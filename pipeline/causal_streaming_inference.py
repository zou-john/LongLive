# Adopted from https://github.com/guandeh17/Self-Forcing
# SPDX-License-Identifier: Apache-2.0
from typing import List, Generator, Tuple
import torch

from pipeline.causal_inference import CausalInferencePipeline
from utils.memory import get_cuda_free_memory_gb, gpu


class CausalStreamingInferencePipeline(CausalInferencePipeline):
    """
    Streaming variant of CausalInferencePipeline that yields partial results
    every N blocks instead of waiting for the entire video to complete.
    """

    def streaming_inference(
        self,
        noise: torch.Tensor,
        text_prompts: List[str],
        blocks_per_chunk: int = 5,
        low_memory: bool = False,
    ) -> Generator[Tuple[torch.Tensor, torch.Tensor, bool], None, None]:
        """
        Streaming inference that yields (video_chunk, latents_chunk, is_final) every N blocks.

        Args:
            noise: Input noise tensor [batch_size, num_output_frames, num_channels, height, width]
            text_prompts: List of text prompts
            blocks_per_chunk: Number of blocks to process before yielding (default: 5)
            low_memory: Whether to use low memory mode

        Yields:
            Tuple of (video_chunk, latents_chunk, is_final):
                - video_chunk: Decoded video frames for this chunk [batch, frames, C, H, W], normalized [0, 1]
                - latents_chunk: Raw latent frames for this chunk [batch, frames, channels, H, W]
                - is_final: True if this is the last chunk
        """
        # noise shape for the whole video excluding chunking
        batch_size, num_output_frames, num_channels, height, width = noise.shape
        # ensure that output frame is divisible by num_frame_per_block
        assert num_output_frames % self.num_frame_per_block == 0
        # if so then, create the number of blocks
        num_blocks = num_output_frames // self.num_frame_per_block

        # generation is going to be conditioned by the prompt
        conditional_dict = self.text_encoder(text_prompts=text_prompts)

        # move the text encoder to the gpu and doing it so it doesn't get an OOM error
        if low_memory:
            from utils.memory import move_model_to_device_with_memory_preservation
            gpu_memory_preservation = get_cuda_free_memory_gb(gpu) + 5
            move_model_to_device_with_memory_preservation(
                self.text_encoder, target_device=gpu, preserved_memory_gb=gpu_memory_preservation
            )

        # to perserve more space, let's put the output on the cpu
        output_device = torch.device('cpu') if low_memory else noise.device

        # initialize KV cache
        local_attn_cfg = getattr(self.args.model_kwargs, "local_attn_size", -1)
        if local_attn_cfg != -1:
            # each token can attend up to local_attn_cfg frames
            kv_cache_size = local_attn_cfg * self.frame_seq_length
        else:
            # each token attend to all token level frames
            kv_cache_size = num_output_frames * self.frame_seq_length

        # shape: num_layers * (K + V) x batch_size x kv_cache_size x num_head x head_dim 
        self._initialize_kv_cache(
            batch_size=batch_size,
            dtype=noise.dtype,
            device=noise.device,
            kv_cache_size_override=kv_cache_size
        )

        # shape: num_layers x (K + V) x batch_size x 512 x num_head x head_dim
        self._initialize_crossattn_cache(
            batch_size=batch_size,
            dtype=noise.dtype,
            device=noise.device
        )

        current_start_frame = 0
        self.generator.model.local_attn_size = self.local_attn_size
        self._set_all_modules_max_attention_size(self.local_attn_size)

        # buffer to accumulate latents for current chunk
        # chunks over blocks and blocks over frames
        chunk_latents = []
        chunk_start_frame = 0 
        blocks_in_chunk = 0

        all_num_frames = [self.num_frame_per_block] * num_blocks

        for block_idx, current_num_frames in enumerate(all_num_frames):
            # number of frames in this block 
            # NOTE: note that denoising is block by block but output is chunk by chunk
            noisy_input = noise[:, current_start_frame:current_start_frame + current_num_frames]

            # spatial: denoise for the number of frames in that current block
            for index, current_timestep in enumerate(self.denoising_step_list):
                timestep = torch.ones(
                    [batch_size, current_num_frames],
                    device=noise.device,
                    dtype=torch.int64
                ) * current_timestep

                # except for the last input, run the generator with the noise
                if index < len(self.denoising_step_list) - 1:
                    # generate prediced noise base off current timestep
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length
                    )

                    next_timestep = self.denoising_step_list[index + 1]
                    # add noise for the next timestep
                    noisy_input = self.scheduler.add_noise(
                        denoised_pred.flatten(0, 1),
                        torch.randn_like(denoised_pred.flatten(0, 1)),
                        next_timestep * torch.ones(
                            [batch_size * current_num_frames], device=noise.device, dtype=torch.long
                        )
                    ).unflatten(0, denoised_pred.shape[:2])
                else:
                    # otherwise just generate the final clean frame and 
                    # cache is not updated for the final frame
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length
                    )

            # store denoised output for this block
            chunk_latents.append(denoised_pred.to(output_device))

            # shape: [batch_size, current_num_frames]
            # self.args.context_noise is usually 0 during inference so there IS real CONTEXT
            context_timestep = torch.ones_like(timestep) * self.args.context_noise
            
            # rerun with timestep zero to update KV cache using clean context
            # update KV cache with clean latent (block-level and diffusion-based)
            self.generator(
                noisy_image_or_video=denoised_pred,
                conditional_dict=conditional_dict,
                timestep=context_timestep,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=current_start_frame * self.frame_seq_length,
            )
            # after every block increment to the next
            current_start_frame += current_num_frames
            blocks_in_chunk += 1 

            # check if the block_idx is the same
            is_final = (block_idx == len(all_num_frames) - 1)
            # check if we should yield a chunk (encapsulates block)
            should_yield = (blocks_in_chunk >= blocks_per_chunk) or is_final

            if should_yield:

                # concatenate accumulated latents along the temporal dimension
                latents_chunk = torch.cat(chunk_latents, dim=1)

                # decode to video using VAE cache for temporal continuity across chunks
                # use_cache=True maintains temporal state so chunk boundaries are seamless
                video_chunk = self.vae.decode_to_pixel(latents_chunk.to(noise.device), use_cache=True)

                video_chunk = (video_chunk * 0.5 + 0.5).clamp(0, 1)

                yield video_chunk, latents_chunk, is_final

                # reset the chunk wise generation
                chunk_latents = []
                chunk_start_frame = current_start_frame
                blocks_in_chunk = 0
