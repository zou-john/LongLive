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
        batch_size, num_output_frames, num_channels, height, width = noise.shape
        assert num_output_frames % self.num_frame_per_block == 0
        num_blocks = num_output_frames // self.num_frame_per_block

        conditional_dict = self.text_encoder(text_prompts=text_prompts)

        if low_memory:
            from utils.memory import move_model_to_device_with_memory_preservation
            gpu_memory_preservation = get_cuda_free_memory_gb(gpu) + 5
            move_model_to_device_with_memory_preservation(
                self.text_encoder, target_device=gpu, preserved_memory_gb=gpu_memory_preservation
            )

        # Output device based on low_memory mode
        output_device = torch.device('cpu') if low_memory else noise.device

        # Initialize KV cache
        local_attn_cfg = getattr(self.args.model_kwargs, "local_attn_size", -1)
        if local_attn_cfg != -1:
            kv_cache_size = local_attn_cfg * self.frame_seq_length
        else:
            kv_cache_size = num_output_frames * self.frame_seq_length

        self._initialize_kv_cache(
            batch_size=batch_size,
            dtype=noise.dtype,
            device=noise.device,
            kv_cache_size_override=kv_cache_size
        )
        self._initialize_crossattn_cache(
            batch_size=batch_size,
            dtype=noise.dtype,
            device=noise.device
        )

        current_start_frame = 0
        self.generator.model.local_attn_size = self.local_attn_size
        self._set_all_modules_max_attention_size(self.local_attn_size)

        # Buffer to accumulate latents for current chunk
        chunk_latents = []
        chunk_start_frame = 0
        blocks_in_chunk = 0

        all_num_frames = [self.num_frame_per_block] * num_blocks

        for block_idx, current_num_frames in enumerate(all_num_frames):
            noisy_input = noise[:, current_start_frame:current_start_frame + current_num_frames]

            # Spatial denoising loop
            for index, current_timestep in enumerate(self.denoising_step_list):
                timestep = torch.ones(
                    [batch_size, current_num_frames],
                    device=noise.device,
                    dtype=torch.int64
                ) * current_timestep

                if index < len(self.denoising_step_list) - 1:
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length
                    )
                    next_timestep = self.denoising_step_list[index + 1]
                    noisy_input = self.scheduler.add_noise(
                        denoised_pred.flatten(0, 1),
                        torch.randn_like(denoised_pred.flatten(0, 1)),
                        next_timestep * torch.ones(
                            [batch_size * current_num_frames], device=noise.device, dtype=torch.long
                        )
                    ).unflatten(0, denoised_pred.shape[:2])
                else:
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length
                    )

            # Store denoised output for this block
            chunk_latents.append(denoised_pred.to(output_device))

            # Rerun with timestep zero to update KV cache using clean context
            context_timestep = torch.ones_like(timestep) * self.args.context_noise
            self.generator(
                noisy_image_or_video=denoised_pred,
                conditional_dict=conditional_dict,
                timestep=context_timestep,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=current_start_frame * self.frame_seq_length,
            )

            current_start_frame += current_num_frames
            blocks_in_chunk += 1

            # Check if we should yield a chunk
            is_final = (block_idx == len(all_num_frames) - 1)
            should_yield = (blocks_in_chunk >= blocks_per_chunk) or is_final

            if should_yield:
                # Concatenate accumulated latents
                latents_chunk = torch.cat(chunk_latents, dim=1)

                # Decode to video
                video_chunk = self.vae.decode_to_pixel(latents_chunk.to(noise.device), use_cache=False)
                video_chunk = (video_chunk * 0.5 + 0.5).clamp(0, 1)

                yield video_chunk, latents_chunk, is_final

                # Reset chunk buffer
                chunk_latents = []
                chunk_start_frame = current_start_frame
                blocks_in_chunk = 0
