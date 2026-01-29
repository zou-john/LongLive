# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# To view a copy of this license, visit http://www.apache.org/licenses/LICENSE-2.0
#
# No warranties are given. The work is provided "AS IS", without warranty of any kind, express or implied.
#
# SPDX-License-Identifier: Apache-2.0
from typing import List, Generator, Tuple
import torch

from pipeline.interactive_causal_inference import InteractiveCausalInferencePipeline
from utils.memory import get_cuda_free_memory_gb, gpu, move_model_to_device_with_memory_preservation
from utils.debug_option import DEBUG


class InteractiveCausalStreamingInferencePipeline(InteractiveCausalInferencePipeline):
    """
    Streaming variant of InteractiveCausalInferencePipeline that yields partial results
    every N blocks instead of waiting for the entire video to complete.
    Supports prompt switching at specified frame indices.
    """

    def streaming_inference(
        self,
        noise: torch.Tensor,
        *,
        text_prompts_list: List[List[str]],
        switch_frame_indices: List[int],
        blocks_per_chunk: int = 5,
        low_memory: bool = False,
    ) -> Generator[Tuple[torch.Tensor, torch.Tensor, bool], None, None]:
        """
        Streaming inference that yields (video_chunk, latents_chunk, is_final) every N blocks
        with support for prompt switching at specified frame indices.

        Args:
            noise: Input noise tensor [batch_size, num_output_frames, num_channels, height, width]
            text_prompts_list: List[List[str]], length = N_seg. Prompt list used for segment i (aligned with batch).
            switch_frame_indices: List[int], length = N_seg - 1. The i-th value indicates that when generation reaches this frame (inclusive)
                we start using the prompts for segment i+1.
            blocks_per_chunk: Number of blocks to process before yielding (default: 5)
            low_memory: Whether to use low memory mode

        Yields:
            Tuple of (video_chunk, latents_chunk, is_final):
                - video_chunk: Decoded video frames for this chunk [batch, frames, C, H, W], normalized [0, 1]
                - latents_chunk: Raw latent frames for this chunk [batch, frames, channels, H, W]
                - is_final: True if this is the last chunk
        """
        batch_size, num_output_frames, num_channels, height, width = noise.shape
        assert len(text_prompts_list) >= 1, "text_prompts_list must not be empty"
        assert len(switch_frame_indices) == len(text_prompts_list) - 1, (
            "length of switch_frame_indices should be one less than text_prompts_list"
        )
        assert num_output_frames % self.num_frame_per_block == 0
        num_blocks = num_output_frames // self.num_frame_per_block

        # encode all prompts
        print(text_prompts_list)
        cond_list = [self.text_encoder(text_prompts=p) for p in text_prompts_list]

        if low_memory:
            gpu_memory_preservation = get_cuda_free_memory_gb(gpu) + 5
            move_model_to_device_with_memory_preservation(
                self.text_encoder,
                target_device=gpu,
                preserved_memory_gb=gpu_memory_preservation,
            )

        # Output device based on low_memory mode
        output_device = torch.device('cpu') if low_memory else noise.device

        # Initialize caches
        local_attn_cfg = getattr(self.args.model_kwargs, "local_attn_size", -1)
        if local_attn_cfg != -1:
            kv_cache_size = local_attn_cfg * self.frame_seq_length
            kv_policy = f"int->local, size={local_attn_cfg}"
        else:
            kv_cache_size = num_output_frames * self.frame_seq_length
            kv_policy = "global (-1)"
        print(f"kv_cache_size: {kv_cache_size} (policy: {kv_policy}, frame_seq_length: {self.frame_seq_length}, num_output_frames: {num_output_frames})")

        self._initialize_kv_cache(
            batch_size,
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
        print(f"[streaming_inference] local_attn_size set on model: {self.generator.model.local_attn_size}")
        self._set_all_modules_max_attention_size(self.local_attn_size)

        # Buffer to accumulate latents for current chunk
        chunk_latents = []
        chunk_start_frame = 0
        blocks_in_chunk = 0

        # temporal denoising by blocks
        all_num_frames = [self.num_frame_per_block] * num_blocks
        segment_idx = 0  # current segment index
        next_switch_pos = (
            switch_frame_indices[segment_idx]
            if segment_idx < len(switch_frame_indices)
            else None
        )

        # Track full output for recaching after switch
        full_output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=output_device,
            dtype=noise.dtype
        )

        if DEBUG:
            print("[MultipleSwitch Streaming] all_num_frames", all_num_frames)
            print("[MultipleSwitch Streaming] switch_frame_indices", switch_frame_indices)

        for block_idx, current_num_frames in enumerate(all_num_frames):
            # Check for prompt switch
            if next_switch_pos is not None and current_start_frame >= next_switch_pos:
                segment_idx += 1
                self._recache_after_switch(full_output, current_start_frame, cond_list[segment_idx])
                if DEBUG:
                    print(
                        f"[MultipleSwitch Streaming] Switch to segment {segment_idx} at frame {current_start_frame}"
                    )
                next_switch_pos = (
                    switch_frame_indices[segment_idx]
                    if segment_idx < len(switch_frame_indices)
                    else None
                )
                print(f"segment_idx: {segment_idx}")
                print(f"text_prompts_list[segment_idx]: {text_prompts_list[segment_idx]}")
            
            cond_in_use = cond_list[segment_idx]

            noisy_input = noise[:, current_start_frame : current_start_frame + current_num_frames]

            # ---------------- Spatial denoising loop ----------------
            for index, current_timestep in enumerate(self.denoising_step_list):
                timestep = (
                    torch.ones([batch_size, current_num_frames],
                    device=noise.device,
                    dtype=torch.int64)
                    * current_timestep
                )

                if index < len(self.denoising_step_list) - 1:
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=cond_in_use,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length,
                    )
                    next_timestep = self.denoising_step_list[index + 1]
                    noisy_input = self.scheduler.add_noise(
                        denoised_pred.flatten(0, 1),
                        torch.randn_like(denoised_pred.flatten(0, 1)),
                        next_timestep
                        * torch.ones(
                            [batch_size * current_num_frames], device=noise.device, dtype=torch.long
                        ),
                    ).unflatten(0, denoised_pred.shape[:2])
                else:
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=cond_in_use,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length,
                    )

            # Store denoised output for this block
            full_output[:, current_start_frame : current_start_frame + current_num_frames] = denoised_pred.to(output_device)
            chunk_latents.append(denoised_pred.to(output_device))

            # rerun with clean context to update cache
            context_timestep = torch.ones_like(timestep) * self.args.context_noise
            self.generator(
                noisy_image_or_video=denoised_pred,
                conditional_dict=cond_in_use,
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