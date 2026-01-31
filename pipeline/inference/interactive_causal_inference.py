# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# To view a copy of this license, visit http://www.apache.org/licenses/LICENSE-2.0
#
# No warranties are given. The work is provided "AS IS", without warranty of any kind, express or implied.
#
# SPDX-License-Identifier: Apache-2.0
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
import torch

from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper
from utils.memory import gpu, get_cuda_free_memory_gb, move_model_to_device_with_memory_preservation
from pipeline.inference.causal_inference import CausalInferencePipeline
import torch.distributed as dist
from utils.debug_option import DEBUG


@dataclass
class KVCacheCheckpoint:
    """Snapshot of KV-cache state saved after a scene completes.

    checkpoint[N] stores the state after scene N finishes (1-indexed).
    To edit from scene N onwards, restore checkpoint[N-1].
    Example: checkpoint[2] is saved after scene 2 completes.
             Restore checkpoint[2] to regenerate scenes 3, 4, 5.
    """
    scene: int  # The scene that just finished (1-indexed)
    frame_index: int  # Frame index where this scene ended
    kv_cache: List[Dict[str, torch.Tensor]] = field(default_factory=list)
    crossattn_cache: List[Dict[str, Any]] = field(default_factory=list)
    output_latents: Optional[torch.Tensor] = None


class InteractiveCausalInferencePipeline(CausalInferencePipeline):
    def __init__(
        self,
        args,
        device,
        *,
        generator: WanDiffusionWrapper | None = None,
        text_encoder: WanTextEncoder | None = None,
        vae: WanVAEWrapper | None = None,
    ):
        super().__init__(args, device, generator=generator, text_encoder=text_encoder, vae=vae)
        self.global_sink = getattr(args, "global_sink", False)
        # Checkpoint storage: maps scene number (1-indexed) -> KVCacheCheckpoint
        self.checkpoints: Dict[int, KVCacheCheckpoint] = {}

    def _save_checkpoint(self, scene: int, frame_index: int, output: torch.Tensor):
        """Save a checkpoint after a scene completes.

        Args:
            scene: Scene number that just finished (1-indexed, e.g., 1 for scene 1)
            frame_index: The frame index where this scene ended
            output: The full output tensor (latents) - we'll copy up to frame_index
        """
        # Deep copy KV cache
        kv_cache_copy = []
        for cache in self.kv_cache1:  # FOR EVERY TRANSFORMER BLOCK
            kv_cache_copy.append({
                "k": cache["k"].clone(),
                "v": cache["v"].clone(),
                "global_end_index": cache["global_end_index"].clone(),
                "local_end_index": cache["local_end_index"].clone(),
            })

        # Deep copy cross-attention cache
        crossattn_cache_copy = []
        for cache in self.crossattn_cache: # FOR EVERY TRANSFORMER BLOCK
            crossattn_cache_copy.append({
                "k": cache["k"].clone(),
                "v": cache["v"].clone(),
                "is_init": cache["is_init"],
            })

        # Store checkpoint
        self.checkpoints[scene] = KVCacheCheckpoint(
            scene=scene,
            frame_index=frame_index,
            kv_cache=kv_cache_copy,
            crossattn_cache=crossattn_cache_copy,
            output_latents=output[:, :frame_index].clone(),
        )

        if DEBUG:
            print(f"[Checkpoint] Saved checkpoint[{scene}] after scene {scene} at frame {frame_index}")

    def _restore_checkpoint(self, scene: int) -> int:
        """Restore state from a checkpoint.

        Args:
            scene: Scene number to restore from (1-indexed). Will resume AFTER this scene.

        Returns:
            The frame index to resume from
        """
        if scene not in self.checkpoints:
            raise ValueError(f"No checkpoint[{scene}] found. "
                           f"Available: {list(self.checkpoints.keys())}")

        checkpoint = self.checkpoints[scene]

        # Restore KV cache
        for i, cache in enumerate(checkpoint.kv_cache): # FOR EVERY TRANSFORMER BLOCK
            self.kv_cache1[i]["k"].copy_(cache["k"])
            self.kv_cache1[i]["v"].copy_(cache["v"])
            self.kv_cache1[i]["global_end_index"].copy_(cache["global_end_index"])
            self.kv_cache1[i]["local_end_index"].copy_(cache["local_end_index"])

        # Restore cross-attention cache
        for i, cache in enumerate(checkpoint.crossattn_cache):
            self.crossattn_cache[i]["k"].copy_(cache["k"])
            self.crossattn_cache[i]["v"].copy_(cache["v"])
            self.crossattn_cache[i]["is_init"] = cache["is_init"]

        # Clear any checkpoints after this one (they're now invalid)
        scenes_to_remove = [s for s in self.checkpoints.keys() if s > scene]
        for s in scenes_to_remove:
            del self.checkpoints[s]

        if DEBUG:
            print(f"[Checkpoint] Restored checkpoint[{scene}], resuming at frame {checkpoint.frame_index}")

        return checkpoint.frame_index

    def get_checkpoint_latents(self, scene: int) -> Optional[torch.Tensor]:
        """Get the output latents from a checkpoint."""
        if scene not in self.checkpoints:
            return None
        return self.checkpoints[scene].output_latents

    def clear_checkpoints(self):
        """Clear all stored checkpoints to free memory."""
        self.checkpoints.clear()

    # Internal helpers
    def _recache_after_switch(self, output, current_start_frame, new_conditional_dict):
        if not self.global_sink:
            # reset kv cache
            for block_idx in range(self.num_transformer_blocks):
                cache = self.kv_cache1[block_idx]
                cache["k"].zero_()
                cache["v"].zero_()
                # cache["global_end_index"].zero_()
                # cache["local_end_index"].zero_()
            
        # reset cross-attention cache
        for blk in self.crossattn_cache:
            blk["k"].zero_()
            blk["v"].zero_()
            blk["is_init"] = False

        # recache
        if current_start_frame == 0:
            return
        
        num_recache_frames = current_start_frame if self.local_attn_size == -1 else min(self.local_attn_size, current_start_frame)
        recache_start_frame = current_start_frame - num_recache_frames
        
        frames_to_recache = output[:, recache_start_frame:current_start_frame]
        # move to gpu
        if frames_to_recache.device.type == 'cpu':
            target_device = next(self.generator.parameters()).device
            frames_to_recache = frames_to_recache.to(target_device)
        batch_size = frames_to_recache.shape[0]
        print(f"num_recache_frames: {num_recache_frames}, recache_start_frame: {recache_start_frame}, current_start_frame: {current_start_frame}")
        
        # prepare blockwise causal mask
        device = frames_to_recache.device
        block_mask = self.generator.model._prepare_blockwise_causal_attn_mask(
            device=device,
            num_frames=num_recache_frames,
            frame_seqlen=self.frame_seq_length,
            num_frame_per_block=self.num_frame_per_block,
            local_attn_size=self.local_attn_size
        )
        
        context_timestep = torch.ones([batch_size, num_recache_frames], 
                                    device=device, dtype=torch.int64) * self.args.context_noise
        
        self.generator.model.block_mask = block_mask
        
        # recache
        with torch.no_grad():
            self.generator(
                noisy_image_or_video=frames_to_recache,
                conditional_dict=new_conditional_dict,
                timestep=context_timestep,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=recache_start_frame * self.frame_seq_length,
                sink_recache_after_switch=not self.global_sink,
            )
        
        # reset cross-attention cache
        for blk in self.crossattn_cache:
            blk["k"].zero_()
            blk["v"].zero_()
            blk["is_init"] = False

    def inference(
        self,
        noise: torch.Tensor,
        *,
        text_prompts_list: List[List[str]],
        switch_frame_indices: List[int],
        return_latents: bool = False,
        low_memory: bool = False,
        save_checkpoints: bool = False,
    ):
        """Generate a video and switch prompts at specified frame indices.

        Args:
            noise: Noise tensor, shape = (B, T_out, C, H, W).
            text_prompts_list: List[List[str]], length = N_seg. Prompt list used for segment i (aligned with batch).
            switch_frame_indices: List[int], length = N_seg - 1. The i-th value indicates that when generation reaches this frame (inclusive)
                we start using the prompts for segment i+1.
            return_latents: Whether to also return the latent tensor.
            low_memory: Enable low-memory mode.
            save_checkpoints: If True, save KV-cache checkpoints after each segment completes.
                              Use restore_and_continue() to resume from a checkpoint.
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

        output_device = torch.device('cpu') if low_memory else noise.device
        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=output_device,
            dtype=noise.dtype
        )

        # initialize KV cache
        local_attn_cfg = getattr(self.args.model_kwargs, "local_attn_size", -1)
        kv_policy = ""
        if local_attn_cfg != -1:
            # local attention
            kv_cache_size = local_attn_cfg * self.frame_seq_length
            kv_policy = f"int->local, size={local_attn_cfg}"
        else:
            # global attention
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
        print(f"[inference] local_attn_size set on model: {self.generator.model.local_attn_size}")
        self._set_all_modules_max_attention_size(self.local_attn_size)

        # temporal denoising by blocks
        all_num_frames = [self.num_frame_per_block] * num_blocks
        segment_idx = 0  # current segment index
        next_switch_pos = (
            switch_frame_indices[segment_idx]
            if segment_idx < len(switch_frame_indices)
            else None
        )

        if DEBUG:
            print("[MultipleSwitch] all_num_frames", all_num_frames)
            print("[MultipleSwitch] switch_frame_indices", switch_frame_indices)

        for current_num_frames in all_num_frames:
            if next_switch_pos is not None and current_start_frame >= next_switch_pos:
                # Save checkpoint for the scene that just completed (before switching)
                # segment_idx is 0-indexed, scene is 1-indexed, so scene = segment_idx + 1
                if save_checkpoints:
                    self._save_checkpoint(segment_idx + 1, current_start_frame, output)

                segment_idx += 1
                self._recache_after_switch(output, current_start_frame, cond_list[segment_idx])
                if DEBUG:
                    print(
                        f"[MultipleSwitch] Switch to segment {segment_idx} at frame {current_start_frame}"
                    )
                next_switch_pos = (
                    switch_frame_indices[segment_idx]
                    if segment_idx < len(switch_frame_indices)
                    else None
                )
                print(f"segment_idx: {segment_idx}")
                print(f"text_prompts_list[segment_idx]: {text_prompts_list[segment_idx]}")
            cond_in_use = cond_list[segment_idx]

            noisy_input = noise[
                :, current_start_frame : current_start_frame + current_num_frames
            ]

            # ---------------- Spatial denoising loop ----------------
            for index, current_timestep in enumerate(self.denoising_step_list):
                # denoise frames within a block
                timestep = (
                    torch.ones([batch_size, current_num_frames],
                    device=noise.device,
                    dtype=torch.int64)
                    * current_timestep
                )
                # except for the last input, run the generator with the noise
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

            # Record output
            output[:, current_start_frame : current_start_frame + current_num_frames] = denoised_pred.to(output.device)

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

            # Update frame pointer
            current_start_frame += current_num_frames

        # Standard decoding
        video = self.vae.decode_to_pixel_chunk(output.to(noise.device), use_cache=False, chunk_size=60)
        video = (video * 0.5 + 0.5).clamp(0, 1)

        if return_latents:
            return video, output
        return video

    def inference_from_checkpoint(
        self,
        noise: torch.Tensor,
        *,
        restore_from_scene: int,
        text_prompts_list: List[List[str]],
        switch_frame_indices: List[int],
        return_latents: bool = False,
        low_memory: bool = False,
        save_checkpoints: bool = False,
    ):
        """Resume video generation from a saved checkpoint.

        Example: 5 prompts = 5 scenes. To edit scenes 3-5, call with restore_from_scene=2.
        This restores checkpoint[2] (saved after scene 2), retains scenes 1-2, regenerates 3-5.

        Args:
            noise: Noise tensor, shape = (B, T_out, C, H, W).
            restore_from_scene: Scene number to restore from (1-indexed). Generation resumes AFTER this scene.
                                e.g., restore_from_scene=2 keeps scenes 1-2, regenerates 3+.
            text_prompts_list: Full list of prompts for ALL scenes. Only prompts after
                               restore_from_scene will be encoded (skips encoding for retained scenes).
            switch_frame_indices: Full list of switch indices (same as original inference).
            return_latents: Whether to also return the latent tensor.
            low_memory: Enable low-memory mode.
            save_checkpoints: If True, save new checkpoints as scenes complete.
        """
        if restore_from_scene not in self.checkpoints: # number in the checkpoint
            raise ValueError(f"No checkpoint[{restore_from_scene}] found. "
                           f"Available: {list(self.checkpoints.keys())}")

        checkpoint = self.checkpoints[restore_from_scene]
        resume_frame = checkpoint.frame_index
        # Convert 1-indexed scene to 0-indexed segment for internal use
        start_segment = restore_from_scene  # scene 2 -> segment 2 (which is the 3rd prompt, 0-indexed)

        batch_size, num_output_frames, num_channels, height, width = noise.shape
        assert len(text_prompts_list) >= start_segment + 1, (
            f"text_prompts_list must have prompts for scene {restore_from_scene + 1} onwards"
        )
        assert len(switch_frame_indices) == len(text_prompts_list) - 1, (
            "length of switch_frame_indices should be one less than text_prompts_list"
        )
        assert num_output_frames % self.num_frame_per_block == 0
        num_blocks = num_output_frames // self.num_frame_per_block

        # Only encode prompts we need (skip prompts for retained scenes)
        print(f"[Resume] Restoring checkpoint[{restore_from_scene}] at frame {resume_frame}")
        print(f"[Resume] Retaining scenes 1-{restore_from_scene}, regenerating {restore_from_scene + 1}-{len(text_prompts_list)}")
        prompts_to_encode = text_prompts_list[start_segment:]
        cond_list = [self.text_encoder(text_prompts=p) for p in prompts_to_encode]

        if low_memory:
            gpu_memory_preservation = get_cuda_free_memory_gb(gpu) + 5
            move_model_to_device_with_memory_preservation(
                self.text_encoder,
                target_device=gpu,
                preserved_memory_gb=gpu_memory_preservation,
            )

        output_device = torch.device('cpu') if low_memory else noise.device
        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=output_device,
            dtype=noise.dtype
        )

        # Copy retained latents from checkpoint
        output[:, :resume_frame] = checkpoint.output_latents.to(output_device)

        # Restore KV cache state
        self._restore_checkpoint(restore_from_scene)

        # Recache with the first new prompt
        self._recache_after_switch(output, resume_frame, cond_list[0])

        current_start_frame = resume_frame
        self.generator.model.local_attn_size = self.local_attn_size
        self._set_all_modules_max_attention_size(self.local_attn_size)

        # Calculate blocks to generate
        start_block = resume_frame // self.num_frame_per_block
        num_blocks_to_generate = num_blocks - start_block
        all_num_frames = [self.num_frame_per_block] * num_blocks_to_generate

        # Global segment index
        segment_idx = start_segment
        next_switch_pos = (
            switch_frame_indices[segment_idx]
            if segment_idx < len(switch_frame_indices)
            else None
        )

        if DEBUG:
            print(f"[Resume] Starting at scene {segment_idx + 1}")

        for current_num_frames in all_num_frames:
            # Check for prompt switch
            if next_switch_pos is not None and current_start_frame >= next_switch_pos:
                # Save checkpoint for the scene that just completed (1-indexed)
                if save_checkpoints:
                    self._save_checkpoint(segment_idx + 1, current_start_frame, output)

                segment_idx += 1
                cond_idx = segment_idx - start_segment
                self._recache_after_switch(output, current_start_frame, cond_list[cond_idx])
                if DEBUG:
                    print(f"[Resume] Switch to scene {segment_idx + 1} at frame {current_start_frame}")
                next_switch_pos = (
                    switch_frame_indices[segment_idx]
                    if segment_idx < len(switch_frame_indices)
                    else None
                )

            cond_idx = segment_idx - start_segment
            cond_in_use = cond_list[cond_idx]
            noisy_input = noise[:, current_start_frame:current_start_frame + current_num_frames]

            # Spatial denoising loop
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

            output[:, current_start_frame:current_start_frame + current_num_frames] = denoised_pred.to(output.device)

            # Rerun with clean context to update cache
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

        # Decode full video (retained + regenerated)
        video = self.vae.decode_to_pixel_chunk(output.to(noise.device), use_cache=False, chunk_size=60)
        video = (video * 0.5 + 0.5).clamp(0, 1)

        if return_latents:
            return video, output
        return video