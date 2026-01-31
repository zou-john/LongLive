"""
Test script to debug video flickering when using encode_frame() for websocket transmission.

This script:
1. Runs pipeline.inference() to generate video
2. Saves the raw video (same as inference.py - no flicker baseline)
3. Encodes each frame using encode_frame() -> decodes back
4. Saves the reconstructed video from decoded frames
5. Saves individual frames for comparison (before/after encoding)

Output structure in videos/byte/:
  - raw_video.mp4           : Direct from pipeline (baseline, no flicker)
  - encoded_video.mp4       : Reconstructed from encode_frame() -> decode
  - frames_raw/             : Individual raw frames as PNG
  - frames_encoded/         : Individual frames after encode_frame() round-trip
  - comparison_stats.txt    : Statistics about value ranges, differences

Usage:
    python tests/02_test_encode_frame.py --config_path configs/longlive_inference.yaml
"""

import os
import argparse
import base64
import io
import json
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
from einops import rearrange
from torchvision.io import write_video

from pipeline import CausalInferencePipeline
from utils.misc import set_seed
from utils.memory import get_cuda_free_memory_gb, DynamicSwapInstaller

torch.set_grad_enabled(False)

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------

OUTPUT_DIR = Path("/Users/johnbzou/VSCode/random stuff/LongLive/videos/byte")
TEST_PROMPT = "A cat walking in a garden with flowers, cinematic lighting"
NUM_FRAMES_TO_SAVE = 10  # Save first N individual frames for detailed comparison


def encode_frame(frame: torch.Tensor) -> dict:
    """Original encode_frame from release_server.py"""
    frame = (frame.clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
    img = Image.fromarray(frame)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)

    return {
        "data": base64.b64encode(buf.getvalue()).decode("ascii"),
        "format": "jpeg",
    }


def decode_frame(payload: dict) -> np.ndarray:
    """Decode a frame payload back to numpy array (simulating client-side decode)"""
    data = base64.b64decode(payload["data"])
    buf = io.BytesIO(data)
    img = Image.open(buf)
    return np.array(img)


def encode_frame_png(frame: torch.Tensor) -> dict:
    """Alternative encode using PNG (lossless) for comparison"""
    frame = (frame.clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
    img = Image.fromarray(frame)

    buf = io.BytesIO()
    img.save(buf, format="PNG")

    return {
        "data": base64.b64encode(buf.getvalue()).decode("ascii"),
        "format": "png",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--prompt", type=str, default=TEST_PROMPT)
    parser.add_argument("--num_frames", type=int, default=21, help="Number of frames to generate")
    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)

    # Override for testing
    config.num_output_frames = args.num_frames

    device = torch.device("cuda")
    set_seed(config.seed)

    low_memory = get_cuda_free_memory_gb(device) < 40
    low_memory = True  # force for safety

    print(f"[Init] Device: {device}")
    print(f"[Init] Free VRAM: {get_cuda_free_memory_gb(device):.2f} GB")
    print(f"[Init] Generating {args.num_frames} frames")

    # Create output directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "frames_raw").mkdir(exist_ok=True)
    (OUTPUT_DIR / "frames_encoded_jpeg").mkdir(exist_ok=True)
    (OUTPUT_DIR / "frames_encoded_png").mkdir(exist_ok=True)

    # -----------------------------------------------------------------------------
    # Load Pipeline
    # -----------------------------------------------------------------------------
    print("[Init] Loading pipeline...")
    pipeline = CausalInferencePipeline(config, device=device)

    if config.generator_ckpt:
        state = torch.load(config.generator_ckpt, map_location="cpu")
        if "generator_ema" in state and config.use_ema:
            state = state["generator_ema"]
        elif "generator" in state:
            state = state["generator"]
        elif "model" in state:
            state = state["model"]
        pipeline.generator.load_state_dict(state, strict=False)

    pipeline = pipeline.to(dtype=torch.bfloat16)
    pipeline.generator.to(device)
    pipeline.vae.to(device)

    if low_memory:
        DynamicSwapInstaller.install_model(pipeline.text_encoder, device=device)

    print("[Init] Pipeline ready")

    # -----------------------------------------------------------------------------
    # Run Inference
    # -----------------------------------------------------------------------------
    prompts = [args.prompt]

    sampled_noise = torch.randn(
        [1, config.num_output_frames, 16, 60, 104],
        device=device,
        dtype=torch.bfloat16,
    )

    print(f"[Inference] Running with prompt: {args.prompt[:50]}...")

    video, latents = pipeline.inference(
        noise=sampled_noise,
        text_prompts=prompts,
        return_latents=True,
        low_memory=low_memory,
        profile=False,
    )

    # Shape: (B, T, C, H, W) -> (B, T, H, W, C)
    video_thwc = rearrange(video, 'b t c h w -> b t h w c').cpu()
    print(f"[Inference] Output shape: {video_thwc.shape}")
    print(f"[Inference] Value range: min={video_thwc.min():.4f}, max={video_thwc.max():.4f}")

    pipeline.vae.model.clear_cache()

    # -----------------------------------------------------------------------------
    # Save Raw Video (baseline - same as inference.py)
    # -----------------------------------------------------------------------------
    print("[Save] Saving raw video (baseline)...")

    # This is exactly how inference.py does it
    raw_video_for_save = 255.0 * video_thwc[0]  # Shape: (T, H, W, C)
    raw_video_path = OUTPUT_DIR / "raw_video.mp4"
    write_video(str(raw_video_path), raw_video_for_save, fps=16)
    print(f"[Save] Raw video saved to: {raw_video_path}")

    # -----------------------------------------------------------------------------
    # Process Through encode_frame() and Reconstruct (JPEG)
    # -----------------------------------------------------------------------------
    print("[Encode] Processing frames through encode_frame() (JPEG)...")

    stats = {
        "num_frames": video_thwc.shape[1],
        "raw_value_range": {
            "min": float(video_thwc.min()),
            "max": float(video_thwc.max()),
        },
        "frames_outside_01": [],
        "jpeg_sizes_bytes": [],
        "png_sizes_bytes": [],
        "frame_differences": [],
    }

    encoded_frames_jpeg = []
    encoded_frames_png = []

    for idx in range(video_thwc.shape[1]):
        frame = video_thwc[0, idx]  # Shape: (H, W, C)

        # Check if values are outside [0, 1]
        if frame.min() < 0 or frame.max() > 1:
            stats["frames_outside_01"].append({
                "frame_idx": idx,
                "min": float(frame.min()),
                "max": float(frame.max()),
            })

        # Encode with JPEG
        payload_jpeg = encode_frame(frame)
        decoded_jpeg = decode_frame(payload_jpeg)
        encoded_frames_jpeg.append(decoded_jpeg)
        stats["jpeg_sizes_bytes"].append(len(payload_jpeg["data"]))

        # Encode with PNG (lossless comparison)
        payload_png = encode_frame_png(frame)
        decoded_png = decode_frame(payload_png)
        encoded_frames_png.append(decoded_png)
        stats["png_sizes_bytes"].append(len(payload_png["data"]))

        # Calculate difference between raw and JPEG-encoded
        raw_uint8 = (frame.clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
        diff = np.abs(raw_uint8.astype(np.int16) - decoded_jpeg.astype(np.int16))
        stats["frame_differences"].append({
            "frame_idx": idx,
            "max_diff": int(diff.max()),
            "mean_diff": float(diff.mean()),
        })

        # Save individual frames for first N frames
        if idx < NUM_FRAMES_TO_SAVE:
            # Save raw frame as PNG
            raw_img = Image.fromarray(raw_uint8)
            raw_img.save(OUTPUT_DIR / "frames_raw" / f"frame_{idx:04d}.png")

            # Save JPEG-encoded frame
            jpeg_img = Image.fromarray(decoded_jpeg)
            jpeg_img.save(OUTPUT_DIR / "frames_encoded_jpeg" / f"frame_{idx:04d}.png")

            # Save PNG-encoded frame
            png_img = Image.fromarray(decoded_png)
            png_img.save(OUTPUT_DIR / "frames_encoded_png" / f"frame_{idx:04d}.png")

        if (idx + 1) % 20 == 0:
            print(f"  Processed {idx + 1}/{video_thwc.shape[1]} frames")

    # -----------------------------------------------------------------------------
    # Reconstruct Video from Encoded Frames
    # -----------------------------------------------------------------------------
    print("[Reconstruct] Building video from JPEG-encoded frames...")

    # Stack frames: (T, H, W, C)
    encoded_video_jpeg = torch.from_numpy(np.stack(encoded_frames_jpeg, axis=0)).float()
    encoded_video_jpeg_path = OUTPUT_DIR / "encoded_video_jpeg.mp4"
    write_video(str(encoded_video_jpeg_path), encoded_video_jpeg, fps=16)
    print(f"[Save] JPEG-encoded video saved to: {encoded_video_jpeg_path}")

    print("[Reconstruct] Building video from PNG-encoded frames...")
    encoded_video_png = torch.from_numpy(np.stack(encoded_frames_png, axis=0)).float()
    encoded_video_png_path = OUTPUT_DIR / "encoded_video_png.mp4"
    write_video(str(encoded_video_png_path), encoded_video_png, fps=16)
    print(f"[Save] PNG-encoded video saved to: {encoded_video_png_path}")

    # -----------------------------------------------------------------------------
    # Save Statistics
    # -----------------------------------------------------------------------------
    stats_summary = {
        "num_frames": stats["num_frames"],
        "raw_value_range": stats["raw_value_range"],
        "num_frames_outside_01": len(stats["frames_outside_01"]),
        "frames_outside_01_details": stats["frames_outside_01"][:5],  # First 5
        "avg_jpeg_size_bytes": sum(stats["jpeg_sizes_bytes"]) / len(stats["jpeg_sizes_bytes"]),
        "avg_png_size_bytes": sum(stats["png_sizes_bytes"]) / len(stats["png_sizes_bytes"]),
        "max_frame_diff_from_jpeg": max(d["max_diff"] for d in stats["frame_differences"]),
        "avg_frame_diff_from_jpeg": sum(d["mean_diff"] for d in stats["frame_differences"]) / len(stats["frame_differences"]),
        "frame_diff_details": stats["frame_differences"][:5],  # First 5
    }

    stats_path = OUTPUT_DIR / "comparison_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats_summary, f, indent=2)

    # Also print summary
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"Number of frames: {stats_summary['num_frames']}")
    print(f"Raw value range: [{stats_summary['raw_value_range']['min']:.4f}, {stats_summary['raw_value_range']['max']:.4f}]")
    print(f"Frames with values outside [0,1]: {stats_summary['num_frames_outside_01']}")
    print(f"Avg JPEG size: {stats_summary['avg_jpeg_size_bytes']:.0f} bytes")
    print(f"Avg PNG size: {stats_summary['avg_png_size_bytes']:.0f} bytes")
    print(f"Max pixel diff (JPEG vs raw): {stats_summary['max_frame_diff_from_jpeg']}")
    print(f"Avg pixel diff (JPEG vs raw): {stats_summary['avg_frame_diff_from_jpeg']:.2f}")
    print("=" * 60)

    print(f"\nOutput files:")
    print(f"  - {raw_video_path} (baseline - should not flicker)")
    print(f"  - {encoded_video_jpeg_path} (reconstructed from JPEG encode_frame)")
    print(f"  - {encoded_video_png_path} (reconstructed from PNG - lossless)")
    print(f"  - {OUTPUT_DIR}/frames_raw/ (individual raw frames)")
    print(f"  - {OUTPUT_DIR}/frames_encoded_jpeg/ (frames after JPEG round-trip)")
    print(f"  - {OUTPUT_DIR}/frames_encoded_png/ (frames after PNG round-trip)")
    print(f"  - {stats_path} (detailed statistics)")

    print("\n[Done] Compare raw_video.mp4 vs encoded_video_jpeg.mp4 to identify flickering source.")
    print("If JPEG video flickers but PNG doesn't, the JPEG compression is the cause.")
    print("If both encoded videos flicker, check the clamp(0,1) or dtype conversion.")


if __name__ == "__main__":
    main()
