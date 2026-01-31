#!/usr/bin/env python3
"""
Self-contained WebSocket video streaming test for LongLive.

This script runs BOTH a local websocket server AND client to test the full pipeline:
1. Runs inference to generate video frames
2. Encodes and sends frames through websocket (server side)
3. Receives and decodes frames through websocket (client side)
4. Saves received frames and reconstructs video

This isolates whether flickering is caused by websocket transmission.

Usage:
    cd LongLive
    python tests/02_test_websocket.py --config_path configs/longlive_inference_le.yaml

Output:
    - output/websocket_test/raw_video.mp4       : Video saved BEFORE websocket (baseline)
    - output/websocket_test/received_video.mp4  : Video reconstructed from websocket frames
    - output/websocket_test/frames_received/    : Individual frames after websocket round-trip
    - output/websocket_test/diagnostics.json    : Timing and comparison stats
"""

import os
import sys
import asyncio
import argparse
import base64
import io
import json
import time
import threading
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Optional
from queue import Queue

# Add parent directory to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

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

OUTPUT_DIR = PROJECT_ROOT / "videos" / "websocket"
TEST_PROMPT = "A cat walking in a garden with flowers, cinematic lighting"
WS_PORT = 18765  # Use non-standard port to avoid conflicts


@dataclass
class FrameData:
    """Data for a single received frame."""
    index: int
    server_index: int
    timestamp: float
    size_bytes: int
    shape: tuple
    data: np.ndarray = field(repr=False)


@dataclass
class DiagnosticReport:
    """Diagnostic information about the websocket test."""
    total_frames_sent: int = 0
    total_frames_received: int = 0
    total_time_seconds: float = 0

    # Timing
    avg_frame_interval_ms: float = 0
    min_frame_interval_ms: float = 0
    max_frame_interval_ms: float = 0

    # Comparison
    frames_match_exactly: int = 0
    max_pixel_diff: int = 0
    avg_pixel_diff: float = 0

    # Issues
    out_of_order_frames: List[dict] = field(default_factory=list)
    missing_frames: List[int] = field(default_factory=list)


def encode_frame(frame: torch.Tensor) -> dict:
    """Encode frame for websocket transmission (same as release_server.py)."""
    frame = (frame.clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
    img = Image.fromarray(frame)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)

    return {
        "data": base64.b64encode(buf.getvalue()).decode("ascii"),
        "format": "jpeg",
    }


def decode_frame(payload: dict) -> np.ndarray:
    """Decode frame from websocket payload."""
    data = base64.b64decode(payload["data"])
    buf = io.BytesIO(data)
    img = Image.open(buf)
    return np.array(img)


async def run_websocket_server(video_frames: torch.Tensor, ready_event: asyncio.Event, results_queue: Queue):
    """
    Run websocket server that sends video frames.

    Args:
        video_frames: Tensor of shape (T, H, W, C) with values in [0, 1]
        ready_event: Event to signal when server is ready
        results_queue: Queue to store sent frame info for comparison
    """
    try:
        import websockets
    except ImportError:
        print("Missing websockets library. Install with: pip install websockets")
        sys.exit(1)

    frames_sent = []

    async def handler(websocket):
        print(f"[Server] Client connected")

        try:
            # Wait for client request
            message = await websocket.recv()
            request = json.loads(message)
            print(f"[Server] Received request: {request.get('prompt', 'no prompt')[:50]}...")

            # Send frames
            print(f"[Server] Sending {len(video_frames)} frames...")
            for idx in range(len(video_frames)):
                frame = video_frames[idx]
                payload = encode_frame(frame)

                # Include frame index for ordering verification
                await websocket.send(json.dumps({
                    "type": "frame",
                    "frame_index": idx,
                    **payload,
                }))

                # Store original frame for comparison
                original_uint8 = (frame.clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
                frames_sent.append({
                    "index": idx,
                    "original": original_uint8,
                })

                if (idx + 1) % 20 == 0:
                    print(f"[Server] Sent {idx + 1}/{len(video_frames)} frames")

            # Send done signal
            await websocket.send(json.dumps({"type": "done"}))
            print(f"[Server] All frames sent")

        except Exception as e:
            print(f"[Server] Error: {e}")
            await websocket.send(json.dumps({"type": "error", "message": str(e)}))

    async with websockets.serve(handler, "localhost", WS_PORT):
        print(f"[Server] Listening on ws://localhost:{WS_PORT}")
        ready_event.set()

        # Wait for client to finish (with timeout)
        await asyncio.sleep(300)  # 5 minute timeout

    # Put sent frames in queue for comparison
    results_queue.put(frames_sent)


async def run_websocket_client(prompt: str, ready_event: asyncio.Event) -> List[FrameData]:
    """
    Run websocket client that receives video frames.

    Returns:
        List of received frames with metadata
    """
    try:
        import websockets
    except ImportError:
        print("Missing websockets library. Install with: pip install websockets")
        sys.exit(1)

    # Wait for server to be ready
    await ready_event.wait()
    await asyncio.sleep(0.5)  # Small delay to ensure server is listening

    frames_received: List[FrameData] = []
    frame_intervals: List[float] = []
    start_time = time.time()
    last_frame_time = start_time

    print(f"[Client] Connecting to ws://localhost:{WS_PORT}...")

    async with websockets.connect(f"ws://localhost:{WS_PORT}", max_size=10 * 1024 * 1024) as websocket:
        print(f"[Client] Connected!")

        # Send request
        request = {"prompt": prompt, "num_samples": 1}
        await websocket.send(json.dumps(request))
        print(f"[Client] Sent request, waiting for frames...")

        while True:
            try:
                message = await asyncio.wait_for(websocket.recv(), timeout=60.0)
                current_time = time.time()
                data = json.loads(message)

                if data["type"] == "frame":
                    elapsed = current_time - start_time
                    interval = current_time - last_frame_time
                    last_frame_time = current_time

                    if len(frames_received) > 0:
                        frame_intervals.append(interval * 1000)

                    # Decode frame
                    frame_array = decode_frame(data)
                    server_idx = data.get("frame_index", len(frames_received))

                    frame = FrameData(
                        index=len(frames_received),
                        server_index=server_idx,
                        timestamp=elapsed,
                        size_bytes=len(data["data"]),
                        shape=frame_array.shape,
                        data=frame_array,
                    )
                    frames_received.append(frame)

                    if (len(frames_received)) % 20 == 0:
                        print(f"[Client] Received {len(frames_received)} frames")

                elif data["type"] == "done":
                    print(f"[Client] Received done signal, total frames: {len(frames_received)}")
                    break

                elif data["type"] == "error":
                    print(f"[Client] Server error: {data.get('message')}")
                    break

            except asyncio.TimeoutError:
                print(f"[Client] Timeout waiting for frame")
                break

    return frames_received


async def run_test(video_frames: torch.Tensor, prompt: str) -> tuple[List[FrameData], List[dict]]:
    """Run both server and client concurrently."""
    ready_event = asyncio.Event()
    results_queue = Queue()

    # Run server and client concurrently
    server_task = asyncio.create_task(
        run_websocket_server(video_frames, ready_event, results_queue)
    )

    # Run client
    frames_received = await run_websocket_client(prompt, ready_event)

    # Cancel server (it's waiting)
    server_task.cancel()
    try:
        await server_task
    except asyncio.CancelledError:
        pass

    # Get sent frames from queue
    frames_sent = results_queue.get() if not results_queue.empty() else []

    return frames_received, frames_sent


def compare_frames(frames_received: List[FrameData], frames_sent: List[dict]) -> DiagnosticReport:
    """Compare received frames with original sent frames."""
    report = DiagnosticReport()
    report.total_frames_sent = len(frames_sent)
    report.total_frames_received = len(frames_received)

    if not frames_received:
        return report

    report.total_time_seconds = frames_received[-1].timestamp

    # Check ordering
    server_indices = [f.server_index for f in frames_received]
    for i in range(1, len(server_indices)):
        if server_indices[i] < server_indices[i-1]:
            report.out_of_order_frames.append({
                "position": i,
                "got": server_indices[i],
                "expected_after": server_indices[i-1],
            })

    # Check for missing frames
    expected_indices = set(range(len(frames_sent)))
    received_indices = set(server_indices)
    report.missing_frames = list(expected_indices - received_indices)

    # Compare pixel values
    pixel_diffs = []
    exact_matches = 0

    for received in frames_received:
        # Find corresponding sent frame
        sent_idx = received.server_index
        if sent_idx < len(frames_sent):
            original = frames_sent[sent_idx]["original"]
            received_arr = received.data

            # Calculate difference
            diff = np.abs(original.astype(np.int16) - received_arr.astype(np.int16))
            max_diff = int(diff.max())
            mean_diff = float(diff.mean())

            pixel_diffs.append({"max": max_diff, "mean": mean_diff})

            if max_diff == 0:
                exact_matches += 1

    report.frames_match_exactly = exact_matches
    if pixel_diffs:
        report.max_pixel_diff = max(d["max"] for d in pixel_diffs)
        report.avg_pixel_diff = sum(d["mean"] for d in pixel_diffs) / len(pixel_diffs)

    return report


def main():
    parser = argparse.ArgumentParser(description="Self-contained websocket video test")
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--prompt", type=str, default=TEST_PROMPT)
    parser.add_argument("--num_frames", type=int, default=60, help="Number of frames to generate")
    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)
    config.num_output_frames = args.num_frames

    device = torch.device("cuda")
    set_seed(config.seed)

    low_memory = get_cuda_free_memory_gb(device) < 40
    low_memory = True

    print(f"\n{'='*70}")
    print("LONGLIVE WEBSOCKET ROUND-TRIP TEST")
    print(f"{'='*70}")
    print(f"Config: {args.config_path}")
    print(f"Prompt: {args.prompt[:50]}...")
    print(f"Frames: {args.num_frames}")
    print(f"Device: {device}")
    print(f"{'='*70}\n")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "frames_received").mkdir(exist_ok=True)

    # -----------------------------------------------------------------------------
    # Load Pipeline and Run Inference
    # -----------------------------------------------------------------------------
    print("[Pipeline] Loading...")
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

    print("[Pipeline] Ready")

    # Run inference
    print(f"\n[Inference] Generating {args.num_frames} frames...")
    sampled_noise = torch.randn(
        [1, config.num_output_frames, 16, 60, 104],
        device=device,
        dtype=torch.bfloat16,
    )

    video, latents = pipeline.inference(
        noise=sampled_noise,
        text_prompts=[args.prompt],
        return_latents=True,
        low_memory=low_memory,
        profile=False,
    )

    # Shape: (B, T, C, H, W) -> (T, H, W, C)
    video_frames = rearrange(video, 'b t c h w -> b t h w c')[0].cpu()
    print(f"[Inference] Generated video shape: {video_frames.shape}")
    print(f"[Inference] Value range: [{video_frames.min():.4f}, {video_frames.max():.4f}]")

    pipeline.vae.model.clear_cache()

    # -----------------------------------------------------------------------------
    # Save Raw Video (baseline)
    # -----------------------------------------------------------------------------
    print(f"\n[Save] Saving raw video (baseline)...")
    raw_video_path = OUTPUT_DIR / "raw_video.mp4"
    raw_video_for_save = 255.0 * video_frames
    write_video(str(raw_video_path), raw_video_for_save, fps=16)
    print(f"[Save] Raw video: {raw_video_path}")

    # -----------------------------------------------------------------------------
    # Run WebSocket Test
    # -----------------------------------------------------------------------------
    print(f"\n[WebSocket] Starting server/client test...")

    frames_received, frames_sent = asyncio.run(run_test(video_frames, args.prompt))

    print(f"\n[WebSocket] Test complete")
    print(f"  Frames sent: {len(frames_sent)}")
    print(f"  Frames received: {len(frames_received)}")

    # -----------------------------------------------------------------------------
    # Save Received Frames and Video
    # -----------------------------------------------------------------------------
    if frames_received:
        print(f"\n[Save] Saving received frames...")

        # Save individual frames
        for frame in frames_received:
            img = Image.fromarray(frame.data)
            img.save(OUTPUT_DIR / "frames_received" / f"frame_{frame.server_index:04d}.png")

        # Reconstruct video from received frames (in server index order)
        frames_sorted = sorted(frames_received, key=lambda f: f.server_index)
        received_video = torch.from_numpy(
            np.stack([f.data for f in frames_sorted], axis=0)
        ).float()

        received_video_path = OUTPUT_DIR / "received_video.mp4"
        write_video(str(received_video_path), received_video, fps=16)
        print(f"[Save] Received video: {received_video_path}")

    # -----------------------------------------------------------------------------
    # Compare and Generate Report
    # -----------------------------------------------------------------------------
    print(f"\n[Analysis] Comparing frames...")
    report = compare_frames(frames_received, frames_sent)

    print(f"\n{'='*70}")
    print("DIAGNOSTIC REPORT")
    print(f"{'='*70}")
    print(f"\n[TRANSMISSION]")
    print(f"  Frames sent:     {report.total_frames_sent}")
    print(f"  Frames received: {report.total_frames_received}")
    print(f"  Total time:      {report.total_time_seconds:.2f}s")

    print(f"\n[PIXEL COMPARISON]")
    print(f"  Exact matches:   {report.frames_match_exactly}/{report.total_frames_received}")
    print(f"  Max pixel diff:  {report.max_pixel_diff}")
    print(f"  Avg pixel diff:  {report.avg_pixel_diff:.2f}")

    issues_found = False
    print(f"\n[ISSUES]")

    if report.out_of_order_frames:
        print(f"  ❌ OUT-OF-ORDER FRAMES: {len(report.out_of_order_frames)}")
        for ooo in report.out_of_order_frames[:3]:
            print(f"      At position {ooo['position']}: got {ooo['got']}, expected > {ooo['expected_after']}")
        issues_found = True

    if report.missing_frames:
        print(f"  ❌ MISSING FRAMES: {report.missing_frames}")
        issues_found = True

    if report.max_pixel_diff > 10:
        print(f"  ⚠️  SIGNIFICANT PIXEL DIFFERENCES (max: {report.max_pixel_diff})")
        print(f"      This is expected with JPEG compression (quality=90)")
        print(f"      To reduce: increase JPEG quality or switch to PNG")
        issues_found = True

    if not issues_found:
        print(f"  ✅ No major issues detected")

    print(f"\n{'='*70}")
    print(f"[OUTPUT FILES]")
    print(f"  Raw video (baseline):    {raw_video_path}")
    print(f"  Received video (via WS): {received_video_path}")
    print(f"  Received frames:         {OUTPUT_DIR / 'frames_received/'}")
    print(f"{'='*70}")

    print(f"\n[NEXT STEPS]")
    print(f"1. Compare raw_video.mp4 vs received_video.mp4")
    print(f"2. If received video flickers but raw doesn't -> websocket/encoding issue")
    print(f"3. If both look the same -> issue is in your frontend rendering")

    # Save report
    report_dict = asdict(report)
    with open(OUTPUT_DIR / "diagnostics.json", "w") as f:
        json.dump(report_dict, f, indent=2)

    print(f"\nDiagnostics saved to {OUTPUT_DIR / 'diagnostics.json'}")


if __name__ == "__main__":
    main()
