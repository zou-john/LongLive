# SPDX-License-Identifier: Apache-2.0
# Streaming T2V API using CausalChunkInferencePipeline

import os
import argparse
import asyncio
import base64
from typing import List

import torch
from omegaconf import OmegaConf
from einops import rearrange

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from pipeline import CausalChunkInferencePipeline, CausalInferencePipeline
from utils.dataset import TextDataset
from utils.misc import set_seed
from utils.memory import get_cuda_free_memory_gb, DynamicSwapInstaller

# -----------------------------------------------------------------------------
# CONFIG / INIT
# -----------------------------------------------------------------------------

# Debug configuration
DEBUG_SAVE_VIDEOS = True  # Set to False to disable video saving
DEBUG_OUTPUT_DIR = "/tmp/debug_videos"  # Change to Modal volume path like "/data/debug_videos"

torch.set_grad_enabled(False)

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, required=True)
args = parser.parse_args()

config = OmegaConf.load(args.config_path)

device = torch.device("cuda")
set_seed(config.seed)

low_memory = get_cuda_free_memory_gb(device) < 40
low_memory = True  # force for safety

print(f"[Init] Device: {device}")
print(f"[Init] Free VRAM: {get_cuda_free_memory_gb(device):.2f} GB")

# -----------------------------------------------------------------------------
# PIPELINE LOADING
# -----------------------------------------------------------------------------

print("[Init] Loading pipeline...")
pipeline = CausalInferencePipeline(config, device=device)

# Load generator checkpoint
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
# FASTAPI
# -----------------------------------------------------------------------------

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------

from PIL import Image
import io

def encode_frame(frame: torch.Tensor) -> dict:
    frame = (frame.clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
    img = Image.fromarray(frame)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)

    return {
        "data": base64.b64encode(buf.getvalue()).decode("ascii"),
        "format": "jpeg",
    }

def save_video_for_debugging(video_tensor: torch.Tensor, prompt: str, output_dir: str = "/tmp/debug_videos"):
    """
    Save video to disk for debugging purposes - EXACT same logic as inference.py
    
    Args:
        video_tensor: Tensor of shape (T, H, W, C) with values in [0, 1] (already rearranged)
        prompt: Text prompt used for generation
        output_dir: Directory to save videos
    """
    from torchvision.io import write_video
    import re
    import time
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate a safe filename from prompt
    safe_prompt = re.sub(r'[^\w\s-]', '', prompt)[:50]
    safe_prompt = re.sub(r'[-\s]+', '_', safe_prompt)
    timestamp = int(time.time())
    
    # EXACT same as inference.py: multiply by 255.0 (no clamping, no uint8 conversion here)
    video_to_save = 255.0 * video_tensor
    
    output_path = os.path.join(output_dir, f"{safe_prompt}_{timestamp}.mp4")
    write_video(output_path, video_to_save, fps=16)
    print(f"[DEBUG] Saved video to: {output_path}")
    
    return output_path

# -----------------------------------------------------------------------------
# WEBSOCKET ENDPOINT
# -----------------------------------------------------------------------------

@app.websocket("/ws/generate")
async def ws_generate(ws: WebSocket):
    await ws.accept()
    print("[WS] Connected")

    try:
        init = await ws.receive_json()
        prompt: str = init["prompt"]
        num_samples: int = init.get("num_samples", 1)
        blocks_per_chunk: int = init.get("blocks_per_chunk", 5)

        prompts = [prompt] * num_samples

        sampled_noise = torch.randn(
            [
                num_samples,
                config.num_output_frames,
                16,
                60,
                104,
            ],
            device=device,
            dtype=torch.bfloat16,
        )

        print("[WS] Starting streaming inference")

        async def stream():
            try:
                # causal infernece (non streaming)
                video, latents = pipeline.inference(
                    noise=sampled_noise,
                    text_prompts=prompts,
                    return_latents=True,
                    low_memory=low_memory,
                    profile=False,
                )
                
                current_video = rearrange(video, 'b t c h w -> b t h w c').cpu()
                
                # Save video for debugging BEFORE streaming
                # This ensures we save exactly what we're about to send
                if DEBUG_SAVE_VIDEOS:
                    debug_video_path = save_video_for_debugging(
                        current_video[0],  # First sample
                        prompt,
                        output_dir=DEBUG_OUTPUT_DIR
                    )
                    print(f"[DEBUG] Video saved to: {debug_video_path}")
                
                # Stream frames to client
                print(f"[DEBUG] Streaming {len(current_video[0])} frames to client...")
                for idx, frame in enumerate(current_video[0]):  # Iterate over frames of the first sample
                    payload = encode_frame(frame)
                    await ws.send_json({
                        "type": "frame",
                        **payload,
                    })
                    # Log progress every 20 frames
                    if (idx + 1) % 20 == 0:
                        print(f"[DEBUG] Sent {idx + 1}/{len(current_video[0])} frames")
                print("[WS] All frames sent - Johnny")

                # print(
                #     "[WS] Using chunked causal inference for streaming generation"
                # )

                # for video_chunk, _, is_final in pipeline.chunk_inference(
                #     noise=sampled_noise,
                #     text_prompts=prompts,
                #     blocks_per_chunk=blocks_per_chunk,
                #     low_memory=low_memory,
                # ):
                #     # video_chunk: (B, T, C, H, W)
                #     frames = rearrange(
                #         video_chunk, "b t c h w -> b t h w c"
                #     )

                #     # send frames
                #     for t in range(frames.shape[1]):
                #         frame = frames[0, t]
                #         payload = encode_frame(frame)
                #         await ws.send_json({
                #             "type": "frame",
                #             **payload,
                #         })
                #     # NOTE: problem is here 
                #     if is_final:
                #         await ws.send_json({"type": "done"})
                #         break

                pipeline.vae.model.clear_cache()
                print("[WS] Generation complete")

            except Exception as e:
                await ws.send_json({
                    "type": "error",
                    "message": str(e),
                })
                raise

        await stream()

    except WebSocketDisconnect:
        print("[WS] Disconnected")
    except Exception as e:
        print("[WS] Error:", e)
        await ws.close()

# -----------------------------------------------------------------------------
# HEALTH
# -----------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "release_server:app",  # <-- your filename without .py
        host="0.0.0.0",
        port=8000,
        reload=False,          # reload can cause GPU OOM, better off for prod
    )