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

def encode_frame(frame: torch.Tensor, debug=False) -> dict:
    # Frame should be in range [0, 1] from the pipeline
    # Let's check the actual range
    if debug:
        print(f"Frame range: [{frame.min().item():.4f}, {frame.max().item():.4f}]")
    
    # Don't clamp - let's see if values are actually outside [0, 1]
    # If they are, we need to understand why
    frame = (frame * 255.0).to(torch.uint8)
    
    # Ensure it's on CPU and contiguous
    if frame.is_cuda:
        frame = frame.cpu()
    frame = frame.contiguous()
    
    frame_np = frame.numpy()
    img = Image.fromarray(frame_np)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95, optimize=False)

    return {
        "data": base64.b64encode(buf.getvalue()).decode("ascii"),
        "format": "jpeg",
    }

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

        # Clear VAE cache before starting
        pipeline.vae.model.clear_cache()
        torch.cuda.empty_cache()
        
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
                # Generate video (returns values in [0, 1] range)
                video, latents = pipeline.inference(
                    noise=sampled_noise,
                    text_prompts=prompts,
                    return_latents=True,
                    low_memory=low_memory,
                    profile=False,
                )
                
                # Match inference.py processing exactly
                torch.cuda.synchronize()
                
                # DEBUG: Check video tensor stats before rearrange
                print(f"[DEBUG] Video tensor - shape: {video.shape}, dtype: {video.dtype}")
                print(f"[DEBUG] Video range: [{video.min().item():.4f}, {video.max().item():.4f}]")
                print(f"[DEBUG] Video mean: {video.mean().item():.4f}, std: {video.std().item():.4f}")
                
                current_video = rearrange(video, 'b t c h w -> b t h w c').cpu()
                
                # DEBUG: Check per-frame stats
                print(f"[DEBUG] Checking first 5 frames:")
                for i in range(min(5, current_video.shape[1])):
                    frame = current_video[0, i]
                    print(f"  Frame {i}: min={frame.min().item():.4f}, max={frame.max().item():.4f}, mean={frame.mean().item():.4f}")
                
                # Clear cache before sending frames
                pipeline.vae.model.clear_cache()
                
                # Clean up GPU memory
                del video, latents
                torch.cuda.empty_cache()
                
                # Send frames one by one
                for idx, frame in enumerate(current_video[0]):
                    # Debug first, middle, and last frames
                    debug = (idx < 3) or (idx == len(current_video[0]) // 2) or (idx >= len(current_video[0]) - 3)
                    payload = encode_frame(frame, debug=debug)
                    await ws.send_json({
                        "type": "frame",
                        **payload,
                    })
                
                del current_video
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