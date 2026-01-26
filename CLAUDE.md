# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LongLive is an NVIDIA Labs deep learning framework for generating long videos (up to 240 seconds) with real-time interactive prompt capabilities. It uses frame-level autoregressive diffusion with causal attention and KV-caching, built on the Wan2.1-T2V-1.3B base model.

## Commands

### Installation
```bash
conda create -n longlive python=3.10 -y && conda activate longlive
conda install nvidia/label/cuda-12.4.1::cuda cudatoolkit
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

### Download Models
```bash
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir wan_models/Wan2.1-T2V-1.3B
huggingface-cli download Efficient-Large-Model/LongLive --local-dir longlive_models
```

### Inference
```bash
bash inference.sh                    # Single-prompt video generation
bash interactive_inference.sh        # Interactive multi-prompt generation
```

### Training
```bash
bash train_init.sh                   # Phase 1: Self-Forcing initialization (21-frame, short window + frame sink)
bash train_long.sh                   # Phase 2: Streaming long tuning (up to 240 frames)
```

All scripts use `torchrun` for distributed execution. Training uses 8 GPUs per node.

## Architecture

### Key Concepts

**KV-Cache**: Stores Key/Value matrices from previous frames during autoregressive generation. Per-transformer-block caching for the 30-block model.

**Frame Sink**: Attention sink tokens that summarize history for long-range consistency. Combined with local attention window (default 12 frames).

**KV-Recache**: For interactive prompt switching - recaches recent N frames (15-21) with new prompt encoding for smooth transitions. See `InteractiveCausalInferencePipeline._recache_after_switch()`.

**Streaming Long Tuning**: Trains on long sequences by processing in chunks (21 frames), reusing historical KV cache between chunks.

### Core Components

| Component | Location | Purpose |
|-----------|----------|---------|
| `CausalInferencePipeline` | `pipeline/causal_inference.py` | Single-prompt inference with KV-cache |
| `InteractiveCausalInferencePipeline` | `pipeline/interactive_causal_inference.py` | Multi-prompt with recaching |
| `CausalWanModel` | `wan/modules/causal_model.py` | Causal attention transformer |
| `CausalWanModel` (Infinity) | `wan/modules/causal_model_infinity.py` | Blockwise relative RoPE for infinite-length |
| `WanDiffusionWrapper` | `utils/wan_wrapper.py` | Wraps Wan model with diffusion pipeline |
| `ScoreDistillationTrainer` | `trainer/distillation.py` | Main training loop |

### Data Flow
```
Text Prompt → WanTextEncoder (T5) → Prompt Embeddings
→ Initialize noise → CausalInferencePipeline.inference()
  (denoise through timesteps [1000, 750, 500, 250] with KV-cache)
→ Generated Latents → WanVAEWrapper.decode_to_pixel() → MP4
```

## Configuration

Configs use OmegaConf YAML, loaded from `configs/default_config.yaml` then merged with specific config.

**Key Parameters:**
- `model_kwargs.local_attn_size`: Local attention window (default: 12 frames)
- `model_kwargs.sink_size`: Number of sink tokens (default: 3)
- `denoising_step_list`: Timesteps to denoise (default: [1000, 750, 500, 250])
- `streaming_chunk_size`: Training chunk size (default: 21)
- `num_output_frames`: Output video length

**Config Selection:**
- Inference: `longlive_inference.yaml`
- Interactive: `longlive_interactive_inference.yaml`
- Infinite-length: `longlive_inference_infinity.yaml`
- Training Phase 1: `longlive_train_init.yaml`
- Training Phase 2: `longlive_train_long.yaml`

## Code Patterns

**FSDP**: All training uses `FullyShardedDataParallel`. State dict loading requires cleaning `_fsdp_wrapped_module.` prefix (see `_cleanup_key()` functions).

**Mixed Precision**: Default `torch.bfloat16` for generator; encoder/VAE stay FP32.

**Flash Attention**: FA3 auto-enabled on Hopper GPUs (H100/H800/H20, CC 9.0+), FA2 fallback on Ampere. Detection via `torch.cuda.get_device_capability()`.

**Debug Flags**: `utils/debug_option.py` contains `DEBUG`, `LOG_GPU_MEMORY`, `DEBUG_GRADIENT`.

**Memory Management**: `utils/memory.py` has GPU utilities; `DynamicSwapInstaller` swaps models between GPU/CPU.

## Interactive Prompt Format

From `example/interactive_example.jsonl`:
```json
{"prompts": [
  {"text": "Initial prompt...", "end_frame": 21},
  {"text": "New prompt after switch...", "end_frame": 42}
]}
```

Loaded via `MultiTextDataset` which extracts text sequences and switch points.
