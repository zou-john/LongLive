PROMPT_1 = """
A wide cinematic establishing shot of a massive futuristic metropolis at dawn, towering glass skyscrapers stretching into low clouds, neon signage still glowing faintly as the sun rises. Flying vehicles glide between buildings on invisible lanes. The camera slowly dollies forward from above the city, soft golden light reflecting off steel and glass, ultra-realistic, 4K, cinematic depth of field, calm and anticipatory mood.
"""
PROMPT_2 = """
Street-level view inside the same metropolis during early morning rush. Diverse crowds of pedestrians in modern cyberpunk fashion walk past holographic billboards and transparent storefronts. Autonomous cars silently pass by. The camera tracks sideways at human height, shallow depth of field, subtle motion blur, cool blue tones mixed with warm sunlight, hyper-detailed textures, realistic lighting.
""" 

PROMPT_3 = """
Interior shot of a high-rise control room overlooking the city. Floor-to-ceiling windows show the metropolis now fully awake. Floating holographic city maps, data streams, and glowing UI elements hover in the air. A single figure stands in silhouette facing the window. The camera slowly pushes in from behind, dramatic contrast lighting, cinematic sci-fi atmosphere, ultra-sharp details.
"""

PROMPT_4 = """
A dynamic aerial chase through the metropolis at sunset. Flying vehicles weave between skyscrapers at high speed, neon lights reflecting off metallic surfaces. The camera follows closely behind one vehicle, fast motion, dramatic perspective shifts, orange-purple sky, intense energy, high frame-rate cinematic action, realistic physics, lens flares.
"""

PROMPT_5 = """
Nighttime panoramic shot of the entire metropolis, now glowing with thousands of lights. Rain begins to fall, creating reflections on rooftops and streets far below. The camera slowly pulls back and upward, revealing the city stretching endlessly into the horizon. Moody cyberpunk color palette, reflective surfaces, soft rain particles, epic and contemplative ending, ultra-cinematic realism.
"""
# reprompt entry happens at prompt 2 to 5

REPROMPT_3 = """Interior of a towering high-rise command center at dusk, overlooking a vast futuristic metropolis. Floor-to-ceiling glass walls reveal the city transitioning from day to night. Floating holographic interfaces display live traffic flows, energy grids, and city diagnostics. A lone figure stands at the center console, softly illuminated by blue holographic light. The camera slowly orbits behind the figure, cinematic sci-fi realism, controlled and tense atmosphere, ultra-detailed.
"""

REPROMPT_4 = """
The city suddenly surges into motion at night. A high-speed aerial sequence weaving between illuminated skyscrapers and glowing skyways. Emergency lights ripple across buildings as autonomous flying vehicles accelerate and diverge. The camera aggressively follows one vehicle from behind, sharp turns, dramatic parallax, neon reflections, rain beginning to streak across the lens, intense cyberpunk action, realistic motion blur.
"""

REPROMPT_5 = """
A wide, contemplative aerial shot above the metropolis late at night. Rain pours steadily, coating rooftops and streets in reflective light. The chaos below stabilizes as traffic patterns smooth out. The camera slowly pulls upward and backward, revealing the full city glowing beneath storm clouds. Neon lights shimmer through rain and mist, moody futuristic palette, epic yet calm resolution, cinematic realism.
"""

# regular packages
import argparse
import os 
from typing import List

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from tqdm import tqdm
from torch.utils.data import DataLoader, SequentialSampler
from torchvision.io import write_video
from torchvision import transforms
from einops import rearrange

from utils.misc import set_seed
from utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller

from pipeline import InteractiveCausalInferencePipeline

# arugment parsing
parser = argparse.ArgumentParser("Interactive causal inference")
parser.add_argument("--config_path", type=str, help="Path to the config file")
args = parser.parse_args()

# load the config and prepare for one GPU
config = OmegaConf.load(args.config_path)
local_rank = 0
rank = 0
device = torch.device("cuda")
set_seed(config.seed)
print(f"Single GPU mode on device {device}")

# memory optimization
low_memory = get_cuda_free_memory_gb(device) < 40
torch.set_grad_enabled(False)

# initialize the pipeline and load the pretrained weights 
pipeline = InteractiveCausalInferencePipeline(config, device)

if config.generator_ckpt:
    state_dict = torch.load(config.generator_ckpt, map_location="cpu")
    raw_gen_state_dict = state_dict["generator_ema" if config.use_ema else "generator"]

    if config.use_ema:
        def _clean_key(name: str) -> str:
            return name.replace("_fsdp_wrapped_module.", "")

        cleaned_state_dict = {_clean_key(k): v for k, v in raw_gen_state_dict.items()}
        missing, unexpected = pipeline.generator.load_state_dict(
            cleaned_state_dict, strict=False
        )
        if local_rank == 0:
            if missing:
                print(f"[Warning] {len(missing)} parameters missing: {missing[:8]} ...")
            if unexpected:
                print(f"[Warning] {len(unexpected)} unexpected params: {unexpected[:8]} ...")
    else:
        pipeline.generator.load_state_dict(raw_gen_state_dict)

# lora for optimization (from config/longlive_interactive_inference_le.yaml)
pipeline.is_lora_enabled = False
from utils.lora_utils import configure_lora_for_model
import peft
if getattr(config, "adapter", None) and configure_lora_for_model is not None:
    if local_rank == 0:
        print(f"LoRA enabled with config: {config.adapter}")
        print("Applying LoRA to generator (inference)...")
    
    # after loading base weights, apply LoRA wrapper to the generator's transformer model
    pipeline.generator.model = configure_lora_for_model(
        pipeline.generator.model,
        model_name="generator",
        lora_config=config.adapter,
        is_main_process=(local_rank == 0),
    )

    # load LoRA weights (if lora_ckpt is provided)
    lora_ckpt_path = getattr(config, "lora_ckpt", None)
    if lora_ckpt_path:
        if local_rank == 0:
            print(f"Loading LoRA checkpoint from {lora_ckpt_path}")
        lora_checkpoint = torch.load(lora_ckpt_path, map_location="cpu")
        # support both formats: containing the `generator_lora` key or a raw LoRA state dict
        if isinstance(lora_checkpoint, dict) and "generator_lora" in lora_checkpoint:
            peft.set_peft_model_state_dict(pipeline.generator.model, lora_checkpoint["generator_lora"])  # type: ignore
        else:
            peft.set_peft_model_state_dict(pipeline.generator.model, lora_checkpoint)  # type: ignore
        if local_rank == 0:
            print("LoRA weights loaded for generator")
    else:
        if local_rank == 0:
            print("No LoRA checkpoint specified; using base weights with LoRA adapters initialized")

    pipeline.is_lora_enabled = True

# device setup
print("dtype", pipeline.generator.model.dtype)
pipeline = pipeline.to(dtype=torch.bfloat16)
if low_memory:
    DynamicSwapInstaller.install_model(pipeline.text_encoder, device=device)
pipeline.generator.to(device=device)
pipeline.vae.to(device=device)

# build prompts 
original_prompts = [PROMPT_1, PROMPT_2, PROMPT_3, PROMPT_4, PROMPT_5]
reprompted_prompts = [REPROMPT_3, REPROMPT_4, REPROMPT_5]

# convert switch_frame_indices to list of ints 
if isinstance(config.switch_frame_indices, int):
    switch_frame_indices: List[int] = [int(config.switch_frame_indices)]
else:
    switch_frame_indices: List[int] = [
        int(x) for x in str(config.switch_frame_indices).split(",") if str(x).strip()
    ]

# my arguments
num_segments = 5
assert len(switch_frame_indices) == num_segments - 1, (
    "The number of switch_frame_indices should be the number of prompt segments minus 1"
)

# create the batch data dimensions
prompt_list = original_prompts

# inference the original prompt_list
sampled_noise = torch.randn(
    [
        config.num_samples,  # 1
        config.num_output_frames,  # 120 frames for 30 seconds
        16,  # channels = 16
        60,  # height = 480 // 8 = 60
        104,  # width = 832 // 8 = 104
    ],
    device=device,
    dtype=torch.bfloat16,
)

# Step 1: Generate original video with checkpoints enabled
print("=" * 50)
print("Step 1: Generating original video with checkpoints")
print("=" * 50)
video = pipeline.inference(
    noise=sampled_noise,
    text_prompts_list=prompt_list,
    switch_frame_indices=switch_frame_indices,
    return_latents=False,
    save_checkpoints=True,  # Enable checkpoint saving
)

# Save original video
current_video = rearrange(video, "b t c h w -> b t h w c").cpu() * 255.0
output_path = os.path.join(config.output_folder, "myprompt-original.mp4")
write_video(output_path, current_video[0].to(torch.uint8), fps=16)
print(f"Original video saved to {output_path}")
print(f"Available checkpoints: {list(pipeline.checkpoints.keys())}")

# Step 2: Edit from scene 3 onwards using checkpoint[2] (after scene 2)
# This retains scenes 1-2 and regenerates scenes 3-5 with new prompts
print("=" * 50)
print("Step 2: Editing scenes 3-5 from checkpoint[2]")
print("=" * 50)

# Build new prompt list: keep P1, P2, use REPROMPT for 3, 4, 5
edited_prompts = [PROMPT_1, PROMPT_2, REPROMPT_3, REPROMPT_4, REPROMPT_5]

video_edited = pipeline.inference_from_checkpoint(
    noise=sampled_noise,
    restore_from_scene=2,  # Restore checkpoint[2] (after scene 2 completes)
    text_prompts_list=edited_prompts,
    switch_frame_indices=switch_frame_indices,
    return_latents=False,
    save_checkpoints=True,  # Save new checkpoints for further editing
)

# Save edited video
edited_video = rearrange(video_edited, "b t c h w -> b t h w c").cpu() * 255.0
output_path = os.path.join(config.output_folder, "myprompt-edited.mp4")
write_video(output_path, edited_video[0].to(torch.uint8), fps=16)
print(f"Edited video saved to {output_path}")

# Clean up checkpoints to free memory
pipeline.clear_checkpoints()
print("Checkpoints cleared.")


