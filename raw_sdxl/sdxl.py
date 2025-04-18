from diffusers import DiffusionPipeline
import torch
import json
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--cuda_index', type=int, default=0, help="CUDA device index to use")
args = parser.parse_args() 

# Set environment variable to reduce memory fragmentation (optional but recommended)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Set a fixed seed for reproducibility
seed = 2024  # You can change this to any integer value
generator = torch.Generator(device=f"cuda:{args.cuda_index}").manual_seed(seed)

# Load the pipeline with memory-efficient settings
pipe = DiffusionPipeline.from_pretrained(
    "diffusion_models/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,  # Use half-precision to reduce memory usage
    use_safetensors=True,
    variant="fp16"
)
pipe.to(f"cuda:{args.cuda_index}")

# Enable memory optimizations
pipe.enable_vae_tiling()  # Break VAE decoding into tiles to reduce memory demand
# Optional: Uncomment the following if still encountering OOM
# pipe.enable_model_cpu_offload()  # Offload parts of the model to CPU
# pipe.enable_xformers_memory_efficient_attention()  # Optimize attention mechanism (if available)

# Read prompts from prompts.json
with open("9_grids/prompts_selected4.json", "r") as f:
    data = json.load(f)
    prompt_pairs = data["images"]

# Ensure output directory exists
output_dir = "results/"
os.makedirs(output_dir, exist_ok=True)

# Generate and save images for each prompt
for pair in prompt_pairs:
    prompt = pair["prompt"]
    filename = pair["filename"]
    output_path = os.path.join(output_dir, filename)
    os.makedirs(output_path, exist_ok=True)
    
    # Generate image with the fixed seed generator
    # Reduced resolution as an example; adjust as needed
    image = pipe(
        prompt=prompt,
        width=4096,  # High resolution; reduce if OOM persists (e.g., 2048)
        height=4096,  # High resolution; reduce if OOM persists (e.g., 2048)
        generator=generator,
        guidance_scale=7.5,  # Default value; adjust if needed
        num_inference_steps=50  # Explicit steps; reduce if OOM persists (e.g., 30)
    ).images[0]

    # Save image with a unique filename
    for folder_name in os.listdir(output_path):
        saved_path = os.path.join(output_path, folder_name)
        if not os.path.isdir(saved_path):
            continue
        saved_path = os.path.join(saved_path, "sdxl.jpg")
        image.save(saved_path)
        print(f"Saved {saved_path}")
    
    # Clean up memory after each image
    del image
    torch.cuda.empty_cache()

# Final cleanup
del pipe
torch.cuda.empty_cache()
print("All images generated and memory cleaned.")