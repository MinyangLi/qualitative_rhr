from diffusers import StableDiffusionXLPipeline, DDIMScheduler
import torch
import json
import os
import argparse
import sys
# sys.path.append(os.path.dirname(__file__)) 
# print(f'current dir: {os.path.dirname(__file__)}')
# print(f'sys.path {sys.path}')
from hidiffusion import apply_hidiffusion, remove_hidiffusion

parser = argparse.ArgumentParser()
parser.add_argument('--cuda_index', type=int, default=0, help="CUDA device index to use")
args = parser.parse_args() 

# 
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

method_name='hidiffusion'

# 
seed = 2024  # 
generator = torch.Generator(device=f"cuda:{args.cuda_index}").manual_seed(seed)

# 
pretrain_model = "diffusion_models/stable-diffusion-xl-base-1.0"
scheduler = DDIMScheduler.from_pretrained(pretrain_model, subfolder="scheduler")
pipe = StableDiffusionXLPipeline.from_pretrained(
    pretrain_model,
    scheduler=scheduler,
    torch_dtype=torch.float16,
    variant="fp16"
).to(f"cuda:{args.cuda_index}")

# 
pipe.enable_vae_tiling()  # 
# pipe.enable_model_cpu_offload()  # 
# pipe.enable_xformers_memory_efficient_attention()  # 

# 
output_dir = "results/"
os.makedirs(output_dir, exist_ok=True)

# 
apply_hidiffusion(pipe)

# 
with open("9_grids/prompts_selected4.json", "r") as f:
    data = json.load(f)
    prompt_pairs = data["images"]


# 
for pair in prompt_pairs:
    resolutions = pair["resolutions"]
    print(resolutions)
    # Iterate over all resolutions
    for resolution in [resolutions[1]]:
        print(resolution)
        # Parse the string into width and height
        height, width = map(int, resolution.split(','))
        filename = f"{pair['filename']}"
        output_path=os.path.join(output_dir,filename)
        output_path=os.path.join(output_path,f'{height}_{width}_jpg')
        os.makedirs(output_path, exist_ok=True)
        image = pipe(
            prompt=pair["prompt"],
            guidance_scale=7.5,
            height=height,
            width=width,
            eta=1.0,
            negative_prompt=pair["negative_prompt"],
            generator=generator,  
            num_inference_steps=50  
        ).images[0]
        
        # 
        image.save(os.path.join(output_path, f"{method_name}.jpg"))
        print(f"Saved to {os.path.join(output_path, method_name + '.jpg')}")
        
        #
        del image
        torch.cuda.empty_cache()

#
remove_hidiffusion(pipe)
del pipe
torch.cuda.empty_cache()
print("All images generated and memory cleaned.")