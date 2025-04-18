import torch
import sys
import os
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--cuda_index', type=int, default=0, help="CUDA device index to use")
args = parser.parse_args() 

# 获取当前脚本目录并添加到 sys.path
# current_dir = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(current_dir)
# print(f'current file: {current_dir}')
# print(f'sys path is {sys.path}')

from pipeline_demofusion_sdxl import DemoFusionSDXLPipeline

method_name='demofusion'

# 确保输出目录存在
output_dir = "results/"
os.makedirs(output_dir, exist_ok=True)

seed = 2024  # 
generator = torch.Generator(device=f"cuda:{args.cuda_index}").manual_seed(seed)
# 预训练模型设置，指定缓存目录
print('This is demofusion working')
model_ckpt = "diffusion_models/stable-diffusion-xl-base-1.0"
pipe = DemoFusionSDXLPipeline.from_pretrained(
    model_ckpt,
    torch_dtype=torch.float16,
    variant="fp16"
)
pipe = pipe.to(f"cuda:{args.cuda_index}")

with open("9_grids/prompts_selected4.json", "r") as f:
    data = json.load(f)
    prompt_pairs = data["images"]

# 为每个提示词生成不同分辨率的图像
for pair in prompt_pairs:
    resolutions = pair["resolutions"]
    # print(resolutions)
    # Iterate over all resolutions
    for resolution in [resolutions[1]]:
        # print(resolution)
        # Parse the string into width and height
        height, width = map(int, resolution.split(','))
        filename = f"{pair['filename']}"
        output_path=os.path.join(output_dir,filename)
        output_path=os.path.join(output_path,f'{height}_{width}_jpg')
        os.makedirs(output_path, exist_ok=True)
        images = pipe(
            prompt=pair["prompt"],
            negative_prompt=pair["negative_prompt"],
            height=height,
            width=width,
            generator=generator,
            view_batch_size=16,
            stride=64,
            num_inference_steps=50,
            guidance_scale=7.5,
            cosine_scale_1=3,
            cosine_scale_2=1,
            cosine_scale_3=1,
            sigma=0.8,
            multi_decoder=True,
            show_image=False
        )
        for i, image in enumerate(images):
            image.save(os.path.join(output_path, f"{method_name}.jpg"))
            print(f"Saved to {os.path.join(output_path, method_name + '.jpg')}")