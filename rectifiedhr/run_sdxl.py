import torch, os
from diffusers import DDIMScheduler
from pipelines.pipeline_sdxl import SDXLPipeline
from diffusers.utils.torch_utils import randn_tensor
from utils.preprocess import latent2image, image2latent
from utils.main_tools import quantic_HW, quantic_cfg, quantic_step
from configs import (
    model_id, prompts, generator, It_base_path, num_inference_steps,
    N, cfg_min, cfg_max, M_cfg, T_min, T_max, M_T, res_min, res_max,
)
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--cuda_index', type=int, default=0, help="CUDA device index to use")
args = parser.parse_args() 

def main():
    # 1. init
    refresh_step_list = quantic_step(T_min, T_max, N, M_T)
    cfg_list = quantic_cfg(cfg_min, cfg_max, N, M_cfg)
    resolution_list = quantic_HW(res_min, res_max, N)
    os.makedirs(It_base_path, exist_ok=True)
    pipe = SDXLPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(f"cuda:{args.cuda_index}")
    pipe.vae.enable_tiling()
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    MEMORY = {}

    textprompts=[]
    negative_prompts=[]
    filenames=[]
    with open("9_grids/prompts_selected4.json", "r") as f:
        data = json.load(f)
        prompt_pairs = data["images"]
    for pair in prompt_pairs:
        prompt = pair["prompt"]
        ng_prompt = pair["negative_prompt"]
        textprompts.append(prompt)
        negative_prompts.append(ng_prompt)
        filename = f"{pair['filename']}"
        filenames.append(filename)

    # 2. run
    for idx in range(len(textprompts)):
        prompt = textprompts[idx]
        negative_prompt = negative_prompts[idx]
        filename = filenames[idx]
        MEMORY.update({
            'predict_x0_list': [],
        })
        start_latent = randn_tensor((1, 4, res_min[0] // pipe.vae_scale_factor, res_min[1] // pipe.vae_scale_factor), dtype=pipe.dtype, device=pipe.device, generator=generator)
        for i in range(len(resolution_list)):
            if i == 0:
                predict_x0_latent_noisey = start_latent
            else:
                predict_x0_latent = image2latent(latent2image(MEMORY['predict_x0_list'][-1], pipe).resize((resolution_list[i][1], resolution_list[i][0])), pipe)
                noise_ = randn_tensor(predict_x0_latent.shape, dtype=predict_x0_latent.dtype, device=predict_x0_latent.device, generator=generator)
                predict_x0_latent_noisey = pipe.scheduler.add_noise(predict_x0_latent, noise_, pipe.scheduler.timesteps[len(MEMORY['predict_x0_list']) - num_inference_steps])
            hr_output = pipe(
                    prompt=prompt,
                    negative_prompt = negative_prompt,
                    generator=generator,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=cfg_list[i],
                    latents=predict_x0_latent_noisey,
                    MEMORY=MEMORY,
                    strength=(num_inference_steps - len(MEMORY['predict_x0_list'])) / num_inference_steps,
                    denoising_end=refresh_step_list[i + 1] / num_inference_steps,
                )
        output_path = os.path.join(It_base_path, filename)
        saved_path = os.path.join(output_path, "4096_4096_jpg")
        os.makedirs(saved_path, exist_ok=True)
        hr_output.images[0].save(os.path.join(saved_path, f'Ours.jpg'))



if __name__ == '__main__':
    main()



