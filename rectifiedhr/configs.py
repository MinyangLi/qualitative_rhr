import torch

seed = 2024
num_inference_steps = 50
generator = torch.Generator("cuda").manual_seed(seed)
model_id = "diffusion_models/stable-diffusion-xl-base-1.0"
It_base_path = f"results/"

# For 2048 x 2048
# res_min, res_max = (1024, 1024), (2048, 2048)
# N = 2
# cfg_min, cfg_max, M_cfg = 5, 30, 1
# T_min, T_max, M_T = 40, num_inference_steps, 1

# # For 4096 x 4096
res_min, res_max = (1024, 1024), (4096, 4096)
N = 3
cfg_min, cfg_max, M_cfg = 5, 50, 0.5
T_min, T_max, M_T = 40, num_inference_steps, 0.5

# For 2048 x 4096
# res_min, res_max = (1536, 768), (4096, 2048)
# res_min, res_max = (768, 1536), (2048, 4096)
# M_cfg, M_T, T_max, cfg_min = 1, 1, num_inference_steps, 5
# N = 3 # resize次数, 范围(2, 3), teaser图里的那些我都用的3
# cfg_max = 50 # cfg最大的选取, 范围(50, 30), 控制细节的生成，越大细节越多，但是越容易出现过曝现象
# T_min = 40 # 开始进行resize的step, 范围(35, 40), 控制自由度越大自由度越小，但可以生成更少的重复pattern


prompts = [
    "Background shows an industrial revolution cityscape with smoky skies and tall, metal structures",
    "A giant tree with glowing roots, standing in the center of a mystical forest under a starry sky.",
    "An ancient warrior standing on a mountain peak, gazing at a vast battlefield below, with a stormy sky overhead.",
    "A knight facing a dragon on a cliff edge.",
    "A panda walking down 5th Avenue, beautiful sunset, close up, high definition, 4k.",
    "Warm light spills from the windows of the cottage.",
    "A hero in a cape surrounded by golden butterflies",
]