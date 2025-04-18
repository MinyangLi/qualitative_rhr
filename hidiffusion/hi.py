from hidiffusion import apply_hidiffusion, remove_hidiffusion
from diffusers import StableDiffusionXLPipeline, DDIMScheduler
import torch
import os

# 设置环境变量以减少显存碎片
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 设置固定种子
seed = 23
generator = torch.Generator(device="cuda:3").manual_seed(seed)  # 主设备为 cuda:3

# 预训练模型和调度器设置
pretrain_model = "stabilityai/stable-diffusion-xl-base-1.0"
scheduler = DDIMScheduler.from_pretrained(pretrain_model, subfolder="scheduler")

# 初始化管道（先在 CPU 上加载）
pipe = StableDiffusionXLPipeline.from_pretrained(
    pretrain_model,
    scheduler=scheduler,
    torch_dtype=torch.float16,
    variant="fp16"
)

# 先应用 hidiffusion（在 CPU 上）
apply_hidiffusion(pipe)

# 定义要使用的 GPU 设备
devices = ["cuda:2", "cuda:3", "cuda:4", "cuda:5"]

# 手动分配管道组件到不同 GPU
pipe.unet.to(devices[0])          # UNet 放 cuda:3
pipe.vae.to(devices[1])           # VAE 放 cuda:4
pipe.text_encoder.to(devices[2])  # Text Encoder 放 cuda:5
pipe.text_encoder_2.to(devices[3])# Text Encoder 2 放 cuda:6

# 启用内存优化
pipe.enable_model_cpu_offload(gpu_id=3)  # 主 GPU 为 cuda:3
pipe.enable_vae_tiling()  # 启用 VAE 平铺，降低解码显存需求

# 定义提示词和负向提示词
prompt = "A breathtaking sunset over a serene lake, surrounded by vibrant wildflowers, painted in the style of Claude Monet, with soft glowing light and intricate details, ultra-realistic textures"
negative_prompt = "blurry, dark, dull colors, low resolution, cartoonish, distorted shapes, modern buildings, people"

# 生成 4096x4096 图像
image = pipe(
    prompt,
    guidance_scale=7.5,
    height=4096,
    width=4096,
    eta=1.0,
    negative_prompt=negative_prompt,
    generator=generator,
    num_inference_steps=50  # 明确指定步数，与进度条一致
).images[0]

# 保存图像
output_dir = "results/hidiffusion/"
os.makedirs(output_dir, exist_ok=True)
image.save(os.path.join(output_dir, "golem.jpg"))

# 清理显存
del image
del pipe
for device in devices:
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
print("Image generated and memory cleaned.")