#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import functools
from functools import partial
import gc
import itertools
import json
import math
import os
import sys
import time
import random
import shutil
from pathlib import Path
from typing import List, Union
import cv2

from loguru import logger

import accelerate
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torchvision
import torchvision.transforms.v2.functional as TF
import transformers
import webdataset as wds
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from braceexpand import braceexpand
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torch.utils.data import default_collate
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, CLIPTextModel, PretrainedConfig
from webdataset.tariterators import (
    base_plus_ext,
    tar_file_expander,
    url_opener,
    valid_sample,
)

import diffusers
from diffusers import (
    DDIMScheduler,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    StableDiffusion3Pipeline
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from safetensors import safe_open

MAX_SEQ_LENGTH = 77

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.25.0")

from utils.wrapper_utils import wrap, unwrap, obtain_origin
from utils.sd_utils import vae_encode, vae_decode, sdxl_vae_wrapper, sdxl_vae_unwrapper
from utils.facfg_wrapper import facfg_wrapper, facfg_unwrapper
from utils.scheduler_wrapper import scheduler_wrapper, scheduler_unwrapper
from utils.camap_wrapper import camap_wrapper, camap_unwrapper
from utils.scheduling_flow_match_euler_discrete_modified import FlowMatchEulerDiscreteModifiedScheduler

# 设置 PyTorch 内存优化（第三个方法）
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--output_dir", type=str, default="results/", help="The output directory where the model predictions will be written.")
    parser.add_argument("--seed", type=int, default=2024, help="A seed for reproducible training.")
    parser.add_argument("--prompts", default="an elegant woman is drinking coffee", help="the prompt str or prompts file.")
    parser.add_argument("--images-per-prompt", type=int, default=1, help="The generated images for each prompt.")
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--msp_endtimes", nargs="+", type=int, default=[200,0])
    parser.add_argument("--msp_steps", nargs="+", type=int, default=[40,10])
    parser.add_argument("--msp_gamma", type=float, default=2.0)
    parser.add_argument("--gs", type=float, default="5.0")
    parser.add_argument("--tsize", type=str, default="[[768,1536],[2048,4096]]")
    parser.add_argument("--name", type=str, default="sdxl", choices=["sd21", "sdxl", "sd3"])
    parser.add_argument("--vae_tiling", action="store_true")
    parser.add_argument("--facfg_weight", nargs="+", type=float, default=[7.5])
    parser.add_argument("--facfg_mode", type=str, default="bilinear")
    parser.add_argument('--cuda_index', type=int, default=0, help="CUDA device index to use")
    parser.add_argument("--camap_weight", type=float, default=1.0)

    args = parser.parse_args()
    args.tsize = eval(args.tsize)

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    return args

def main(args):
    # ===== prepare the environment =====
    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    logger.info(f"weight_dtype: {weight_dtype}")

    # ===== FreCaS =====
    pipeline_paths = {
        "sd21": "/home/notebook/data/group/LowLevelLLM/models/diffusion_models/stable-diffusion-2-1-base",
        "sdxl": "diffusion_models/stable-diffusion-xl-base-1.0",
        "sd3": "/home/notebook/data/group/LowLevelLLM/models/diffusion_models/stable-diffusion-3-medium-diffusers",
    }

    # --- load the pipeline ---
    if args.name == "sd21":
        pipeline = StableDiffusionPipeline.from_pretrained(pipeline_paths[args.name], torch_dtype=weight_dtype, safety_checker=None)
        pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config, torch_dtype=weight_dtype, device=f"cuda:{args.cuda_index}")
    if args.name == "sdxl":
        pipeline = StableDiffusionXLPipeline.from_pretrained(pipeline_paths[args.name], torch_dtype=weight_dtype, safety_checker=None)
        pipeline.watermark = None
        pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config, torch_dtype=weight_dtype, device=f"cuda:{args.cuda_index}")
    if args.name == "sd3":
        pipeline = StableDiffusion3Pipeline.from_pretrained(pipeline_paths[args.name], text_encoder_3=None, tokenizer_3=None, torch_dtype=weight_dtype)
        pipeline.scheduler = FlowMatchEulerDiscreteModifiedScheduler.from_config(pipeline.scheduler.config, torch_dtype=weight_dtype)

    # --- move the pipeline to cuda ---
    pipeline.to(f"cuda:{args.cuda_index}")
    pipeline.vae.to(f"cuda:{args.cuda_index}")
    if isinstance(pipeline, StableDiffusion3Pipeline):
        pipeline.transformer.to(f"cuda:{args.cuda_index}")
    else:
        pipeline.unet.to(f"cuda:{args.cuda_index}")

    pipeline._weight_dtype = weight_dtype
    pipeline._device = pipeline.vae.device

    if isinstance(pipeline, StableDiffusionXLPipeline):
        sdxl_vae_wrapper(pipeline)

    if args.vae_tiling:
        pipeline.enable_vae_tiling()
    logger.info(f"create pipeline: {args.name}")

    # ===== enable pipeline to use CA-maps reutilization and FA-CFGs =====
    camap_wrapper(pipeline, camap_w=args.camap_weight)
    facfg_wrapper(pipeline, gs=args.gs, gsw=args.facfg_weight, mode=args.facfg_mode, sizes=[(sh//8,sw//8) for sh,sw in args.tsize])

    # --- set the schedulers ---
    schedulers = []
    for i in range(len(args.tsize)):
        schedulers.append({
            "endtime": args.msp_endtimes[i],
            "steps": args.msp_steps[i],
            "size": args.tsize[i],
        })
    scheduler_wrapper(pipeline, schedulers=schedulers, gamma=args.msp_gamma)

    # ===== load the prompts =====
    textprompts = []
    negative_prompts = []
    filenames = []
    with open(args.prompts, "r") as f:
        data = json.load(f)
        prompt_pairs = data["images"]
    for pair in prompt_pairs:
        prompt = pair["prompt"]
        ng_prompt = pair["negative_prompt"]
        textprompts.append(prompt)
        negative_prompts.append(ng_prompt)
        filename = f"{pair['filename']}"
        filenames.append(filename)

    # ===== generate the images =====
    mapping = []
    for pi, prompt in enumerate(textprompts):
        set_seed(args.seed)

        # --- generate the images ---
        with torch.no_grad():
            negative_prompt = [negative_prompts[pi]]
            filename = filenames[pi]
            num_inference_steps = sum([_s["steps"] for _s in schedulers])
            latents = torch.randn(args.images_per_prompt,
                                  pipeline.vae.config.latent_channels,
                                  args.tsize[0][0]//8, args.tsize[0][1]//8)
            latents = latents.to(pipeline._weight_dtype).to(pipeline._device)
            
            # 优化内存：分块处理图像生成（第四个方法）
            images = pipeline(prompt=[prompt], negative_prompt=negative_prompt, 
                              num_images_per_prompt=args.images_per_prompt, 
                              guidance_scale=args.gs, num_inference_steps=num_inference_steps, 
                              latents=latents)[0]
            
            # 立即释放 latents
            del latents
            torch.cuda.empty_cache()  # 释放缓存

        # --- saving the images ---
        for bi in range(len(images)):
            folder_path = os.path.join(args.output_dir, filename)
            saved_path = os.path.join(folder_path, f'{args.tsize[1][0]}_{args.tsize[1][1]}_jpg')
            os.makedirs(saved_path, exist_ok=True)
            location = os.path.join(saved_path, 'FreCaS.jpg')
            images[bi].save(location)
            print(f'saved to {location}')
            mapping.append(f'{filename}: {prompt[bi//args.images_per_prompt]}\n')
            logger.info(f'generated image {saved_path} of prompt "{pi}"')

        # 释放 images 内存
        del images
        torch.cuda.empty_cache()  # 再次清理缓存

    # 最终清理
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    try:
        args = parse_args()
        main(args)
    except Exception as e:
        import traceback; traceback.print_exc()
        logger.error(f"runing into a exception: {e}")
        import pdb; pdb.post_mortem()