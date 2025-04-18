import subprocess
import os

# 定义环境路径
env1_python = "/home/KeyuHu/anaconda3/envs/crfm-helm/bin/python"
env2_python = "/home/KeyuHu/anaconda3/envs/rhr/bin/python"
env2_accelerate = "/home/KeyuHu/anaconda3/envs/fouriscale/bin/accelerate"

# 所有指令按顺序运行，绑定到GPU 4
all_commands = [
    [env1_python, "AccDiffusion-main/accdiffusion_sdxl.py", "--prompt", "9_grids/prompts_selected4.json", "--cuda_index", "4"],
    [env1_python, "hidiffusion/hidiff.py", "--cuda_index", "4"],
    [env1_python, "demofusion/demofusion.py", "--cuda_index", "4"],
    [env2_python, "rectifiedhr/run_sdxl.py", "--cuda_index", "4"],
    [
        env1_python, "FreCaS-master/main.py",
        "--gs", "7.5",
        "--prompts", "9_grids/prompts_selected4.json",
        "--tsize", "[[1024,1024],[4096,4096]]",
        "--msp_endtimes", "200", "0",
        "--msp_steps", "40", "10",
        "--msp_gamma", "1.5",
        "--name", "sdxl",
        "--images-per-prompt", "1",
        "--facfg_weight", "25.0", "7.5",
        "--camap_weight", "0.8",
        "--output", "results/",
        "--cuda_index", "4",
        "--vae_tiling"
    ],
    [env2_accelerate, "launch", "--gpu_ids", "4", "--num_processes", "1", "ScaleCrafter-main/text2image_xl.py", "--validation_prompt", "9_grids/prompts_selected4.json"],
    [env2_accelerate, "launch", "--gpu_ids", "4", "--num_processes", "1", "FouriScale-main/text2image_xl.py", "--validation_prompt", "9_grids/prompts_selected4.json"],
    [env1_python, "raw_sdxl/sdxl.py", "--cuda_index", "4"]
]

# 创建输出目录
output_dir = "logs"
os.makedirs(output_dir, exist_ok=True)

# 顺序运行所有命令
print("Starting sequential execution on GPU 4...")
for i, cmd in enumerate(all_commands):
    log_file = os.path.join(output_dir, f"task_{i}.log")
    print(f"Running {' '.join(cmd)} (output redirected to {log_file})...")
    with open(log_file, "w") as f:
        process = subprocess.run(cmd, stdout=f, stderr=f)
    print(f"{' '.join(cmd)} finished with return code: {process.returncode}")