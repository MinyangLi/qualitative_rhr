import subprocess
import os

# 定义环境路径
env1_python = "/home/KeyuHu/anaconda3/envs/crfm-helm/bin/python"
env2_python = "/home/KeyuHu/anaconda3/envs/rhr/bin/python"
env2_accelerate = "/home/KeyuHu/anaconda3/envs/fouriscale/bin/accelerate"

# 第一组并行运行的指令 (env1，前四个)
group1_commands = [
    [env1_python, "AccDiffusion-main/accdiffusion_sdxl.py", "--prompt", "9_grids/prompts_selected.json", "--cuda_index", "5"],
    [env1_python, "hidiffusion/hidiff.py", "--cuda_index", "1"],
    [env1_python, "demofusion/demofusion.py", "--cuda_index", "2"],
    [env2_python, "rectifiedhr/run_sdxl.py", "--cuda_index", "4"],
    [
        env1_python, "FreCaS-master/main.py",
        "--gs", "7.5",
        "--prompts", "9_grids/prompts_selected.json",
        "--tsize", "[[1024,1024],[2048,2048]]",
        "--msp_endtimes", "200", "0",
        "--msp_steps", "40", "10",
        "--msp_gamma", "1.5",
        "--name", "sdxl",
        "--images-per-prompt", "1",
        "--facfg_weight", "25.0", "7.5",
        "--camap_weight", "0.8",
        "--output", "results/",
        "--cuda_index", "3"
    ]
]

# 第二组并行运行的指令 (env2，用 accelerate)
group2_commands = [
    [env2_accelerate, "launch", "--gpu_ids", "4", "--num_processes", "1", "ScaleCrafter-main/text2image_xl.py", "--validation_prompt", "9_grids/prompts_selected.json"],
    [env2_accelerate, "launch", "--gpu_ids", "5", "--num_processes", "1", "FouriScale-main/text2image_xl.py", "--validation_prompt", "9_grids/prompts_selected.json"]
]

# 第三组单独运行的指令 (env1，最后一个)
group3_command = [env1_python, "raw_sdxl/sdxl.py", "--cuda_index", "0"]

# 创建输出目录
output_dir = "logs"
os.makedirs(output_dir, exist_ok=True)

# 第一步：并行运行 group1
print("Starting Group 1 (parallel execution)...")
group1_processes = []
for i, cmd in enumerate(group1_commands):
    log_file = os.path.join(output_dir, f"group1_task_{i}.log")
    print(f"Starting {' '.join(cmd)} (output redirected to {log_file})...")
    with open(log_file, "w") as f:
        process = subprocess.Popen(cmd, stdout=f, stderr=f)
        group1_processes.append((process, cmd))

# 等待 group1 全部完成
for process, cmd in group1_processes:
    process.wait()
    print(f"{' '.join(cmd)} finished with return code: {process.returncode}")

# 第二步：并行运行 group2 (用 accelerate)
print("\nStarting Group 2 (parallel execution with accelerate)...")
group2_processes = []
for i, cmd in enumerate(group2_commands):
    log_file = os.path.join(output_dir, f"group2_task_{i}.log")
    print(f"Starting {' '.join(cmd)} (output redirected to {log_file})...")
    with open(log_file, "w") as f:
        process = subprocess.Popen(cmd, stdout=f, stderr=f)
        group2_processes.append((process, cmd))

# 等待 group2 全部完成
for process, cmd in group2_processes:
    process.wait()
    print(f"{' '.join(cmd)} finished with return code: {process.returncode}")

# 第三步：单独运行 group3
print("\nStarting Group 3 (sequential execution)...")
log_file = os.path.join(output_dir, "group3_task_0.log")
print(f"Running {' '.join(group3_command)} (output redirected to {log_file})...")
with open(log_file, "w") as f:
    process = subprocess.run(group3_command, stdout=f, stderr=f)
print(f"{' '.join(group3_command)} finished with return code: {process.returncode}")