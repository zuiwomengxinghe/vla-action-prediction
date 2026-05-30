import os
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# 获取当前工作目录的绝对路径
current_dir = os.getcwd()
# 或者使用 pathlib: from pathlib import Path; current_dir = str(Path.cwd())

dataset = LeRobotDataset(
    "lerobot/aloha_sim_transfer_cube_human",
    root=current_dir   # 数据集将保存在 current_dir/lerobot/aloha_sim_transfer_cube_human/
)