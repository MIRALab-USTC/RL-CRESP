import torch
import numpy as np
import os
os.environ['MUJOCO_GL'] = 'egl'
from pathlib import Path

from dcs_make_env import dcs
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
torch.backends.cudnn.benchmark = True


import matplotlib.pyplot as plt
import torchvision.transforms as T
import time

fig = plt.figure()


task_name = 'cartpole_swingup'
# num_videos = 1 #suite_utils.DIFFICULTY_NUM_VIDEOS['hard']
# background_kwargs = suite_utils.get_background_kwargs(
#     domain_name, num_videos, False, background_dataset_path, 'train')
# background_kwargs['start_idx'] = 0

# env = dcs.make(
#     task_name, 2, 84, 8, 0, 3, 'hard', True,
#     'background', 1, None, 'train'
# )[1]

env = dcs.make(
    task_name, 1, 84, 8, 0, 3, 'hard', True,
    'background', None, None, 'val')[0]

start_time = time.time()
time_step = env.reset()
time_step = env.reset()
for _ in range(10):
    for i in range(125):
        o = torch.tensor(time_step.observation[-3:])
        o = T.ToPILImage()(o)
        plt.imshow(o)
        plt.axis('off')
        plt.savefig(f"./dcs_make_env/demo/{i}.jpg", bbox_inches='tight', dpi=fig.dpi, pad_inches=0.0)
        time_step = env.step(env.action_space.sample())
    time_step = env.reset()
print(time.time() - start_time)
