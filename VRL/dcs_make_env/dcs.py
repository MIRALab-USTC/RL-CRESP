import numpy as np
from distracting_control import suite_utils
from .suite import load as suite_load
from .wrappers import DMCWrapper, ExtendedTimeStepFrameStackWrapper


_DIFFICULTY = ['easy', 'medium', 'hard']
_DISTRACT_MODE = ['train', 'training', 'val', 'validation']


def load(domain_name, task_name, seed=1, image_size=84, action_repeat=2,
         frame_stack=3, background_dataset_path=None, background_dataset_videos="train",
         difficulty=None, background_kwargs=None, camera_kwargs=None, color_kwargs=None):

    env = suite_load(domain_name, task_name, difficulty, seed=seed,
                     background_dataset_path=background_dataset_path,
                     background_dataset_videos=background_dataset_videos,
                     background_kwargs=background_kwargs,
                     camera_kwargs=camera_kwargs,
                     color_kwargs=color_kwargs,
                     visualize_reward=False)

    camera_id = 2 if domain_name == "quadruped" else 0
    env = DMCWrapper(env, seed=seed, from_pixels=True, height=image_size,
                     width=image_size, camera_id=camera_id, frame_skip=action_repeat)
    env = ExtendedTimeStepFrameStackWrapper(env, k=frame_stack)
    return env


def make(name, num_sources, image_size, action_repeat, seed, frame_stack, difficulty='hard',
         dynamic=True, dcs_type=None, num_videos=None, scale=None, background_dataset_videos='train',
         video_start_idxs=None, background_dataset_path="/home/amax/data/DAVIS/JPEGImages/480p/"):

    # handle the task name
    domain_name, task_name = name.split('_', 1)
    domain_name = dict(ball='ball_in_cup').get(domain_name, domain_name)
    task_name = dict(in_cup_catch='catch').get(task_name, task_name)

    assert difficulty in _DIFFICULTY, print(difficulty)
    assert background_dataset_videos in _DISTRACT_MODE, print(background_dataset_videos)

    # handle the scale of distractions
    if isinstance(scale, list) or isinstance(scale, tuple):
        scales = np.linspace(scale[0], scale[1], num_sources)
    elif scale is None:
        scales = [suite_utils.DIFFICULTY_SCALE[difficulty]] * num_sources
    else:
        scales = [scale] * num_sources

    # obtain the type of distractions
    # assert dcs_type in ['background', 'color', 'camera'], print(dcs_type)
    dyn = ' Dynamic' if dynamic else ''
    if dcs_type == 'background':
        background, camera, color = True, False, False
        print(f"Distraction Type:{dyn} Background.")
    elif dcs_type == 'camera':
        background, camera, color = False, True, False
        print(f"Distraction Type:{dyn} Camera.")
        print("Camera Scale:", scales)
    elif dcs_type == 'color':
        background, camera, color = False, False, True
        print(f"Distraction Type:{dyn} Color.")
        print("Color Scale:", scales)
    else:
        raise ValueError(dcs_type)

    if video_start_idxs is None and num_videos is not None:
        video_start_idxs = [num_videos * i for i in range(num_sources)]

    envs = []
    for i in range(num_sources):
        background_kwargs, camera_kwargs, color_kwargs = None, None, None

        if background:
            background_kwargs = suite_utils.get_background_kwargs(
                domain_name, num_videos, dynamic, background_dataset_path, background_dataset_videos)
            if video_start_idxs is not None:
                background_kwargs['start_idx'] = video_start_idxs[i]
            background_kwargs['seed'] = seed + i

        if camera:
            camera_kwargs = suite_utils.get_camera_kwargs(domain_name, scales[i], dynamic)
            camera_kwargs['seed'] = seed + i

        if color:
            color_kwargs = suite_utils.get_color_kwargs(scales[i], dynamic)
            color_kwargs['seed'] = seed + i

        env = load(domain_name, task_name, seed+i, image_size,
                   action_repeat, frame_stack, background_dataset_path,
                   background_dataset_videos=background_dataset_videos,
                   background_kwargs=background_kwargs,
                   camera_kwargs=camera_kwargs, color_kwargs=color_kwargs)
        env.seed(seed+i)
        envs.append(env)

    return envs
