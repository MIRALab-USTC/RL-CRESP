import numpy as np
from distracting_control import suite_utils
from common import suite, wrappers


_DIFFICULTY = ['easy', 'medium', 'hard']
_DISTRACT_MODE = ['train', 'training', 'val', 'validation']


def make(domain_name,
         task_name,
         seed=1,
         image_size=84,
         action_repeat=2,
         frame_stack=3,
         background_dataset_path=None,
         difficulty=None,
         background_kwargs=None,
         camera_kwargs=None,
         color_kwargs=None):

    env = suite.load(domain_name,
                     task_name,
                     difficulty,
                     seed=seed,
                     background_dataset_path=background_dataset_path,
                     background_kwargs=background_kwargs,
                     camera_kwargs=camera_kwargs,
                     color_kwargs=color_kwargs,
                     visualize_reward=False)

    camera_id = 2 if domain_name == "quadruped" else 0
    env = wrappers.DMCWrapper(env,
                              seed=seed,
                              from_pixels=True,
                              height=image_size,
                              width=image_size,
                              camera_id=camera_id,
                              frame_skip=action_repeat)
    env = wrappers.FrameStack(env, k=frame_stack)
    return env


def set_dcs_multisources(domain_name, task_name, image_size, action_repeat,
                         seed, frame_stack, background_dataset_path, num_sources,
                         difficulty='hard', dynamic=True, distract_mode='train',
                         background=False, camera=False, color=False, num_videos=1,
                         test_background=False, test_camera=False, test_color=False,
                         video_start_idxs=None, camera_scale=None, test_camera_scale=0.5,
                         color_scale=None, test_color_scale=None, **kargs):
    print("background: %s, camera: %s, color: %s" % (background, camera, color))
    assert difficulty in _DIFFICULTY
    assert distract_mode in _DISTRACT_MODE
    if video_start_idxs is None:
        video_start_idxs = [num_videos * i for i in range(num_sources)]

    train_envs = []
    for i in range(num_sources):
        background_kwargs, camera_kwargs, color_kwargs = None, None, None

        if background:
            if num_videos is None:
                num_videos = suite_utils.DIFFICULTY_NUM_VIDEOS[difficulty]
            background_kwargs = suite_utils.get_background_kwargs(
                domain_name, num_videos, dynamic, background_dataset_path, distract_mode)
            background_kwargs['start_idx'] = video_start_idxs[i]

        if camera:
            if camera_scale is None:
                camera_scale = suite_utils.DIFFICULTY_SCALE[difficulty]
            camera_scale += i * camera_scale
            camera_kwargs = suite_utils.get_camera_kwargs(domain_name, camera_scale, dynamic)

        if color:
            if color_scale is None:
                color_scale = suite_utils.DIFFICULTY_SCALE[difficulty]
            color_scale += i * color_scale
            color_kwargs = suite_utils.get_color_kwargs(color_scale, dynamic)

        env = make(domain_name, task_name, seed, image_size,
                   action_repeat, frame_stack, background_dataset_path,
                   background_kwargs=background_kwargs,
                   camera_kwargs=camera_kwargs, color_kwargs=color_kwargs)
        env.seed(seed)
        train_envs.append(env)

    # stack several consecutive frames together
    obs_shape = (3 * frame_stack, image_size, image_size)
    pre_aug_obs_shape = (3, image_size, image_size)

    # Setting Test Environment
    test_seed = np.random.randint(100, 100000) + seed
    background_kwargs, camera_kwargs, color_kwargs = None, None, None

    if test_background:
        background_kwargs = suite_utils.get_background_kwargs(
            domain_name, None, dynamic, background_dataset_path, 'val')

    if test_camera:
        if test_camera_scale is None:
            test_camera_scale = suite_utils.DIFFICULTY_SCALE[difficulty]
        camera_kwargs = suite_utils.get_camera_kwargs(domain_name, test_camera_scale, dynamic)

    if test_color:
        if test_color_scale is None:
            test_color_scale = suite_utils.DIFFICULTY_SCALE[difficulty]
        color_kwargs = suite_utils.get_color_kwargs(test_color_scale, dynamic)

    test_env = make(domain_name, task_name, seed, image_size,
                    action_repeat, frame_stack, background_dataset_path,
                    background_kwargs=background_kwargs,
                    camera_kwargs=camera_kwargs, color_kwargs=color_kwargs)
    test_env.seed(test_seed)

    return train_envs, test_env, (obs_shape, pre_aug_obs_shape)
