import torch
import numpy as np
import os
os.environ['MUJOCO_GL'] = 'egl'
from pathlib import Path

from common import utils, make_env
from common.buffer_trajectory import BReplayBuffer
# from common.buffer import ReplayBuffer
from common.video import VideoRecorder

from argument import parse_args
from module.init_module import init_agent
from train import train_agent

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
torch.backends.cudnn.benchmark = True

from agent import *
from algo import *
from auxiliary import *

_AVAILABLE_AGENT = {'drq': DrQAgent, 'curl': CurlAgent}
_AVAILABLE_AUXILIARY = {'cresp': CRESP}
_AVAILABLE_ALGORITHM = {'sac': SAC, 'td3': TD3}


def run(args, device, work_dir, config):
    if args.seed == -1: 
        args.__dict__["seed"] = np.random.randint(1, 1000000)
    utils.set_seed_everywhere(args.seed)
    _, domain_name, task_name = args.env.split('.')

    # Initialize Logger and Save Hyperparameters
    logger, work_dir = utils.init_logger(args, config, work_dir)
    video_dir = utils.make_dir(work_dir / 'video')
    model_dir = utils.make_dir(work_dir / 'model')
    buffer_dir = utils.make_dir(work_dir / 'buffer')
    
    video = VideoRecorder(video_dir if args.save_video else None, height=448, width=448)

    # Initialize Environment
    train_envs, test_env, obs_dict = make_env.set_dcs_multisources(
        domain_name,
        task_name,
        config['buffer_params']['image_size'],
        config['train_params']['action_repeat'],
        test_background=args.test_background,
        test_camera=args.test_camera,
        test_color=args.test_color,
        **config['setting']
    )
    obs_shape, pre_aug_obs_shape = obs_dict
    action_shape = train_envs[0].action_space.shape
    action_limit = train_envs[0].action_space.high[0]

    # Initialize Replay Buffer
    replay_buffer = BReplayBuffer(obs_shape=pre_aug_obs_shape,
                                  action_shape=action_shape,
                                  buffer_dir=buffer_dir,
                                  batch_size=args.batch_size,
                                  device=device,
                                  **config['buffer_params'])

    config.update(dict(obs_shape=obs_shape, batch_size=args.batch_size, device=device))
    config['algo_params'].update(dict(action_shape=action_shape,
                                      action_limit=action_limit,
                                      device=device))

    # Initialize Agent
    assert args.agent in _AVAILABLE_AGENT
    config['aux_task'] = None
    if args.auxiliary is not None:
        assert args.auxiliary in _AVAILABLE_AUXILIARY
        config['aux_task'] = _AVAILABLE_AUXILIARY[args.auxiliary]
    assert args.base in _AVAILABLE_ALGORITHM
    config['base'] = _AVAILABLE_ALGORITHM[args.base]
    agent = init_agent(_AVAILABLE_AGENT[args.agent], config)

    # Train Agent
    train_agent(train_envs=train_envs,
                test_env=test_env,
                agent=agent,
                replay_buffer=replay_buffer,
                logger=logger,
                video=video,
                model_dir=model_dir,
                num_updates=args.num_updates,
                device=device,
                **config['train_params'])

    for env in train_envs:
        env.close()
    test_env.close()


if __name__ == '__main__':

    args = parse_args()
    cuda_id = "cuda:%d" % args.cuda_id
    device = torch.device(cuda_id if args.cuda else "cpu")
    for i in args.seed_list:
        work_dir = Path.cwd()
        config = utils.read_config(args, work_dir / args.config_dir)
        torch.multiprocessing.set_start_method('spawn', force=True)
        args.seed, config['setting']['seed'] = i, i
        run(args, device, work_dir, config)
