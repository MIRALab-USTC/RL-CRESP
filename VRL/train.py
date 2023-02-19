# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)

import os

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

from pathlib import Path
import hydra
import yaml
import numpy as np
import torch
from dm_env import specs

import csv
import dmc
from dcs_make_env import dcs
import utils
from logger import Logger
from logx import EpochLogger
from evaluate import d4rl_eval_fn
from numpy_replay_buffer import EfficientReplayBuffer
from video import TrainVideoRecorder, VideoRecorder

torch.backends.cudnn.benchmark = True

_ACTION_REPEAT = {
    'ball_in_cup_catch': 4, 'cartpole_swingup': 8, 'cheetah_run': 4,
    'finger_spin': 2, 'reacher_easy': 4, 'walker_walk': 2
}


def make_agent(obs_spec, action_spec, cfg):
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = action_spec.shape
    return hydra.utils.instantiate(cfg)


class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        cuda_id = "cuda:%d" % cfg.cuda_id
        self.device = torch.device(cuda_id if cfg.device == "cuda" else "cpu")
        print("-----DEVICE: ", self.device, "------")
        self.cfg.action_repeat = _ACTION_REPEAT.get(self.cfg.task_name, self.cfg.action_repeat)
        self.setup()

        self.agent = make_agent(self.eval_env.observation_spec(),
                                self.eval_env.action_spec(),
                                self.cfg.agent)
        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0

    def setup(self):
        # create logger
        self.logger = Logger(self.work_dir, use_tb=self.cfg.use_tb)
        self.splogger = EpochLogger(**dict(output_dir=self.work_dir, exp_name=self.cfg.exp_name))
        with open(self.work_dir / '.hydra' / 'config.yaml') as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)
        self.splogger.save_config(config)
        utils.init_sp_logger(self.splogger)

        # create envs
        self.init_dcs_env()
        [print("DisTrainEnv %d: " % i, self.train_env[i]) for i in range(len(self.train_env))]

        # create replay buffer
        data_specs = (self.eval_env.observation_spec(),
                      self.eval_env.action_spec(),
                      specs.Array((1,), np.float32, 'reward'),
                      specs.Array((1,), np.float32, 'discount'))

        if self.cfg.rew_noise is not None:
            self.cfg.rew_noise = {'mu': self.cfg.rew_noise_mu, 'std': self.cfg.rew_noise_std}

        self.replay_buffer = EfficientReplayBuffer(self.cfg.replay_buffer_size,
                                                   self.cfg.batch_size,
                                                   self.cfg.nstep,
                                                   self.cfg.discount,
                                                   self.cfg.frame_stack,
                                                   self.cfg.agent.rsd_nstep if 'cres' in self.cfg.experiment else 1,
                                                   self.cfg.replay_buffer_num_workers,
                                                   data_specs,
                                                   rew_noise=self.cfg.rew_noise)

        self.video_recorder = VideoRecorder(
            self.work_dir if self.cfg.save_video else None)
        self.train_video_recorder = TrainVideoRecorder(
            self.work_dir if self.cfg.save_train_video else None)

        # self.eval_on_distracting = self.cfg.eval_on_distracting
        # self.eval_on_multitask = self.cfg.eval_on_multitask

        self.eval_single_env = d4rl_eval_fn(self.video_recorder,
                                            self.cfg.action_repeat)

    def init_dcs_env(self):
        self.train_env = dcs.make(self.cfg.task_name, self.cfg.num_sources, 84, self.cfg.action_repeat,
                                  self.cfg.seed, self.cfg.frame_stack, self.cfg.distracting_mode, True,
                                  self.cfg.dcs_type, self.cfg.num_videos,
                                  [self.cfg.scale, self.cfg.scale_end], 'train',
                                  background_dataset_path=self.cfg.data_path)
        self.eval_env  = dmc.make(self.cfg.task_name, self.cfg.frame_stack,
                                  self.cfg.action_repeat, self.cfg.seed)

        if self.cfg.eval_on_dcs:
            self.distracting_envs, self.distraction_modes = [], []
            self.distracting_envs.append(
                dcs.make(self.cfg.task_name, 1, 84, self.cfg.action_repeat, self.cfg.seed,
                         self.cfg.frame_stack, self.cfg.distracting_mode, True,
                         self.cfg.dcs_type, None, self.cfg.eval_scale, 'val',
                         background_dataset_path=self.cfg.data_path)[0])
            self.distraction_modes.append(self.cfg.dcs_type)

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.action_repeat

    @property
    def env_id(self):
        return self._global_episode % self.cfg.num_sources

    @property
    def env(self):
        if isinstance(self.train_env, list):
            return self.train_env[self.env_id]
        return self.train_env


    def eval(self, env, env_name='', record_video=True, write_log=False):
        env_name = '%s_' % env_name if env_name != '' else env_name
        eval_info = self.eval_single_env(env, env_name, self.agent, self.global_episode,
                                         self.global_step, self.global_frame, record_video,
                                         self.cfg.num_eval_episodes)
        self.splogger.store(**eval_info)
        if not write_log:
            for k, v in eval_info.items():
                self.logger.log(f'eval/{k}', v, self.global_frame)
        else:
            with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
                for k, v in eval_info.items():
                    log(k, v)

    def eval_dcs(self, record_video):
        for env, env_name in zip(self.distracting_envs, self.distraction_modes):
            self.eval(env, env_name, record_video)

    def train(self):
        if self.load_agent():
            self.train_online()

        write_dir = "./main_results/"
        self.save_log_for_the_end(
            f"{write_dir}{self.cfg.task_name}",
            self.distracting_envs[0], max_episode=10)
        self.save_log_for_the_end(
            f"{write_dir}{self.cfg.task_name}",
            self.distracting_envs[0], max_episode=100)


    def save_log_for_the_end(self, write_dir, env, max_episode=100):
        eval_info = self.eval_single_env(env, '', self.agent, self.global_episode,
                                         self.global_step, self.global_frame, False,
                                         max_episode)
        write_dir = f"{write_dir}-{max_episode}.csv"
        if self.cfg.dcs_type == 'background':
            start_scale, end_scale, target_scale = 1, self.cfg.num_sources, 30
        else:
            start_scale, end_scale, target_scale = self.cfg.scale, self.cfg.scale_end, self.cfg.eval_scale

        with open(write_dir, 'a') as f:
            csv_writer = csv.DictWriter(f, fieldnames=[
                'return', 'seed', 'type', 'num_domains', 'start_scale', 'end_scale', 'target_scale',
                'length', 'discount', 'algo', 'batch_size'
            ])
            data={}
            data['return'] = eval_info['episode_reward']
            data['seed'] = self.cfg.seed
            data['type'] = self.cfg.dcs_type
            data['num_domains'] = self.cfg.num_sources
            data['start_scale'] = start_scale
            data['end_scale'] = end_scale
            data['target_scale'] = target_scale
            try:
                data['length'] = self.cfg.agent.rsd_nstep
            except:
                data['length'] = 0
            data['discount'] = self.cfg.rsd_discount if self.cfg.rsd_discount else self.cfg.discount
            data['algo'] = self.cfg.experiment
            data['batch_size'] = self.cfg.batch_size
            csv_writer.writerow(data)
        print(max_episode, "return: ", eval_info['episode_reward'])

    def train_online(self):
        # predicates
        train_until_step = utils.Until(self.cfg.num_train_frames,
                                       self.cfg.action_repeat)
        seed_until_step = utils.Until(self.cfg.num_seed_frames,
                                      self.cfg.action_repeat)
        eval_every_step = utils.Every(self.cfg.eval_every_frames,
                                      self.cfg.action_repeat)
        # only in distracting evaluation mode
        eval_save_vid_every_step = utils.Every(self.cfg.eval_save_vid_every_step,
                                               self.cfg.action_repeat)

        print(self.replay_buffer.nstep, self.replay_buffer.rsd_nstep)
        print(self.agent.update_times, self.agent.actor_update_times, self.agent.critic_update_times, self.agent.critic_target_update_freq)
        episode_step, episode_reward = 0, 0
        time_step = self.env.reset()
        self.replay_buffer.add(time_step)
        self.train_video_recorder.init(time_step.observation)
        metrics = None
        while train_until_step(self.global_step):
            if time_step.last():
                self._global_episode += 1
                self.train_video_recorder.save(f'{self.global_frame}.mp4')
                # wait until all the metrics schema is populated
                if metrics is not None:
                    # log stats
                    elapsed_time, total_time = self.timer.reset()
                    episode_frame = episode_step * self.cfg.action_repeat
                    with self.logger.log_and_dump_ctx(self.global_frame,
                                                      ty='train') as log:
                        log('fps', episode_frame / elapsed_time)
                        log('total_time', total_time)
                        log('episode_reward', episode_reward)
                        log('episode_length', episode_frame)
                        log('episode', self.global_episode)
                        log('buffer_size', len(self.replay_buffer))
                        log('step', self.global_step)
                    train_info = dict(EpRet=episode_reward, EpLen=episode_frame,
                                      EpNum=self.global_episode)
                    self.splogger.store(**train_info)

                # reset env
                time_step = self.env.reset()
                self.replay_buffer.add(time_step)
                self.train_video_recorder.init(time_step.observation)
                # try to save snapshot
                if self.cfg.save_snapshot:
                    self.save_snapshot()
                episode_step = 0
                episode_reward = 0

            # try to evaluate
            if eval_every_step(self.global_step):
                self.logger.log('eval_total_time', self.timer.total_time(),
                                self.global_frame)
                if self.cfg.eval_on_dcs:
                    self.eval_dcs(eval_save_vid_every_step(self.global_step))
                self.eval(self.eval_env, write_log=True)
                utils.print_log(self.splogger, self.global_step, self.global_frame,
                                self.distraction_modes)
                self.agent.print_update_record()

            # sample action
            with torch.no_grad(), utils.eval_mode(self.agent):
                action = self.agent.act(time_step.observation,
                                        self.global_step,
                                        eval_mode=False)

            # try to update the agent
            if not seed_until_step(self.global_step):
                metrics = self.agent.update(self.replay_buffer, self.global_step)
                self.logger.log_metrics(metrics, self.global_frame, ty='train')
                self.splogger.store(**metrics)

            # take env step
            time_step = self.env.step(action)
            episode_reward += time_step.reward
            self.replay_buffer.add(time_step)
            self.train_video_recorder.record(time_step.observation)
            episode_step += 1
            self._global_step += 1

    def load_agent(self):
        if not self.cfg.load_model:
            return True
        print("| Load Agent.")
        print("| Save Dir: ", self.cfg.save_dir)
        self.load_snapshot(self.cfg.save_dir)
        return False

    def save_snapshot(self):
        snapshot = self.work_dir / 'snapshot.pt'
        keys_to_save = ['agent', 'timer', '_global_step', '_global_episode']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open('wb') as f:
            torch.save(payload, f)

    def load_snapshot(self, work_dir):
        snapshot = Path(work_dir) / 'snapshot.pt'
        with snapshot.open('rb') as f:
            payload = torch.load(f)
        for k, v in payload.items():
            self.__dict__[k] = v


@hydra.main(config_path='cfgs', config_name='config')
def main(cfg):
    from train import Workspace as W
    root_dir = Path.cwd()
    workspace = W(cfg)
    # print(cfg)
    snapshot = root_dir / 'snapshot.pt'
    if snapshot.exists():
        print(f'resuming: {snapshot}')
        workspace.load_snapshot()
    workspace.train()
    print(" __________")
    print("|_TASK_END_|")


if __name__ == '__main__':
    main()
