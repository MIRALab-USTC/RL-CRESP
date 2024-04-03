# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
from utils import random_overlay

from svea import SVEAAgent
from utils import random_mask_freq_v2, random_mask_freq_v1


class SRMAgent(SVEAAgent):
    def __init__(self, obs_shape, action_shape, device, lr, feature_dim,
                 hidden_dim, critic_target_tau, num_expl_steps,
                 update_every_steps, stddev_schedule, stddev_clip, use_tb,
                 super_lr, update_super_times, super_start_step, use_super):
        super().__init__(
            obs_shape, action_shape, device, lr, feature_dim,
            hidden_dim, critic_target_tau, num_expl_steps,
            update_every_steps, stddev_schedule, stddev_clip, use_tb,
            super_lr, update_super_times, super_start_step, use_super)


    def update(self, replay_buffer, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_buffer)
        obs, action, reward, discount, next_obs = utils.to_torch(
            batch, self.device)

        # augment
        obs = self.aug(obs.float())
        original_obs = obs.clone()
        next_obs = self.aug(next_obs.float())

        # strong augmentation + SRM
        aug_obs = random_mask_freq_v2(random_overlay(original_obs))

        # update reward model
        for i in range(self.update_super_times):
            self.super_model.update_reward_model(
                obs, action, reward, discount,
                self.encoder, self.aug, replay_buffer, step)
            reward_metrics = self.super_model.update_reward_model(
                aug_obs, action, reward, discount,
                self.encoder, self.aug, replay_buffer, step)
        metrics.update(reward_metrics)

        # with torch.no_grad():
        #     self.super_model.update_reward_feature(obs, action, step)

        # encode
        obs = self.encoder(obs)
        aug_obs = self.encoder(aug_obs)
        with torch.no_grad():
            next_obs = self.encoder(next_obs)

        if self.use_tb:
            metrics['batch_reward'] = reward.mean().item()

        # update critic
        metrics.update(
            self.update_critic(obs, action, reward, discount, next_obs, step, aug_obs))

        # update actor
        metrics.update(self.update_actor(obs.detach(), step))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)

        return metrics
