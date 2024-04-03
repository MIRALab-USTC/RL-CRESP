# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.import hydra
import abc
import time
import torch
import itertools
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Normal, independent, kl
from pathlib import Path, PosixPath
from typing import Tuple, Union, Optional

import utils
from drqv2 import Encoder


def make_super_model(obs_shape: Tuple[int], action_shape: Tuple[int], repr_dim: int,
                     device: str, lr: float, feature_dim: int, rew_feature_dim: int,
                     hidden_dim: int, num_expl_steps: int, beta_start_value: float,
                     beta_end_value: float, use_super: bool, start_step: int = 5000,
                     positive: bool = False):
    if not use_super:
        return EmptySuperModel(device=device,
                               num_expl_steps=num_expl_steps,
                               positive=positive)
    return SuperModel(obs_shape, action_shape, repr_dim, device, lr, feature_dim,
                      rew_feature_dim, hidden_dim, num_expl_steps, beta_start_value,
                      beta_end_value, start_step, positive=positive)


class QfCritic(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim, rew_feature_dim=512):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.Qf1 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, rew_feature_dim))

        self.Qf2 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, rew_feature_dim))

        self.Q1 = nn.Sequential(
            nn.Linear(rew_feature_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1))

        self.Q2 = nn.Sequential(
            nn.Linear(rew_feature_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1))

        self.apply(utils.weight_init)

    def forward(self, obs, action):
        h = self.trunk(obs)
        h_action = torch.cat([h, action], dim=-1)
        qf1 = self.Qf1(h_action)
        qf2 = self.Qf2(h_action)

        q1 = self.Q1(qf1)
        q2 = self.Q2(qf2)
        return q1, q2, qf1, qf2


class EnsembleMLP(nn.Module):
    def __init__(self, repr_dim: int, action_shape: Tuple[int], feature_dim: int,
                 rew_feature_dim: int, hidden_dim: int, output_dim: int = 1, positive: bool = False):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.ReLU())
        self.fc = nn.Sequential(
            nn.Linear(rew_feature_dim, rew_feature_dim, bias=False))

        self.sa_feature_encoder = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.LayerNorm(hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, rew_feature_dim * 2))#, nn.ReLU())
        self.rf_feature_encoder = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, rew_feature_dim * 2))#, nn.ReLU())

        self.sa_recons_weight = nn.Parameter(torch.rand(rew_feature_dim, output_dim))
        self.sa_recons_logvar = nn.Sequential(nn.Linear(rew_feature_dim, output_dim))
        self.rf_recons_weight = nn.Parameter(torch.rand(rew_feature_dim, output_dim))
        self.rf_recons_logvar = nn.Sequential(nn.Linear(rew_feature_dim, output_dim))

        self.logvar_max = nn.Parameter(
            torch.ones(1, output_dim).float() / 2, requires_grad=True)
        self.logvar_min = nn.Parameter(
            -torch.ones((1, output_dim)).float() * 10, requires_grad=True)
        self.positive = positive

        self.apply(utils.weight_init)

    @property
    def weight(self):
        if self.positive:
            return torch.sigmoid(self.sa_recons_weight)
        return self.sa_recons_weight

    def forward(self, obs: Tensor, action: Tensor, reward: Tensor) -> Tuple[Tensor]:
        state_action = torch.cat([self.trunk(obs), action], dim=-1)

        # (batch_size, rew_feature_dim*2) --> h_inputs: (2, batch_size, rew_feature_dim*2)
        sa_feature = self.sa_feature_encoder(state_action)
        rf_feature = self.rf_feature_encoder(reward)
        features = torch.stack([sa_feature, rf_feature], dim=0)

        # (2, batch_size, rew_feature_dim)
        reward_feature, reward_feature_logvar = features.chunk(2, dim=-1)
        reward_feature_logvar = utils.soft_clamp(reward_feature_logvar, -10, 0.5)
        # reward_feature_std = F.softplus(reward_feature_std) + 0.1
        return reward_feature, torch.exp(reward_feature_logvar / 2)

    def sa_forward(self, obs: Tensor, action: Tensor) -> Tuple[Tensor]:
        state_action = torch.cat([self.trunk(obs), action], dim=-1)
        sa, sa_logvar = self.sa_feature_encoder(state_action).chunk(2, dim=-1)
        sa_logvar = utils.soft_clamp(sa_logvar, -10, 0.5)
        return sa, torch.exp(sa_logvar / 2)

    def rf_forward(self, reward: Tensor) -> Tuple[Tensor]:
        rf, rf_logvar = self.rf_feature_encoder(reward).chunk(2, dim=-1)
        rf_logvar = utils.soft_clamp(rf_logvar, -10, 0.5)
        return rf, torch.exp(rf_logvar / 2)

    def reconst(self, reward_feature: Tensor):
        # reward_feature: (2, batch_size, rew_feature_dim)
        #  --> mu / log_var: (2, batch_size, 1)
        sa_weight = torch.sigmoid(self.sa_recons_weight) if self.positive \
                else self.sa_recons_weight
        rf_weight = torch.sigmoid(self.rf_recons_weight) if self.positive \
                else self.rf_recons_weight

        mu_sa = torch.matmul(reward_feature[0], sa_weight)
        mu_rf = torch.matmul(reward_feature[1], rf_weight)
        mu = torch.stack([mu_sa, mu_rf], dim=0)

        logvar_sa = self.sa_recons_logvar(reward_feature[0])
        logvar_rf = self.rf_recons_logvar(reward_feature[1])
        log_var = torch.stack([logvar_sa, logvar_rf], dim=0)
        log_var = utils.soft_clamp(log_var, -10, 0.5)
        return mu, log_var


class EmptySuperModel(object, metaclass=abc.ABCMeta):
    def __init__(self, device: str, num_expl_steps: int, deterministic: bool = True,
                 start_step: int = 2000, positive: bool = False):
        # Setting hyperparameters
        self.device = device

        # Default training parameters
        self.pretrain_done = False
        self.training = False
        self.total_time_steps = 0
        self.num_expl_steps = num_expl_steps
        self.start_step = start_step
        self.deterministic = deterministic
        self.positive = positive

        # Default models
        self.encoder: Encoder = None
        self.reward_model: EnsembleMLP = None
        self.encoder_opt: torch.optim.Adam = None
        self.reward_model_opt: torch.optim.Adam = None

        # Reward feature signal
        self.current_step = 0
        self.critic_f_loss: list = []
        self.reward_feature: Tensor = None
        self.q_feature_target: list = None
        self.metrics = dict(critic_target_qf=0.0, critic_qf1=0.0, critic_qf2=0.0)

    def train(self, training: bool = True):
        self.training = training
        if self.encoder is not None:
            self.encoder.train(training)
        if self.reward_model is not None:
            self.reward_model.train(training)

    def eval(self):
        if self.encoder is not None:
            self.encoder.eval()
        if self.reward_model is not None:
            self.reward_model.eval()

    @torch.no_grad()
    def update_reward_feature(self, reward_feature: Tensor, step: int):
        self.reward_feature = reward_feature
        self.current_step = step

    @torch.no_grad()
    def update_q_feature_target(self, q_feature: Tensor):
        if self.q_feature_target is None:
            self.q_feature_target = [q_feature]
        else:
            self.q_feature_target.append(q_feature)
        self.metrics['critic_target_qf'] = torch.stack(self.q_feature_target).mean().item()

    def update_reward_model(
            self, obs, action, reward, discount, encoder, aug, replay_buffer, step):
        return dict()

    def compute_Q(self, Q: Tuple[Tensor], output_qf: bool = False, compute_qf_loss: bool = True) \
            -> Tuple[Tensor]:
        if len(Q) == 4:
            Q1, Q2, Qf1, Qf2 = Q
        elif len(Q) == 2:
            Q1, Q2 = Q
            Qf1, Qf2 = None, None
        else:
            raise ValueError(Q.shape)
        if self.current_step >= self.start_step and self.current_step >= self.initial_training_step \
                and len(Q) == 4 and compute_qf_loss and self.q_feature_target is not None:
            assert torch.is_tensor(Qf1) and torch.is_tensor(Qf2), print(Qf1, Qf2)
            target_Qf = torch.stack(self.q_feature_target).mean(0).detach()
            critic_f_loss = (Qf1 - target_Qf).pow(2).mean() + (
                Qf2 - target_Qf).pow(2).mean()
            self.critic_f_loss.append(critic_f_loss)
            self.metrics['critic_qf1'] = Qf1.mean().item()
            self.metrics['critic_qf2'] = Qf2.mean().item()
        if output_qf:
            return Q1, Q2, Qf1, Qf2
        return Q1, Q2

    @torch.no_grad()
    def compute_targetQ(self, target_Q: Tuple[Tensor], discount: Tensor, step: int,
                        entropy: Tensor = None) -> Tuple[Tensor]:
        if self.current_step >= self.start_step and self.current_step >= self.initial_training_step \
                and len(target_Q) == 4:
            target_Qf1, target_Qf2 = target_Q[2:]
            if not self.positive:
                idx = (self.reward_feature > 0).float()
                target_Vf = torch.min(target_Qf1, target_Qf2) * idx + torch.max(target_Qf1, target_Qf2) * (1.0 - idx)
            else:
                target_Vf = torch.min(target_Qf1, target_Qf2)
            if self.current_step == step:
                if entropy is not None:
                    if entropy.ndim == 1:
                        entropy = entropy.unsqueeze(-1)
                    weight = 1.0 / (self.reward_model.weight.mean().item() + 1.e-4)
                    target_Vf += entropy * weight
                target_Qf = self.reward_feature + (discount * target_Vf)
                self.update_q_feature_target(target_Qf)
        return target_Q[:2]

    def compute_qf_loss(self, idx: int = 0):
        if len(self.critic_f_loss) == 0:
            return torch.zeros(1).to(self.device).mean().detach()
        assert idx < len(self.critic_f_loss), print(idx, len(self.critic_f_loss))
        return self.critic_f_loss[idx]

    def clear(self):
        self.critic_f_loss = []
        self.reward_feature = None
        self.q_feature_target = None
        self.metrics = dict(critic_target_qf=0.0, critic_qf1=0.0, critic_qf2=0.0)

    def save_snapshot(self, model_dir: str, num_step: str) -> None:
        if self.encoder is not None:
            torch.save(self.encoder.state_dict(),
                    '%s/reward_encoder%s.pt' % (model_dir, num_step))
        if self.reward_model is not None:
            torch.save(self.reward_model.state_dict(),
                    '%s/reward_model%s.pt' % (model_dir, num_step))

    def load_snapshot(self, model_dir: str, num_step: str) -> None:
        if self.encoder is not None:
            self.encoder.load_state_dict(
                torch.load('%s/reward_encoder%s.pt' % (model_dir, num_step),
                           map_location=self.device))
        if self.reward_model is not None:
            self.reward_model.load_state_dict(
                torch.load('%s/reward_model%s.pt' % (model_dir, num_step),
                           map_location=self.device))
        print("Loaded Predictor%s." % num_step)


class SuperModel(EmptySuperModel):
    def __init__(self, obs_shape: Tuple, action_shape: Tuple, repr_dim: int, device: str,
                 lr: float, feature_dim: int, rew_feature_dim: int, hidden_dim: int,
                 num_expl_steps: int, beta_start_value: float, beta_end_value: float,
                 deterministic: bool = True, initial_training_step: int = 5000, positive: bool = False):
        super().__init__(device, num_expl_steps, deterministic, initial_training_step, positive)
        self.initial_training_step = initial_training_step

        # Default training parameters
        self.beta_scheduler = utils.ExponentialScheduler(
            start_value=beta_start_value, end_value=beta_end_value,
            n_iterations=initial_training_step+5e4, start_iteration=initial_training_step+10000)

        # Initial models
        # self.encoder = Encoder(obs_shape).to(device)#self.encoder.repr_dim
        self.reward_model = EnsembleMLP(repr_dim, action_shape, feature_dim,
                                        rew_feature_dim, hidden_dim, 1, positive).to(device)

        # Initial optimizers
        # self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr*0.5)
        self.reward_model_opt = torch.optim.Adam(self.reward_model.parameters(), lr=lr)

        self.train()

    def update_model_via_reconst(self, reward_feature: Tensor, reward: Tensor):
        # reward_feature: (2, batch_size, rew_feature_dim)
        metrics = dict()

        # Compute recon loss to update reward model, such that reward features are
        # linearly dependent on the mean of rewards.
        mu, log_var = self.reward_model.reconst(reward_feature)
        if mu.ndim == 3:
            reward = reward.view(1, -1, 1)

        inv_var = torch.exp(-log_var)
        recon_mse_loss = ((mu - reward).pow(2) * inv_var)
        recon_loss = recon_mse_loss.sum() + log_var.sum()

        # recon_loss += 0.01 * self.reward_model.logvar_max.sum() \
        #     - 0.01 * self.reward_model.logvar_min.sum()

        metrics['super_recon_loss'] = recon_loss.item()
        metrics['super_recon_mse_loss'] = recon_mse_loss.mean().item()

        return recon_loss, metrics

    def update_model_via_inforbn(self, reward_feature: Tensor, reward_feature_std: Tensor,
                                 step: int):
        # reward_feature: (2, batch_size, rew_feature_dim)
        metrics = dict()

        ## 1. Compute the mutual information loss
        rew_relevant_loss = utils.compute_curl_loss(reward_feature[0],
                                                    self.reward_model.fc(reward_feature[1]))

        ## 2. Compute the symmetrized KL divergence
        rew_irrelevant_loss = utils.kl_divergence(reward_feature[0], reward_feature_std[0],
                                                  reward_feature[1], reward_feature_std[1]) \
                                    + utils.kl_divergence(reward_feature[1], reward_feature_std[1],
                                                          reward_feature[0], reward_feature_std[0])
        beta = self.beta_scheduler(step)

        # 3. Compute total inforbn loss
        infor_loss = rew_relevant_loss + rew_irrelevant_loss * beta * 2.0

        metrics['super_rew_relevant_loss'] = rew_relevant_loss.item()
        metrics['super_rew_irrelevant_loss'] = rew_irrelevant_loss.item()
        metrics['super_rew_irrelevant_beta'] = beta

        return infor_loss, metrics

    def update_reward_model(self, obs, action, reward, discount, encoder, aug,
                            replay_buffer, step) -> Tuple[Tensor, dict]:
        metrics = dict()
        self.total_time_steps += 1
        assert encoder is not None or self.encoder is not None, print(encoder, self.encoder)
        encoder_opt = None
        if self.encoder is not None:
            encoder, encoder_opt = self.encoder, self.encoder_opt

        if step < self.initial_training_step:
            metrics['super_recon_loss'] = 0.0
            metrics['super_recon_mse_loss'] = 0.0
            metrics['super_rew_relevant_loss'] = 0.0
            metrics['super_rew_irrelevant_loss'] = 0.0
            metrics['super_rew_irrelevant_beta'] = 0.0
            return metrics

        elif step == self.initial_training_step:
            self.pertrain_model(replay_buffer, aug, encoder, encoder_opt, step)

        # Compute reward features
        #     reward_feature: (2, batch_size, rew_feature_dim)
        reward_feature, reward_feature_std = self.reward_model(encoder(obs), action, reward)

        recon_loss, recon_metrics = self.update_model_via_reconst(
            reward_feature, reward)
        infor_loss, infor_metrics = self.update_model_via_inforbn(
            reward_feature, reward_feature_std, step)
        # infor_loss, infor_metrics = 0.0, dict()
        # infor_metrics['super_rew_relevant_loss'] = 0.0
        # infor_metrics['super_rew_irrelevant_loss'] = 0.0
        # infor_metrics['super_rew_irrelevant_beta'] = 0.0

        # Optimize reward mode
        if encoder_opt is not None:
            encoder_opt.zero_grad(set_to_none=True)
        self.reward_model_opt.zero_grad(set_to_none=True)
        (recon_loss + infor_loss).backward()
        if encoder_opt is not None:
            encoder_opt.step()
        self.reward_model_opt.step()

        # Record infor
        metrics.update(recon_metrics)
        metrics.update(infor_metrics)
        # self.reward_model.log(metrics)

        # reward_feature, _ = self.reward_model(obs, action, reward)
        # self.update_reward_feature(self.reward_model.rf_forward(reward)[0], step)
        self.update_reward_feature(self.reward_model.sa_forward(encoder(obs), action)[0], step)
        return metrics


    def pertrain_model(self, replay_buffer, aug, encoder, encoder_opt, step):
        max_epoch_step = 500
        total_recon_loss, total_mse_loss = 0, 0
        total_relevant_loss, total_rew_irrelevant_loss = 0, 0
        if self.initial_training_step <= self.num_expl_steps or self.pretrain_done:
            return

        for pretrain_step in range(self.initial_training_step - self.num_expl_steps):
            metrics = dict()
            batch = next(replay_buffer)
            obs, action, reward, *_ = utils.to_torch(batch, self.device)
            obs = aug(obs.float())
            # metrics = self.update_reward_model(obs, action, reward, step + pretrain_step)

            # Compute reward features
            #     reward_feature: (2, batch_size, rew_feature_dim)
            reward_feature, reward_feature_std = self.reward_model(encoder(obs), action, reward)

            recon_loss, recon_metrics = self.update_model_via_reconst(
                reward_feature, reward)

            infor_loss, infor_metrics = self.update_model_via_inforbn(
                reward_feature, reward_feature_std, step)

            # Optimize reward mode
            if encoder_opt is not None:
                encoder_opt.zero_grad(set_to_none=True)
            self.reward_model_opt.zero_grad(set_to_none=True)
            (recon_loss + infor_loss).backward()
            if encoder_opt is not None:
                encoder_opt.step()
            self.reward_model_opt.step()

            metrics.update(recon_metrics)
            metrics.update(infor_metrics)

            total_recon_loss += metrics['super_recon_loss']
            total_mse_loss += metrics['super_recon_mse_loss']
            total_relevant_loss += metrics['super_rew_relevant_loss']
            total_rew_irrelevant_loss += metrics['super_rew_irrelevant_loss']

            if (pretrain_step + 1) % max_epoch_step == 0:
                print("| Train | Step: %d | Recon Loss: %.6f | MSE Loss: %.6f | TR_Loss: %.6f | TIR_Loss: %.6f " % (
                    pretrain_step+1, total_recon_loss / 500.0, total_mse_loss / 500.0,
                    total_relevant_loss / 500.0, total_rew_irrelevant_loss / 500.0))
                total_recon_loss, total_mse_loss = 0, 0
                total_relevant_loss, total_rew_irrelevant_loss = 0, 0

        print("| PreTrain Done.")
        self.pretrain_done = True
