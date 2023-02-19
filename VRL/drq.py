# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

import utils


class RandomShiftsAug(nn.Module):
    def __init__(self, pad=4):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False)


class NoShiftAug(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class Encoder(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()

        assert len(obs_shape) == 3
        self.repr_dim = 32 * 35 * 35
        self.num_layers = 4

        self.convnet = nn.Sequential(nn.Conv2d(obs_shape[0], 32, 3, stride=2),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Flatten())
        self.infos = dict()
        self.apply(utils.weight_init)

    def forward_conv(self, conv):
        for i in range(self.num_layers):
            conv = self.convnet[2*i+1](self.convnet[2*i](conv))
            self.infos['conv%s_img' % (i + 1)] = conv.detach().clone()
        return self.convnet[-1](conv)

    def forward(self, obs):
        obs = obs / 255.0
        h = self.forward_conv(obs)
        return h

    def log(self, metrics):
        metrics.update(self.infos)
        return metrics


class Actor(nn.Module):
    log_std_min = -5
    log_std_max = 2
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()
        self.repr_dim = repr_dim
        self.feature_dim = feature_dim
        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.policy = nn.Sequential(nn.Linear(feature_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True))
        self.mean_head = nn.Sequential(nn.Linear(hidden_dim, action_shape[0]))
        self.logstd_head = nn.Sequential(nn.Linear(hidden_dim, action_shape[0]))

        self.apply(utils.weight_init)

    def forward_trunk(self, x):
        if x.size(-1) == self.feature_dim:
            return x
        return self.trunk(x)

    def dist(self, latent):
        h = self.policy(latent)
        mu = self.mean_head(h)
        log_std = self.logstd_head(h)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1)
        return mu, log_std

    def forward(self, obs, compute_pi=True, with_logprob=True):
        mu, log_std = self.dist(self.forward_trunk(obs))

        if compute_pi:
            std = log_std.exp()
            noise = torch.randn_like(mu)
            pi = mu + noise * std
        else:
            pi = None

        if with_logprob:
            log_pi = utils.gaussian_logprob(noise, log_std)
        else:
            log_pi = None

        mu, pi, log_pi = utils.squash(mu, pi, log_pi)

        return mu, pi, log_pi, log_std


class Critic(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())
        self.Q1 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))
        self.Q2 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))
        self.apply(utils.weight_init)

    def forward(self, obs, action):
        h = self.trunk(obs)
        h_action = torch.cat([h, action], dim=-1)
        q1 = self.Q1(h_action)
        q2 = self.Q2(h_action)
        return q1, q2


class ECritic(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())
        self.Q = nn.Sequential(
            utils.EnsembleLinear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), utils.EnsembleLinear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), utils.EnsembleLinear(hidden_dim, 1))
        self.apply(utils.weight_init)

    def forward(self, obs, action):
        h = self.trunk(obs)
        h_action = torch.cat([h, action], dim=-1)
        q = self.Q(h_action)
        return q


class DrQAgent:
    def __init__(self, obs_shape, action_shape, device, lr, feature_dim,
                 hidden_dim, critic_target_tau, num_expl_steps,
                 update_every_steps, init_temperature, use_tb,
                 actor_update_times=1, critic_update_times=1, critic_target_update_freq=1,
                 augmentation=RandomShiftsAug(pad=4)):
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.use_tb = use_tb
        self.num_expl_steps = num_expl_steps
        self.init_temperature = init_temperature
        self.actor_update_times = actor_update_times
        self.critic_update_times = critic_update_times
        self.critic_target_update_freq = critic_target_update_freq
        # record
        self.update_critic_steps = 0
        self.update_actor_steps = 0
        self.update_encoder_steps = 0

        # models
        self.encoder = Encoder(obs_shape).to(device)
        self.actor = Actor(self.encoder.repr_dim, action_shape, feature_dim,
                           hidden_dim).to(device)

        self.critic = Critic(self.encoder.repr_dim, action_shape, feature_dim,
                             hidden_dim).to(device)
        self.critic_target = Critic(self.encoder.repr_dim, action_shape,
                                    feature_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        self.target_entropy = -np.prod(action_shape)

        # optimizers
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=lr, betas=(0.5, 0.999))

        # data augmentation
        self.aug = augmentation

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)

    def act(self, obs, step=1.0, eval_mode=False):
        obs = torch.as_tensor(obs, device=self.device)
        obs = self.encoder(obs.unsqueeze(0))
        mu_action, pi_action, *_ = self.actor(obs,
                                              not eval_mode,
                                              with_logprob=False)
        if eval_mode:
            action = mu_action
        else:
            action = pi_action
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        return action.cpu().numpy()[0]

    @property
    def update_times(self):
        return max(max(self.critic_update_times, self.actor_update_times), 1)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def print_update_record(self):
        print("| Update Actor Times: %d | Update Critic Times: %d | Update Encoder Times: %d" % (
            self.update_actor_steps, self.update_critic_steps, self.update_encoder_steps
        ))

    def calc_value(self, obs, is_target=True):
        _, action, log_prob, _ = self.actor(obs)
        critic = self.critic if not is_target else self.critic_target
        Q1, Q2 = critic(obs, action)
        return torch.min(Q1, Q2) - self.alpha * log_prob.unsqueeze(-1)

    def update_critic(self, obses, action, reward, discount, next_obses, step):
        metrics = dict()
        self.update_critic_steps += 1
        self.update_encoder_steps += 1
        obs1, obs2 = obses
        next_obs1, next_obs2 = next_obses

        with torch.no_grad():
            target_V = (self.calc_value(next_obs1) + self.calc_value(next_obs2)) / 2
            target_Q = reward.float() + (discount * target_V)

        Q1, Q2 = self.critic(obs1, action)
        Q3, Q4 = self.critic(obs2, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q) \
                      + F.mse_loss(Q3, target_Q) + F.mse_loss(Q4, target_Q)

        if self.use_tb:
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q1'] = Q1.mean().item()
            metrics['critic_q2'] = Q2.mean().item()
            metrics['critic_loss'] = critic_loss.item() / 2
            Qmin, Qmax = torch.min(Q1, Q2), torch.max(Q1, Q2)
            plt.scatter(Qmin.detach().cpu().numpy(), target_Q.detach().cpu().numpy(), s=50, alpha=0.3)
            plt.scatter(Qmax.detach().cpu().numpy(), target_Q.detach().cpu().numpy(), s=50, alpha=0.3)
            plt.plot([target_Q.min().item(), target_Q.max().item()], [target_Q.min().item(), target_Q.max().item()], c='r')
            metrics['Q_targQ_fig'] = plt.gcf()

        # optimize encoder and critic
        optim = dict(opt1=self.encoder_opt, opt2=self.critic_opt)
        utils.update_params(optim, critic_loss)

        return metrics

    def update_actor(self, obs, step, behavioural_action=None):
        metrics = dict()
        self.update_actor_steps += 1

        _, action, log_prob, _ = self.actor(obs)

        # optimize alpha
        alpha_loss = (self.alpha * (-log_prob - self.target_entropy).detach()).mean()
        utils.update_params(self.alpha_opt, alpha_loss)

        Q1, Q2 = self.critic(obs, action)
        Q = torch.min(Q1, Q2)

        actor_policy_improvement_loss = (self.alpha.detach() * log_prob - Q).mean()

        actor_loss = actor_policy_improvement_loss

        # optimize actor
        utils.update_params(self.actor_opt, actor_loss)

        if self.use_tb:
            metrics['actor_loss'] = actor_policy_improvement_loss.item()
            metrics['actor_logprob'] = log_prob.mean().item()
            metrics['actor_ent'] = -(self.alpha * log_prob).mean().item()
            metrics['alpha_loss'] = self.alpha.item()

        return metrics

    def update(self, replay_buffer, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_buffer)
        obs, action, reward, discount, next_obs = utils.to_torch(
            batch, self.device)

        # augment
        o1, o2 = self.aug(obs.float()), self.aug(obs.float())
        next_o1, next_o2 = self.aug(next_obs.float()), self.aug(next_obs.float())
        # encode
        o1, o2 = self.encoder(o1), self.encoder(o2)
        with torch.no_grad():
            next_o1, next_o2 = self.encoder(next_o1), self.encoder(next_o2)

        if self.use_tb:
            metrics['batch_reward'] = reward.mean().item()

        # update critic
        metrics.update(self.update_critic((o1, o2), action, reward, discount,
                                          (next_o1, next_o2), step))

        # update actor
        metrics.update(self.update_actor(o1.detach(), step))

        # update critic target
        if step % self.critic_target_update_freq == 0:
            utils.soft_update_params(self.critic, self.critic_target, self.critic_target_tau)
        metrics = self.encoder.log(metrics)
        return metrics
