# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import random
import re
import time
import math
import datetime

import numpy as np
import h5py
from collections import deque
import dmc
from dm_env import StepType
from VRL.numpy_replay_buffer import EfficientReplayBuffer

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as pyd
from torch.distributions.utils import _standard_normal


class eval_mode:
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data +
                                (1 - tau) * target_param.data)


def to_torch(xs, device):
    return tuple(torch.as_tensor(x, device=device).float() for x in xs)


def calc_time(start_time):
    return str(datetime.timedelta(seconds=int(time.time() - start_time)))


def set_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def weight_init(m):
    if isinstance(m, EnsembleLinear) or isinstance(m, nn.Linear):
        # nn.init.orthogonal_(m.weight.data)
        nn.init.xavier_uniform_(m.weight.data, gain=1)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


def update_params(optim: dict, loss: torch.Tensor, retain_graph=False,
                  set_to_none=True, grad_cliping=False, networks=None):
    if not isinstance(optim, dict):
        optim = dict(optimizer=optim)
    for opt in optim:
        optim[opt].zero_grad(set_to_none=set_to_none)
    loss.backward(retain_graph=retain_graph)
    # Clip norms of gradients to stebilize training.
    if grad_cliping:
        try:
            for net in networks:
                nn.utils.clip_grad_norm_(net.parameters(), grad_cliping)
        except:
            nn.utils.clip_grad_norm_(networks.parameters(), grad_cliping)
    for opt in optim:
        optim[opt].step()


def init_sp_logger(logger):
    infos = dict(EpRet=0, EpNum=0, EpLen=0, critic_q1=0, critic_q2=0,
                 critic_target_q=0, critic_loss=0, actor_loss=0)
    logger.store(**infos)


def print_log(logger, step, frame, env_names=None):
    # Log info about epoch
    logger.log_tabular('Epoch', frame // 10000)
    logger.log_tabular('TotalEnvInteracts', frame)
    logger.log_tabular('Step', step)
    logger.log_tabular('EpRet', average_only=True)
    logger.log_tabular('EpNum', average_only=True)
    logger.log_tabular('EpLen', average_only=True)

    logger.log_tabular('episode_score', average_only=True)
    logger.log_tabular('episode_reward', average_only=True)
    logger.log_tabular('episode_std', average_only=True)
    if env_names is not None:
        if not isinstance(env_names, list) and isinstance(env_names, str):
            env_names = [env_names]
        for env_name in env_names:
            logger.log_tabular(f'{env_name}_episode_score', average_only=True)
            logger.log_tabular(f'{env_name}_episode_reward', average_only=True)
            logger.log_tabular(f'{env_name}_episode_std', average_only=True)

    logger.log_tabular('critic_q1', average_only=True)
    logger.log_tabular('critic_q2', average_only=True)
    logger.log_tabular('critic_target_q', average_only=True)
    logger.log_tabular('critic_loss', average_only=True)
    logger.log_tabular('actor_loss', average_only=True)
    logger.dump_tabular()


class Until:
    def __init__(self, until, action_repeat=1):
        self._until = until
        self._action_repeat = action_repeat

    def __call__(self, step):
        if self._until is None:
            return True
        until = self._until // self._action_repeat
        return step <= until


class Every:
    def __init__(self, every, action_repeat=1):
        self._every = every
        self._action_repeat = action_repeat

    def __call__(self, step):
        if self._every is None:
            return False
        every = self._every // self._action_repeat
        if step % every == 0:
            return True
        return False


class Timer:
    def __init__(self):
        self._start_time = time.time()
        self._last_time = time.time()

    def reset(self):
        elapsed_time = time.time() - self._last_time
        self._last_time = time.time()
        total_time = time.time() - self._start_time
        return elapsed_time, total_time

    def total_time(self):
        return time.time() - self._start_time


def gaussian_logprob(noise, log_std):
    """Compute Gaussian log probability."""
    residual = (-0.5 * noise.pow(2) - log_std).sum(-1, keepdim=True)
    return residual - 0.5 * np.log(2 * np.pi) * noise.size(-1)


def squash(mu, pi, log_pi):
    """Apply squashing function.
    See appendix C from https://arxiv.org/pdf/1812.05905.pdf.
    """
    mu = torch.tanh(mu)
    if pi is not None:
        pi = torch.tanh(pi)
    if log_pi is not None:
        log_pi -= torch.log(F.relu(1 - pi.pow(2)) + 1e-6).sum(-1, keepdim=True)
        log_pi = log_pi.squeeze(-1)
    return mu, pi, log_pi


class EnsembleLinear(nn.Module):

    def __init__(self, in_features, out_features, in_channels=5, weight_decay=0., bias=True):
        super(EnsembleLinear, self).__init__()
        self.in_channels = in_channels
        self.in_features = in_features
        self.out_features = out_features
        self.weight_decay = weight_decay

        self.weight = nn.Parameter(torch.empty((in_channels, in_features, out_features)))
        if bias:
            self.bias = nn.Parameter(torch.empty(in_channels, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # nn.init.trunc_normal_(self.weight, std=1/(2*self.in_features**0.5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        # # input: (batch_size, in_features)
        # output = input @ self.weight + self.bias[:,None,:] if self.bias is not None else input @ self.weight
        # return output # output: (in_channels, batch_size, out_features)
        if input.ndim == 2 or input.ndim == 3:
            # input: (batch_size, in_features) or (in_channels, batch_size, in_features)
            # output: (in_channels, batch_size, out_features)
            output = input @ self.weight
            output = output + self.bias[:,None,:] if self.bias is not None else output
        elif input.ndim == 4:
            # input: (in_channels, num_sample, batch_size, in_features)
            # output: (in_channels, num_sample, batch_size, out_features)
            output = input @ self.weight.unsqueeze(1)
            output = output + self.bias[:,None,None,:] if self.bias is not None else output
        else:
            raise NotImplementedError
        return output

    def get_params(self, indexes):
        assert len(indexes) <= self.in_channels and max(indexes) < self.in_channels
        if self.bias is not None:
            return self.weight.data[indexes], self.bias.data[indexes]
        return self.weight.data[indexes]

    def save_params(self, indexes, params):
        assert len(indexes) <= self.in_channels and max(indexes) < self.in_channels
        if self.bias is not None:
            weight, bias = params
            self.weight.data[indexes] = weight
            self.bias.data[indexes] = bias
        else:
            self.weight.data[indexes] = params

    def extra_repr(self):
        return 'in_features={}, out_features={}, in_channels={}, bias={}'.format(
            self.in_features, self.out_features, self.in_channels, self.bias is not None
        )


step_type_lookup = {
    0: StepType.FIRST,
    1: StepType.MID,
    2: StepType.LAST
}


def cosine_similarity(z1, z2):
    z1 = z1 / torch.norm(z1, dim=-1, p=2, keepdim=True)
    z2 = z2 / torch.norm(z2, dim=-1, p=2, keepdim=True) 
    similarity = z1 @ z2.transpose(-1, -2)
    return similarity


def compute_ce_loss(z1, z2, temperature=1.0):
    # case 1:
    #       z1: (N, X, B, Z); z2: (N, X, B, Z); output: (N, X)
    # case 2:
    #       z1: (B, Z); z2: (N, B, Z); output: (N, )
    similarity = cosine_similarity(z1, z2) / temperature
    batch_size = similarity.size(-1)
    with torch.no_grad():
        target = torch.eye(batch_size).to(z1.device)
        pred_prob = torch.softmax(similarity, dim=-1)
        accuracy = (pred_prob * target).sum(-1).mean()
        diff = pred_prob - target.float()
    loss = (similarity * diff).sum(-1).mean(-1)
    return loss, accuracy


def compute_cl_loss(z1, z2, labels=None, mask=None, temperature=1.0):
    similarity = cosine_similarity(z1, z2) / temperature
    if mask is not None:
        similarity = similarity[~mask].view(similarity.size(0), -1) # [B, B-1]
    with torch.no_grad():
        if labels is None:
            labels = torch.arange(z1.size(0)).to(z1.device)
            target = torch.eye(z1.size(0), dtype=torch.bool).to(z1.device)
        else:
            target = F.one_hot(labels, similarity.size(1)).to(z1.device)
        pred_prob = torch.softmax(similarity, dim=-1)
        accuracy = (pred_prob * target).sum(-1).mean()
        diff = pred_prob - target.float()
    loss = (similarity * diff).sum(-1).mean(-1)
    return loss, accuracy