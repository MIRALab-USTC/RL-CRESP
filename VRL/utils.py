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


def kl_divergence(p_mu, p_std, q_mu, q_std):
    var_ratio = (p_std / q_std).pow(2)
    t1 = ((p_mu - q_mu) / q_std).pow(2)
    kl_divergence = 0.5 * (var_ratio + t1 - 1 - var_ratio.log()).mean()
    return kl_divergence


def soft_clamp(x : torch.Tensor, _min=None, _max=None):
    # clamp tensor values while mataining the gradient
    if _max is not None:
        x = _max - F.softplus(_max - x)
    if _min is not None:
        x = _min + F.softplus(x - _min)
    return x


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


def cross_entropy(inputs, labels):
    # inputs: (K, B, B); labels: (B,); output: (K,)
    K = inputs.shape[0]
    inputs = inputs.view(-1, inputs.shape[-1])
    log_softmax = torch.nn.functional.log_softmax(inputs, dim=-1) # (K*B, B)
    indices = labels.view(-1, 1).repeat(K, 1) # (K*B, 1)
    log_probs = log_softmax.gather(1, indices).view(K, -1) # (K, B)
    return -log_probs.mean(-1)


# def compute_curl_loss(z1, z2):
#     # z1: (K, B, Z); z2: (B, Z); output: (K, )
#     logits = z1 @ z2.T # (K, B, B)
#     logits = logits - torch.max(logits, -1)[0][:, :, None] # (K, B, B) - (K, B, 1)
#     labels = torch.arange(logits.shape[-1]).long().to(z1.device) # (B,)
#     return cross_entropy(logits, labels)


def compute_curl_loss(z1, z2):
    logits = torch.matmul(z1, z2.T)
    logits = logits - torch.max(logits, 1)[0][:, None]
    labels = torch.arange(logits.shape[0]).long().to(z1.device)
    return F.cross_entropy(logits, labels)


def schedule(schdl, step):
    try:
        return float(schdl)
    except ValueError:
        match = re.match(r'linear\((.+),(.+),(.+)\)', schdl)
        if match:
            init, final, duration = [float(g) for g in match.groups()]
            mix = np.clip(step / duration, 0.0, 1.0)
            return (1.0 - mix) * init + mix * final
        match = re.match(r'step_linear\((.+),(.+),(.+),(.+),(.+)\)', schdl)
        if match:
            init, final1, duration1, final2, duration2 = [
                float(g) for g in match.groups()
            ]
            if step <= duration1:
                mix = np.clip(step / duration1, 0.0, 1.0)
                return (1.0 - mix) * init + mix * final1
            else:
                mix = np.clip((step - duration1) / duration2, 0.0, 1.0)
                return (1.0 - mix) * final1 + mix * final2
    raise NotImplementedError(schdl)


# Schedulers for beta
class Scheduler(object):
    def __call__(self, **kwargs):
        raise NotImplementedError()

class LinearScheduler(Scheduler):
    def __init__(
        self, start_value, end_value, n_iterations, start_iteration=0
    ):
        self.start_value = start_value
        self.end_value = end_value
        self.n_iterations = n_iterations
        self.start_iteration = start_iteration
        self.m = (end_value - start_value) / n_iterations

    def __call__(self, iteration):
        if iteration > self.start_iteration + self.n_iterations:
            return self.end_value
        elif iteration <= self.start_iteration:
            return self.start_value
        else:
            return (
                (iteration - self.start_iteration) * self.m + self.start_value
            )

class ExponentialScheduler(LinearScheduler):
    def __init__(
        self, start_value, end_value, n_iterations, start_iteration=0,
    ):
        super(ExponentialScheduler, self).__init__(
            start_value=np.log10(start_value),
            end_value=np.log10(end_value),
            n_iterations=n_iterations,
            start_iteration=start_iteration
        )

    def __call__(self, iteration):
        linear_value = super(ExponentialScheduler, self).__call__(iteration)
        return np.power(10., linear_value)


class TruncatedNormal(pyd.Normal):
    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6):
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps

    def _clamp(self, x):
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        x = x - x.detach() + clamped_x.detach()
        return x

    def sample(self, clip=None, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape,
                               dtype=self.loc.dtype,
                               device=self.loc.device)
        eps *= self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        x = self.loc + eps
        return self._clamp(x)


import csv

CSV_HEAD = ['return', 'seed', 'type', 'num_domains', 'start_scale', 'end_scale',
            'target_scale', 'length', 'nstep', 'discount', 'algo', 'batch_size']

def save_log_in_csv(eval_info: dict, write_dir: str, max_episode: int, cfg: dict):
    write_dir = f"{write_dir}-{max_episode}.csv"
    if cfg.dcs_type == 'background':
        start_scale, end_scale, target_scale = 1, cfg.num_sources, 30
    elif cfg.dcs_type == 'color':
        start_scale, end_scale = cfg.scale, cfg.scale_end
        target_scale = cfg.eval_scale
    elif cfg.dcs_type == 'common':
        start_scale, end_scale, target_scale = 0, 0, 0
    else:
        raise ValueError("Environment type %s is not enable." % cfg.dcs_type)

    try:
        with open(write_dir, 'r') as f:
            line = f.readline()
            write_head = line[:6] != CSV_HEAD[0]
    except:
        write_head = True

    with open(write_dir, 'a') as f:
        csv_writer = csv.DictWriter(f, fieldnames=CSV_HEAD)
        data = {}
        if write_head:
            for key in CSV_HEAD:
                data[key] = key
            csv_writer.writerow(data)
            data = {}
        data['return'] = eval_info['episode_reward']
        data['seed'] = cfg.seed
        data['type'] = cfg.dcs_type
        data['num_domains'] = cfg.num_sources if not cfg.dcs_type == 'common' else 1
        data['start_scale'] = start_scale
        data['end_scale'] = end_scale
        data['target_scale'] = target_scale
        # data['length'] = cfg.rsd_nstep
        data['nstep'] = cfg.nstep
        data['discount'] = cfg.rsd_discount if cfg.rsd_discount else cfg.discount
        data['algo'] = cfg.exp_name
        data['batch_size'] = cfg.batch_size
        csv_writer.writerow(data)
    print(max_episode, "return: ", eval_info['episode_reward'])


import json
import os
import torchvision.transforms as TF
import torchvision.datasets as datasets

places_dataloader = None
places_iter = None

def load_config(key=None):
    current_dir = os.path.dirname(__file__)
    path = os.path.join(f'{current_dir}/cfgs', 'aug_config.cfg')
    with open(path) as f:
        data = json.load(f)
    if key is not None:
        return data[key]
    return data

def _load_places(batch_size=256, image_size=84, num_workers=8, use_val=False):
	global places_dataloader, places_iter
	partition = 'val' if use_val else 'train'
	print(f'Loading {partition} partition of places365_standard...')
	for data_dir in load_config('datasets'):
		if os.path.exists(data_dir):
			fp = os.path.join(data_dir, 'places365_standard', partition)
			if not os.path.exists(fp):
				print(f'Warning: path {fp} does not exist, falling back to {data_dir}')
				fp = data_dir
			places_dataloader = torch.utils.data.DataLoader(
				datasets.ImageFolder(fp, TF.Compose([
					TF.RandomResizedCrop(image_size),
					TF.RandomHorizontalFlip(),
					TF.ToTensor()
				])),
				batch_size=batch_size, shuffle=True,
				num_workers=num_workers, pin_memory=True)
			places_iter = iter(places_dataloader)
			break
	if places_iter is None:
		raise FileNotFoundError('failed to find places365 data at any of the specified paths')
	print('Loaded dataset from', data_dir)

def _get_places_batch(batch_size):
	global places_iter
	try:
		imgs, _ = next(places_iter)
		if imgs.size(0) < batch_size:
			places_iter = iter(places_dataloader)
			imgs, _ = next(places_iter)
	except StopIteration:
		places_iter = iter(places_dataloader)
		imgs, _ = next(places_iter)
	return imgs.cuda()

def random_overlay(x, dataset='places365_standard'):
	"""Randomly overlay an image from Places"""
	global places_iter
	alpha = 0.5

	if dataset == 'places365_standard':
		if places_dataloader is None:
			_load_places(batch_size=x.size(0), image_size=x.size(-1))
		imgs = _get_places_batch(batch_size=x.size(0)).repeat(1, x.size(1)//3, 1, 1)
	else:
		raise NotImplementedError(f'overlay has not been implemented for dataset "{dataset}"')

	return ((1-alpha)*(x/255.) + (alpha)*imgs)*255.



# The SRM code, version 1, circle-ring shaped mask
def random_mask_freq_v1(x):
        p = random.uniform(0, 1)
        if p > 0.5:
             return x
        # need to adjust r1 r2 and delta for best performance
        r1=random.uniform(0,0.5)
        delta_r=random.uniform(0,0.035)
        r2=np.min((r1+delta_r,0.5))
        # print(r2)
        # generate Mask M
        B,C,H,W = x.shape
        center = (int(H/2), int(W/2))
        diagonal_lenth = max(H,W) # np.sqrt(H**2+W**2) is also ok, use a smaller r1
        r1_pix = diagonal_lenth * r1
        r2_pix = diagonal_lenth * r2
        Y_coord, X_coord = np.ogrid[:H, :W]
        dist_from_center = np.sqrt((Y_coord - center[0])**2 + (X_coord - center[1])**2)
        M = dist_from_center <= r2_pix
        M = M * (dist_from_center >= r1_pix)
        M = ~M

        # mask Fourier spectrum
        M = torch.from_numpy(M).float().to(x.device)
        srm_out = torch.zeros_like(x)
        for i in range(C):
            x_c = x[:,i,:,:]
            x_spectrum = torch.fft.fftn(x_c, dim=(-2,-1))
            x_spectrum = torch.fft.fftshift(x_spectrum, dim=(-2,-1))
            out_spectrum = x_spectrum * M
            out_spectrum = torch.fft.ifftshift(out_spectrum, dim=(-2,-1))
            srm_out[:,i,:,:] = torch.fft.ifftn(out_spectrum, dim=(-2,-1)).float()
        return srm_out

def random_mask_freq_v2(x):
    p = random.uniform(0, 1)
    if p > 0.5:
        return x

    # dynamicly select freq range to erase
    A = 0
    B = 0.5
    a = random.uniform(A, B)
    C = 2
    freq_limit_low = round(a, C)

    A = 0
    B = 0.05
    a = random.uniform(A, B)
    C = 2
    diff = round(a, C)
    freq_limit_hi = freq_limit_low + diff

    # b, 9, h, w
    b, c, h, w = x.shape
    x0, x1, x2 = torch.chunk(x, 3, dim=1)
    # b, 3, 3, h, w
    x = torch.cat((x0.unsqueeze(1), x1.unsqueeze(1), x2.unsqueeze(1)), dim=1)

    pass1 = torch.abs(torch.fft.fftfreq(x.shape[-1], device=x.device)) < freq_limit_hi
    pass2 = torch.abs(torch.fft.fftfreq(x.shape[-2], device=x.device)) < freq_limit_hi
    kernel1 = torch.outer(pass2, pass1)  # freq_limit_hi square is true

    pass1 = torch.abs(torch.fft.fftfreq(x.shape[-1], device=x.device)) < freq_limit_low
    pass2 = torch.abs(torch.fft.fftfreq(x.shape[-2], device=x.device)) < freq_limit_low
    kernel2 = torch.outer(pass2, pass1)  # freq_limit_low square is true

    kernel = kernel1 * (~kernel2)  # a square ring is true
    fft_1 = torch.fft.fftn(x, dim=(2, 3, 4))
    imgs = torch.fft.ifftn(fft_1 * (~kernel), dim=(2, 3, 4)).float()
    x0, x1, x2 = torch.chunk(imgs, 3, dim=1)
    imgs = torch.cat((x0.squeeze(1), x1.squeeze(1), x2.squeeze(1)), dim=1)

    return imgs
