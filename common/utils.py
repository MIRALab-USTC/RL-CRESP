from turtle import back
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import datetime
import yaml
import random
import numpy as np
from torch.distributions import Normal
from torch.distributions.utils import _standard_normal
from common.logger_tb import Logger
from common.logx import EpochLogger, setup_logger_kwargs


_ACTION_REPEAT = {
    'dmc.ball_in_cup.catch': 4, 'dmc.cartpole.swingup': 8, 'dmc.cheetah.run': 4,
    'dmc.finger.spin': 2, 'dmc.reacher.easy': 4, 'dmc.walker.walk': 2
}

overwrite = lambda key, value, dictionary: dictionary[key] if key in dictionary.keys() else value


def read_config(args, config_dir):
    # read common.yaml
    with open(config_dir / 'common.yaml') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    # read algo.yaml and update config.algo_params
    with open(config_dir / 'algo.yaml') as f:
        alg_config = yaml.load(f, Loader=yaml.SafeLoader)
    config['algo_params'].update(alg_config)

    # read auxiliary.yaml and update config
    if args.auxiliary is not None:
        with open(config_dir / 'auxiliary.yaml') as f:
            aux_config = yaml.load(f, Loader=yaml.SafeLoader)
        config['auxiliary_params'] = aux_config[args.auxiliary]

    # read agent.yaml and update config
    with open(config_dir / 'agent.yaml') as f:
        agent_config = yaml.load(f, Loader=yaml.SafeLoader)

    for key in agent_config[args.agent].keys():
        if isinstance(agent_config[args.agent][key], dict):
            config[key].update(agent_config[args.agent][key])
        else:
            config[key] = agent_config[args.agent][key]

    config['train_params']['action_repeat'] = _ACTION_REPEAT[args.env]
    config['agent_base_params']['action_repeat'] = _ACTION_REPEAT[args.env]
    config['agent_base_params']['num_sources'] = config['setting']['num_sources']

    # overwrite the config by using the args
    args_dict = vars(args)
    if args.disenable_default:
        config = overwrite_config(config, args_dict)
    else:
        config['setting'] = overwrite_config(config['setting'], args_dict)

    config['train_params']['eval_freq'] = config['steps_per_epoch'] // config['train_params']['action_repeat']
    config['train_params']['total_steps'] = config['train_params']['total_steps'] // config['train_params']['action_repeat']
    return config


def overwrite_config(config: dict, args: dict):
    for key, value in config.items():
        if isinstance(value, dict):
            overwrite_config(config[key], args)
        else:
            config[key] = overwrite(key, value, args)
    return config


def calc_exp_name(args, config):
    def env_dis_mode(background, camera, color):
        if not (background | camera | color):
            return 'Clean'
        if background:
            return 'Ba'
        elif camera:
            return 'Ca'
        elif color:
            return 'Co'
        raise ValueError(background, camera, color)

    dyn = 'Dyn' if args.dynamic else 'NDyn'
    bf_nstep_rsd = config['buffer_params']['nstep_of_rsd']
    extr_dim = '' if not args.extr_update_via_qfloss \
        else 'euf_qf%d' % args.extr_update_freq_via_qfloss
    mode = '%s-%s-%s' % (args.agent, args.base, extr_dim)
    if args.auxiliary is not None:
        discount_rs = 'rs'
        if args.rs_fc:
            discount_rs += 'fc'
        discount_rs += ('%s' % args.discount_of_rs)
        cf_config = 'CF%s_%d.%d_%s_%d' % (
            args.opt_mode, args.opt_num, args.num_ensemble, args.omega_opt_mode, args.num_sample)
        mode = '%s-%s-%s-nrs%d-%s' % (
            args.auxiliary, mode, cf_config, bf_nstep_rsd, discount_rs)

    return '%s-%s-%s-%s' % (
        args.env,
        dyn,
        env_dis_mode(args.background, args.camera, args.color) + '%s' % args.num_sources,
        mode
    )


def init_logger(args, config, work_dir):
    exp_name = calc_exp_name(args, config)
    config['exp_name'] = exp_name
    work_dir = work_dir / 'data' / ('%s' % args.env)
    # logger_kwargs = setup_logger_kwargs(exp_name, args.seed, work_dir)
    work_dir = work_dir / ('%s' % exp_name) / ('%s_s%d' % (exp_name, args.seed))
    logger_kwargs = dict(output_dir=work_dir, exp_name=exp_name)
    logsp = EpochLogger(**logger_kwargs)
    logsp.save_config(config)
    # utils.make_dir(work_dir)
    logtb = Logger(work_dir, action_repeat=config['train_params']['action_repeat'], use_tb=args.save_tb)
    return dict(tb=logtb, sp=logsp), work_dir


def local_time(t = time.time()):
    local_time = time.localtime(t)
    tm_mon = '0%d' % local_time.tm_mon if local_time.tm_mon // 10 == 0 \
        else local_time.tm_mon
    tm_mday = '0%d' % local_time.tm_mday if local_time.tm_mday // 10 == 0 \
        else local_time.tm_mday
    return "%s.%s.%s" % (local_time.tm_year, tm_mon, tm_mday)


def calc_time(start_time):
    return str(datetime.timedelta(seconds=int(time.time() - start_time)))


def update_linear_schedule(optimizer, lr):
    """Decreases the learning rate linearly"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(
            tau * param.data + (1 - tau) * target_param.data
        )


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


def _handle_data(data):
    if (data > 1.0).any():
        return data / 255.0# - 0.5
    return data


def rank(data, value, dim, num, mode):
    assert data.size(dim) >= num, print(data.size(), dim, num)
    idx = value.sort(dim=dim)[1]
    if mode == 'max':
        return data[idx[-num:]]
    elif mode == 'min':
        return data[idx[:num]]
    elif mode == 'random':
        idx = np.random.randint(0, idx.size(0), num)
        return data[idx]
    raise ValueError(mode)


def log(logger, key, value_dict, epoch):
    assert isinstance(logger, dict) and isinstance(value_dict, dict), \
            print(type(logger), type(value_dict))
    logger['tb'].save_log_dict(key, value_dict, epoch)
    logger['sp'].store(**value_dict)


def update_params(optim, loss, retain_graph=False, grad_cliping=False, networks=None):
    if not isinstance(optim, dict):
        optim = dict(optimizer=optim)
    for opt in optim:
        optim[opt].zero_grad()
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


def cosine_similarity(z1, z2):
    z1 = z1 / torch.norm(z1, dim=-1, p=2, keepdim=True)
    z2 = z2 / torch.norm(z2, dim=-1, p=2, keepdim=True) 
    similarity = z1 @ z2.transpose(-1, -2)
    return similarity


def compute_cl_loss(z1, z2, labels=None, mask=None, temperature=1.0, output_acc=False):
    similarity = cosine_similarity(z1, z2) / temperature
    if similarity.ndim == 3:
        similarity = similarity.squeeze(1)
    if mask is not None:
        if (mask.sum(-1) != 1.0).any():
            similarity[mask] = similarity[mask] * 0.0
        else:
            similarity = similarity[~mask].view(similarity.size(0), -1)
    with torch.no_grad():
        if labels is None:
            labels = torch.arange(z1.size(0)).to(z1.device)
            target = torch.eye(z1.size(0), dtype=torch.bool).to(z1.device)
        else:
            target = F.one_hot(labels, similarity.size(1)).to(z1.device)
        pred_prob = torch.softmax(similarity, dim=-1)
        i = pred_prob.max(dim=-1)[1]
        accuracy = (i==labels).sum(-1).float() / labels.size(0)
        if accuracy.ndim != 1:
            accuracy = accuracy.mean()
        # accuracy = (pred_prob * target).sum(-1)
        diff = pred_prob - target.float()
    loss = (similarity * diff).sum(-1).mean(-1)
    if output_acc:
        return loss, accuracy
    return loss#, pred_prob, accuracy


def module_hash(module):
    result = 0
    for tensor in module.state_dict().values():
        result += tensor.sum().item()
    return result


def make_dir(dir_path):
    try:
        # os.mkdir(dir_path)
        dir_path.mkdir()
    except OSError:
        pass
    return dir_path


def preprocess_obs(obs, bits=5):
    """Preprocessing image, see https://arxiv.org/abs/1807.03039."""
    bins = 2**bits
    assert obs.dtype == torch.float32
    if bits < 8:
        obs = torch.floor(obs / 2**(8 - bits))
    obs = obs / bins
    obs = obs + torch.rand_like(obs) / bins
    obs = obs - 0.5
    return obs


class TruncatedNormal(Normal):
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


class eval_mode(object):
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


class train_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(True)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def freeze_module(module):
    if isinstance(module, list) or isinstance(module, tuple):
        for m in module:
            for param in m.parameters():
                param.requires_grad = False
    else:
        for param in module.parameters():
            param.requires_grad = False


def activate_module(module):
    if isinstance(module, list) or isinstance(module, tuple):
        for m in module:
            for param in m.parameters():
                param.requires_grad = True
    else:
        for param in module.parameters():
            param.requires_grad = True


def normalize(value):
    mean, std = value.mean().item(), value.std().item()
    return (value - mean) / std


def np2torch(x, device):
    return torch.as_tensor(x).to(device)


def log_sum_exp(value, dim):
    m, _ = torch.max(value, dim=dim, keepdim=True)
    value_norm = value - m
    m = m.squeeze(dim)
    return m+torch.log(torch.sum(torch.exp(value_norm), dim=dim))


def atanh(x):
    one_plus_x = (1 + x).clamp(min=1e-6)
    one_minus_x = (1 - x).clamp(min=1e-6)
    return 0.5 * torch.log(one_plus_x/ one_minus_x)


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

    def __init__(self, in_features, out_features, in_channels, weight_decay=0., bias=True):
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
            # output = torch.einsum('ij, kjl -> kil', input, self.weight)
            # output = torch.einsum('kij, kjl -> kil', input, self.weight)
            output = input @ self.weight
            output = output + self.bias[:,None,:] if self.bias is not None else output
        elif input.ndim == 4:
            # input: (in_channels, num_sample, batch_size, in_features)
            # output: (in_channels, num_sample, batch_size, out_features)
            # output = torch.einsum('skij, skjl -> skil', input, self.weight.unsqueeze(1))
            output = input @ self.weight.unsqueeze(1)
            output = output + self.bias[:,None,None,:] if self.bias is not None else output
        else:
            raise NotImplementedError
        return output

    def extra_repr(self):
        return 'in_features={}, out_features={}, in_channels={}, bias={}'.format(
            self.in_features, self.out_features, self.in_channels, self.bias is not None
        )

def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, EnsembleLinear) or isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


def mlp(input_dim=None, hidden_dim=1024, output_dim=None, hidden_depth=2, output_mod=None,
        inplace=False, handle_dim=None, channel_dim=1, weight_decay=None, ensemble=False, activation=nn.ReLU):
    '''
    output_mod:     output activation function
        output_mod=nn.ReLU(inplace):            inplace-->False or True;
        output_mod=nn.LayerNorm(handle_dim):    handle_dim-->int
        output_mod=nn.Softmax(handle_dim):      handle_dim-->0 or 1
    linear:         choice[nn.Linear, EnsembleLinear]
        linear=EnsembleLinear:                  channel_dim-->int: ensemble number
    '''
    if not ensemble:
        linear = lambda n_input, n_output, channel_dim, weight_decay: nn.Linear(n_input, n_output)
    else:
        linear = lambda n_input, n_output, channel_dim, weight_decay: EnsembleLinear(
            in_features=n_input, out_features=n_output, in_channels=channel_dim, weight_decay=weight_decay
        )

    if weight_decay is None:
        weight_decay = [0.]
    if len(weight_decay) == 1:
        weight_decay = list(weight_decay) * (hidden_depth + 1)

    if hidden_depth == 0:
        mods = [linear(input_dim, output_dim, channel_dim, weight_decay[0])]
    else:
        mods = [linear(input_dim, hidden_dim, channel_dim, weight_decay[0]), activation(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [linear(hidden_dim, hidden_dim, channel_dim, weight_decay[i+1]), activation(inplace=True)]
        mods.append(linear(hidden_dim, output_dim, channel_dim, weight_decay[-1]))
    if output_mod is not None:
        try:
            mods.append(output_mod(inplace=inplace))
        except:
            if handle_dim in [0, 1, -1]:
                mods.append(output_mod(dim=handle_dim))
            elif handle_dim is not None:
                mods.append(output_mod(handle_dim))
            else:
                mods.append(output_mod())
    trunk = nn.Sequential(*mods)
    return trunk


def get_decay_loss(model):
    decay_loss = 0.
    for m in model.children():
        if isinstance(m, EnsembleLinear):
            decay_loss += m.weight_decay * torch.sum(torch.square(m.weight)) / 2.
    return decay_loss

class Swish(nn.Module):

    def __init__(self, inplace=False):
        super(Swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x.mul_(torch.sigmoid(x)) if self.inplace else x * torch.sigmoid(x)
