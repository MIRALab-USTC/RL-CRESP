from turtle import forward
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from zmq import has
from common.utils import gaussian_logprob, squash, weight_init, mlp, Swish


class SGMLPActor(nn.Module):

    def __init__(
        self, action_shape, hidden_dim, repr_dim, encoder_feature_dim,
        log_std_min, log_std_max, l, action_limit
    ):
        super(SGMLPActor, self).__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.act_limit = action_limit
        self.state_dim = encoder_feature_dim
        self.repr_dim = repr_dim
        self.hidden_depth = l

        if repr_dim is not None:
            self.trunk = mlp(repr_dim, 0, self.state_dim, 0, nn.LayerNorm,
                             handle_dim=self.state_dim)
        self.pi_trunk = mlp(self.state_dim, hidden_dim, hidden_dim, l-1, nn.ReLU, True)
        self.pi_mean = mlp(hidden_dim, 0, action_shape[0], 0)
        self.pi_logstd = mlp(hidden_dim, 0, action_shape[0], 0)
        self.infos = dict()
        self.apply(weight_init)

    def _reprocess(self, pi):
        pi[pi == 1.0] -= 1e-10
        pi[pi == -1.0] += 1e-10
        return pi

    def _output(self, pi):
        if pi is None:
            return None
        return self._reprocess(self.act_limit * pi)

    def forward_trunk(self, obs, tanh=True):
        if self.repr_dim is None:
            return obs
        state = self.trunk(obs)
        self.infos['ln'] = state
        if tanh:
            state = torch.tanh(state)
            self.infos['tanh'] = state
        return state

    def dist(self, state, tanh=True):
        h = self.pi_trunk(self.forward_trunk(state, tanh))
        # mu, log_std = self.pi(state).chunk(2, dim=-1)
        mu = self.pi_mean(h)
        log_std = self.pi_logstd(h)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1)

        self.infos['mu'] = mu
        self.infos['std'] = log_std.exp()
        return mu, log_std, h

    def forward(self, state, compute_pi=True, with_logprob=True, tanh=True):
        mu, log_std, _ = self.dist(state, tanh)

        if compute_pi:
            std = log_std.exp()
            noise = torch.randn_like(mu)
            pi = mu + noise * std
        else:
            noise = torch.zeros_like(mu)
            pi = mu

        if with_logprob:
            log_pi = gaussian_logprob(noise, log_std)
        else:
            log_pi = None

        mu, pi, log_pi = squash(mu, pi, log_pi)
        self.infos['act'] = self._output(pi)

        return self._output(mu), self._output(pi), log_pi, log_std

    def act(self, state, deterministic=False, tanh=True):
        mu_action, pi_action, _, _ = self.forward(state, not deterministic, False, tanh)
        if deterministic:
            return mu_action
        return pi_action

    def log(self, L, step, log_freq, params=False):
        if not log_freq or step % log_freq != 0:
            return

        for k, v in self.infos.items():
            L.log_histogram('train_actor/%s_hist' % k, v, step)

        if params:
            if self.repr_dim is not None:
                L.log_param('train_actor/fc', self.trunk[0], step)
                L.log_param('train_actor/ln', self.trunk[1], step)
            for i in range(self.hidden_depth):
                L.log_param('train_actor/pi_fc%d' % i, self.pi_trunk[i * 2], step)
            L.log_param('train_actor/pi_mean', self.pi_mean[0], step)
            L.log_param('train_actor/pi_logstd', self.pi_logstd[0], step)


from common.utils import TruncatedNormal

class MLPActor(nn.Module):

    def __init__(self, action_shape, hidden_dim, repr_dim, encoder_feature_dim,
                 l, act_limit, act_noise=0.1, eps=1e-6):
        super(MLPActor, self).__init__()
        self.act_limit = act_limit
        self.act_noise = act_noise
        self.hidden_depth = l
        self.state_dim = encoder_feature_dim
        self.repr_dim = repr_dim
        self.eps = eps

        if repr_dim is not None:
            self.trunk = mlp(repr_dim, 0, self.state_dim, 0, nn.LayerNorm,
                            handle_dim=self.state_dim)
        self.pi = mlp(self.state_dim, hidden_dim, action_shape[0], l, nn.Tanh)
        self.infos = dict()
        self.apply(weight_init)

    def forward_trunk(self, obs, tanh=True):
        if self.repr_dim is None:
            return obs
        state = self.trunk(obs)
        self.infos['ln'] = state
        if tanh:
            state = torch.tanh(state)
            self.infos['tanh'] = state
        return state

    def forward(self, obs, deterministic=False, act_noise=None, clip=None, tanh=True, with_logprob=True):
        state = self.forward_trunk(obs, tanh)
        mu = self.act_limit * self.pi(state)
        self.infos['mu'] = mu

        if act_noise is None:
            act_noise = self.act_noise
        dist = TruncatedNormal(mu, torch.ones_like(mu) * act_noise)

        if deterministic:
            pi_action = dist.mean
        else:
            pi_action = dist.sample(clip=clip)
        
        if with_logprob:
            log_pi = dist.log_prob(pi_action).sum(-1, keepdim=True)
            return pi_action, log_pi, dist.entropy().sum(dim=-1)

        return pi_action

    def act(self, state, deterministic=False, act_noise=None, clip=None, tanh=True):
        return self.forward(state, deterministic, act_noise, clip, tanh, False)

    def log(self, L, step, log_freq, params=False):
        if not log_freq or step % log_freq != 0:
            return

        for k, v in self.infos.items():
            L.log_histogram('train_actor/%s_hist' % k, v, step)

        if params:
            if self.repr_dim is not None:
                L.log_param('train_actor/fc', self.trunk[0], step)
                L.log_param('train_actor/ln', self.trunk[1], step)
            for i in range(self.hidden_depth+1):
                L.log_param('train_actor/pi_fc%d' % i, self.pi[i * 2], step)


class Critic(nn.Module):
    """Critic network, employes two q-functions."""
    def __init__(self, action_shape, hidden_dim, repr_dim, encoder_feature_dim, l=2,
                 output_mod=None, num_q=2, output_dim=1):
        super(Critic, self).__init__()

        self.state_dim = encoder_feature_dim
        self.output_dim = output_dim
        self.hidden_depth = l
        self.repr_dim = repr_dim
        self.num_q = num_q
        
        if repr_dim is not None:
            self.trunk = nn.Sequential(nn.Linear(repr_dim, encoder_feature_dim),
                                       nn.LayerNorm(encoder_feature_dim))
        # self.q1 = mlp(self.state_dim + action_shape[0], hidden_dim, output_dim, l, output_mod)
        self.q1 = mlp(self.state_dim + action_shape[0], hidden_dim, output_dim, l, output_mod)
        self.q2 = mlp(self.state_dim + action_shape[0], hidden_dim, output_dim, l, output_mod) if num_q == 2 else None
        self.infos = dict()
        self.apply(weight_init)

    def forward_trunk(self, obs, tanh=True):
        if self.repr_dim is None:
            return obs
        state = self.trunk(obs)
        self.infos['ln'] = state
        if tanh:
            state = torch.tanh(state)
            self.infos['tanh'] = state
        return state

    def forward(self, state, action, tanh=True):
        state = self.forward_trunk(state)
        assert state.size()[:-1] == action.size()[:-1], print(state.size(), action.size())
        sa = torch.cat([state, action], 1)
        q1 = torch.squeeze(self.q1(sa), -1) # q:(batch_size,)
        q2 = torch.squeeze(self.q2(sa), -1)
        self.infos['q1'] = q1
        self.infos['q2'] = q2
        return q1, q2

    def forward_q(self, state, action):
        assert state.size()[:-1] == action.size()[:-1], print(state.size(), action.size())
        sa = torch.cat([state, action], 1)
        q1 = torch.squeeze(self.q1(sa), -1) # q:(batch_size,)
        q2 = torch.squeeze(self.q2(sa), -1)
        self.infos['q1'] = q1
        self.infos['q2'] = q2
        return q1, q2

    def Q1(self, state, action, tanh=True):
        state = self.forward_trunk(state)
        assert state.size()[:-1] == action.size()[:-1], print(state.size(), action.size())
        sa = torch.cat([state, action], 1)
        return torch.squeeze(self.q1(sa), -1)

    def log(self, L, step, log_freq, params=False):
        if not log_freq or step % log_freq != 0:
            return

        for k, v in self.infos.items():
            L.log_histogram('train_critic/%s_hist' % k, v, step)

        if params:
            if self.repr_dim is not None:
                L.log_param('train_critic/fc', self.trunk[0], step)
                L.log_param('train_critic/ln', self.trunk[1], step)
            for i in range(self.hidden_depth+1):
                L.log_param('train_critic/q1_fc%d' % i, self.q1[i * 2], step)
                L.log_param('train_critic/q2_fc%d' % i, self.q2[i * 2], step) if self.q2 is not None else 0


class EnsembleCritic(nn.Module):

    def __init__(self, action_shape, hidden_dim, repr_dim, encoder_feature_dim, l=2,
                 output_mod=None, num_q=2, output_dim=1, handle_dim=None):
        super(EnsembleCritic, self).__init__()

        self.state_dim = encoder_feature_dim
        self.output_dim = output_dim
        self.num_q = num_q
        self.hidden_depth = l
        self.repr_dim = repr_dim
        
        if repr_dim is not None:
            self.trunk = mlp(repr_dim, 0, self.state_dim, 0, nn.LayerNorm, handle_dim=self.state_dim)
        self.q = mlp(
            self.state_dim + action_shape[0], hidden_dim, output_dim, l,
            output_mod, handle_dim=handle_dim, channel_dim=num_q, ensemble=True
        )
        self.infos = dict()
        self.apply(weight_init)

    def forward_trunk(self, obs, tanh=True):
        if self.repr_dim is None:
            return obs
        state = self.trunk(obs)
        self.infos['ln'] = state
        if tanh:
            state = torch.tanh(state)
            self.infos['tanh'] = state
        return state

    def forward(self, state, action, minimize=True, tanh=True):
        state = self.forward_trunk(state, tanh)
        assert state.size()[:-1] == action.size()[:-1], print(state.size(), action.size())
        sa = torch.cat([state, action], -1)
        if sa.ndim == 3:
            sa = sa.unsqueeze(0)
        q = self.q(sa) # (batch_size, 1) or (num_q, batch_size, 1)
        q = torch.squeeze(q, -1) if q.size(-1) == 1 else q # q:(num_q, batch_size)

        for i in range(q.size(0)):
            self.infos['q%s' % (i + 1)] = q[i]

        if minimize:
            q = q.min(dim=0)[0] if q.size(0) == self.num_q else q # q:(batch_size,)
            self.infos['q_min'] = q
        return q

    def forward_q(self, state, action, minimize=True):
        assert state.size()[:-1] == action.size()[:-1], print(state.size(), action.size())
        sa = torch.cat([state, action], -1)
        if sa.ndim == 3:
            sa = sa.unsqueeze(0)
        q = self.q(sa) # (batch_size, 1) or (num_q, batch_size, 1)
        q = torch.squeeze(q, -1) if q.size(-1) == 1 else q # q:(num_q, batch_size)

        for i in range(q.size(0)):
            self.infos['q%s' % (i + 1)] = q[i]

        if minimize:
            q = q.min(dim=0)[0] if q.size(0) == self.num_q else q # q:(batch_size,)
            self.infos['q_min'] = q
        return q

    def log(self, L, step, log_freq, params=False):
        if not log_freq or step % log_freq != 0:
            return

        for k, v in self.infos.items():
            L.log_histogram('train_critic/%s_hist' % k, v, step)

        if params:
            if self.repr_dim is not None:
                L.log_param('train_critic/fc', self.trunk[0], step)
                L.log_param('train_critic/ln', self.trunk[1], step)
            for i in range(self.hidden_depth+1):
                L.log_param('train_critic/q_ensemble_fc%d' % i, self.q[i * 2], step)


class EnsembleValue(nn.Module):

    def __init__(self, hidden_dim, repr_dim, encoder_feature_dim, l=2, output_mod=None,
                 num_q=2, output_dim=1, handle_dim=None, activation=nn.ReLU):
        super(EnsembleValue, self).__init__()

        self.state_dim = encoder_feature_dim
        self.output_dim = output_dim
        self.num_q = num_q
        self.hidden_depth = l
        self.repr_dim = repr_dim
        
        if repr_dim is not None:
            self.trunk = mlp(repr_dim, 0, self.state_dim, 0, nn.LayerNorm, handle_dim=self.state_dim)

        self.output_dim = output_dim
        self.v = mlp(self.state_dim, hidden_dim, output_dim, l, output_mod,
                     handle_dim=handle_dim, channel_dim=num_q, ensemble=True,
                     activation=activation)
        self.infos = dict()
        self.apply(weight_init)

    def forward_trunk(self, obs, tanh=True):
        if self.repr_dim is None:
            return obs
        state = self.trunk(obs)
        self.infos['ln'] = state
        if tanh:
            state = torch.tanh(state)
            self.infos['tanh'] = state
        return state

    def output_v(self, v, mode):
        if v.size(0) == 1 or mode == False:
            return v.squeeze(0)
        elif mode == 'random':
            idx = np.random.choice(v.size(0))
            v = v[idx]
        elif mode == 'min':
            v = v.min(0)[0]
        else:
            raise ValueError(mode)
        return v

    def forward(self, state, mode=False, tanh=True):
        state = self.forward_trunk(state, tanh)

        if state.ndim == 3:
            state = state.unsqueeze(0)
        v = self.v(state) # (batch_size, 1) or (num_q, batch_size, 1)
        v = torch.squeeze(v, -1) if v.size(-1) == 1 else v # (num_q, batch_size)

        for i in range(v.size(0)):
            self.infos['v%s' % (i + 1)] = v[i]
        v = self.output_v(v, mode)

        self.infos['v'] = v # (batch_size,)
        return v

    def forward_v(self, state, mode=True):
        if state.ndim == 3:
            state = state.unsqueeze(0)
        v = self.v(state) # (batch_size, 1) or (num_q, batch_size, 1)
        v = torch.squeeze(v, -1) if v.size(-1) == 1 else v # (num_q, batch_size)

        for i in range(v.size(0)):
            self.infos['v%s' % (i + 1)] = v[i]
        v = self.output_v(v, mode)

        self.infos['v'] = v # (batch_size,)
        return v

    def log(self, L, step, log_freq, param=False):
        if not log_freq or step % log_freq != 0:
            return

        for k, v in self.infos.items():
            L.log_histogram('train_value/%s_hist' % k, v, step)

        if param:
            for i in range(self.hidden_depth+1):
                L.log_param('train_value/ensemble_v_fc%d' % i, self.v[i * 2], step)


class CURL(nn.Module):

    def __init__(self, extr, critic, extr_targ, critic_targ, feature_dim):
        super(CURL, self).__init__()
        self.extr = extr
        self.trunk = None
        if critic.repr_dim is not None:
            self.trunk = critic.trunk

        self.extr_targ = extr_targ
        self.trunk_targ = None
        if critic_targ is not None and critic_targ.repr_dim is not None:
            self.trunk_targ = critic_targ.trunk
        self.W = nn.Parameter(torch.rand(feature_dim, feature_dim))

    def output(self, extr, trunk, x, tanh=True):
        y = extr(x)
        if trunk is not None:
            y = trunk(y)
            if tanh is True:
                y = torch.tanh(y)
        return y

    def encode(self, x, detach=False, ema=False):
        if ema:
            z_out = self.output(
                self.extr_targ, self.trunk_targ, x).detach()
        else:
            z_out = self.output(self.extr, self.trunk, x)
        if detach:
            z_out = z_out.detach()
        return z_out

    def compute_logits(self, z_anc, z_pos):
        Wz = torch.matmul(self.W, z_pos.T)
        logits = torch.matmul(z_anc, Wz)
        logits = logits - torch.max(logits, 1)[0][:, None]
        return logits


LOG_SIG_MIN = -6
LOG_STD_MAX = 2
class CFPredictor(nn.Module):
    def __init__(self, latent_state_dim, act_seq_in_dim, rew_seq_in_dim, hidden_dim,
                 act_seq_out_dim=5, omg_seq_out_dim=5, output_dim=1, l=2, output_mod=None,
                 ensemble=False, num_ensemble=1, activation=Swish, rs_fc=False,
                 omega_opt_mode=None, num_sample=256):
        super(CFPredictor, self).__init__()

        # Initialize hyperparameters
        self.output_dim = 2 * output_dim
        self.num_ensemble = num_ensemble
        self.hidden_depth = l
        self.num_sample = num_sample
        self.omega_opt_mode = omega_opt_mode

        # Initialize modules
        self.aseq_fc = mlp(act_seq_in_dim, 0, act_seq_out_dim, 0, output_mod)
        self.omeg_fc = mlp(rew_seq_in_dim, 0, omg_seq_out_dim, 0, output_mod)
        self.rseq_fc = mlp(rew_seq_in_dim, 0, rew_seq_in_dim, 0, output_mod) if rs_fc \
            else None
        
        self.predictor = mlp(
            latent_state_dim + act_seq_out_dim + omg_seq_out_dim,
            hidden_dim, self.output_dim, l, output_mod, ensemble=ensemble,
            channel_dim=num_ensemble, activation=activation
        )

        # Initialize a random variable
        self._init_omega(omega_opt_mode, rew_seq_in_dim)

        self.infos = dict()
        self.apply(weight_init)

    def _init_omega(self, omega_opt_mode, rs_dim):
        assert omega_opt_mode in [None, 'min_mu', 'min_all'], print(omega_opt_mode)
        self.omega_mu = nn.Parameter(torch.zeros(rs_dim, requires_grad=True))
        self.omega_logstd = nn.Parameter(
            torch.ones(rs_dim, requires_grad=True) * math.atanh(-LOG_SIG_MIN/(LOG_STD_MAX - LOG_SIG_MIN))
        )
        if omega_opt_mode is None:
            self.omega_mu.requires_grad = False
            self.omega_logstd.requires_grad = False
        elif omega_opt_mode == 'min_mu':
            self.omega_logstd.requires_grad = False

    @property
    def omega(self):
        # output.shape == torch.Size([num_sample, rew_seq_dim])
        if self.omega_opt_mode != 'sample':
            log_std = torch.tanh(self.omega_logstd)
            log_std = LOG_SIG_MIN + log_std * (LOG_STD_MAX - LOG_SIG_MIN)
            std = torch.exp(log_std)
            noise = torch.randn(self.num_sample, log_std.size(0)).to(log_std.device)

            self.infos['omega_mu'] = self.omega_mu.detach()
            self.infos['omega_std'] = std.detach()
            return self.omega_mu + noise * std
        return self.omega_mu

    def _get_input(self, h, h_as, h_ws):
        if h_as.ndim == 3:
            h = h.unsqueeze(0).expand(h_as.size(0), *h.size())
        cat = torch.cat([h, h_as], dim=-1)
        if h_ws.ndim == 4:
            cat = cat.unsqueeze(1).expand(*h_ws.size()[:3], cat.size(-1))
        elif h_ws.ndim == 3:
            cat = cat.unsqueeze(0).expand(h_ws.size(0), *cat.size())
        return torch.cat([cat, h_ws], dim=-1)

    def forward_as(self, action_sequence: torch.tensor):
        # action_sequence: torch.Size([batch_size, seq_dim, act_dim])
        # case1: output.shape == torch.Size([batch_size, act_seq_out_dim])
        # case2: output.shape == torch.Size([num_ensemble, batch_size, act_seq_out_dim])
        assert action_sequence.ndim == 3
        h_as = self.aseq_fc(action_sequence.view(action_sequence.size(0), -1))
        self.infos['h_as'] = h_as[0] if h_as.ndim == 3 else h_as
        return h_as

    def forward_ws(self, omega_sequence: torch.tensor, batch_size: int):
        # omega_sequence: torch.Size([num_sample, rew_seq_dim])
        # case1: output.shape == torch.Size([num_sample, batch_size, rew_seq_dim])
        # case2: output.shape == torch.Size([num_ensemble, num_sample, batch_size, rew_seq_dim])
        assert omega_sequence.ndim == 2
        omega_sequence = omega_sequence.unsqueeze(1).expand(
            omega_sequence.size(0), batch_size, omega_sequence.size(-1)
        ).unsqueeze(0)
        h_ws = self.omeg_fc(omega_sequence)
        self.infos['h_ws'] = h_ws[0].mean(0)
        return h_ws.squeeze(0)

    def forward_rs(self, reward_sequence: torch.tensor):
        # reward_sequence: torch.Size([batch_size, rew_seq_dim])
        # output: torch.Size([batch_size, rew_seq_dim])
        assert reward_sequence.ndim == 2
        h_rs = self.rseq_fc(reward_sequence) if self.rseq_fc is not None else reward_sequence
        self.infos['h_rs'] = h_rs
        return h_rs

    def forward(self, latent_state, action_sequence, omega_sequence):
        # case1: output.shape == torch.Size([num_sample, batch_size, output_dim])
        # case2: output.shape == torch.Size([num_ensemble, num_sample, batch_size, output_dim])
        batch_size = latent_state.size(0)
        latent_a_seq = self.forward_as(action_sequence)
        latent_w_seq = self.forward_ws(omega_sequence, batch_size)
        input = self._get_input(latent_state, latent_a_seq, latent_w_seq)
        return self.predictor(input)

    def log(self, L, step, log_freq, params=True):
        if not log_freq or step % log_freq != 0:
            return

        for k, v in self.infos.items():
            L.log_histogram('train_cresp/%s_hist' % k, v, step)

        if params:
            L.log_param('train_cresp/aseq_fc', self.aseq_fc[0], step)
            L.log_param('train_cresp/omeg_fc', self.omeg_fc[0], step)
            L.log_param('train_cresp/rseq_fc', self.rseq_fc[0], step) if self.rseq_fc is not None else None
            for i in range(self.hidden_depth+1):
                L.log_param('train_cresp/pred_fc%d' % i, self.predictor[i * 2], step)
