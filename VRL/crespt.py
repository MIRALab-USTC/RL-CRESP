import hydra
import time
import math
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path

import utils
from drq import Actor, Critic, ECritic, RandomShiftsAug, NoShiftAug


def tie_weights(src, trg):
    assert type(src) == type(trg)
    trg.weight = src.weight
    trg.bias = src.bias


class Encoder(nn.Module):
    def __init__(self, obs_shape, output_dim):
        super().__init__()

        assert len(obs_shape) == 3

        self.repr_dim = 32 * 35 * 35
        self.num_conv_layers = 4

        self.convnet = nn.Sequential(nn.Conv2d(obs_shape[0], 32, 3, stride=2), nn.ReLU(),
                                     nn.Conv2d(32, 32, 3, stride=1), nn.ReLU(),
                                     nn.Conv2d(32, 32, 3, stride=1), nn.ReLU(),
                                     nn.Conv2d(32, 32, 3, stride=1), nn.ReLU(), nn.Flatten())
        self.embed_state = nn.Sequential(nn.Linear(self.repr_dim, output_dim), nn.LayerNorm(output_dim), nn.Tanh())
        self.infos = dict()
        self.apply(utils.weight_init)

    def forward_conv(self, conv):
        for i in range(self.num_conv_layers):
            conv = self.convnet[2*i+1](self.convnet[2*i](conv))
            self.infos['conv%s' % (i + 1)] = conv
        return self.convnet[-1](conv)

    def forward(self, obs, embed=False):
        obs = obs / 255.0
        h = self.forward_conv(obs)
        if embed:
            return self.embed_state(h)
        return h

    def copy_linear_weights_from(self, source):
        """Tie linear layers"""
        tie_weights(src=source.trunk[0], trg=self.embed_state[0])

    def log(self, metrics):
        for k, v in self.infos.items():
            metrics["%s_img" % k] = v
        return metrics


class Swish(nn.Module):

    def __init__(self, inplace=False):
        super(Swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x.mul_(torch.sigmoid(x)) if self.inplace else x * torch.sigmoid(x)


class GELU(nn.Module):
    def forward(self, input):
        return F.gelu(input)

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, num_head, feature_dim, block_size, attn_pdrop, resid_pdrop):
        super().__init__()
        assert feature_dim % num_head == 0
        self.num_head = num_head
        # key, query, value projections for all heads
        self.key = nn.Linear(feature_dim, feature_dim)
        self.query = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(feature_dim, feature_dim)

    def forward(self, x, layer_past=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.num_head, C // self.num_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.num_head, C // self.num_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.num_head, C // self.num_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, num_head, feature_dim, block_size, attn_pdrop, resid_pdrop):
        super().__init__()
        self.ln1 = nn.LayerNorm(feature_dim)
        self.ln2 = nn.LayerNorm(feature_dim)
        self.attn = CausalSelfAttention(
            num_head, feature_dim, block_size, attn_pdrop, resid_pdrop)
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, 4 * feature_dim),
            GELU(),
            nn.Linear(4 * feature_dim, feature_dim),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


LOG_SIG_MIN = -6
LOG_STD_MAX = 2


class Transformer(nn.Module):
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1
    num_layer=2
    num_head=2

    def __init__(self, obs_shape, feature_dim, act_dim, rsd_nstep, omega_num_sample, hidden_dim, output_dim, device):
        super().__init__()
        self.act_dim  = act_dim
        self.rsd_nstep = rsd_nstep
        self.output_dim = output_dim
        self.block_size = rsd_nstep + 1
        self.feature_dim = feature_dim

        self.encoder = Encoder(obs_shape, feature_dim)

        # input embedding stem
        self.timestep = torch.arange(omega_num_sample).to(device)
        self.embed_timestep = nn.Embedding(omega_num_sample, feature_dim)
        self.drop = nn.Dropout(self.embd_pdrop)

        self.embed_seq = nn.Sequential(nn.Linear((self.act_dim+1)*self.rsd_nstep, feature_dim), nn.Tanh())

        self.blocks = nn.Sequential(*[Block(self.num_head, feature_dim, self.block_size, self.attn_pdrop, self.resid_pdrop) for _ in range(self.num_layer-1)])
        self.block_sin = nn.Sequential(*[Block(self.num_head, feature_dim, self.block_size, self.attn_pdrop, self.resid_pdrop) for _ in range(1)])
        self.block_cos = nn.Sequential(*[Block(self.num_head, feature_dim, self.block_size, self.attn_pdrop, self.resid_pdrop) for _ in range(1)])

        # decoder head
        self.ln_f = nn.LayerNorm(feature_dim)
        self.pred_sin_cf = nn.Linear(feature_dim, output_dim)
        self.pred_cos_cf = nn.Linear(feature_dim, output_dim)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _init_omega(self, omega_output_mode, reward_dim):
        assert omega_output_mode in [None, 'min_mu', 'min_all'], print(omega_output_mode)
        self.omega_mu = nn.Parameter(torch.zeros(reward_dim, requires_grad=True))
        self.omega_logstd = nn.Parameter(torch.ones(reward_dim, requires_grad=True)
                                         * math.atanh(-LOG_SIG_MIN/(LOG_STD_MAX - LOG_SIG_MIN)))
        if omega_output_mode is None:
            self.omega_mu.requires_grad = False
            self.omega_logstd.requires_grad = False
        elif omega_output_mode == 'min_mu':
            self.omega_logstd.requires_grad = False
        else:
            return 

    # constraint guassian distribution
    def omega(self, omega_num_sample):
        # output.shape == [N, T]
        log_std = torch.tanh(self.omega_logstd)
        log_std = LOG_SIG_MIN + log_std * (LOG_STD_MAX - LOG_SIG_MIN)
        std = torch.exp(log_std)
        noise = torch.randn(omega_num_sample, log_std.size(0)).to(log_std.device)
        idx = noise.abs().mean(-1).sort()[1]
        omega_seq = self.omega_mu + noise[idx] * std # [N, T]
        # omega_seq = (torch.tanh(omega_seq) + 1.0) * 0.5
        return omega_seq # [N, T]

    def configure_optimizers(self, learning_rate, weight_decay):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('omega_mu')
        no_decay.add('omega_logstd')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95))
        return optimizer

    def forward(self, z: torch.Tensor, a_seq: torch.Tensor, w_seq: torch.Tensor):
        # z     --> [B, F]
        # a_seq --> [B, T, A]
        # w_seq --> [N, T])
        assert a_seq.ndim == 3 and w_seq.ndim == 2, print(a_seq.ndim, w_seq.ndim)
        B = z.size(0)
        N, T = w_seq.size()
        H = self.feature_dim
        a_seq = a_seq.reshape(B, -1)  # a_seq --> [B, T*A]

        z = z.unsqueeze(1).expand(B, N, H)   # z --> [B, N, H]
        a_seq = a_seq.unsqueeze(0).expand(N, *a_seq.size()).transpose(0, 1) # a_seq --> [B, N, T*A]
        w_seq = w_seq.unsqueeze(0).expand(B, N, T)   # w_seq --> [B, N, T]

        seq_cat = torch.cat([a_seq, w_seq], dim=-1) # seq_cat --> [B, N, T*(A+1)]
        seq_cat = self.embed_seq(seq_cat) + self.embed_timestep(self.timestep)  # seq_cat --> [B, N, H]
        seq_cat = torch.cat([z.unsqueeze(-2), seq_cat.unsqueeze(-2)], dim=-2) # seq_cat --> [B, N, 2, H]

        x = seq_cat.reshape([-1, 2, H]) # seq_cat --> [B*N, 2, H]
        x = self.drop(x)
        x = self.blocks(x)
        x1 = self.ln_f(self.block_cos(x))
        x2 = self.ln_f(self.block_sin(x))
        pred_cos = self.pred_cos_cf(x1)[:, 0].reshape(B, N, -1)
        pred_sin = self.pred_sin_cf(x2)[:, 0].reshape(B, N, -1)  # [B, N, output_dim]

        return pred_cos, pred_sin


class CFPredictor(nn.Module):
    def __init__(self, obs_shape, feature_dim, action_shape, reward_dim, output_dim,
                 elite_output_dim, hidden_dim, trunk=None, output_mode='min',
                 omega_num_sample=256, omega_output_mode=None, device='cpu'):
        super().__init__()

        # Setting modules
        self.predictor = Transformer(obs_shape, feature_dim, action_shape[0], reward_dim,
                                     omega_num_sample, hidden_dim, output_dim, device)

        # Setting hyperparameters
        self.output_dim = output_dim
        self.elite_output_dim = elite_output_dim
        self.output_mode = output_mode
        self.omega_num_sample = omega_num_sample
        self.omega_output_mode = omega_output_mode

        # Initialize a random variable
        self._init_omega(omega_output_mode, reward_dim)

        self.apply(utils.weight_init)

    def _init_omega(self, omega_output_mode, reward_dim):
        self.predictor._init_omega(omega_output_mode, reward_dim)

    def configure_optimizers(self, learning_rate, weight_decay):
        return self.predictor.configure_optimizers(learning_rate, weight_decay)

    @property
    def omega(self):
        # output.shape == [N, T]
        return self.predictor.omega(self.omega_num_sample)

    def elite_rank(self, data, loss):
        if torch.is_tensor(loss):
            loss = loss.cpu().numpy()
        elite_rank = np.argsort(loss)[:self.elite_output_dim]
        return data[elite_rank]

    def forward(self, latent_state, action_sequence, omega_sequence):
        # latent_state:     torch.Size([B, H])
        # action_sequence:  torch.Size([B, T, A])
        # omega_sequence:   torch.Size([N, T])
        # psi_cos/sin:      torch.Size([B, N, output_dim])
        # psi:              torch.Size([output_dim, B, N, 2])
        psi_cos, psi_sin = self.predictor(
            latent_state, action_sequence, omega_sequence)
        psi = torch.stack([psi_cos, psi_sin], dim=0).transpose(0, -1)
        return psi

    def save_snapshot(self, work_dir, best=False):
        work_dir = Path(work_dir)
        snapshot = work_dir / 'snapshot.pt' if not best else work_dir / 'snapshot_best.pt'
        payload = {'predictor': self.predictor}
        with snapshot.open('wb') as f:
            torch.save(payload, f)

    def load_snapshot(self, work_dir, best=False):
        work_dir = Path(work_dir)
        snapshot = work_dir / 'snapshot.pt' if not best else work_dir / 'snapshot_best.pt'
        with snapshot.open('rb') as f:
            payload = torch.load(f)
        self.predictor = payload['predictor']


class Agent:
    def __init__(self, obs_shape, action_shape, device, lr, feature_dim, hidden_dim,
                 critic_target_tau, num_expl_steps, update_every_steps, init_temperature,
                 use_tb, use_cf, actor_update_times=1, critic_update_times=1,
                 critic_target_update_freq=2, augmentation=RandomShiftsAug(pad=4), # CRESP-T
                 rsd_nstep=5, rsd_discount=0.99, cf_temp=0.1, cf_output_dim=7, cf_elite_num=5,
                 cf_output_mode='min', cf_weight=1.0, cf_update_freq=10, omega_num_sample=256,
                 omega_cf_output_mode='min_mu'):
        self.use_cf = use_cf
        self.device = device
        self.actor_update_times = actor_update_times
        self.critic_update_times = critic_update_times
        self.critic_target_tau = critic_target_tau
        self.critic_target_update_freq = critic_target_update_freq
        self.update_every_steps = update_every_steps
        self.use_tb = use_tb
        self.num_expl_steps = num_expl_steps
        self.init_temperature = init_temperature
        # Initialize hyperparameters of CREST
        self.rsd_nstep = rsd_nstep
        self.rsd_discount = rsd_discount
        self.cf_temp = cf_temp
        self.cf_elite_num = cf_elite_num
        self.cf_output_dim = cf_output_dim
        self.cf_output_mode = cf_output_mode
        self.cf_weight = cf_weight
        self.cf_update_freq = cf_update_freq
        # record
        self.update_critic_steps = 0
        self.update_actor_steps = 0
        self.update_encoder_steps = 0
        self.update_cf_steps = 0

        # models
        self.cfpred = CFPredictor(obs_shape, feature_dim, action_shape, rsd_nstep, cf_output_dim,
                                  cf_elite_num, hidden_dim, None, cf_output_mode, 
                                  omega_num_sample, omega_cf_output_mode, device).to(device)

        self.encoder = self.cfpred.predictor.encoder
        self.actor = Actor(self.encoder.repr_dim, action_shape, feature_dim, hidden_dim).to(device)
        self.encoder.copy_linear_weights_from(self.actor)

        # self.critic = ECritic(self.encoder.repr_dim, action_shape, feature_dim, hidden_dim).to(device)
        # self.critic_target = ECritic(self.encoder.repr_dim, action_shape, feature_dim, hidden_dim).to(device)
        self.critic = Critic(self.encoder.repr_dim, action_shape, feature_dim, hidden_dim).to(device)
        self.critic_target = Critic(self.encoder.repr_dim, action_shape, feature_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        self.target_entropy = -np.prod(action_shape)

        # optimizers
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=lr, betas=(0.5, 0.999))
        self.cfpred_opt = self.cfpred.configure_optimizers(learning_rate=lr, weight_decay=0.0)

        # data augmentation
        self.aug = augmentation

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)
        self.cfpred.train(training)

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
        print("| Update Actor Times: %d | Update Critic Times: %d | Update Encoder Times: %d | Update CF Times: %d" % (
            self.update_actor_steps, self.update_critic_steps, self.update_encoder_steps, self.update_cf_steps
        ))

    def calc_psi(self, r_seq, w_seq):
        # r_seq:    [B, T]
        # w_seq:    [N, T]
        # outputs:  [B, N, 2]
        inner_product = r_seq @ w_seq.t()
        psi_targ_cos = np.pi / 2 * torch.cos(inner_product)
        psi_targ_sin = np.pi / 2 * torch.sin(inner_product)
        return torch.stack([psi_targ_cos, psi_targ_sin], dim=-1)

    def update_extr(self, obses, action_seqs, reward_seqs, step, cfp_opt=None):
        metrics = dict()

        self.update_cf_steps += 1
        reward_seqs = reward_seqs * (self.rsd_discount ** torch.arange(reward_seqs.size(1)).to(self.device))

        obs1 = self.encoder(self.aug(obses.float()), True)
        obs2 = self.encoder(self.aug(obses.float()), True)

        omega_seqs = self.cfpred.omega
        psi_targ = self.calc_psi(reward_seqs, omega_seqs)

        psi1 = self.cfpred(obs1, action_seqs, omega_seqs) # (K, B, N, 2)
        psi2 = self.cfpred(obs2, action_seqs, omega_seqs) # (K, B, N, 2)
        # psi1 = torch.sin(psi1)
        # psi2 = torch.sin(psi2)
        # psi1[psi1 > 0] = psi1[psi1 > 0] % 1
        # psi1[psi1 < 0] = psi1[psi1 < 0] % -1
        # psi2[psi2 > 0] = psi2[psi2 > 0] % 1
        # psi2[psi2 < 0] = psi2[psi2 < 0] % -1

        # MSE
        psi1_error = (psi1 - psi_targ.unsqueeze(0)).pow(2)
        psi2_error = (psi2 - psi_targ.unsqueeze(0)).pow(2)
        psi1_error = self.cfpred.elite_rank(psi1_error, psi1_error.mean([1,2,3]).detach())
        psi2_error = self.cfpred.elite_rank(psi2_error, psi2_error.mean([1,2,3]).detach())
        loss_psi_mse = (psi1_error + psi2_error).sum(0).sum(-1).mean()
        loss_psi_std = psi1_error.std(2).mean() + psi2_error.std(2).mean()

        # Cross Entropy
        batch_size, *_ = psi_targ.size()
        psi_ce_targ = psi_targ.reshape(batch_size, -1) # (B, N*2)
        psi1_ce = psi1.reshape(psi1.size(0), batch_size, -1) # (K, B, N*2)
        psi2_ce = psi2.reshape(psi2.size(0), batch_size, -1) # (K, B, N*2)

        loss_psi1_ce, acc1 = utils.compute_ce_loss(psi1_ce, psi_ce_targ, self.cf_temp)
        loss_psi2_ce, acc2 = utils.compute_ce_loss(psi2_ce, psi_ce_targ, self.cf_temp)
        loss_psi_ce = self.cfpred.elite_rank(loss_psi1_ce, loss_psi1_ce.detach()) \
                        + self.cfpred.elite_rank(loss_psi2_ce, loss_psi2_ce.detach())
        # Total Loss
        # loss_psi = loss_psi_mse + loss_psi_ce.sum()
        # loss_psi = loss_psi_ce.sum()
        loss_psi = loss_psi_mse + 0.01 * loss_psi_std + loss_psi_ce.sum()

        if self.use_tb:
            metrics['extr_loss'] = loss_psi.item() / 2
            metrics['extr_mse_loss'] = loss_psi_mse.item() / 2
            metrics['extr_std_loss'] = loss_psi_std.item() / 2
            metrics['extr_ce_loss'] = loss_psi_ce.sum().item() / 2
            metrics['extr_ce_acc'] = (acc1 + acc2).mean().item() / 2
            if step % 100 == 0:
                pe = psi1_error[0, :, 0].reshape(psi1_error.size(1), -1)#[:20]
                plt.figure()
                [plt.plot(np.arange(p.size(0)), p.detach().cpu().numpy()) for p in pe]
                metrics['psi_error_fig'] = plt.gcf()

        # optimize encoder and cfpred
        if cfp_opt is not None:
            utils.update_params(cfp_opt, loss_psi * self.cf_weight)
        return metrics

    def calc_value(self, obs, is_target=True):
        _, action, log_prob, _ = self.actor(obs)
        critic = self.critic if not is_target else self.critic_target
        # Q = critic(obs, action)
        # return Q.min(0)[0] - self.alpha * log_prob.unsqueeze(-1)
        Q1, Q2 = critic(obs, action)
        return torch.min(Q1, Q2) - self.alpha * log_prob.unsqueeze(-1)


    def update_critic(self, obses, action, reward, discount, next_obses, step):
        metrics = dict()

        if step >= self.critic_update_times:
            return metrics

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

        # Q1 = self.critic(obs1, action)
        # Q2 = self.critic(obs2, action)

        # critic_loss = utils.inv_huber_loss(Q1, target_Q) + utils.inv_huber_loss(Q2, target_Q)

        if self.use_tb:
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q1'] = Q1.mean().item()
            metrics['critic_q2'] = Q2.mean().item()
            metrics['critic_loss'] = critic_loss.item() / 2
            # Qmin, Qmax = Q1.min(0)[0], Q2.max(0)[0]
            Qmin, Qmax = torch.min(Q1, Q2), torch.max(Q1, Q2)
            plt.clf()
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

        if step >= self.actor_update_times:
            return metrics
        self.update_actor_steps += 1

        _, action, log_prob, _ = self.actor(obs)

        # optimize alpha
        alpha_loss = (self.alpha * (-log_prob - self.target_entropy).detach()).mean()
        utils.update_params(self.alpha_opt, alpha_loss)

        Q1, Q2 = self.critic(obs, action)
        Q = torch.min(Q1, Q2)
        # Q = self.critic(obs, action).min(0)[0]

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

    def update_actor_critic(self, obses, action, reward, discount, next_obs, step):
        metrics = dict()

        # augment
        obs1, obs2 = self.aug(obses.float()), self.aug(obses.float())
        next_obs1, next_obs2 = self.aug(next_obs.float()), self.aug(next_obs.float())
        # encode
        obs1, obs2 = self.encoder(obs1), self.encoder(obs2)
        with torch.no_grad():
            next_obs1, next_obs2 = self.encoder(next_obs1), self.encoder(next_obs2)

        if self.use_tb:
            metrics['batch_reward'] = reward.mean().item()

        # update critic
        metrics.update(self.update_critic((obs1, obs2), action, reward, discount,
                                          (next_obs1, next_obs2), step))

        # update actor
        metrics.update(self.update_actor(obs1.detach(), step))

        return metrics

    def crest(self, replay_buffer, step):
        metrics = dict()
        # update encoder & cfpred
        if step % self.cf_update_freq == 0:
            replay_buffer.rsd = True
            batch = next(replay_buffer)
            replay_buffer.rsd = False
            obs, act_seqs, rew_seqs = utils.to_torch(batch, self.device)
            metrics = self.update_extr(obs, act_seqs, rew_seqs, step, self.cfpred_opt)
        if step % 10000 == 0:
            print("Omega: ", self.cfpred.predictor.omega_mu.detach().cpu())
        return metrics

    def update(self, replay_buffer, step):
        batch = next(replay_buffer)
        obs, action, reward, discount, next_obs = utils.to_torch(batch, self.device)

        for i in range(self.update_times):
            metrics = self.update_actor_critic(obs, action, reward, discount, next_obs, i)

            if self.use_cf:
                metrics.update(self.crest(replay_buffer, step+i))

        # update critic target
        if step % self.critic_target_update_freq == 0:
            utils.soft_update_params(self.critic, self.critic_target, self.critic_target_tau)

        metrics = self.encoder.log(metrics)
        return metrics
