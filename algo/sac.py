import copy
import torch
import torch.nn.functional as F
import numpy as np

from .algo_base import ALGOBase
from module.rl_module import SGMLPActor, Critic, EnsembleCritic
from common.utils import update_params, soft_update_params

_AVAILABLE_CRITIC = {'normal': Critic, 'ensemble': EnsembleCritic}


class SAC(ALGOBase):

    def __init__(self, action_shape, action_limit, critic_lr, critic_beta,
                 critic_type, actor_lr, actor_beta, alpha_lr, alpha_beta,
                 device='cpu', critic_tau=0.05, num_q=2, hidden_dim=1024,
                 extr_latent_dim=50, repr_dict=dict(), init_temperature=0.1,
                 l=2, actor_log_std_min=-10, actor_log_std_max=2, **kwargs):
        super().__init__(action_limit, num_q, critic_tau, extr_latent_dim)
        # Setting hyperparameters
        extr_has_fc, actor_repr_dim, critic_repr_dim = repr_dict
        # self.is_fc = not extr_has_fc
        actor_repr_dim = None if extr_has_fc else actor_repr_dim
        critic_repr_dim = None if extr_has_fc else critic_repr_dim

        # Setting modules
        self.actor = SGMLPActor(
            action_shape, hidden_dim, actor_repr_dim, extr_latent_dim,
            actor_log_std_min, actor_log_std_max, l, self.action_limit
        ).to(device)
        self.critic = _AVAILABLE_CRITIC[critic_type](
            action_shape, hidden_dim, critic_repr_dim, extr_latent_dim,
            l, num_q=num_q
        ).to(device)
        self.critic_targ = copy.deepcopy(self.critic)

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -np.prod(action_shape)

        # Setting optimizers
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr, betas=(critic_beta, 0.999))
        self.actor_optimizer  = torch.optim.Adam(self.actor.parameters(),
                                                 lr=actor_lr, betas=(actor_beta, 0.999))
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                     lr=alpha_lr, betas=(alpha_beta, 0.999))

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def get_q_target(self, s2, r, nd, gamma):
        _, a2, logp_a2, _ = self.actor(s2)
        q_pi_targ = self.critic_targ(s2, a2, False) # (num_q, batch_size)
        q_targ = r + gamma * nd * (q_pi_targ.min(dim=0)[0] - self.alpha * logp_a2)
        q_targ_max = r + gamma * nd * (q_pi_targ.max(dim=0)[0] - self.alpha * logp_a2)
        return q_targ, q_targ_max # (batch_size,)

    def calculate_q(self, s, a, num_aug, batch_size):
        _s = self.critic.forward_trunk(s)
        _s = _s.view(num_aug, batch_size, -1) if _s.ndim == 2 else _s # (num_aug, batch_size, enc_dim)
        if a.ndim == 2:
            _a = a.view(num_aug, batch_size, -1) if a.size(0) == (num_aug*batch_size) \
                else a.unsqueeze(0).expand((num_aug, *a.size()))  # (num_aug, batch_size, act_dim)
        return self.critic.forward_q(_s, _a, False)  # (num_aug, num_q, batch_size)

    def update_critic(self, s, a, r, s2, nd, gamma, num_aug=None):
        with torch.no_grad():
            q_targ, q_targ_max = self.get_q_target(s2, r, nd, gamma)
            if num_aug is not None and isinstance(num_aug, int):
                q_targ = q_targ.view(num_aug, -1).mean(0)
                q_targ_max = q_targ_max.view(num_aug, -1).mean(0)

        q = self.calculate_q(s, a, num_aug, s.size(0)//num_aug) # (num_aug, num_q, batch_size)
        # loss_q = F.mse_loss(
        #     q, q_targ.view(1, 1, -1).expand(*q.size()), reduction='none')
        loss_q = (q - q_targ.view(1, 1, -1)).pow(2).mean(-1)
        q_info_dict = dict(Qvals=q.min(dim=1)[0].mean([0, 1]),
                           Qmaxs=q.max(dim=1)[0].mean([0, -1]),
                           TQvals=q_targ, TQmaxs=q_targ_max,
                           LossQ=loss_q.mean())
        return loss_q.sum([0, 1]).mean(), dict(opt_q=self.critic_optimizer), q_info_dict

    def update_actor(self, s, mode='min', num_aug=1):
        _, a, logp_a, _ = self.actor(s)

        loss_alpha = (self.alpha * (-logp_a - self.target_entropy).detach()).mean()
        update_params(self.log_alpha_optimizer, loss_alpha)

        q_pi = self.calculate_q(s, a, num_aug, s.size(0)//num_aug) # (num_aug, num_q, batch_size)
        q_pi = torch.min(q_pi, dim=1)[0] - self.alpha.detach() * logp_a.view(num_aug, -1) # (num_aug, batch_size)
        loss_pi = -self.select_q_pi(q_pi, mode)

        pi_info_dict = dict(HPi=-logp_a.view(num_aug, -1).mean(0),
                            LossPi=loss_pi, Alpha=self.alpha.item(),
                            LossAlpha=loss_alpha)
        return loss_pi.mean(), dict(opt_pi=self.actor_optimizer), pi_info_dict

    def soft_update_params(self):
        soft_update_params(self.critic, self.critic_targ, self.critic_tau)

    def _save(self, model_dir, step):
        pass

    def _load(self, model_dir, step):
        pass

    def _print_log(self, logger):
        logger.log_tabular('HPi', average_only=True)
        logger.log_tabular('Alpha', average_only=True)
        logger.log_tabular('LossAlpha', average_only=True)
