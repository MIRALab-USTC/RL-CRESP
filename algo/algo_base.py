import abc
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ALGOBase(object, metaclass=abc.ABCMeta):
    
    def __init__(self, action_limit, num_q, critic_tau, extr_latent_dim):
        # Setting hyperparameters
        self.action_limit = action_limit
        self.num_q = num_q
        self.critic_tau = critic_tau
        self.extr_latent_dim = extr_latent_dim

        # Setting modules
        self.actor = None
        self.actor_targ = None
        self.critic = None
        self.critic_targ = None

        # Setting optimizers
        self.actor_optimizer = None
        self.critic_optimizer = None

    def train(self, training=True):
        self.actor.train(training)
        self.critic.train(training)

    def train_targ(self, training):
        self.actor_targ.train(training) if self.actor_targ is not None else None
        self.critic_targ.train(training) if self.critic_targ is not None else None

    def select_action(self, s, deterministic=False, tanh=False, to_numpy=True):
        pi = self.actor.act(s, deterministic, tanh)
        return pi.cpu().data.numpy().flatten() if to_numpy else pi.squeeze(0).detach()

    def select_q_pi(self, q_pi, mode):
        if mode == 'min':
            q_pi = torch.min(q_pi, dim=0)[0]
        elif mode == 'mean':
            q_pi = q_pi.mean(0)
        elif mode == 'sample':
            idx = np.random.choice(q_pi.size(0), 2, replace=False)
            q_pi = q_pi[idx].mean(0)
        elif mode == 'sum':
            q_pi = q_pi.sum(0)
        elif mode == 'single':
            q_pi = q_pi[0]
        return q_pi # q_pi: (batch_size,)

    @abc.abstractmethod
    def update_critic(self, s, a, r, s2, nd):
        pass

    @abc.abstractmethod
    def update_actor(self, s):
        pass

    def save(self, model_dir, step):
        torch.save(
            self.actor.state_dict(), '%s/actor_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.critic.state_dict(), '%s/critic_%s.pt' % (model_dir, step)
        )
        self._save(model_dir, step)

    @abc.abstractmethod
    def _save(self, model_dir, step):
        pass

    def load(self, model_dir, step):
        self.actor.load_state_dict(
            torch.load('%s/actor_%s.pt' % (model_dir, step))#, map_location=torch.device('cpu'))
        )
        self.actor_targ = copy.deepcopy(self.actor) if self.actor_targ else None
        self.critic.load_state_dict(
            torch.load('%s/critic_%s.pt' % (model_dir, step))#, map_location=torch.device('cpu'))
        )
        self.critic_targ = copy.deepcopy(self.critic) if self.critic_targ else None
        self._load(model_dir, step)

    @abc.abstractmethod
    def _load(self, model_dir, step):
        pass

    @abc.abstractmethod
    def _print_log(self, logger):
        pass