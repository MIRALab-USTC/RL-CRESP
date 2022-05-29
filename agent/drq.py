import torch
from .agent_base import AGENTBase
from common import utils
from module.init_module import init_algo, init_auxiliary_task


class DrQAgent(AGENTBase):

    def __init__(self, aug_func, extr_lr, extr_beta, config, **kwargs):
        super().__init__(device=config['device'], **config['agent_base_params'])
        # Initialize hyperparameters
        obs_shape = config['obs_shape']
        self.batch_size = config['batch_size']

        # Initialize modules
        self._init_extractor(obs_shape, config['extr_params'])
        repr_dim = self.extr.repr_dim
        self.rl = init_algo(config['base'],
                            (False, repr_dim, repr_dim),
                            config['algo_params'])
        self.aux_task = init_auxiliary_task(config['aux_task'],
                                            config['algo_params']['action_shape'],
                                            config['auxiliary_params'],
                                            self.device)
        # Initialize augmentation
        self.aug_func = aug_func.to(self.device)

        # Initialize optimizers
        self.extr_q_optimizer = torch.optim.Adam(self.extr.parameters(),
                                                 lr=extr_lr, betas=(extr_beta, 0.999))
        self.train()
        self.train_targ()
        self.print_module()

    def update_critic(self, aug_o, aug_a, aug_r, aug_o2, aug_nd, gamma):
        extr_targ = self.extr_targ if self.extr_targ is not None else self.extr

        for i in range(self.update_to_data):
            self.update_critic_steps += 1

            aug_s = self.extr(aug_o)
            aug_s2 = extr_targ(aug_o2).detach()

            loss_q, qf_opt_dict, q_info_dict = self.rl.update_critic(
                aug_s, aug_a, aug_r, aug_s2, aug_nd, gamma, self.num_aug)

            if self.extr_update_via_qfloss and self.total_time_steps % self.extr_update_freq_via_qfloss == 0:
                qf_opt_dict['opt_e'] = self.extr_q_optimizer
                self.update_extr_steps += 1

            utils.update_params(qf_opt_dict, loss_q)
        return q_info_dict

    def update_actor(self, aug_o, step):
        pi_info_dict = None
        if step % self.actor_update_freq == 0:
            self.update_actor_steps += 1

            aug_s = self.extr(aug_o).detach()
            loss_pi, pi_opt_dict, pi_info_dict = self.rl.update_actor(
                aug_s, self.actor_update_mode, self.num_aug)
            utils.update_params(pi_opt_dict, loss_pi)
        return pi_info_dict

    def _update(self, data, logger, step, save_log):
        """Augment a mini-batch data"""
        batch_size, aug_o, aug_a, aug_r, aug_o2, aug_nd, gamma, envl = self.augment(data, self.num_aug)

        """Update critic"""
        q_info_dict = self.update_critic(aug_o, aug_a, aug_r, aug_o2, aug_nd, gamma)

        """Update actor"""
        pi_info_dict = self.update_actor(aug_o, step)

        """Save logger"""
        if save_log:
            self.extr.log(logger['tb'], step, True)
            self.rl.critic.log(logger['tb'], step, True)
            utils.log(logger, 'train_critic', q_info_dict, step)
            if pi_info_dict is not None:
                self.rl.actor.log(logger['tb'], step, True)
                utils.log(logger, 'train_actor', pi_info_dict, step)

        '''Smooth update'''
        if step % self.critic_target_update_freq == 0:
            self.rl.soft_update_params()
            if self.extr_targ is not None:
                utils.soft_update_params(self.extr, self.extr_targ, self.extr_tau)


    def _save(self, model_dir, step):
        pass

    def _load(self, model_dir, step):
        pass

    def _print_log(self, logger):
        pass
