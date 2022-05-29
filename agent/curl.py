import torch
from .agent_base import AGENTBase
from common import utils
from module.init_module import init_algo, init_auxiliary_task
from module.rl_module import CURL


class CurlAgent(AGENTBase):

    def __init__(self, aug_func, extr_lr, extr_beta, config, **kwargs):
        super().__init__(device=config['device'], **config['agent_base_params'])
        # Setting hyperparameters
        obs_shape = config['obs_shape']
        self.batch_size = config['batch_size']
        # Setting modules
        self._init_extractor(obs_shape, config['extr_params'])
        repr_dim = self.extr.repr_dim
        self.rl = init_algo(config['base'],
                            (False, repr_dim, repr_dim),
                            config['algo_params'])

        self.curl = CURL(self.extr,
                         self.rl.critic,
                         self.extr_targ,
                         self.rl.critic_targ,
                         config['extr_params']['extr_latent_dim']).to(self.device)

        self.aux_task = init_auxiliary_task(config['aux_task'],
                                            config['algo_params']['action_shape'],
                                            config['auxiliary_params'],
                                            self.device)
        # Setting augmentation
        self.aug_func = aug_func.to(self.device)

        # Setting optimizers
        self.extr_q_optimizer = torch.optim.Adam(self.extr.parameters(),
                                                 lr=extr_lr, betas=(extr_beta, 0.999))

        self.cpc_optimizer = torch.optim.Adam(self.curl.parameters(),
                                              lr=extr_lr, betas=(extr_beta, 0.999))
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
        
        self.train()
        self.train_targ()
        self.print_module()


    def _print_module(self):
        print("CURL:", self.curl)

    def update_critic(self, aug_o, aug_a, aug_r, aug_o2, aug_nd, gamma):
        extr_targ = self.extr_targ if self.extr_targ is not None else self.extr

        for i in range(self.update_to_data):
            self.update_critic_steps += 1

            aug_s = self.extr(aug_o)
            aug_s2 = extr_targ(aug_o2).detach()

            loss_q, qf_opt_dict, q_info_dict = self.rl.update_critic(
                aug_s, aug_a, aug_r, aug_s2, aug_nd, gamma, 1)

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
            loss_pi, pi_opt_dict, pi_info_dict = self.rl.update_actor(aug_s)
            utils.update_params(pi_opt_dict, loss_pi)
        return pi_info_dict

    def update_curl(self, aug_o, num_aug):
        o_anc, o_pos = aug_o#.view(num_aug, -1, *aug_o.size()[1:])
        z_a = self.curl.encode(o_anc)
        z_pos = self.curl.encode(o_pos, ema=True)

        logits = self.curl.compute_logits(z_a, z_pos)
        labels = torch.arange(logits.shape[0]).long().to(self.device)

        loss_cl = self.cross_entropy_loss(logits, labels)
        opt_dict = dict(opt1=self.cpc_optimizer,
                        opt2=self.extr_q_optimizer)
        utils.update_params(opt_dict, loss_cl)
        return dict(LossCPC=loss_cl.mean().item())

    def _update(self, data, logger, step, save_log):
        """Augment a mini-batch data"""
        batch_size, aug_o, aug_a, aug_r, aug_o2, aug_nd, gamma, envl = self.augment(data, self.num_aug)
        aug_o = aug_o.view(self.num_aug, -1, *aug_o.size()[1:])
        aug_o2 = aug_o2.view(self.num_aug, -1, *aug_o2.size()[1:])
        aug_a = aug_a.view(self.num_aug, -1, aug_a.size(-1))
        aug_r = aug_r.view(self.num_aug, -1)
        aug_nd = aug_nd.view(self.num_aug, -1)

        """Update critic"""
        q_info_dict = self.update_critic(aug_o[0], aug_a[0], aug_r[0], aug_o2[0], aug_nd[0], gamma)

        """Update actor"""
        pi_info_dict = self.update_actor(aug_o[0], step)

        """Update CURL"""
        cl_info_dict = self.update_curl(aug_o, self.num_aug)

        """Save logger"""
        if save_log:
            self.extr.log(logger['tb'], step, True)
            self.rl.critic.log(logger['tb'], step, True)
            utils.log(logger, 'train_critic', q_info_dict, step)
            if pi_info_dict is not None:
                self.rl.actor.log(logger['tb'], step, True)
                utils.log(logger, 'train_actor', pi_info_dict, step)
            utils.log(logger, 'train_cpc', cl_info_dict, step)

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
        logger.log_tabular('LossCPC', average_only=True)
