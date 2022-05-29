import abc
import time
import copy
import torch

from module.init_module import init_extractor
from common.utils import _handle_data, update_params, log


class AGENTBase(object, metaclass=abc.ABCMeta):
    
    def __init__(
        self,
        action_repeat,
        actor_update_mode,
        actor_update_freq,
        critic_target_update_freq,
        update_to_data,
        extr_update_via_qfloss,
        extr_update_freq_via_qfloss,
        num_sources,
        num,
        device
    ):
        # Setting hyperparameters
        self.action_repeat = action_repeat
        self.actor_update_mode = actor_update_mode
        self.actor_update_freq = actor_update_freq
        self.critic_target_update_freq = critic_target_update_freq
        self.update_to_data = update_to_data
        self.extr_update_via_qfloss = extr_update_via_qfloss
        self.extr_update_freq_via_qfloss = extr_update_freq_via_qfloss
        self.num_sources = num_sources
        self.num_aug = num
        self.device = device
        self.debug_info = {}
        self.training = False

        self.total_time_steps = 0
        self.update_steps = 0
        self.update_critic_steps = 0
        self.update_actor_steps = 0
        self.update_extr_steps = 0
        self.update_extr_steps_via_aux = 0

        # Setting modules
        self.aug_func = None
        self.aux_task = None
        self.extr = None
        self.extr_targ = None
        self.rl = None

        # Setting optimizers
        self.extr_q_optimizer = None

    def train(self, training=True):
        self.training = training
        self.extr.train(self.training)
        self._train()
        self.rl.train(self.training)
        self.aux_task.train(self.training) if self.aux_task is not None else None

    def _train(self):
        pass

    def train_targ(self):
        self.extr_targ.train(self.training) if self.extr_targ is not None else None
        self._train_targ()
        self.rl.train_targ(self.training)

    def _train_targ(self):
        pass

    @property
    def update_extr_total_steps(self):
        if self.update_extr_steps_via_aux == 0:
            return self.update_extr_steps
        return "Qf-%d Aux-%d" % (self.update_extr_steps, self.update_extr_steps_via_aux)

    def print_module(self):
        print("Augment:", self.aug_func)
        print("AuxiliaryTask:", self.aux_task) if self.aux_task is None else self.aux_task.print_module()
        print("Extractor:", self.extr)
        print("Target Extractor:", self.extr_targ)
        print("Critic:", self.rl.critic)
        print("Actor:", self.rl.actor)
        self._print_module()

    def _print_module(self):
        pass

    def _init_extractor(self, obs_shape, extr_config):
        module_dict = init_extractor(obs_shape, self.device, extr_config)
        self.extr = module_dict['extr']
        self.extr_targ = module_dict['extr_targ']
        self.extr_tau = module_dict['extr_tau']

    def augment(self, data, num_aug=2, name=''):
        o, a, r, o2, nd, el = data['obs'], data['act'], data['rew'], data['obs2'], data['not_done'], data['env_labels']
        infos = data['infos']
        r = r.squeeze(-1) if r.size(-1) == 1 else r
        nd = nd.squeeze(-1) if nd.size(-1) == 1 else nd
        self.debug_info['obs%s' % name] = o # (batch_size, *o.size())
        self.debug_info['next_obs%s' % name] = o2 # (batch_size, *o2.size())
        # Augment data
        batch_size = o.size(0)
        if num_aug > 1:
            o, a, r, o2, nd = o.repeat(num_aug, 1, 1, 1), a.repeat(num_aug, 1), r.repeat(num_aug), o2.repeat(num_aug, 1, 1, 1), nd.repeat(num_aug)
        aug_o, aug_o2 = self.aug_func(o), self.aug_func(o2)
        aug_o, aug_o2 = _handle_data(aug_o), _handle_data(aug_o2)
        data['aug_o'] = aug_o
        return batch_size, aug_o, a, r, aug_o2, nd, infos['gamma'], el

    @torch.no_grad()
    def select_action(self, obs, deterministic=False, tanh=True, to_numpy=True):
        if not torch.is_tensor(obs):
            obs = torch.FloatTensor(obs).to(self.device)
        if obs.ndim == 3:
            obs = obs.unsqueeze(0)
        s = self.extr(_handle_data(obs))
        return self.rl.select_action(s, deterministic, tanh, to_numpy)

    @torch.no_grad()
    def estimate_q_val(self, obs, act, minimize=True):
        obs = torch.FloatTensor(obs).to(self.device).unsqueeze(0)
        state = self.extr(_handle_data(obs))
        if act.ndim == 1:
            act = act.unsqueeze(0)
        q_vals = self.rl.critic(state, act, minimize)
        return q_vals

    def update(self, replay_buffer, logger, step, save_log, batch_size=None):
        self.update_steps += 1
        '''Sample an augmented mini-batch data from replay buffer'''
        data1 = replay_buffer.sample_batch(batch_size) if self.aux_task is None \
            else replay_buffer.sample_batch_with_rs(batch_size)
        data2 = replay_buffer.sample_batch(batch_size) if self.aux_task is None \
            else replay_buffer.sample_batch_with_rs(batch_size)
        self._update(data1, logger, step, save_log)
        self._update(data2, logger, step, save_log)
        """Update extr by the auxiliary task"""
        self.auxiliary_update(data1, logger, step, save_log)
        self.auxiliary_update(data2, logger, step, save_log)

    def auxiliary_update(self, data, logger, step, save_log):
        if self.aux_task is not None:
            self.update_extr_steps_via_aux += 1

            aug_s = self.rl.actor.forward_trunk(self.extr(data['aug_o']))
            loss, opt_dict, info_dict = self.aux_task.update_extr(
                data, aug_s, self.num_aug)
            opt_dict['opt_e'] = self.extr_q_optimizer
            opt_dict['opt_pi'] = self.rl.actor_optimizer
            update_params(opt_dict, loss)

            if save_log:
                self.aux_task.network.log(logger['tb'], step, True)
                log(logger, 'train_auxiliary', info_dict, step)

    @abc.abstractmethod
    def _update(self, data, logger, step, save_log):
        pass

    def save(self, model_dir, step):
        torch.save(
            self.extr.state_dict(), '%s/extr_%s.pt' % (model_dir, step)
        )
        self.rl.save(model_dir, step)
        self.aux_task.save(model_dir, step) if self.aux_task is not None else None
        self._save(model_dir, step)

    @abc.abstractmethod
    def _save(self, model_dir, step):
        raise NotImplementedError

    def load(self, model_dir, step):
        self.extr.load_state_dict(
            torch.load('%s/extr_%s.pt' % (model_dir, step))#, map_location=torch.device('cpu')
        )
        self.extr_targ = copy.deepcopy(self.extr) if self.extr_targ else None
        self.rl.load(model_dir, step)
        self.aux_task.load(model_dir, step) if self.aux_task is not None else None
        self._load(model_dir, step)

    @abc.abstractmethod
    def _load(self, model_dir, step):
        raise NotImplementedError

    def print_log(self, logger, test_env, epoch, step, ar, test, start_time, epoch_fps):
        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('TotalEnvInteracts', step * ar)
        logger.log_tabular('Step', step)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpNum', average_only=True)
        logger.log_tabular('EpLen', average_only=True)
        if test:
            if isinstance(test_env, list):
                for i in range(len(test_env)):
                    logger.log_tabular('TestEpRet%s' % i, with_min_and_max=True)
            else:
                logger.log_tabular('TestEpRet0', with_min_and_max=True)
        logger.log_tabular('Qvals', average_only=True)
        logger.log_tabular('Qmaxs', average_only=True)
        logger.log_tabular('TQvals', average_only=True)
        logger.log_tabular('LossQ', average_only=True)
        logger.log_tabular('LossPi', average_only=True)

        self.rl._print_log(logger)
        self.aux_task._print_log(logger) if self.aux_task is not None else None
        self._print_log(logger)

        logger.log_tabular('Time', (time.time() - start_time)/3600)
        logger.log_tabular('FPS', epoch_fps)
        logger.dump_tabular()

    @abc.abstractmethod
    def _print_log(logger):
        raise NotImplementedError
