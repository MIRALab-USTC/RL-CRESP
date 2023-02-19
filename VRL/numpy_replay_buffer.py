# This code is mainly excerpted from the code:
# https://github.com/conglu1997/v-d4rl/blob/main/drqbc/numpy_replay_buffer.py
import numpy as np
import abc
import torch
from torch.utils.data import Dataset, DataLoader


class AbstractReplayBuffer(abc.ABC):
    @abc.abstractmethod
    def add(self, time_step):
        pass

    @abc.abstractmethod
    def __next__(self, ):
        pass

    @abc.abstractmethod
    def __len__(self, ):
        pass


class EfficientReplayBuffer(AbstractReplayBuffer):
    '''Fast + efficient replay buffer implementation in numpy.'''

    def __init__(self, buffer_size, batch_size, nstep, discount, frame_stack, rsd_nstep=1,
                 replay_buffer_num_workers=8, data_specs=None, sarsa=False, rew_noise=None):
        self.buffer_size = buffer_size
        self.data_dict = {}
        self.index = -1
        self.traj_index = 0
        self.frame_stack = frame_stack
        self._recorded_frames = frame_stack + 1
        self.batch_size = batch_size
        self.nstep = nstep
        self.rsd = False
        self.only_rsd = True
        self.osd = False
        self.rsd_nstep = rsd_nstep
        self.num_workers = replay_buffer_num_workers
        self.discount = discount
        self.full = False
        self.discount_vec = np.power(discount, np.arange(nstep))  # n_step - first dim should broadcast
        self.next_dis = discount ** nstep
        self.sarsa = sarsa
        self.rew_noise = rew_noise # dict('mu'=0, 'std'=?) or None

    def _initial_setup(self, time_step):
        self.index = 0
        self.obs_shape = list(time_step.observation.shape)
        self.ims_channels = self.obs_shape[0] // self.frame_stack
        self.act_shape = time_step.action.shape

        self.obs = np.zeros([self.buffer_size, self.ims_channels, *self.obs_shape[1:]], dtype=np.uint8)
        self.act = np.zeros([self.buffer_size, *self.act_shape], dtype=np.float32)
        self.rew = np.zeros([self.buffer_size], dtype=np.float32)
        self.dis = np.zeros([self.buffer_size], dtype=np.float32)
        self.valid = np.zeros([self.buffer_size], dtype=np.bool_)
        self.rsd_valid = np.zeros([self.buffer_size], dtype=np.bool_)

    @property
    def multi_step(self):
        return self.nstep if self.nstep >= self.rsd_nstep else self.rsd_nstep

    def add_data_point(self, time_step):
        first = time_step.first()
        latest_obs = time_step.observation[-self.ims_channels:]
        if first:
            # Save the first obs and repeat frame_stack times
            end_index = self.index + self.frame_stack
            end_invalid = end_index + self.frame_stack + 1
            if end_invalid > self.buffer_size:
                if end_index > self.buffer_size:
                    end_index = end_index % self.buffer_size
                    self.obs[self.index:self.buffer_size] = latest_obs
                    self.obs[0:end_index] = latest_obs
                    self.full = True
                else:
                    self.obs[self.index:end_index] = latest_obs
                end_invalid = end_invalid % self.buffer_size
                self.valid[self.index:self.buffer_size] = False
                self.valid[0:end_invalid] = False
                self.rsd_valid[self.index:self.buffer_size] = False
                self.rsd_valid[0:end_invalid] = False
            else:
                self.obs[self.index:end_index] = latest_obs
                self.valid[self.index:end_invalid] = False
                self.rsd_valid[self.index:end_invalid] = False
            self.index = end_index
            self.traj_index = 1
        else:
            np.copyto(self.obs[self.index], latest_obs)  # Check most recent image
            np.copyto(self.act[self.index], time_step.action)
            self.rew[self.index] = time_step.reward
            self.dis[self.index] = time_step.discount
            self.valid[(self.index + self.frame_stack) % self.buffer_size] = False
            self.rsd_valid[(self.index + self.frame_stack) % self.buffer_size] = False
            if self.traj_index >= self.nstep:
                self.valid[(self.index - self.nstep + 1) % self.buffer_size] = True
            if self.traj_index >= self.rsd_nstep:
                self.rsd_valid[(self.index - self.rsd_nstep + 1) % self.buffer_size] = True
            self.index += 1
            self.traj_index += 1
            if self.index == self.buffer_size:
                self.index = 0
                self.full = True

    def add(self, time_step):
        if self.index == -1:
            self._initial_setup(time_step)
        self.add_data_point(time_step)

    def __next__(self):
        if self.rsd:
            indices = np.random.choice(self.rsd_valid.nonzero()[0], size=self.batch_size)
            return self.gather_rsd_nstep_indices(indices, self.only_rsd)
        elif self.osd:
            indices = np.random.choice(self.rsd_valid.nonzero()[0], size=self.batch_size)
            return self.gather_osd_nstep_indices(indices)
        indices = np.random.choice(self.valid.nonzero()[0], size=self.batch_size)
        return self.gather_nstep_indices(indices)

    def gather_nstep_indices(self, indices):
        B = indices.shape[0]
        all_gather_ranges = np.stack([np.arange(indices[i] - self.frame_stack, indices[i] + self.nstep)
                                      for i in range(B)], axis=0) % self.buffer_size
        gather_ranges = all_gather_ranges[:, self.frame_stack:]  # (B, MS)
        obs_gather_ranges = all_gather_ranges[:, :self.frame_stack]  # (B, FS)
        nobs_gather_ranges = all_gather_ranges[:, -self.frame_stack:]  # (B, FS)

        all_rewards = self.rew[gather_ranges]
        if self.rew_noise is not None:
            noise = np.random.randn(*all_rewards.shape) * self.rew_noise['std'] + self.rew_noise['mu']
            all_rewards += noise
            all_rewards = all_rewards.clip(0.0, 1.0)

        # Could implement below operation as a matmul in pytorch for marginal additional speed improvement
        rew = np.sum(all_rewards * self.discount_vec, axis=1, keepdims=True)

        obs = np.reshape(self.obs[obs_gather_ranges], [B, *self.obs_shape])
        nobs = np.reshape(self.obs[nobs_gather_ranges], [B, *self.obs_shape])

        act = self.act[indices]
        dis = np.expand_dims(self.next_dis * self.dis[nobs_gather_ranges[:, -1]], axis=-1)

        if self.sarsa:
            nact = self.act[indices + self.nstep]
            return (obs, act, rew, dis, nobs, nact)

        return (obs, act, rew, dis, nobs)

    def gather_rsd_nstep_indices(self, indices, only_rsd=True):
        B = indices.shape[0]
        all_gather_ranges = np.stack([np.arange(indices[i] - self.frame_stack, indices[i] + self.multi_step)
                                      for i in range(B)], axis=0) % self.buffer_size
        gather_ranges = all_gather_ranges[:, self.frame_stack:]  # (B, MS)
        obs_gather_ranges = all_gather_ranges[:, :self.frame_stack]
        nobs_gather_ranges = all_gather_ranges[:, self.nstep:self.frame_stack+self.nstep]

        obs = np.reshape(self.obs[obs_gather_ranges], [B, *self.obs_shape])
        all_acts = self.act[gather_ranges]  # (B, MS, A)
        all_rewards = self.rew[gather_ranges]# * np.power(self.discount, np.arange(self.multi_step))  # (B, MS)
        if only_rsd:
            return (obs, all_acts, all_rewards)

        act = self.act[indices] # all_acts[:, 0]
        # Could implement below operation as a matmul in pytorch for marginal additional speed improvement
        rew = np.sum(all_rewards[:, :self.nstep] * self.discount_vec, axis=1, keepdims=True)
        dis = np.expand_dims(self.next_dis * self.dis[nobs_gather_ranges[:, -1]], axis=-1)
        nobs = np.reshape(self.obs[nobs_gather_ranges], [B, *self.obs_shape])

        return (obs, act, rew, dis, nobs, all_acts, all_rewards)

    def gather_osd_nstep_indices(self, indices):
        B = indices.shape[0]
        all_gather_ranges = np.stack([np.arange(indices[i] - self.frame_stack, indices[i] + self.multi_step)
                                      for i in range(B)], axis=0) % self.buffer_size
        gather_ranges = all_gather_ranges[:, self.frame_stack:]  # (B, MS)
        obs_gather_ranges = all_gather_ranges[:, :self.frame_stack]  # (B, FS)
        nobs_gather_ranges = all_gather_ranges[:, self.nstep:self.frame_stack+self.nstep]  # (B, FS)

        obs = np.reshape(self.obs[obs_gather_ranges], [B, *self.obs_shape])
        all_acts = self.act[gather_ranges]  # (B, MS, A)
        all_rewards = self.rew[gather_ranges]# * np.power(self.discount, np.arange(self.multi_step))  # (B, MS)
        # act = self.act[indices] # all_acts[:, 0]

        obses_gather_ranges = [all_gather_ranges[:, i+1:self.frame_stack+i+1] for i in range(self.multi_step)]
        obses_gather_ranges = np.stack(obses_gather_ranges, -1).reshape(B, -1)  # (B, MS * FS)
        all_nobs = np.reshape(self.obs[obses_gather_ranges], [B*self.multi_step, *self.obs_shape])  # (B*MS, *Image_Shape)
        return (obs, all_acts, all_rewards, all_nobs)

    def __len__(self):
        if self.full:
            return self.buffer_size
        else:
            return self.index

    def get_train_and_val_indices(self, useful_data, validation_percentage, max_num):
        all_indices = useful_data.nonzero()[0]
        num_indices = all_indices.shape[0]
        num_val = min(int(num_indices * validation_percentage), max_num)
        np.random.shuffle(all_indices)
        val_indices, train_indices = np.split(all_indices,
                                              [num_val])
        return train_indices, val_indices

    def get_obs_act_batch(self, indices):
        B = indices.shape[0]
        obs_gather_ranges = np.stack([np.arange(indices[i] - self.frame_stack, indices[i])
                                      for i in range(B)], axis=0) % self.buffer_size
        obs = np.reshape(self.obs[obs_gather_ranges], [B, *self.obs_shape])
        act = self.act[indices]
        return obs, act

    def generate_dataloader(self, rsd=False, val_ratio=None, max_num=1000):
        valid = False if val_ratio is None else True
        val_ratio = 0 if val_ratio is None else val_ratio

        if rsd:
            train_indices, val_indices = self.get_train_and_val_indices(self.rsd_valid, val_ratio, max_num)

            train_buffer = self.gather_rsd_nstep_indices(train_indices, True)
            val_buffer = self.gather_rsd_nstep_indices(val_indices, True) if valid else None

            train_data = RSDBuffer(train_buffer, self.batch_size)
            val_data = RSDBuffer(val_buffer, self.batch_size) if valid else None
        else:
            train_indices, val_indices = self.get_train_and_val_indices(self.valid, val_ratio, max_num)

            train_buffer = self.gather_nstep_indices(train_indices)
            val_buffer = self.gather_nstep_indices(val_indices) if valid else None

            train_data = RSDBuffer(train_buffer, self.batch_size)
            val_data = RSDBuffer(val_buffer, self.batch_size) if valid else None

        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True, 
                                  num_workers=self.num_workers, pin_memory=True)
        val_loader = DataLoader(val_data, batch_size=self.batch_size, shuffle=True, 
                                num_workers=self.num_workers, pin_memory=True) if valid else None

        return train_loader, val_loader


class DataBuffer(Dataset):
    """
    Save the Data with Normalization to Train Models
    """
    def __init__(self, buffer, batch_size):
        super().__init__()
        self.buffer = buffer
        self.batch_size = batch_size

    def __len__(self):
        return len(self.buffer[0])

    def __getitem__(self, idx):
        obs, act, rew, dis, nobs = self.buffer
        return obs[idx], act[idx], rew[idx], dis[idx], nobs[idx]


class RSDBuffer(Dataset):
    """
    Save the Data with Normalization to Train Models
    """
    def __init__(self, buffer, batch_size):
        super().__init__()
        self.buffer = buffer
        self.batch_size = batch_size

    def __len__(self):
        return len(self.buffer[0])

    def __getitem__(self, idx):
        obses, all_acts, all_rewards = self.buffer
        return obses[idx], all_acts[idx], all_rewards[idx]
