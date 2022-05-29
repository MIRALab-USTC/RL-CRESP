import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np


class ReplayBuffer(Dataset):
    """Buffer to store environment transitions."""
    def __init__(self, obs_shape, state_shape, action_shape, capacity, buffer_dir, batch_size, nstep,
                 gamma, lambd, device, save_buffer, buf_dtype, image_size=84, pre_image_size=84, **kwargs):
        obs_shape = (9, 84, 84)
        self.capacity = capacity
        self.buffer_dir = buffer_dir
        self.batch_size = batch_size
        self.nstep = nstep
        self.gamma = gamma
        self.device = device
        self.save_buffer = save_buffer
        self.image_size = image_size
        self.pre_image_size = pre_image_size
        # the proprioceptive obs is stored as float32, pixels obs as uint8
        obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8
        
        self.obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.states = np.empty((capacity, *state_shape), dtype=np.float32)
        self.next_states = np.empty((capacity, *state_shape), dtype=np.float32)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.logp_a = np.empty((capacity, 1), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)
        self.discount = np.empty((capacity, 1), dtype=np.float32)
        self.env_labels = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.idx_epoch = 0
        self.idx_val_epoch = 0
        self.last_save = 0
        self.full = False

    @property
    def size(self):
        return self.capacity if self.full else self.idx

    def add_obs(self, obs, episode):
        pass

    def add(self, obs, action, reward, next_obs, done, d, ep_num, ep_len,
            env_label=0, logp_a=None, infos=None, state=None, next_state=None):

        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)
        np.copyto(self.env_labels[self.idx], env_label)
        np.copyto(self.discount[self.idx], infos['discount'])
        if logp_a is not None:
            np.copyto(self.logp_a[self.idx], logp_a)
        if state is not None:
            np.copyto(self.states[self.idx], state)
            np.copyto(self.next_states[self.idx], next_state)

        if d and self.save_buffer:
            self.save(self.buffer_dir, ep_num, ep_len)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def get_rew(self, idxs):
        gamma = np.ones_like(self.discount[idxs])
        if self.nstep == 1:
            return torch.as_tensor(self.rewards[idxs], device=self.device), self.gamma
        rewards = np.zeros((idxs.shape[0], 1))
        for i in range(self.nstep):
            step_rewards = self.rewards[(idxs + i) % self.capacity]
            rewards += gamma * step_rewards
            gamma *= self.discount[(idxs + i) % self.capacity] * self.gamma
        return torch.as_tensor(rewards, device=self.device), torch.as_tensor(gamma, device=self.device)

    def sample_idxs(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        if self.nstep == 1:
            return idxs, None
        return idxs, (idxs + self.nstep - 1) % self.capacity

    def sample(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        idxs, next_idxs = self.sample_idxs(batch_size)
        
        obses = self.obses[idxs]
        next_obses = self.next_obses[idxs] if self.nstep == 1 else self.next_obses[next_idxs]

        obses = torch.as_tensor(obses, device=self.device).float()
        states = torch.as_tensor(self.states[idxs], device=self.device)
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards, gamma = self.get_rew(idxs)
        next_obses = torch.as_tensor(next_obses, device=self.device).float()
        next_states = torch.as_tensor(self.next_states[idxs], device=self.device)
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        env_labels = torch.as_tensor(self.env_labels[idxs], device=self.device)

        transition = dict(obs=obses, obs2=next_obses, act=actions, rew=rewards,
                          not_done=not_dones, gamma=gamma, env_label=env_labels,
                          state=states, next_state=next_states)
        return transition

    def sample_obs(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        idxs, next_idxs = self.sample_idxs(batch_size)
        
        obses = self.obses[idxs]
        obses = torch.as_tensor(obses, device=self.device).float()
        return obses

    def shrank(self, k):
        self.add(self.obses[-1], self.actions[-1], self.rewards[-1], self.next_obses[-1],
                 not self.not_dones[-1], False, 1, 1, self.env_labels[-1], None, {'discount': 0.99},
                 self.states[-1], self.next_states[-1])
        print("The capacity of the replay buffer now is {}".format(self.size))

        self.obses = self.obses[:self.idx]
        self.states = self.states[:self.idx]
        self.env_labels = self.env_labels[:self.idx]
        self.shuffle_idx = np.arange(self.idx)

        np.random.shuffle(self.shuffle_idx)
        self.obses = self.obses[self.shuffle_idx]
        self.states = self.states[self.shuffle_idx]
        self.env_labels = self.env_labels[self.shuffle_idx]

        self.obses_val = self.obses[self.idx-k:self.idx]
        self.states_val = self.states[self.idx-k:self.idx]
        self.env_labels_val = self.env_labels[self.idx-k:self.idx]

        self.obses = self.obses[:self.idx-k]
        self.states = self.states[:self.idx-k]
        self.env_labels = self.env_labels[:self.idx-k]
        self.shuffle_idx = np.arange(self.idx-k)
        self.idx -= k
        self.idx_val = k

    def sample_ablate(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        if self.idx_epoch == 0:
            np.random.shuffle(self.shuffle_idx)
            self.obses = self.obses[self.shuffle_idx]
            self.states = self.states[self.shuffle_idx]
            self.env_labels = self.env_labels[self.shuffle_idx]

        obses = self.obses[self.idx_epoch:self.idx_epoch+batch_size]
        obses = torch.as_tensor(obses, device=self.device).float()
        states = torch.as_tensor(self.states[self.idx_epoch:self.idx_epoch+batch_size], device=self.device)
        env_labels = torch.as_tensor(self.env_labels[self.idx_epoch:self.idx_epoch+batch_size], device=self.device)
        self.idx_epoch = (self.idx_epoch + batch_size) % self.idx
        return obses, states, env_labels, self.idx_epoch

    def sample_val_ablate(self):
        obses = self.obses_val[self.idx_val_epoch:self.idx_val_epoch+500]
        obses = torch.as_tensor(obses, device=self.device).float()
        states = torch.as_tensor(self.states_val[self.idx_val_epoch:self.idx_val_epoch+500], device=self.device)
        env_labels = torch.as_tensor(self.env_labels_val[self.idx_val_epoch:self.idx_val_epoch+500], device=self.device)
        self.idx_val_epoch = (self.idx_val_epoch + 500) % self.idx_val
        return obses, states, env_labels, self.idx_val_epoch

    def sample_np(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        idxs, next_idxs = self.sample_idxs(batch_size)
        
        obses = self.obses[idxs]
        next_obses = self.next_obses[idxs] if self.nstep == 1 else self.next_obses[next_idxs]
        actions = self.actions[idxs]
        not_dones = self.not_dones[idxs]

        gamma = np.ones_like(self.discount[idxs])
        if self.nstep == 1:
            rewards = self.rewards[idxs]
        else:
            rewards = np.zeros((idxs.shape[0], 1))
            for i in range(self.nstep):
                step_rewards = self.rewards[(idxs + i) % self.capacity]
                rewards += gamma * step_rewards
                gamma *= self.discount[(idxs + i) % self.capacity] * self.gamma

        transition = dict(obs=obses, obs2=next_obses, act=actions, rew=rewards,
                          not_done=not_dones, gamma=gamma)
        return transition

    def save(self, save_dir, ep_num, ep_len):
        if self.idx == self.last_save:
            return
        path = save_dir / ('%d_%d_%d_%d.pt' % (ep_num, ep_len, self.last_save, self.idx))
        payload = [
            self.obses[self.last_save:self.idx],
            self.next_obses[self.last_save:self.idx],
            self.states[self.last_save:self.idx],
            self.next_states[self.last_save:self.idx],
            self.actions[self.last_save:self.idx],
            self.rewards[self.last_save:self.idx],
            self.not_dones[self.last_save:self.idx],
            self.env_labels[self.last_save:self.idx]
        ]
        self.last_save = self.idx
        torch.save(payload, path)

    def load(self, save_dir):
        chunks = os.listdir(save_dir)
        chucks = sorted(chunks, key=lambda x: int(x.split('_')[0]))
        for chunk in chucks:
            ep_num, ep_len, start, end = [int(x) for x in chunk.split('.')[0].split('_')]
            path = os.path.join(save_dir, chunk)
            payload = torch.load(path)
            assert self.idx == start
            self.obses[start:end] = payload[0]
            self.next_obses[start:end] = payload[1]
            self.states[start:end] = payload[2]
            self.next_states[start:end] = payload[3]
            self.actions[start:end] = payload[4]
            self.rewards[start:end] = payload[5]
            self.not_dones[start:end] = payload[6]
            self.env_labels[start:end] = payload[7]
            self.idx = end

    def __getitem__(self, idx):
        idx = np.random.randint(0, self.size, size=1)
        idx = idx[0]
        obs = self.obses[idx]
        state = self.states[idx]
        action = self.actions[idx]
        reward = self.rewards[idx]
        next_obs = self.next_obses[idx]
        next_state = self.next_states[idx]
        not_done = self.not_dones[idx]
        env_label = self.env_labels[idx]

        if self.transform:
            obs = self.transform(obs)
            next_obs = self.transform(next_obs)

        return obs, state, action, reward, next_obs, next_state, not_done, env_label

    def __len__(self):
        return self.capacity
