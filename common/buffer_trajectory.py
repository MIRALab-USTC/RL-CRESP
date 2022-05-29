import torch
from torch.utils.data import Dataset
import os
import numpy as np


class BReplayBuffer(Dataset):
    """Buffer to store environment transitions."""
    def __init__(self, obs_shape, action_shape, capacity, buffer_dir, batch_size, nstep,
                 nstep_of_rsd, gamma, device, save_buffer, image_size=84, **kwargs):
        self.capacity = capacity
        self.buffer_dir = buffer_dir
        self.batch_size = batch_size
        self.nstep = 1#nstep
        self.nstep_rsd = nstep_of_rsd
        self.gamma = gamma
        self.device = device
        self.save_buffer = save_buffer
        self.image_size = image_size
        # the proprioceptive obs is stored as float32, pixels obs as uint8
        obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8

        self.obses = np.zeros((capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.zeros((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.not_dones = np.zeros((capacity, 1), dtype=np.float32)
        self.env_labels = np.zeros((capacity, 1), dtype=np.float32)
        self.obs_idxes = np.ones(capacity, dtype=np.int) * -1

        # save the start point and end point of an episode
        self.ep_start_idxs = np.zeros((capacity,), dtype=np.int)
        self.ep_end_idxs = np.zeros((capacity,), dtype=np.int)
        self.ep_end_idxs_rew = np.zeros((capacity,), dtype=np.int)
        # save episode num which can be used to update the agent
        self.ep_num_list = []

        self.num = 0
        self.ep_num = 0
        self.anchor_idx = 0
        self.last_save = 0
        self.full = False
        self.valid_start_ep_num = 0
        self.infos = {}

    @property
    def idx(self):
        return self.num % self.capacity

    @property
    def size(self):
        return self.capacity if self.full else self.idx

    def minus(self, y, x):
        return y - x if y >= x else y - x + self.capacity

    def add_obs(self, obs):
        np.copyto(self.obses[self.idx : self.idx + 2], np.stack((obs[:3], obs[3:6]), axis=0))
        self.num += 2
        self.ep_start_idxs[self.ep_num] = self.idx

    def add(self, obs, action, reward, next_obs, done, done_bool, ep_len, env_label=0):
        if self.full and self.anchor_idx == self.idx:
            if self.ep_end_idxs[self.valid_start_ep_num] == self.ep_start_idxs[self.valid_start_ep_num]:
                del self.ep_num_list[0]
                self.valid_start_ep_num = self.ep_num_list[0]
                self.anchor_idx = self.minus(self.ep_start_idxs[self.valid_start_ep_num], 2)
            else:
                self.anchor_idx = (self.anchor_idx + 1) % self.capacity
                self.ep_start_idxs[self.valid_start_ep_num] = (self.anchor_idx + 2) % self.capacity

        self.obses[self.idx] = obs[-3:].copy()
        self.actions[self.idx] = action.copy()
        self.rewards[self.idx] = reward
        self.not_dones[self.idx] = 1. - done_bool
        self.env_labels[self.idx] = env_label
        self.obs_idxes[self.idx] = self.num

        if done:
            # save the terminal obs
            self.num += 1
            self.obses[self.idx] = next_obs[-3:].copy()
            nstep = self.nstep - 1 if self.nstep > 1 else 1 # prepare for predictor
            nstep_rsd = self.nstep_rsd - 1 if self.nstep_rsd > 1 else 1
            self.ep_end_idxs[self.ep_num] = self.minus(self.idx, nstep)
            self.ep_end_idxs_rew[self.ep_num] = self.minus(self.idx, nstep_rsd)
            self.ep_num_list.append(self.ep_num)
            if self.save_buffer:
                self.save(self.buffer_dir, self.idx+1, self.ep_num, ep_len)
            self.ep_num += 1

        self.num += 1
        self.full = self.full or self.idx == 0

    def get_rew(self, idxs):
        infos = dict(gamma=self.gamma, nstep=self.nstep)
        if self.nstep == 1:
            return torch.as_tensor(self.rewards[idxs], device=self.device).float(), infos
        # nstep
        rewards, gamma = np.zeros((idxs.shape[0], 1)), 1.0
        for i in range(self.nstep):
            step_rewards = self.rewards[(idxs + i) % self.capacity]
            rewards += gamma * step_rewards
            gamma *= self.gamma
        infos['gamma'] = gamma
        return torch.as_tensor(rewards, device=self.device).float(), infos


    def sample_batch_with_obs_only(self, batch_size=None, enable_labels=False):
        if batch_size is None:
            batch_size = self.batch_size

        # sample indexes
        indexes = self.obs_idxes.copy()
        indexes[self.idx - 1] = -1
        idxs = np.random.choice(indexes[indexes != -1], batch_size)
        obs_idxs = idxs.repeat(3).reshape(-1, 3)
        obs_idxs[:, 0], obs_idxs[:, 1] = obs_idxs[:, 0]-2, obs_idxs[:, 1]-1
        idxs, obs_idxs = idxs % self.capacity, obs_idxs.flatten() % self.capacity

        obses = self.obses[obs_idxs].reshape(-1, 9, *self.obses.shape[-2:])
        obses = torch.as_tensor(obses, device=self.device).float()
        if enable_labels:
            env_labels = torch.as_tensor(self.env_labels[idxs], device=self.device).float()
            return obses, env_labels
        return obses

    def sample_batch_without_reward(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        # sample indexes
        indexes = self.obs_idxes.copy()
        indexes[self.idx - 1] = -1
        idxs = np.random.choice(indexes[indexes != -1], batch_size)
        obs_idxs = idxs.repeat(3).reshape(-1, 3)
        obs_idxs[:, 0], obs_idxs[:, 1] = obs_idxs[:, 0]-2, obs_idxs[:, 1]-1
        idxs, obs_idxs = idxs % self.capacity, obs_idxs.flatten() % self.capacity

        # get batch data without reward
        batch = dict(
            obs = self.obses[obs_idxs].reshape(-1, 9, *self.obses.shape[-2:]),
            obs2 = self.obses[(obs_idxs + self.nstep) % self.capacity].reshape(-1, 9, *self.obses.shape[-2:]),
            act = self.actions[idxs],
            not_done = self.not_dones[(idxs + self.nstep - 1) % self.capacity],
            env_labels = self.env_labels[idxs]
        )
        return {k: torch.as_tensor(v, device=self.device).float() for k, v in batch.items()}, idxs

    def sample_batch(self, batch_size=None):
        transition, idxs = self.sample_batch_without_reward(batch_size)
        rewards, infos = self.get_rew(idxs)
        transition['rew'], transition['infos'] = rewards, infos
        return transition

    def sample_batch_with_rs(self, batch_size=None):
        transition, idxs = self.sample_batch_without_reward(batch_size)
        rewards, infos = self.get_rew(idxs)
        transition['rew'], transition['infos'] = rewards, infos

        traj_act = transition['act'].unsqueeze(1)
        traj_rew = torch.as_tensor(self.rewards[idxs], device=self.device)
        gamma = self.gamma

        for step in range(1, self.nstep_rsd):
            # action sequence
            next_act = torch.as_tensor(self.actions[(idxs + step) % self.capacity],
                                       device=self.device).float()
            traj_act = torch.cat([traj_act, next_act.unsqueeze(1)], dim=1)
            # reward sequence
            step_rew = torch.as_tensor(self.rewards[(idxs + step) % self.capacity],
                                       device=self.device).float() * gamma
            traj_rew = torch.cat([traj_rew, step_rew], dim=-1)
            gamma *= self.gamma

        assert traj_rew.ndim == 2, print(traj_rew.size())
        transition['traj_a'] = traj_act # (batch_size, nstep, act_dim)
        transition['traj_r'] = traj_rew # (batch_size, nstep)
        return transition


    """when nstep != 1"""
    def sample_image_fs3_idxs(self, start, end):
        end[start >= end] += self.capacity
        idxs = np.random.randint(start, end)
        obs_idxs = idxs.repeat(3).reshape(-1, 3)
        obs_idxs[:, 0] -= 2
        obs_idxs[:, 1] -= 1
        obs_idxs = obs_idxs.reshape(-1)
        obs_idxs[obs_idxs < 0] += self.capacity
        return idxs % self.capacity, obs_idxs % self.capacity

    def sample_by_recording_list(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        ep_idxs = np.random.choice(np.array(self.ep_num_list), batch_size)
        ep_start_idxs = self.ep_start_idxs[ep_idxs]
        ep_end_idxs = self.ep_end_idxs[ep_idxs]
        idxs, obs_idxs = self.sample_image_fs3_idxs(ep_start_idxs, ep_end_idxs)

        obses = self.obses[obs_idxs].reshape(-1, 9, *self.obses.shape[-2:])
        next_obses = self.obses[(obs_idxs + self.nstep) % self.capacity].reshape(-1, 9, *self.obses.shape[-2:])

        obses = torch.as_tensor(obses, device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards, infos = self.get_rew(idxs)
        next_obses = torch.as_tensor(next_obses, device=self.device).float()
        not_dones = torch.as_tensor(self.not_dones[(idxs + self.nstep - 1) % self.capacity], device=self.device).float()
        env_labels = torch.as_tensor(self.env_labels[idxs], device=self.device)
        obs_num = torch.as_tensor(self.obs_idxes[idxs], device=self.device)

        transition = dict(obs=obses, obs2=next_obses, act=actions, rew=rewards.float(),
                          not_done=not_dones, env_labels=env_labels, obs_num=obs_num, infos=infos)
        return transition

    def sample_obs_by_recording_list(self, batch_size=None, enable_labels=False):
        if batch_size is None:
            batch_size = self.batch_size
        
        ep_idxs = np.random.choice(np.array(self.ep_num_list), batch_size)
        ep_start_idxs = self.ep_start_idxs[ep_idxs]
        ep_end_idxs = self.ep_end_idxs[ep_idxs]
        idxs, obs_idxs = self.sample_image_fs3_idxs(ep_start_idxs, ep_end_idxs)

        obses = self.obses[obs_idxs].reshape(-1, 9, *self.obses.shape[-2:])
        obses = torch.as_tensor(obses, device=self.device).float()
        if enable_labels:
            env_labels = torch.as_tensor(self.env_labels[idxs], device=self.device)
            return obses, env_labels
        return obses

    def sample_abst_traj_by_recording_list(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        ep_idxs = np.random.choice(np.array(self.ep_num_list), batch_size)
        ep_start_idxs = self.ep_start_idxs[ep_idxs]
        ep_end_idxs = self.ep_end_idxs_rew[ep_idxs]
        idxs, obs_idxs = self.sample_image_fs3_idxs(ep_start_idxs, ep_end_idxs)

        traj_a, traj_r = [], []
        obs = self.obses[obs_idxs].reshape(-1, 9, *self.obses.shape[-2:])
        obs = torch.as_tensor(obs, device=self.device).float()
        for i in range(self.nstep_rsd):
            act = self.actions[(idxs + i) % self.capacity]
            traj_a.append(torch.as_tensor(act, device=self.device))
            step_rewards = self.rewards[(idxs + i) % self.capacity]
            traj_r.append(torch.as_tensor(step_rewards, device=self.device))

        traj_a = torch.stack(traj_a, dim=0) # (nstep, batch_size, act_dim)
        traj_r = torch.stack(traj_r, dim=0).squeeze(-1) # (nstep, batch_size)
        o_end = self.obses[(obs_idxs + self.nstep_rsd) % self.capacity].reshape(-1, 9, *self.obses.shape[-2:])
        o_end = torch.as_tensor(o_end, device=self.device).float()
        envl = torch.as_tensor(self.env_labels[idxs], device=self.device)
        obs_num = torch.as_tensor(self.obs_idxes[idxs], device=self.device)
        traj = dict(traj_o=obs, traj_a=traj_a, traj_r=traj_r, obs_end=o_end, env_labels=envl, obs_num=obs_num)
        return traj


    def save(self, save_dir, idx, ep_num, ep_len):
        if idx == self.last_save:
            return
        _idx = idx + self.capacity if idx < self.last_save else idx
        path = save_dir / '%d_%d_from%d_to%d.pt' % (self.ep_num, ep_len, self.last_save, _idx)
        payload = [
            self.obses[self.last_save:idx] if _idx == idx else np.concatenate((self.obses[self.last_save:], self.obses[:idx%self.capacity]), axis=0),
            self.actions[self.last_save:idx] if _idx == idx else np.concatenate((self.actions[self.last_save:], self.actions[:idx%self.capacity]), axis=0),
            self.rewards[self.last_save:idx] if _idx == idx else np.concatenate((self.rewards[self.last_save:], self.rewards[:idx%self.capacity]), axis=0),
            self.not_dones[self.last_save:idx] if _idx == idx else np.concatenate((self.not_dones[self.last_save:], self.not_dones[:idx%self.capacity]), axis=0),
            self.obs_idxes[self.last_save:idx] if _idx == idx else np.concatenate((self.obs_idxes[self.last_save:], self.obs_idxes[:idx%self.capacity]), axis=0),
            self.env_labels[self.last_save:idx] if _idx == idx else np.concatenate((self.env_labels[self.last_save:], self.env_labels[:idx%self.capacity]), axis=0),
        ]
        self.last_save = idx
        torch.save(payload, path)

    def load(self, save_dir):
        chunks = os.listdir(save_dir)
        chucks = sorted(chunks, key=lambda x: int(x.split('_')[0]))
        for chunk in chucks:
            ep_num, ep_len, start, end = [int(x) for x in chunk.split('.')[0].split('_')]
            path = save_dir / chunk
            payload = torch.load(path)
            end = (end - 1 + self.capacity) % self.capacity + 1
            if end < start:
                self.obses[start:], self.obses[:end] = payload[0][:self.capacity-start], payload[0][self.capacity-start:]
                self.actions[start:], self.actions[:end] = payload[1][:self.capacity-start], payload[1][self.capacity-start:]
                self.rewards[start:], self.rewards[:end] = payload[2][:self.capacity-start], payload[2][self.capacity-start:]
                self.not_dones[start:], self.not_dones[:end] = payload[3][:self.capacity-start], payload[3][self.capacity-start:]
                self.obs_idxes[start:], self.obs_idxes[:end] = payload[4][:self.capacity-start], payload[4][self.capacity-start:]
                self.env_labels[start:], self.env_labels[:end] = payload[5][:self.capacity-start], payload[5][self.capacity-start:]
            else:
                self.obses[start:end] = payload[0]
                self.actions[start:end] = payload[1]
                self.rewards[start:end] = payload[2]
                self.not_dones[start:end] = payload[3]
                self.obs_idxes[start:end] = payload[4]
                self.env_labels[start:end] = payload[5]
            self.ep_num_list.append(ep_num)
            self.ep_start_idxs[ep_num] = (start + 2) % self.capacity
            self.ep_end_idxs[ep_num] = self.minus(end, self.nstep)
            self.num = end

    def __len__(self):
        return self.capacity

