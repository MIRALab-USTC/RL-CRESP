import gym.spaces as spaces
from gym import ObservationWrapper


class ObservationByKey(ObservationWrapper):
    r"""Observation wrapper that flattens the observation."""

    def __init__(self, env, obs_key):
        super().__init__(env)
        self.obs_key = obs_key
        self.observation_space = env.observation_space[obs_key]

    def observation(self, observation):
        return observation[self.obs_key]
