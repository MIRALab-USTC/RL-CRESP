# coding=utf-8
# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""A wrapper for dm_control environments which applies color distractions."""

import numpy as np
from dm_control.rl import control

from .mixins import GetStateMixin

class DistractingColorEnv(control.Environment, GetStateMixin):
    """Environment wrapper for color visual distraction.

    **NOTE**: This wrapper should be applied BEFORE the pixel wrapper to make sure
    the color changes are applied before rendering occurs.
    """

    def __init__(self, env, step_std, max_delta, fix_color=False, seed=None, sample_from_edge=False):
        """Initialize the environment wrapper.

        Args:
          env: instance of dm_control Environment to wrap with augmentations.
        """
        if step_std < 0:
            raise ValueError("`step_std` must be greater than or equal to 0.")
        if max_delta < 0:
            raise ValueError("`max_delta` must be greater than or equal to 0.")

        self._env = env
        self._step_std = step_std
        self._max_delta = max_delta
        self._random_state = np.random.RandomState(seed=seed)

        self._cam_type = None
        self._current_rgb = None
        self._max_rgb = None
        self._min_rgb = None
        self._original_rgb = None

        # Fix color across episode (This is different from `dynamic`!)
        self._fix_color = fix_color
        self._sample_from_edge = sample_from_edge
        self._seed = seed

    def reset(self):
        """Reset the distractions state."""
        time_step = self._env.reset()
        if self._original_rgb is None or not self._fix_color:
            self._reset_color()
        return time_step

    def _reset_color(self):
        from .suite_utils import sample
        # Save all original colors.
        if self._original_rgb is None:
            self._original_rgb = np.copy(self._env.physics.model.mat_rgba)[:, :3]
            # Determine minimum and maximum rgb values.
            self._max_rgb = np.clip(self._original_rgb + self._max_delta, 0.0, 1.0)
            self._min_rgb = np.clip(self._original_rgb - self._max_delta, 0.0, 1.0)

        # Pick random colors in the allowed ranges.
        distribution = 'edge' if self._sample_from_edge else 'uniform'
        r = sample(self._random_state, size=self._min_rgb.shape, distribution=distribution)
        self._current_rgb = self._min_rgb + r * (self._max_rgb - self._min_rgb)

        # Apply the color changes.
        self._env.physics.model.mat_rgba[:, :3] = self._current_rgb

    def step(self, action):
        time_step = self._env.step(action)

        if time_step.first() and not self._fix_color:
            self._reset_color()
            return time_step

        color_change = self._random_state.randn(*self._current_rgb.shape)
        color_change = color_change * self._step_std

        new_color = self._current_rgb + color_change

        self._current_rgb = np.clip(
            new_color,
            a_min=self._min_rgb,
            a_max=self._max_rgb,
        )

        # Apply the color changes.
        self._env.physics.model.mat_rgba[:, :3] = self._current_rgb
        return time_step

    # Forward property and method calls to self._env.
    def __getattr__(self, attr):
        if hasattr(self._env, attr):
            return getattr(self._env, attr)
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    @classmethod
    def from_dict(cls, env, state):
        # instance._original_rgb must be obtained from the current env, never reuse the old one.
        state.pop('_original_rgb')

        # Instantiate the class in whatever way and set attributes
        instance = cls(env,
                       state['_step_std'],
                       state['_max_delta'])
        for key, val in state.items():
            setattr(instance, key, val)

        assert instance._fix_color

        # Directly apply the color changes.
        instance._env.physics.model.mat_rgba[:, :3] = instance._current_rgb
        instance._original_rgb = np.copy(instance._env.physics.model.mat_rgba)[:, :3]

        return instance
