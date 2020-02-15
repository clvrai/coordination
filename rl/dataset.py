from collections import defaultdict
from time import time

import numpy as np


class ReplayBuffer:
    """ Replay Buffer. """

    def __init__(self, keys, buffer_size, sample_func):
        self._size = buffer_size

        # memory management
        self._idx = 0
        self._current_size = 0
        self._sample_func = sample_func

        # create the buffer to store info
        self._keys = keys
        self._buffers = defaultdict(list)

    def clear(self):
        self._idx = 0
        self._current_size = 0
        self._buffers = defaultdict(list)

    def store_episode(self, rollout):
        """ Stores the episode. """
        idx = self._idx = (self._idx + 1) % self._size
        self._current_size += 1

        if self._current_size > self._size:
            for k in self._keys:
                self._buffers[k][idx] = rollout[k]
        else:
            for k in self._keys:
                self._buffers[k].append(rollout[k])

    def sample(self, batch_size):
        """ Samples the data from the replay buffer. """
        # sample transitions
        transitions = self._sample_func(self._buffers, batch_size)
        return transitions

    def state_dict(self):
        return self._buffers

    def load_state_dict(self, state_dict):
        self._buffers = state_dict
        self._current_size = len(self._buffers['ac'])


class RandomSampler:
    """ Samples a batch of transitions from replay buffer. """

    def sample_func(self, episode_batch, batch_size_in_transitions):
        rollout_batch_size = len(episode_batch['ac'])
        batch_size = batch_size_in_transitions

        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        t_samples = [np.random.randint(len(episode_batch['ac'][episode_idx])) for episode_idx in episode_idxs]

        transitions = {}
        for key in episode_batch.keys():
            transitions[key] = \
                [episode_batch[key][episode_idx][t] for episode_idx, t in zip(episode_idxs, t_samples)]

        transitions['ob_next'] = [
            episode_batch['ob'][episode_idx][t + 1] for episode_idx, t in zip(episode_idxs, t_samples)]

        new_transitions = {}
        for k, v in transitions.items():
            if isinstance(v[0], dict):
                sub_keys = v[0].keys()
                new_transitions[k] = {
                    sub_key: np.stack([v_[sub_key] for v_ in v]) for sub_key in sub_keys
                }
            else:
                new_transitions[k] = np.stack(v)

        return new_transitions

