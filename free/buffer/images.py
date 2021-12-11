from dataclasses import dataclass, field
import random
from typing import Sequence

import jax
import numpy as np
from jax import numpy as jnp


class SampleBuffer:
    def __init__(self, image_shape: Sequence[int], max_samples: int = 10000, num_classes: int = 10):
        self.max_samples = max_samples
        self.image_shape = image_shape
        self.num_classes = num_classes
        self.buffer = []

    def __len__(self):
        return len(self.buffer)

    def push(self, samples):
        samples = np.asarray(samples)

        for sample in samples:
            self.buffer.append(sample)

            if len(self.buffer) > self.max_samples:
                self.buffer.pop(0)

    def get(self, n_samples):
        items = random.choices(self.buffer, k=n_samples)
        samples = jnp.stack(items, 0)
        return samples

    def sample(self, batch_size, key, p: float = 0.95):
        if len(self) == 0:
            return jax.random.uniform(key, shape=[batch_size, *self.image_shape])

        n_replay = (np.random.rand(batch_size) < p).sum()

        replay_sample = self.get(n_replay)
        random_sample = jax.random.uniform(key, shape=[batch_size - n_replay, *self.image_shape])
        return jnp.concatenate((replay_sample, random_sample))
