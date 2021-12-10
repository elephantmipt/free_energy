from dataclasses import dataclass, field
import random
from typing import List, Any, Sequence

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

    def push(self, samples, class_ids=None):
        samples = np.asarray(samples)
        class_ids = np.asarray(class_ids)

        for sample, class_id in zip(samples, class_ids):
            self.buffer.append((sample, class_id))

            if len(self.buffer) > self.max_samples:
                self.buffer.pop(0)

    def get(self, n_samples):
        items = random.choices(self.buffer, k=n_samples)
        samples, class_ids = zip(*items)
        samples = jnp.stack(samples, 0)
        class_ids = jnp.array(class_ids)
        return samples, class_ids

    def sample(self, batch_size, key, p: float = 0.95):
        if len(self) == 0:
            return (
                jax.random.uniform(key, shape=[batch_size, *self.image_shape]),
                jax.random.randint(key, shape=(batch_size,), minval=0, maxval=self.num_classes)
            )
        n_replay = (np.random.rand(batch_size) < p).sum()

        replay_sample, replay_id = self.get(n_replay)
        random_sample = jax.random.uniform(key, shape=[batch_size - n_replay, *self.image_shape])
        random_id = jax.random.randint(key, shape=(batch_size - n_replay,), minval=0, maxval=self.num_classes)
        return (
            jnp.concatenate((replay_sample, random_sample)),
            jnp.concatenate((replay_id, random_id))
        )

def sample_data(loader):
    loader_iter = iter(loader)

    while True:
        try:
            yield next(loader_iter)

        except StopIteration:
            loader_iter = iter(loader)

            yield next(loader_iter)
