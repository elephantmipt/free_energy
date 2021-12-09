from dataclasses import dataclass
from typing import Callable

import haiku as hk
import jax.random


@dataclass
class Sampler:

    apply_fn: Callable

    def step(self, img, params, key, *args, **kwargs):
        pass

    def perform_generation(self, start_img, params: hk.Params, key, num_steps: int, *args, **kwargs):
        img = start_img
        for _ in range(num_steps):
            step_key, key = jax.random.split(key)
            img = self.step(img, params, key=step_key)
        return img, key
