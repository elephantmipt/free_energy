from dataclasses import dataclass, field
from typing import List, Any, Sequence

import jax
import numpy as np
from jax import numpy as jnp


@dataclass
class ImageBuffer:

    buffer_size: int
    img_shape: Sequence[int]
    storage: List[Any] = field(default_factory=list, init=False)

    def __call__(self, key):
        if self.is_empty():
            return jax.random.normal(key=key, shape=self.img_shape)
        current_index = jax.random.randint(
            key=key, shape=[1], minval=0, maxval=len(self.storage)
        )[0]
        return jnp.asarray(self.storage[current_index])

    def push(self, img):
        self.storage.append(np.asarray(img))
        if len(self.storage) > self.buffer_size:
            self.storage.pop(0)

    def is_empty(self):
        return len(self.storage) == 0
