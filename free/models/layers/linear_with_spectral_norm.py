from typing import Optional

import haiku as hk
import numpy as np
from jax import numpy as jnp, lax


class LinearWithSpectralNorm(hk.Module):
    def __init__(
            self,
            output_size: int,
            with_bias: bool = True,
            num_power_iterations: int = 5,
            w_init: Optional[hk.initializers.Initializer] = None,
            b_init: Optional[hk.initializers.Initializer] = None,
            name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.input_size = None
        self.output_size = output_size
        self.num_power_iterations = num_power_iterations
        self.with_bias = with_bias
        self.w_init = w_init
        self.b_init = b_init or jnp.zeros
        self.normalizer = hk.SpectralNorm()

    def __call__(
            self,
            inputs: jnp.ndarray,
            *,
            update_stats: bool = True,
            precision: Optional[lax.Precision] = None,
    ) -> jnp.ndarray:
        """Computes a linear transform of the input."""
        if not inputs.shape:
            raise ValueError("Input must not be scalar.")

        input_size = self.input_size = inputs.shape[-1]
        output_size = self.output_size
        dtype = inputs.dtype

        w_init = self.w_init
        if w_init is None:
            stddev = 1. / np.sqrt(self.input_size)
            w_init = hk.initializers.TruncatedNormal(stddev=stddev)

        w = hk.get_parameter("w", [input_size, output_size], dtype, init=w_init)
        w = self.normalizer(w, update_stats=update_stats)
        hk.set_state("w", w)

        out = jnp.dot(inputs, w, precision=precision)

        if self.with_bias:
            b = hk.get_parameter("b", [self.output_size], dtype, init=self.b_init)
            b = jnp.broadcast_to(b, out.shape)
            out = out + b

        return out
