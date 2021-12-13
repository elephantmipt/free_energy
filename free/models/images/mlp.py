import jax
import haiku as hk

from free.models.layers import LinearWithSpectralNorm


class LeNet(hk.Module):
    def __init__(self, activation=jax.nn.leaky_relu):
        super().__init__()
        self.lin_1 = LinearWithSpectralNorm(128)
        self.lin_2 = LinearWithSpectralNorm(64)
        self.lin_3 = hk.Linear(1, )
        self.activation = activation

    def __call__(self, x):
        x = hk.Flatten()(x)
        x = self.lin_1(x)
        x = self.activation(x)
        x = self.lin_2(x)
        x = self.activation(x)
        x = self.lin_3(x)
        return x
