import jax
import haiku as hk


class LeNet(hk.Module):
    def __init__(self, activation=jax.nn.leaky_relu):
        super().__init__()
        self.lin_1 = hk.Linear(128)
        self.lin_2 = hk.Linear(64)
        self.lin_3 = hk.Linear(1)
        self.activation = activation

    def __call__(self, x):
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)
        x = self.lin_1(x)
        x = self.activation(x)
        x = self.lin_2(x)
        x = self.activation(x)
        x = self.lin_3(x)
        return x
