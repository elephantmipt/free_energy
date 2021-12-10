import jax
import haiku as hk


class LeNet(hk.Module):
    def __init__(self, activation=jax.nn.leaky_relu):
        super().__init__()
        self.conv_1 = hk.Conv2D(32, kernel_shape=(4, 4))
        self.conv_2 = hk.Conv2D(64, kernel_shape=(4, 4))
        self.conv_3 = hk.Conv2D(128, kernel_shape=(4, 4))
        self.lin_1 = hk.Linear(64)
        self.lin_2 = hk.Linear(1)
        self.activation = activation

    def __call__(self, x):
        batch_size = x.shape[0]
        x = self.conv_1(x)
        x = self.activation(x)
        x = self.conv_2(x)
        x = self.activation(x)
        x = self.conv_3(x)
        x = x.reshape(batch_size, -1)
        x = self.activation(x)
        x = self.lin_1(x)
        x = self.activation(x)
        x = self.lin_2(x)
        return x
