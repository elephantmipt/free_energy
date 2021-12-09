import jax

from free.sampling.base import Sampler


class MCMCSampler(Sampler):

    step_size: float = 0.02

    def step(self, img):
        energy_grad = jax.grad(self.apply_fn)
        next_img = img - self.step_size * energy_grad(img)
        return next_img

