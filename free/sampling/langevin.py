import jax

from free.sampling.base import Sampler







class LangevinSampler(Sampler):

    step_size: float = 0.02

    def step(self, img, params, key, *args, **kwargs):

        energy_grad = jax.grad(energy)
        noise = jax.random.normal(key=key, shape=img.shape) * self.step_size * 2
        e_grad = energy_grad(img, params)
        e_grad = jax.lax.clamp(-0.01, e_grad, 0.01)
        next_img = img - self.step_size * e_grad + noise
        next_img = jax.lax.clamp(0., next_img, 1.)
        return next_img

