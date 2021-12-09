import jax

from free.sampling.base import Sampler


class LangevinSampler(Sampler):

    step_size: float = 0.02

    def step(self, img, params, key, *args, **kwargs):
        @jax.jit
        def energy(img, params):
            return self.apply_fn(params, img).mean()

        energy_grad = jax.grad(energy)
        noise = jax.random.normal(key=key, shape=img.shape) * self.step_size * 2
        next_img = img - self.step_size * energy_grad(img, params) + noise
        next_img = jax.lax.clamp(0., next_img, 1.)
        return next_img

