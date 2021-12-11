import argparse
from typing import Generator, Mapping, Tuple

import numpy as np
import jax
from jax import numpy as jnp

import haiku as hk
import optax

import tensorflow_datasets as tfds
from tqdm.auto import trange, tqdm

from free.buffer.images import SampleBuffer
from free.models.images import LeNet

Batch = Mapping[str, np.ndarray]

batch_size = 8
num_steps = 100

num_training_steps = 10000

def model_fn(image):
    x = image.astype(jnp.float32) / 255.
    le_net = LeNet()
    return le_net(x)


def load_dataset(
    split: str,
    *,
    is_training: bool,
    batch_size: int,
) -> Generator[Batch, None, None]:
    """Loads the dataset as a generator of batches."""
    ds = tfds.load("mnist:3.*.*", split=split).cache().repeat()
    if is_training:
        ds = ds.shuffle(10 * batch_size, seed=0)
    ds = ds.batch(batch_size)
    return iter(tfds.as_numpy(ds))


model = hk.without_apply_rng(
    hk.transform(model_fn)
)
opt = optax.adam(1e-3)


@jax.jit
def loss(params: hk.Params, pos_image, neg_image, alpha) -> jnp.ndarray:
    positive_energy = model.apply(params, pos_image)
    negative_energy = model.apply(params, neg_image)

    loss = alpha * (positive_energy ** 2 + negative_energy ** 2)
    loss = loss + (positive_energy - negative_energy)

    return loss.mean()


@jax.jit
def update(
    params: hk.Params,
    opt_state: optax.OptState,
    pos_image,
    neg_image,
    alpha,
) -> Tuple[jnp.ndarray, hk.Params, optax.OptState]:
    """Learning rule (stochastic gradient descent)."""
    loss_value, grads = jax.value_and_grad(loss)(params, pos_image=pos_image, neg_image=neg_image, alpha=alpha)
    updates, opt_state = opt.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return loss_value, new_params, opt_state

train = load_dataset("train", is_training=True, batch_size=batch_size)
prng = jax.random.PRNGKey(42)
init_rng, prng = jax.random.split(prng)
params = model.init(init_rng, next(train)["image"])
opt_state = opt.init(params)
buffer = SampleBuffer(image_shape=next(train)["image"][0].shape)


@jax.jit
def kinetic_energy(velocity):
    return 0.5 * velocity ** 2


@jax.jit
def energy(img, params):
    return model.apply(params, img).sum()


@jax.jit
def hamiltonian(position, velocity, params):
    batch_size = velocity.shape[0]
    kinetic_energy_flat = kinetic_energy(velocity).reshape(batch_size, -1)
    return energy(position, params) + kinetic_energy_flat.sum(1)


@jax.jit
def leapfrog_step(x0, v0, step_size, params):
    energy_grad = jax.grad(energy)

    v = v0 - 0.5 * step_size * energy_grad(x0, params)

    x = x0 + v * step_size

    for i in range(num_steps):
        v = v - step_size * energy_grad(x, params)

        x = x + step_size * v

    v = v - 0.5 * step_size * energy_grad(x, params)
    return x, v


def hmc(initial_x, step_size, params, key):
    v0 = jax.random.normal(key=key, shape=initial_x.shape)
    x, v = leapfrog_step(
        x0=initial_x,
        v0=v0,
        step_size=step_size,
        params=params,
    )


    #orig = hamiltonian(initial_x, v0, params=params)
    #current = hamiltonian(x, v, params=params)

    #acceptance_prob = jnp.exp(current - orig)

    #uniform = jax.random.uniform(key=key, shape=acceptance_prob.shape)

    #keep_mask = acceptance_prob > uniform
    #x_new = jnp.where(keep_mask, x, initial_x)
    #print(x_new.shape)
    return x


it = tqdm(enumerate(train), total=num_training_steps)
for step, batch in it:
    if step + 1 == num_training_steps:
        break
    buffer_key, prng = jax.random.split(prng)
    neg_image = buffer.sample(batch_size=batch_size, key=buffer_key)
    step_key, prng = jax.random.split(prng)
    neg_image = hmc(initial_x=neg_image, step_size=0.1, params=params, key=step_key)
    neg_image = jax.lax.stop_gradient(neg_image)
    pos_image = batch["image"]
    loss_value, params, opt_state = update(
        params=params, opt_state=opt_state, pos_image=pos_image, neg_image=neg_image, alpha=1
    )
    it.set_description(f"Loss value: {loss_value:.4f}")
    buffer.push(neg_image)

