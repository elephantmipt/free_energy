from typing import Generator, Mapping, Tuple

import numpy as np
import jax
from jax import numpy as jnp

import haiku as hk
import optax

import tensorflow_datasets as tfds
from tqdm.auto import tqdm

from free.buffer.images import SampleBuffer
from free.models.images import LeNet

from matplotlib import pyplot as plt
import seaborn as sns

sns.set(style="whitegrid", font_scale=1.4)


Batch = Mapping[str, np.ndarray]

batch_size = 128
num_steps = 60

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
    hk.transform_with_state(model_fn)
)
opt = optax.adam(1e-2)


@jax.jit
def loss(params: hk.Params, state: hk.State, pos_image, neg_image, alpha) -> Tuple[jnp.ndarray, hk.State]:
    positive_energy, state = model.apply(params, state, pos_image)
    negative_energy, state = model.apply(params, state, neg_image)
    loss = alpha * (positive_energy ** 2 + negative_energy ** 2)
    loss = loss + (positive_energy - negative_energy)

    return loss.mean(), state


@jax.jit
def update(
        params: hk.Params,
        state: hk.State,
        opt_state: optax.OptState,
        pos_image,
        neg_image,
        alpha,
) -> Tuple[jnp.ndarray, hk.Params, hk.State, optax.OptState]:
    """Learning rule (stochastic gradient descent)."""
    (loss_value, state), grads = jax.value_and_grad(loss, has_aux=True)(params, state=state, pos_image=pos_image, neg_image=neg_image, alpha=alpha)
    updates, opt_state = opt.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return loss_value, new_params, state, opt_state


train = load_dataset("train", is_training=True, batch_size=batch_size)
prng = jax.random.PRNGKey(42)
init_rng, prng = jax.random.split(prng)
params, state = model.init(init_rng, image=next(iter(train))["image"])
opt_state = opt.init(params)
buffer = SampleBuffer(image_shape=next(train)["image"][0].shape)


@jax.jit
def kinetic_energy(velocity):
    return 0.5 * velocity ** 2


@jax.jit
def hamiltonian(position, velocity, params):
    batch_size = velocity.shape[0]
    kinetic_energy_flat = kinetic_energy(velocity).reshape(batch_size, -1)
    return model.apply(params, position) + kinetic_energy_flat.sum(1)


@jax.jit
def energy(x, params, state):
    energy_value, state = model.apply(params, state, x)
    return energy_value.sum(), state


energy_grad = jax.value_and_grad(energy, has_aux=True)


@jax.jit
def leapfrog_step(x0, v0, step_size, params):
    e_value, e_grad = energy_grad(x0, params)

    v = v0 - 0.5 * step_size * e_grad

    x = x0 + v * step_size

    history = jnp.zeros(num_steps)

    for i in range(num_steps):
        e_value, e_grad = energy_grad(x, params)

        history[i] = e_value

        v = v - step_size * e_grad

        x = x + step_size * v

    v = v - 0.5 * step_size * energy_grad(x, params)
    return x, v, history


def mcmc(initial_x, step_size, params, state, key):
    x = initial_x
    history = np.zeros(num_steps)
    for i in range(num_steps):
        (e_value, state), e_grad = energy_grad(x, params, state)

        x = x - step_size * e_grad
        history[i] = e_value.astype(float)

    return x, state, history


def hmc(initial_x, step_size, params, key):
    v0 = jax.random.normal(key=key, shape=initial_x.shape)
    x, v, history = leapfrog_step(
        x0=initial_x,
        v0=v0,
        step_size=step_size,
        params=params,
    )

    # orig = hamiltonian(initial_x, v0, params=params)
    # current = hamiltonian(x, v, params=params)

    # acceptance_prob = jnp.exp(current - orig)

    # uniform = jax.random.uniform(key=key, shape=acceptance_prob.shape)

    # keep_mask = acceptance_prob > uniform
    # x_new = jnp.where(keep_mask, x, initial_x)
    # print(x_new.shape)
    return x, history

it = tqdm(enumerate(train), total=num_training_steps)
meta_hist = []
for step, batch in it:
    if step + 1 == num_training_steps:
        break
    buffer_key, prng = jax.random.split(prng)
    neg_image = buffer.sample(batch_size=batch_size, key=buffer_key)
    step_key, prng = jax.random.split(prng)
    neg_image, state, history = mcmc(initial_x=neg_image, step_size=5000, params=params, state=state, key=step_key)
    meta_hist.append(np.asarray(history))
    neg_image = jax.lax.stop_gradient(neg_image)
    pos_image = batch["image"]
    #if step % 1000 == 0:
        #jnp.save(f"checkpoint_{step}.jax", params)

        #for i in range(neg_image.shape[0]):
            #plt.figure(figsize=(12, 8))
            #plt.grid()
            #plt.imshow(neg_image[i, :, :, 0])
            #plt.savefig(f"step_{step}_batch_{i}.svg", format="svg")
            #plt.close()

    loss_value, params, state, opt_state = update(
        params=params, opt_state=opt_state, state=state, pos_image=pos_image, neg_image=neg_image, alpha=1
    )

    it.set_description(f"Loss value: {loss_value:.4f}")
    buffer.push(neg_image)


