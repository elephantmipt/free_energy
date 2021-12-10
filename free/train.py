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


def main(args):
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

    train = load_dataset("train", is_training=True, batch_size=128)
    prng = jax.random.PRNGKey(42)
    init_rng, prng = jax.random.split(prng)
    params = model.init(init_rng, next(train)["image"])
    opt_state = opt.init(params)
    buffer = SampleBuffer(image_shape=next(train)["image"][0].shape)
    batch_size = next(train)["image"].shape[0]

    @jax.jit
    def energy(img, params):
        return model.apply(params, img).mean()

    @jax.jit
    def energy_step(img, params, key, step_size=0.02):
        energy_grad = jax.grad(energy)
        noise = jax.random.normal(key=key, shape=img.shape) * step_size * 2
        e_grad = energy_grad(img, params)
        e_grad = jax.lax.clamp(-0.01, e_grad, 0.01)
        next_img = img - step_size * e_grad + noise
        next_img = jax.lax.clamp(0., next_img, 1.)
        return next_img

    for epoch in trange(args.num_epochs):
        iter_train = tqdm(enumerate(train), leave=False)
        for step, batch in iter_train:
            buffer_key, prng = jax.random.split(prng)
            neg_image, neg_ids = buffer.sample(batch_size=batch_size, key=buffer_key)
            for e_s in range(args.num_gen_steps):
                step_key, prng = jax.random.split(prng)
                neg_image = energy_step(img=neg_image, params=params, key=step_key)
            neg_image = jax.lax.stop_gradient(neg_image)
            pos_image = batch["image"]
            loss_value, params, opt_state = update(
                params=params, opt_state=opt_state, pos_image=pos_image, neg_image=neg_image, alpha=1.
            )
            iter_train.set_description(f"Loss value: {loss_value:.4f}")
            buffer.push(neg_image, class_ids=jnp.zeros((batch_size,)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-epochs", default=10, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--num-gen-steps", default=60, type=int)
    parser.add_argument("--batch-size", default=128, type=int)
    args = parser.parse_args()
    main(args=args)





