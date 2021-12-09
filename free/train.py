import argparse
from typing import Generator, Mapping, Tuple

import numpy as np
import jax
from jax import numpy as jnp

import haiku as hk
import optax

import tensorflow_datasets as tfds

from free.buffer.images import ImageBuffer
from free.models.images import LeNet
from free.sampling.langevin import LangevinSampler

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

    def loss(params: hk.Params, positive_image, negative_img, alpha) -> jnp.ndarray:
        positive_energy = model.apply(params, positive_image)
        negative_energy = model.apply(params, negative_img)

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
    ) -> Tuple[hk.Params, optax.OptState]:
        """Learning rule (stochastic gradient descent)."""
        grads = jax.grad(loss)(params, pos_image, neg_image, alpha)
        updates, opt_state = opt.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, opt_state

    train = load_dataset("train", is_training=True, batch_size=128)
    prng = jax.random.PRNGKey(42)
    init_rng, prng = jax.random.split(prng)
    params = model.init(init_rng, next(train)["image"])
    opt_state = opt.init(params)
    buffer = ImageBuffer(buffer_size=500, img_shape=next(train)["image"].shape)
    sampler = LangevinSampler(apply_fn=model.apply)
    for epoch in range(args.num_epochs):

        for step, batch in enumerate(train):
            buffer_key, prng = jax.random.split(prng)
            neg_image = buffer(buffer_key)
            neg_image, prng = sampler.perform_generation(
                start_img=neg_image, params=params, key=prng, num_steps=args.num_gen_steps
            )
            neg_image = jax.lax.stop_gradient(neg_image)
            pos_image = batch["image"]
            update(params=params, opt_state=opt_state, pos_image=pos_image, neg_image=neg_image, alpha=1.)
            buffer.push(neg_image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-epochs", default=10, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--num-gen-steps", default=100, type=int)
    parser.add_argument("--batch-size", default=128, type=int)
    args = parser.parse_args()
    main(args=args)





