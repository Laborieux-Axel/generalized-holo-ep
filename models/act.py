import jax.numpy as jnp
from jax.tree_util import tree_map
from flax import linen as nn
from typing import Callable


class SimpleAct(nn.Module):
    act: Callable[[jnp.ndarray], jnp.ndarray]

    def __call__(self, x):
        return tree_map(self.act, x)
