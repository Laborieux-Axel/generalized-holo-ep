from typing import Optional, Sequence, Tuple, Union, Any
import numpy as np

import jax.numpy as jnp
from jax import grad, lax
from flax import linen as nn



def _infer_shape(
    x: jnp.ndarray,
    size: Union[int, Sequence[int]],
    channel_axis: Optional[int] = -1,
) -> Tuple[int, ...]:
    if isinstance(size, int):
        if channel_axis and not 0 <= abs(channel_axis) < x.ndim:
            raise ValueError(f"Invalid channel axis {channel_axis} for {x.shape}")
        if channel_axis and channel_axis < 0:
            channel_axis = x.ndim + channel_axis
        return (1,) + tuple(size if d != channel_axis else 1
                        for d in range(1, x.ndim))
    elif len(size) < x.ndim:
    # Assume additional dimensions are batch dimensions.
        return (1,) * (x.ndim - len(size)) + tuple(size)
    else:
        assert x.ndim == len(size)
        return tuple(size)


def window_wise_max(x, ws, strides, padding):
    
    holo = (x.dtype == jnp.complex64)
    ones = jnp.ones_like(x)

    def f(o, x):
        sum_pooled = lax.reduce_window(o, 0., lax.add, ws, strides, padding)
        max_pooled = lax.reduce_window(x, -jnp.inf, lax.max, ws, strides, padding)
        return jnp.sum(sum_pooled * max_pooled)

    return grad(f, holomorphic=holo)(ones, x)





def sfm_pool(
    value: jnp.ndarray,
    window_shape: Union[int, Sequence[int]],
    strides: Union[int, Sequence[int]],
    padding: str,
    channel_axis: Optional[int] = -1,
    T: Optional[float] = 1.0,
) -> jnp.ndarray:

    if padding not in ("SAME", "VALID"):
        raise ValueError(f"Invalid padding '{padding}', must be 'SAME' or 'VALID'.")

    window_shape = _infer_shape(value, window_shape, channel_axis)
    strides = _infer_shape(value, strides, channel_axis)

    T = jnp.array(T).astype(value.dtype)
    tmp_v = lax.div(value, T)
    max_v = window_wise_max(tmp_v, window_shape, strides, padding)
    safe_exp = lax.exp(tmp_v - lax.stop_gradient(max_v))
    numer = lax.mul(value, safe_exp)

    reduce_window_args = (0., lax.add, window_shape, strides, padding)

    pooled_num = lax.reduce_window(numer, *reduce_window_args)
    pooled_den = lax.reduce_window(safe_exp, *reduce_window_args)
    out = lax.div(pooled_num, pooled_den)
    return out





class SfmPool(nn.Module):
    window_shape: Union[int, Sequence[int]]
    strides: Union[int, Sequence[int]]
    padding: str
    channel_axis: Optional[int] = -1
    T: Optional[float] = 1.0

    def __call__(self, value: jnp.ndarray) -> jnp.ndarray:
        return sfm_pool(value, self.window_shape, self.strides,
                        self.padding, self.channel_axis, self.T)

