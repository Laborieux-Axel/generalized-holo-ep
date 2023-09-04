import pprint
import jax.numpy as jnp
from jax.nn import standardize
from jax import device_get, device_put_replicated
from jax import random, local_devices, local_device_count
from jax.tree_util import tree_map, tree_flatten, tree_unflatten, tree_leaves
from flax.core.frozen_dict import freeze


def sigmoid(z):
    return 1/(1 + jnp.exp(-4 * z + 2))


def dsilu(z):
    return 0.5*z/(1 + jnp.exp(-z)) + (1-0.5*z)/(1 + jnp.exp(-(z-2)) )


def get_activation(name):
    if name == 'sigmoid':
        return sigmoid
    elif name == 'dsilu':
        return dsilu
    else:
        raise ValueError(f'Activation {name} not supported')


def concat_flat_leaves(tree):
    leaves = tree_leaves(tree_map(lambda x: x.flatten(), tree))
    return jnp.concatenate(leaves, axis=-1)

def cosine_sim(a, b):
    a, b = a.flatten(), b.flatten()
    return jnp.inner(a, b)/(jnp.linalg.norm(a) * jnp.linalg.norm(b))

def dict_to_vec(a):
    flat_a = tree_map(lambda x: x.flatten(), a)
    return jnp.concatenate(tree_leaves(flat_a))
    
def dict_cossim(a, b):
    return cosine_sim(dict_to_vec(a), dict_to_vec(b))

def load_params(old, new):
    _, tree_def = tree_flatten(old)
    leaves, _ = tree_flatten(new)
    return tree_unflatten(tree_def, leaves)

def to_complex(ptree):
    return tree_map(lambda x: x.astype(jnp.complex64), ptree)

def to_float16(ptree):
    return tree_map(lambda x: x.astype(jnp.float16), ptree)

def to_float32(ptree):
    return tree_map(lambda x: x.astype(jnp.float32), ptree)

def pretty_print(dictio):
    print()
    d = tree_map(lambda x: (x.shape, x.dtype) if isinstance(x, jnp.ndarray) else x, dictio)
    pprint.PrettyPrinter(depth=6).pprint(d)
    print()

def params_std(params):
    out = {}
    for k in params['params']:
        if 'readout' in k:
            continue
        out[f'{k}_std'] = jnp.std(params['params'][k]['kernel'])
    return out



def crank_gain(params, gain):
    """cranks the inhibition by a factor of `gain`"""
    params = params.unfreeze()
    for k in params['params']:
        if 'readout' in k:
            continue
        if 'BatchNorm' in k:
            continue
        params['params'][k]['kernel'] = params['params'][k]['kernel'] * gain
    return freeze(params)





def rotate(params, alpha=0.0, eb=False):
    if eb: # nothing to do
        return params
    else:
        alpha = jnp.pi * alpha / 180.0 # convert to radians
        params = params.unfreeze()
        for k in params['params']:
            if 'readout' in k:
                continue
            if 'BatchNorm' in k:
                continue
            if 'bias' not in params['params'][k]: # then top down weights

                pre, post = k.split('-')
                bk = f'{post}-{pre}'
                if bk not in params['params']:
                    continue

                if len(params['params'][k]['kernel'].shape) == 2:
                    axes = (1, 0)
                elif len(params['params'][k]['kernel'].shape) == 4:
                    axes = (0, 1, 2, 3)

                transposed = jnp.transpose(params['params'][bk]['kernel'], axes=axes)

                if transposed.shape != params['params'][k]['kernel'].shape:
                    continue

                params['params'][k]['kernel'] = \
                    jnp.sin(alpha) * params['params'][k]['kernel'] + \
                    jnp.cos(alpha) * transposed

        return freeze(params)




def angle(params, eb=False):
    if eb: # nothing to do
        return 0.0
    else:
        keys = []
        # get pairs of layers for which angle is defined
        for k in params['params']:
            if 'readout' in k:
                continue
            if 'BatchNorm' in k:
                continue
            pre, post = k.split('-')
            bk = f'{post}-{pre}'
            # only ff layers have bias
            if 'bias' in params['params'][k] and bk in params['params']:
                keys.append((k, bk))

        if len(keys)>0:
            params = to_float32(params) # risk of overflow if float16
            inner_sum = 0.0
            sq_sum_1 = 0.0
            sq_sum_2 = 0.0

            for k, bk in keys:

                if len(params['params'][k]['kernel'].shape) == 2: # dense
                    axes = (1, 0)
                elif len(params['params'][k]['kernel'].shape) == 4: # conv
                    axes = (0, 1, 2, 3)
                transposed = jnp.transpose(params['params'][bk]['kernel'], axes=axes)

                if transposed.shape != params['params'][k]['kernel'].shape:
                    return jnp.nan # shape mismatch
                else:
                    inner_sum += jnp.sum(params['params'][k]['kernel'] * \
                                         transposed)
                    sq_sum_1 += jnp.sum(params['params'][k]['kernel']**2)
                    sq_sum_2 += jnp.sum(params['params'][bk]['kernel']**2)

            cos = inner_sum / jnp.sqrt(sq_sum_1 * sq_sum_2)
            cos = jnp.clip(cos, -1.0, 1.0) # numerical stability
            angle = jnp.arccos(cos) * (180.0/jnp.pi)
            return jnp.clip(angle, a_min=0.0, a_max=180.0)
        else:
            return jnp.nan


def batch_keys(key, bsz):
    key, *subkeys = random.split(key, num=bsz+1)
    subkeys = tree_map(lambda a: jnp.expand_dims(a, 0), subkeys)
    subkeys = jnp.concatenate(subkeys, axis=0)
    return key, subkeys


def _pmap_device_order():
    return local_devices()


def split_batch(batch):
    split = lambda x: x.reshape((local_device_count(), -1) + x.shape[1:])
    return tree_map(split, batch)

def unsplit_batch(batch):
    batch = device_get(batch) # ensure batch is on host
    unsplit = lambda x: x.reshape((-1,) + x.shape[2:])
    return tree_map(unsplit, batch)

def replicate(tree, devices=None):
    """Replicates arrays to multiple devices.
    Args:
    tree: a pytree containing the arrays that should be replicated.
    devices: the devices the data is replicated to
        (default: same order as expected by `jax.pmap()`).
    Returns:
    A new pytree containing the replicated arrays.
    """
    devices = devices or _pmap_device_order()
    return device_put_replicated(tree, devices)


def unreplicate(tree):
    return tree_map(lambda x: x[0], device_get(tree))
