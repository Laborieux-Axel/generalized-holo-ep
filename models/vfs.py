import jax.numpy as jnp
from jax import grad, vmap, value_and_grad
from jax.tree_util import tree_map
from jax.nn import log_softmax, softmax
from flax import linen as nn
from flax.core.frozen_dict import freeze
from typing import Callable, Mapping, Tuple

from utils.pool import SfmPool
from models.act import SimpleAct

USE_BIAS = True




#############################################
#        CLASSES FOR VECTOR FIELDS          #
#############################################


    

class mlp_discrete_vf(nn.Module):
    shapes: Mapping[str, Tuple[int]]
    sizes: Mapping[str, int]
    connections: Mapping[str, str]
    loss: str
    act: Callable[[jnp.ndarray], jnp.ndarray]
    eb: bool = False
    discrete: bool = True
    
    def setup(self):

        layers = {}
        for k, lname in self.connections.items():
            pre, post = k.split('-')
            if lname=='dense':
                layers[k] = nn.Dense(self.sizes[post], use_bias=USE_BIAS, name=k)
            elif lname=='dense_nobias':
                layers[k] = nn.Dense(self.sizes[post], use_bias=False, name=k)

        self.layers = freeze(layers)
        self.activate = SimpleAct(self.act)
        
    @nn.compact
    def __call__(self, u, x, y, beta, activate=True):
        
        if activate:
            act_u = self.activate(u)
        else:
            act_u = u

        l = {'in': x, **act_u}
        new_u = tree_map(jnp.zeros_like, u)
        
        for k in self.layers:
            pre, post = k.split('-')
            m = self.layers[k]
            new_u[post] = new_u[post] + m(l[pre])

        if self.loss=='xent':
            dLdu, out = softmax_readout(y.shape[-1], name='readout')(l['out'], y)
        elif self.loss=='mse':
            dLdu = (l['out'] - y)
            out = l['out']

        new_u['out'] = new_u['out'] + beta * dLdu
        vf = tree_map(jnp.subtract, new_u, u)
        
        return vf, out, act_u

    def __hash__(self):
        return id(self)



        ############################
        #          CONV NET        #
        ############################



class cnn_discrete_vf(nn.Module):
    shapes: Mapping[str, Tuple[int]]
    sizes: Mapping[str, int]
    loss: str
    act: Callable[[jnp.ndarray], jnp.ndarray]
    eb: bool = False
    discrete: bool = True
    local: bool = False
    
    def setup(self):

        conv = nn.ConvLocal if self.local else nn.Conv

        layers = {
            'in-conv1': conv(128, (3, 3), name='in-conv1'), 

            'conv1-conv2': conv(256, (3, 3), name='conv1-conv2'),
            'conv2-conv1': conv(256, (3, 3), name='conv2-conv1', use_bias=False),

            'conv2-conv3': conv(512, (3, 3), name='conv2-conv3'),
            'conv3-conv2': conv(512, (3, 3), name='conv3-conv2', use_bias=False),

            'conv3-out': conv(512, (3, 3), name='conv3-out', padding='VALID'),
            'out-conv3': conv(512, (3, 3), name='out-conv3', padding='VALID', use_bias=False)
        }

        self.layers = freeze(layers)
        self.pool = SfmPool((2,2,1), (2,2,1), padding='VALID', T=1.0, name='pool')
        self.activate = SimpleAct(self.act)
        
    @nn.compact
    def __call__(self, u, x, y, beta, activate=True):
        
        if activate:
            act_u = self.activate(u)
        else:
            act_u = u

        l = {'in': x, **act_u}
        new_u = tree_map(jnp.zeros_like, u)
        
        f = lambda post, pre, bk: jnp.sum(self.pool(self.layers[bk](post)) * pre)
        top_down = lambda post, pre, bk: grad(f)(post, pre, bk)

        new_u['conv1'] = self.pool(self.layers['in-conv1'](l['in'])) + \
                         top_down(l['conv1'], l['conv2'], 'conv2-conv1')
        
        new_u['conv2'] = self.pool(self.layers['conv1-conv2'](l['conv1'])) + \
                         top_down(l['conv2'], l['conv3'], 'conv3-conv2')
        
        new_u['conv3'] = self.pool(self.layers['conv2-conv3'](l['conv2'])) + \
                         top_down(l['conv3'], l['out'], 'out-conv3')
        
        new_u['out'] = self.pool(self.layers['conv3-out'](l['conv3']))

        if self.loss=='xent':
            flat_out = l['out'].reshape(y.shape[0], -1)
            dLdu, out = softmax_readout(y.shape[-1], name='readout')(flat_out, y)
        elif self.loss=='mse':
            dLdu = (l['out'] - y)
            out = l['out']

        new_u['out'] = new_u['out'] + beta * dLdu.reshape(l['out'].shape)
        vf = tree_map(jnp.subtract, new_u, u)
        
        return vf, out, act_u

    def __hash__(self):
        return id(self)


class softmax_readout(nn.Module):
    num_classes: int

    @nn.compact
    def __call__(self, x, y):
        holo = (x.dtype == jnp.complex64)
        def readout(x, y):
            logits = nn.Dense(self.num_classes)(x)
            loss = -jnp.sum(y*log_softmax(logits))
            return loss, logits
        return grad(readout, has_aux=True, holomorphic=holo)(x, y)
    
