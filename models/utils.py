import optax
import numpy as np
import jax.numpy as jnp
from jax import local_device_count
from jax.tree_util import tree_leaves, tree_map
import wandb
import diffrax
from flax.training.train_state import TrainState
from flax.training import checkpoints
from flax.core.frozen_dict import freeze
from typing import Mapping
import json
import math

from models.vfs import mlp_continuous_vf
from models.vfs import mlp_discrete_vf, mlp_discrete_ff_vf
from models.vfs import cnn_discrete_vf, cnn_discrete_mnist_vf
from models.dyn import ContinuousDynamics, DiscreteDynamics
from utils.funcs import pretty_print, get_activation, dalify, rotate


class VFTrainState(TrainState):
    last_steady: Mapping[str, jnp.DeviceArray]
    key: jnp.ndarray

    @classmethod
    def create(cls, *, apply_fn, params, tx, last_steady, key, **kwargs):
        """Creates a new instance with `step=0` and initialized `opt_state`."""
        opt_state = tx.init(params)
        return cls(
            step=0,
            apply_fn=apply_fn,
            params=params,
            tx=tx,
            opt_state=opt_state,
            last_steady=last_steady,
            key=key,
            **kwargs,
        )


def mask(p, key):
    mask = tree_map(lambda x: False, p.unfreeze())
    for k in p['params']:
        if k==key:
            mask['params'][k] = True
    return freeze(mask)

def create_cosine_sgd(lr, wd, mmt, num_epochs, num_batches):
    steps = num_epochs * num_batches
    lr_fn = optax.cosine_decay_schedule(init_value=lr, decay_steps=steps)
    opt = optax.inject_hyperparams(optax.sgd)(lr_fn, momentum=mmt)
    opt = optax.chain(optax.add_decayed_weights(wd), opt)
    return opt

def create_piecewise_adamw_cos(lr, wd, num_epochs, num_batches, start_epoch):
    steps = num_epochs * num_batches
    start_step = start_epoch * num_batches
    zero = optax.constant_schedule(0.0)
    cos_lr_fn = optax.cosine_decay_schedule(init_value=lr, decay_steps=(steps-start_step))
    lr_fn = optax.join_schedules(schedules=[zero, cos_lr_fn], boundaries=[start_step])
    opt = optax.adamw(learning_rate=lr_fn, weight_decay=wd)
    return opt

def create_adamw_cos(lr, wd, num_epochs, num_batches):
    steps = num_epochs * num_batches
    lr_fn = optax.cosine_decay_schedule(init_value=lr, decay_steps=steps)
    opt = optax.adamw(learning_rate=lr_fn, weight_decay=wd)
    return opt

def create_adamw(lr, wd):
    opt = optax.adamw(learning_rate=lr, weight_decay=wd)
    return opt

def get_mlp_opt(args):
    
    if args.opt == 'sgd':
        tx = optax.sgd(args.lr, args.mmt)
        tx = optax.chain(optax.add_decayed_weights(args.wd), tx)
    elif args.opt == 'adamw':
        tx = optax.adamw(learning_rate=args.lr, weight_decay=args.wd)
    else:
        raise NotImplementedError
    return tx

def get_cnn_opt(args, params, num_batches):
    
    if args.opt == 'sgd':
        opt = optax.sgd(args.lr, args.mmt)
        opt = optax.chain(optax.add_decayed_weights(args.wd), opt)
    elif args.opt =='sgd_cos':

	    lrs = {'in-conv1':    25*args.lr,
    		   'conv1-conv2': 15*args.lr,
    		   'conv2-conv1': 15*args.lr,
    		   'conv2-conv3': 10*args.lr,
    		   'conv3-conv2': 10*args.lr,
    		   'conv3-out':    8*args.lr,
    		   'out-conv3':    8*args.lr,
    		   'readout':      5*args.lr}
    	
    	wds = {'in-conv1':    args.wd,
    		   'conv1-conv2': args.wd,
    		   'conv2-conv1': args.wd,
    		   'conv2-conv3': args.wd,
    		   'conv3-conv2': args.wd,
    		   'conv3-out':   args.wd,
    		   'out-conv3':   args.wd,
    		   'readout':     2*args.wd}



        chain = tuple(optax.masked(create_cosine_sgd(lrs[k], wds[k], args.mmt,
                                                    args.num_epochs, num_batches),
                                   mask(params, k)) for k in params['params'])

        opt = optax.chain(*chain)

    elif args.opt == 'adamw':
        
	    lrs = {'in-conv1':     1.0*args.lr,
    			'conv1-conv2': 0.8*args.lr,
    			'conv2-conv1': 0.8*args.lr,
    			'conv2-conv3': 0.5*args.lr,
    			'conv3-conv2': 0.5*args.lr,
    			'conv3-out':   0.2*args.lr,
    			'out-conv3':   0.2*args.lr,
    			'readout':     0.1*args.lr}
    	
    	wds = {'in-conv1':     args.wd,
    			'conv1-conv2': args.wd,
    			'conv2-conv1': args.wd,
    			'conv2-conv3': args.wd,
    			'conv3-conv2': args.wd,
    			'conv3-out':   args.wd,
    			'out-conv3':   args.wd,
    			'readout':     args.wd}



        chain = tuple(optax.masked(create_adamw(lrs[k], wds[k]),
                                   mask(params, k)) for k in params['params'])

        opt = optax.chain(*chain)

    elif args.opt == 'adamw_cos':

        lrs = {'in-conv1':     3*args.lr,
                'conv1-conv2': 2*args.lr,
                'conv2-conv1': 2*args.lr,
                'conv2-conv3': 1.5*args.lr,
                'conv3-conv2': 1.5*args.lr,
                'conv3-out':   args.lr,
                'out-conv3':   args.lr,
                'readout':     args.lr}
        
        wds = {'in-conv1':     args.wd,
                'conv1-conv2': args.wd,
                'conv2-conv1': args.wd,
                'conv2-conv3': args.wd,
                'conv3-conv2': args.wd,
                'conv3-out':   args.wd,
                'out-conv3':   args.wd,
                'readout':     10*args.wd}

        chain = tuple(optax.masked(create_adamw_cos(lrs[k], wds[k], args.num_epochs, num_batches),
                                   mask(params, k)) for k in params['params'])
        opt = optax.chain(*chain)

    return opt

def get_params(args):

    if args.net == 'mlp':
        params, model = select_mlp(args)
    elif args.net == 'cnn':
        params, model = select_cnn(args)
    else:
        raise NotImplementedError

    if args.rotate:
        params = rotate(params, alpha=args.angle)

    if args.wconstraint == 'dale':
        params = dalify(params)

    if args.load_params:
        params = load_params(args.model_name, args.load_epoch)

    return params, model


def select_mlp(args):

    with open(f'arch/{args.arch}.json') as f:
        arch = json.load(f)

    pretty_print(arch)

    in_size = arch['in_size']   # int
    out_size = arch['out_size'] # int

    shapes = {k: tuple(v) for k, v in arch['shapes'].items()}
    sizes = {k: math.prod(v) for k, v in shapes.items()}
    tot_size = sum([p for p in tree_leaves(sizes)])
    connections = arch['connections'] # dict

    loss = args.loss
    seed = args.seed
    act = get_activation(args.act)

    if hasattr(args, 'noreset'):      # only for train script
        noreset = args.noreset
    else:
        noreset = False

    if hasattr(args, 'jac_reg'):      # only for train script
        jac_reg = args.jac_reg
        jac_reg_coef = args.jac_reg_coef
    else:
        jac_reg = False
        jac_reg_coef = 0.0


    half_prec = hasattr(args, 'half_prec') and args.half_prec

    dum_x, dum_y = jnp.ones((1, in_size)), jnp.zeros((1, out_size))

	vf = mlp_discrete_vf(shapes=shapes, sizes=sizes, 
							connections=connections,
							loss=loss, act=act)

    jac_safe = tot_size < 1e3
    model = DiscreteDynamics(vf=vf, seed=seed, noreset=noreset,
    						 wconstraint=wconstraint, jac_safe=jac_safe,
    						 half_prec=half_prec, jac_reg=jac_reg,
    						 jac_reg_coef=jac_reg_coef)



    params = model.init_params(dum_x, dum_y)

    return params, model


def select_cnn(args):

    if args.dataset in ['cifar10', 'cifar100', 'imagenet32']:
        in_size = (1, 32, 32, 3)

        shapes = {'conv1': (16, 16, 128),
                'conv2': (8, 8, 256),
                'conv3': (4, 4, 512),
                'out': (1, 1, 512)}


    pretty_print(shapes)
    sizes = {k: math.prod(v) for k, v in shapes.items()}
    tot_size = sum([p for p in tree_leaves(sizes)])

    if args.dataset == 'cifar10':
        out_size = (1, 10)
    elif args.dataset == 'cifar100':
        out_size = (1, 100)
    elif args.dataset == 'imagenet32':
        out_size = (1, 1000)

    if hasattr(args, 'jac_reg'):      # only for train script
        jac_reg = args.jac_reg
        jac_reg_coef = args.jac_reg_coef
    else:
        jac_reg = False
        jac_reg_coef = 0.0

    loss = args.loss
    seed = args.seed
    act = get_activation(args.act)

    if args.parallel:
        n_devices = local_device_count()
    else:
        n_devices = 1

    vf = cnn_discrete_vf(shapes=shapes, sizes=sizes, loss=loss,
    					act=act, local=args.local)


    jac_safe = tot_size < 1e3
    model = DiscreteDynamics(vf=vf, seed=seed, noreset=False, wconstraint='none',
                             jac_safe=jac_safe, n_devices=n_devices,
                             half_prec=args.half_prec, jac_reg=jac_reg,
                             jac_reg_coef=jac_reg_coef)

    dum_x, dum_y = jnp.ones(in_size), jnp.zeros(out_size)
    params = model.init_params(dum_x, dum_y)

    return params, model


def print_metrics(metrics):
    """prints metrics"""
    keys = ['epoch', 'train_acc', 'test_acc', 'train_conv', 'test_conv', 'angle']
    print(' | '.join([f'{k}: {metrics[k]:.3f}' for k in keys]))
    print()

def normalize_metrics(metrics):
    """assumes one of the key is the size"""
    for k in ['loss', 'conv', 'jac_sym', 'acc', 'top5']:
        metrics[k] = metrics[k] / metrics['size']
    if 'homeo_loss' in metrics:
        metrics['homeo_loss'] = metrics['homeo_loss'] / metrics['size']
    return metrics

def make_pixels(w, thresh=1e-4):
    """converts weights to images for logging"""
    w = np.array(w)
    if len(w.shape) == 4: # conv layer -> take first filter
        w = w[:,:,0,0]
    if w.shape[0]>w.shape[1]:
        w = w.T
    w = np.expand_dims(w, axis=2)
    w = np.repeat(w, 3, axis=2) 

    absn_w = np.abs(w)/np.max(np.abs(w))
    pixels = np.ones_like(w) # white background

    # subtract the positive weights magnitude from GB channels 
    # to interpolate between white and red
    pixels[:,:,1:] = np.where(w[:,:,1:] > thresh,
                              pixels[:,:,1:] - absn_w[:,:,1:], 
                              pixels[:,:,1:])

    # subtract the negative weights magnitude from RG channels 
    # to interpolate between white and blue
    pixels[:,:,:-1] = np.where(w[:,:,:-1] < -thresh,
                               pixels[:,:,:-1] - absn_w[:,:,:-1],
                               pixels[:,:,:-1])

    return pixels

def get_weights_as_images(params, current_key='params'):
    """converts weights to images for logging"""
    images = []
    if 'kernel' in params:
        pixels = make_pixels(params['kernel'])
        image = wandb.Image(pixels, caption=current_key)
        images.append((current_key, image))
    else:
        for k in params:
            images = images + get_weights_as_images(params[k], current_key=k)
    return images



def save_state(state, args, epoch):
    ckpt = {'model': state}
    checkpoints.save_checkpoint(ckpt_dir=f'checkpoints/{args.name}', target=ckpt,
                                step=epoch, overwrite=False, keep=2)


def load_state(state, name, epoch):
    target = {'model': state}
    ckpt = checkpoints.restore_checkpoint(f'checkpoints/{name}', target=target, step=epoch)
    state = ckpt['model']
    return state


def load_params(name, epoch):
    ckpt = checkpoints.restore_checkpoint(f'checkpoints/{name}', target=None, step=epoch)
    state = ckpt['model']
    return freeze(state['params'])
