from itertools import accumulate
from functools import partial
from typing import Tuple, Optional

import jax
from jax import grad, vmap, jacfwd, jit, vjp, jvp, random, value_and_grad
import jax.numpy as jnp
from jax.nn import log_softmax
from jax.tree_util import tree_map, tree_leaves, tree_flatten, tree_unflatten
from jax.lax import pmean, psum, top_k, fori_loop

import flax
from flax import linen as nn
from flax.core.frozen_dict import freeze

from utils.funcs import dict_cossim, to_complex, cosine_sim, to_float16, to_float32
from utils.funcs import dalify, ff_exc_fb_inh, batch_keys, concat_flat_leaves




class DiscreteDynamics(flax.struct.PyTreeNode):
    """general class for running, differentiating, and training a discrete
    dynamical system where the underlying system is abstracted away"""
    vf: nn.Module
    seed: int
    noreset: bool = False
    jac_safe: bool = False
    n_devices: int = 1
    half_prec: bool = False
    jac_reg: bool = False
    jac_reg_coef: float = 0.0

    #############################################
    #               PARAMS INIT                 #
    #############################################
        
    def init_params(self, x, y):
        u = self.batch_nrn(x)
        key = random.PRNGKey(self.seed)
        params = self.vf.init(key, u, x, y, 0.0)
        return params

    
    
    #############################################
    #           MANIPULATING NEURONS            #
    #############################################
    
    def init_nrn(self, x):
        f = lambda a, b: jnp.zeros(a).reshape(b)
        return tree_map(f, self.vf.sizes, self.vf.shapes)
    
    def batch_nrn(self, batch_x):
        return vmap(self.init_nrn)(batch_x)
    
    def concat_nrn(self, u):
        return concat_flat_leaves(u)
    
    def split_nrn(self, u):
        indices = tuple(accumulate(tree_leaves(self.vf.sizes)))
        _, treedef = tree_flatten(self.vf.sizes)
        leaves = jnp.split(u, indices, axis=-1)[:-1]
        new_u = tree_unflatten(treedef, leaves)
        new_u = tree_map(lambda a, b: a.reshape(b), new_u, self.vf.shapes)
        return new_u
    
    def out_nrn(self, params, u, x, y, beta):
        _, out, _ = self.vf.apply(params, u, x, y, beta)
        return out
    
    
    #############################################
    #           RUNNING THE DYNAMICS            #
    #############################################
    
    
    def vfield(self, params, u, x, y, beta):
        vf, _, _ = self.vf.apply(params, u, x, y, beta)
        return vf
    
    def vfield_no_act(self, params, u, x, y, beta):
        vf, _, _ = self.vf.apply(params, u, x, y, beta, activate=False)
        return vf
    
    def activate(self, params, u, x, y, beta):
        _, _, act_u = self.vf.apply(params, u, x, y, beta)
        return act_u

    @partial(jax.jit, static_argnums=(0,6))
    def fwd(self, params, u0, x, y, beta, T):
            
        u = u0
        for _ in range(T):
            vf = self.vfield(params, u, x, y, beta)
            u = tree_map(jnp.add, vf, u) # discrete integration
    
        return u
    
    @partial(jax.jit, static_argnums=(0,6))
    def fwd_f16(self, params, u0, x, y, beta, T):
        (params, u0, x, y) = to_float16((params, u0, x, y))
        u1 = self.fwd(params, u0, x, y, beta, T)
        return to_float32(u1)
    
    def online_fwd(self, params, u, x, y, beta, T1, T2, dt):

        def body_fun(i, val):
            u, avg_u, amp_u = val
            cbeta = beta * jnp.exp(2 * jnp.pi * 1j * dt * i)
            u = self.fwd(params, u, x, y, cbeta, T2)
            avg_u = tree_map(lambda a, b: a + b, avg_u, u)
            act_u = self.activate(params, u, x, y, cbeta)
            amp_u = tree_map(lambda a, b: a + b/cbeta, amp_u, act_u)
            return u, avg_u, amp_u

        (params, u, x, y) = to_complex((params, u, x, y))

        init_val = (u, u, u)
        u, avg_u, amp_u = fori_loop(0, T1, body_fun, init_val)

        avg_u = tree_map(lambda a: jnp.real(a) / T1, avg_u)
        amp_u = tree_map(lambda a: jnp.real(a) / T1, amp_u)

        return avg_u, amp_u

    def eval_conv(self, params, u, x, y, beta):
        vf = self.vfield(params, u, x, y, beta)
        vf = concat_flat_leaves(vf)
        return jnp.log10(jnp.linalg.norm(vf) + 1e-8)
    
    def batch_eval_conv(self, *args):
        vf = self.vfield(*args)
        vf = vmap(concat_flat_leaves)(vf)
        return jnp.log10(jnp.linalg.norm(vf, axis=1) + 1e-8)
    
    def batch_time_fwd(self, params, batch, T):

        t = 0
        x, y = batch
        u = self.batch_nrn(x)
        convs = self.batch_eval_conv(params, u, x, y, 0.0)
        while jnp.max(convs) > -5.5 and t < T:
            u = self.fwd(params, u, x, y, 0.0, 1)
            convs = self.batch_eval_conv(params, u, x, y, 0.0)
            t += 1
        
        return t
    
    def fwd_fractal(self, params, u0, x, y, beta, T, batched):
    
        u = self.fwd(params, u0, x, y, beta, T)
        vf = self.vfield(params, u, x, y, beta)
        vf = vmap(concat_flat_leaves)(vf) if batched else concat_flat_leaves(vf)
        l2 = jnp.nan_to_num(jnp.linalg.norm(vf, axis=-1))

        return jnp.mean(l2)
    
    @partial(jit, static_argnums=(0,6,7))
    def batch_fwd_fractal(self, *args):
        axes = (None, None, None, None, 0, None, None)
        return vmap(self.fwd_fractal, in_axes=axes)(*args)

    def trajectory(self, params, u0, x, y, beta, T):
        if self.half_prec:
            (params, u0, x, y) = to_float16((params, u0, x, y))
        u = u0
        log_l2s = []
        ts = jnp.arange(1, T+1)
        for t in range(T):
            vf = self.vfield(params, u, x, y, beta)
            flat_vf = vmap(concat_flat_leaves)(to_float32(vf))
            log_l2 = jnp.log10(jnp.linalg.norm(flat_vf, axis=-1) + 1e-8)
            log_l2s.append(log_l2.reshape(1, -1))
            u = tree_map(jnp.add, vf, u)

        return u, ts, jnp.concatenate(log_l2s)

    #############################################
    #             JACOBIAN ANALYSIS             #
    #############################################
    
    def vf_jac(self, params, u, x, y, beta):

        def f(v):
            v = self.split_nrn(v)
            vf = self.vfield_no_act(params, v, x, y, beta)
            return self.concat_nrn(vf)
        
        u = self.concat_nrn(self.activate(params, u, x, y, beta))
        return jacfwd(f)(u)

    def batch_vf_jac(self, *args):
        in_axes = (None, 0, 0, 0, None)
        return vmap(self.vf_jac, in_axes=in_axes)(*args)
    
    def jac_sym(self, *args):
        
        if self.jac_safe:
            J = self.vf_jac(*args)
            J_sym = 0.5 * (J + J.T)
            J_ant = 0.5 * (J - J.T)
            
            s = jnp.linalg.norm(J_sym)
            a = jnp.linalg.norm(J_ant)
            
            return s/(s+a)
        else:
            return jnp.nan
    
    def batch_jac_sym(self, *args):
        in_axes = (None, 0, 0, 0, None)
        return vmap(self.jac_sym, in_axes=in_axes)(*args)
    
    def jac_evs(self, params, u, x, y, beta):
        J = self.vf_jac(params, u, x, y, beta)
        evs = jnp.linalg.eigvals(J)
        return evs
        
    def batch_jac_evs(self, *args):
        in_axes = (None, 0, 0, 0, None)
        return vmap(self.jac_evs, in_axes=in_axes)(*args)
    

    #############################################
    #           GRADIENT COMPUTATION            #
    #############################################

    def loss(self, out, y):
        if self.vf.loss=='mse':
            loss = 0.5 * jnp.sum(jnp.power(out - y, 2))
        elif self.vf.loss=='xent':
            loss = -jnp.sum(y*log_softmax(out))
        return loss
    
    def dLdu(self, params, u, x, y):
        
        def f(*args):
            out = self.out_nrn(*args, 0.0)
            return self.loss(out, y)

        return grad(f, argnums=1)(params, u, x, y)

    def bp_grad(self, params, u, x, y, T):
        
        def graph(params, u, x, y, t):
            u1 = self.fwd(params, u, x, y, 0.0, t)
            out = self.out_nrn(params, u1, x, y, 0.0)
            loss = self.loss(out, y)/y.shape[0]
            return loss, (out, u1)
        
        g, aux = grad(graph, has_aux=True)(params, u, x, y, T)
        g = tree_map(jnp.nan_to_num, g)
        out, u1 = aux
        
        return g, (out, u1)
    
    def readout_grad(self, params, u, x, y):
        
        def f(*args):
            out = self.out_nrn(*args, 0.0)
            return self.loss(out, y)/y.shape[0]
        
        return grad(f)(params, u, x, y)
    
    @partial(jit, static_argnums=(0,5,7))
    def du_dbeta(self, params, u1, x, y, T, beta, N):
        
        if N==0:
            return self.du_dbeta_autodiff(params, u1, x, y, T)

        u = self.fwd(params, u1, x, y, beta, T)
        au = self.activate(params, u, x, y, beta)

        if N==1:
            au1 = self.activate(params, u1, x, y, 0.0)
            du_dbeta = tree_map(lambda a, b: (a-b)/beta, au, au1)
        else:
            du_dbeta = tree_map(lambda a: a/beta, au)

            if N%2==0:
                u = self.fwd(params, u1, x, y, -beta, T)
                au = self.activate(params, u, x, y, -beta)
                du_dbeta = tree_map(lambda a, b: a-b/beta, du_dbeta, au)

            if N>2:
                params = to_complex(params)
                u1 = to_complex(u1)
                x = x.astype(jnp.complex64)

            for t in range(1, (N+1)//2): 

                cbeta = beta * jnp.exp((2 * 1j * jnp.pi * t)/N)
                u = self.fwd(params, u1, x, y, cbeta, T)
                au = self.activate(params, u, x, y, cbeta)
                au = tree_map(jnp.nan_to_num, au)
                du_dbeta = tree_map(lambda a, b: a+2*jnp.real(b/cbeta), du_dbeta, au)

            du_dbeta = tree_map(lambda a: a/N, du_dbeta)
  
        return du_dbeta
    
    @partial(jit, static_argnums=(0,5))
    def du_dbeta_autodiff(self, params, u1, x, y, T):
        def f(params, u, x, y, beta, T):
            u = self.fwd(params, u, x, y, beta, T)
            return self.activate(params, u, x, y, beta)
        du_dbeta = jacfwd(f, argnums=4)(params, u1, x, y, 0.0, T)
        return tree_map(jnp.nan_to_num, du_dbeta)

    def hep_vf(self, params, u1, x, y, T, beta, N):

        du_dbeta = self.du_dbeta(params, u1, x, y, T, beta, N)

        F_of_w = lambda w: self.vfield(w, u1, x, y, 0.0)
        _, dFdwT = vjp(F_of_w, params)
        g, = dFdwT(du_dbeta)
        g = tree_map(lambda a: a/x.shape[0], g)
        g = tree_map(jnp.add, g, self.readout_grad(params, u1, x, y))

        return g

    def online_grad(self, params, avg_u, amp_u, x, y):

        F_of_w = lambda w: self.vfield(w, avg_u, x, y, 0.0)
        _, dFdwT = vjp(F_of_w, params)
        g, = dFdwT(amp_u)
        g = tree_map(lambda a: a/x.shape[0], g)
        g = tree_map(jnp.add, g, self.readout_grad(params, avg_u, x, y))

        return g

    @partial(jit, static_argnums=(0,5))
    def delta_rbp(self, params, u1, x, y, T):
        
        dLdu1 = self.dLdu(params, u1, x, y)
        delta = tree_map(jnp.zeros_like, dLdu1)
        F_of_u = lambda u: self.vfield(params, u, x, y, 0.0)
        _, dFduT = vjp(F_of_u, u1)

        for _ in range(T):
            dFduT_delta, = dFduT(delta)
            dFduT_delta = tree_map(jnp.nan_to_num, dFduT_delta)
            update = tree_map(jnp.add, dFduT_delta, dLdu1)
            delta = tree_map(jnp.add, delta, update)
        
        conv = jnp.log10(jnp.linalg.norm(vmap(concat_flat_leaves)(update), axis=-1)+1e-9)
        return delta, conv

    def rbp(self, params, u1, x, y, T):

        delta, _ = self.delta_rbp(params, u1, x, y, T)

        F_of_w = lambda w: self.vfield(w, u1, x, y, 0.0)
        _, dFdwT = vjp(F_of_w, params)
        g, = dFdwT(delta)
        g = tree_map(lambda a: a/x.shape[0], g)
        g = tree_map(jnp.add, g, self.readout_grad(params, u1, x, y))

        return g

    @partial(jit, static_argnums=(0,5))
    def delta_neumann_rbp(self, params, u1, x, y, T):
        
        dLdu1 = self.dLdu(params, u1, x, y)
        v = dLdu1
        delta = dLdu1
        F_of_u = lambda u: self.vfield(params, u, x, y, 0.0)
        _, dFduT = vjp(F_of_u, u1)

        for _ in range(T):
            dFduT_v, = dFduT(v)
            dFduT_v = tree_map(jnp.nan_to_num, dFduT_v)
            v = tree_map(jnp.add, dFduT_v, v)
            delta = tree_map(jnp.add, delta, v)
        
        conv = jnp.log10(jnp.linalg.norm(vmap(concat_flat_leaves)(v), axis=-1)+1e-9)
        return delta, conv

    def neumann_rbp(self, params, u1, x, y, T):

        delta, _ = self.delta_neumann_rbp(params, u1, x, y, T)

        F_of_w = lambda w: self.vfield(w, u1, x, y, 0.0)
        _, dFdwT = vjp(F_of_w, params)
        g, = dFdwT(delta)
        g = tree_map(lambda a: a/x.shape[0], g)
        g = tree_map(jnp.add, g, self.readout_grad(params, u1, x, y))

        return g

    def homeo_loss(self, params, u1, x, y, key):
        
        eps = tree_map(jnp.zeros_like, u1) 
        for k in eps:
            key, subkey = random.split(key)
            eps[k] = random.normal(subkey, eps[k].shape)

        au1 = self.activate(params, u1, x, y, 0.0)
        F_of_u = lambda u: self.vfield_no_act(params, u, x, y, 0.0)

        _, J_eps = jvp(F_of_u, (au1,), (eps,))
        _, J2_eps = jvp(F_of_u, (au1,), (J_eps,))

        eps = vmap(concat_flat_leaves)(eps)
        J_eps = vmap(concat_flat_leaves)(J_eps)
        J2_eps = vmap(concat_flat_leaves)(J2_eps)

        loss = (jnp.sum(jnp.square(J_eps), axis=1) - \
                 vmap(jnp.inner)(eps, J2_eps))/J_eps.shape[-1]

        return jnp.mean(loss)
    
    def batch_homeo_loss(self, *args):
        in_axes = (None, None, None, None, 0)
        f = lambda *args: pmean(self.homeo_loss(*args), axis_name='j')
        return vmap(f, in_axes=in_axes, out_axes=None, axis_name='j')(*args)

    def homeo_grad(self, *args):
        homeo_loss, g = value_and_grad(self.batch_homeo_loss)(*args)
        g = tree_map(lambda a: a * self.jac_reg_coef, g)
        return tree_map(jnp.nan_to_num, g), homeo_loss
    
    def stoch_homeo_grad(self, params, u1, x, y, keys):
        operands = (params, u1, x, y, keys)
        no_grad = lambda *args: freeze(tree_map(jnp.zeros_like, args[0].unfreeze()))
        cond = random.bernoulli(keys[0], 0.4)
        return jax.lax.cond(cond, self.homeo_grad, no_grad, *operands)

    def sweep_beta(self, params, u1, x, y, T, betas, N, g):
        
        args = (params, u1, x, y, T, betas, N, g)
        in_axes = (None, None, None, None, None, 0, None, None)
        
        def f(*args):
            true_g = args[-1]
            est = self.hep_vf(*args[:-1])
            true_g = true_g.unfreeze()
            est = est.unfreeze()
            true_g['params'].pop('readout', None)
            est['params'].pop('readout', None)
            out = {}
            for k in est['params']:
                out[k] = cosine_sim(est['params'][k]['kernel'], 
                                    true_g['params'][k]['kernel'])
            out['total'] = dict_cossim(est, true_g)
            return out
        
        return vmap(f, in_axes=in_axes)(*args)
    
    def sweep_beta_delta(self, params, u1, x, y, T, betas, N):
        
        args = (params, u1, x, y, T, betas, N)
        in_axes = (None, None, None, None, None, 0, None)
        
        def f(*args):
            deltas, _ = self.delta_rbp(*args[:-2])
            du_dbetas = self.du_dbeta(*args)
            out = {}
            for k in deltas:
                out[k] = jnp.mean(vmap(cosine_sim)(deltas[k], du_dbetas[k]))
            out['total'] = jnp.mean(vmap(dict_cossim)(deltas, du_dbetas))
            return out
        
        return vmap(f, in_axes=in_axes)(*args)

    #############################################
    #                 TRAINING!                 #
    #############################################
    
    @partial(jit, static_argnums=(0,3,5))
    def batch_train_step(self, state, batch, Ts, beta, algo):
        
        T1, T2, N = Ts
        x, y = batch
        if self.noreset:
            u = state.last_steady
        else:
            u = self.batch_nrn(x)
        
        fwd = self.fwd_f16 if self.half_prec else self.fwd

        if algo=='bp':
            u = fwd(state.params, u, x, y, 0.0, T1)
            out = self.out_nrn(state.params, u, x, y, 0.0)
            grads, _ = self.bp_grad(state.params, u, x, y, T2)
        elif algo=='hep_vf':
            u = fwd(state.params, u, x, y, 0.0, T1)
            out = self.out_nrn(state.params, u, x, y, 0.0)
            grads = self.hep_vf(state.params, u, x, y, T2, beta, N)
        elif algo=='rec_bp':
            u = fwd(state.params, u, x, y, 0.0, T1)
            out = self.out_nrn(state.params, u, x, y, 0.0)
            grads = self.rbp(state.params, u, x, y, T2)
        elif algo=='neu_rbp':
            u = self.fwd(state.params, u, x, y, 0.0, T1)
            out = self.out_nrn(state.params, u, x, y, 0.0)
            grads = self.neumann_rbp(state.params, u, x, y, T2)
        elif algo=='online_hvf':
            dt = 1/N
            q = T1//T2
            u, amp_u = self.online_fwd(state.params, u, x, y, beta, q, T2, dt)
            out = self.out_nrn(state.params, u, x, y, 0.0)
            grads = self.online_grad(state.params, u, amp_u, x, y)
        
        metrics = self.batch_metrics(state.params, u, x, y, out)

        if self.jac_reg:
            key, subkeys = batch_keys(state.key, 5)
            reg_grads, homeo_loss = self.homeo_grad(state.params, u, x, y, subkeys)
            grads = tree_map(jnp.add, grads, reg_grads)
            state = state.replace(key=key)
            metrics['homeo_loss'] = homeo_loss * x.shape[0]

        if self.n_devices>1:
            grads = pmean(grads, axis_name='device')

        state = state.apply_gradients(grads=grads)
        state = state.replace(last_steady=u)
        
        return state, metrics
    
    @partial(jit, static_argnums=0)
    def batch_metrics(self, params, u, x, y, out):
        
        metrics = {'size': out.shape[0]}

        conv = self.batch_eval_conv(params, u, x, y, 0.0)
        metrics['conv'] = jnp.sum(conv)

        sub_u = tree_map(lambda a: a[:10], u)
        sub_x, sub_y = x[:10], y[:10]
        jac_sym = self.batch_jac_sym(params, sub_u, sub_x, sub_y, 0.0)
        metrics['jac_sym'] = jnp.mean(jac_sym) * x.shape[0]
        
        labels = jnp.argmax(y, -1)
        
        metrics['loss'] = self.loss(out, y)
        
        pred1 = jnp.argmax(out, -1)
        metrics['acc'] = jnp.sum(jnp.equal(pred1, labels))
        
        pred5 = top_k(out, 5)[1] 
        corr5 = jnp.sum(jnp.equal(pred5, labels.reshape(-1,1)).any(axis=1))
        metrics['top5'] = corr5
        
        return metrics
    
    @partial(jit, static_argnums=(0,3))
    def eval_step(self, params, batch, T):
        x, y = batch
        u = self.batch_nrn(x)
        fwd = self.fwd_f16 if self.half_prec else self.fwd
        u = fwd(params, u, x, y, 0.0, T)
        out = self.out_nrn(params, u, x, y, 0.0)        
        return self.batch_metrics(params, u, x, y, out)


