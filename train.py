import jax
import argparse
import wandb
import sys
import jax.numpy as jnp
from jax.tree_util import tree_map
import time

from utils.data import get_data, get_iters
from models.utils import normalize_metrics, get_weights_as_images
from models.utils import VFTrainState, save_state, load_state
from models.utils import get_params, get_mlp_opt, get_cnn_opt, print_metrics
from utils.funcs import angle, cosine_sim, replicate
from utils.funcs import unreplicate, unsplit_batch, params_std

parser = argparse.ArgumentParser(description='Compare autodiff and holo_vf')

parser.add_argument('--nolog', action='store_true', help='stop log to wandb')
parser.add_argument('--name', type=str, default='no particular experiment')

parser.add_argument('--arch', type=str, default='')
parser.add_argument('--act', type=str, default='sigmoid')
parser.add_argument('--discrete', action='store_true')
parser.add_argument('--net', choices=['mlp', 'cnn'], default='mlp')
parser.add_argument('--local', action='store_true')
parser.add_argument('--rotate', action='store_true')
parser.add_argument('--angle', type=float, default=0.0)
parser.add_argument('--beta', type=float, default=0.01)
parser.add_argument('--T1', type=int, default=170)
parser.add_argument('--T2', type=int, default=50)
parser.add_argument('--N', type=int, default=0)

parser.add_argument('--dataset', choices=['fmnist', 'cifar10', 'cifar100', 'imagenet32'], default='fmnist')
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--parallel', action='store_true')
parser.add_argument('--half_prec', action='store_true')
parser.add_argument('--bsz', type=int, default=20)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--loss', choices=['mse', 'xent'], default='xent')

parser.add_argument('--noreset', action='store_true')
parser.add_argument('--jac_reg', action='store_true')
parser.add_argument('--jac_reg_coef', type=float, default=0.5)
parser.add_argument('--algo', choices=['hep_vf', 'bp', 'neu_rbp', 'rec_bp', 'online_hvf'], default='hep_vf')
parser.add_argument('--num_epochs', type=int, default=50)
parser.add_argument('--opt', choices=['adamw', 'adamw_cos', 'sgd', 'sgd_cos'], default='sgd')
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--mmt', type=float, default=0.0)
parser.add_argument('--wd', type=float, default=0.0)

parser.add_argument('--save_model', action='store_true')
parser.add_argument('--load_state', action='store_true')
parser.add_argument('--load_params', action='store_true')
parser.add_argument('--model_name', type=str, default='')
parser.add_argument('--load_epoch', type=int, default=0)

args = parser.parse_args()

if not args.nolog:
    wandb.init(project="", entity="REDACTED",
               name=args.name, config=args)
    wandb.config.update({'script_name': sys.argv[0]})
    print(f'\nRunning:\n\npython {" ".join(sys.argv)}\n')

jax.config.update("jax_default_device", jax.devices()[args.device]) 


train_ds, test_ds = get_data(args)

beta = args.beta
algo = args.algo
Ts = (args.T1, args.T2, args.N)
T1, T2, N = Ts

params, model = get_params(args)

if args.net == 'mlp':
    tx = get_mlp_opt(args)
elif args.net == 'cnn':
    tx = get_cnn_opt(args, params, len(train_ds))

state = VFTrainState.create(apply_fn=model.fwd, params=params, tx=tx,
                            last_steady=None, key=jax.random.PRNGKey(args.seed))

if args.load_state:
    state = load_state(state, args.model_name, args.load_epoch)

if not args.nolog:
    wandb.log({'angle': angle(state.params), 'epoch': args.load_epoch})


if args.parallel:
    state = replicate(state)
    train_step = jax.pmap(model.batch_train_step, axis_name='device',
                          static_broadcasted_argnums=(2,3,4))
    delta_rbp = jax.pmap(model.delta_rbp, static_broadcasted_argnums=4)
    du_dbeta = jax.pmap(model.du_dbeta, static_broadcasted_argnums=(4,5,6))
    eval_step = jax.pmap(model.eval_step, static_broadcasted_argnums=2)
else:
    train_step = model.batch_train_step
    delta_rbp = model.delta_rbp
    du_dbeta = model.du_dbeta
    eval_step = model.eval_step


for epoch in range(args.load_epoch + 1, args.num_epochs + 1):
    
    train_it, test_it = get_iters(train_ds, test_ds, args.parallel)

    other_metrics = params_std(state.params)

    start_time = time.perf_counter()

    # train loop
    for idx, train_batch in enumerate(train_it):

        state, metrics = train_step(state, train_batch, Ts, beta, algo)
        train_metrics = metrics if idx==0 else tree_map(jnp.add, train_metrics, metrics)
         
    epoch_time = time.perf_counter() - start_time
    start_time = time.perf_counter()

    # eval loop
    for idx, test_batch in enumerate(test_it):
        
        metrics = eval_step(state.params, test_batch, T1)
        test_metrics = metrics if idx==0 else tree_map(jnp.add, test_metrics, metrics)

    eval_time = time.perf_counter() - start_time

    if args.parallel: # gather and sum sharded metrics back to host
        train_metrics = tree_map(jnp.sum, jax.device_get(train_metrics))
        test_metrics = tree_map(jnp.sum, jax.device_get(test_metrics))

    train_metrics = normalize_metrics(train_metrics)
    test_metrics = normalize_metrics(test_metrics)

    metrics = {}
    last_steady = unreplicate(state.last_steady) if args.parallel else state.last_steady
    for k in last_steady:
        act_u = model.vf.act(last_steady[k][0, :])
        metrics[f'steady_{k}'] = tree_map(jnp.nan_to_num, act_u)

    for k in train_metrics:
        metrics[f'train_{k}'] = train_metrics[k]
    for k in test_metrics:
        metrics[f'test_{k}'] = test_metrics[k]
    for k in other_metrics:
        metrics[k] = other_metrics[k]

    metrics['epoch_time'] = epoch_time
    metrics['eval_time'] = eval_time
    metrics['epoch'] = epoch

    params_for_log = unreplicate(state.params) if args.parallel else state.params
    metrics['angle'] = angle(params_for_log)

    if (epoch-1)%max(args.num_epochs//10, 1) == 0 :
        images = get_weights_as_images(params_for_log['params'])
        for (k, im) in images:
            metrics[k] = im

    # comparing with RBP
    deltas, delta_convs = delta_rbp(state.params, state.last_steady, *train_batch, T2)
    du_db = du_dbeta(state.params, state.last_steady, *train_batch, T2, beta, N)

    if args.parallel:
        deltas = unsplit_batch(deltas)
        delta_convs = unsplit_batch(delta_convs)
        du_db = unsplit_batch(du_db)

    coses = tree_map(jax.vmap(cosine_sim), deltas, du_db)
    metrics['mean_delta_convs'] = jnp.mean(delta_convs)
    for k in coses:
        metrics[f'mean_cos_delta({k})'] = jnp.mean(jnp.nan_to_num(coses[k]))
        metrics[f'std_cos_delta({k})'] = jnp.std(jnp.nan_to_num(coses[k]))


    if not args.nolog:
        wandb.log(metrics)
    else:
        print_metrics(metrics)
    
    if args.save_model:
        to_save = unreplicate(state) if args.parallel else state
        save_state(to_save, args, epoch)
