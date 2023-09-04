import jax
import jax.numpy as jnp
from jax.nn import one_hot
from flax.jax_utils import prefetch_to_device
import tensorflow as tf
import tensorflow_datasets as tfds
import json

from utils.funcs import split_batch


# Ensure TF does not see GPU and grab all GPU memory.
tf.config.set_visible_devices([], device_type='GPU') 






def get_data(args):

    if args.dataset == 'fmnist':
        train_ds, _, test_ds = tfds_mnist_mlp(args.bsz, args.seed)
    elif args.dataset in ['cifar10', 'cifar100', 'imagenet32']:
        assert args.net == 'cnn', f'{args.dataset} only implemented with CNN'
        train_ds, test_ds = get_tfds_loaders(args.dataset, args.bsz, args.seed)
    else:
        raise NotImplementedError(f'Dataset {args.dataset} not implemented')

    return train_ds, test_ds


def get_iters(train_ds, test_ds, parallel):
    if parallel:
        train_it, test_it = map(split_batch, train_ds), map(split_batch, test_ds)
        train_it = prefetch_to_device(train_it, 2)
        test_it = prefetch_to_device(test_it, 2)
    else:
        train_it, test_it = iter(train_ds), iter(test_ds)
    return train_it, test_it







def tfds_mnist_mlp(batch_size, seed):

    tf.random.set_seed(seed)
    data_dir = './datasets'
    num_pixels = 784
    ds_name = 'fashion_mnist'
    
    def data_transform(x, y):
        x = x/255
        x = tf.reshape(x, (len(x), num_pixels))
        y = tf.one_hot(y, 10)
        return x, y

    train_ds = tfds.load(name=ds_name,
                         split='train',
                         as_supervised=True,
                         data_dir=data_dir)

    train_ds = (
        train_ds
        .shuffle(len(train_ds))
        .batch(batch_size)
        .map(data_transform)
        .prefetch(1)
    )
    
    fulltrain_ds = tfds.load(name=ds_name,
                             split='train',
                             as_supervised=True,
                             data_dir=data_dir)

    fulltrain_ds = (
        fulltrain_ds
        .batch(10000)
        .map(data_transform)
        .prefetch(1)
    )

    test_ds = tfds.load(name=ds_name,
                        split='test',
                        as_supervised=True,
                        data_dir=data_dir)

    test_ds = (
        test_ds
        .batch(10000)
        .map(data_transform)
        .prefetch(1)
    )

    train_ds = tfds.as_numpy(train_ds)
    fulltrain_ds = tfds.as_numpy(fulltrain_ds)
    test_ds = tfds.as_numpy(test_ds)

    return train_ds, fulltrain_ds, test_ds






def get_tfds_loaders(task, batch_size, seed):

    tf.random.set_seed(seed)
    data_dir = './datasets/tfds'
    
    if task=='cifar10':

        rng = tf.random.Generator.from_seed(seed, alg='philox')

        def normalize(image, label):
            image = tf.cast(image, tf.float32)
            image = tf.divide(image, 255)
            mean = tf.constant([0.4914, 0.4822, 0.4465])
            mean = tf.reshape(mean, (1,1,3))
            std = tf.constant([1.5*0.2023, 1.5*0.1994, 1.5*0.2010])
            std = tf.reshape(std, (1,1,3))
            image = (image - mean)/std
            label = tf.one_hot(label, 10)
            return image, label
        
        def augment(image_label, seed):
            image, label = image_label
            image = tf.image.stateless_random_flip_left_right(image, seed=seed)
            image = tf.image.resize_with_crop_or_pad(image, 36, 36)
            image = tf.image.stateless_random_crop(image, size=[32,32,3], seed=seed)
            return image, label 

        def rnd_aug(x, y):
            seed = rng.make_seeds(2)[0]
            image, label = augment((x, y), seed)
            return image, label


        train_ds = tfds.load(name='cifar10', split='train',
                             as_supervised=True, data_dir=data_dir)

        train_ds = (
            train_ds
            .shuffle(len(train_ds))
            .map(rnd_aug)
            .map(normalize)
            .batch(batch_size, drop_remainder=True)
            .prefetch(1)
        )

        train_ds = tfds.as_numpy(train_ds)
    
        test_ds = tfds.load(name='cifar10', split='test',
                            as_supervised=True, data_dir=data_dir)

        test_ds = (
            test_ds
            .map(normalize)
            .batch(5000)
            .prefetch(1)
        )

        test_ds = tfds.as_numpy(test_ds)

    elif task=='cifar100':

        rng = tf.random.Generator.from_seed(seed, alg='philox')

        def normalize(image, label):
            image = tf.cast(image, tf.float32)
            image = tf.divide(image, 255)
            mean = tf.constant([0.4914, 0.4822, 0.4465])
            mean = tf.reshape(mean, (1,1,3))
            std = tf.constant([1.5*0.2023, 1.5*0.1994, 1.5*0.2010])
            std = tf.reshape(std, (1,1,3))
            image = (image - mean)/std
            label = tf.one_hot(label, 100)
            return image, label
        
        def augment(image_label, seed):
            image, label = image_label
            image = tf.image.stateless_random_flip_left_right(image, seed=seed)
            image = tf.image.resize_with_crop_or_pad(image, 36, 36)
            image = tf.image.stateless_random_crop(image, size=[32,32,3], seed=seed)
            return image, label 

        def rnd_aug(x, y):
            seed = rng.make_seeds(2)[0]
            image, label = augment((x, y), seed)
            return image, label

        # cifar100_data = tfds.load(name="cifar100", batch_size=-1,
                                  # data_dir=data_dir, download=True) 

        train_ds = tfds.load(name='cifar100', split='train',
                             as_supervised=True, data_dir=data_dir)

        train_ds = (
            train_ds
            .shuffle(len(train_ds))
            .map(rnd_aug)
            .map(normalize)
            .batch(batch_size, drop_remainder=True)
            .prefetch(1)
        )

        train_ds = tfds.as_numpy(train_ds)

        test_ds = tfds.load(name='cifar100', split='test',
                            as_supervised=True, data_dir=data_dir)

        test_ds = (
            test_ds
            .map(normalize)
            .batch(5000)
            .prefetch(1)
        )

        test_ds = tfds.as_numpy(test_ds)


    elif task=='imagenet32':
    
        rng = tf.random.Generator.from_seed(seed, alg='philox')

        def normalize(image, label):
            image = tf.cast(image, tf.float32)
            image = tf.divide(image, 255)
            mean = tf.constant([0.485, 0.456, 0.406])
            mean = tf.reshape(mean, (1,1,3))
            std = tf.constant([1.5*0.229, 1.5*0.224, 1.5*0.225])
            std = tf.reshape(std, (1,1,3))
            image = (image - mean)/std
            label = tf.one_hot(label, 1000)
            return image, label
        
        def augment(image_label, seed):
            image, label = image_label
            image = tf.image.stateless_random_flip_left_right(image, seed=seed)
            image = tf.image.resize_with_crop_or_pad(image, 36, 36)
            image = tf.image.stateless_random_crop(image, size=[32,32,3], seed=seed)
            return image, label 

        def rnd_aug(x, y):
            seed = rng.make_seeds(2)[0]
            image, label = augment((x, y), seed)
            return image, label

        split = tfds.split_for_jax_process('train', drop_remainder=True)
        train_ds = tfds.load(name='imagenet_resized/32x32', split=split,
                             as_supervised=True, data_dir=data_dir, 
                             shuffle_files=True, download=False)

        train_ds = (
            train_ds
            .shuffle(len(train_ds))
            .map(rnd_aug)
            .map(normalize)
            .batch(batch_size, drop_remainder=True)
            .prefetch(1)
        )

        train_ds = tfds.as_numpy(train_ds)
    
        test_ds = tfds.load(name='imagenet_resized/32x32', split='validation',
                            as_supervised=True, data_dir=data_dir)

        test_ds = (
            test_ds
            .map(normalize)
            .batch(5000)
            .prefetch(1)
        )

        test_ds = tfds.as_numpy(test_ds)

    return train_ds, test_ds 
