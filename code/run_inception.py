import os
import sys
import math
import scipy.misc
import numpy as np
from PIL import Image
import pickle
import tensorflow as tf
import time
import argparse
from inception.slim import slim

# Batch normalization. Constant governing the exponential moving average of
# the 'global' mean and variance for all activations.
BATCHNORM_MOVING_AVERAGE_DECAY = 0.9997
# The decay to use for the moving average.
MOVING_AVERAGE_DECAY = 0.9999


def inference(images, num_classes, for_training=False, restore_logits=True,
              scope=None):
    """Build Inception v3 model architecture.

    See here for reference: http://arxiv.org/abs/1512.00567

    Args:
    images: Images returned from inputs() or distorted_inputs().
    num_classes: number of classes
    for_training: If set to `True`, build the inference model for training.
      Kernels that operate differently for inference during training
      e.g. dropout, are appropriately configured.
    restore_logits: whether or not the logits layers should be restored.
      Useful for fine-tuning a model with different num_classes.
    scope: optional prefix string identifying the ImageNet tower.

    Returns:
    Logits. 2-D float Tensor.
    Auxiliary Logits. 2-D float Tensor of side-head. Used for training only.
    """
    # Parameters for BatchNorm.
    batch_norm_params = {
        # Decay for the moving averages.
        'decay': BATCHNORM_MOVING_AVERAGE_DECAY,
        # epsilon to prevent 0s in variance.
        'epsilon': 0.001,
    }
    # Set weight_decay for weights in Conv and FC layers.
    with slim.arg_scope([slim.ops.conv2d, slim.ops.fc], weight_decay=0.00004):
        with slim.arg_scope([slim.ops.conv2d],
                            stddev=0.1,
                            activation=tf.nn.relu,
                            batch_norm_params=batch_norm_params):
            logits, endpoints = slim.inception.inception_v3(
                images,
                dropout_keep_prob=0.8,
                num_classes=num_classes,
                is_training=for_training,
                restore_logits=restore_logits,
                scope=scope)

    # Grab the logits associated with the side head. Employed during training.
    auxiliary_logits = endpoints['aux_logits']
    return logits, auxiliary_logits


def load_data(fullpath):
    images = []
    for path, subdirs, files in os.walk(fullpath):
        for name in files:
            if name.rfind('jpg') != -1 or name.rfind('png') != -1:
                filename = os.path.join(path, name)
                # print('filename', filename)
                # print('path', path, '\nname', name)
                # print('filename', filename)
                if os.path.isfile(filename):
                    img = scipy.misc.imread(filename)
                    images.append(img)
#     print('images', len(images), images[0].shape)
    print('Load images from {}, Numbers: {}, Shape: {}'.format(fullpath, len(images), images[0].shape))
    return images


def preprocess(img):
    # print('img', img.shape, img.max(), img.min())
    # img = Image.fromarray(img, 'RGB')
    if len(img.shape) == 2:
        img = np.resize(img, (img.shape[0], img.shape[1], 3))
    img = scipy.misc.imresize(img, (299, 299, 3),
                              interp='bilinear')
    img = img.astype(np.float32)
    # [0, 255] --> [0, 1] --> [-1, 1]
    img = img / 127.5 - 1.
    # print('img', img.shape, img.max(), img.min())
    return np.expand_dims(img, 0)


def load_data_new(fullpath):
    # contain load_data and preprocess in single function
    images = []
    print(fullpath)
    for path, subdirs, files in os.walk(fullpath):
        for name in files:
            if name.rfind('jpg') != -1 or name.rfind('png') != -1:
                filename = os.path.join(path, name)
                if os.path.isfile(filename):
                    img_PIL = Image.open(filename)
                    img_np = np.array(img_PIL.resize((299, 299), Image.BILINEAR))
                    img_np = images.append(np.expand_dims(img_np, 0))

    images = np.concatenate(images, axis=0)
    images = images / 127.5 - 1
    print('Load images from {}, Numbers: {}, Shape: {}'.format(fullpath, len(images), images[0].shape))
    return images


def get_inception_score(sess, images, pred_op, ep_info=None):
    splits = number_splits
    bs = batch_size
    preds = []
    num_examples = len(images)
    n_batches = int(math.floor(float(num_examples) / float(bs)))
    indices = list(np.arange(num_examples))
    np.random.shuffle(indices)

    s_time = time.time()
    for i in range(n_batches):
        inp = images[i * bs: (i+1) * bs]
        pred = sess.run(pred_op, {'inputs:0': inp})
        preds.append(pred)

        if i % 500 == 0:
            e_time = time.time()
            if ep_info is None:
                print('Step: {}/{} Times: {}'.format(i, n_batches, e_time-s_time))
            elif len(ep_info) == 3:
                print('Epoch {} ({}/{}) Step: {}/{} Times: {}'.format(ep_info[0], ep_info[1], ep_info[2],
                                                                      i, n_batches, e_time-s_time))
            s_time = e_time

    preds = np.concatenate(preds, 0)
    scores = []
    for i in range(splits):
        istart = i * preds.shape[0] // splits
        iend = (i + 1) * preds.shape[0] // splits
        part = preds[istart:iend, :]
        kl = (part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0))))
        kl = np.mean(np.sum(kl, 1))
        scores.append(np.exp(kl))
    return np.mean(scores), np.std(scores)


def parse_args():
    parser = argparse.ArgumentParser(description='Run inception score for model')
    parser.add_argument('--gpu', dest='gpu_id', type=int, default=0)
    parser.add_argument('--name', type=str, default='', help='name for the output dir / model')
    parser.add_argument('--bz', type=int, default=8, help='batch size')
    parser.add_argument('--dataset', type=str, default='test',
                        help='dataset used for validation [train | test]')
    parser.add_argument('--ep_start', type=int, default=0, help='start epoch')
    parser.add_argument('--ep_end', type=int, default=100, help='end epoch')
    parser.add_argument('--ep_interval', type=int, default=5, help='epoch interval')
    parser.add_argument('--ckpt_path', type=str, help='ckpt path for inception net',
                        default='/space1/leoxing/data/inception/inception_finetuned_models/birds_valid299/model.ckpt')
    parser.add_argument('--offical', action='store_true')
    parser.add_argument('--fource', action='store_true')

    args = parser.parse_args()
    return args


# parameters for forward
num_classes = 50  # ???
number_splits = 10  # in original code


if __name__ == '__main__':
    # TODO add argparse here
    args = parse_args()

    assert args.name != ''
    model_name = args.name

    assert args.dataset in ['train', 'test']
    split_name = args.dataset

    isOffical = args.offical
    fource = args.fource

    record_path = os.path.join('../val_output/', model_name, split_name, 'inception.pkl')
    if isOffical:
        image_folder_base = os.path.join('../val_output/', model_name, split_name, 'results')
    else:
        image_folder_base = os.path.join('../val_output/', model_name, split_name, 'EP_{}')
    ckpt_path = args.ckpt_path

    ep_start = args.ep_start
    ep_end = args.ep_end
    ep_interval = args.ep_interval
    epoch_range = [-1] if isOffical else [ep for ep in range(ep_start, ep_end+1, ep_interval)]

    batch_size = args.bz
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    # build inception model
    with tf.Graph().as_default():
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            with tf.device("/gpu:%d" % 0):
                # Number of classes in the Dataset label set plus 1.
                # Label 0 is reserved for an (unused) background class.
                num_classes = num_classes + 1

                # Build a Graph that computes the logits predictions from the
                # inference model.
                inputs = tf.placeholder(
                    tf.float32, [batch_size, 299, 299, 3],
                    name='inputs')
                # print(inputs)

                logits, _ = inference(inputs, num_classes)
                # calculate softmax after remove 0 which reserve for BG
                known_logits = tf.slice(logits, [0, 1],
                                        [batch_size, num_classes - 1])
                pred_op = tf.nn.softmax(known_logits)

                # Restore the moving average version of the
                # learned variables for eval.
                variable_averages = \
                    tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
                variables_to_restore = variable_averages.variables_to_restore()
                saver = tf.train.Saver(variables_to_restore)
                saver.restore(sess, ckpt_path)
                print('Restore the model from %s).' % ckpt_path)

                if os.path.exists(record_path):
                    if fource:
                        print('Fource ReRun Inception Score on All Epoches !')
                        inp_dict = {}
                    else:
                        with open(record_path, 'rb') as file:
                            inp_dict = pickle.load(file)
                            print('Load Inception Score Dict in {}'.format(record_path))
                else:
                    inp_dict = {}  # {ep: np.array([MEAN, STD])}

                if isOffical:
                    print('Run Inception evaluation for Offical Model.')
                    print('Image dir: {}'.format(image_folder_base))
                    images = load_data_new(image_folder_base)
                    mean, var = get_inception_score(sess, images, pred_op, None)
                    pass
                # Evaluation loop --> non-offical loop
                for num_ep, ep in enumerate(epoch_range):
                    if ep in inp_dict.keys():
                        print('Epoch {} has already validate, Skip to next Epoch'.format(ep))
                        continue

                    print('')
                    print('Epoch {} ({}/{}) Start'.format(ep, num_ep, len(epoch_range)))
                    img_folder = image_folder_base.format(ep)
                    images = load_data_new(img_folder)
                    mean, var = get_inception_score(sess, images, pred_op, [ep, num_ep, len(epoch_range)])
                    inp_dict[ep] = np.array([mean, var])

                    with open(record_path, 'wb') as file:
                        pickle.dump(inp_dict, file)
