from __future__ import print_function

from miscc.config import cfg, cfg_from_file
from datasets import TextDataset
from trainer_dist import condGANTrainer as trainer
import torch.distributed as dist
import torch.backends.cudnn as cudnn

import os
import sys
import time
import random
import pprint
import datetime
import dateutil.tz
import argparse
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image
from torch.autograd import Variable
import numpy.random

dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)


def parse_args():
    parser = argparse.ArgumentParser(description='Train a main module of the ManiGAN network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfg/train_bird.yml', type=str)
    # parser.add_argument('--gpu', dest='gpu_id', type=int, default=-1)
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--name', type=str, help='name for output dir')
    parser.add_argument('--bz', type=int, help='batch size for training')
    parser.add_argument('--thread', type=int, help='thread for dataloader')
    parser.add_argument('--epoch', type=int, help='max epoch for training')
    parser.add_argument('--ngpus', default=4, type=int, help='number of gpus')
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='node rank for distributed training')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    cfg.CUDA = True

    if args.data_dir != '':
        cfg.DATA_DIR = args.data_dir

    if args.bz is not None:
        cfg.TRAIN.BATCH_SIZE = args.bz
        cfg.TRAIN.BATCH_SIZE_PER_GPU = args.bz // args.ngpus
        cfg.N_GPUS = args.ngpus

    if args.thread is not None:
        cfg.WORKERS = args.thread

    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    random.seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    torch.cuda.manual_seed_all(args.manualSeed)
    cudnn.deterministic = True

    if args.epoch is not None:
        cfg.TRAIN.MAX_EPOCH = args.epoch

    rank = args.local_rank
    if rank == 0:
        # save path
        now = datetime.datetime.now(dateutil.tz.tzlocal())
        timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
        if args.name is None:
            output_dir = '../output/%s_%s_%s' % \
                (cfg.DATASET_NAME, cfg.CONFIG_NAME, timestamp)
        else:
            output_dir = '../output/%s_%s_%s_%s' % \
                (cfg.DATASET_NAME, cfg.CONFIG_NAME, args.name, timestamp)
    cfg.OUTPUT_DIR = output_dir

    if rank == 0:
        print('Using config:')
        pprint.pprint(cfg)

    try:
        model_worker(rank, args.ngpus, args)
    except KeyboardInterrupt:
        if rank == 0:
            print('KeyboardInterrupt, Stop Running')
        dist.destroy_process_group()


def getDataLoader(rank):
    split_dir = 'train'
    imsize = cfg.TREE.BASE_SIZE * (2 ** (cfg.TREE.BRANCH_NUM - 1))
    image_transform = transforms.Compose([
        transforms.Scale(int(imsize * 76 / 64))])

    dataset = TextDataset(cfg.DATA_DIR, split_dir,
                          base_size=cfg.TREE.BASE_SIZE,
                          transform=image_transform, rank=rank)

    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)

    assert dataset
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU,
        drop_last=True, shuffle=None, num_workers=int(cfg.WORKERS),
        sampler=train_sampler)
    return dataloader, dataset.n_words, dataset.ixtoword


def model_worker(gpu, ngpus_per_node, args):
    dist.init_process_group(backend='nccl')

    dataloader, n_words, ixtoword = getDataLoader(gpu)
    start_t = time.time()
    algo = trainer(dataloader, n_words, ixtoword)
    algo.train()
    end_t = time.time()
    dist.barrier()
    if gpu == 0:
        print('Total time for training:', end_t - start_t)


if __name__ == "__main__":
    main()
