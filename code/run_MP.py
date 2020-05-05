from __future__ import print_function

from miscc.config import cfg, cfg_from_file
from datasets import TextDataset
from trainerDCM import condGANTrainer as trainer

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

dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)


def parse_args():
    parser = argparse.ArgumentParser(description='Train a main module of the ManiGAN network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfg/eval_bird.yml', type=str)
    parser.add_argument('--gpu', dest='gpu_id', type=int, default=-1)
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--name', type=str, help='name of this config')
    parser.add_argument('--dataset', type=str, help='dataset used for validation [train | test]')
    parser.add_argument('--ep_start', type=int, help='start epoch')
    parser.add_argument('--ep_end', type=int, help='end epoch')
    parser.add_argument('--net_G', type=str, help='')
    parser.add_argument('--net_C', type=str, help='base name for net_C')
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    
    cfg.TRAIN.FLAG = False
    if args.gpu_id != -1:
        cfg.GPU_ID = args.gpu_id
    else:
        cfg.CUDA = False

    if args.data_dir != '':
        cfg.DATA_DIR = args.data_dir

    if args.dataset != '':
        assert args.dataset in ['train', 'test']
        cfg.VALIDATION_SET = args.dataset
    
    if args.net_C is not None:
        cfg.TRAIN.NET_C_BASE = args.net_C

    if args.net_G is not None:
        cfg.TRAIN.NET_G = args.net_G
    
    if args.name is not None:
        dir_name = args.name
    else:
        dir_name = cfg.TRAIN.NET_G.split('/')[-1].split('.')[0]
    
    dir_name = os.path.join(cfg.VALIDATION_OUTPUT, dir_name, cfg.VALIDATION_SET)
    
    assert args.ep_start is not None and args.ep_end is not None
    ep_range = [ep for ep in range(args.ep_start, args.ep_end+1, 5)]
    
    cfg.B_VALIDATION = True
    print('Using config:')
    pprint.pprint(cfg)

    if not cfg.TRAIN.FLAG:
        args.manualSeed = 100
    elif args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    random.seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if cfg.CUDA:
        torch.cuda.manual_seed_all(args.manualSeed)

    output_dir = ''
    bshuffle = True
    split_dir = cfg.VALIDATION_SET

    # Get data loader
    imsize = cfg.TREE.BASE_SIZE * (2 ** (cfg.TREE.BRANCH_NUM - 1))

    image_transform = transforms.Compose([
        transforms.Scale(int(imsize * 76 / 64))])
   
    dataset = TextDataset(cfg.DATA_DIR, split_dir,
                          base_size=cfg.TREE.BASE_SIZE,
                          transform=image_transform)
    assert dataset
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.TRAIN.BATCH_SIZE,
        drop_last=True, shuffle=bshuffle, num_workers=int(cfg.WORKERS))

    # Define models and go to train/evaluate
    algo = trainer(output_dir, dataloader, dataset.n_words, dataset.ixtoword)

    if cfg.DATASET_NAME == 'birds':
        data_dir = cfg.DATA_DIR + '/CUB_200_2011'
    else:
        data_dir = cfg.DATA_DIR

    base_size = cfg.TREE.BASE_SIZE
    imsize_list = []
    for i in range(cfg.TREE.BRANCH_NUM):
        imsize_list.append(base_size)
        base_size = base_size * 2

    norm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    start_t = time.time()

    '''generate images from pre-extracted embeddings'''
    with torch.no_grad():
        algo.sampling_with_MP(dir_name, ep_range)
            
    end_t = time.time()
    print('Total time for Validation:', end_t - start_t)
