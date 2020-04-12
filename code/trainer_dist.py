from __future__ import print_function
from six.moves import range

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.nn import SyncBatchNorm

from PIL import Image

from miscc.config import cfg
from miscc.utils import mkdir_p, apply_running_mean
from miscc.utils import build_super_images, build_super_images2
from miscc.utils import weights_init, load_params, copy_G_params
from model import G_DCGAN, G_NET, DCM_Net
from datasets import prepare_data
from model import RNN_ENCODER, CNN_ENCODER
from VGGFeatureLoss import VGGNet

from miscc.losses import words_loss
from miscc.losses import discriminator_loss, generator_loss, KL_loss
from miscc.losses import D_loss_dist, G_loss_dist

import os
import time
import numpy as np
import sys
from copy import deepcopy


class condGANTrainer(object):
    def __init__(self, data_loader, n_words, ixtoword):
        self.gpu = dist.get_rank()
        torch.cuda.set_device(self.gpu)
        cudnn.benchmark = True

        if cfg.TRAIN.FLAG:
            output_dir = cfg.OUTPUT_DIR
            self.model_dir = os.path.join(output_dir, 'Model', 'GPU'+str(self.gpu))
            self.image_dir = os.path.join(output_dir, 'Image', 'GPU'+str(self.gpu))
            mkdir_p(self.model_dir)
            mkdir_p(self.image_dir)

        self.batch_size = cfg.TRAIN.BATCH_SIZE_PER_GPU
        self.max_epoch = cfg.TRAIN.MAX_EPOCH
        self.snapshot_interval = cfg.TRAIN.SNAPSHOT_INTERVAL

        self.n_words = n_words
        self.ixtoword = ixtoword
        self.data_loader = data_loader
        self.num_batches = len(self.data_loader)
        self.text_encoder, self.image_encoder, \
            self.netG, self.netsD, self.start_epoch, self.VGG = self.build_models()
        self.optG, self.optD = self.define_optimizers(self.netG, self.netsD)

    def build_models(self):
        # ################# Text and Image encoders ##########################
        if cfg.TRAIN.NET_E == '':
            print('Error: no pretrained text-image encoders')
            return

        VGG = VGGNet()

        for p in VGG.parameters():
            p.requires_grad = False

        if self.gpu == 0:
            print("Load the VGG model")

        VGG.eval()

        image_encoder = CNN_ENCODER(cfg.TEXT.EMBEDDING_DIM)
        img_encoder_path = cfg.TRAIN.NET_E.replace('text_encoder',
                                                   'image_encoder')
        state_dict = torch.load(img_encoder_path,
                                map_location=lambda storage, loc: storage)
        image_encoder.load_state_dict(state_dict)
        for p in image_encoder.parameters():
            p.requires_grad = False
        if self.gpu == 0:
            print('Load image encoder from:', img_encoder_path)
        image_encoder.eval()

        text_encoder = RNN_ENCODER(self.n_words,
                                   nhidden=cfg.TEXT.EMBEDDING_DIM)
        state_dict = torch.load(cfg.TRAIN.NET_E,
                                map_location=lambda storage, loc: storage)
        text_encoder.load_state_dict(state_dict)
        for p in text_encoder.parameters():
            p.requires_grad = False
        if self.gpu == 0:
            print('Load text encoder from:', cfg.TRAIN.NET_E)
        text_encoder.eval()

        # ##################### Generator and Discriminators ##############
        netsD = []
        if cfg.GAN.B_DCGAN:
            if cfg.TREE.BRANCH_NUM == 1:
                from model import D_NET64 as D_NET
            elif cfg.TREE.BRANCH_NUM == 2:
                from model import D_NET128 as D_NET
            else:  # cfg.TREE.BRANCH_NUM == 3:
                from model import D_NET256 as D_NET
            netG = G_DCGAN()
            netsD = [D_NET(b_jcu=False)]
        else:
            from model import D_NET64, D_NET128, D_NET256
            netG = G_NET()
            if cfg.TREE.BRANCH_NUM > 0:
                netsD.append(D_NET64())
            if cfg.TREE.BRANCH_NUM > 1:
                netsD.append(D_NET128())
            if cfg.TREE.BRANCH_NUM > 2:
                netsD.append(D_NET256())

        netG.apply(weights_init)
        for i in range(len(netsD)):
            netsD[i].apply(weights_init)
        if self.gpu == 0:
            print('# of netsD', len(netsD))

        epoch = 0
        if cfg.TRAIN.NET_G != '':
            state_dict = torch.load(cfg.TRAIN.NET_G,
                                    map_location=lambda storage, loc: storage)
            try:
                netG.load_state_dict(state_dict)
            except:
                # print('Load parameters in list version.')
                load_params(netG, state_dict)

            if self.gpu == 0:
                print('Load G from: ', cfg.TRAIN.NET_G)
            istart = cfg.TRAIN.NET_G.rfind('_') + 1
            iend = cfg.TRAIN.NET_G.rfind('.')
            epoch = cfg.TRAIN.NET_G[istart:iend]
            epoch = int(epoch) + 1
            if cfg.TRAIN.B_NET_D:
                Gname = cfg.TRAIN.NET_G
                for i in range(len(netsD)):
                    s_tmp = Gname[:Gname.rfind('/')]
                    Dname = '%s/netD%d.pth' % (s_tmp, i)
                    if self.gpu == 0:
                        print('Load D from: ', Dname)
                    state_dict = torch.load(Dname,
                                            map_location=lambda storage, loc: storage)
                    netsD[i].load_state_dict(state_dict)
        # ########################################################### #
        text_encoder = text_encoder.cuda()
        image_encoder = image_encoder.cuda()
        VGG = VGG.cuda()
        netG.cuda()
        netG = DDP(SyncBatchNorm.convert_sync_batchnorm(netG),
                   device_ids=[self.gpu], find_unused_parameters=True)
        # netG = DDP(netG, device_ids=[self.gpu], find_unused_parameters=True)

        # for i in range(len(netsD)):
        #    netsD[i].cuda()
        netsD = [DDP(SyncBatchNorm.convert_sync_batchnorm(netD.cuda()),
                     device_ids=[self.gpu],
                     find_unused_parameters=True) for netD in netsD]
        # netsD = [DDP(netD.cuda(),
        #              device_ids=[self.gpu],
        #              find_unused_parameters=True) for netD in netsD]
        return [text_encoder, image_encoder, netG, netsD, epoch, VGG]

    def define_optimizers(self, netG, netsD):
        optimizersD = []
        num_Ds = len(netsD)
        for i in range(num_Ds):
            opt = optim.Adam(netsD[i].parameters(),
                             lr=cfg.TRAIN.DISCRIMINATOR_LR,
                             betas=(0.5, 0.999))
            optimizersD.append(opt)

        optimizerG = optim.Adam(netG.parameters(),
                                lr=cfg.TRAIN.GENERATOR_LR,
                                betas=(0.5, 0.999))

        return optimizerG, optimizersD

    def prepare_labels(self):
        batch_size = self.batch_size
        real_labels = Variable(torch.FloatTensor(batch_size).fill_(1))
        fake_labels = Variable(torch.FloatTensor(batch_size).fill_(0))
        match_labels = Variable(torch.LongTensor(range(batch_size)))
        if cfg.CUDA:
            real_labels = real_labels.cuda()
            fake_labels = fake_labels.cuda()
            match_labels = match_labels.cuda()

        return real_labels, fake_labels, match_labels

    def save_model(self, avg_dict_G, netsD, epoch):
        # if self.gpu != 0:
        #     return
        print('GPU: {} start save G'.format(self.gpu))
        # torch.save([p for p in avg_param_G],
        # '%s/netG_epoch_%d.pth' % (self.model_dir, epoch))
        torch.save(avg_dict_G,
                   '%s/netG_epoch_%d.pth' % (self.model_dir, epoch))

        print('GPU: {} start save D'.format(self.gpu))
        for i in range(len(netsD)):
            netD = netsD[i]
            torch.save(netD.module.state_dict(),
                       '%s/netD%d_epoch_%d.pth' % (self.model_dir, i, epoch))
        print('Save G/Ds models of GPU: {}.'.format(self.gpu))

    def save_img_results(self, netG, noise, sent_emb, words_embs, mask,
                         image_encoder, captions, cap_lens,
                         gen_iterations, cnn_code, region_features,
                         real_imgs, name='current'):
        # Save images
        print('run inference on GPU: ', self.gpu)
        with torch.no_grad():
            netG.eval()
            fake_imgs, attention_maps, _, _, _, _ = netG(noise, sent_emb,
                                                         words_embs, mask,
                                                         cnn_code,
                                                         region_features)
            for i in range(len(attention_maps)):
                if len(fake_imgs) > 1:
                    img = fake_imgs[i + 1].detach().cpu()
                    lr_img = fake_imgs[i].detach().cpu()
                else:
                    img = fake_imgs[0].detach().cpu()
                    lr_img = None
                attn_maps = attention_maps[i]
                att_sze = attn_maps.size(2)
                img_set, _ = \
                    build_super_images(img, captions, self.ixtoword,
                                       attn_maps, att_sze, lr_imgs=lr_img)
                if img_set is not None:
                    im = Image.fromarray(img_set)
                    fullpath = '%s/G_%s_%d_%d.png'\
                        % (self.image_dir, name, gen_iterations, i)
                    im.save(fullpath)

            i = -1
            img = fake_imgs[i].detach()
            region_features, _ = image_encoder(img)
            att_sze = region_features.size(2)
            _, _, att_maps = words_loss(region_features.detach(),
                                        words_embs.detach(),
                                        None, cap_lens,
                                        None, self.batch_size)
            img_set, _ = \
                build_super_images(fake_imgs[i].detach().cpu(),
                                   captions, self.ixtoword, att_maps, att_sze)
            if img_set is not None:
                im = Image.fromarray(img_set)
                fullpath = '%s/D_%s_%d.png'\
                    % (self.image_dir, name, gen_iterations)
                im.save(fullpath)
            netG.train()

        # save the real images
        nvis = min(len(real_imgs[-1]), 8)
        for k in range(nvis):
            im = real_imgs[-1][k].data.cpu().numpy()
            im = (im + 1.0) * 127.5
            im = im.astype(np.uint8)
            im = np.transpose(im, (1, 2, 0))
            im = Image.fromarray(im)
            fullpath = '%s/R_%s_%d_%d.png'\
                    % (self.image_dir, name, gen_iterations, k)
            im.save(fullpath)

    def train(self):
        text_encoder = self.text_encoder
        image_encoder = self.image_encoder
        netG = self.netG
        netsD = self.netsD
        start_epoch = self.start_epoch
        VGG = self.VGG
        optimizerG, optimizersD = self.optG, self.optD
        avg_dict_G = deepcopy(netG.state_dict())
        real_labels, fake_labels, match_labels = self.prepare_labels()

        batch_size = self.batch_size
        nz = cfg.GAN.Z_DIM
        noise = Variable(torch.FloatTensor(batch_size, nz))
        fixed_noise = Variable(torch.FloatTensor(batch_size, nz).normal_(0, 1))

        noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

        gen_iterations = 0
        for epoch in range(start_epoch, self.max_epoch):
            self.data_loader.sampler.set_epoch(epoch)
            start_t = time.time()

            data_iter = iter(self.data_loader)
            step = 0
            while step < self.num_batches:
                # print('step {} for gpu {}'.format(step, self.gpu))

                ######################################################
                # (1) Prepare training data and Compute text embeddings
                ######################################################
                data = data_iter.next()
                imgs, captions, cap_lens, class_ids, keys, wrong_caps, \
                    wrong_caps_len, wrong_cls_id = prepare_data(data)

                hidden = text_encoder.init_hidden(batch_size)
                # words_embs: batch_size x nef x seq_len
                # sent_emb: batch_size x nef

                # matched text embeddings
                words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
                words_embs, sent_emb = words_embs.detach(), sent_emb.detach()

                # mismatched text embeddings
                w_words_embs, w_sent_emb = text_encoder(wrong_caps, wrong_caps_len, hidden)
                w_words_embs, w_sent_emb = w_words_embs.detach(), w_sent_emb.detach()

                # image features: regional and global
                region_features, cnn_code = image_encoder(imgs[len(netsD)-1])

                mask = (captions == 0)
                num_words = words_embs.size(2)
                if mask.size(1) > num_words:
                    mask = mask[:, :num_words]

                #######################################################
                # (2) Modify real images
                ######################################################
                noise.data.normal_(0, 1)
                fake_imgs, _, mu, logvar, _, _ = netG(noise, sent_emb,
                                                      words_embs, mask,
                                                      cnn_code,
                                                      region_features)
                #######################################################
                # (3) Update D network
                ######################################################
                errD_total = 0
                D_logs = ''
                for i in range(len(netsD)):
                    netsD[i].zero_grad()
                    # errD = discriminator_loss(netsD[i].module, imgs[i],
                    #                           fake_imgs[i],
                    #                           sent_emb, real_labels,
                    #                           fake_labels, words_embs,
                    #                           cap_lens, image_encoder,
                    #                           class_ids, w_words_embs,
                    #                           wrong_caps_len, wrong_cls_id)
                    # use distribute method
                    D_output = netsD[i](imgs[i], fake_imgs[i],
                                        sent_emb, real_labels,
                                        fake_labels, target='D')
                    errD = D_loss_dist(D_output, imgs[i], words_embs,
                                       cap_lens, image_encoder,
                                       class_ids, w_words_embs,
                                       wrong_caps_len, wrong_cls_id,
                                       real_labels, fake_labels)
                    # backward and update parameters
                    errD.backward(retain_graph=True)
                    optimizersD[i].step()
                    errD_total += errD
                    D_logs += 'errD%d: %.2f ' % (i, errD.item())

                #######################################################
                # (4) Update G network: maximize log(D(G(z)))
                ######################################################
                # compute total loss for training G
                step += 1
                gen_iterations += 1

                netG.zero_grad()
                D_outputs = [netD(img, real_labels, sent_emb, target='G')
                             for netD, img in zip(netsD, fake_imgs)]
                errG_total, G_logs = G_loss_dist(D_outputs, image_encoder,
                                                 fake_imgs, words_embs,
                                                 sent_emb, match_labels,
                                                 cap_lens, class_ids,
                                                 VGG, imgs)
                # errG_total, G_logs = \
                #     generator_loss(netsD, image_encoder, fake_imgs,
                #                    real_labels, words_embs, sent_emb,
                #                    match_labels, cap_lens, class_ids,
                #                    VGG, imgs)
                kl_loss = KL_loss(mu, logvar)
                errG_total += kl_loss
                G_logs += 'kl_loss: %.2f ' % kl_loss.item()
                # backward and update parameters
                errG_total.backward()
                optimizerG.step()

                # for p, avg_p in zip(netG.parameters(), avg_param_G):
                #     avg_p.mul_(0.999).add_(0.001, p.data)

                avg_dict_G = apply_running_mean(netG.state_dict(), avg_dict_G)

                if gen_iterations % 100 == 0 and self.gpu == 0:
                    step_log = '[%d/%d][%d/%d] ' % (epoch+1, self.max_epoch,
                                                    step, self.num_batches)
                    logs = step_log + D_logs + '\n' + step_log + G_logs
                    # print(D_logs + '\n' + G_logs)
                    print(logs)
                # save images
                if gen_iterations % 1000 == 0:
                    self.save_img_results(netG, fixed_noise, sent_emb,
                                          words_embs, mask, image_encoder,
                                          captions, cap_lens, epoch, cnn_code,
                                          region_features, imgs,
                                          name='current')
                    dist.barrier()

            end_t = time.time()
            if self.gpu == 0:
                print('''[%d/%d][%d] Loss_D: %.2f Loss_G: %.2f Time: %.2fs'''
                      % (epoch+1, self.max_epoch, self.num_batches,
                         errD_total, errG_total, end_t - start_t), end='\n')
                # print('''[%d/%d][%d] Loss_D: %.2f Loss_G: %.2f'''
                #       % (epoch+1, self.max_epoch, self.num_batches,
                #          errD_total, errG_total), end='\n')
            dist.barrier()

            if epoch % cfg.TRAIN.SNAPSHOT_INTERVAL == 0:
                self.save_model(avg_dict_G, netsD, epoch)
                dist.barrier()  # make sure all model finish saving

        self.save_model(avg_dict_G, netsD, self.max_epoch)
        dist.barrier()

    def train_noupdate(self):
        text_encoder = self.text_encoder
        image_encoder = self.image_encoder
        netG = self.netG
        netsD = self.netsD
        start_epoch = self.start_epoch
        real_labels, fake_labels, match_labels = self.prepare_labels()

        batch_size = self.batch_size
        nz = cfg.GAN.Z_DIM
        noise = Variable(torch.FloatTensor(batch_size, nz))
        fixed_noise = Variable(torch.FloatTensor(batch_size, nz).normal_(0, 1))

        noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

        gen_iterations = 0
        for epoch in range(start_epoch, self.max_epoch):
            data_iter = iter(self.data_loader)
            dist.barrier()
            step = 0
            while step < self.num_batches:

                data = data_iter.next()
                dist.barrier()
                imgs, captions, cap_lens, class_ids, keys, wrong_caps, \
                    wrong_caps_len, wrong_cls_id = prepare_data(data)
                dist.barrier()
                step += 1
                gen_iterations += 1

                if gen_iterations % 100 == 0:
                    print('GPU: %d [%d/%d][%d/%d]' % (self.gpu, epoch+1, self.max_epoch, step, self.num_batches))

            print('''GPU: %d[%d/%d][%d]'''
                  % (self.gpu, epoch+1, self.max_epoch, self.num_batches), end='\n')
        dist.barrier()
