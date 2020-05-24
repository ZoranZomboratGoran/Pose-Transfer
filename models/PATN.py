import numpy as np
import torch
import os
import re
from collections import OrderedDict
import itertools
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
# losses
from losses.L1_plus_perceptualLoss import L1_plus_perceptualLoss

import sys
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn

class TransferModel(BaseModel):
    def name(self):
        return 'TransferModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        nb = opt.batchSize
        size = opt.fineSize
        self.running_err = OrderedDict()

        input_nc = [opt.P_input_nc, opt.BP_input_nc+opt.BP_input_nc]
        self.netG = networks.define_G(input_nc, opt.P_input_nc,
                                        opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids,
                                        n_downsampling=opt.G_n_downsampling)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            if opt.with_D_PB:
                self.netD_PB = networks.define_D(opt.P_input_nc+opt.BP_input_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids,
                                            not opt.no_dropout_D,
                                            n_downsampling = opt.D_n_downsampling)

            if opt.with_D_PP:
                self.netD_PP = networks.define_D(opt.P_input_nc+opt.P_input_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids,
                                            not opt.no_dropout_D,
                                            n_downsampling = opt.D_n_downsampling)

        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.netG, 'netG', which_epoch)
            if self.isTrain:
                if opt.with_D_PB:
                    self.load_network(self.netD_PB, 'netD_PB', which_epoch)
                if opt.with_D_PP:
                    self.load_network(self.netD_PP, 'netD_PP', which_epoch)


        if self.isTrain:
            self.old_lr = opt.lr
            self.fake_PP_pool = ImagePool(opt.pool_size)
            self.fake_PB_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)

            if opt.L1_type == 'origin':
                self.criterionL1 = torch.nn.L1Loss()
            elif opt.L1_type == 'l1_plus_perL1':
                self.criterionL1 = L1_plus_perceptualLoss(opt.lambda_A, opt.lambda_B, opt.perceptual_layers, self.gpu_ids, opt.percep_is_l1)
            else:
                raise Excption('Unsurportted type of L1!')
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            if opt.with_D_PB:
                self.optimizer_D_PB = torch.optim.Adam(self.netD_PB.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            if opt.with_D_PP:
                self.optimizer_D_PP = torch.optim.Adam(self.netD_PP.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizers = []
            self.schedulers = []
            self.optimizers.append(self.optimizer_G)
            if opt.with_D_PB:
                self.optimizers.append(self.optimizer_D_PB)
            if opt.with_D_PP:
                self.optimizers.append(self.optimizer_D_PP)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG)
        if self.isTrain:
            if opt.with_D_PB:
                networks.print_network(self.netD_PB)
            if opt.with_D_PP:
                networks.print_network(self.netD_PP)
        print('-----------------------------------------------')

    def set_input(self, input):
        self.input_P1, self.input_BP1 = input['P1'], input['BP1']
        self.input_P2, self.input_BP2 = input['P2'], input['BP2']
        self.image_paths = input['P1_path'][0] + '___' + input['P2_path'][0]
        result = re.search('(.*c)([0-9])(.*)', input['P1_path'][0])
        self.fake_path = result.group(1)+str(int(result.group(2)) + 10)+result.group(3)
        self.fake_id = [re.search('(.*)_c.*', x).group(1) for x in input['P1_path']]

        if len(self.gpu_ids) > 0:
            self.input_P1 = self.input_P1.cuda()
            self.input_BP1 = self.input_BP1.cuda()
            self.input_P2 = self.input_P2.cuda()
            self.input_BP2 = self.input_BP2.cuda()

    def forward(self):
        G_input = [self.input_P1,
                   torch.cat((self.input_BP1, self.input_BP2), 1)]
        self.fake_p2 = self.netG(G_input)


    def test(self):
        with torch.no_grad():
            G_input = [self.input_P1,
                       torch.cat((self.input_BP1, self.input_BP2), 1)]
            self.fake_p2 = self.netG(G_input)


    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def get_fake_path(self):
        return self.fake_path


    def backward_G(self, mode = 'train'):
        if self.opt.with_D_PB:
            pred_fake_PB = self.netD_PB(torch.cat((self.fake_p2, self.input_BP2), 1))
            loss_G_GAN_PB = self.criterionGAN(pred_fake_PB, True)

        if self.opt.with_D_PP:
            pred_fake_PP = self.netD_PP(torch.cat((self.fake_p2, self.input_P1), 1))
            loss_G_GAN_PP = self.criterionGAN(pred_fake_PP, True)

        # L1 loss
        if self.opt.L1_type == 'l1_plus_perL1' :
            losses = self.criterionL1(self.fake_p2, self.input_P2)
            self.loss_G_L1 = losses[0]
            self.loss_originL1 = losses[1].item()
            self.loss_perceptual = losses[2].item()
        else:
            self.loss_G_L1 = self.criterionL1(self.fake_p2, self.input_P2) * self.opt.lambda_A


        pair_L1loss = self.loss_G_L1
        if self.opt.with_D_PB:
            pair_GANloss = loss_G_GAN_PB * self.opt.lambda_GAN
            if self.opt.with_D_PP:
                pair_GANloss += loss_G_GAN_PP * self.opt.lambda_GAN
                pair_GANloss = pair_GANloss / 2
        else:
            if self.opt.with_D_PP:
                pair_GANloss = loss_G_GAN_PP * self.opt.lambda_GAN

        if self.opt.with_D_PB or self.opt.with_D_PP:
            pair_loss = pair_L1loss + pair_GANloss
        else:
            pair_loss = pair_L1loss

        if mode == 'train':
            pair_loss.backward()

        self.pair_L1loss = pair_L1loss.item()
        if self.opt.with_D_PB or self.opt.with_D_PP:
            self.pair_GANloss = pair_GANloss.item()

        self.loss_G_GAN_PB = loss_G_GAN_PB.item()
        self.loss_G_GAN_PP = loss_G_GAN_PP.item()


    def backward_D_basic(self, netD, real, fake, mode = 'train'):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True) * self.opt.lambda_GAN
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False) * self.opt.lambda_GAN
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        if mode == 'train':
            loss_D.backward()
        return loss_D

    # D: take(P, B) as input
    def backward_D_PB(self, mode = 'train'):
        real_PB = torch.cat((self.input_P2, self.input_BP2), 1)
        # fake_PB = self.fake_PB_pool.query(torch.cat((self.fake_p2, self.input_BP2), 1))
        fake_PB = self.fake_PB_pool.query( torch.cat((self.fake_p2, self.input_BP2), 1).data )
        loss_D_PB = self.backward_D_basic(self.netD_PB, real_PB, fake_PB, mode)
        self.loss_D_PB = loss_D_PB.item()

    # D: take(P, P') as input
    def backward_D_PP(self, mode = 'train'):
        real_PP = torch.cat((self.input_P2, self.input_P1), 1)
        # fake_PP = self.fake_PP_pool.query(torch.cat((self.fake_p2, self.input_P1), 1))
        fake_PP = self.fake_PP_pool.query( torch.cat((self.fake_p2, self.input_P1), 1).data )
        loss_D_PP = self.backward_D_basic(self.netD_PP, real_PP, fake_PP, mode)
        self.loss_D_PP = loss_D_PP.item()


    def optimize_parameters(self, mode = 'train'):
        # forward
        self.forward()

        self.optimizer_G.zero_grad()
        self.backward_G(mode)
        if mode == 'train':
            self.optimizer_G.step()

        # D_P
        if self.opt.with_D_PP:
            for _ in range(self.opt.DG_ratio):
                self.optimizer_D_PP.zero_grad()
                self.backward_D_PP(mode)
                if mode == 'train':
                    self.optimizer_D_PP.step()

        # D_BP
        if self.opt.with_D_PB:
            for _ in range(self.opt.DG_ratio):
                self.optimizer_D_PB.zero_grad()
                self.backward_D_PB(mode)
                if mode == 'train':
                    self.optimizer_D_PB.step()

        self.add_to_running_error()

    def clear_running_error(self):
        self.running_err['G_L1loss'] = 0
        self.running_err['G_origin_L1'] = 0
        self.running_err['G_perceptual_L1'] = 0
        self.running_err['G_GANloss'] = 0
        self.running_err['G_PP_GANloss'] = 0
        self.running_err['G_PB_GANloss'] = 0
        self.running_err['D_PP_GANloss'] = 0
        self.running_err['D_PB_GANloss'] = 0

    def get_epoch_errors(self, iterations):
        for k, v in self.running_err.items():
            self.running_err[k] = v / iterations
        return self.running_err

    def add_to_running_error(self):
        cur_err = self.get_current_errors()
        for k, v in cur_err.items():
            self.running_err[k] += v

    def get_current_errors(self):
        ret_errors = OrderedDict()

        # Generator L1 loss
        ret_errors ['G_L1loss'] = self.pair_L1loss
        if self.opt.L1_type == 'l1_plus_perL1':
            ret_errors['G_origin_L1'] = self.loss_originL1
            ret_errors['G_perceptual_L1'] = self.loss_perceptual

        # Generator GAN loss
        if self.opt.with_D_PB or self.opt.with_D_PP:
            ret_errors['G_GANloss'] = self.pair_GANloss
        if self.opt.with_D_PP:
            ret_errors['G_PP_GANloss'] = self.loss_G_GAN_PP
        if self.opt.with_D_PB:
            ret_errors['G_PB_GANloss'] = self.loss_G_GAN_PB

        # Discriminator GAN loss
        # D_PP appearance discriminator
        # D_PB shape discriminator
        if self.opt.with_D_PP:
            ret_errors['D_PP_GANloss'] = self.loss_D_PP
        if self.opt.with_D_PB:
            ret_errors['D_PB_GANloss'] = self.loss_D_PB

        return ret_errors

    def get_current_visuals(self):
        height, width = self.input_P1.size(2), self.input_P1.size(3)
        input_P1 = util.tensor2im(self.input_P1.data)
        input_P2 = util.tensor2im(self.input_P2.data)

        input_BP1 = util.draw_pose_from_map(self.input_BP1.data)[0]
        input_BP2 = util.draw_pose_from_map(self.input_BP2.data)[0]

        fake_p2 = util.tensor2im(self.fake_p2.data)

        vis = np.zeros((height, width*5, 3)).astype(np.uint8) #h, w, c
        vis[:, :width, :] = input_P1
        vis[:, width:width*2, :] = input_BP1
        vis[:, width*2:width*3, :] = input_P2
        vis[:, width*3:width*4, :] = input_BP2
        vis[:, width*4:, :] = fake_p2

        ret_visuals = OrderedDict([('vis', vis)])

        return ret_visuals

    def get_current_visuals_batched(self):
        height, width = self.input_P1.size(2), self.input_P1.size(3)
        input_P1 = util.tensor2im_batch(self.input_P1.data)
        input_P2 = util.tensor2im_batch(self.input_P2.data)

        input_BP1 = util.draw_pose_from_map_batched(self.input_BP1.data)
        input_BP2 = util.draw_pose_from_map_batched(self.input_BP2.data)

        fake_p2 = util.tensor2im_batch(self.fake_p2.data)

        ret_visuals = []
        for batch in range(len(fake_p2)):
            vis = np.zeros((height, width*5, 3)).astype(np.uint8) #h, w, c
            vis[:, :width, :] = input_P1[batch]
            vis[:, width:width*2, :] = input_BP1[batch][0]
            vis[:, width*2:width*3, :] = input_P2[batch]
            vis[:, width*3:width*4, :] = input_BP2[batch][0]
            vis[:, width*4:, :] = fake_p2[batch]
            ret_visuals.append(vis)

        return ret_visuals

    def get_current_fakes(self):
        fake_p2 = util.tensor2im(self.fake_p2.data)
        return fake_p2

    def save_fake(self):
        util.save_image(util.tensor2im(self.fake_p2.data),
                    os.path.join("fakes", self.fake_path))

    def save_visuals(self, epoch = 0):
        visuals = self.get_current_visuals_batched()
        for idx, image_numpy in enumerate(visuals):
            img_path = os.path.join(self.opt.img_dir, 'epoch%.3d_%s.png' % (epoch, self.fake_id[idx]))
            util.save_image(image_numpy, img_path)

    def save(self, label):
        self.save_network(self.netG,  'netG',  label, self.gpu_ids)
        if self.opt.with_D_PB:
            self.save_network(self.netD_PB,  'netD_PB',  label, self.gpu_ids)
        if self.opt.with_D_PP:
            self.save_network(self.netD_PP, 'netD_PP', label, self.gpu_ids)

