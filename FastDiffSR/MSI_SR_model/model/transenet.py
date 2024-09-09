"""
Super-resolution of CelebA using Generative Adversarial Networks.
The dataset can be downloaded from: https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AADIKlz8PR9zr6Y20qbkunrba/Img/img_align_celeba.zip?dl=0
Instrustion on running the script:
1. Download the dataset from the provided link
2. Save the folder 'img_align_celeba' to '../../data/'
4. Run the sript using command 'python3 srgan.py'
"""
import argparse
import os
import numpy as np
#import math
import time
from collections import OrderedDict
import sys
from PIL import Image
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torch.autograd import Variable

import torch
import torch.nn as nn
import torch.nn.functional as F

from skimage.measure import compare_ssim
from skimage.measure import compare_mse
from skimage.measure import compare_psnr
from torchvision.transforms import ToPILImage, ToTensor
from torchvision.models import vgg19

from utils import utils as srutils
from model.base_networks import *
from utils.logger import Logger, PrintLogger
from data.data import get_training_datasets, get_test_datasets, get_RGB_trainDataset, get_RGB_testDataset
import math
# def weights_init_normal(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1:
#         torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
#     elif classname.find('BatchNorm') != -1:
#         torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
#         torch.nn.init.constant_(m.bias.data, 0.0)
def weights_init_normal(m, mean=0.0, std=0.02):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(mean, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Conv2d') != -1:
        m.weight.data.normal_(mean, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('ConvTranspose2d') != -1:
        m.weight.data.normal_(mean, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Norm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        if m.bias is not None:
            m.bias.data.zero_()

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()

        vgg19_model = vgg19(pretrained=True)

        # Extracts features at the 11th layer
        self.feature_extractor = nn.Sequential(*list(vgg19_model.features.children())[:12])

    def forward(self, img):
        out = self.feature_extractor(img)
        return out

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size//2), bias=bias)

class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False

class BasicBlock(nn.Sequential):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size, stride=1, bias=True,
        bn=False, act=nn.ReLU(True)):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)
        super(BasicBlock, self).__init__(*m)

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if i == 0: m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feat, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feat, 4 * n_feat, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn: m.append(nn.BatchNorm2d(n_feat))
                if act: m.append(act())
        elif scale == 3:
            m.append(conv(n_feat, 9 * n_feat, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if act: m.append(act())
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


class DownBlock(nn.Module):
    def __init__(self, scale):
        super().__init__()

        self.scale = scale

    def forward(self, x):
        n, c, h, w = x.size()
        x = x.view(n, c, h//self.scale, self.scale, w//self.scale, self.scale)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()
        x = x.view(n, c * (self.scale**2), h//self.scale, w//self.scale)
        return x
class NonLocalBlock2D(nn.Module):
    def __init__(self, in_channels, inter_channels):
        super(NonLocalBlock2D, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        self.g = nn.Conv2d(in_channels=in_channels, out_channels=inter_channels,
                           kernel_size=1, stride=1, padding=0)
        self.W = nn.Conv2d(in_channels=inter_channels, out_channels=in_channels,
                           kernel_size=1, stride=1, padding=0)
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)

        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

    def forward(self, x):

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)

        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)

        # use dot production
        # N = f.size(-1)
        # f_div_C = f / N

        # use embedding gaussian
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z

# ref:NONLocalBlock2D
class AdjustedNonLocalBlock(nn.Module):
    def __init__(self, in_channels, inter_channels):
        super(AdjustedNonLocalBlock, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        self.g = nn.Conv2d(in_channels=in_channels, out_channels=inter_channels,
                           kernel_size=1, stride=1, padding=0)
        self.W = nn.Conv2d(in_channels=inter_channels, out_channels=in_channels,
                           kernel_size=1, stride=1, padding=0)
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)

        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

    def forward(self, x0, x1):

        batch_size = x0.size(0)

        g_x = self.g(x0).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x1).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)

        phi_x = self.phi(x0).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)

        # use embedding gaussian
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x0.size()[2:])
        W_y = self.W(y)
        z = W_y + x0

        return z

from einops import rearrange
from model.transformer import TransformerEncoder, TransformerDecoder

MIN_NUM_PATCHES = 12

class BasicModule(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, block_type='basic', bias=True,
                 bn=False, act=nn.ReLU(True)):
        super(BasicModule, self).__init__()

        self.block_type = block_type

        m_body = []
        if block_type == 'basic':
            n_blocks = 10
            m_body = [
                BasicBlock(conv, n_feat, n_feat, kernel_size, bias=bias, bn=bn)
                # common.ResBlock(conv, n_feat, kernel_size)
                for _ in range(n_blocks)
            ]
        elif block_type == 'residual':
            n_blocks = 5
            m_body = [
                ResBlock(conv, n_feat, kernel_size)
                for _ in range(n_blocks)
            ]
        else:
            print('Error: not support this type')
        self.body = nn.Sequential(*m_body)

    def forward(self, x):

        res = self.body(x)
        if self.block_type == 'basic':
            out = res + x
        elif self.block_type == 'residual':
            out = res

        return out


class TransENet(nn.Module):

    def __init__(self, n_feats=64, scale=4, n_basic_modules=10, n_colors=3, hr_patch_size=256, back_projection_iters=10, en_depth=8, de_depth=1, conv=default_conv):
        super(TransENet, self).__init__()
        self.hr_patch_size = hr_patch_size
        self.scale = scale
        n_feats = n_feats
        kernel_size = 3
        act = nn.ReLU(True)
        rgb_range = 1.0
        rgb_mean = (0.4916, 0.4991, 0.4565)  # UCMerced data
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = MeanShift(rgb_range, rgb_mean, rgb_std)

        # define head body
        m_head = [
            conv(n_colors, n_feats, kernel_size)
        ]
        self.head = nn.Sequential(*m_head)

        # define main body
        self.feat_extrat_stage1 = BasicModule(conv, n_feats, kernel_size, block_type='residual', act=act)
        self.feat_extrat_stage2 = BasicModule(conv, n_feats, kernel_size, block_type='residual', act=act)
        self.feat_extrat_stage3 = BasicModule(conv, n_feats, kernel_size, block_type='residual', act=act)

        reduction = 4
        self.stage1_conv1x1 = conv(n_feats, n_feats // reduction, 1)
        self.stage2_conv1x1 = conv(n_feats, n_feats // reduction, 1)
        self.stage3_conv1x1 = conv(n_feats, n_feats // reduction, 1)
        self.up_conv1x1 = conv(n_feats, n_feats // reduction, 1)
        self.span_conv1x1 = conv(n_feats // reduction, n_feats, 1)

        self.upsampler = Upsampler(conv, self.scale, n_feats, act=False)

        # define tail body
        self.tail = conv(n_feats, n_colors, kernel_size)
        self.add_mean = MeanShift(rgb_range, rgb_mean, rgb_std, 1)

        # define transformer
        image_size = hr_patch_size // self.scale
        patch_size = 8
        dim = 512
        en_depth = en_depth
        de_depth = de_depth
        heads = 6
        mlp_dim = 512
        channels = n_feats // reduction
        dim_head = 32
        dropout = 0.0

        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2

        patch_dim = channels * patch_size ** 2
        assert num_patches > MIN_NUM_PATCHES, f'your number of patches ({num_patches}) is way too small for attention to be effective (at least 16). Try decreasing your patch size'

        self.patch_size = patch_size
        self.patch_to_embedding_low1 = nn.Linear(patch_dim, dim)
        self.patch_to_embedding_low2 = nn.Linear(patch_dim, dim)
        self.patch_to_embedding_low3 = nn.Linear(patch_dim, dim)
        self.patch_to_embedding_high = nn.Linear(patch_dim, dim)

        self.embedding_to_patch = nn.Linear(dim, patch_dim)

        self.encoder_stage1 = TransformerEncoder(dim, en_depth, heads, dim_head, mlp_dim, dropout)
        self.encoder_stage2 = TransformerEncoder(dim, en_depth, heads, dim_head, mlp_dim, dropout)
        self.encoder_stage3 = TransformerEncoder(dim, en_depth, heads, dim_head, mlp_dim, dropout)
        self.encoder_up = TransformerEncoder(dim, en_depth, heads, dim_head, mlp_dim, dropout)

        self.decoder1 = TransformerDecoder(dim, de_depth, heads, dim_head, mlp_dim, dropout)
        self.decoder2 = TransformerDecoder(dim, de_depth, heads, dim_head, mlp_dim, dropout)
        self.decoder3 = TransformerDecoder(dim, de_depth, heads, dim_head, mlp_dim, dropout)


    def forward(self, x):

        x = self.sub_mean(x)
        x = self.head(x)

        # feature extraction part
        feat_stage1 = self.feat_extrat_stage1(x)
        feat_stage2 = self.feat_extrat_stage2(x)
        feat_stage3 = self.feat_extrat_stage3(x)
        feat_ups = self.upsampler(feat_stage3)

        feat_stage1 = self.stage1_conv1x1(feat_stage1)
        feat_stage2 = self.stage2_conv1x1(feat_stage2)
        feat_stage3 = self.stage3_conv1x1(feat_stage3)
        feat_ups = self.up_conv1x1(feat_ups)

        # transformer part:
        p = self.patch_size

        feat_stage1 = rearrange(feat_stage1, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = p, p2 = p)
        feat_stage2 = rearrange(feat_stage2, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)
        feat_stage3 = rearrange(feat_stage3, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)
        feat_ups = rearrange(feat_ups, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = p, p2 = p)

        feat_stage1 = self.patch_to_embedding_low1(feat_stage1)
        feat_stage2 = self.patch_to_embedding_low2(feat_stage2)
        feat_stage3 = self.patch_to_embedding_low3(feat_stage3)
        feat_ups = self.patch_to_embedding_high(feat_ups)

        # encoder
        feat_stage1 = self.encoder_stage1(feat_stage1)
        feat_stage2 = self.encoder_stage2(feat_stage2)
        feat_stage3 = self.encoder_stage3(feat_stage3)
        feat_ups = self.encoder_up(feat_ups)

        feat_ups = self.decoder3(feat_ups, feat_stage3)
        feat_ups = self.decoder2(feat_ups, feat_stage2)
        feat_ups = self.decoder1(feat_ups, feat_stage1)

        feat_ups = self.embedding_to_patch(feat_ups)
        feat_ups = rearrange(feat_ups, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h=self.hr_patch_size // p, p1=p, p2=p)

        feat_ups = self.span_conv1x1(feat_ups)

        x = self.tail(feat_ups)
        x = self.add_mean(x)

        return x

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))

class TransENetModel(object):
    def __init__(self, args):
        # parameters
        self.model_name = args.model_name
        self.train_dataset = args.train_dataset
        self.test_dataset = args.test_dataset
        self.crop_size = args.crop_size
        self.test_crop_size = args.test_crop_size
        self.hr_height = args.hr_height
        self.hr_width = args.hr_width
        self.num_threads = args.num_threads
        self.num_channels = args.num_channels
        self.scale_factor = args.scale_factor
        self.epoch = args.epoch
        self.num_epochs = args.num_epochs
        self.save_epochs = args.save_epochs
        self.batch_size = args.batch_size
        self.test_batch_size = args.test_batch_size
        self.lr = args.lr
        self.b1 = args.b1
        self.b2 = args.b2
        self.data_dir = args.data_dir
        self.root_dir = args.root_dir
        self.save_dir = args.save_dir
        self.gpu_mode = args.gpu_mode
        self.n_cpu = args.n_cpu
        self.sample_interval = args.sample_interval

        cuda = True if torch.cuda.is_available() else False
        from utils import PerceptualSimilarity
        self.PerceptualModel = PerceptualSimilarity.PerceptualLoss(model='net-lin', net='alex', use_gpu=cuda)
        self.clip_value = args.clip_value
        self.lambda_gp = args.lambda_gp
        self.gp = args.gp
        self.penalty_type = args.penalty_type
        self.grad_penalty_Lp_norm = args.grad_penalty_Lp_norm
        self.relative = args.relativeGan
        self.loss_Lp_norm = args.loss_Lp_norm
        self.weight_gan = args.weight_gan
        self.weight_content = args.weight_content
        self.max_train_samples = args.max_train_samples

        self.log_dict = OrderedDict()

        # set the logger
        log_dir = os.path.join(self.root_dir, self.save_dir, 'logs')
        log_freq = 200
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        self.logger = Logger(self.model_name, log_dir, log_freq)

    def compute_gradient_penalty(self, discriminator, real_samples, fake_samples):
        """Calculates the gradient penalty loss for WGAN GP"""
        #cuda = True if torch.cuda.is_available() else False
        Tensor = torch.cuda.FloatTensor if self.gpu_mode else torch.FloatTensor
        # Random weight term for interpolation between real and fake samples
        alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = discriminator(interpolates)
        # fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)

        # add by lyd
        Tensor = torch.cuda.FloatTensor if self.gpu_mode else torch.Tensor
        grad_outputs = Tensor(d_interpolates.size())
        grad_outputs.resize_(d_interpolates.size()).fill_(1.0)

        # Get gradient w.r.t. interpolates
        gradients = torch.autograd.grad(outputs=d_interpolates, inputs=interpolates,
                                  grad_outputs=grad_outputs, create_graph=True, retain_graph=True,
                                  only_inputs=True)[0]
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def gradient_penalty(self, discriminator, real_samples, fake_samples,
                         grad_penalty_Lp_norm='L2', penalty_type='LS'):
        """Calculates the gradient penalty loss for WGAN GP"""
        # Best setting (novel Hinge Linfinity gradient penalty)
        #grad_penalty_Lp_norm = 'Linf'
        #penalty_type = 'hinge'
        # Default setting from WGAN-GP and most cases (L2 gradient penalty)
        #grad_penalty_Lp_norm = 'L2'
        #penalty_type = 'LS'
        # Calculate gradient
        #penalty_weight = 20 # 10 is the more usual choice

        #cuda = True if torch.cuda.is_available() else False
        Tensor = torch.cuda.FloatTensor if self.gpu_mode else torch.FloatTensor
        # Random weight term for interpolation between real and fake samples
        alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = discriminator(interpolates)
        # fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)

        Tensor = torch.cuda.FloatTensor if self.gpu_mode else torch.Tensor
        grad_outputs = Tensor(d_interpolates.size())
        grad_outputs.resize_(d_interpolates.size()).fill_(1.0)
        #grad_outputs = torch.ones(param.batch_size)

        # Get gradient w.r.t. interpolates
        gradients = torch.autograd.grad(outputs=d_interpolates, inputs=interpolates,grad_outputs=grad_outputs, create_graph=True, retain_graph=True,only_inputs=True)[0]
        #grad = grad.view(current_batch_size,-1)

        if grad_penalty_Lp_norm == 'Linf': # Linfinity gradient norm penalty (Corresponds to L1 margin, BEST results)
          grad_abs = torch.abs(gradients) # Absolute value of gradient
          grad_norm , _ = torch.max(grad_abs,1)
        elif grad_penalty_Lp_norm == 'L1': # L1 gradient norm penalty (Corresponds to Linfinity margin, WORST results)
          grad_norm = gradients.norm(1,1)
        else: # L2 gradient norm penalty (Corresponds to L2 margin, this is what people generally use)
          grad_norm = gradients.norm(2,1)

        if penalty_type == 'LS': # The usual choice, penalize values below 1 and above 1 (too constraining to properly estimate the Wasserstein distance)
          constraint = (grad_norm-1).pow(2)
        elif penalty_type == 'hinge': # Penalize values above 1 only (best choice)
          constraint = torch.nn.ReLU()(grad_norm - 1)

        constraint = constraint.mean()
        gradient_penalty = constraint
        gradient_penalty.backward(retain_graph=True)

        return gradient_penalty

    def load_dataset(self, dataset='train', max_samples=10000):
        if self.num_channels == 1:
            is_gray = True
        else:
            is_gray = False

        if dataset == 'train':
            print('Loading train dct_datasets...')
            train_set = get_RGB_trainDataset(self.data_dir, self.train_dataset, self.crop_size, self.scale_factor, is_gray=is_gray)
            return DataLoader(dataset=train_set, num_workers=self.num_threads, batch_size=self.batch_size,
                              shuffle=True)
        elif dataset == 'test':
            print('Loading test dct_datasets...')
            # test_set = get_test_sets(self.data_dir, self.test_dataset, self.scale_factor, is_gray=is_gray)
            test_set = get_RGB_testDataset(self.data_dir, self.test_dataset, self.crop_size, self.scale_factor, is_gray=is_gray)
            return DataLoader(dataset=test_set, num_workers=self.num_threads,
                              batch_size=self.test_batch_size,
                              shuffle=False)

    def train(self):
        # os.makedirs('images', exist_ok=True)
        # os.makedirs('saved_models', exist_ok=True)

        # torch.cuda.set_device(0)
        # torch.backends.cudnn.benchmark = True
        device_ids = [0]
        cuda = True if torch.cuda.is_available() else False
        # cuda = self.gpu_mode
        self.device = torch.device('cuda' if cuda else 'cpu')

        # Calculate output of image discriminator (PatchGAN)
        patch_h, patch_w = int(self.crop_size / 2 ** self.scale_factor), int(self.crop_size / 2 ** self.scale_factor)
        patch = (self.batch_size, 1, patch_h, patch_w)

        # Initialize generator and discriminator
        # self.generator = GeneratorResNet(ResidualDenseBlock_5Block_WithAttention, n_residual_blocks=12, n_basic_blocks=5,
        #                                  ca=True, sa=True, bca=True, bsa=True, cam=True, sam=True, cascading=True, withMaxPool=True, addconv=True)
        self.generator = TransENet(n_feats=64, scale=self.scale_factor, n_basic_modules=10, n_colors=3, hr_patch_size=self.crop_size, back_projection_iters=10, en_depth=8, de_depth=1, conv=default_conv)
        self.feature_extractor = FeatureExtractor()
        # self.feature_extractor = VGGFeatureExtractor(feature_layer=34, use_bn=False, use_input_norm=True, device=self.device)
        # srutils.print_network_to_file(self.generator, save_dir=self.save_dir, tag='Generator')
        # srutils.print_network_to_file(self.feature_extractor, save_dir=self.save_dir, tag='FeatureExtractor')

        # Losses
        # self.criterion_GAN = torch.nn.MSELoss().to(self.device)
        if self.loss_Lp_norm=="L1":
            self.criterion_content = torch.nn.L1Loss().to(self.device)
        else:
            self.criterion_content = torch.nn.MSELoss().to(self.device)

        if len(device_ids) > 1:
            self.generator = nn.DataParallel(self.generator).to(self.device)
            self.feature_extractor = nn.DataParallel(self.feature_extractor).to(self.device)

        if cuda:
            self.generator = self.generator.cuda(device_ids[0])
            self.feature_extractor = self.feature_extractor.cuda(device_ids[0])
            # self.criterion_GAN = self.criterion_GAN.cuda(device_ids[0])
            self.criterion_content = self.criterion_content.cuda(device_ids[0])


        model_dir = os.path.join(self.save_dir, 'model')
        if self.epoch != 0:
            G_model_path = model_dir+'/generator_param_epoch_%d.pkl' % self.epoch
            self.load_epoch_network(load_path=G_model_path, network=self.generator)
        else:
            self.generator.apply(srutils.weights_init_normal)

        # Optimizers
        optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=(self.b1, self.b2))

        # Inputs & targets memory allocation
        Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
        input_lr = Tensor(self.batch_size, self.num_channels, self.crop_size // self.scale_factor, self.crop_size // self.scale_factor)
        input_hr = Tensor(self.batch_size, self.num_channels, self.crop_size, self.crop_size)
        input_bc = Tensor(self.batch_size, self.num_channels, self.crop_size, self.crop_size)

        # Adversarial ground truths
        valid = Variable(Tensor(np.ones(patch)), requires_grad=False)
        fake = Variable(Tensor(np.zeros(patch)), requires_grad=False)
        if cuda:
            valid = valid.cuda()
            fake = fake.cuda()
        #load train dataset
        dataloader = self.load_dataset(dataset='train', max_samples=self.max_train_samples)
        #test_data_loader = self.load_dataset(dataset='test')


        sys.stdout = PrintLogger(os.path.join(self.root_dir, self.save_dir))
        # ----------
        #  Training
        # ----------
        print('Training is started.')
        avg_loss_G = []
        step = 0
        start_time = time.time()

        tensor_to_img = ToPILImage()
        global val_psnr
        global val_ssim
        global val_ergas
        global val_lpips
        global val_psnr_max
        global val_ssim_max
        global val_ergas_max
        global val_lpips_max
        global val_loss_no_improve_count
        global val_loss_noimprove_max_count
        global val_loss_best_step
        val_psnr = 0
        val_ssim = 0
        val_ergas = 0
        val_lpips = 0
        val_psnr_max = 0
        val_ssim_max = 0
        val_ergas_max = 10000
        val_lpips_max = 10000
        val_loss_no_improve_count = 0
        val_loss_noimprove_max_count = 5
        val_loss_best_step = 0
        global epoch
        epoch = self.epoch
        while epoch < self.num_epochs and self.lr >= 0.00001:
        # for epoch in range(self.epoch, self.num_epochs):
            # learning rate is decayed by a factor of 10 every 20 epochs
            # if (epoch + 1) % 10 == 0:
            #     for param_group in optimizer_G.param_groups:
            #         param_group["lr"] /= 2.0
            #     print("Learning rate decay: lr={}".format(optimizer_G.param_groups[0]["lr"]))
            # # if (epoch + 1) % 20 == 0:
            #     for param_group in optimizer_D.param_groups:
            #         param_group["lr"] /= 2.0
            #     print("Learning rate decay: lr={}".format(optimizer_D.param_groups[0]["lr"]))

            epoch_loss_G = 0
            for i, (input, target, bc_input, imgpath) in enumerate(dataloader):

                # Configure model input
                imgs_lr = Variable(input_lr.copy_(input))
                imgs_hr = Variable(input_hr.copy_(target))
                imgs_bc = Variable(input_bc.copy_(bc_input))

                # ------------------
                #  Train Generators
                # ------------------

                optimizer_G.zero_grad()

                # Generate a high resolution image from low resolution input
                gen_hr = self.generator(imgs_lr)

                # pixel loss
                mse_loss_G = self.criterion_content(gen_hr, imgs_hr)
                # Content loss
                gen_features = self.feature_extractor(gen_hr)
                real_features = Variable(self.feature_extractor(imgs_hr).data, requires_grad=False)
                loss_content = self.criterion_content(gen_features, real_features)
                loss_G = mse_loss_G + self.weight_content * loss_content
                # Total loss
                # loss_G = loss_content + 1e-3 * loss_GAN
                # loss_G = mse_loss_G + 2e-6 * loss_content + 1e-3 * loss_gan
                # loss_G = mse_loss_G + 1e-3 * loss_content + 1e-2 * loss_gan

                loss_G.backward()
                optimizer_G.step()
                # optimizer_G.module.step()
                # --------------
                #  Log Progress
                # --------------
                print("[Epoch %d/%d] [Batch %d/%d] [G loss: %f]" %
                      (epoch, self.num_epochs, i, len(dataloader), loss_G.item()))
                epoch_loss_G += loss_G.data.item()
                self.logger.scalar_summary('loss_G', loss_G.item(), step + 1)

                epoch_loss_G += loss_G.data.item()
                self.logger.scalar_summary('loss_G', loss_G.item(), step + 1)

                step += 1

                batches_done = epoch * len(dataloader) + i
                if batches_done % self.sample_interval == 0:
                    # Save image sample
                    # bc_imgs = srutils.img_interp(imgs_lr.cpu(), 4)
                    bc_img = tensor_to_img(imgs_bc[0].cpu())
                    bc_img = np.array(bc_img)

                    hr_img = tensor_to_img(imgs_hr[0].cpu())
                    hr_img = np.array(hr_img)

                    gen_img = tensor_to_img(gen_hr[0].cpu())
                    gen_img = np.array(gen_img)

                    # mse = compare_mse(gen_img, hr_img)
                    recon_psnr = compare_psnr(gen_img, hr_img)
                    recon_ssim = compare_ssim(gen_img, hr_img, multichannel=True)
                    recon_ergas = srutils.compare_ergas2(hr_img, gen_img, scale=self.scale_factor)
                    # recon_lpips = srutils.compare_lpips(imgs_hr[0], gen_hr[0], use_gpu=cuda)
                    recon_lpips = self.PerceptualModel.forward(gen_hr[0], imgs_hr[0]).detach().item()


                    # mse1 = compare_mse(bc_img, hr_img)
                    bc_psnr = compare_psnr(bc_img, hr_img)
                    bc_ssim = compare_ssim(bc_img, hr_img, multichannel=True)
                    bc_ergas = srutils.compare_ergas2(hr_img, bc_img, scale=self.scale_factor)
                    # bc_lpips = srutils.compare_lpips(imgs_hr[0], imgs_bc[0], use_gpu=cuda)
                    bc_lpips = self.PerceptualModel.forward(imgs_bc[0], imgs_hr[0]).detach().item()

                    result_imgs = [imgs_hr[0].cpu(), imgs_lr[0].cpu(), imgs_bc[0].cpu(), gen_hr[0].cpu()]
                    psnrs = [None, None, bc_psnr, recon_psnr]
                    ssims = [None, None, bc_ssim, recon_ssim]
                    ergas = [None, None, bc_ergas, recon_ergas]
                    lpips = [None, None, bc_lpips, recon_lpips]
                    indicators = {"PSNR":psnrs, "SSIM":ssims, "ERGAS":ergas, "LPIPS":lpips}
                    srutils.plot_result_by_name(result_imgs, indicators, batches_done, imgpath[0],
                                                     save_dir=self.save_dir, is_training=True, show=False)
                    # srutils.plot_test_result_by_name(result_imgs, psnrs, ssims, batches_done, imgpath[0], save_dir=self.save_dir, is_training=True, show=False)
                    # save_image(torch.cat((gen_hr.data, imgs_hr.data), -2),
                    #          'images/%d.png' % batches_done, normalize=True)
                    # log
                    time_elapsed = time.time() - start_time
                    print_rlt = OrderedDict()
                    print_rlt['model'] = self.model_name
                    print_rlt['epoch'] = epoch
                    print_rlt['iters'] = step
                    print_rlt['G_lr'] = optimizer_G.param_groups[0]['lr']
                    print_rlt['time'] = time_elapsed
                    print_rlt['G_loss'] = loss_G.item()
                    print_rlt['bicubic_psnr'] = bc_psnr
                    print_rlt['bicubic_ssim'] = bc_ssim
                    print_rlt['bicubic_ergas'] = bc_ergas
                    print_rlt['bicubic_lpips'] = bc_lpips
                    print_rlt['srwgan_psnr'] = recon_psnr
                    print_rlt['srwgan_ssim'] = recon_ssim
                    print_rlt['srwgan_ergas'] = recon_ergas
                    print_rlt['srwgan_lpips'] = recon_lpips
                    for k, v in self.log_dict.items():
                        print_rlt[k] = v
                    self.logger.print_format_results('train', print_rlt)

                    del bc_img, hr_img, gen_img, recon_psnr, recon_ssim, recon_ergas, recon_lpips, bc_ssim, bc_psnr, bc_ergas, bc_lpips, result_imgs, psnrs, ssims
                del imgs_hr, imgs_bc, imgs_lr, gen_hr, gen_features, real_features
                torch.cuda.empty_cache()

            # avg. loss per epoch
            avg_loss_G.append(epoch_loss_G / len(dataloader))

            val_psnr, val_ssim, val_ergas, val_lpips = self.validate(epoch=epoch, mode='train', save_img=((epoch + 1) % self.save_epochs == 0))

            if val_psnr - val_psnr_max > 0:
                val_psnr_max = val_psnr
                val_loss_no_improve_count = 0
                val_loss_best_step = epoch
            elif val_ssim - val_ssim_max > 0:
                val_ssim_max = val_ssim
                val_loss_no_improve_count = 0
                val_loss_best_step = epoch
            elif val_ergas - val_ergas_max < 0:
                val_ergas_max = val_ergas
                val_loss_no_improve_count = 0
                val_loss_best_step = epoch
            elif val_lpips - val_lpips_max < 0:
                val_lpips_max = val_lpips
                val_loss_no_improve_count = 0
                val_loss_best_step = epoch
            else:
                val_loss_no_improve_count = val_loss_no_improve_count + 1

            self.save_epoch_network(save_dir=model_dir, network=self.generator, network_label='generator',
                                    iter_label=epoch + 1)
            # global epoch
            epoch += 1
            if val_loss_no_improve_count >= val_loss_noimprove_max_count:
                G_model_path = model_dir + '/generator_param_epoch_%d.pkl' % (val_loss_best_step + 1)
                self.load_epoch_network(load_path=G_model_path, network=self.generator)
                for param_group in optimizer_G.param_groups:
                    param_group["lr"] /= 2.0
                print("Learning rate decay: lr={}".format(optimizer_G.param_groups[0]["lr"]))
                # global epoch
                epoch = val_loss_best_step + 1
                val_loss_no_improve_count = 0
            # if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
            #     # Save model checkpoints
            #     torch.save(generator.state_dict(), 'saved_models/generator_%d.pth' % epoch)
            #     torch.save(discriminator.state_dict(), 'saved_models/discriminator_%d.pth' % epoch)
            # Save trained parameters of model
            # if (epoch + 1) % self.save_epochs == 0:
            #     # self.save_model(epoch + 1)
            #     self.save_epoch_network(save_dir=model_dir, network=self.generator, network_label='generator', iter_label=epoch+1)
            #     self.save_epoch_network(save_dir=model_dir, network=self.discriminator, network_label='discriminator', iter_label=epoch+1)
            torch.cuda.empty_cache()
        # Plot avg. loss
        srutils.plot_loss([avg_loss_G], self.num_epochs, save_dir=self.save_dir)
        print("Training is finished.")

        # Save final trained parameters of model
        self.save_model(epoch=None)
    @torch.no_grad()
    def validate(self, epoch=0, mode='test', save_img=False):
        # networks
        cuda = True if torch.cuda.is_available() else False
        device_ids = [0,1]

        # load model
        if mode == 'test':
            # networks
            self.generator = TransENet(n_feats=64, scale=self.scale_factor, n_basic_modules=10, n_colors=3, hr_patch_size=self.crop_size, back_projection_iters=10, en_depth=8, de_depth=1, conv=default_conv)
            self.load_epoch_model(epoch)

        if self.gpu_mode:
            self.generator.cuda()

        # load dataset
        test_data_loader = self.load_dataset(dataset='test')

        # Inputs & targets memory allocation
        Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
        input_lr = Tensor(self.test_batch_size, self.num_channels, self.hr_height // self.scale_factor, self.hr_width // self.scale_factor)
        input_hr = Tensor(self.test_batch_size, self.num_channels, self.hr_height, self.hr_width)
        input_bc = Tensor(self.test_batch_size, self.num_channels, self.hr_height, self.hr_width)
        tensor_to_img = ToPILImage()
        # Test
        print('Test is started.')
        img_num = 0
        self.generator.eval()
        start_time = time.time()
        avg_bicubic_mse = 0.0
        avg_bicubic_psnr = 0.0
        avg_bicubic_ssim = 0.0
        avg_bicubic_ergas = 0.0
        avg_bicubic_lpips = 0.0
        avg_srcnn_mse = 0.0
        avg_srcnn_psnr = 0.0
        avg_srcnn_ssim = 0.0
        avg_srcnn_ergas = 0.0
        avg_srcnn_lpips = 0.0
        for iter, (input, target, bc_input, imgpath) in enumerate(test_data_loader):
            # Configure model input
            imgs_lr = Variable(input_lr.copy_(input))
            imgs_hr = Variable(input_hr.copy_(target))
            imgs_bc = Variable(input_bc.copy_(bc_input))

            # prediction
            recon_imgs = self.generator(imgs_lr)
            for i in range(self.test_batch_size):
                img_num += 1
                recon_img = recon_imgs[i].cpu().data
                gt_img = imgs_hr[i]  # utils.shave(target[i], border_size=8)
                # lr_img = imgs_lr[i]
                bc_img = imgs_bc[i]  # srutils.img_interp(lr_img.cpu(), 4)

                # calculate psnrs
                bc_img = tensor_to_img(bc_img.cpu())
                bc_img = np.array(bc_img)

                gt_img = tensor_to_img(gt_img.cpu())
                gt_img = np.array(gt_img)

                recon_img = tensor_to_img(recon_img.cpu())
                recon_img = np.array(recon_img)
                recon_mse = compare_mse(recon_img, gt_img)
                recon_psnr = compare_psnr(recon_img, gt_img)
                recon_ssim = compare_ssim(recon_img, gt_img, multichannel=True)
                recon_ergas = srutils.compare_ergas2(gt_img, recon_img, scale=self.scale_factor)
                # recon_lpips = srutils.compare_lpips(imgs_hr[i], recon_imgs[i], use_gpu=cuda)
                recon_lpips = self.PerceptualModel.forward(recon_imgs[i], imgs_hr[i]).detach().item()

                bc_mse = compare_mse(bc_img, gt_img)
                bc_psnr = compare_psnr(bc_img, gt_img)
                bc_ssim = compare_ssim(bc_img, gt_img, multichannel=True)
                bc_ergas = srutils.compare_ergas2(gt_img, bc_img, scale=self.scale_factor)
                # bc_lpips = srutils.compare_lpips(imgs_hr[i], imgs_bc[i], use_gpu=cuda)
                bc_lpips = self.PerceptualModel.forward(imgs_bc[i], imgs_hr[i]).detach().item()

                if save_img and iter % 50 == 0:
                    # save result images
                    result_imgs = [imgs_hr[i].cpu(), imgs_lr[i].cpu(), imgs_bc[i].cpu(), recon_imgs[i].cpu()]
                    psnrs = [None, None, bc_psnr, recon_psnr]
                    ssims = [None, None, bc_ssim, recon_ssim]
                    ergas = [None, None, bc_ergas, recon_ergas]
                    lpips = [None, None, bc_lpips, recon_lpips]
                    indicators = {"PSNR": psnrs, "SSIM": ssims, "ERGAS": ergas, "LPIPS": lpips}
                    # srutils.plot_test_result_by_name(result_imgs, psnrs, ssims, epoch, imgpath[i],
                                                     # save_dir=self.save_dir)
                    srutils.plot_result_by_name(result_imgs, indicators, epoch, imgpath[i],
                                                save_dir=self.save_dir)

                avg_bicubic_mse += bc_mse
                avg_bicubic_psnr += bc_psnr
                avg_bicubic_ssim += bc_ssim
                avg_bicubic_ergas += bc_ergas
                avg_bicubic_lpips += bc_lpips
                avg_srcnn_mse += recon_mse
                avg_srcnn_psnr += recon_psnr
                avg_srcnn_ssim += recon_ssim
                avg_srcnn_ergas += recon_ergas
                avg_srcnn_lpips += recon_lpips
                print("Saving %d test result images..." % img_num)

                del bc_img, gt_img, recon_img, recon_psnr, recon_ssim, recon_mse, recon_ergas, recon_lpips, bc_mse, bc_ssim, bc_psnr, bc_ergas, bc_lpips
            del imgs_hr, imgs_bc, imgs_lr, recon_imgs
            torch.cuda.empty_cache()

        avg_bicubic_mse = avg_bicubic_mse / img_num
        avg_bicubic_psnr = avg_bicubic_psnr / img_num
        avg_bicubic_ssim = avg_bicubic_ssim / img_num
        avg_bicubic_ergas = avg_bicubic_ergas / img_num
        avg_bicubic_lpips = avg_bicubic_lpips / img_num
        avg_srcnn_mse = avg_srcnn_mse / img_num
        avg_srcnn_psnr = avg_srcnn_psnr / img_num
        avg_srcnn_ssim = avg_srcnn_ssim / img_num
        avg_srcnn_ergas = avg_srcnn_ergas / img_num
        avg_srcnn_lpips = avg_srcnn_lpips / img_num
        time_elapsed = time.time() - start_time
        # Save to log
        print_rlt = OrderedDict()
        print_rlt['model'] = self.model_name
        print_rlt['epoch'] = epoch
        print_rlt['iters'] = epoch
        print_rlt['time'] = time_elapsed
        print_rlt['bicubic_mse'] = avg_bicubic_mse
        print_rlt['bicubic_psnr'] = avg_bicubic_psnr
        print_rlt['bicubic_ssim'] = avg_bicubic_ssim
        print_rlt['bicubic_ergas'] = avg_bicubic_ergas
        print_rlt['bicubic_lpips'] = avg_bicubic_lpips
        print_rlt['srcnn_mse'] = avg_srcnn_mse
        print_rlt['srcnn_psnr'] = avg_srcnn_psnr
        print_rlt['srcnn_ssim'] = avg_srcnn_ssim
        print_rlt['srcnn_ergas'] = avg_srcnn_ergas
        print_rlt['srcnn_lpips'] = avg_srcnn_lpips
        self.logger.print_format_results('val', print_rlt)
        print('-----------------------------------')
        del test_data_loader
        torch.cuda.empty_cache()
        return avg_srcnn_psnr, avg_srcnn_ssim, avg_srcnn_ergas, avg_srcnn_lpips

    @torch.no_grad()
    def validateByClass(self, epoch, mode='test', save_img=False):
        # networks
        cuda = True if torch.cuda.is_available() else False
        device_ids = [0, 1]
        self.device = torch.device('cuda' if cuda else 'cpu')
        # load model
        if mode == 'test':
            # networks
            # self.generator = GeneratorResNet(n_residual_blocks=28).to(self.device)
            self.generator = TransENet(n_feats=64, scale=self.scale_factor, n_basic_modules=10, n_colors=3, hr_patch_size=self.crop_size, back_projection_iters=10, en_depth=8, de_depth=1, conv=default_conv)
            self.load_epoch_model(epoch)
            # srutils.print_network_to_file(self.generator, save_dir=self.save_dir, tag='Generator')

        if self.gpu_mode:
            self.generator.cuda()

        if len(device_ids) > 1:
            self.generator = nn.DataParallel(self.generator).to(self.device)

        self.generator.eval()
        # Inputs & targets memory allocation
        Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
        input_lr = Tensor(self.batch_size, self.num_channels, self.hr_height // self.scale_factor, self.hr_width // self.scale_factor)
        input_hr = Tensor(self.batch_size, self.num_channels, self.hr_height, self.hr_width)
        input_bc = Tensor(self.batch_size, self.num_channels, self.hr_height, self.hr_width)
        # Test
        print('Test is started.')
        tensor_to_img = ToPILImage()
        start_time = time.time()
        average_bicubic_psnr = 0.0
        average_bicubic_ssim = 0.0
        average_bicubic_mse = 0.0
        average_bicubic_ergas = 0.0
        average_bicubic_lpips = 0.0
        average_srrgan_mse = 0.0
        average_srrgan_psnr = 0.0
        average_srrgan_ssim = 0.0
        average_srrgan_ergas = 0.0
        average_srrgan_lpips = 0.0
        num = 0
        # create train and val dataloader
        for datasetDir in self.test_dataset:

            test_set = get_RGB_testDataset(self.data_dir, [datasetDir], self.crop_size, self.scale_factor,
                                                 is_gray=False)
            test_data_loader = DataLoader(dataset=test_set, num_workers=self.num_threads,
                              batch_size=self.test_batch_size,
                              shuffle=False)
            # val_loader.append(create_dataloader(val_set, dataset_opt))
            print('Number of val images in [{:s}]: {:d}'.format(datasetDir, len(test_data_loader)))
            datasetName = os.path.split(datasetDir)[-1]

            avg_bicubic_psnr = 0.0
            avg_bicubic_ssim = 0.0
            avg_bicubic_mse = 0.0
            avg_bicubic_ergas = 0.0
            avg_bicubic_lpips = 0.0
            avg_srcnn_mse = 0.0
            avg_srcnn_psnr = 0.0
            avg_srcnn_ssim = 0.0
            avg_srcnn_ergas = 0.0
            avg_srcnn_lpips = 0.0
            img_num = 0
            for i, (input, target, bc_input, imgpath) in enumerate(test_data_loader):
                # Configure model input
                imgs_lr = Variable(input_lr.copy_(input))
                imgs_hr = Variable(input_hr.copy_(target))
                imgs_bc = Variable(input_bc.copy_(bc_input))

                # prediction
                recon_imgs = self.generator(imgs_lr)
                for i in range(self.test_batch_size):
                    img_num += 1
                    num += 1
                    recon_img = recon_imgs[i].cpu().data
                    gt_img = imgs_hr[i]  # utils.shave(target[i], border_size=8)
                    lr_img = imgs_lr[i]
                    bc_img = imgs_bc[i]  # srutils.img_interp(lr_img.cpu(), 4)

                    # calculate psnrs
                    bc_img = tensor_to_img(bc_img.cpu())
                    bc_img = np.array(bc_img)

                    gt_img = tensor_to_img(gt_img.cpu())
                    gt_img = np.array(gt_img)

                    recon_img = tensor_to_img(recon_img.cpu())
                    recon_img = np.array(recon_img)
                    recon_mse = compare_mse(recon_img, gt_img)
                    recon_psnr = compare_psnr(recon_img, gt_img)
                    recon_ssim = compare_ssim(recon_img, gt_img, multichannel=True)
                    recon_ergas = srutils.compare_ergas2(gt_img, recon_img, scale=self.scale_factor)
                    # recon_lpips = srutils.compare_lpips(imgs_hr[i], recon_imgs[i], use_gpu=cuda)
                    recon_lpips = self.PerceptualModel.forward(imgs_hr[i], recon_imgs[i]).detach().item()

                    bc_mse = compare_mse(bc_img, gt_img)
                    bc_psnr = compare_psnr(bc_img, gt_img)
                    bc_ssim = compare_ssim(bc_img, gt_img, multichannel=True)
                    bc_ergas = srutils.compare_ergas2(gt_img, bc_img, scale=self.scale_factor)
                    # bc_lpips = srutils.compare_lpips(imgs_hr[i], imgs_bc[i], use_gpu=cuda)
                    bc_lpips = self.PerceptualModel.forward(imgs_hr[i], imgs_bc[i]).detach().item()

                    if save_img:
                        # save result images
                        result_imgs = [imgs_hr[i].cpu(), imgs_lr[i].cpu(), imgs_bc[i].cpu(), recon_imgs[i].cpu()]
                        psnrs = [None, None, bc_psnr, recon_psnr]
                        ssims = [None, None, bc_ssim, recon_ssim]
                        ergas = [None, None, bc_ergas, recon_ergas]
                        lpips = [None, None, bc_lpips, recon_lpips]
                        indicators = {"PSNR": psnrs, "SSIM": ssims, "ERGAS": ergas, "LPIPS": lpips}
                        srutils.plot_result_by_name(result_imgs, indicators, epoch, imgpath[i], is_training=False,
                                                    is_validation=True, classname=datasetName, save_dir=self.save_dir)
                        # srutils.plot_test_result_by_class(result_imgs, psnrs, ssims, epoch, datasetName, imgpath[i],
                        #                                  save_dir=self.save_dir)
                        img_name = os.path.splitext(os.path.basename(imgpath[i]))[0]
                        img_dir = os.path.join(self.save_dir, 'validate', datasetName, '{:s}_x{:d}_{:d}.png'.format(img_name, self.scale_factor, epoch))
                        srutils.save_img1(recon_imgs[i].cpu(), os.path.join(self.save_dir, 'validate'), img_dir)

                    avg_bicubic_mse += bc_mse
                    avg_bicubic_psnr += bc_psnr
                    avg_bicubic_ssim += bc_ssim
                    avg_bicubic_ergas += bc_ergas
                    avg_bicubic_lpips += bc_lpips
                    avg_srcnn_mse += recon_mse
                    avg_srcnn_psnr += recon_psnr
                    avg_srcnn_ssim += recon_ssim
                    avg_srcnn_ergas += recon_ergas
                    avg_srcnn_lpips += recon_lpips

                    average_bicubic_mse += bc_mse
                    average_bicubic_psnr += bc_psnr
                    average_bicubic_ssim += bc_ssim
                    average_bicubic_ergas += bc_ergas
                    average_bicubic_lpips += bc_lpips
                    average_srrgan_mse += recon_mse
                    average_srrgan_psnr += recon_psnr
                    average_srrgan_ssim += recon_ssim
                    average_srrgan_ergas += recon_ergas
                    average_srrgan_lpips += recon_lpips
                    # loss_mse += mse
                    print("Saving %d test result images..." % img_num)

                    del bc_img, gt_img, recon_img, recon_mse, recon_psnr, recon_ssim, recon_ergas, recon_lpips, bc_mse, bc_ssim, bc_psnr, bc_ergas, bc_lpips
                del imgs_hr, imgs_bc, imgs_lr, recon_imgs

            avg_bicubic_mse = avg_bicubic_mse / img_num
            avg_bicubic_psnr = avg_bicubic_psnr / img_num
            avg_bicubic_ssim = avg_bicubic_ssim / img_num
            avg_bicubic_ergas = avg_bicubic_ergas / img_num
            avg_bicubic_lpips = avg_bicubic_lpips / img_num
            avg_srcnn_mse = avg_srcnn_mse / img_num
            avg_srcnn_psnr = avg_srcnn_psnr / img_num
            avg_srcnn_ssim = avg_srcnn_ssim / img_num
            avg_srcnn_ergas = avg_srcnn_ergas / img_num
            avg_srcnn_lpips = avg_srcnn_lpips / img_num
            time_elapsed = time.time() - start_time
            # Save to log
            print_rlt = OrderedDict()
            print_rlt['model'] = datasetName
            print_rlt['epoch'] = epoch
            print_rlt['iters'] = epoch
            print_rlt['time'] = time_elapsed
            print_rlt['bicubic_mse'] = avg_bicubic_mse
            print_rlt['bicubic_psnr'] = avg_bicubic_psnr
            print_rlt['bicubic_ssim'] = avg_bicubic_ssim
            print_rlt['bicubic_ergas'] = avg_bicubic_ergas
            print_rlt['bicubic_lpips'] = avg_bicubic_lpips
            print_rlt['srcnn_mse'] = avg_srcnn_mse
            print_rlt['srcnn_psnr'] = avg_srcnn_psnr
            print_rlt['srcnn_ssim'] = avg_srcnn_ssim
            print_rlt['srcnn_ergas'] = avg_srcnn_ergas
            print_rlt['srcnn_lpips'] = avg_srcnn_lpips
            self.logger.print_format_results('val', print_rlt)

            del test_data_loader
            del test_set

        average_bicubic_mse = average_bicubic_mse / num
        average_bicubic_psnr = average_bicubic_psnr / num
        average_bicubic_ssim = average_bicubic_ssim / num
        average_bicubic_ergas = average_bicubic_ergas / num
        average_bicubic_lpips = average_bicubic_lpips / num
        average_srrgan_mse = average_srrgan_mse / num
        average_srrgan_psnr = average_srrgan_psnr / num
        average_srrgan_ssim = average_srrgan_ssim / num
        average_srrgan_ergas = average_srrgan_ergas / num
        average_srrgan_lpips = average_srrgan_lpips / num
        time_elapsed = time.time() - start_time
        # Save to log
        print_rlt = OrderedDict()
        print_rlt['model'] = "Total"  # opt['model']
        print_rlt['epoch'] = epoch
        print_rlt['iters'] = epoch
        print_rlt['time'] = time_elapsed
        print_rlt['bicubic_mse'] = average_bicubic_mse
        print_rlt['bicubic_psnr'] = average_bicubic_psnr
        print_rlt['bicubic_ssim'] = average_bicubic_ssim
        print_rlt['bicubic_ergas'] = average_bicubic_ergas
        print_rlt['bicubic_lpips'] = average_bicubic_lpips
        print_rlt['srrgan_mse'] = average_srrgan_mse
        print_rlt['srrgan_psnr'] = average_srrgan_psnr
        print_rlt['srrgan_ssim'] = average_srrgan_ssim
        print_rlt['srrgan_ergas'] = average_srrgan_ergas
        print_rlt['srrgan_lpips'] = average_srrgan_lpips
        self.logger.print_format_results('val', print_rlt)
        print('-----------------------------------')
        torch.cuda.empty_cache()

    def test(self):
        # networks
        cuda = True if torch.cuda.is_available() else False
        # Initialize generator and discriminator
        self.generator = TransENet()

        if cuda:
            self.generator = self.generator.cuda()

        # load model
        self.load_model()

        # Inputs & targets memory allocation
        Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
        input_lr = Tensor(self.batch_size, self.num_channels, self.hr_height // self.scale_factor, self.hr_width // self.scale_factor)
        input_hr = Tensor(self.batch_size, self.num_channels, self.hr_height, self.hr_width)
        input_bc = Tensor(self.batch_size, self.num_channels, self.hr_height, self.hr_width)

        tensor_to_img = ToPILImage()

        # load test dataset
        test_data_loader = self.load_dataset(dataset='test')

        # Test
        print('Test is started.')
        img_num = 0
        for i, (input, target, bc_input) in enumerate(test_data_loader):
            # Configure model input
            imgs_lr = Variable(input_lr.copy_(input))
            imgs_hr = Variable(input_hr.copy_(target))
            imgs_bc = Variable(input_bc.copy_(bc_input))

            # prediction
            recon_imgs = self.generator(imgs_lr)
            for i in range(self.test_batch_size):
                img_num += 1
                recon_img = recon_imgs[i].cpu().data
                gt_img = imgs_hr[i]  # utils.shave(target[i], border_size=8)
                lr_img = imgs_lr[i]
                bc_img = imgs_bc[i]  #srutils.img_interp(lr_img.cpu(), 4)

                # calculate psnrs
                bc_img = tensor_to_img(bc_img.cpu())
                bc_img = np.array(bc_img)

                gt_img = tensor_to_img(gt_img.cpu())
                gt_img = np.array(gt_img)

                recon_img = tensor_to_img(recon_img.cpu())
                recon_img = np.array(recon_img)
                # mse = compare_mse(recon_imgs[0], imgs_hr[0])
                recon_psnr = compare_psnr(recon_img, gt_img)
                recon_ssim = compare_ssim(recon_img, gt_img, multichannel=True)

                mse1 = compare_mse(bc_img, gt_img)
                bc_psnr = compare_psnr(bc_img, gt_img)
                bc_ssim = compare_ssim(bc_img, gt_img, multichannel=True)
                # save result images
                result_imgs = [imgs_hr[i].cpu(), imgs_lr[i].cpu(), imgs_bc[i].cpu(), recon_imgs[i].cpu()]
                psnrs = [None, None, bc_psnr, recon_psnr]
                ssims = [None, None, bc_ssim, recon_ssim]
                srutils.plot_test_result2(result_imgs, psnrs, ssims, img_num, save_dir=self.save_dir)

                print("Saving %d test result images..." % img_num)

    def validate_single(self, epoch, imgpath):
        # networks
        #cuda = True if torch.cuda.is_available() else False
        # Initialize generator and discriminator
        self.generator = TransENet()

        # if cuda:
        #     self.generator = self.generator.cuda()

        # load model
        self.load_epoch_model(epoch)
        self.generator.eval()

        s_attention_map = srutils.LayerActivations(self.generator.res_blocks[15].conv5.sa)
        # c_attention_map = srutils.LayerActivations(self.generator.res_blocks[15].ca)

        img = Image.open(imgpath)
        toTensor = transforms.Compose([
            #transforms.CenterCrop(),
            transforms.ToTensor()
        ])
        input = toTensor(img)
        size = list(input.shape)
        input_lr = torch.Tensor(1, size[0], size[1], size[2])
        input_lr[0, :, :, :] = input
        bc_imgs = srutils.img_interp(input_lr, self.scale_factor)

        recon_imgs = self.generator(input_lr.cpu())
        s_attention_map.remove()
        s_attentionmap = s_attention_map.features
        # c_attention_map.remove()
        # c_attentionmap = c_attention_map.features

        img_name = os.path.splitext(os.path.basename(imgpath))[0]
        img_dir = os.path.join(self.save_dir, 'validate', '{:s}_{:d}.png'.format(img_name, epoch))
        img_dir1 = os.path.join(self.save_dir, 'validate', 's_attention_{:s}_{:d}.png'.format(img_name, epoch))
        img_dir2 = os.path.join(self.save_dir, 'validate', 'c_attention_{:s}_{:d}.png'.format(img_name, epoch))
        # compare_img_dir = os.path.join(self.save_dir, 'validate', 'compare_{:s}_{:d}.png'.format(img_name, epoch))
        srutils.save_img1(recon_imgs[0].cpu(), os.path.join(self.save_dir, 'validate'), img_dir)
        # srutils.save_img1(bc_imgs[0].cpu(), img_dir1)
        srutils.save_img1(s_attentionmap[0].cpu(), os.path.join(self.save_dir, 'validate'), img_dir1)
        # srutils.save_img1(c_attentionmap[0].cpu(), os.path.join(self.save_dir, 'validate'), img_dir2)
        # srutils.plot_test_single_result([input_lr[0], bc_imgs[0].cpu(), recon_imgs[0].cpu()], [0, 0, 0], [0, 0, 0],
        #                                 save_path=compare_img_dir)

    def test_single(self, img_fn):
        # networks
        cuda = True if torch.cuda.is_available() else False
        # Initialize generator and discriminator
        self.generator = TransENet()

        if cuda:
            self.generator = self.generator.cuda()

        # load model
        self.load_model()

        tensor_to_img = ToPILImage()
        # load data
        img = Image.open(img_fn)
        # lr_transform = transforms.Compose(self.lr_transforms)
        # hr_transform = transforms.Compose(self.hr_transforms)
        toTensor = transforms.Compose([
            transforms.CenterCrop(self.test_crop_size),
            transforms.ToTensor()
        ])
        input = toTensor(img)
        # input = Variable(ToTensor()(img)).view(1, -1, img.size[1], img.size[0])
        # if self.gpu_mode:
        #     input = input.cuda()
        # Inputs & targets memory allocation
        Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
        input_lr = Tensor(self.batch_size, self.num_channels, self.test_crop_size, self.test_crop_size)
        input_img = Variable(input_lr.copy_(input))
        bc_img = srutils.img_interp(input_img.cpu(), self.scale_factor)

        # img_lr = lr_transform(img)
        # img_hr = hr_transform(img)

        # prediction
        recon_imgs = self.generator(input_img)
        recon_img = recon_imgs[0]
        # if list(recon_img.shape)[0] == 3:
        #     # Scale to 0-255
        #     # img = (((img - img.min()) * 255) / (img.max() - img.min())).numpy().transpose(1, 2, 0).astype(np.uint8)
        #     recon_img *= 255.0
        #     recon_img = recon_img.cpu().clamp(0, 255).detach().numpy().transpose(1, 2, 0).astype(np.uint8)
        # else:
        #     # img = ((img - img.min()) / (img.max() - img.min())).numpy().transpose(1, 2, 0)
        #     recon_img = recon_img.cpu().squeeze().clamp(0, 1).detach().numpy()

        # save result images
        img_name = img_fn.split("/")[-1]
        # save_image(torch.cat((recon_imgs[0].cpu().data, bc_img[0].cpu().data), -2),
        #            self.save_dir+'/SR_%s' % img_name, normalize=False)
        srutils.save_img1(recon_imgs[0].cpu().data, self.save_dir+'/SR_GAN_%s' % img_name)
        srutils.save_img1(bc_img[0].cpu().data, self.save_dir+'/SR_Bicubic_%s' % img_name)
        srutils.plot_test_single_result([recon_imgs[0].cpu(), bc_img[0].cpu()], [0,0], [0,0], save_path=self.save_dir+'/SR_%s' % img_name)

        # recon_img = recon_imgs[0].cpu().data
        # gt_img = img_hr  # utils.shave(target[i], border_size=8)
        # lr_img = img_lr
        # bc_imgs = srutils.img_interp(lr_img.cpu(), self.scale_factor)
        #
        # # calculate psnrs
        # bc_img = tensor_to_img(bc_imgs)
        # bc_img = np.array(bc_img)
        #
        # gt_img = tensor_to_img(gt_img.cpu())
        # gt_img = np.array(gt_img)
        #
        # recon_img = tensor_to_img(recon_img.cpu())
        # recon_img = np.array(recon_img)
        # # mse = compare_mse(recon_imgs[0], imgs_hr[0])
        # recon_psnr = compare_psnr(recon_img, gt_img)
        # recon_ssim = compare_ssim(recon_img, gt_img, multichannel=True)
        #
        # mse1 = compare_mse(bc_img, gt_img)
        # bc_psnr = compare_psnr(bc_img, gt_img)
        # bc_ssim = compare_ssim(bc_img, gt_img, multichannel=True)
        # # save result images
        # result_imgs = [img_hr.cpu(), img_lr.cpu(), bc_imgs.cpu(), recon_imgs.cpu()]
        # psnrs = [None, None, bc_psnr, recon_psnr]
        # ssims = [None, None, bc_ssim, recon_ssim]

        # save img
        # result_dir = os.path.join(self.save_dir, 'result')
        # if not os.path.exists(result_dir):
        #     os.mkdir(result_dir)
        # save_fn = result_dir + '/SR_result.png'
        # srutils.plot_test_result3(result_imgs, psnrs, ssims, save_path=save_fn)

    # helper saving function that can be used by subclasses
    def save_epoch_network(self, save_dir, network, network_label, iter_label):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_filename = '{}_param_epoch_{}.pkl'.format(network_label, iter_label)
        save_path = os.path.join(save_dir, save_filename)
        if isinstance(network, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path)

    # helper loading function that can be used by subclasses
    def load_epoch_network(self, load_path, network, strict=True):
        if isinstance(network, nn.DataParallel):
            network = network.module
        network.load_state_dict(torch.load(load_path), strict=strict)
        print('Trained model is loaded.')

    def save_model(self, epoch=None):
        model_dir = os.path.join(self.save_dir, 'model')
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        if epoch is not None:
            torch.save(self.generator.state_dict(), model_dir + '/generator_param_epoch_%d.pkl' % epoch)
        else:
            torch.save(self.generator.state_dict(), model_dir + '/generator_param.pkl')

        print('Trained model is saved.')

    def load_model(self):
        model_dir = os.path.join(self.save_dir, 'model')

        model_name = model_dir + '/generator_param.pkl'
        if os.path.exists(model_name):
            self.generator.load_state_dict(torch.load(model_name))
            print('Trained model is loaded.')
            return True
        else:
            print('No model exists to load.')
            return False

    def load_epoch_model(self, epoch):
        model_dir = os.path.join(self.save_dir, 'model')
        model_name = model_dir + '/generator_param_epoch_%d.pkl' % epoch
        if os.path.exists(model_name):
            # model_dict_G = torch.load(model_dir + '/generator_param_epoch_%d.pkl' % self.epoch)
            # trans_param_G = self.generator.state_dict()
            # for item, value in model_dict_G.items():
            #     name = '.'.join(item.split('.')[1:])
            #     trans_param_G[name] = value
            # self.generator.load_state_dict(trans_param_G)
            self.generator.load_state_dict(torch.load(model_name))
            print('Trained model is loaded.')
            return True
        else:
            print('No model exists to load.')
            return False


    def mfeNew_validate(self, epoch=100, modelpath=None):
        # networks
        cuda = True if torch.cuda.is_available() else False

        # load model
        self.generator = TransENet(n_feats=64, scale=self.scale_factor, n_basic_modules=10, n_colors=3, hr_patch_size=self.crop_size, back_projection_iters=10, en_depth=8, de_depth=1, conv=default_conv)
        if self.gpu_mode:
            self.generator.cuda()

        if modelpath is not None:
            self.generator.load_state_dict(torch.load(modelpath), strict=False)

        # self.generator.eval()

        # load dataset
        test_data_loader = self.load_dataset(dataset='test')

        # Inputs & targets memory allocation
        Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
        input_lr = Tensor(self.test_batch_size, self.num_channels, self.hr_height // self.scale_factor, self.hr_width // self.scale_factor)
        input_hr = Tensor(self.test_batch_size, self.num_channels, self.hr_height, self.hr_width)
        input_bc = Tensor(self.test_batch_size, self.num_channels, self.hr_height, self.hr_width)
        tensor_to_img = ToPILImage()
        # Test
        print('Test is started.')
        img_num = 0
        self.generator.eval()
        start_time = time.time()
        avg_bicubic_mse = 0.0
        avg_bicubic_psnr = 0.0
        avg_bicubic_ssim = 0.0
        avg_bicubic_ergas = 0.0
        avg_bicubic_lpips = 0.0
        avg_srcnn_mse = 0.0
        avg_srcnn_psnr = 0.0
        avg_srcnn_ssim = 0.0
        avg_srcnn_ergas = 0.0
        avg_srcnn_lpips = 0.0
        for iter, (input, target, bc_input, imgpath) in enumerate(test_data_loader):
            # Configure model input
            imgs_lr = Variable(input_lr.copy_(input))
            imgs_hr = Variable(input_hr.copy_(target))
            imgs_bc = Variable(input_bc.copy_(bc_input))

            # prediction
            recon_imgs = self.generator(imgs_lr)
            for i in range(self.test_batch_size):
                img_num += 1
                recon_img = recon_imgs[i].cpu().data
                gt_img = imgs_hr[i]  # utils.shave(target[i], border_size=8)
                # lr_img = imgs_lr[i]
                bc_img = imgs_bc[i]  # srutils.img_interp(lr_img.cpu(), 4)

                # calculate psnrs
                bc_img = tensor_to_img(bc_img.cpu())
                bc_img = np.array(bc_img)

                gt_img = tensor_to_img(gt_img.cpu())
                gt_img = np.array(gt_img)

                recon_img = tensor_to_img(recon_img.cpu())
                recon_img = np.array(recon_img)
                recon_mse = compare_mse(recon_img, gt_img)
                recon_psnr = compare_psnr(recon_img, gt_img)
                recon_ssim = compare_ssim(recon_img, gt_img, multichannel=True)
                recon_ergas = srutils.compare_ergas2(gt_img, recon_img, scale=self.scale_factor)
                recon_lpips = self.PerceptualModel.forward(recon_imgs[i], imgs_hr[i], normalize=True).detach().item()

                bc_mse = compare_mse(bc_img, gt_img)
                bc_psnr = compare_psnr(bc_img, gt_img)
                bc_ssim = compare_ssim(bc_img, gt_img, multichannel=True)
                bc_ergas = srutils.compare_ergas2(gt_img, bc_img, scale=self.scale_factor)
                bc_lpips = self.PerceptualModel.forward(imgs_bc[i], imgs_hr[i], normalize=True).detach().item()



                result_imgs = [imgs_hr[i].cpu(), imgs_lr[i].cpu(), imgs_bc[i].cpu(), recon_imgs[i].cpu()]
                mses = [None, None, bc_mse, recon_mse]
                psnrs = [None, None, bc_psnr, recon_psnr]
                ssims = [None, None, bc_ssim, recon_ssim]
                ergas = [None, None, bc_ergas, recon_ergas]
                lpips = [None, None, bc_lpips, recon_lpips]
                srutils.mfe_plot_test_result2(result_imgs, mses, psnrs, ssims, ergas, lpips, img_num, save_dir=self.save_dir)

                avg_bicubic_mse += bc_mse
                avg_bicubic_psnr += bc_psnr
                avg_bicubic_ssim += bc_ssim
                avg_bicubic_ergas += bc_ergas
                avg_bicubic_lpips += bc_lpips
                avg_srcnn_mse += recon_mse
                avg_srcnn_psnr += recon_psnr
                avg_srcnn_ssim += recon_ssim
                avg_srcnn_ergas += recon_ergas
                avg_srcnn_lpips += recon_lpips
                print("Saving the %d test dataset images" % img_num)

                del bc_img, gt_img, recon_img, recon_psnr, recon_ssim, recon_mse, recon_ergas, recon_lpips, bc_mse, bc_ssim, bc_psnr, bc_ergas, bc_lpips
            del imgs_hr, imgs_bc, imgs_lr, recon_imgs
            torch.cuda.empty_cache()

        avg_bicubic_mse = avg_bicubic_mse / img_num
        avg_bicubic_psnr = avg_bicubic_psnr / img_num
        avg_bicubic_ssim = avg_bicubic_ssim / img_num
        avg_bicubic_ergas = avg_bicubic_ergas / img_num
        avg_bicubic_lpips = avg_bicubic_lpips / img_num
        avg_srcnn_mse = avg_srcnn_mse / img_num
        avg_srcnn_psnr = avg_srcnn_psnr / img_num
        avg_srcnn_ssim = avg_srcnn_ssim / img_num
        avg_srcnn_ergas = avg_srcnn_ergas / img_num
        avg_srcnn_lpips = avg_srcnn_lpips / img_num
        time_elapsed = time.time() - start_time
        # Save to log
        print_rlt = OrderedDict()
        print_rlt['model'] = self.model_name
        print_rlt['epoch'] = epoch
        print_rlt['iters'] = epoch
        print_rlt['time'] = time_elapsed
        print_rlt['bicubic_mse'] = avg_bicubic_mse
        print_rlt['bicubic_psnr'] = avg_bicubic_psnr
        print_rlt['bicubic_ssim'] = avg_bicubic_ssim
        print_rlt['bicubic_ergas'] = avg_bicubic_ergas
        print_rlt['bicubic_lpips'] = avg_bicubic_lpips
        print_rlt['transenet_mse'] = avg_srcnn_mse
        print_rlt['transenet_psnr'] = avg_srcnn_psnr
        print_rlt['transenet_ssim'] = avg_srcnn_ssim
        print_rlt['transenet_ergas'] = avg_srcnn_ergas
        print_rlt['transenet_lpips'] = avg_srcnn_lpips
        self.logger.print_format_results('val', print_rlt)
        print('-----------------------------------')
        del test_data_loader
        torch.cuda.empty_cache()
        return avg_srcnn_psnr, avg_srcnn_ssim, avg_srcnn_ergas, avg_srcnn_lpips



    def mfeNew_validateByClass(self, epoch, save_img=False, modelpath=None):
        # networks
        cuda = True if torch.cuda.is_available() else False
        # load model
        self.generator = TransENet(n_feats=64, scale=self.scale_factor, n_basic_modules=10, n_colors=3, hr_patch_size=self.crop_size, back_projection_iters=10, en_depth=8, de_depth=1, conv=default_conv)

        if modelpath is not None:
            self.generator.load_state_dict(torch.load(modelpath), strict=False)
        #srutils.print_network_to_file(self.generator, save_dir=self.save_dir, tag='TransENet')

        if self.gpu_mode:
            self.generator.cuda()

        self.generator.eval()
        # Inputs & targets memory allocation
        Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
        input_lr = Tensor(self.test_batch_size, self.num_channels, self.hr_height // self.scale_factor, self.hr_width // self.scale_factor)
        input_hr = Tensor(self.test_batch_size, self.num_channels, self.hr_height, self.hr_width)
        input_bc = Tensor(self.test_batch_size, self.num_channels, self.hr_height, self.hr_width)
        # Test
        print('Test is started.')
        tensor_to_img = ToPILImage()
        start_time = time.time()
        average_bicubic_psnr = 0.0
        average_bicubic_ssim = 0.0
        average_bicubic_mse = 0.0
        average_bicubic_ergas = 0.0
        average_bicubic_lpips = 0.0
        average_srrgan_mse = 0.0
        average_srrgan_psnr = 0.0
        average_srrgan_ssim = 0.0
        average_srrgan_ergas = 0.0
        average_srrgan_lpips = 0.0
        num = 0
        # create train and val dataloader
        testDataset_dirs = []
        dataset = self.test_dataset[0]
        if dataset == 'Test':
            file_path = os.path.join(self.data_dir, dataset)
            dirs = sorted(os.listdir(file_path))
            for dir in dirs:
                dir_path = os.path.join(file_path, dir)
                if os.path.isdir(dir_path):
                    testDataset_dirs.append(dir_path)  # 最后一层类别文件夹
        for datasetDir in testDataset_dirs:
            datasetDir_list = []
            datasetDir_list.append(datasetDir)
            test_set = get_RGB_testDataset(self.data_dir, datasetDir_list, self.crop_size, self.scale_factor,
                                    is_gray=False)
            test_data_loader = DataLoader(dataset=test_set, num_workers=self.num_threads, batch_size=self.test_batch_size, shuffle=False, drop_last=True)
            print('Number of val images in [{:s}]: {:d}'.format(datasetDir, len(test_data_loader)))
            datasetName = os.path.split(datasetDir)[-1]

            avg_bicubic_psnr = 0.0
            avg_bicubic_ssim = 0.0
            avg_bicubic_mse = 0.0
            avg_bicubic_ergas = 0.0
            avg_bicubic_lpips = 0.0
            avg_srcnn_mse = 0.0
            avg_srcnn_psnr = 0.0
            avg_srcnn_ssim = 0.0
            avg_srcnn_ergas = 0.0
            avg_srcnn_lpips = 0.0
            img_num = 0
            for i, (input, target, bc_input, imgpath) in enumerate(test_data_loader):
                # Configure model input
                imgs_lr = Variable(input_lr.copy_(input))
                imgs_hr = Variable(input_hr.copy_(target))
                imgs_bc = Variable(input_bc.copy_(bc_input))

                # prediction
                recon_imgs = self.generator(imgs_lr)
                for i in range(self.test_batch_size):
                    img_num += 1
                    num += 1
                    recon_img = recon_imgs[i].cpu().data
                    gt_img = imgs_hr[i]  # utils.shave(target[i], border_size=8)
                    lr_img = imgs_lr[i]
                    bc_img = imgs_bc[i]  # srutils.img_interp(lr_img.cpu(), 4)

                    # calculate psnrs
                    bc_img = tensor_to_img(bc_img.cpu())
                    bc_img = np.array(bc_img)

                    gt_img = tensor_to_img(gt_img.cpu())
                    gt_img = np.array(gt_img)

                    recon_img = tensor_to_img(recon_img.cpu())
                    recon_img = np.array(recon_img)
                    recon_mse = compare_mse(recon_img, gt_img)
                    recon_psnr = compare_psnr(recon_img, gt_img)
                    recon_ssim = compare_ssim(recon_img, gt_img, multichannel=True)
                    recon_ergas = srutils.compare_ergas2(gt_img, recon_img, scale=self.scale_factor)
                    # recon_lpips = srutils.compare_lpips(gt_img, recon_img, use_gpu=cuda)
                    recon_lpips = self.PerceptualModel.forward(recon_imgs[i], imgs_hr[i], normalize=True).detach().item()

                    bc_mse = compare_mse(bc_img, gt_img)
                    bc_psnr = compare_psnr(bc_img, gt_img)
                    bc_ssim = compare_ssim(bc_img, gt_img, multichannel=True)
                    bc_ergas = srutils.compare_ergas2(gt_img, bc_img, scale=self.scale_factor)
                    # bc_lpips = srutils.compare_lpips(gt_img, bc_img, use_gpu=cuda)
                    bc_lpips = self.PerceptualModel.forward(imgs_bc[i], imgs_hr[i], normalize=True).detach().item()

                    if save_img:
                        # save result images
                        result_imgs = [imgs_hr[i].cpu(), imgs_lr[i].cpu(), imgs_bc[i].cpu(), recon_imgs[i].cpu()]
                        mses = [None, None, bc_mse, recon_mse]
                        psnrs = [None, None, bc_psnr, recon_psnr]
                        ssims = [None, None, bc_ssim, recon_ssim]
                        ergas = [None, None, bc_ergas, recon_ergas]
                        lpips = [None, None, bc_lpips, recon_lpips]
                        indicators = {"PSNR": psnrs, "SSIM": ssims, "ERGAS": ergas, "LPIPS": lpips}
                        srutils.plot_result_by_name(result_imgs, indicators, epoch, imgpath[i], is_training=False,
                                                    is_validation=True, classname=datasetName, save_dir=self.save_dir)
                        img_name = os.path.splitext(os.path.basename(imgpath[i]))[0]
                        img_dir = os.path.join(self.save_dir, 'validate', datasetName, '{:s}_x{:d}_{:d}.png'.format(img_name, self.scale_factor, epoch))
                        srutils.save_img1(recon_imgs[i].cpu(), os.path.join(self.save_dir, 'validate'), img_dir, cuda=cuda)

                    avg_bicubic_mse += bc_mse
                    avg_bicubic_psnr += bc_psnr
                    avg_bicubic_ssim += bc_ssim
                    avg_bicubic_ergas += bc_ergas
                    avg_bicubic_lpips += bc_lpips
                    avg_srcnn_mse += recon_mse
                    avg_srcnn_psnr += recon_psnr
                    avg_srcnn_ssim += recon_ssim
                    avg_srcnn_ergas += recon_ergas
                    avg_srcnn_lpips += recon_lpips

                    average_bicubic_mse += bc_mse
                    average_bicubic_psnr += bc_psnr
                    average_bicubic_ssim += bc_ssim
                    average_bicubic_ergas += bc_ergas
                    average_bicubic_lpips += bc_lpips
                    average_srrgan_mse += recon_mse
                    average_srrgan_psnr += recon_psnr
                    average_srrgan_ssim += recon_ssim
                    average_srrgan_ergas += recon_ergas
                    average_srrgan_lpips += recon_lpips
                    # loss_mse += mse
                    print("Saving %d test result images..." % img_num)

                    del bc_img, gt_img, recon_img, recon_mse, recon_psnr, recon_ssim, recon_ergas, recon_lpips, bc_mse, bc_ssim, bc_psnr, bc_ergas, bc_lpips
                del imgs_hr, imgs_bc, imgs_lr, recon_imgs

            avg_bicubic_mse = avg_bicubic_mse / img_num
            avg_bicubic_psnr = avg_bicubic_psnr / img_num
            avg_bicubic_ssim = avg_bicubic_ssim / img_num
            avg_bicubic_ergas = avg_bicubic_ergas / img_num
            avg_bicubic_lpips = avg_bicubic_lpips / img_num
            avg_srcnn_mse = avg_srcnn_mse / img_num
            avg_srcnn_psnr = avg_srcnn_psnr / img_num
            avg_srcnn_ssim = avg_srcnn_ssim / img_num
            avg_srcnn_ergas = avg_srcnn_ergas / img_num
            avg_srcnn_lpips = avg_srcnn_lpips / img_num
            time_elapsed = time.time() - start_time
            # Save to log
            print_rlt = OrderedDict()
            print_rlt['model'] = datasetName
            print_rlt['epoch'] = epoch
            print_rlt['iters'] = epoch
            print_rlt['time'] = time_elapsed
            print_rlt['bicubic_mse'] = avg_bicubic_mse
            print_rlt['bicubic_psnr'] = avg_bicubic_psnr
            print_rlt['bicubic_ssim'] = avg_bicubic_ssim
            print_rlt['bicubic_ergas'] = avg_bicubic_ergas
            print_rlt['bicubic_lpips'] = avg_bicubic_lpips
            print_rlt['transenet_mse'] = avg_srcnn_mse
            print_rlt['transenet_psnr'] = avg_srcnn_psnr
            print_rlt['transenet_ssim'] = avg_srcnn_ssim
            print_rlt['transenet_ergas'] = avg_srcnn_ergas
            print_rlt['transenet_lpips'] = avg_srcnn_lpips
            self.logger.print_format_results('val', print_rlt)

            del test_data_loader
            del test_set

        average_bicubic_mse = average_bicubic_mse / num
        average_bicubic_psnr = average_bicubic_psnr / num
        average_bicubic_ssim = average_bicubic_ssim / num
        average_bicubic_ergas = average_bicubic_ergas / num
        average_bicubic_lpips = average_bicubic_lpips / num
        average_srrgan_mse = average_srrgan_mse / num
        average_srrgan_psnr = average_srrgan_psnr / num
        average_srrgan_ssim = average_srrgan_ssim / num
        average_srrgan_ergas = average_srrgan_ergas / num
        average_srrgan_lpips = average_srrgan_lpips / num
        time_elapsed = time.time() - start_time
        # Save to log
        print_rlt = OrderedDict()
        print_rlt['model'] = "Total"  # opt['model']
        print_rlt['epoch'] = epoch
        print_rlt['iters'] = epoch
        print_rlt['time'] = time_elapsed
        print_rlt['bicubic_mse'] = average_bicubic_mse
        print_rlt['bicubic_psnr'] = average_bicubic_psnr
        print_rlt['bicubic_ssim'] = average_bicubic_ssim
        print_rlt['bicubic_ergas'] = average_bicubic_ergas
        print_rlt['bicubic_lpips'] = average_bicubic_lpips
        print_rlt['transenet_mse'] = average_srrgan_mse
        print_rlt['transenet_psnr'] = average_srrgan_psnr
        print_rlt['transenet_ssim'] = average_srrgan_ssim
        print_rlt['transenet_ergas'] = average_srrgan_ergas
        print_rlt['transenet_lpips'] = average_srrgan_lpips
        self.logger.print_format_results('val', print_rlt)
        print('-----------------------------------')
        torch.cuda.empty_cache()

    def mfeNew_total_ops(self, modelpath=None):
        # load model
        self.generator = TransENet(n_feats=64, scale=self.scale_factor, n_basic_modules=10, n_colors=3, hr_patch_size=self.crop_size, back_projection_iters=10, en_depth=8, de_depth=1, conv=default_conv)

        if modelpath is not None:
            self.generator.load_state_dict(torch.load(modelpath), strict=False)
        srutils.print_network_to_file(self.generator, save_dir=self.save_dir, tag='TransENet')



    def dm_test_single(self, img_fn, modelpath=None):
        # networks
        cuda = True if torch.cuda.is_available() else False
        self.generator = TransENet(n_feats=64, scale=4, n_basic_modules=10, n_colors=3, hr_patch_size=512, back_projection_iters=10, en_depth=8, de_depth=1, conv=default_conv)

        # load model
        if modelpath is not None:
            self.generator.load_state_dict(torch.load(modelpath), strict=False)

        if cuda:
            self.generator = self.generator.cuda()

        self.generator.eval()

        # load data
        imgs = os.listdir(img_fn)
        for img_ in imgs:
            img_dir = os.path.join(img_fn, img_)
            img = Image.open(img_dir)

            toTensor = transforms.Compose([
                transforms.CenterCrop(128),
                transforms.ToTensor()
            ])
            input = toTensor(img)

            Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
            # input_lr = Tensor(self.test_batch_size, self.num_channels, 32, 32)
            input_lr = Tensor(1, 3, 128, 128)
            input_img = Variable(input_lr.copy_(input))

            # prediction
            recon_imgs = self.generator(input_img)
            img_name, img_type = img_.split(".")
            srutils.save_img1(recon_imgs[0].cpu().data, self.save_dir, self.save_dir + '/SR_transenet_x4_%s' % img_)

