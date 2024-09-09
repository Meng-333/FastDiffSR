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

class GANLoss(nn.Module):
    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type.lower()
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'wgan-gp':

            def wgan_loss(input, target):
                # target is boolean
                return -1 * input.mean() if target else input.mean()

            self.loss = wgan_loss
        else:
            raise NotImplementedError('GAN type [{:s}] is not found'.format(self.gan_type))

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()

        vgg19_model = vgg19(pretrained=True)

        # Extracts features at the 11th layer
        self.feature_extractor = nn.Sequential(*list(vgg19_model.features.children())[:12])

    def forward(self, img):
        out = self.feature_extractor(img)
        return out


def make_model(args, parent=False):
    return HSENET(args)


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

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
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?  # &：按位与运算； 参与运算的两个值,如果两个相应位都为1,则该位的结果为1,否则为0
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feat, 4 * n_feat, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn: m.append(nn.BatchNorm2d(n_feat))
                if act: m.append(act())
        elif scale % 3 == 0:
            for _ in range(int(math.log(scale, 3))):
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
        #view()是做维度调整,就是按照参数表里的数字重新调整tensor的各维度上的大小, 最后那个-1的意思是把剩下的维度全都合并归到最后的一维;
        #g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        #比如示例中的g(x)算完之后是[128, 16, 7, 7] : (b, c, h, w), 示例中做的是把它变成[128, 16, 49], 就是把h和w两个维度合成一个
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)

        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)    #两个矩阵乘积运算

        # use dot production
        # N = f.size(-1)
        # f_div_C = f / N

        # use embedding gaussian
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()    #调用contiguous()之后，PyTorch会开辟一块新的内存空间存放变换之后的数据，并会真正改变Tensor的内容，按照变换之后的顺序存放数据
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])  #展开维度
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


# hybrid-scale self-similarity exploitation module
class HSEM(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, bias=True,
                 bn=False, act=nn.ReLU(True)):
        super(HSEM, self).__init__()

        base_scale = []
        base_scale.append(SSEM(conv, n_feats, kernel_size, bias=bias, bn=bn, act=act))

        down_scale = []
        down_scale.append(SSEM(conv, n_feats, kernel_size, bias=bias, bn=bn, act=act))

        tail = []
        tail.append(BasicBlock(conv, n_feats, n_feats, kernel_size, bias=bias, bn=bn, act=act))

        self.NonLocal_base = AdjustedNonLocalBlock(n_feats, n_feats // 2)

        self.base_scale = nn.Sequential(*base_scale)
        self.down_scale = nn.Sequential(*down_scale)
        self.tail = nn.Sequential(*tail)

    def forward(self, x):

        add_out = x

        # base scale
        x_base = self.base_scale(x)

        # 1/2 scale
        x_down = F.interpolate(x, scale_factor=0.5, mode='bilinear')
        x_down = self.down_scale(x_down)

        # fusion x_down and x_down2
        x_down = F.interpolate(x_down, size=(x_base.shape[2], x_base.shape[3]),
                               mode='bilinear')
        ms = self.NonLocal_base(x_base, x_down)
        ms = self.tail(ms)

        add_out = add_out + ms

        return add_out


# single-scale self-similarity exploitation module
class SSEM(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, bias=True,
                 bn=False, act=nn.ReLU(True)):
        super(SSEM, self).__init__()

        head = []
        head.append(BasicBlock(conv, n_feats, n_feats, kernel_size, bias=bias, bn=bn))

        MB = [] # main branch
        MB.append(BasicBlock(conv, n_feats, n_feats, kernel_size, bias=bias, bn=bn))
        MB.append(BasicBlock(conv, n_feats, n_feats, kernel_size, bias=bias, bn=bn))

        AB = []  # attention branch
        AB.append(NonLocalBlock2D(n_feats, n_feats//2))
        AB.append(nn.Conv2d(n_feats, n_feats, 1, padding=0, bias=True))

        sigmoid = []
        sigmoid.append(nn.Sigmoid())

        tail = []
        tail.append(BasicBlock(conv, n_feats, n_feats, kernel_size, bias=bias, bn=bn))

        self.head = nn.Sequential(*head)
        self.MB = nn.Sequential(*MB)
        self.AB = nn.Sequential(*AB)
        self.sigmoid = nn.Sequential(*sigmoid)
        self.tail = nn.Sequential(*tail)

    def forward(self, x):

        add_out = x
        x_head = self.head(x)
        x_MB = self.MB(x_head)
        x_AB = self.AB(x_head)
        x_AB = self.sigmoid(x_AB)
        x_MB_AB = x_MB * x_AB
        x_tail = self.tail(x_MB_AB)

        add_out = add_out + x_tail
        return add_out


# multi-scale self-similarity block
class BasicModule(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, bias=True,
                 bn=False, act=nn.ReLU(True)):
        super(BasicModule, self).__init__()

        head = [
            BasicBlock(conv, n_feats, n_feats, kernel_size, bias=bias, bn=bn, act=act) \
            for _ in range(2)
        ]

        body = []
        body.append(HSEM(conv, n_feats, kernel_size, bias=bias, bn=bn, act=act))

        tail = [
            BasicBlock(conv, n_feats, n_feats, kernel_size, bias=bias, bn=bn, act=act) \
            for _ in range(2)
        ]

        self.head = nn.Sequential(*head)
        self.body = nn.Sequential(*body)
        self.tail = nn.Sequential(*tail)

    def forward(self, x):

        add_out = x

        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)

        add_out = add_out + x

        return add_out


class HSENET(nn.Module):
    def __init__(self, n_feats=64, scale=3, n_basic_modules=10, n_colors=3, conv=default_conv):   #默认n_basic_modules=10
        super(HSENET, self).__init__()

        n_feats = n_feats
        kernel_size = 3
        scale = scale
        act = nn.ReLU(True)
        rgb_range = 1.0

        self.n_BMs = n_basic_modules

        rgb_mean = (0.4916, 0.4991, 0.4565)  # UCMerced data
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = MeanShift(rgb_range, rgb_mean, rgb_std)

        # define head body
        m_head = [conv(n_colors, n_feats, kernel_size)]

        # define main body
        self.body_modulist = nn.ModuleList([
            BasicModule(conv, n_feats, kernel_size, act=act) \
            for _ in range(self.n_BMs)    #for循环来创建重复的层
        ])

        # define tail body
        m_tail = [
            Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, n_colors, kernel_size)
        ]

        self.add_mean = MeanShift(rgb_range, rgb_mean, rgb_std, 1)
        self.head = nn.Sequential(*m_head)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        # main body
        add_out = x

        for i in range(self.n_BMs):
            x = self.body_modulist[i](x)
        add_out = add_out + x

        x = self.tail(add_out)
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
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

class HSENet(object):
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

        self.lr_transforms = [
            transforms.CenterCrop(self.crop_size),
            transforms.Resize(self.crop_size // self.scale_factor),
            transforms.ToTensor()
        ]
        self.hr_transforms = [
            transforms.CenterCrop(self.crop_size),
            transforms.ToTensor()
        ]
        # self.lr_transforms = [transforms.Resize((self.hr_height // self.scale_factor, self.hr_height // self.scale_factor), Image.BICUBIC),
        #                  transforms.ToTensor(),
        #                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        #
        # self.hr_transforms = [transforms.Resize((self.hr_height, self.hr_height), Image.BICUBIC),
        #                  transforms.ToTensor(),
        #                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

        # self.use_gan = args.use_gan
        self.use_gan = False
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

        cuda = True if torch.cuda.is_available() else False
        # cuda = False
        from utils import PerceptualSimilarity
        self.PerceptualModel = PerceptualSimilarity.PerceptualLoss(model='net-lin', net='alex', use_gpu=cuda)

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
                              shuffle=True, drop_last=True)
        elif dataset == 'test':
            print('Loading test dct_datasets...')
            test_set = get_RGB_testDataset(self.data_dir, self.test_dataset, self.crop_size, self.scale_factor, is_gray=is_gray)
            return DataLoader(dataset=test_set, num_workers=self.num_threads,
                              batch_size=self.test_batch_size,
                              shuffle=False, drop_last=True)

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
        self.generator = HSENET(n_feats=64, scale=self.scale_factor, n_basic_modules=10, n_colors=3, conv=default_conv)  #默认n_basic_modules=10
        self.feature_extractor = FeatureExtractor()
        # self.feature_extractor = VGGFeatureExtractor(feature_layer=34, use_bn=False, use_input_norm=True, device=self.device)
        #srutils.print_network_to_file(self.generator, save_dir=self.save_dir, tag='Generator')
        #srutils.print_network_to_file(self.feature_extractor, save_dir=self.save_dir, tag='FeatureExtractor')

        if self.use_gan:
            self.discriminator = Discriminator(norm_type='batch', use_spectralnorm=False, attention=False)
            #srutils.print_network_to_file(self.discriminator, save_dir=self.save_dir, tag='Discriminator')
            self.criterion_raGAN = GANLoss(gan_type='wgan-gp', real_label_val=1.0, fake_label_val=0.0).to(self.device)
        # Losses
        # self.criterion_GAN = torch.nn.MSELoss().to(self.device)
        if self.loss_Lp_norm=="L1":
            self.criterion_content = torch.nn.L1Loss().to(self.device)
        else:
            self.criterion_content = torch.nn.MSELoss().to(self.device)

        if len(device_ids) > 1:
            self.generator = nn.DataParallel(self.generator).to(self.device)
            self.feature_extractor = nn.DataParallel(self.feature_extractor).to(self.device)
            if self.use_gan:
                self.discriminator = nn.DataParallel(self.discriminator).to(self.device)

        if cuda:
            self.generator = self.generator.cuda(device_ids[0])
            self.feature_extractor = self.feature_extractor.cuda(device_ids[0])
            # self.criterion_GAN = self.criterion_GAN.cuda(device_ids[0])
            self.criterion_content = self.criterion_content.cuda(device_ids[0])
            if self.use_gan:
                self.discriminator = self.discriminator.cuda(device_ids[0])
                self.criterion_raGAN = self.criterion_raGAN.cuda(device_ids[0])

        model_dir = os.path.join(self.save_dir, 'model')
        if self.epoch != 0:
            # Load pretrained models
            # model_dict_G = torch.load(model_dir+'/generator_param_epoch_%d.pkl' % self.epoch)
            # trans_param_G = self.generator.state_dict()
            # for item, value in model_dict_G.items():
            #     name = '.'.join(item.split('.')[1:])
            #     trans_param_G[name] = value
            #
            # model_dict_D = torch.load(model_dir+'/discriminator_param_epoch_%d.pkl' % self.epoch)
            # trans_param_D = self.discriminator.state_dict()
            # for item, value in model_dict_D.items():
            #     name = '.'.join(item.split('.')[1:])
            #     trans_param_D[name] = value
            #
            # self.generator.load_state_dict(trans_param_G)
            # self.discriminator.load_state_dict(trans_param_D)
            # self.generator.load_state_dict(torch.load(model_dir+'/generator_param_epoch_%d.pkl' % self.epoch))
            # self.discriminator.load_state_dict(torch.load(model_dir+'/discriminator_param_epoch_%d.pkl' % self.epoch))
            G_model_path = model_dir+'/generator_param_epoch_%d.pkl' % self.epoch
            # D_model_path = model_dir+'/discriminator_param_epoch_%d.pkl' % self.epoch
            self.load_epoch_network(load_path=G_model_path, network=self.generator)
            # self.load_epoch_network(load_path=D_model_path, network=self.discriminator)
        else:
            # load_path_G = "pretrained_models/SRResNet_param_epoch_64.pkl"
            # print('loading model for G [{:s}] ...'.format(load_path_G))
            # # self.generator.load_state_dict(torch.load(load_path_G), strict=True)
            # self.load_epoch_network(load_path=load_path_G, network=self.generator)
            # Initialize weights
            self.generator.apply(srutils.weights_init_normal)
            if self.use_gan:
                self.discriminator.apply(srutils.weights_init_normal)
            # G_model_path = '/home/zju/lyd/SuperResolution/Result/generator_param_epoch_49.pkl'
            # D_model_path = '/home/zju/lyd/SuperResolution-1/Result/discriminator_param_epoch_85.pkl'
            # self.load_epoch_network(load_path=G_model_path, network=self.generator, strict=False)
            # self.load_epoch_network(load_path=D_model_path, network=self.discriminator, strict=False)

        # Optimizers
        optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        if self.use_gan:
            optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        # optimizer_G = torch.optim.RMSprop(self.generator.parameters(), lr=self.lr)
        # optimizer_D = torch.optim.RMSprop(self.discriminator.parameters(), lr=self.lr)

        # Multi-GPU Training
        # if torch.cuda.device_count() > 1:
        #     self.generator = nn.DataParallel(self.generator, device_ids=device_ids)
        #     #self.discriminator = nn.DataParallel(self.discriminator, device_ids=device_ids)
        #     self.feature_extractor = nn.DataParallel(self.feature_extractor, device_ids=device_ids)
        #     optimizer_G = nn.DataParallel(optimizer_G, device_ids=device_ids)
        #     #optimizer_D = nn.DataParallel(optimizer_D, device_ids=device_ids)
        #     if self.use_gan:
        #         self.discriminator = nn.DataParallel(self.discriminator, device_ids=device_ids)
        #         optimizer_D = nn.DataParallel(optimizer_D, device_ids=device_ids)

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

        # set the logger
        # G_log_dir = os.path.join(self.save_dir, 'G_logs')
        # if not os.path.exists(G_log_dir):
        #     os.mkdir(G_log_dir)
        # G_logger = Logger(G_log_dir)
        #
        # D_log_dir = os.path.join(self.save_dir, 'D_logs')
        # if not os.path.exists(D_log_dir):
        #     os.mkdir(D_log_dir)
        # D_logger = Logger(D_log_dir)

        sys.stdout = PrintLogger(os.path.join(self.root_dir, self.save_dir))
        # ----------
        #  Training
        # ----------
        print('Training is started.')
        avg_loss_G = []
        avg_loss_D = []
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
        while epoch < self.num_epochs and self.lr > 0.00001:
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
            epoch_loss_D = 0
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
                # gen_features = self.feature_extractor(gen_hr)
                # real_features = Variable(self.feature_extractor(imgs_hr).data, requires_grad=False)
                # loss_content = self.criterion_content(gen_features, real_features)
                loss_G = mse_loss_G #+ self.weight_content * loss_content
                # Adversarial loss
                if self.use_gan:
                    if self.relative:
                        pred_g_fake = self.discriminator(gen_hr)
                        pred_d_real = self.discriminator(imgs_hr).detach()
                        loss_gan = (self.criterion_raGAN(pred_d_real - torch.mean(pred_g_fake), False) +
                                    self.criterion_raGAN(pred_g_fake - torch.mean(pred_d_real), True)) / 2
                    else:
                        # loss_gan = -torch.mean(self.discriminator(gen_hr))
                        gen_validity = self.discriminator(gen_hr)
                        loss_gan = self.criterion_raGAN(gen_validity, True)
                    loss_G += self.weight_gan * loss_gan

                # Total loss
                # loss_G = loss_content + 1e-3 * loss_GAN
                # loss_G = mse_loss_G + 2e-6 * loss_content + 1e-3 * loss_gan
                # loss_G = mse_loss_G + 1e-3 * loss_content + 1e-2 * loss_gan

                loss_G.backward()
                optimizer_G.step()
                # optimizer_G.module.step()

                # ---------------------
                #  Train Discriminator
                # ---------------------
                if self.use_gan:
                    optimizer_D.zero_grad()

                    # Loss of real and fake images
                    if self.relative:
                        pred_d_real = self.discriminator(imgs_hr)
                        pred_d_fake = self.discriminator(gen_hr.detach())  # detach to avoid BP to G
                        l_d_real = self.criterion_raGAN(pred_d_real - torch.mean(pred_d_fake), True)
                        l_d_fake = self.criterion_raGAN(pred_d_fake - torch.mean(pred_d_real), False)
                        loss_D = (l_d_real + l_d_fake) / 2
                    else:
                        # loss_D = -torch.mean(self.discriminator(imgs_hr)) + torch.mean(self.discriminator(gen_hr.detach()))
                        loss_real = self.criterion_raGAN(self.discriminator(imgs_hr), True)
                        loss_fake = self.criterion_raGAN(self.discriminator(gen_hr.detach()), False)
                        loss_D = loss_real + loss_fake
                    # Gradient penalty
                    # gradient_penalty = self.compute_gradient_penalty(self.discriminator, imgs_hr.data, gen_hr.detach().data)
                    if self.gp:
                        gradient_penalty = self.gradient_penalty(self.discriminator, imgs_hr.data, gen_hr.detach().data,
                                                                 grad_penalty_Lp_norm=self.grad_penalty_Lp_norm,
                                                                 penalty_type=self.penalty_type)
                        loss_D += self.lambda_gp * gradient_penalty

                    loss_D.backward()
                    optimizer_D.step()
                    # optimizer_D.module.step()

                    # Clip weights of discriminator
                    for p in self.discriminator.parameters():
                        p.data.clamp_(-self.clip_value, self.clip_value)
                # --------------
                #  Log Progress
                # --------------
                print("[Epoch %d/%d] [Batch %d/%d] [G loss: %f]" %
                      (epoch, self.num_epochs, i, len(dataloader), loss_G.item()))
                epoch_loss_G += loss_G.data.item()
                self.logger.scalar_summary('loss_G', loss_G.item(), step + 1)

                epoch_loss_G += loss_G.data.item()
                self.logger.scalar_summary('loss_G', loss_G.item(), step + 1)
                if self.use_gan:
                    print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f]" %
                          (epoch, self.num_epochs, i, len(dataloader), loss_D.item()))
                    epoch_loss_D += loss_D.data.item()
                    # tensorboard logging
                    self.logger.scalar_summary('loss_D', loss_D.item(), step + 1)
                    epoch_loss_D += loss_D.data.item()
                    # tensorboard logging
                    self.logger.scalar_summary('loss_D', loss_D.item(), step + 1)

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
                    # print_rlt['G_lr'] = optimizer_G.module.param_groups[0]['lr']
                    # print_rlt['D_lr'] = optimizer_D.module.param_groups[0]['lr'] if self.use_gan else 0
                    print_rlt['G_lr'] = optimizer_G.param_groups[0]['lr']
                    print_rlt['D_lr'] = optimizer_D.param_groups[0]['lr'] if self.use_gan else 0
                    print_rlt['time'] = time_elapsed
                    print_rlt['G_loss'] = loss_G.item()
                    print_rlt['D_loss'] = loss_D.item() if self.use_gan else 0
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
                del imgs_hr, imgs_bc, imgs_lr, gen_hr#, gen_features, real_features
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
                # for param_group in optimizer_G.module.param_groups:
                #     param_group["lr"] /= 2.0
                # print("Learning rate decay: lr={}".format(optimizer_G.module.param_groups[0]["lr"]))
                # if self.use_gan:
                #     for param_group in optimizer_D.module.param_groups:
                #         param_group["lr"] /= 2.0
                #     print("Learning rate decay: lr={}".format(optimizer_D.module.param_groups[0]["lr"]))
                for param_group in optimizer_G.param_groups:
                    param_group["lr"] /= 2.0
                print("Learning rate decay: lr={}".format(optimizer_G.param_groups[0]["lr"]))
                self.lr /= 2.0
                if self.use_gan:
                    for param_group in optimizer_D.param_groups:
                        param_group["lr"] /= 2.0
                    print("Learning rate decay: lr={}".format(optimizer_D.param_groups[0]["lr"]))
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
        # srutils.plot_loss([avg_loss_G, avg_loss_D], self.num_epochs, save_dir=self.save_dir)
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
            self.generator = HSENET(n_feats=64, scale=self.scale_factor, n_basic_modules=10, n_colors=3, conv=default_conv)   #默认n_basic_modules=10
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

    def validateByClass(self, epoch, modelpath=None, mode='test', save_img=False):
        # networks
        cuda = True if torch.cuda.is_available() else False
        cuda = False
        device_ids = [0]
        self.device = torch.device('cuda' if cuda else 'cpu')
        # load model
        if mode == 'test':
            # networks
            # self.generator = GeneratorResNet(n_residual_blocks=28).to(self.device)
            self.generator = HSENET(n_feats=64, scale=self.scale_factor, n_basic_modules=10, n_colors=3, conv=default_conv)   #默认n_basic_modules=10
            if modelpath is not None:
                self.generator.load_state_dict(torch.load(modelpath), strict=True)
            else:
                self.load_epoch_model(epoch)

        if cuda and self.gpu_mode:
            self.generator.cuda()

        if len(device_ids) > 1:
            self.generator = nn.DataParallel(self.generator).to(self.device)

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
        for datasetDir in self.test_dataset:

            test_set = get_RGB_testDataset(self.data_dir, [datasetDir], self.crop_size, self.scale_factor,
                                                 is_gray=False)
            test_data_loader = DataLoader(dataset=test_set, num_workers=self.num_threads,
                              batch_size=self.test_batch_size,
                              shuffle=False, drop_last=True)
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
            for iter, (input, target, bc_input, imgpath) in enumerate(test_data_loader):
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
                        srutils.save_img1(recon_imgs[i].cpu(), os.path.join(self.save_dir, 'validate'), img_dir, normalize=cuda)

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
        self.generator = HSENET()

        if cuda:
            self.generator = self.generator.cuda()

        # load model
        self.load_model()

        # Inputs & targets memory allocation
        Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
        input_lr = Tensor(self.test_batch_size, self.num_channels, self.hr_height // self.scale_factor, self.hr_width // self.scale_factor)
        input_hr = Tensor(self.test_batch_size, self.num_channels, self.hr_height, self.hr_width)
        input_bc = Tensor(self.test_batch_size, self.num_channels, self.hr_height, self.hr_width)

        tensor_to_img = ToPILImage()

        # load test dataset
        test_data_loader = self.load_dataset(dataset='test')

        # Test
        print('Test dataset is started.')
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

                print("Saving the %d test dataset images" % img_num)

    def validate_single(self, epoch, imgpath):
        # networks
        #cuda = True if torch.cuda.is_available() else False
        # Initialize generator and discriminator
        self.generator = HSENET()

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
        self.generator = HSENET()

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
    def load_epoch_network(self, load_path, network, strict=False):
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
            # torch.save(self.discriminator.state_dict(), model_dir + '/discriminator_param_epoch_%d.pkl' % epoch)
        else:
            torch.save(self.generator.state_dict(), model_dir + '/generator_param.pkl')
            # torch.save(self.discriminator.state_dict(), model_dir + '/discriminator_param.pkl')

        print('Trained model is saved.')

    def load_model(self):
        model_dir = os.path.join(self.save_dir, 'model')

        model_name = model_dir + '/generator_param.pkl'
        if os.path.exists(model_name):
            self.generator.load_state_dict(torch.load(model_name), strict=False)
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
            self.generator.load_state_dict(torch.load(model_name), strict=False)
            print('Trained model is loaded.')
            return True
        else:
            print('No model exists to load.')
            return False



    def mfeNew_validate(self, epoch=100, modelpath=None):
        # networks
        cuda = True if torch.cuda.is_available() else False

        # load model
        self.generator = HSENET(n_feats=64, scale=self.scale_factor, n_basic_modules=10, n_colors=3, conv=default_conv)   #默认n_basic_modules=10
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
        print_rlt['hsenet_mse'] = avg_srcnn_mse
        print_rlt['hsenet_psnr'] = avg_srcnn_psnr
        print_rlt['hsenet_ssim'] = avg_srcnn_ssim
        print_rlt['hsenet_ergas'] = avg_srcnn_ergas
        print_rlt['hsenet_lpips'] = avg_srcnn_lpips
        self.logger.print_format_results('val', print_rlt)
        print('-----------------------------------')
        del test_data_loader
        torch.cuda.empty_cache()
        return avg_srcnn_psnr, avg_srcnn_ssim, avg_srcnn_ergas, avg_srcnn_lpips

    def mfeNew_validateByClass(self, epoch, save_img=False, modelpath=None):
        # networks
        cuda = True if torch.cuda.is_available() else False
        # load model
        self.generator = HSENET(n_feats=64, scale=self.scale_factor, n_basic_modules=10, n_colors=3, conv=default_conv)   #默认n_basic_modules=10

        if modelpath is not None:
            self.generator.load_state_dict(torch.load(modelpath), strict=False)
        #srutils.print_network_to_file(self.generator, save_dir=self.save_dir, tag='HSENET_validateByClass')

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
            print_rlt['hsenet_mse'] = avg_srcnn_mse
            print_rlt['hsenet_psnr'] = avg_srcnn_psnr
            print_rlt['hsenet_ssim'] = avg_srcnn_ssim
            print_rlt['hsenet_ergas'] = avg_srcnn_ergas
            print_rlt['hsenet_lpips'] = avg_srcnn_lpips
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
        print_rlt['hsenet_mse'] = average_srrgan_mse
        print_rlt['hsenet_psnr'] = average_srrgan_psnr
        print_rlt['hsenet_ssim'] = average_srrgan_ssim
        print_rlt['hsenet_ergas'] = average_srrgan_ergas
        print_rlt['hsenet_lpips'] = average_srrgan_lpips
        self.logger.print_format_results('val', print_rlt)
        print('-----------------------------------')
        torch.cuda.empty_cache()


    def mfeNew_total_ops(self, modelpath=None):
        # load model
        self.generator = HSENET(n_feats=64, scale=self.scale_factor, n_basic_modules=10, n_colors=3, conv=default_conv)   #默认n_basic_modules=10

        if modelpath is not None:
            self.generator.load_state_dict(torch.load(modelpath), strict=False)
        srutils.print_network_to_file(self.generator, save_dir=self.save_dir, tag='HSENET')


    def dm_test_single(self, img_fn, modelpath=None):
        # networks
        cuda = True if torch.cuda.is_available() else False
        self.generator = HSENET(n_feats=64, scale=4, n_basic_modules=10, n_colors=3, conv=default_conv)   #默认n_basic_modules=10

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
                transforms.CenterCrop(125),
                transforms.ToTensor()
            ])
            input = toTensor(img)

            Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
            # input_lr = Tensor(self.test_batch_size, self.num_channels, 32, 32)
            input_lr = Tensor(1, 3, 125, 125)
            input_img = Variable(input_lr.copy_(input))

            # prediction
            recon_imgs = self.generator(input_img)
            img_name, img_type = img_.split(".")
            srutils.save_img1(recon_imgs[0].cpu().data, self.save_dir, self.save_dir + '/SR_hsenet_x4_%s' % img_)


