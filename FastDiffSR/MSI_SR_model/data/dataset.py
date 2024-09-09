import torch.utils.data as data
from torchvision.transforms import *
from torchvision.transforms import functional as functional
from os import listdir
from os.path import join
from PIL import Image
import random
import glob
import cv2
from torch.utils.data import DataLoader
# from prefetch_generator import BackgroundGenerator
import torch
import numpy as np

class AddSaltPepperNoise(object):

    def __init__(self, density=0):
        self.density = density

    def __call__(self, img):

        img = np.array(img)                                                             # 图片转numpy
        h, w, c = img.shape
        Nd = self.density
        Sd = 1 - Nd
        mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[Nd/2.0, Nd/2.0, Sd])      # 生成一个通道的mask
        mask = np.repeat(mask, c, axis=2)                                               # 在通道的维度复制，生成彩色的mask
        img[mask == 0] = 0                                                              # 椒
        img[mask == 1] = 255                                                            # 盐
        img= Image.fromarray(img.astype('uint8')).convert('RGB')                        # numpy转图片
        return img

class AddGaussianNoise(object):

    def __init__(self, mean=0.0, variance=1.0, amplitude=1.0):

        self.mean = mean
        self.variance = variance
        self.amplitude = amplitude

    def __call__(self, img):
        img = np.array(img)
        h, w, c = img.shape
        N = self.amplitude * np.random.normal(loc=self.mean, scale=self.variance, size=(h, w, 1))
        N = np.repeat(N, c, axis=2)
        img = N + img
        img[img > 255] = 255                       # 避免有值超过255而反转
        img = Image.fromarray(img.astype('uint8')).convert('RGB')
        return img

# class DataLoaderX(DataLoader):
#     def __iter__(self):
#         return BackgroundGenerator(super().__iter__())

class DataPrefetcher():
    def __init__(self, loader, opt):
        self.loader = iter(loader)
        self.opt = opt
        self.stream = torch.cuda.Stream()
        # with Amp, it isn't necessary to manually convert data to half
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.preload()

    def preload(self):
        try:
            self.batch = next(self.loader)
        except StopIteration:
            self.batch = None
            return
        with torch.cuda.stream(self.stream):
            for k in self.batch:
                if k != 'meta':
                    self.batch[k] = self.batch[k].to(device=self.opt.device, non_blocking=True)
            # with Amp, it isn't necessary to manually convert data to half
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            #     self.next_input = self.next_input.float()

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        self.preload()
        return batch

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".bmp", ".tif"])


def load_img(filepath):

    img = Image.open(filepath).convert('RGB')
    # img = Image.open(filepath)
    #img = np.array(img)
    return img


def calculate_valid_crop_size(crop_size, scale_factor):
    return crop_size - (crop_size % scale_factor)



class TestDatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, is_gray=False, scale_factor=4):
        super(TestDatasetFromFolder, self).__init__()
        self.image_filenames = []
        for dir in image_dir:
            self.image_filenames.extend(join(dir, x) for x in sorted(listdir(dir)) if is_image_file(x))
        self.is_gray = is_gray
        self.scale_factor = scale_factor

    def __getitem__(self, index):
        # load image
        img = load_img(self.image_filenames[index])

        # original HR image size
        w = img.size[0]
        h = img.size[1]

        # determine valid HR image size with scale factor
        hr_img_w = calculate_valid_crop_size(w, self.scale_factor)
        hr_img_h = calculate_valid_crop_size(h, self.scale_factor)

        # determine lr_img LR image size
        lr_img_w = hr_img_w // self.scale_factor
        lr_img_h = hr_img_h // self.scale_factor

        # only Y-channel is super-resolved
        if self.is_gray:
            img = img.convert('YCbCr')
            # img, _, _ = lr_img.split()

        # hr_img HR image
        hr_transform = Compose([Resize((hr_img_w, hr_img_h), interpolation=Image.BICUBIC), ToTensor()])
        hr_img = hr_transform(img)

        # lr_img LR image
        lr_transform = Compose([Resize((lr_img_w, lr_img_h), interpolation=Image.BICUBIC), ToTensor()])
        lr_img = lr_transform(img)

        # Bicubic interpolated image
        bc_transform = Compose([ToPILImage(), Resize((hr_img_w, hr_img_h), interpolation=Image.BICUBIC), ToTensor()])
        bc_img = bc_transform(lr_img)

        return lr_img, hr_img, bc_img, self.image_filenames[index]

    def __len__(self):
        return len(self.image_filenames)


class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dirs, crop_size=256, noise=None, input_transform=None, input_resize_transform=None, target_transform=None, is_gray=False, max_sample_num=100000):
        super(DatasetFromFolder, self).__init__()
        #self.image_filenames = [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]
        image_files = []
        for image_dir in image_dirs:
            image_files.extend(join(image_dir, x) for x in sorted(listdir(image_dir)) if is_image_file(x))

        sampleNum = len(image_files)
        if sampleNum > max_sample_num:
            indicies = random.sample(range(sampleNum), max_sample_num)
            self.image_filenames = [image_files[i] for i in indicies]
        else:
            self.image_filenames = image_files

        self.input_transform = input_transform
        self.input_resize_transform = input_resize_transform
        self.target_transform = target_transform
        self.is_gray = is_gray
        self.crop_size = crop_size
        self.noise = noise

    def __getitem__(self, index):
        input = load_img(self.image_filenames[index])
        # only Y-channel is super-resolved
        if self.is_gray:
            input = input.convert('YCbCr')

        # transform = CenterCrop(self.crop_size)
        # img = transform(input)

        target = input.copy()
        input_resize = input.copy()
        if self.input_transform:
            input = self.input_transform(input)
            if self.noise is not None:  # and random.random() < 0.5
                noise_type = self.noise[0]
                noise_value = self.noise[1]
                # img_array = np.asarray(input)
                # if noise_type == 'Gaussain':
                #     noises = np.random.normal(loc=0, scale=noise_value, size=img_array.shape)
                # elif noise_type == 'Poisson':
                #     noises = np.random.poisson(input.numpy() * noise_value) / noise_value
                #     noises = noises - noises.mean(axis=0).mean(axis=0)
                # # gauss = np.random.normal(mean, std, img.shape).astype(np.float32)
                # x_noise = np.clip((1 + noises) * img_array.astype(np.float32), 0, 255).astype(np.uint8)
                # # x_noise = img_array + noises
                # # x_noise = x_noise.clip(0, 255).astype(np.uint8)
                # input = ToTensor()(x_noise)
                if noise_type == 'Gaussain':
                    input_noise = AddGaussianNoise(amplitude=noise_value)(input)
                elif noise_type == 'Poisson':
                    input_noise = AddSaltPepperNoise(density=noise_value)(input)
                input = ToTensor()(input_noise)
                input_resize = input_noise
            else:
                input_resize = input
                input = ToTensor()(input)
        if self.input_resize_transform:
            # input_resize = self.input_resize_transform(input_resize)
            input_resize = self.input_resize_transform(input_resize)
        if self.target_transform:
            target = self.target_transform(target)

        return input, target, input_resize, self.image_filenames[index]

    def __len__(self):
        return len(self.image_filenames)

class Dataset(data.Dataset):
    def __init__(self, image_dirs, crop_size=256, scale_factor=4, is_gray=False, random_crop=False, random_scale=False, rotate=False,
                 fliplr=False, fliptb=False, noise=None, max_sample_num=100000):
        super(Dataset, self).__init__()
        #self.image_filenames = [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]
        image_files = []
        for image_dir in image_dirs:
            image_files.extend(join(image_dir, x) for x in sorted(listdir(image_dir)) if is_image_file(x))

        sampleNum = len(image_files)
        if sampleNum > max_sample_num:
            indicies = random.sample(range(sampleNum), max_sample_num)
            self.image_filenames = [image_files[i] for i in indicies]
        else:
            self.image_filenames = image_files

        self.is_gray = is_gray
        self.random_crop = random_crop
        self.random_scale = random_scale
        self.crop_size = crop_size
        self.scale_factor = scale_factor
        self.rotate = rotate
        self.fliplr = fliplr
        self.fliptb = fliptb
        self.noise = noise

    def __getitem__(self, index):
        img = load_img(self.image_filenames[index])
        # determine valid HR image size with scale factor
        self.crop_size = calculate_valid_crop_size(self.crop_size, self.scale_factor)
        hr_img_w = self.crop_size
        hr_img_h = self.crop_size

        # determine LR image size
        lr_img_w = hr_img_w // self.scale_factor
        lr_img_h = hr_img_h // self.scale_factor

        # random scaling between [0.5, 1.0]
        if self.random_scale:
            eps = 1e-3
            ratio = random.randint(5, 10) * 0.1
            if hr_img_w * ratio < self.crop_size:
                ratio = self.crop_size / hr_img_w + eps
            if hr_img_h * ratio < self.crop_size:
                ratio = self.crop_size / hr_img_h + eps

            scale_w = int(hr_img_w * ratio)
            scale_h = int(hr_img_h * ratio)
            transform = Resize((scale_w, scale_h), interpolation=Image.BICUBIC)
            img = transform(img)

        # random crop
        if self.random_crop:
            transform = RandomCrop(self.crop_size)
        else:
            transform = CenterCrop(self.crop_size)
        img = transform(img)

        if self.noise is not None:# and random.random() < 0.5
            noise_type = self.noise[0]
            noise_value = self.noise[1]
            img_array = np.asarray(img)
            if noise_type == 'Gaussain':
                noises = np.random.normal(loc=0, scale=noise_value, size=img_array.shape)
            elif noise_type == 'Poisson':
                noises = np.random.poisson(img.numpy() * noise_value) / noise_value
                noises = noises - noises.mean(axis=0).mean(axis=0)
            x_noise = img_array + noises
            x_noise = x_noise.clip(0, 255).astype(np.uint8)
            img = ToPILImage()(x_noise)

        # random rotation between [90, 180, 270] degrees
        if self.rotate:
            rv = random.randint(1, 3)
            img = img.rotate(90 * rv, expand=True)

        # random horizontal flip
        if self.fliplr:
            transform = RandomHorizontalFlip()
            img = transform(img)

        # random vertical flip
        if self.fliptb:
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_TOP_BOTTOM)

        # only Y-channel is super-resolved
        if self.is_gray:
            img = img.convert('YCbCr')
            # img, _, _ = img.split()

        # hr_img HR image
        hr_transform = Compose([Resize((hr_img_w, hr_img_h), interpolation=Image.BICUBIC), ToTensor()])
        hr_img = hr_transform(img)

        # lr_img LR image
        lr_transform = Compose([Resize((lr_img_w, lr_img_h), interpolation=Image.BICUBIC), ToTensor()])
        lr_img = lr_transform(img)

        # Bicubic interpolated image
        bc_transform = Compose(
            [ToPILImage(), Resize((hr_img_w, hr_img_h), interpolation=Image.BICUBIC), ToTensor()])
        bc_img = bc_transform(lr_img)

        return lr_img, hr_img, bc_img, self.image_filenames[index]

    def __len__(self):
        return len(self.image_filenames)

class RandomCropDatasetFromFolder(data.Dataset):
    def __init__(self, image_dirs, crop_size, scale_factor):
        super(RandomCropDatasetFromFolder, self).__init__()
        #self.image_filenames = [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]
        self.image_filenames = []
        for image_dir in image_dirs:
            self.image_filenames.extend(join(image_dir, x) for x in sorted(listdir(image_dir)) if is_image_file(x))
        self.target_transform = Compose([
            RandomCrop(crop_size)
        ])
        self.input_transform = Compose([
            Resize(crop_size // scale_factor),
            ToTensor()
        ])
        self.input_resize_transform = Compose([
            Resize(crop_size // scale_factor),
            Resize(crop_size, Image.BICUBIC),
            ToTensor()
        ])

    def __getitem__(self, index):
        img = load_img(self.image_filenames[index])
        # 高斯模糊（高斯滤波）
        blur_img = cv2.GaussianBlur(img, (5, 5), 0) # 高斯核的宽和高必须是技术，0分别为X方向和Y方向的标准差
        # RandomCrop
        crop_img = self.target_transform(blur_img)
        # ToTensor
        target = ToTensor()(crop_img)
        input = self.input_transform(crop_img)
        input_resize = self.input_resize_transform(crop_img)

        return input, target, input_resize, self.image_filenames[index]

    def __len__(self):
        return len(self.image_filenames)

#SRGAN dataset loader
class ImageDataset(data.Dataset):
    def __init__(self, root, lr_transforms=None, hr_transforms=None):
        self.lr_transform = transforms.Compose(lr_transforms)
        self.hr_transform = transforms.Compose(hr_transforms)

        self.files = sorted(glob.glob(root + '/*.*'))

    def __getitem__(self, index):
        img = Image.open(self.files[index % len(self.files)])
        img_lr = self.lr_transform(img)
        img_hr = self.hr_transform(img)

        return {'lr': img_lr, 'hr': img_hr}

    def __len__(self):
        return len(self.files)

#user-define dataset loader
# class DatasetFromFolder(data.Dataset):
#     def __init__(self, image_dir, input_transform=None, target_transform=None):
#         super(DatasetFromFolder, self).__init__()
#         self.image_filenames = [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]
#
#         self.input_transform = input_transform
#         self.target_transform = target_transform
#
#     def __getitem__(self, index):
#         input = load_img(self.image_filenames[index])
#         target = input.copy()
#         if self.input_transform:
#             input = self.input_transform(input)
#         if self.target_transform:
#             target = self.target_transform(target)
#
#         return input, target
#
#     def __len__(self):
#         return len(self.image_filenames)



# L8, S2, P
class TrainDatasetFromFolder(data.Dataset):
    def __init__(self, L8_image_dirs, s2_image_dirs, is_gray=False, random_scale=True, crop_size=432, rotate=True, fliplr=True,
                 fliptb=True, scale_factor=3):
        super(TrainDatasetFromFolder, self).__init__()

        self.L8_image_files = []
        self.s2_image_files = []
        self.L8_image_filenames = []
        self.s2_image_filenames = []
        # print('dataset_________________L8_image_dirs', L8_image_dirs)
        # print('dataset_________________s2_image_dirs', s2_image_dirs)
        #遍历到文件夹ROIs100_spring_land_L8_5这一层级
        for L8_image_dir in L8_image_dirs:
            # print('dataset___________L8_image_dir', L8_image_dir)
            # print('_________________', listdir(L8_image_dir))
            self.L8_image_files.extend(join(L8_image_dir, x) for x in sorted(listdir(L8_image_dir)))
            # print('dataset_____________________self.L8_image_files', self.L8_image_files)
        for s2_image_dir in s2_image_dirs:
            self.s2_image_files.extend(join(s2_image_dir, x) for x in sorted(listdir(s2_image_dir)))
            # print("dataset___________________________self.s2_image_files", self.s2_image_files)
        # 遍历到文件夹ROIs100_spring_land_L8_5/.tif这一层级
        for L8_image_filename in self.L8_image_files:
            self.L8_image_filenames.extend(join(L8_image_filename, x) for x in sorted(listdir(L8_image_filename)) if is_image_file(x))
            # print('______________self.L8_image_filenames', self.L8_image_filenames, len(self.L8_image_filenames))
        for s2_image_filename in self.s2_image_files:
            self.s2_image_filenames.extend(join(s2_image_filename, x) for x in sorted(listdir(s2_image_filename)) if is_image_file(x))
            # print('______________self.s2_image_filenames', self.s2_image_filenames, len(self.s2_image_filenames))
        self.is_gray = is_gray
        # self.random_scale = random_scale
        self.crop_size = crop_size
        # self.rotate = rotate
        # self.fliplr = fliplr
        # self.fliptb = fliptb
        self.scale_factor = scale_factor
        # print('dataset____________________self.s2_image_filenames', self.s2_image_filenames, len(self.s2_image_filenames))

    def __getitem__(self, index):
        # load image
        L8_img = load_img(self.L8_image_filenames[index])
        s2_img = load_img(self.s2_image_filenames[index])

        #print('__________________________L8_img', L8_img, L8_img.size)
        # print('__________________________s2_img', s2_img)

        # determine valid HR image size with scale factor
        self.crop_size = calculate_valid_crop_size(self.crop_size, self.scale_factor)
        hr_img_w = self.crop_size
        hr_img_h = self.crop_size

        # determine LR image size
        lr_img_w = hr_img_w // self.scale_factor
        lr_img_h = hr_img_h // self.scale_factor

        # random scaling between [0.5, 1.0]
        # if self.random_scale:
        #     eps = 1e-3
        #     ratio = random.randint(5, 10) * 0.1
        #     if hr_img_w * ratio < self.crop_size:
        #         ratio = self.crop_size / hr_img_w + eps
        #     if hr_img_h * ratio < self.crop_size:
        #         ratio = self.crop_size / hr_img_h + eps
        #
        #     scale_w = int(hr_img_w * ratio)
        #     scale_h = int(hr_img_h * ratio)
        #     transform = Resize((scale_w, scale_h), interpolation=Image.BICUBIC)
        #     img = transform(img)

        # random crop
        # transform = RandomCrop(self.crop_size)
        # img = transform(img)

        # random rotation between [90, 180, 270] degrees
        # if self.rotate:
        #     rv = random.randint(1, 3)
        #     img = img.rotate(90 * rv, expand=True)

        # random horizontal flip
        # if self.fliplr:
        #     transform = RandomHorizontalFlip()
        #     img = transform(img)

        # random vertical flip
        # if self.fliptb:
        #     if random.random() < 0.5:
        #         img = img.transpose(Image.FLIP_TOP_BOTTOM)

        # only Y-channel is super-resolved
        if self.is_gray:
            L8_img = L8_img.convert('YCbCr')
            s2_img = s2_img.convert('YCbCr')
            # img, _, _ = img.split()

        # hr_img HR image
        #hr_transform = Compose([Scale((hr_img_w, hr_img_h), interpolation=Image.BICUBIC), ToTensor()])
        # hr_transform = Compose([Resize((hr_img_w, hr_img_h), interpolation=Image.BICUBIC), ToTensor()])
        #hr_transform = Compose([ToTensor()])
        #hr_img = hr_transform(s2_img)
        # hr_transform = functional.to_tensor()
        hr_img = functional.to_tensor(s2_img)

        # lr_img LR image
        #lr_transform = Compose([Scale((lr_img_w, lr_img_h), interpolation=Image.BICUBIC), ToTensor()])
        # lr_transform = Compose([Resize((lr_img_w, lr_img_h), interpolation=Image.BICUBIC), ToTensor()])
        #lr_transform = Compose([ToTensor()])
        #lr_img = lr_transform(L8_img)
        lr_img = functional.to_tensor(L8_img)

        # Bicubic interpolated image
        #bc_transform = Compose([ToPILImage(), Scale((hr_img_w, hr_img_h), interpolation=Image.BICUBIC), ToTensor()])
        #bc_transform = Compose([ToPILImage(), Resize((hr_img_w, hr_img_h), interpolation=Image.BICUBIC), ToTensor()])
        #bc_img = bc_transform(L8_img)
        bc_img = functional.resize(L8_img, size=[hr_img_w, hr_img_h], interpolation=Image.BICUBIC)
        bc_img = functional.to_tensor(bc_img)

        return lr_img, hr_img, bc_img, self.L8_image_filenames[index]

    def __len__(self):
        return len(self.L8_image_filenames)

class DatasetFromFolder2(data.Dataset):
    def __init__(self, L8_test_dirs, s2_test_dirs, input_transform=None, input_resize_transform=None, target_transform=None, is_gray=False, max_sample_num=100000):
        super(DatasetFromFolder2, self).__init__()
        #self.image_filenames = [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]
        self.L8_image_files = []
        self.s2_image_files = []
        self.L8_image_filenames = []
        self.s2_image_filenames = []

        # image_files = []
        #遍历到ROIs777_winter_land_L8_2文件夹这一层级
        for L8_test_dir in L8_test_dirs:
            self.L8_image_files.extend(join(L8_test_dir, x) for x in sorted(listdir(L8_test_dir)))
        for s2_test_dir in s2_test_dirs:
            self.s2_image_files.extend(join(s2_test_dir, x) for x in sorted(listdir(s2_test_dir)))
        #遍历到ROIs777_winter_land_L8_2/.tif这一层级
        for L8_image_file in self.L8_image_files:
            self.L8_image_filenames.extend(join(L8_image_file, x) for x in sorted(listdir(L8_image_file)) if is_image_file(x))
        for s2_image_file in self.s2_image_files:
            self.s2_image_filenames.extend(join(s2_image_file, x) for x in sorted(listdir(s2_image_file)) if is_image_file(x))

        # sampleNum = len(image_files)
        # if sampleNum > max_sample_num:
        #     indicies = random.sample(range(sampleNum), max_sample_num)
        #     self.image_filenames = [image_files[i] for i in indicies]
        # else:
        #     self.image_filenames = image_files

        self.input_transform = input_transform
        self.input_resize_transform = input_resize_transform
        self.target_transform = target_transform
        self.is_gray = is_gray

    def __getitem__(self, index):
        L8_input = load_img(self.L8_image_filenames[index])
        s2_input = load_img(self.s2_image_filenames[index])
        # only Y-channel is super-resolved
        if self.is_gray:
            L8_input = L8_input.convert('YCbCr')
            s2_input = s2_input.convert('YCbCr')
        target = s2_input.copy()
        input_resize = L8_input.copy()
        if self.input_transform:
            L8_input = self.input_transform(input_resize)
        if self.input_resize_transform:
            input_resize = self.input_resize_transform(input_resize)
        if self.target_transform:
            target = self.target_transform(target)

        return L8_input, target, input_resize, self.L8_image_filenames[index]

    def __len__(self):
        return len(self.L8_image_filenames)



# Train, Test
class RGB_TrainDatasetFromFolder(data.Dataset):
    def __init__(self, image_dirs, is_gray=False, random_scale=True, crop_size=256, rotate=True, fliplr=True,
                 fliptb=True, scale_factor=3):
        super(RGB_TrainDatasetFromFolder, self).__init__()

        self.image_filenames = []
        for image_dir in image_dirs:
            self.image_filenames.extend(join(image_dir, x) for x in sorted(listdir(image_dir)) if is_image_file(x))
        #print('train_dataset---------self.image_filenames:', self.image_filenames, len(self.image_filenames))
        self.is_gray = is_gray
        self.random_scale = random_scale
        self.crop_size = crop_size
        self.rotate = rotate
        self.fliplr = fliplr
        self.fliptb = fliptb
        self.scale_factor = scale_factor

    def __getitem__(self, index):
        # load image
        img = load_img(self.image_filenames[index])
        # img = functional.center_crop(img, self.crop_size)  # 288*288 center_crop to 256*256

        # determine valid HR image size with scale factor
        self.crop_size = calculate_valid_crop_size(self.crop_size, self.scale_factor)
        hr_img_w = self.crop_size
        hr_img_h = self.crop_size

        # determine LR image size
        lr_img_w = hr_img_w // self.scale_factor
        lr_img_h = hr_img_h // self.scale_factor

        # # random scaling between [0.5, 1.0]
        # if self.random_scale:
        #     eps = 1e-3
        #     ratio = random.randint(5, 10) * 0.1
        #     if hr_img_w * ratio < self.crop_size:
        #         ratio = self.crop_size / hr_img_w + eps
        #     if hr_img_h * ratio < self.crop_size:
        #         ratio = self.crop_size / hr_img_h + eps
        #
        #     scale_w = int(hr_img_w * ratio)
        #     scale_h = int(hr_img_h * ratio)
        #     transform = Resize((scale_w, scale_h), interpolation=Image.BICUBIC)
        #     img = transform(img)
        #
        # # random crop
        # transform = RandomCrop(self.crop_size)
        # img = transform(img)
        #
        # # random rotation between [90, 180, 270] degrees
        # if self.rotate:
        #     rv = random.randint(1, 3)
        #     img = img.rotate(90 * rv, expand=True)
        #
        # # random horizontal flip
        # if self.fliplr:
        #     transform = RandomHorizontalFlip()
        #     img = transform(img)
        #
        # # random vertical flip
        # if self.fliptb:
        #     if random.random() < 0.5:
        #         img = img.transpose(Image.FLIP_TOP_BOTTOM)
        #
        # # only Y-channel is super-resolved
        # if self.is_gray:
        #     img = img.convert('YCbCr')
        #     # img, _, _ = img.split()

        # hr_img HR image
        #hr_transform = Compose([Scale((hr_img_w, hr_img_h), interpolation=Image.BICUBIC), ToTensor()])

        # hr_transform = Compose([Resize((hr_img_w, hr_img_h), interpolation=Image.BICUBIC), ToTensor()])
        # hr_img = hr_transform(img)

        hr_img = functional.to_tensor(img)

        # lr_img LR image
        # lr_transform = Compose([Resize((lr_img_w, lr_img_h), interpolation=Image.BICUBIC), ToTensor()])
        # lr_img = lr_transform(img)
        lr_img = functional.resize(img, size=[lr_img_w, lr_img_h], interpolation=Image.BICUBIC)
        lr_input = lr_img.copy()
        lr_img = functional.to_tensor(lr_img)

        # Bicubic interpolated image
        # bc_transform = Compose([ToPILImage(), Resize((hr_img_w, hr_img_h), interpolation=Image.BICUBIC), ToTensor()])
        # bc_img = bc_transform(lr_img)
        bc_img = functional.resize(lr_input, size=[hr_img_w, hr_img_h], interpolation=Image.BICUBIC)
        bc_img = functional.to_tensor(bc_img)

        return lr_img, hr_img, bc_img, self.image_filenames[index]

    def __len__(self):
        return len(self.image_filenames)


class RGB_DatasetFromFolder2(data.Dataset):
    def __init__(self, image_dirs, input_transform=None, input_resize_transform=None, target_transform=None, is_gray=False, scale_factor=3, max_sample_num=100000):
        super(RGB_DatasetFromFolder2, self).__init__()
        #self.image_filenames = [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]
        image_files = []
        for image_dir in image_dirs:
            image_files.extend(join(image_dir, x) for x in sorted(listdir(image_dir)) if is_image_file(x))

        self.image_filenames = image_files
        #print('test_dataset--------------self.image_filenames:', self.image_filenames, len(self.image_filenames))
        self.input_transform = input_transform  # lr
        self.input_resize_transform = input_resize_transform   # bicubic
        self.target_transform = target_transform   # hr
        self.is_gray = is_gray

    def __getitem__(self, index):
        input = load_img(self.image_filenames[index])
        # only Y-channel is super-resolved
        if self.is_gray:
            input = input.convert('YCbCr')
        target = input.copy()
        input_resize = input.copy()
        if self.input_transform:
            input = self.input_transform(input)
        if self.input_resize_transform:
            input_resize = self.input_resize_transform(input_resize)
        if self.target_transform:
            target = self.target_transform(target)

        return input, target, input_resize, self.image_filenames[index]

    def __len__(self):
        return len(self.image_filenames)