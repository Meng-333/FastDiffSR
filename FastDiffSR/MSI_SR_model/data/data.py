from os.path import exists, join, basename
from os import makedirs, remove
import os
from six.moves import urllib
import tarfile
from .dataset import *

def download_bsds300(dest="dataset"):
    output_image_dir = join(dest, "BSDS300/images")

    if not exists(output_image_dir):
        if not exists(dest):
            makedirs(dest)
        url = "http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/segbench/BSDS300-images.tgz"
        print("downloading url ", url)

        data = urllib.request.urlopen(url)

        file_path = join(dest, basename(url))
        with open(file_path, 'wb') as f:
            f.write(data.read())

        print("Extracting data")
        with tarfile.open(file_path) as tar:
            for item in tar:
                tar.extract(item, dest)

        remove(file_path)

    return output_image_dir

def get_training_set(data_dir, datasets, crop_size, scale_factor, is_gray=False):
    train_dir = []
    for dataset in datasets:
        if dataset == 'bsds300':
            root_dir = download_bsds300(data_dir)
            train_dir.append(join(root_dir, "train"))
        elif dataset == 'DIV2K':
            train_dir.append(join(data_dir, dataset, 'DIV2K_train_LR_bicubic/X4'))
        elif dataset in ["UCMerced_LandUse", "AID", "RS_images_2800", "RSI-CB256", "DOTA"]:
            datasetpth = join(data_dir, dataset)
            train_dir.append(datasetpth)
            dirs = os.listdir(datasetpth)
            for dir in dirs:
                dirpath = join(datasetpth, dir)
                if os.path.isdir(dirpath):
                    train_dir.append(dirpath)
        else:
            datasetpth = join(data_dir, dataset)
            train_dir.append(datasetpth)
            dirs = os.listdir(datasetpth)
            for dir in dirs:
                dirpath = join(datasetpth, dir)
                if os.path.isdir(dirpath):
                    train_dir.append(dirpath)

    return TrainDatasetFromFolder(train_dir,
                                  is_gray=is_gray,
                                  random_scale=True,    # random scaling
                                  crop_size=crop_size,  # random crop
                                  rotate=True,          # random rotate
                                  fliplr=True,          # random flip
                                  fliptb=True,
                                  scale_factor=scale_factor)

def get_training_set_random(data_dir, datasets, crop_size, scale_factor, is_gray=False):
    train_dir = []
    for dataset in datasets:
        if dataset == 'bsds300':
            root_dir = download_bsds300(data_dir)
            train_dir.append(join(root_dir, "train"))
        elif dataset == 'DIV2K':
            train_dir.append(join(data_dir, dataset, 'DIV2K_train_LR_bicubic/X4'))
        elif dataset in ["UCMerced_LandUse", "AID", "RS_images_2800", "RSI-CB256", "DOTA"]:
            datasetpth = join(data_dir, dataset)
            train_dir.append(datasetpth)
            dirs = os.listdir(datasetpth)
            for dir in dirs:
                dirpath = join(datasetpth, dir)
                if os.path.isdir(dirpath):
                    train_dir.append(dirpath)
        else:
            datasetpth = join(data_dir, dataset)
            train_dir.append(datasetpth)
            dirs = os.listdir(datasetpth)
            for dir in dirs:
                dirpath = join(datasetpth, dir)
                if os.path.isdir(dirpath):
                    train_dir.append(dirpath)

    return TrainDatasetFromFolder(train_dir,
                                  is_gray=is_gray,
                                  random_scale=True,    # random scaling
                                  crop_size=crop_size,  # random crop
                                  rotate=True,          # random rotate
                                  fliplr=True,          # random flip
                                  fliptb=True,
                                  scale_factor=scale_factor)

def get_training_set_centercrop(data_dir, datasets, crop_size, scale_factor, is_gray=False, max_sample_num=100000):
    train_dir = []
    for dataset in datasets:
        if dataset == 'bsds300':
            root_dir = download_bsds300(data_dir)
            train_dir.append(join(root_dir, "train"))
        elif dataset == 'DIV2K':
            train_dir.append(join(data_dir, dataset, 'DIV2K_train_LR_bicubic/X4'))
        elif dataset in ["UCMerced_LandUse", "AID", "RS_images_2800", "RSI-CB256", "DOTA"]:
            datasetpth = join(data_dir, dataset)
            train_dir.append(datasetpth)
            dirs = os.listdir(datasetpth)
            for dir in dirs:
                dirpath = join(datasetpth, dir)
                if os.path.isdir(dirpath):
                    train_dir.append(dirpath)
        else:
            datasetpth = join(data_dir, dataset)
            train_dir.append(datasetpth)
            dirs = os.listdir(datasetpth)
            for dir in dirs:
                dirpath = join(datasetpth, dir)
                if os.path.isdir(dirpath):
                    train_dir.append(dirpath)
    input_transform = Compose([
        CenterCrop(crop_size),
        Resize(crop_size // scale_factor),
        ToTensor()
    ])
    srcnn_input_transform = Compose([
        CenterCrop(crop_size),
        Resize(crop_size // scale_factor),
        Resize(crop_size, interpolation=Image.BICUBIC),
        ToTensor()
    ])
    target_transform = Compose([
        CenterCrop(crop_size),
        ToTensor()
    ])
    return DatasetFromFolder2(train_dir, input_transform=input_transform, input_resize_transform=srcnn_input_transform, target_transform=target_transform, is_gray=is_gray, max_sample_num=max_sample_num)

def get_training_set_randomcrop(data_dir, datasets, crop_size, scale_factor, is_gray=False):
    train_dir = []
    for dataset in datasets:
        if dataset == 'bsds300':
            root_dir = download_bsds300(data_dir)
            train_dir.append(join(root_dir, "train"))
        elif dataset == 'DIV2K':
            train_dir.append(join(data_dir, dataset, 'DIV2K_train_LR_bicubic/X4'))
        elif dataset in ["UCMerced_LandUse", "AID", "RS_images_2800", "RSI-CB256", "DOTA"]:
            datasetpth = join(data_dir, dataset)
            train_dir.append(datasetpth)
            dirs = os.listdir(datasetpth)
            for dir in dirs:
                dirpath = join(datasetpth, dir)
                if os.path.isdir(dirpath):
                    train_dir.append(dirpath)
        else:
            datasetpth = join(data_dir, dataset)
            train_dir.append(datasetpth)
            dirs = os.listdir(datasetpth)
            for dir in dirs:
                dirpath = join(datasetpth, dir)
                if os.path.isdir(dirpath):
                    train_dir.append(dirpath)

    return RandomCropDatasetFromFolder(train_dir, crop_size,scale_factor)

def get_test_sets(data_dir, datasets, scale_factor, is_gray=False):
    test_dir = []
    for dataset in datasets:
        if dataset == 'bsds300':
            root_dir = download_bsds300(data_dir)
            test_dir.append(join(root_dir, "test"))
        elif dataset == 'DIV2K':
            test_dir.append(join(data_dir, dataset, 'DIV2K_test_LR_bicubic/X4'))
        elif dataset == "RSI":
            datasetpth = join(data_dir, dataset)
            test_dir.append(datasetpth)
            dirs = os.listdir(datasetpth)
            for dir in dirs:
                dirpath = join(datasetpth, dir)
                if os.path.isdir(dirpath):
                    test_dir.append(dirpath)
        else:
            datasetpth = join(data_dir, dataset)
            test_dir.append(datasetpth)
            dirs = os.listdir(datasetpth)
            for dir in dirs:
                dirpath = join(datasetpth, dir)
                if os.path.isdir(dirpath):
                    test_dir.append(dirpath)

    return TestDatasetFromFolder(test_dir,
                                 is_gray=is_gray,
                                 scale_factor=scale_factor)

def get_test_sets_by_cropsize(data_dir, datasets, crop_size, scale_factor, is_gray=False):
    test_dir = []
    for dataset in datasets:
        if dataset == 'bsds300':
            root_dir = download_bsds300(data_dir)
            test_dir.append(join(root_dir, "test"))
        elif dataset == 'DIV2K':
            test_dir.append(join(data_dir, dataset, 'DIV2K_test_LR_bicubic/X4'))
        elif dataset == "RSI":
            datasetpth = join(data_dir, dataset)
            test_dir.append(datasetpth)
            dirs = os.listdir(datasetpth)
            for dir in dirs:
                dirpath = join(datasetpth, dir)
                if os.path.isdir(dirpath):
                    test_dir.append(dirpath)
        else:
            datasetpth = join(data_dir, dataset)
            test_dir.append(datasetpth)
            dirs = os.listdir(datasetpth)
            for dir in dirs:
                dirpath = join(datasetpth, dir)
                if os.path.isdir(dirpath):
                    test_dir.append(dirpath)
    input_transform = Compose([
        CenterCrop(crop_size),
        Resize(crop_size // scale_factor),
        ToTensor()
    ])
    srcnn_input_transform = Compose([
        CenterCrop(crop_size),
        Resize(crop_size // scale_factor),
        Resize(crop_size, Image.BICUBIC),
        ToTensor()
    ])
    target_transform = Compose([
        CenterCrop(crop_size),
        ToTensor()
    ])
    return DatasetFromFolder2(test_dir, input_transform=input_transform, input_resize_transform=srcnn_input_transform, target_transform=target_transform)

def get_datasets(data_dir, datasets, crop_size, scale_factor, is_gray=False, noise=['Gaussain', 1], max_sample_num=100000):
    test_dir = []
    for dataset in datasets:
        if dataset == 'bsds300':
            root_dir = download_bsds300(data_dir)
            test_dir.append(join(root_dir, "test"))
        elif dataset == 'DIV2K':
            test_dir.append(join(data_dir, dataset, 'DIV2K_test_LR_bicubic/X4'))
        elif dataset == "RSI":
            datasetpth = join(data_dir, dataset)
            test_dir.append(datasetpth)
            dirs = os.listdir(datasetpth)
            for dir in dirs:
                dirpath = join(datasetpth, dir)
                if os.path.isdir(dirpath):
                    test_dir.append(dirpath)
        else:
            datasetpth = join(data_dir, dataset)
            test_dir.append(datasetpth)
            dirs = os.listdir(datasetpth)
            for dir in dirs:
                dirpath = join(datasetpth, dir)
                if os.path.isdir(dirpath):
                    test_dir.append(dirpath)
    input_transform = Compose([
        CenterCrop(crop_size),
        Resize(crop_size // scale_factor),
        # ToTensor()
    ])
    srcnn_input_transform = Compose([
        # CenterCrop(crop_size),
        # Resize(crop_size // scale_factor),
        Resize(crop_size, Image.BICUBIC),
        ToTensor()
    ])
    target_transform = Compose([
        CenterCrop(crop_size),
        ToTensor()
    ])
    return DatasetFromFolder(test_dir, crop_size=crop_size, noise=noise, input_transform=input_transform,
                             input_resize_transform=srcnn_input_transform, target_transform=target_transform, max_sample_num=max_sample_num)

def get_test_set(data_dir, dataset, scale_factor, is_gray=False):
    dataset = dataset[0]
    if dataset == 'bsds300':
        root_dir = download_bsds300(data_dir)
        test_dir = join(root_dir, "test")
    elif dataset == 'DIV2K':
        test_dir = join(data_dir, dataset, 'DIV2K_test_LR_bicubic/X4')
    else:
        test_dir = join(data_dir, dataset)

    return TestDatasetFromFolder(test_dir,
                                 is_gray=is_gray,
                                 scale_factor=scale_factor)


# L8 S2 P
def get_training_datasets(data_dir, datasets, crop_size, scale_factor, is_gray=False):
    L8_train_dir = []
    s2_train_dir = []
    for dataset in datasets:
        if dataset == 'train/train_S2':
            L8_train_dir.append(join(data_dir, dataset))
        elif dataset == 'train/train_P':
            s2_train_dir.append(join(data_dir, dataset))
    return TrainDatasetFromFolder(L8_train_dir,
                                  s2_train_dir,
                                  is_gray=is_gray,
                                  random_scale=True,    # random scaling
                                  crop_size=crop_size,  # random crop
                                  rotate=True,          # random rotate
                                  fliplr=True,          # random flip
                                  fliptb=True,
                                  scale_factor=scale_factor)

def get_test_datasets(data_dir, datasets, crop_size, scale_factor, is_gray=False):
    L8_test_dir = []
    s2_test_dir = []
    for dataset in datasets:
        if dataset == 'test/test_S2':
            L8_test_dir.append(join(data_dir, dataset))
        elif dataset == 'test/test_P':
            s2_test_dir.append(join(data_dir, dataset))
        # elif dataset == "RSI":
        #     datasetpth = join(data_dir, dataset)
        #     test_dir.append(datasetpth)
        #     dirs = os.listdir(datasetpth)
        #     for dir in dirs:
        #         dirpath = join(datasetpth, dir)
        #         if os.path.isdir(dirpath):
        #             test_dir.append(dirpath)
        # else:
        #     datasetpth = join(data_dir, dataset)
        #     test_dir.append(datasetpth)
        #     dirs = os.listdir(datasetpth)
        #     for dir in dirs:
        #         dirpath = join(datasetpth, dir)
        #         if os.path.isdir(dirpath):
        #             test_dir.append(dirpath)
    input_transform = Compose([
        # CenterCrop(crop_size),
        # Resize(crop_size // scale_factor),
        # ToTensor()
        # CenterCrop(crop_size // scale_factor),
        ToTensor()
    ])
    srcnn_input_transform = Compose([
        # CenterCrop(crop_size),
        # Resize(crop_size // scale_factor),
        # Resize(crop_size, Image.BICUBIC),
        # ToTensor()
        # CenterCrop(crop_size // scale_factor),
        Resize(crop_size, Image.BICUBIC),
        ToTensor()
    ])
    target_transform = Compose([
        # CenterCrop(crop_size),
        ToTensor()
    ])
    return DatasetFromFolder2(L8_test_dir, s2_test_dir, input_transform=input_transform, input_resize_transform=srcnn_input_transform, target_transform=target_transform)

# AID_288, LoveDA_288, WHU-RS19_288
# def get_RGB_trainDataset(data_dir, datasets, crop_size, scale_factor, is_gray=False):
#     train_dirs = []
#     for dataset in datasets:
#         if dataset == 'AID_288':  # AID
#             file_path = join(data_dir, dataset)
#             dirs = os.listdir(file_path)
#             #print('train_data---------------AIDdirs:', dirs, len(dirs))
#             for dir in dirs:
#                 dir_path = join(file_path, dir)
#                 if os.path.isdir(dir_path):
#                     train_dirs.append(dir_path)  # 最后一层类别文件夹
#         elif dataset == 'LoveDA_288':   # LoveDA
#             filePath = join(data_dir, dataset)
#             img_dirs = os.listdir(filePath)  # Test Train
#             #print('train_data---------------LoveDA_img_dirs:', img_dirs, len(img_dirs))
#             for dirs in img_dirs:
#                 dirPath = join(filePath, dirs)
#                 dirs = sorted(os.listdir(dirPath))  # Rural Urban
#                 #print('train_data---------------LoveDAdirs:', dirs, len(dirs))
#                 for d in dirs:
#                     d_path = join(dirPath, d)
#                     if os.path.isdir(d_path):
#                         train_dirs.append(d_path)  # 最后一层Rural, Urban文件夹
#     #print('data--------------train_dirs:', train_dirs, len(train_dirs))
#     return RGB_TrainDatasetFromFolder(train_dirs,
#                                   is_gray=is_gray,
#                                   random_scale=True,    # random scaling
#                                   crop_size=crop_size,  # random crop
#                                   rotate=True,          # random rotate
#                                   fliplr=True,          # random flip
#                                   fliptb=True,
#                                   scale_factor=scale_factor)
#
# def get_RGB_testDataset(data_dir, datasets, crop_size, scale_factor, is_gray=False):
#     test_dirs = []
#     dataset = datasets[0]
#     if dataset == 'WHU-RS19_288':
#         file_path = join(data_dir, dataset)
#         dirs = sorted(os.listdir(file_path))
#         #print('test_data---------------dirs:', dirs, len(dirs))
#         for dir in dirs:
#             dir_path = join(file_path, dir)
#             if os.path.isdir(dir_path):
#                 test_dirs.append(dir_path)  # 最后一层类别文件夹
#     else:
#         test_dirs = datasets
#     #print('data---------------test_dirs:', test_dirs, len(test_dirs))
#     input_transform = Compose([  # lr
#         CenterCrop(crop_size),
#         Resize(crop_size // scale_factor),
#         ToTensor()
#     ])
#     srcnn_input_transform = Compose([  # bicubic
#         CenterCrop(crop_size),
#         Resize(crop_size // scale_factor),
#         Resize(crop_size, Image.BICUBIC),
#         ToTensor()
#     ])
#     target_transform = Compose([   # hr
#         CenterCrop(crop_size),
#         ToTensor()
#     ])
#     return RGB_DatasetFromFolder2(test_dirs,
#                                   input_transform=input_transform, input_resize_transform=srcnn_input_transform,
#                                    target_transform=target_transform, is_gray=is_gray, scale_factor=scale_factor)



# Train, Test     --- all is 256*256 size ---
def get_RGB_trainDataset(data_dir, datasets, crop_size, scale_factor, is_gray=False):
    train_dirs = []
    for dataset in datasets:
        if dataset == 'Train':   # Train
            dir_path = join(data_dir, dataset)
            train_dirs.append(dir_path)  # 最后一层类别文件夹
    return RGB_TrainDatasetFromFolder(train_dirs,
                                  is_gray=is_gray,
                                  random_scale=True,    # random scaling
                                  crop_size=crop_size,  # random crop
                                  rotate=True,          # random rotate
                                  fliplr=True,          # random flip
                                  fliptb=True,
                                  scale_factor=scale_factor)

def get_RGB_testDataset(data_dir, datasets, crop_size, scale_factor, is_gray=False):
    test_dirs = []
    dataset = datasets[0]
    if dataset == 'Test':
        file_path = join(data_dir, dataset)
        dirs = sorted(os.listdir(file_path))
        for dir in dirs:
            dir_path = join(file_path, dir)
            if os.path.isdir(dir_path):
                test_dirs.append(dir_path)  # 最后一层类别文件夹
    else:
        test_dirs = datasets
    input_transform = Compose([  # lr
        # CenterCrop(crop_size),
        Resize(crop_size // scale_factor),
        ToTensor()
    ])
    srcnn_input_transform = Compose([  # bicubic
        # CenterCrop(crop_size),
        Resize(crop_size // scale_factor),
        Resize(crop_size, Image.BICUBIC),
        ToTensor()
    ])
    target_transform = Compose([   # hr
        # CenterCrop(crop_size),
        ToTensor()
    ])
    return RGB_DatasetFromFolder2(test_dirs,
                                  input_transform=input_transform, input_resize_transform=srcnn_input_transform,
                                   target_transform=target_transform, is_gray=is_gray, scale_factor=scale_factor)