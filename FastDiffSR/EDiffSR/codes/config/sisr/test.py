import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import argparse
import logging
import os.path
import sys
import time
from collections import OrderedDict
import torchvision.utils as tvutils

import numpy as np
import torch
import lpips

import options as option
from models import create_model

sys.path.insert(0, "../../")
import utils as util
from data import create_dataloader, create_dataset
from data.util import bgr2ycbcr

#### options
parser = argparse.ArgumentParser()
parser.add_argument("-opt", type=str, default="options/test/aid.yml", help="Path to options YMAL file.")
opt = option.parse(parser.parse_args().opt, is_train=False)

opt = option.dict_to_nonedict(opt)

#### mkdir and logger
util.mkdirs(
    (
        path
        for key, path in opt["path"].items()
        if not key == "experiments_root"
        and "pretrain_model" not in key
        and "resume" not in key
    )
)

os.system("rm ./result")
os.symlink(os.path.join(opt["path"]["results_root"], ".."), "./result")

util.setup_logger(
    "base",
    opt["path"]["log"],
    "test_" + opt["name"],
    level=logging.INFO,
    screen=True,
    tofile=True,
)
logger = logging.getLogger("base")
logger.info(option.dict2str(opt))

#### Create test dataset and dataloader
test_loaders = []
for phase, dataset_opt in sorted(opt["datasets"].items()):
    test_set = create_dataset(dataset_opt)
    test_loader = create_dataloader(test_set, dataset_opt)
    logger.info(
        "Number of test images in [{:s}]: {:d}".format(
            dataset_opt["name"], len(test_set)
        )
    )
    test_loaders.append(test_loader)

# load pretrained model by default
model = create_model(opt)
device = model.device

sde = util.IRSDE(max_sigma=opt["sde"]["max_sigma"], T=opt["sde"]["T"], schedule=opt["sde"]["schedule"], eps=opt["sde"]["eps"], device=device)

sde.set_model(model.model)


scale = opt['degradation']['scale']

for test_loader in test_loaders:
    test_set_name = test_loader.dataset.opt["name"]  # path opt['']
    logger.info("\nTesting [{:s}]...".format(test_set_name))
    test_start_time = time.time()
    dataset_dir = os.path.join(opt["path"]["results_root"], test_set_name)
    util.mkdir(dataset_dir)

    test_times = []
    avg_psnr = 0.0
    avg_ssim = 0.0
    avg_ergas = 0.0
    avg_lpips = 0.0
    idx = 0

    for i, test_data in enumerate(test_loader):

        need_GT = False if test_loader.dataset.opt["dataroot_GT"] is None else True
        img_path = test_data["GT_path"][0] if need_GT else test_data["LQ_path"][0]
        img_name = os.path.splitext(os.path.basename(img_path))[0]

        #### input dataset_LQ
        LQ, GT = test_data["LQ"], test_data["GT"]
        LQ = util.upscale(LQ, scale)
        noisy_state = sde.noise_state(LQ)

        model.feed_data(noisy_state, LQ, GT)
        tic = time.time()
        model.test(sde, save_states=True)
        toc = time.time()
        test_times.append(toc - tic)

        visuals = model.get_current_visuals()
        SR_img = visuals["Output"]
        output = util.tensor2img(SR_img.squeeze())  # uint8
        LQ_ = util.tensor2img(visuals["Input"].squeeze())  # uint8
        GT_ = util.tensor2img(visuals["GT"].squeeze())  # uint8

        # calculate PSNR/ssim/ergas/lpips
        SR_psnr = util.calculate_psnr(output, GT_)
        SR_ssim = util.calculate_ssim(output, GT_)
        SR_ergas = util.calc_ergas(output, GT_, scale=scale)
        SR_lpips = util.calc_lpips(output, GT_)

        avg_psnr += SR_psnr
        avg_ssim += SR_ssim
        avg_ergas += SR_ergas
        avg_lpips += SR_lpips

        idx += 1
        
        suffix = opt["suffix"]
        if suffix:
            save_img_path = os.path.join(dataset_dir, img_name + suffix + ".png")
        else:
            save_img_path = os.path.join(dataset_dir, img_name + ".png")
        util.save_img(output, save_img_path)

        result_imgs = [GT_, LQ_, LQ_, output]
        mses = [None, None, 0, 0]
        psnrs = [None, None, 0, SR_psnr]
        ssims = [None, None, 0, SR_ssim]
        ergas = [None, None, 0, SR_ergas]
        lpips = [None, None, 0, SR_lpips]
        util.plot_img(result_imgs, mses, psnrs, ssims, ergas, lpips, '{}/{}_plot.png'.format(dataset_dir, img_name))

    avg_psnr = avg_psnr / idx
    avg_ssim = avg_ssim / idx
    avg_ergas = avg_ergas / idx
    avg_lpips = avg_lpips / idx

    # log
    #logger.info("# Validation # PSNR: {:.5f}, SSIM: {:.5f}, ERGAS: {:.5f}, LPIPS: {:.5f},".format(avg_psnr, avg_ssim, avg_ergas, avg_lpips))
    #logger.info(f"average test time: {np.mean(test_times):.4f}")

    logger.info('Test # PSNR: %0.5e, SSIM：%0.5e, ERGAS: %0.4e, LPIPS: %0.5e' % (avg_psnr, avg_ssim, avg_ergas, avg_lpips))
    avg_test_time = np.mean(test_times)
    logger.info('average test time: %0.5e' % (avg_test_time))


    print(f"average test time: {np.mean(test_times):.5f}")
