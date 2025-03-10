import torch
import os, argparse
from model.transenet import TransENetModel
from utils.utils import mkdir_and_rename

"""parsing and configuration"""

def parse_args():
    desc = "PyTorch implementation of SR collections"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--model_name', type=str, default='TransENet',
                        choices=['SwinIR', 'HSENet', 'TransENet', 'NDSRGAN', 'HAT'], help='The type of model')
    parser.add_argument('--root_dir', type=str, default='/root/mfe/FastDiffSR/MSI_SR_model/')
    parser.add_argument('--data_dir', type=str, default='/root/mfe/FastDiffSR/dataset/Train/')
    parser.add_argument('--train_dataset', type=list, default=["Train"], choices=["Train"],
                        help='The name of training dataset')
    parser.add_argument('--test_dataset', type=list,
                        default=["Test"],
                        help='The name of test dataset')
    parser.add_argument('--crop_size', type=int, default=256, help='Size of cropped HR image')
    parser.add_argument('--num_threads', type=int, default=8, help='number of threads for data loader to use')
    parser.add_argument('--num_channels', type=int, default=3, help='The number of channels to super-resolve')
    parser.add_argument('--scale_factor', type=int, default=4, help='Size of scale factor')
    parser.add_argument('--epoch', type=int, default=0, help='epoch to start training from')
    parser.add_argument('--num_epochs', type=int, default=100, help='The number of epochs to run')
    parser.add_argument('--save_epochs', type=int, default=1, help='Save trained model every this epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='training batch size')
    parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size')
    parser.add_argument('--save_dir', type=str, default='Result', help='Directory name to save the results')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate default 0.0002')
    parser.add_argument('--b1', type=float, default=0.9, help='adam: decay of first order momentum of gradient')
    parser.add_argument('--b2', type=float, default=0.99, help='adam: decay of first order momentum of gradient')
    parser.add_argument('--gpu_mode', type=bool, default=True)

    parser.add_argument('--test_crop_size', type=int, default=256, help='Size of cropped HR image')
    parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
    parser.add_argument('--hr_height', type=int, default=256, help='size of high res. image height')
    parser.add_argument('--hr_width', type=int, default=256, help='size of high res. image width')
    parser.add_argument('--sample_interval', type=int, default=1000,
                        help='interval between sampling of images from generators')
    # wgan & wgan_gp
    parser.add_argument('--clip_value', type=float, default=0.01, help='lower and upper clip value for disc. weights')
    parser.add_argument('--lambda_gp', type=float, default=10, help='Loss weight for gradient penalty')
    parser.add_argument('--gp', type=bool, default=True, help='gradient penalty')
    parser.add_argument('--penalty_type', type=str, default='LS', choices=["LS", "hinge"], help='gradient type')
    parser.add_argument('--grad_penalty_Lp_norm', type=str, default='L2', choices=["L2", "L1", "Linf"],
                        help='gradient penalty Lp norm')
    parser.add_argument('--relativeGan', type=bool, default=False, help='relative GAN')
    parser.add_argument('--loss_Lp_norm', type=str, default='L1', choices=["L2", "L1"], help='loss Lp norm')
    parser.add_argument('--weight_content', type=float, default=1e-2, help='Loss weight for content loss　')
    parser.add_argument('--weight_gan', type=float, default=1e-3, help='Loss weight for gan loss')

    parser.add_argument('--max_train_samples', type=int, default=40000, help='Max training samples')
    # srragan
    parser.add_argument('--is_train', type=bool, default=True, help='if at training stage')

    return check_args(parser.parse_args())


"""checking arguments"""

def check_args(args):
    # --save_dir
    args.save_dir = os.path.join(args.save_dir, args.model_name)
    if args.epoch == 0:
        mkdir_and_rename(os.path.join(args.root_dir, args.save_dir))  # rename old experiments if exists

    # --epoch
    try:
        assert args.num_epochs >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')

    return args


"""main"""


def main():
    # parse arguments
    args = parse_args()
    if args is None:
        exit()

    if args.gpu_mode and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --gpu_mode=False")

    # model
    if args.model_name == 'TransENetModel':
        net = TransENetModel(args)
    else:
        raise Exception("[!] There is no option for " + args.model_name)

    # train
    net.train()

    # test
    #net.mfeNew_validate(epoch=100, modelpath='./Result/TransENetModel/model/generator_param.pkl')
    net.mfeNew_validateByClass(100, save_img=True, modelpath="./Result/TransENetModel/model/generator_param.pkl")

    # # 单张图片推理
    # #UCM_input_img scale_factor=4 crop_size=test_crop_size=hr_height=hr_width=128
    #net.dm_test_single(img_fn="../dataset/UCM_128_512/lr_128/", modelpath="../checkpoint/transenet_x4_generator_param.pkl")
    

if __name__ == '__main__':
    main()
