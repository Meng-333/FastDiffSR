import argparse
import os
from pytorch_fid import fid_score

def calc_fid(paths, batch_size=1, device="cuda", dims=2048):
    return fid_score.calculate_fid_given_paths(paths, batch_size, device, dims)

rslt_path = "/data/mfe/FastDiffSR/FID/FastDiffSR_potsdam_x4"
#rslt_path = "/data/mfe/FastDiffSR/FID/FastDiffSR_potsdam_x8"
hr_path = "/data/mfe/FastDiffSR/MSI_SR_model_4/dataset/Test/Potsdam"
#rslt_path = "/data/mfe/FastDiffSR/FID/FastDiffSR_toronto_x4"
#rslt_path = "/data/mfe/FastDiffSR/FID/FastDiffSR_toronto_x8"
#hr_path = "/data/mfe/FastDiffSR/MSI_SR_model_4/dataset/Test/Toronto"

paths = [rslt_path, hr_path]
fid_score = calc_fid(paths)
print("- SR_FID : {:.3f}".format(fid_score))
print16 = "- SR_FID : {:.3f}".format(fid_score)



