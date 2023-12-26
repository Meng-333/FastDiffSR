# **FastDiffSR**

**A fast denoising Diffusion model with conditional guidance and fast  sampling for remote sensing image super-resolution**

  - Fanen Meng,  Haoyu Jing, Laifu Zhang, Sensen Wu, Tian Feng, Renyi Liu, Zhenhong Du
  - Neurocomputing, Volume 479, pp 47-59
  - https://doi.org/10.1016/j.neucom.2022.01.029

## Folder Structure

Our folder structure is as follows:

```
├── FastDiffSR (code)
│   ├── config
│   │   ├── sr_fastdiffsr_train_64_256.json
│   │   ├── sr_fastdiffsr_train_32_256.json
│   │   ├── sr_fastdiffsr_test_64_256.json
│   │   ├── sr_fastdiffsr_test_32_256.json
│   ├── core
│   ├── data
│   ├── dataset
│   │   ├── Train_64_256
│   │   |   ├── hr_256
│   │   |   ├── lr_64
│   │   |   ├── sr_64_256
│   │   ├── Train_32_256 (the same as above)
│   │   ├── val_64_256   (the same as above)
│   │   ├── val_32_256   (the same as above)
│   │   ├── Test_Potsdam_64_256 (the same as above)
│   │   ├── Test_Potsdam_32_256 (the same as above)
│   │   ├── Test_Toronto_64_256 (the same as above)
│   │   ├── Test_Toronto_32_256 (the same as above)
│   ├── deeplabv3+ (semantic segmentation)
│   ├── model        (DDPM,DMDC,TESR,GDP_x0,FastDiffSR model)
│   │   ├── ddpm_modules
│   │   ├── sr3_modules
│   │   ├── tesr_modules
│   │   ├── gdp_modules
│   │   ├── fastdiffsr_modules
│   ├── MSI_SR_model (SwinIR,HSENet,TransENet,NDSRGAN,HAT model)
│   ├── sr_mfe.py
```

## Introduction

- FastDiffSR (diffusion model architecture): This project is based on [[sr3]](https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement)

  - Contains ten super-resolution models: ['SwinIR', 'HSENet', 'TransENet', 'NDSRGAN', 'HAT', 'DDPM', 'DMDC', 'TESR', 'GDP_x0', '**FastDiffSR**']
  - MSI_SR_model (traditional generative model architecture): This project is based on [[sradsgan]](https://github.com/Meng-333/SRADSGAN) 
  - deeplabv3+: Semantic segmentation experiments for super-resolution results, this project is based on [[deeplabx3+]](https://github.com/bubbliiiing/deeplabv3-plus-pytorch)


## Environment Installation

The HAT model uses python 3.8, pytorch 1.9, tensorflow-gpu 2.1.0, and the environment of other models is in requirements.txt

```bash
pip install -r requirements.txt
```

## Dataset Preparation

We used two datasets to train our model. After secondary processing, we obtained a total of about 22,854 images of 256*256 size.  

- Train
  
  - ["Potsdam", "Toronto"]
  
- Test
  
- ["Test_Potsdam", "Test_Toronto"]
  
- Link:   https://drive.google.com/drive/folders/1W_ZWHp8BhoeLdEKLF3eAsZTneaVJVgLy?usp=sharing 

  ```bash
  cd FastDiffSR/dataset
  python prepare_data_mfe_dm.py   # Generate corresponding hr, lr, sr folders based on Train and Test data
  ```
  

## Train & Evaluate
1. Prepare environment, datasets and code.
2. Run training / evaluation code. The code is for training on 1 GPU.

```bash
# FastDiffSR
cd FastDiffSR 
python sr_mfe.py -p train -c config/sr_fastdiffsr_train_64_256.json   # train x4
python sr_mfe.py -p val -c config/sr_fastdiffsr_test_64_256.json      # test  x4
python sr_mfe.py -p train -c config/sr_fastdiffsr_train_32_256.json   # train x8
python sr_mfe.py -p val -c config/sr_fastdiffsr_test_32_256.json      # test  x8
---------------------------------------------------------------
# DDPM,DMDC,TESR,GDP_x0 (the same as above)
# SwinIR,HSENet,TransENet,NDSRGAN,HAT:
cd MSI_SR_model
python main_swinir.py
python main_hsenet.py
python main_transenet.py
python main_ndsrgan.py
python main_hat.py
---------------------------------------------------------------
net.train()                       # train
net.mfeNew_validate()             # test
net.mfeNew_validateByClass()      # classes test
---------------------------------------------------------------
# deeplabv3+
cd deeplabv3+ 
python train.py          # train
python get_moiu.py       # get metric results of semantic segmentation
python predict.py        # get image label results of semantic segmentation

```

## Results

### 1. Comparisons With The State-of-the-Art Methods



![](D:\桌面\img\srcdm\5.1\5.1_Potsdam_x4x8.png)

**Fig. 8.** ×4 and ×8 visual comparisons with various SR methods on Potsdam dataset.

![](D:\桌面\img\srcdm\5.2\5.2_Toronto_x4x8.png)

**Fig. 9.** ×4 and ×8 visual comparisons with various SR methods on Toronto dataset

### 2. Semantic Segmentation Evaluation of SR Results

![](D:\桌面\img\srcdm\5.3\5.3_se_x4x8.png)

**Fig. 10.** Visualization results of comparison methods on Potsdam dataset. (a) ×4 factor. (b) ×8 factor.



### 3. Evaluation on Real Multispectral Data

![](D:\桌面\img\srcdm\5.4\Vaihingen_image1_5.4.png)

**Fig. 11.** 4-time super-resolution results of different methods on Vaihingen remote sensing image.

![](D:\桌面\img\srcdm\5.4\Vaihingen_image2_5.4.png)

**Fig. 12.** 8-time super-resolution results of different methods on Vaihingen remote sensing image.

## Citation

If our code helps your research or work, please consider citing our paper. 

```bib
Liu, Z.; Chen, X.; Zhou, S.; Yu, H.; Guo, J.; Liu, Y. DUPnet: Water Body Segmentation with Dense Block and Multi-Scale Spatial Pyramid Pooling for Remote Sensing Images. Remote Sens. 2022, 14, 5567.
```
