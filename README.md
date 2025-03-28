# Mamba-Sea
The core part of Mamba-Sea's code is in [Network1](https://github.com/orange-322/SSMDG/blob/main/model/vmunet.py) and [Network2](https://github.com/orange-322/SSMDG/blob/main/model/vmamba.py).By using these two files as a backbone, you can easily incorporate our code into your tasks.

The complete part of our code will be released after the paper is accepted.

## Prepare Datasets
- './dataset/fundus/' or './dataset/ProstateSlice/' or './dataset/skin/'

   -Domain1
    - train
      - images
        - .png
      - masks
        - .png
    - test
      - images
        - .png
      - masks
        - .png
  ......    


## Pre_trained Weights

The weights of the pre-trained VMamba could be downloaded [Google Drive](https://drive.google.com/file/d/16R-zLOYFSKE6mFdHaiPAjc2QwyBHy5SL/view?usp=drive_link). After that, the pre-trained weights should be stored in './pretrained_weights/'.

Thanks a lot for the valuable work of [VM-UNet](https://github.com/JCruan519/VM-UNet) and [VMamba](https://github.com/MzeroMiko/VMamba).

## Dataset
#### Fundus
Download dataset [Fundus](https://drive.google.com/file/d/1p33nsWQaiZMAgsruDoJLyatoq5XAH-TH/view) (Provided by [DoFE](https://github.com/emma-sjwang/Dofe)).
#### Prostate
Download dataset [Prostate](https://drive.google.com/file/d/1sx2FpNySQNjU6_zBa4DPnb9RAmesN0P6/view?usp=sharing) (Originally Provided by [SAML](https://liuquande.github.io/SAML/) and [RAM-DSIR](https://github.com/zzzqzhou/RAM-DSIR)).
#### Skin
Download dataset [ISIC2018](https://challenge.isic-archive.com/data/#2018) and [PH2](https://www.fc.up.pt/addi/ph2%20database.html). (Following [ESP-MedSAM](https://github.com/xq141839/ESP-MedSAM)).
