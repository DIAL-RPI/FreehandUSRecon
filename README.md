# Sensorless Freehand 3D Ultrasound Reconstruction via Deep Contextual Learning
(This repository is under active update.)

## Introduction
In this paper, we propose a deep contextual learning network (DCL-Net), which can efficiently exploit the image feature relationship between US frames and reconstruct 3D US volumes without any tracking device. The proposed DCL-Net utilizes 3D convolutions over a US video segment for feature extraction. An embedded self-attention module makes the network focus on the speckle-rich areas for better spatial movement prediction. We also propose a novel case-wise correlation loss to stabilize the training process for improved accuracy. Highly promising results have been obtained by using the developed method. We will test the method further on US videos with varieties of scanning protocols in future work. For more details, please refer to our pre-print version available on [arXiv](https://arxiv.org/abs/2006.07694).

## Environment
- Set up your environment by anaconda, (**python3.7, torch 1.5.0+cu92**)

### Testing
'''
CUDA_VISIBLE_DEVICES=0 python test_network.py
'''

One of our pretrained model is availabel in GoogleDrive: https://drive.google.com/drive/folders/1fQTHekCs7et95x60WYEG7W5lRM8b4PGx?usp=sharing
