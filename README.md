# Freehand 3D Ultrasound Volume Reconstruction
This repository contains the code for MICCAI 2020 paper, entitled [Sensorless Freehand 3D Ultrasound Reconstruction via Deep Contextual Learning](https://link.springer.com/chapter/10.1007/978-3-030-59716-0_44).

## Introduction
In this paper, we propose a deep contextual learning network (DCL-Net), which can efficiently exploit the image feature relationship between US frames and reconstruct 3D US volumes without any tracking device. The proposed DCL-Net utilizes 3D convolutions over a US video segment for feature extraction. An embedded self-attention module makes the network focus on the speckle-rich areas for better spatial movement prediction. We also propose a novel case-wise correlation loss to stabilize the training process for improved accuracy. Highly promising results have been obtained by using the developed method. We will test the method further on US videos with varieties of scanning protocols in future work. For more details, please refer to our pre-print version available on [arXiv](https://arxiv.org/abs/2006.07694).

## Environment
- Set up your environment by anaconda, (**python3.7, torch 1.5.0+cu92**)

## Testing
One of our pretrained model is availabel in [GoogleDrive](https://drive.google.com/drive/folders/1fQTHekCs7et95x60WYEG7W5lRM8b4PGx?usp=sharing). After downloading it into the "pretrained_model" folder, run the following command to test with the demo case:
```
CUDA_VISIBLE_DEVICES=0 python test_network.py
```

## Citation
If you find this work helpful to you, please cite our paper in your work:
```
@inproceedings{guo2020sensorless,
  title={Sensorless freehand 3D ultrasound reconstruction via deep contextual learning},
  author={Guo, Hengtao and Xu, Sheng and Wood, Bradford and Yan, Pingkun},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={463--472},
  year={2020},
  organization={Springer}
}
```

