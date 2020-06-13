#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 15:35:25 2018

@author: haskig
"""

import keras
import matplotlib.pyplot as plt
from keras.models import Model
from keras.models import load_model
import os
from os import path
import SimpleITK as sitk
import numpy as np
import volume_data_generator as vdg
import mhd_utils as mu
from PyQt5 import QtGui, QtWidgets


def get_paths(n):
    data_folder = '/home/data/uronav_data'
    caseFolder = 'Case{:04d}'.format(n)
    path_base = path.join(data_folder, caseFolder)
    path_MR = path.join(path_base, 'MRVol_adjusted.mhd')
    path_US = path.join(path_base, 'USVol.mhd')
    return path_US, path_MR

data_folder = '/home/data/uronav_data'
vdg_inst = vdg.VolumeDataGenerator(data_folder, (71,750), 
                                    max_registration_error=20)
TRUS_path, MR_path = get_paths(1)
shape= (96,96,32)
MR, TRUS, error_trans, parameters = vdg_inst.create_sample(case_index=1, sample_dim=shape, mat_trans=None)

MR_arr = sitk.GetArrayFromImage(MR)
TRUS_arr = sitk.GetArrayFromImage(TRUS)

folder_model = '/home/haskig/tmp'

fn_model, desc = QtWidgets.QFileDialog.getOpenFileName(self,
                                                            'Load deep network', 
                                                            folder_model, 
                                                            "HDF5 files (*.h5)")


x = np.linspace(1,10,10)
#print(x,x[:-1])

