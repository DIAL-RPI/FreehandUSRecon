

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 17:54:53 2018

@author: haskig
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: yan

Check registration quality by varying translation or rotation.
Compare with MI in the same time.

"""
# %%

from utils import registration_reader as rr
from utils import volume_data_generator as vdg
from utils import volume_resampler_3d as vr3d

import glob
from keras.models import load_model
from matplotlib import pyplot as plt
import numpy as np
from os import path
import SimpleITK as sitk
from resizeimage import resizeimage
from PIL import Image
import random
import scipy
from utils import volume_resampler_3d as vr3D
from stl import mesh
from keras import backend as K


# %% input image dimensions

img_rows, img_cols = 96, 96
depth = 32
# 
img_channels = 2
batch_size = 64

tmp_folder = '/home/haskig/tmp'

# %%
a=0.1

if not 'deep_network' in globals():
    print('Loading deep network...')
    
    #fn_model = 'trained_binary_model_3d.h5'
    """
    fn_model = 'trained_3d_regression_20170627_refined.h5'
    folder_model = '/home/data/models'
    fn_full = path.join(folder_model, fn_model)
    """
    fn_model = '/zion/common/experimental_results/haskins_mrus_reg/trained_3d_regression_new_data.h5'
    deep_network = load_model(fn_model)
    #fn_model = '/home/haskig/tmp/trained_3d_regression_OG_smooth_stdev.h5'
    #deep_network = load_model(fn_model, custom_objects={'mse_var_reg':mse_var_reg})
    
    
    print('Deep network loaded from <{}>'.format(fn_model))


# %%

data_folder = '/home/data/uronav_data'
"""
if not 'vdg_train' in globals():
    vdg_train = vdg.VolumeDataGenerator(data_folder, (1,500))
    print('{} cases for using'.format(vdg_train.get_num_cases()))
"""
vdg_train = vdg.VolumeDataGenerator(data_folder, (71,750))
print('{} cases for using'.format(vdg_train.get_num_cases()))
#trainGen = vdg_train.generate_batch_with_parameters(batch_size=batch_size, shape=(img_cols,img_rows,depth))


# %% Generate samples and check predict values

case_idx = 1
case_folder = 'Case{:04d}'.format(case_idx)
full_case_path = path.join(data_folder, case_folder)

fn_stl = path.join(full_case_path, 'segmentationrtss.uronav.stl')
segMesh = mesh.Mesh.from_file(fn_stl)

folder = '/home/haskig/data/uronav_data'
"""
US_mat_path = path.join(folder, 'Case{:04}/SSC_US'.format(case_idx))
MR_mat_path = path.join(folder, 'Case{:04}/SSC_MR'.format(case_idx))
SSC_moving = scipy.io.loadmat(US_mat_path)['US_SSC']
SSC_fixed = scipy.io.loadmat(MR_mat_path)['MR_SSC']
"""

def get_array_from_itk_matrix(itk_mat):
    mat = np.reshape(np.asarray(itk_mat), (3,3))
    return mat



fns = glob.glob(path.join(full_case_path, '*.txt'))

if len(fns) < 1:
    print('No registration file found!')

fn_gt = fns[0]
for fn_registration in fns:
    if 'refined' in fn_registration:
        fn_gt = fn_registration

trans_gt = rr.load_registration(fn_gt)
print(trans_gt)

R = sitk.ImageRegistrationMethod()
R.SetMetricAsJointHistogramMutualInformation()

scores = []
mis = []
var_range = np.arange(-20, 20, 0.5)
n = 1
e = 0
for x in var_range:
    trans = np.copy(trans_gt)
    trans0 = np.copy(trans_gt)
    x0 = x
    score = 0
    for j in range(n):
        if j == 0:
            trans0[2,3] = trans_gt[2,3] + x
        
            sample0 = vdg_train.create_sample(case_idx, (img_cols,img_rows,depth), trans0)
            sampledFixed0 = sample0[0]
            sampledMoving0 = sample0[1]
        x += random.uniform(-e,e) 
        trans[2,3] = trans_gt[2,3] + x
        
        sample = vdg_train.create_sample(case_idx, (img_cols,img_rows,depth), trans)
        sampledFixed = sample[0]
        sampledMoving = sample[1]

        x = sitk.GetArrayFromImage(sampledFixed)
        y = sitk.GetArrayFromImage(sampledMoving)
        #pos_neg = sample[2]
        error_trans = sample[2]
        (angleX, angleY, angleZ, tX, tY, tZ) = sample[3]
        
        sample4D = np.zeros((1, 32, 96, 96, 2), dtype=np.ubyte)
        #print(sample4D.shape)
        sample4D[0, :,:,:, 0] = sitk.GetArrayFromImage(sampledFixed)
        sample4D[0, :,:,:, 1] = sitk.GetArrayFromImage(sampledMoving)
        
        
        prediction = deep_network.predict(sample4D)
        score_dl = prediction[0,0]
        score += score_dl
        x=x0
    score /= n
    scores.append(score)
    """
    SSD = 0
    trans[2,3] = trans_gt[2,3] + x
    for i in range(SSC_fixed.shape[3]):

        resampler3D = vr3D.VolumeResampler(sitk.GetImageFromArray(SSC_fixed[:,:,:,i]), segMesh, 
                                sitk.GetImageFromArray(SSC_moving[:,:,:,i]),
                                trans)
        resampler3D.set_transform(trans)
        sampledFixed, sampledMoving = resampler3D.resample(96, 96, 32)
        fixed_img = sitk.GetArrayFromImage(sampledFixed)
        moving_img = sitk.GetArrayFromImage(sampledMoving)
        diff = np.subtract(fixed_img, moving_img)
        sq_diff = np.square(diff)
        SSD += np.sum(sq_diff)
    SSC = SSD
    """

    score_mi = R.MetricEvaluate(sampledFixed0, sampledMoving0)
    
    mis.append(score_mi)
    
    print('DL: %.4g <--> MI: %.4g' % (score, score_mi))


# %%

fig, ax1 = plt.subplots()

#num_pts = len(scores)
ax1.plot(var_range, scores, c='b', label='DL')
ax1.set_title('Translation along Z-axis', fontsize=14)
ax1.set_xlabel('Translation along Z axis (mm)'.format(n,e), fontsize=14)
# Make the y-axis label, ticks and tick labels match the line color.
ax1.set_ylabel('CNN score (mm)', color='b', fontsize=14)
ax1.tick_params('y', colors='b')
ax1.legend(loc='lower left',prop={'size': 11})

ax2 = ax1.twinx()
ax2.plot(var_range, -np.asarray(mis), c='r', label='MI')
ax2.set_ylabel('Mutual Information', color='r', fontsize=14)
ax2.tick_params('y', colors='r')

ax2.legend(loc="lower right",prop={'size': 11})

fig.tight_layout()

plt.savefig('/home/haskig/Pictures/MIvLM_trans_bad_case_56.pdf', dpi=600, format='pdf')


# %% Rotations

#import transformations as tfms

scores = []
mis = []
var_range = np.arange(-20, 20, 0.5)
n = 1
e = 0
for x in var_range:
    x0 = x
    trans = np.copy(trans_gt)
    trans0 = np.copy(trans_gt)
    score = 0
    #trans[2,3] = trans_gt[2,3] + x
    
    #mat_rot = tfms.rotation_matrix(x/180.0 * np.pi, (0,0,1))
    #trans = mat_rot.dot(trans)
    rot0, t0 = vdg_train.create_transform(x0, x0, x0, 0, 0, 0, trans_gt)
    trans0[:3,:3] = rot0
    trans0[:3, 3] = t0 + trans_gt[:3,3]
    
    sample0 = vdg_train.create_sample(case_idx, (img_cols,img_rows,depth), trans0)
    sampledFixed0 = sample0[0]
    sampledMoving0 = sample0[1]
    #pos_neg = sample[2]
    error_trans0 = sample0[2]
    (angleX, angleY, angleZ, tX, tY, tZ) = sample0[3]
    for j in range(n):
        x += random.uniform(-e,e)
        rot, t = vdg_train.create_transform(x, x, x, 0, 0, 0, trans_gt)
        trans[:3,:3] = rot
        trans[:3, 3] = t + trans_gt[:3,3]
        
        sample = vdg_train.create_sample(case_idx, (img_cols,img_rows,depth), trans)
        sampledFixed = sample[0]
        sampledMoving = sample[1]
        #pos_neg = sample[2]
        error_trans0 = sample[2]
        (angleX, angleY, angleZ, tX, tY, tZ) = sample[3]
        sample4D = np.zeros((1, 32, 96, 96, 2), dtype=np.ubyte)
        #print(sample4D.shape)
        sample4D[0, :,:,:, 0] = sitk.GetArrayFromImage(sampledFixed)
        sample4D[0, :,:,:, 1] = sitk.GetArrayFromImage(sampledMoving)
        
        prediction = deep_network.predict(sample4D)
        score_dl = prediction[0,0]
        score += score_dl
    score /= n
    scores.append(score)
 
    score_mi = R.MetricEvaluate(sampledFixed0, sampledMoving0)
    
    mis.append(score_mi)
    
    print('DL: %.4g <--> MI: %.4g' % (score_dl, score_mi))



# %

fig, ax1 = plt.subplots()

#num_pts = len(scores)
ax1.plot(var_range, scores, c='b', label='DL')
ax1.set_title('Rotation around all axes simultaneously', fontsize=14)
ax1.set_xlabel('Rotation around all axes (degree)'.format(n,e), fontsize=14)
# Make the y-axis label, ticks and tick labels match the line color.
ax1.set_ylabel('CNN score (mm)', color='b', fontsize=14)
ax1.tick_params('y', colors='b')
ax1.legend(loc='lower left',prop={'size': 11})

ax2 = ax1.twinx()
ax2.plot(var_range, -np.asarray(mis), c='r', label='MI')
ax2.set_ylabel('Mutual Information', color='r', fontsize=14)
ax2.tick_params('y', colors='r')

ax2.legend(loc="lower right",prop={'size': 11})

fig.tight_layout()

plt.savefig('/home/haskig/Pictures/MIvLM_rot_bad_case_56.pdf', dpi=600, format='pdf')

"""
image = Image.open('/home/haskig/Pictures/rotateAll_case10.png')
cover = resizeimage.resize_cover(image, [300,200])
cover.save('/home/haskig/Pictures/rotateAll_case30_resized.png', image.format)
"""


x = [7.84411,7.59224,7.34227,6.69355,6.69355,6.69355,6.69355,6.69355,6.69355,6.69355,6.69355,6.69355,6.69355,6.69355,6.69355,6.69355]
plt.plot(x)
plt.ylabel('CNN Predicted TRE')
plt.xlabel('Generation Number')
plt.title('Differential Evolution plot: bad case (Initial TRE = 16mm)')
plt.savefig('/home/haskig/test_cases/Case0002/diffevo_generation_plot.pdf', dpi=600, format='pdf')