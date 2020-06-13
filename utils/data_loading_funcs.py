# -*- coding: utf-8 -*-
"""
Fuse two images using pseudo color to encode one image and superimposing on the other.
"""

# %%

import numpy as np
import math
import SimpleITK as sitk
from utils import mhd_utils as mu
from utils import transformations as tfms
from utils import adjust_window_level as adwl
from utils import registration_reader as rr
from os import path
import nibabel as nib
from utils import CheckData
import random
import time
import cv2
import imageio
import matplotlib.pyplot as plt
from networks import generators as gens
from networks import evaluators as evas
import torch

def array_normalize(input_array):
    max_value = np.max(input_array)
    min_value = np.min(input_array)
    # print('max {}, min {}'.format(max_value, min_value))
    k = 255 / (max_value - min_value)
    min_array = np.ones_like(input_array) * min_value
    normalized = k * (input_array - min_array)
    return normalized

def fuse_images(img_ref, img_folat, alpha=0.4):
    """
    """
    mask = (img_folat > 5).astype(np.float32)
    # print(alpha)
    mask[mask > 0.5] = alpha
    mask_comp = 1.0 - mask

    img_color = cv2.applyColorMap(img_folat, cv2.COLORMAP_JET)
    # print(img_color.shape)

    dst = np.zeros((img_folat.shape[0], img_folat.shape[1], 3), dtype=np.uint8)

    for i in range(3):
        dst[:, :, i] = (img_ref * mask_comp + img_color[:, :, i] * mask).astype(np.uint8)

    dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)

    return dst

def scale_volume(input_volume, upper_bound=255, lower_bound=0):
    max_value = np.max(input_volume)
    min_value = np.min(input_volume)

    k = (upper_bound - lower_bound) / (max_value - min_value)
    scaled_volume = k * (input_volume - min_value) + lower_bound
    # print('min of scaled {}'.format(np.min(scaled_volume)))
    # print('max of scaled {}'.format(np.max(scaled_volume)))
    return scaled_volume

def estimate_final_transform(base_mat, move_motion, moving_center_mm):
    base_center_mm = coord_rigid_transform(point=moving_center_mm, mat=base_mat)

    test_origin_trans = tfms.translation_matrix(-base_center_mm)
    recon_origin_mat = test_origin_trans.dot(base_mat)

    rot_x = move_motion[3] * np.pi / 180
    rot_y = move_motion[4] * np.pi / 180
    rot_z = move_motion[5] * np.pi / 180
    R_back = tfms.euler_matrix(rot_z, rot_y, rot_x, 'rzyx')
    recon_rotate_mat = R_back.dot(recon_origin_mat)

    recon_back_trans = tfms.translation_matrix(base_center_mm + move_motion[:3])
    recon_mat = recon_back_trans.dot(recon_rotate_mat)
    return recon_mat

# Angles in radian version
def decompose_matrix(trans_matrix):
    eus = tfms.euler_from_matrix(trans_matrix[:3, :3], axes='sxyz')
    # trans = trans_matrix[:3, 3]
    params = np.asarray([trans_matrix[0, 3], trans_matrix[1, 3], trans_matrix[2, 3],
                         eus[0], eus[1], eus[2]])
    return params

# def construct_matrix(params, initial_transform=None, use_initial=False):
#     if use_initial == True:
#         initial_params = decompose_matrix(initial_transform)
#         params += initial_params

#     mat = tfms.euler_matrix(params[3], params[4], params[5], 'sxyz')
#     mat[:3, 3] = np.asarray([params[0], params[1], params[2]])

#     return mat

def construct_matrix(params, initial_transform=None):
    '''
    '''
    mat = tfms.euler_matrix(params[3], params[4], params[5], 'sxyz')
    mat[:3, 3] = np.asarray([params[0], params[1], params[2]])

    if not initial_transform is None:
        mat = mat.dot(initial_transform)

    return mat

# Angles in degree version
def decompose_matrix_degree(trans_matrix):
    eus = tfms.euler_from_matrix(trans_matrix[:3, :3])
    eus = np.asarray(eus, dtype=np.float) / np.pi * 180.0
    params = np.asarray([trans_matrix[0, 3],
                        trans_matrix[1, 3],
                        trans_matrix[2, 3],
                        eus[0], eus[1], eus[2]])
    return params

def construct_matrix_degree(params, initial_transform=None):
    if not params is np.array:
        params = np.asarray(params, dtype=np.float)

    radians = params[3:] / 180.0 * np.pi
    mat = tfms.euler_matrix(radians[0], radians[1], radians[2], 'sxyz')
    mat[:3, 3] = np.asarray([params[0], params[1], params[2]])

    if not initial_transform is None:
        mat = mat.dot(initial_transform)

    return mat

def get_diff_params_as_label(init_mat, target_mat):
    moving_mat = init_mat.dot(np.linalg.inv(target_mat))
    eulers = np.asarray(tfms.euler_from_matrix(moving_mat[:3, :3], axes='sxyz')) / np.pi * 180
    params_rand = np.concatenate((moving_mat[:3, 3], eulers), axis=0)
    return params_rand
# %%

def decompose_matrix_old(trans_matrix):
    # print('trans_matrix\n{}'.format(trans_matrix))
    tX = trans_matrix[0][3]
    tY = trans_matrix[1][3]
    tZ = trans_matrix[2][3]
    # print('tX {}, tY {}, tZ {}'.format(tX, tY, tZ))

    ''' Use online OpenCV codes '''
    ''' radius to degrees '''
    ''' The output angles are degrees! '''
    angleX, angleY, angleZ = rotationMatrixToEulerAngles(trans_matrix[:3, :3])
    angleX = angleX * 180.0 / np.pi
    angleY = angleY * 180.0 / np.pi
    angleZ = angleZ * 180.0 / np.pi

    return np.asarray([tX, tY, tZ, angleX, angleY, angleZ])

def get_array_from_itk_matrix(itk_mat):
    mat = np.reshape(np.asarray(itk_mat), (3, 3))
    return mat

def rotation_matrix(angle, direction='x'):
    rot_mat = np.identity(3)
    sinX = math.sin(angle)
    cosX = math.cos(angle)
    if direction == 'x':
        rot_mat[1][1] = cosX
        rot_mat[1][2] = -sinX
        rot_mat[2][1] = sinX
        rot_mat[2][2] = cosX
    elif direction == 'y':
        rot_mat[0][0] = cosX
        rot_mat[0][2] = sinX
        rot_mat[2][0] = -sinX
        rot_mat[2][2] = cosX
    else:
        rot_mat[0][0] = cosX
        rot_mat[0][1] = -sinX
        rot_mat[1][0] = sinX
        rot_mat[1][1] = cosX
    # print('rot_mat\n{}'.format(rot_mat))
    # time.sleep(30)
    return rot_mat


# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R):
    # assert (isRotationMatrix(R))

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    # return np.array([x, y, z])
    return x, y, z

def construct_matrix_old(tX, tY, tZ, angleX, angleY, angleZ, initial_transform, use_initial=True):
    t_all = np.asarray((tX, tY, tZ))

    # Get the transform
    rotX = sitk.VersorTransform((1, 0, 0), angleX / 180.0 * np.pi)
    matX = get_array_from_itk_matrix(rotX.GetMatrix())
    # print('matX\n{}'.format(matX))
    # rotation_matrix(angleX)
    #
    rotY = sitk.VersorTransform((0, 1, 0), angleY / 180.0 * np.pi)
    matY = get_array_from_itk_matrix(rotY.GetMatrix())
    #
    rotZ = sitk.VersorTransform((0, 0, 1), angleZ / 180.0 * np.pi)
    matZ = get_array_from_itk_matrix(rotZ.GetMatrix())

    # Apply all the rotations
    #
    #Bug fixed on Aug-09-2019:
    # Fixed the order of multiplication to be consistent with
    # euler_matrix(ai, aj, ak, axes='sxyz')
    # in transformations.py
    # previously X.dot(Y.dot(Z))
    # now Z.dot(Y.dot(X))
    if use_initial == True:
        mat_all = initial_transform[:3, :3].dot(matZ.dot(matY.dot(matX)))
        translation = initial_transform[:3, 3]
    else:
        mat_all = np.identity(3).dot(matZ.dot(matY.dot(matX)))
        translation = np.zeros(3)

    itkTrans_us2mr = sitk.AffineTransform(3)
    itkTrans_us2mr.SetMatrix(np.reshape(mat_all, (9,)))

    itkTrans_us2mr.SetTranslation(translation + t_all)
    # print(translation)

    arrTrans_us2mr = np.identity(4)
    arrTrans_us2mr[:3, :3] = mat_all
    arrTrans_us2mr[:3, 3] = translation + t_all
    # print('reconstructed matrix\n{}'.format(arrTrans_us2mr))
    return arrTrans_us2mr

''' Input groundtruth transformation, initial transformation and interpolation ratio '''
''' Output the interpolated transformation matrix'''
''' Ratio=0, use the basic registration; Ratio=1, use the groundtruth registration'''
# def transform_interpolation(gt_reg, base_reg, ratio=0.5):
#     gt_tX, gt_tY, gt_tZ, gt_angleX, gt_angleY, gt_angleZ = decomposite_matrix(trans_matrix=gt_reg)
#     bs_tX, bs_tY, bs_tZ, bs_angleX, bs_angleY, bs_angleZ = decomposite_matrix(trans_matrix=base_reg)

#     gt_params = np.asarray([gt_tX, gt_tY, gt_tZ, gt_angleX, gt_angleY, gt_angleZ])
#     bs_params = np.asarray([bs_tX, bs_tY, bs_tZ, bs_angleX, bs_angleY, bs_angleZ])

#     md_params = (gt_params - bs_params) * ratio + bs_params

#     # print('gt_array {}'.format(gt_array))
#     # print('bs_array {}'.format(bs_array))
#     # print('md_array {}'.format(md_array))

#     md_mat = construct_matrix(tX=md_params[0],
#                               tY=md_params[1],
#                               tZ=md_params[2],
#                               angleX=md_params[3],
#                               angleY=md_params[4],
#                               angleZ=md_params[5],
#                               initial_transform=base_reg,
#                               use_initial=False)

#     # print('gt_reg\n{}'.format(gt_reg))
#     # print('md_reg\n{}'.format(md_reg))
#     # print('bs_reg\n{}'.format(base_reg))
#     # gt_array = gt_params - md_params

#     return bs_params, md_params, gt_params, md_mat


def interpolate_transforms(mat_0, mat_1, ratio=0.5):
    ''' Create a new transform by interpolating between two transforms
        with the given ratio.
    '''

    gt_params = decompose_matrix_degree(mat_0)
    bs_params = decompose_matrix_degree(mat_1)

    md_params = gt_params * ratio + (1.0 - ratio) * bs_params

    md_mat = construct_matrix_degree(md_params)

    return md_mat


def load_mhd_as_sitkImage(fn_mhd, return_header=False):
    """
    """
    rawImg, header = mu.load_raw_data_with_mhd(fn_mhd)
    #
    img = sitk.GetImageFromArray(rawImg)
    img.SetOrigin(header['Offset'])
    img.SetSpacing(header['ElementSpacing'])

    if return_header:
        return img, header
    else:
        return img

def load_gt_registration(folder_path):
    fn_reg = 'coreg.txt'
    fn_reg_refined = 'coreg_refined.txt'

    # By default, load the refined registration
    fn_reg_full = path.join(folder_path, fn_reg_refined)

    if not path.isfile(fn_reg_full):
        fn_reg_full = path.join(folder_path, fn_reg)

    # print('loading {}'.format(fn_reg_full))
    gt_reg = np.loadtxt(fn_reg_full)
    return gt_reg

def coord_rigid_transform(point, mat):
    point = np.append(point, [1])
    trans_pt = np.dot(mat, point)
    trans_pt = trans_pt / trans_pt[3]

    return trans_pt[:3]

def sample_random_point(center, spacing_new, radius_mm_range=(10, 20), random_type='gauss'):
    random_angle_radians = random.random() * 2 * np.pi
    if random_type == 'uniform':
        random_radius_mm = radius_mm_range[0] + random.random() * (radius_mm_range[1] - radius_mm_range[0])
    elif random_type == 'gauss':
        # mean = (radius_mm_range[0] + radius_mm_range[1]) / 2
        # std = (radius_mm_range[1] - radius_mm_range[0]) / 2
        random_radius_mm = np.random.normal(8, 2, 1)[0]
    else:
        print('<{}> is not supported, using uniform instead'.format(random_type))
        random_radius_mm = radius_mm_range[0] + random.random() * (radius_mm_range[1] - radius_mm_range[0])

    random_radius = int(random_radius_mm / spacing_new[0])

    x_coord = center[0] + random_radius * math.cos(random_angle_radians)
    y_coord = center[1] + random_radius * math.sin(random_angle_radians)
    coords = (x_coord, y_coord)
    return coords, random_radius_mm

def generate_random_transform_NIH_circle(gt_mat, mr_header, us_header, fixedImgSize, movingImgSize):
    mr_spacing = np.asarray(mr_header['ElementSpacing'])
    us_spacing = np.asarray(us_header['ElementSpacing'])

    fixedImgSize = np.asarray(fixedImgSize)
    spacing_new = fixedImgSize / np.array([512., 512., 512.]) * mr_spacing
    spacing_new[2] = spacing_new[0]

    # gt_reg = gt_mat

    ''' load MR segmentation '''
    # fn_stl = path.join(case_folder, 'segmentationrtss.uronav.stl')
    # segMesh = mesh.Mesh.from_file(fn_stl)
    # num_triangle = segMesh.points.shape[0]
    # markers = np.reshape(segMesh.points, (num_triangle * 3, 3))

    ''' Landmark calculation is correct! '''
    # avg_markers_mm = np.average(markers - mr_header['Offset'], axis=0)

    ''' Original center of US image'''
    moving_center_mm = np.asarray(movingImgSize) / 2 * us_spacing
    origin_center = moving_center_mm
    #
    # difference = avg_markers_mm - moving_center_mm
    # dist_mm = np.linalg.norm(avg_markers_mm - moving_center_mm)
    # mat_moves = difference
    # print(difference)
    # time.sleep(30)

    ''' Generate the mat that moves the us_center to seg_center'''
    # recenter_reg = load_func.construct_matrix_degree(
    #     params=np.asarray([mat_moves[0], mat_moves[1], mat_moves[2], 0, 0, 0]),
    #     initial_transform=None)
    # recenter_reg = NIH_reg
    # recenter_reg = gt_reg

    # resampler2D = vr2D.Resampler2D(fixedImg, movingImg, recenter_reg)
    # mr_array, us_array = resampler2D.resample(view='ax', loc=0.5)
    # fused_img = fuse_images(mr_array, us_array, alpha=0.4)

    ''' This is the center of segmentation mask'''
    # avg_markers_mm = avg_markers_mm / spacing_new
    # cv2.circle(fused_img, (int(avg_markers_mm[0]), int(avg_markers_mm[1])), 4, (255, 255, 0), -1)
    # cv2.circle(fused_img, (int(avg_markers_mm[0]), int(avg_markers_mm[1])), 40, (255, 255, 0), 1)

    ''' This is the center of ultrasound image'''
    ''' Transform the us_center coords to seg_center'''
    gt_center_mm = coord_rigid_transform(point=moving_center_mm, mat=gt_mat) / spacing_new

    coords, radius_mm = sample_random_point(center=gt_center_mm,
                                            spacing_new=spacing_new,
                                            radius_mm_range=(5, 15))

    params = np.asarray([int(coords[0]) - gt_center_mm[0],
                         int(coords[1]) - gt_center_mm[1],
                         0, 0, 0, 0]) * spacing_new[0]

    this_mat = construct_matrix_degree(params=params,
                                       initial_transform=gt_mat)
    this_center_mm = coord_rigid_transform(point=origin_center, mat=this_mat) / spacing_new[0]
    # print('radius = {:.4f}mm'.format(radius_mm))

    return this_mat, params

def convert_to_sitk_ubyte(img, header, cut_ratio=0.001):
    if header['ElementType'] == 'MET_CHAR':
        img_adjusted = img
    else:
        img_adjusted = adwl.autoAdjustWL(img, cut_ratio)
    img_itk = sitk.GetImageFromArray(img_adjusted)
    img_itk.SetOrigin(header['Offset'])
    img_itk.SetSpacing(header['ElementSpacing'])
    # print('Offset type {}'.format(type(header['Offset'][0])))
    # print('Offset {}'.format(header['Offset']))
    # print('ElementSpacing {}'.format(header['ElementSpacing']))
    # print('GetSize {}'.format(img_itk.GetSize()))
    # time.sleep(30)
    return img_itk

def load_volume2sitk_ubyte_test(img_path, cut_ratio=0.001):
    if img_path.endswith('.mhd'):
        print('Data type is mhd!')
        img_data, header = mu.load_raw_data_with_mhd(img_path)
        # img_data = img_data.astype(np.float32)
        print('mhd img_data shape {}'.format(img_data.shape))
        max_value = np.max(img_data)
        min_value = np.min(img_data)
        print('max {}, min {}'.format(max_value, min_value))
        # time.sleep(30)

        # for i in range(img_data.shape[0]):
        #     img_data[i] = img_data[i] / 255
        #     img_data[i] = np.clip(img_data[i], -2., 2.)
        #
        # file_name = '/zion/guoh9/projects/reg4nih/data_sample/Test01/segmentationrtss.uronav.voi'
        # coords_slice = CheckData.read_voi(file_name)
        # print(coords_slice)
        # unique_slice = np.unique(coords_slice[:, 2]).astype(np.int16)
        # print('unique slices {}'.format(unique_slice))

        # imgs = []
        # for slice_index in range(img_data.shape[0]):
        #     this_slice = img_data[slice_index, :, :]
        #     # imgs.append(this_slice)
        #     # cv2.imshow('this', this_slice)
        #     # cv2.waitKey(0)
        #     plt.imshow(this_slice)
        #     plt.savefig('imgs/slice{:02}.png'.format(slice_index))
        #     # cv2.i('imgs/slice{:02}.png'.format(slice_index), this_slice)
        #     # print('slice{:02} saved'.format(slice_index))
        # # imageio.mimsave('imgs/nii.gif', imgs)
        # print('finished')
        # time.sleep(30)

        # for slice_index in unique_slice:
        #     slice_pts = coords_slice[coords_slice[:, 2] == slice_index, :2]
        #     print('slice {}: {}'.format(slice_index, slice_pts.shape))
        #
        #     slice_img = img_data[slice_index, :, :]
        #
        #     # implot = plt.imshow(slice_img)
        #     # plt.scatter(slice_pts[:, 1], slice_pts[:, 0], c='r', linewidths=0.1)
        #     # plt.show()
        #     for point in slice_pts:
        #         point = point.astype(np.int16)
        #         slice_img = cv2.circle(slice_img, center=(point[0], point[1]),
        #                                radius=1, color=(255, 0, 0), thickness=1)
        #
        #     cv2.imshow('{}'.format(slice_index), slice_img)

        # time.sleep(30)

        # print('img_data shape {}'.format(img_data.shape))
        # cv2.imshow('center_slice', img_data[15, :, :])
        # cv2.waitKey(0)
        # time.sleep(30)
        # time.sleep(30)
        img_data = convert_to_sitk_ubyte(img_data, header, cut_ratio=cut_ratio)
        # sitk.Show(img_data, title='mhd image')
        # print(img_data)
        # time.sleep(30)
        return img_data
    elif img_path.endswith('.nii'):
        print('Data type is nifti!')
        data = nib.load(img_path)
        img_data = data.get_data()
        img_data = np.transpose(img_data, [2, 1, 0])
        print('nii img_data shape {}'.format(img_data.shape))
        max_value = np.max(img_data)
        min_value = np.min(img_data)
        print('max {}, min {}'.format(max_value, min_value))
        # time.sleep(30)

        # for i in range(img_data.shape[0]):
        #     img_data[i] = img_data[i] / 255
        #     img_data[i] = np.clip(img_data[i], -2., 2.)
        # print('nii img_data shape {}'.format(img_data.shape))

        # file_name = '/zion/guoh9/projects/reg4nih/data_sample/MRI_US_Reg_sample2/' \
        #             'Right mid anterior TZ lesion_2nd session.voi'
        file_name = '/zion/guoh9/projects/reg4nih/data_sample/MRI_US_Reg_sample2/' \
                    'wp.voi'
        # file_name = '/zion/guoh9/projects/reg4nih/data_sample/Test01/segmentationrtss.uronav.voi'
        coords_slice = CheckData.read_voi(file_name)
        print(coords_slice)
        unique_slice = np.unique(coords_slice[:, 2]).astype(np.int16)
        print('unique slices {}'.format(unique_slice))

        # imgs = []
        # for slice_index in unique_slice:
        #     slice_pts = coords_slice[coords_slice[:, 2] == slice_index, :2]
        #     print('slice {}: {}'.format(slice_index, slice_pts.shape))
        #
        #     slice_img = img_data[slice_index, :, :]
        #     slice_img = cv2.cvtColor(slice_img, cv2.COLOR_GRAY2BGR)
        #
        #     for point in slice_pts:
        #         point = point.astype(np.int16)
        #         slice_img = cv2.circle(slice_img, center=(point[0], point[1]),
        #                                radius=1, color=(255, 0, 0), thickness=1)
        #     print(np.max(slice_img), np.min(slice_img))
        #     # imgs.append(slice_img)
        #     # cv2.imshow('{}'.format(slice_index), slice_img)
        #     # cv2.waitKey(0)
        #     # plt.imshow(slice_img)
        #     plt.imsave('imgs/slice{:02}.png'.format(slice_index), slice_img)
        # # imageio.mimsave('imgs/seg.gif', imgs)
        # print('seg.gif saved')
        # cv2.imshow('center_slice', img_data[13, :, :])
        # cv2.waitKey(0)
        # time.sleep(30)

        img_header = data.header
        print('nii header\n{}'.format(img_header))
        # print('nii datatype: {}'.format(img_header['quatern_b']))
        print('nii qoffset_x: {}'.format(img_header['qoffset_x']))
        print('nii qoffset_y: {}'.format(img_header['qoffset_y']))
        print('nii qoffset_z: {}'.format(img_header['qoffset_z']))
        print('nii pixdim: {}'.format(img_header['pixdim']))

        nii_spacing = list(img_header['pixdim'][1:4].astype(np.double))
        nii_offset = [float(img_header['qoffset_x']),
                      float(img_header['qoffset_y']),
                      float(img_header['qoffset_z'])]
        nii_size = [int(img_header['dim'][1]),
                    int(img_header['dim'][2]),
                    int(img_header['dim'][0])]
        nii_size = tuple(i for i in nii_size)

        print('nii_spacing: {}'.format(nii_spacing))
        print('nii_offset: {}'.format(nii_offset))
        print('nii_size: {}'.format(nii_size))
        # time.sleep(30)
        print('datatype: {}'.format(img_header['datatype']))

        img_adjusted = adwl.autoAdjustWL(img_data, cut_ratio)
        img_itk = sitk.GetImageFromArray(img_adjusted)
        img_itk.SetOrigin(nii_offset)
        img_itk.SetSpacing(nii_spacing)
        # img_itk.SetSize(nii_size)
        print('nii to itk and sets successfully!')
        # time.sleep(30)
        return img_itk

def load_volume2sitk_ubyte(img_path, cut_ratio=0.001):
    if img_path.endswith('.mhd'):
        print('Data type is mhd!')
        img_data, header = mu.load_raw_data_with_mhd(img_path)
        img_data = convert_to_sitk_ubyte(img_data, header, cut_ratio=cut_ratio)
        return img_data
    elif img_path.endswith('.nii'):
        print('Data type is nifti!')
        data = nib.load(img_path)
        img_data = data.get_data()
        img_data = np.transpose(img_data, [2, 1, 0])

        img_header = data.header
        nii_spacing = list(img_header['pixdim'][1:4].astype(np.double))
        nii_offset = [float(img_header['qoffset_x']),
                      float(img_header['qoffset_y']),
                      float(img_header['qoffset_z'])]

        img_adjusted = adwl.autoAdjustWL(img_data, cut_ratio)
        img_itk = sitk.GetImageFromArray(img_adjusted)
        img_itk.SetOrigin(nii_offset)
        img_itk.SetSpacing(nii_spacing)
        return img_itk

def read_segMesh(file_path):
    if file_path.endswith('.voi'):
        print('voi file!')
    elif file_path.endswith('.stl'):
        print('stl file!')

def load_registration_mat(mat_path, fn_fixed):
    if mat_path.endswith('.xml'):
        this_mat = rr.load_registration_xml(mat_path)
        return this_mat

    elif mat_path.endswith('.txt'):
        this_mat = np.loadtxt(mat_path)
        if this_mat[3, 3] != 1:
            print('Doing automatic transform conversion...')
            print('before conversion was: {}'.format(this_mat))
            this_mat = rr.load_UroNav_registration(fn_reg_UroNav=mat_path, fn_mhd=fn_fixed)
            print('after conversion is: {}'.format(this_mat))
        return this_mat

    else:
        print('Registration format not supported!')

def scale_high_TRE(gt_mat, params_rand, scale_ratio):

    params_rand = params_rand / scale_ratio
    base_mat = construct_matrix_degree(params=params_rand,
                                       initial_transform=gt_mat)
    return base_mat, params_rand


def load_model_stages(init_mode, cardinality=16):
    model = gens.resnet101(sample_size=2, sample_duration=16, cardinality=cardinality)

    if init_mode == 'uniform_SRE2':
        model_path = 'pretrained_models/' \
                     '3d_best_Generator_1107-112210_uniform_SRE1.pth'
    elif init_mode == 'random_SRE2':
        model_path = 'pretrained_models/' \
                     '3d_best_Generator_1107-112020_random_SRE1.pth'
    elif init_mode == 'gauss_nih_SRE2':
        model_path = 'pretrained_models/' \
                     '3d_best_Generator_1107-111933_gauss_nih_SRE1.pth'
    else:
        print('<{}> not supported yet!'.format(init_mode))
        return
    model.load_state_dict(torch.load(model_path, map_location='cuda:0'))
    model.eval()
    print('{} loaded from <{}>!'.format(init_mode, model_path))
    return model




