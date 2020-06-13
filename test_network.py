import numpy as np
import time
import cv2
import copy
import os
import os.path as path
import imageio
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import argparse
from numpy.linalg import inv
import torch
from train_network import data_transform
import train_network
import tools


desc = 'Test reconstruction network'
parser = argparse.ArgumentParser(description=desc)
parser.add_argument('-d', '--device_no',
                    type=int,
                    choices=[0, 1, 2, 3, 4, 5, 6, 7],
                    help='GPU device number [0-7]',
                    default=0)
parser.add_argument('-avg', '--average_dof',
                    type=bool,
                    help='give the average bof within a sample',
                    default=False)

args = parser.parse_args()
device_no = args.device_no

train_ids = np.loadtxt('infos/train_ids.txt').astype(np.int64)
val_ids = np.loadtxt('infos/val_ids.txt').astype(np.int64)
test_ids = np.loadtxt('infos/test_ids.txt').astype(np.int64)

all_ids = np.concatenate((train_ids, val_ids, test_ids), axis=0)

mask_img = cv2.imread('data/US_mask.png', 0)

# frames_folder = '/home/guoh9/tmp/US_vid_frames'
# pos_folder = '/home/guoh9/tmp/US_vid_pos'

# frames_folder = '/zion/guoh9/US_recon/US_vid_frames'
# pos_folder = '/zion/guoh9/US_recon/US_vid_pos'

frames_folder = 'data/US_vid_frames'
pos_folder = 'data/US_vid_pos'
cali_folder = 'data/US_cali_mats'


def read_aurora(file_path):
    """
    Read the Aurora position file and formatly reorganize the shape
    :param file_path: path of Aurora position file
    :return: (frame_number * 9) matrix, each row is a positioning vector
    """
    file = open(file_path, 'r')
    lines = file.readlines()
    pos_np = []
    for line_index in range(1, len(lines) - 1):  # exclude the first line and last line
        line = lines[line_index]
        values = line.split()
        values_np = np.asarray(values[1:]).astype(np.float32)
        pos_np.append(values_np)
    pos_np = np.asarray(pos_np)
    return pos_np


def save_all_aurora_pos():
    """
    This function uses read_aurora function to convert Aurora.pos file into (N x 9) matrix
    Save such txt files for all 640 cases
    """
    check_folder = '/home/guoh9/tmp/US_vid_frames'
    project_folder = '/zion/common/data/uronav_data'
    dst_folder = '/home/guoh9/tmp/US_vid_pos'
    case_list = os.listdir(check_folder)
    case_list.sort()

    for case_index in range(len(case_list)):
        case_id = case_list[case_index]

        pos_path = path.join(project_folder, case_id, '{}_Aurora.pos'.format(case_id))
        pos_np = read_aurora(file_path=pos_path)
        # print(pos_np.shape)

        dst_path = path.join(dst_folder, '{}.txt'.format(case_id))
        np.savetxt(dst_path, pos_np)
        print('{} {} saved'.format(case_id, pos_np.shape))
    print('ALL FINISHED')


def save_vid_gifs():
    """
    Convert the frames of video to a gif
    """
    project_folder = '/home/guoh9/tmp/US_vid_frames'
    dst_folder = '/home/guoh9/tmp/US_vid_gif'
    case_list = os.listdir(project_folder)
    case_list.sort()
    kargs = {'duration': 0.05}

    for case in case_list:
        case_folder = os.path.join(project_folder, case)
        frames_list = os.listdir(case_folder)
        frames_list.sort()

        imgs = []
        for frame in frames_list:
            frame_path = path.join(case_folder, frame)
            frame_img = cv2.imread(frame_path)
            imgs.append(frame_img)
        imageio.mimsave(path.join(dst_folder, '{}.gif'.format(case)), imgs, **kargs)
        print('{}.gif saved'.format(case))
    print('ALL CASES FINISHED!!!')


def segmentation_us(input_img):
    # mask_img = cv2.imread('data/US_mask.png', 0)
    # mask_img[mask_img > 50] = 255
    # mask_img[mask_img <= 50] = 0
    #
    # # input_img[mask_img > 50] = 255
    # input_img[mask_img <= 50] = 0
    #
    # cv2.imshow('mask', input_img)
    # cv2.waitKey(0)

    img = np.log2(input_img, dtype=np.float32)
    img = cv2.medianBlur(img, 5)
    ret, thresh = cv2.threshold(img, 0.5, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    closed_copy = copy.copy(closed)
    cv2.imwrite('closed.jpg', closed)


def mask_us(input_img):
    """
    Use the manually created mask to segment useful US areas
    :param input_img:
    :return: masked US image
    """
    # mask_img[mask_img > 50] = 255
    # mask_img[mask_img <= 50] = 0

    # input_img[mask_img > 50] = 255
    input_img[mask_img <= 20] = 0
    return input_img


def params_to_mat44(trans_params, cam_cali_mat):
    """
    Transform the parameters in Aurora files into 4 x 4 matrix
    :param trans_params: transformation parameters in Aurora.pos. Only the last 7 are useful
    3 are translations, 4 are the quaternion (x, y, z, w) for rotation
    :return: 4 x 4 transformation matrix
    """
    if trans_params.shape[0] == 9:
        trans_params = trans_params[2:]

    translation = trans_params[:3]
    quaternion = trans_params[3:]

    """ Transform quaternion to rotation matrix"""
    r_mat = R.from_quat(quaternion).as_matrix()

    trans_mat = np.zeros((4, 4))
    trans_mat[:3, :3] = r_mat
    trans_mat[:3, 3] = translation
    trans_mat[3, 3] = 1

    trans_mat = np.dot(cam_cali_mat, trans_mat)
    trans_mat = inv(trans_mat)

    # new_qua = np.zeros((4, ))
    # new_qua[0] = quaternion[3]
    # new_qua[1:] = quaternion[:3]
    # eulers_from_mat = tfms.euler_from_matrix(r_mat)
    # eulers_from_qua = tfms.euler_from_quaternion(new_qua, axes='sxyz')
    # print('eulers mat\n{}'.format(eulers_from_mat))
    # print('eulers qua\n{}'.format(eulers_from_qua))
    #
    # recon_R = tfms.euler_matrix(eulers_from_mat[0],
    #                             eulers_from_mat[1],
    #                             eulers_from_mat[2])
    # print('R\n{}'.format(r_mat))
    # print('recon_R\n{}'.format(recon_R))
    return trans_mat


def plot_2d_in_3d(trans_params, frame_color='b', input_img=np.ones((480, 640))):
    """
    Plot a 2D frame into 3D space for sequence visualization
    :param input_img: input image frame
    :param trans_params: Aurora position file line of position
    """
    h, w = input_img.shape
    corner_pts = np.asarray([[0, 0, 0],
                             [0, w, 0],
                             [h, w, 0],
                             [h, 0, 0]])
    corner_pts = np.concatenate((corner_pts, np.ones((4, 1))), axis=1)
    corner_pts = np.transpose(corner_pts)
    print('imgshape {}'.format(input_img.shape))
    print('corner_pts:\n{}'.format(corner_pts))

    trans_mat = params_to_mat44(trans_params=trans_params)
    print('trans_mat:\n{}'.format(trans_mat))

    transformed_corner_pts = np.dot(trans_mat, corner_pts)
    print('transformed_corner_pts:\n{}'.format(transformed_corner_pts))
    # dst = np.linalg.norm(transformed_corner_pts[:, 0] - transformed_corner_pts[:, 2])
    # print(dst)

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # w_weights, h_weights = np.meshgrid(np.linspace(0, 1, w),
    #                                    np.linspace(0, 1, h))
    # X = (1 - w_weights - h_weights) * transformed_corner_pts[0, 0] + \
    #     h_weights * transformed_corner_pts[0, 3] + w_weights * transformed_corner_pts[0, 1]
    # Y = (1 - w_weights - h_weights) * transformed_corner_pts[1, 0] + \
    #     h_weights * transformed_corner_pts[1, 3] + w_weights * transformed_corner_pts[1, 1]
    # Z = (1 - w_weights - h_weights) * transformed_corner_pts[2, 0] + \
    #     h_weights * transformed_corner_pts[2, 3] + w_weights * transformed_corner_pts[2, 1]
    # input_img = cv2.cvtColor(input_img, cv2.COLOR_GRAY2RGB)
    # input_img = input_img / 255
    # ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
    #                 facecolors=input_img)
    # plt.show()
    # time.sleep(30)
    for i in range(-1, 3):
        xs = transformed_corner_pts[0, i], transformed_corner_pts[0, i+1]
        ys = transformed_corner_pts[1, i], transformed_corner_pts[1, i+1]
        zs = transformed_corner_pts[2, i], transformed_corner_pts[2, i+1]
        # line = plt3d.art3d.Line3D(xs, ys, zs)
        # ax.add_line(line)
        ax.plot(xs, ys, zs, color=frame_color)

    # ax.plot(pt1, pt2, color='b')
    # ax.scatter()
    # ax.plot(transformed_corner_pts[:3, 0], transformed_corner_pts[:3, 1], color='b')
    # ax.plot(transformed_corner_pts[:3, 1], transformed_corner_pts[:3, 2], color='b')
    # ax.plot(transformed_corner_pts[:3, 2], transformed_corner_pts[:3, 3], color='b')
    # ax.plot(transformed_corner_pts[:3, 3], transformed_corner_pts[:3, 0], color='b')

    plt.show()

def plot_2d_in_3d_test(trans_params1, trans_params2,
                       frame_color='b', input_img=np.ones((480, 640))):
    """
    Plot a 2D frame into 3D space for sequence visualization
    :param input_img: input image frame
    :param trans_params: Aurora position file line of position
    """
    h, w = input_img.shape
    corner_pts = np.asarray([[0, 0, 0],
                             [0, w, 0],
                             [h, w, 0],
                             [h, 0, 0]])
    corner_pts = np.concatenate((corner_pts, np.ones((4, 1))), axis=1)
    corner_pts = np.transpose(corner_pts)
    print('imgshape {}'.format(input_img.shape))
    print('corner_pts:\n{}'.format(corner_pts))

    trans_mat1 = params_to_mat44(trans_params=trans_params1)
    trans_mat2 = params_to_mat44(trans_params=trans_params2)
    print('trans_mat1 shape {}, trans_mat2 shape {}'.format(trans_mat1.shape, trans_mat2.shape))
    print('trans_mat1 shape\n{}\ntrans_mat2 shape\n{}'.format(trans_mat1, trans_mat2))
    # time.sleep(30)

    relative_mat = np.dot(inv(trans_mat1), trans_mat2)

    original_mat2 = np.dot(trans_mat1, relative_mat)
    print('relative_mat\n{}'.format(relative_mat))
    print('original_mat2\n{}'.format(original_mat2))

    transformed_corner_pts = np.dot(trans_mat1, corner_pts)
    print('transformed_corner_pts:\n{}'.format(transformed_corner_pts))
    # dst = np.linalg.norm(transformed_corner_pts[:, 0] - transformed_corner_pts[:, 2])
    # print(dst)

    fig = plt.figure()
    ax = fig.gca(projection='3d')


    for i in range(-1, 3):
        xs = transformed_corner_pts[0, i], transformed_corner_pts[0, i+1]
        ys = transformed_corner_pts[1, i], transformed_corner_pts[1, i+1]
        zs = transformed_corner_pts[2, i], transformed_corner_pts[2, i+1]
        # line = plt3d.art3d.Line3D(xs, ys, zs)
        # ax.add_line(line)
        ax.plot(xs, ys, zs, color=frame_color)

    # ax.plot(pt1, pt2, color='b')
    # ax.scatter()
    # ax.plot(transformed_corner_pts[:3, 0], transformed_corner_pts[:3, 1], color='b')
    # ax.plot(transformed_corner_pts[:3, 1], transformed_corner_pts[:3, 2], color='b')
    # ax.plot(transformed_corner_pts[:3, 2], transformed_corner_pts[:3, 3], color='b')
    # ax.plot(transformed_corner_pts[:3, 3], transformed_corner_pts[:3, 0], color='b')

    plt.show()

def visualize_frames(case_id):
    case_frames_path = path.join(frames_folder, 'Case{:04}'.format(case_id))
    frames_list = os.listdir(case_frames_path)
    frames_list.sort()

    case_pos_path = path.join(pos_folder, 'Case{:04}.txt'.format(case_id))
    case_pos = np.loadtxt(case_pos_path)
    print('frames_list {}, case_pos {}'.format(len(frames_list), case_pos.shape))

    frames_num = case_pos.shape[0]
    colors_R = np.linspace(0, 255, frames_num).astype(np.int16).reshape((frames_num, 1))
    colors_G = np.zeros((frames_num, 1))
    colors_B = np.linspace(255, 0, frames_num).astype(np.int16).reshape((frames_num, 1))

    colors = np.concatenate((colors_R, colors_G, colors_B), axis=1)

    for frame_id in range(frames_num):
        frame_pos = case_pos[frame_id, :]
        frame_color = tuple(colors[frame_id, :])

        time.sleep(30)

class VisualizeSequence():
    def __init__(self, case_id):
        super(VisualizeSequence, self).__init__()
        self.case_frames_path = path.join(frames_folder, 'Case{:04}'.format(case_id))
        self.frames_list = os.listdir(self.case_frames_path)
        self.frames_list.sort()

        self.cam_cali_mat = np.loadtxt('/zion/common/data/uronav_data/Case{:04}/'
                                       'Case{:04}_USCalib.txt'.format(case_id, case_id))

        case_pos_path = path.join(pos_folder, 'Case{:04}.txt'.format(case_id))
        self.case_pos = np.loadtxt(case_pos_path)
        print('frames_list {}, case_pos {}'.format(len(self.frames_list), self.case_pos.shape))

        self.frames_num = self.case_pos.shape[0]
        colors_R = np.linspace(0, 1, self.frames_num).reshape((self.frames_num, 1))
        colors_G = np.zeros((self.frames_num, 1))
        colors_B = np.linspace(1, 0, self.frames_num).reshape((self.frames_num, 1))

        self.colors = np.concatenate((colors_R, colors_G, colors_B), axis=1)

        self.fig = plt.figure()
        self.ax = self.fig.gca(projection='3d')

        def plot_frame3d(trans_params, frame_color=(255, 0, 0),
                         input_img=np.ones((480, 640)), plot_img=False):
            """
            Plot a 2D frame into 3D space for sequence visualization
            :param input_img: input image frame
            :param trans_params: Aurora position file line of position
            """
            h, w = input_img.shape
            # corner_pts = np.asarray([[0, 0, 0],
            #                          [0, w, 0],
            #                          [h, w, 0],
            #                          [h, 0, 0]])
            corner_pts = np.asarray([[-h, 0, 0],
                                     [-h, -w, 0],
                                     [0, -w, 0],
                                     [0, 0, 0]])
            corner_pts = np.concatenate((corner_pts, np.ones((4, 1))), axis=1)
            corner_pts = np.transpose(corner_pts)
            print('imgshape {}'.format(input_img.shape))
            print('corner_pts:\n{}'.format(corner_pts))
            print('h {}, w {}'.format(h, w))

            trans_mat = params_to_mat44(trans_params=trans_params,
                                        cam_cali_mat=self.cam_cali_mat)
            # trans_mat = trans_mat.transpose()
            # trans_mat = np.dot(self.cam_cali_mat, trans_mat)
            # trans_mat = inv(trans_mat)
            # trans_mat = np.dot(trans_mat, inv(self.cam_cali_mat))
            # trans_mat = np.dot(trans_mat, self.cam_cali_mat)

            print('trans_mat:\n{}'.format(trans_mat))

            transformed_corner_pts = np.dot(trans_mat, corner_pts)
            # time.sleep(30)
            print('transformed_corner_pts:\n{}'.format(transformed_corner_pts))
            # dst = np.linalg.norm(transformed_corner_pts[:, 0] - transformed_corner_pts[:, 1])
            # dst2 = np.linalg.norm(transformed_corner_pts[:, 1] - transformed_corner_pts[:, 2])
            # print(dst, dst2)

            for i in range(-1, 3):
                xs = transformed_corner_pts[0, i], transformed_corner_pts[0, i + 1]
                ys = transformed_corner_pts[1, i], transformed_corner_pts[1, i + 1]
                zs = transformed_corner_pts[2, i], transformed_corner_pts[2, i + 1]
                if i == 0 or i == 2:
                    linewidth = 10
                else:
                    linewidth = 1
                self.ax.plot(xs, ys, zs, color=frame_color, lw=linewidth)

            if plot_img:
                w_weights, h_weights = np.meshgrid(np.linspace(0, 1, w),
                                                   np.linspace(0, 1, h))
                X = (1 - w_weights - h_weights) * transformed_corner_pts[0, 0] + \
                    h_weights * transformed_corner_pts[0, 3] + w_weights * transformed_corner_pts[0, 1]
                Y = (1 - w_weights - h_weights) * transformed_corner_pts[1, 0] + \
                    h_weights * transformed_corner_pts[1, 3] + w_weights * transformed_corner_pts[1, 1]
                Z = (1 - w_weights - h_weights) * transformed_corner_pts[2, 0] + \
                    h_weights * transformed_corner_pts[2, 3] + w_weights * transformed_corner_pts[2, 1]
                input_img = cv2.cvtColor(input_img, cv2.COLOR_GRAY2RGB)
                input_img = input_img / 255
                self.ax.plot_surface(X, Y, Z, rstride=20, cstride=20, facecolors=input_img)


        for frame_id in range(self.frames_num):
            frame_pos = self.case_pos[frame_id, :]
            frame_color = tuple(self.colors[frame_id, :])
            frame_img = cv2.imread(path.join(self.case_frames_path, '{:04}.jpg'.format(frame_id)), 0)
            plot_frame3d(trans_params=frame_pos, frame_color=frame_color,
                         input_img=frame_img, plot_img=False)
            print('{} frame'.format(frame_id))
        plt.show()


def get_6dof_label(trans_params1, trans_params2, cam_cali_mat):
    """
    Given two Aurora position lines of two frames, return the relative 6 degrees of freedom label
    Aurora position line gives the transformation from the ultrasound tracker to Aurora
    :param trans_params1: Aurora position line of the first frame
    :param trans_params2: Aurora position line of the second frame
    :param cam_cali_mat: Camera calibration matrix of this case, which is the transformation from
    the ultrasound image upper left corner (in pixel) to the ultrasound tracker (in mm).
    :return: the relative 6 degrees of freedom (3 translations and 3 rotations xyz) as training label
    Note that this dof is based on the position of the first frame
    """
    trans_mat1 = params_to_mat44(trans_params1, cam_cali_mat=cam_cali_mat)
    trans_mat2 = params_to_mat44(trans_params2, cam_cali_mat=cam_cali_mat)

    relative_mat = np.dot(trans_mat2, inv(trans_mat1))

    translations = relative_mat[:3, 3]
    rotations = R.from_matrix(relative_mat[:3, :3])
    rotations_eulers = rotations.as_euler('xyz')

    dof = np.concatenate((translations, rotations_eulers), axis=0)
    return dof


def get_next_pos(trans_params1, dof, cam_cali_mat):
    """
    Given the first frame's Aurora position line and relative 6dof, return second frame's position line
    :param trans_params1: Aurora position line of the first frame
    :param dof: 6 degrees of freedom based on the first frame
    :param cam_cali_mat: Camera calibration matrix of this case
    :return: Aurora position line of the second frame
    """
    trans_mat1 = params_to_mat44(trans_params1, cam_cali_mat=cam_cali_mat)

    relative_mat = np.identity(4)
    r_recon = R.from_euler('xyz', dof[3:])
    relative_mat[:3, :3] = r_recon.as_matrix()
    relative_mat[:3, 3] = dof[:3]

    next_mat = np.dot(inv(cam_cali_mat), inv(np.dot(relative_mat, trans_mat1)))

    next_params = np.zeros(7)
    next_params[:3] = next_mat[:3, 3]
    next_params[3:] = R.from_matrix(next_mat[:3, :3]).as_quat()
    return next_params


def center_crop(input_img, crop_size=480):
    h, w = input_img.shape
    if crop_size > 480:
        crop_size = 480
    x_start = int((h - crop_size) / 2)
    y_start = int((w - crop_size) / 2)

    patch_img = input_img[x_start:x_start+crop_size, y_start:y_start+crop_size]

    return patch_img


class TestNetwork():
    def __init__(self, case_id):
        super(TestNetwork, self).__init__()
        self.case_id = case_id
        if 1 <= self.case_id <= 71:
            self.data_part = 'test'
        elif 71 < self.case_id <= 140:
            self.data_part = 'val'
        elif 140 < self.case_id <= 747:
            self.data_part = 'train'
        self.case_frames_path = path.join(frames_folder, self.data_part,
                                          'Case{:04}'.format(case_id))
        self.frames_list = os.listdir(self.case_frames_path)
        self.frames_list.sort()

        self.cam_cali_mat_path = path.join(cali_folder, 'Case{:04}_USCalib.txt'.format(case_id, case_id))
        self.cam_cali_mat = np.loadtxt(self.cam_cali_mat_path)

        case_pos_path = path.join(pos_folder, 'Case{:04}.txt'.format(case_id))
        self.case_pos = np.loadtxt(case_pos_path)

        # self.case_pos = self.case_pos[:10, :]

        ''' IF we resample the video to 100 frames'''
        # if self.case_pos.shape[0] >= 110 or self.case_pos.shape[0] <= 150:
        #     self.slice_ids = np.linspace(0, self.case_pos.shape[0]-1, 90).astype(np.uint64)
        #     self.case_pos = self.case_pos[self.slice_ids]
        # else:
        #     self.slice_ids = np.linspace(0, self.case_pos.shape[0]-1, self.case_pos.shape[0]).astype(np.uint64)
        self.slice_ids = np.linspace(0, self.case_pos.shape[0]-1, self.case_pos.shape[0]).astype(np.uint64)
        # print(self.slice_ids)

        print('frames_list {}, case_pos {}'.format(len(self.frames_list), self.case_pos.shape))

        self.frames_num = self.case_pos.shape[0]
        colors_R = np.linspace(0, 1, self.frames_num).reshape((self.frames_num, 1))
        colors_G = np.zeros((self.frames_num, 1))
        colors_B = np.linspace(1, 0, self.frames_num).reshape((self.frames_num, 1))

        self.colors = np.concatenate((colors_R, colors_G, colors_B), axis=1)

        self.fig = plt.figure()
        self.ax = self.fig.gca(projection='3d')

        # def divide_batch(slice_num=end_frame_index, batch_size=32):
        #     """
        #     Divide all the slices into batches for torch parallel computing
        #     :param slice_num: number of slices in a video
        #     :param batch_size: default 32
        #     :return: a list of array, each array is a batch that contains the index of frames
        #     """
        #     end_frame_index = slice_ids.shape[0] - neighbour_slice + 1
        #     print(end_frame_index)
        #     time.sleep(30)
        #     batches_num = slice_ids.shape[0] // batch_size
        #     last_batch_size = slice_ids.shape[0] % batch_size
        #     print('slice_num {}, batch_size {}'.format(slice_ids.shape[0], batch_size))
        #     print('batches_num {}, last_batch_size {}'.format(batches_num, last_batch_size))
        #     batch_ids = []
        #     for i in range(batches_num):
        #         # this_batch_id = np.arange(i * batch_size, (i + 1) * batch_size)
        #         this_batch_id = slice_ids[i * batch_size: (i + 1) * batch_size]
        #         batch_ids.append(this_batch_id)
        #     if last_batch_size != 0:
        #         last_batch_id = np.arange(batches_num * batch_size, batches_num * batch_size + last_batch_size)
        #         # last_batch_id = np.flip(last_batch_id)
        #         batch_ids.append(last_batch_id)
        #     print(batch_ids)
        #     time.sleep(30)
        #     return batch_ids

        def divide_batch(slice_num, batch_size=32):
            """
            Divide all the slices into batches for torch parallel computing
            :param slice_num: number of slices in a video
            :param batch_size: default 32
            :return: a list of array, each array is a batch that contains the index of frames
            """
            batches_num = slice_num // batch_size
            last_batch_size = slice_num % batch_size
            print('slice_num {}, batch_size {}'.format(slice_num, batch_size))
            print('batches_num {}, last_batch_size {}'.format(batches_num, last_batch_size))
            batch_ids = []
            for i in range(batches_num):
                # this_batch_id = np.arange(i * batch_size, (i + 1) * batch_size)
                this_batch_id = self.slice_ids[i * batch_size: (i + 1) * batch_size]
                # this_batch_id = np.flip(this_batch_id)
                batch_ids.append(this_batch_id)
            if last_batch_size != 0:
                # last_batch_id = np.arange(batches_num * batch_size, batches_num * batch_size + last_batch_size)
                last_batch_id = self.slice_ids[batches_num * batch_size:slice_num]
                # last_batch_id = np.flip(last_batch_id)
                batch_ids.append(last_batch_id)
            # print(batch_ids)
            # time.sleep(30)
            return batch_ids

        def get_batch_dofs():
            """
            Give the batches as input
            :return: (frames_num - neighbour_slice + 1) x (neighbour_slice - 1) x 6
            contains the relative motion between two slices within a sample group.
            For example, if a neighbouring sample contains 10 slices, then there are 9 relative
            motions within this group
            """
            end_frame_index = self.frames_num - neighbour_slice + 1
            print('end_frame_index/frame_num {}/{}'.format(end_frame_index, self.frames_num))
            batch_groups = divide_batch(slice_num=end_frame_index, batch_size=batch_size)
            # time.sleep(30)

            if output_type == 'sum_dof':
                result_dof = np.zeros((1, 6))
            else:
                result_dof = np.zeros((1, neighbour_slice - 1, 6))

            for batch_index in range(len(batch_groups)):
                this_batch = batch_groups[batch_index]
                batch_imgs = []
                for group_index in range(len(this_batch)):
                    group_id = this_batch[group_index]
                    sample_slices = []
                    # print(group_id)
                    frame_index = batch_index * neighbour_slice + group_index
                    for i in range(neighbour_slice):
                        frame_id = int(self.slice_ids[frame_index + i])
                        # print('frame_id {}'.format(frame_id))
                        frame_path = path.join(self.case_frames_path, '{:04}.jpg'.format(frame_id))
                        # frame_path = path.join(self.case_frames_path, '{:04}.jpg'.format(20))
                        frame_img = cv2.imread(frame_path, 0)
                        frame_img = data_transform(frame_img, masked_full=False)
                        # frame_img = data_transform(frame_img)
                        sample_slices.append(frame_img)

                    if input_type == 'diff_img':
                        diff_imgs = []
                        for sample_id in range(1, len(sample_slices)):
                            diff_imgs.append(sample_slices[sample_id] - sample_slices[sample_id - 1])
                        sample_slices = np.asarray(diff_imgs)
                    else:
                        sample_slices = np.asarray(sample_slices)

                    batch_imgs.append(sample_slices)
                batch_imgs = np.asarray(batch_imgs)
                if network_type in train_network.networks3D:
                    batch_imgs = np.expand_dims(batch_imgs, axis=1)
                batch_imgs = torch.from_numpy(batch_imgs).float().to(device)

                outputs, maps = model_ft(batch_imgs)

                """ Visualize attention heatmaps """
                # tools.visualize_attention(case_id=self.case_id,
                #                           batch_ids=this_batch,
                #                           batch_imgs=batch_imgs,
                #                           maps=maps, weights=fc_weights)

                print('this_batch {}'.format(this_batch))
                print('maps shape {}'.format(maps.shape))
                print('fc_weights shape {}'.format(fc_weights.shape))
                # print('input shape {}'.format(batch_imgs.shape))
                # print('outputs shape {}'.format(outputs.shape))
                outputs = outputs.data.cpu().numpy()
                if output_type == 'average_dof':
                    outputs = np.expand_dims(outputs, axis=1)
                    outputs_reshape = np.repeat(outputs, neighbour_slice - 1, axis=1)
                elif output_type == 'sum_dof':
                    outputs_reshape = outputs
                else:
                    outputs_reshape = np.reshape(outputs, (outputs.shape[0],
                                                           int(outputs.shape[1] / 6),
                                                           int(outputs.shape[1] / (neighbour_slice - 1))))
                result_dof = np.concatenate((result_dof, outputs_reshape), axis=0)

            if output_type == 'sum_dof':
                result_dof = result_dof[1:, :]
            else:
                result_dof = result_dof[1:, :, :]
            return result_dof

        def get_format_dofs(batch_dofs, merge_option='average_dof'):
            """
            Based on the network outputs, here reformat the result into one row for each frame
            (Because there are many overlapping frames due to the input format)
            :return:
            1) gen_dofs is (slice_num - 1) x 6dof. It is the relative 6dof motion comparing to
            the former frame
            2) pos_params is slice_num x 7params. It is the absolute position, exactly the same
            format as Aurora.pos file
            """
            print('Use <{}> formatting dofs'.format(merge_option))
            if merge_option == 'one':
                gen_dofs = np.zeros((self.frames_num - 1, 6))
                gen_dofs[:batch_dofs.shape[0], :] = batch_dofs[:, 0, :]
                gen_dofs[batch_dofs.shape[0], :] = batch_dofs[-1, 1, :]
                print('gen_dof shape {}'.format(gen_dofs.shape))
                print('not average method')

            elif merge_option == 'baton':
                print('baton batch_dofs shape {}'.format(batch_dofs.shape))
                print('slice_num {}'.format(self.frames_num))
                print('neighboring {}'.format(neighbour_slice))

                gen_dofs = []
                slice_params = []
                for slice_idx in range(self.frames_num):
                    if slice_idx == 0:
                        this_params = self.case_pos[slice_idx, :]
                        slice_params.append(this_params)
                    elif slice_idx < neighbour_slice:
                        this_dof = batch_dofs[0, :] / 4
                        this_params = tools.get_next_pos(trans_params1=slice_params[slice_idx-1],
                                                         dof=this_dof,
                                                         cam_cali_mat=self.cam_cali_mat)
                        gen_dofs.append(this_dof)
                        slice_params.append(this_params)
                    else:
                        baton_idx = slice_idx - neighbour_slice + 1
                        baton_params = slice_params[baton_idx]
                        sample_dof = batch_dofs[baton_idx, :]
                        this_params = tools.get_next_pos(trans_params1=baton_params,
                                                         dof=sample_dof,
                                                         cam_cali_mat=self.cam_cali_mat)
                        this_dof = tools.get_6dof_label(trans_params1=slice_params[slice_idx-1],
                                                        trans_params2=this_params,
                                                        cam_cali_mat=self.cam_cali_mat)
                        gen_dofs.append(this_dof)
                        slice_params.append(this_params)
                gen_dofs = np.asarray(gen_dofs)
                slice_params = np.asarray(slice_params)
                print('gen_dof shape {}'.format(gen_dofs.shape))
                print('slice_params shape {}'.format(slice_params.shape))
                # time.sleep(30)
            else:
                frames_pos = []
                for start_sample_id in range(batch_dofs.shape[0]):
                    for relative_id in range(batch_dofs.shape[1]):
                        this_pos_id = start_sample_id + relative_id + 1
                        # print('this_pos_id {}'.format(this_pos_id))
                        this_pos = batch_dofs[start_sample_id, relative_id, :]
                        this_pos = np.expand_dims(this_pos, axis=0)
                        if len(frames_pos) < this_pos_id:
                            frames_pos.append(this_pos)
                        else:
                            frames_pos[this_pos_id - 1] = np.concatenate((frames_pos[this_pos_id - 1],
                                                                          this_pos), axis=0)

                gen_dofs = []
                for i in range(len(frames_pos)):
                    gen_dof = np.mean(frames_pos[i], axis=0)

                    """This is for Linear Motion"""
                    # gen_dof = train_network.dof_stats[:, 0]
                    # gen_dof = np.asarray([-0.07733258, -1.28508398, 0.37141262,
                    #                       -0.57584312, 0.20969176, 0.51404395]) + 0.1

                    gen_dofs.append(gen_dof)
                gen_dofs = np.asarray(gen_dofs)

                print('batch_dofs {}'.format(batch_dofs.shape))
                print('gen_dofs {}'.format(gen_dofs.shape))
                # time.sleep(30)

            # for dof_id in range(6):
            #     gen_dofs[:, dof_id] = tools.smooth_array(gen_dofs[:, dof_id])
            # time.sleep(30)
            return gen_dofs


        def dof2params(format_dofs):
            gen_param_results = []
            for i in range(format_dofs.shape[0]):
                if i == 0:
                    base_param = self.case_pos[i, :]
                else:
                    base_param = gen_param_results[i-1]
                gen_dof = format_dofs[i, :]
                gen_param = tools.get_next_pos(trans_params1=base_param,
                                               dof=gen_dof, cam_cali_mat=self.cam_cali_mat)
                gen_param_results.append(gen_param)
            # time.sleep(30)
            gen_param_results = np.asarray(gen_param_results)
            pos_params = np.zeros((self.frames_num, 7))
            pos_params[0, :] = self.case_pos[0, 2:]
            pos_params[1:, :] = gen_param_results
            print('pos_params shape {}'.format(pos_params.shape))
            # time.sleep(30)
            return pos_params

        def plot_frame3d(trans_params, frame_color=(255, 0, 0),
                         input_img=np.ones((480, 640)), plot_img=False):
            """
            Plot a 2D frame into 3D space for sequence visualization
            :param input_img: input image frame
            :param trans_params: Aurora position file line of position
            """
            h, w = input_img.shape
            # corner_pts = np.asarray([[0, 0, 0],
            #                          [0, w, 0],
            #                          [h, w, 0],
            #                          [h, 0, 0]])
            corner_pts = np.asarray([[-h, 0, 0],
                                     [-h, -w, 0],
                                     [0, -w, 0],
                                     [0, 0, 0]])
            corner_pts = np.concatenate((corner_pts, np.ones((4, 1))), axis=1)
            corner_pts = np.transpose(corner_pts)
            print('imgshape {}'.format(input_img.shape))
            print('corner_pts:\n{}'.format(corner_pts))
            print('h {}, w {}'.format(h, w))

            trans_mat = params_to_mat44(trans_params=trans_params,
                                        cam_cali_mat=self.cam_cali_mat)
            # trans_mat = trans_mat.transpose()
            # trans_mat = np.dot(self.cam_cali_mat, trans_mat)
            # trans_mat = inv(trans_mat)
            # trans_mat = np.dot(trans_mat, inv(self.cam_cali_mat))
            # trans_mat = np.dot(trans_mat, self.cam_cali_mat)

            print('trans_mat:\n{}'.format(trans_mat))

            transformed_corner_pts = np.dot(trans_mat, corner_pts)
            print('transformed_corner_pts:\n{}'.format(transformed_corner_pts))
            print('transformed_corner_pts shape {}'.format(transformed_corner_pts.shape))
            # dst = np.linalg.norm(transformed_corner_pts[:, 0] - transformed_corner_pts[:, 1])
            # dst2 = np.linalg.norm(transformed_corner_pts[:, 1] - transformed_corner_pts[:, 2])
            # print(dst, dst2)
            # time.sleep(30)

            for i in range(-1, 3):
                xs = transformed_corner_pts[0, i], transformed_corner_pts[0, i + 1]
                ys = transformed_corner_pts[1, i], transformed_corner_pts[1, i + 1]
                zs = transformed_corner_pts[2, i], transformed_corner_pts[2, i + 1]
                if i == 0 or i == 2:
                    linewidth = 10
                else:
                    linewidth = 1
                self.ax.plot(xs, ys, zs, color=frame_color, lw=linewidth)

            if plot_img:
                w_weights, h_weights = np.meshgrid(np.linspace(0, 1, w),
                                                   np.linspace(0, 1, h))
                X = (1 - w_weights - h_weights) * transformed_corner_pts[0, 0] + \
                    h_weights * transformed_corner_pts[0, 3] + w_weights * transformed_corner_pts[0, 1]
                Y = (1 - w_weights - h_weights) * transformed_corner_pts[1, 0] + \
                    h_weights * transformed_corner_pts[1, 3] + w_weights * transformed_corner_pts[1, 1]
                Z = (1 - w_weights - h_weights) * transformed_corner_pts[2, 0] + \
                    h_weights * transformed_corner_pts[2, 3] + w_weights * transformed_corner_pts[2, 1]
                input_img = cv2.cvtColor(input_img, cv2.COLOR_GRAY2RGB)
                input_img = input_img / 255
                self.ax.plot_surface(X, Y, Z, rstride=10, cstride=10, facecolors=input_img)

        def params2corner_pts(params, input_img=np.ones((480, 640))):
            """
            Transform the Aurora params to corner points coordinates of each frame
            :param params: slice_num x 7(or 9) params matrix
            :param input_img: just use for size
            :return: slice_num x 4 x 3. 4 corner points 3d coordinates (x, y, z)
            """
            h, w = input_img.shape
            corner_pts = np.asarray([[-h, 0, 0],
                                     [-h, -w, 0],
                                     [0, -w, 0],
                                     [0, 0, 0]])
            corner_pts = np.concatenate((corner_pts, np.ones((4, 1))), axis=1)
            corner_pts = np.transpose(corner_pts)

            transformed_pts = []
            for frame_id in range(params.shape[0]):
                trans_mat = params_to_mat44(trans_params=params[frame_id, :],
                                            cam_cali_mat=self.cam_cali_mat)
                transformed_corner_pts = np.dot(trans_mat, corner_pts)
                transformed_corner_pts = np.moveaxis(transformed_corner_pts[:3, :], 0, 1)
                transformed_pts.append(transformed_corner_pts)
            transformed_pts = np.asarray(transformed_pts)
            return transformed_pts

        def draw_img_sequence(corner_pts):
            for frame_id in range(corner_pts.shape[0]):
                w_weights, h_weights = np.meshgrid(np.linspace(0, 1, 224),
                                                   np.linspace(0, 1, 224))
                # print('corner_pts shape {}'.format(corner_pts.shape))
                # time.sleep(30)
                X = (1 - w_weights - h_weights) * corner_pts[frame_id, 0, 0] + \
                    h_weights * corner_pts[frame_id, 3, 0] + w_weights * corner_pts[frame_id, 1, 0]
                Y = (1 - w_weights - h_weights) * corner_pts[frame_id, 0, 1] + \
                    h_weights * corner_pts[frame_id, 3, 1] + w_weights * corner_pts[frame_id, 1, 1]
                Z = (1 - w_weights - h_weights) * corner_pts[frame_id, 0, 2] + \
                    h_weights * corner_pts[frame_id, 3, 2] + w_weights * corner_pts[frame_id, 1, 2]

                img_path = path.join(self.case_frames_path, self.frames_list[frame_id])
                input_img = cv2.imread(img_path, 0)
                input_img = train_network.data_transform(input_img)
                print('frame_path\n{}'.format(self.frames_list[frame_id]))
                # time.sleep(30)
                input_img = cv2.cvtColor(input_img, cv2.COLOR_GRAY2RGB)
                input_img = input_img / 255

                if frame_id == 0 or frame_id == corner_pts.shape[0] - 1:
                    stride = 2
                else:
                    stride = 10
                # self.ax.plot_surface(X, Y, Z, rstride=20, cstride=20, facecolors=input_img)
                self.ax.plot_surface(X, Y, Z, rstride=stride, cstride=stride,
                                     facecolors=input_img, zorder=0.1)

        def draw_one_sequence(corner_pts, name, colorRGB=(255, 0, 0), line_width=3, constant=True):
            colorRGB = tuple(channel/255 for channel in colorRGB)
            seg_num = corner_pts.shape[0] + 1


            if constant:
                constant_color = np.asarray(colorRGB)
                constant_color = np.expand_dims(constant_color, axis=0)
                colors = np.repeat(constant_color, seg_num, axis=0)
            else:
                colors_R = np.linspace(0, colorRGB[0], seg_num).reshape((seg_num, 1))
                colors_G = np.linspace(0, colorRGB[1], seg_num).reshape((seg_num, 1))
                colors_B = np.linspace(1, colorRGB[2], seg_num).reshape((seg_num, 1))

                colors = np.concatenate((colors_R, colors_G, colors_B), axis=1)


            # for frame_id in range(int(corner_pts.shape[0] * 0.5), corner_pts.shape[0]):
            #     if frame_id == int(corner_pts.shape[0] * 0.5):
            for frame_id in range(corner_pts.shape[0]):
                if frame_id == 0:
                    """ First frame draw full bounds"""
                    for pt_id in range(-1, 3):
                        xs = corner_pts[frame_id, pt_id, 0], corner_pts[frame_id, pt_id + 1, 0]
                        ys = corner_pts[frame_id, pt_id, 1], corner_pts[frame_id, pt_id + 1, 1]
                        zs = corner_pts[frame_id, pt_id, 2], corner_pts[frame_id, pt_id + 1, 2]
                        self.ax.plot(xs, ys, zs, color=tuple(colors[frame_id, :]), lw=line_width, zorder=1)
                elif frame_id == corner_pts.shape[0] - 1:
                    """ Connect to the former frame """
                    for pt_id in range(-1, 3):
                        xs = corner_pts[frame_id, pt_id, 0], corner_pts[frame_id - 1, pt_id, 0]
                        ys = corner_pts[frame_id, pt_id, 1], corner_pts[frame_id - 1, pt_id, 1]
                        zs = corner_pts[frame_id, pt_id, 2], corner_pts[frame_id - 1, pt_id, 2]
                        self.ax.plot(xs, ys, zs, color=tuple(colors[frame_id, :]), lw=line_width)
                    """ Last frame draw full bounds"""
                    for pt_id in range(-1, 3):
                        xs = corner_pts[frame_id, pt_id, 0], corner_pts[frame_id, pt_id + 1, 0]
                        ys = corner_pts[frame_id, pt_id, 1], corner_pts[frame_id, pt_id + 1, 1]
                        zs = corner_pts[frame_id, pt_id, 2], corner_pts[frame_id, pt_id + 1, 2]
                        self.ax.plot(xs, ys, zs, color=tuple(colors[-1, :]), lw=line_width)
                        if pt_id == -1:
                            self.ax.plot(xs, ys, zs, color=tuple(colors[-1, :]), lw=line_width, label=name)
                else:
                    """ Connect to the former frame """
                    for pt_id in range(-1, 3):
                        xs = corner_pts[frame_id, pt_id, 0], corner_pts[frame_id - 1, pt_id, 0]
                        ys = corner_pts[frame_id, pt_id, 1], corner_pts[frame_id - 1, pt_id, 1]
                        zs = corner_pts[frame_id, pt_id, 2], corner_pts[frame_id - 1, pt_id, 2]
                        self.ax.plot(xs, ys, zs, color=tuple(colors[frame_id, :]), lw=line_width, zorder=1)

                # if plot_img and frame_id==0:


        def visualize_sequences():
            # draw_img_sequence(corner_pts=self.gt_pts1)
            draw_one_sequence(corner_pts=self.gt_pts1, name='Groundtruth',
                              colorRGB=(0, 153, 76), line_width=3)
            draw_one_sequence(corner_pts=self.trans_pts1, name='DCL-Net ({:.4f}mm)'.format(self.trans_pts1_error),
                              colorRGB=(255, 0, 0))

            plt.axis('off')
            self.ax.set_xticklabels([])
            self.ax.set_yticklabels([])
            self.ax.set_zticklabels([])
            plt.legend(loc='lower left')
            plt.tight_layout()

            # views_id = np.linspace(0, 360, 36)
            # for ii in views_id:
            #     self.ax.view_init(elev=10., azim=ii)
            #     plt.savefig('views/{}_img.jpg'.format(ii))
            #     # plt.savefig('views/{}.jpg'.format(ii))
            #     print('{} saved'.format(ii))

            self.ax.view_init(elev=10., azim=0)
            # plt.savefig('views/{}_img.jpg'.format(0))
            plt.savefig('views/all_cases/{}_{}.jpg'.format(model_string, case_id))

            plt.title('Case{:04}'.format(self.case_id))
            plt.show()

        def get_gt_dofs():
            gt_dofs = []
            for slice_id in range(1, self.frames_num):
                params1 = self.case_pos[slice_id-1, :]
                params2 = self.case_pos[slice_id, :]
                this_dof = tools.get_6dof_label(trans_params1=params1,
                                                trans_params2=params2,
                                                cam_cali_mat=self.cam_cali_mat)
                gt_dofs.append(this_dof)
            gt_dofs = np.asarray(gt_dofs)
            print('gt_dof shape {}, frames_num {}'.format(gt_dofs.shape, self.frames_num))
            return gt_dofs

        def visualize_dofs():
            frees = ['tX', 'tY', 'tZ', 'aX', 'aY', 'aZ']
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle('Case{:04}'.format(self.case_id))
            for dof_id in range(len(frees)):
                plot_x = dof_id // 3
                plot_y = dof_id % 3
                axes[plot_x, plot_y].plot(self.gt_dofs[:, dof_id], color='g', label='Groundtruth', alpha=0.5)
                axes[plot_x, plot_y].plot(self.format_dofs[:, dof_id], color='r', label='CNN', alpha=0.5)

                corrcoef = np.corrcoef(self.gt_dofs[:, dof_id], self.format_dofs[:, dof_id])[0, 1]

                axes[plot_x, plot_y].set_title('{}: corrcoef {:.4f}'.format(frees[dof_id], corrcoef))
                axes[plot_x, plot_y].legend(loc='lower left')
                # axes[plot_x, plot_y].show()

                np.savetxt('figures/dof_values/{}_{}_gt.txt'.format(self.case_id, frees[dof_id]),
                           self.gt_dofs[:, dof_id])
                np.savetxt('figures/dof_values/{}_{}_{}_pd.txt'.format(model_string, self.case_id, frees[dof_id]),
                           self.format_dofs[:, dof_id])


            plt.savefig('figures/dof_pred/Case{:04}.jpg'.format(self.case_id))
            # plt.show()





        self.batch_dofs = get_batch_dofs()
        if output_type == 'sum_dof':
            self.format_dofs = get_format_dofs(self.batch_dofs, merge_option='baton')
        else:
            self.format_dofs = get_format_dofs(self.batch_dofs, merge_option='average')

        if normalize_dof:
            self.format_dofs = self.format_dofs * train_network.dof_stats[:, 1] \
                               + train_network.dof_stats[:, 0]


        print('format_dofs\n{}'.format(np.around(self.format_dofs, decimals=2)))
        self.gt_dofs = get_gt_dofs()
        # print('mean gt_dof\n{}'.format(np.mean(self.gt_dofs, axis=0)))
        # time.sleep(30)

        # np.savetxt('infos/gt_dofs.txt', self.gt_dofs)
        # np.savetxt('infos/format_dofs.txt', self.format_dofs)
        # print('saved')
        # time.sleep(30)

        # self.gt_means = np.mean(self.gt_dofs, axis=0)
        # np.savetxt('infos/linear_motion.txt', self.gt_means)

        # self.format_dofs = np.zeros((1, 6))
        # self.format_dofs[0, :] = np.loadtxt('infos/linear_motion.txt')
        # self.format_dofs = np.repeat(self.format_dofs, self.gt_dofs.shape[0], axis=0)
        # print('shapes gt {}, linear {}'.format(self.gt_dofs.shape, self.format_dofs.shape))
        # visualize_dofs()

        self.result_params = dof2params(self.format_dofs)
        print('frame_position shape {}'.format(self.result_params.shape))
        print('self.case_pos shape {}'.format(self.case_pos.shape))

        self.imgs_pts1 = tools.params2corner_pts(params=self.case_pos, cam_cali_mat=self.cam_cali_mat,
                                                 shrink=1)

        self.gt_pts1 = tools.params2corner_pts(params=self.case_pos, cam_cali_mat=self.cam_cali_mat)
        self.trans_pts1 = tools.params2corner_pts(params=self.result_params, cam_cali_mat=self.cam_cali_mat)
        np.save('results/trans_pts/{}_{:04}.npy'.format(model_string, self.case_id), self.trans_pts1)
        np.save('results/trans_pts/GT_{:04}.npy'.format(self.case_id), self.gt_pts1)

        # time.sleep(30)
        self.trans_pts1_error = tools.evaluate_dist(pts1=self.gt_pts1, pts2=self.trans_pts1)
        self.final_drift = tools.final_drift(pts1=self.gt_pts1[-1, :, :], pts2=self.trans_pts1[-1, :, :])
        self.cor_coe = tools.evaluate_correlation(dof1=self.format_dofs, dof2=self.gt_dofs, abs=True)
        print('self.gt_pts1 shape {}'.format(self.gt_pts1.shape))
        print('self.trans_pts1 shape {}'.format(self.trans_pts1.shape))
        print('Case{:04} error {:.4f}mm'.format(self.case_id, self.trans_pts1_error))
        print('Case{:04} final drift {:.4f}mm'.format(self.case_id, self.final_drift))
        print('Case{:04} correlation: {}'.format(self.case_id, self.cor_coe))
        print('*' * 50)
        visualize_sequences()
#

if __name__ == '__main__':
    batch_size = 5
    neighbour_slice = 5
    network_type = 'resnext50'
    input_type = 'org_img'
    output_type = 'average_dof'
    normalize_dof = True
    device = torch.device("cuda:{}".format(device_no))

    model_string = '0312-185335'
    model_folder = 'pretrained_networks'
    model_path = path.join(model_folder, '3d_best_Generator_{}.pth'.format(model_string))
    model_ft = train_network.define_model(model_type=network_type,
                                          pretrained_path=model_path,
                                          input_type=input_type,
                                          output_type=output_type,
                                          neighbour_slice=neighbour_slice)
    print('torch model loaded')


    params = model_ft.state_dict()
    fc_weights = params['fc.weight'].data.cpu().numpy()
    # print(fc_weights.shape)

    since = time.time()
    case = TestNetwork(case_id=5)
    time_elapsed = time.time() - since
    print('One case testing complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


    """ Following parts are for our full dataset testing """
    # errors = []
    # final_drifts = []
    # frame_nums = []
    # corr_coefs = []
    #
    # for i in test_ids:
    # # for i in val_ids:
    # # for i in train_ids:
    #     case = TestNetwork(case_id=i)
    #     results_pos = case.result_params
    #     case_error = case.trans_pts1_error
    #     case_corr = case.cor_coe
    #     errors.append(case_error)
    #     final_drifts.append(case.final_drift)
    #     frame_nums.append([i, case.frames_num])
    #     corr_coefs.append(case_corr)
    #     # np.savetxt('results/pos/Case{:04}_Aurora_result.pos'.format(i), results_pos, fmt='%.6f')
    # errors = np.asarray(errors)
    # avg_error = np.mean(errors)
    # np.savetxt('results/{}.txt'.format(model_string), errors)
    # np.savetxt('infos/frame_nums.txt', np.asarray(frame_nums))
    # np.savetxt('data/{}_corrcoef.txt'.format(model_string), np.asarray(corr_coefs))
    # print('mean corrcoef {}'.format(np.mean(np.asarray(corr_coefs))))
    #
    # final_drifts = np.asarray(final_drifts)
    # avg_final_drift = np.mean(final_drifts)
    # np.savetxt('results/{}_final_drifts.txt'.format(model_string), final_drifts)
    # print(errors)
    # print('neighbour_slice {} average error {:.4f}'.format(neighbour_slice, avg_error))
    # print('neighbour_slice {} average final drift {:.4f}'.format(neighbour_slice, avg_final_drift))
    #
    # print('This model is {}'.format(model_string))
    #
    # med = np.median(errors)
    # maximum = np.max(errors)
    # minimum = np.min(errors)
    # average = np.mean(errors)
    # print('Error: med {:.2f}, max {:.2f}, min {:.2f}, avg {:.2f}'.format(med, maximum, minimum, average))
    #
    # med = np.median(final_drifts)
    # maximum = np.max(final_drifts)
    # minimum = np.min(final_drifts)
    # average = np.mean(final_drifts)
    # print('Drift: med {:.2f}, max {:.2f}, min {:.2f}, avg {:.2f}'.format(med, maximum, minimum, average))







