import numpy as np
import time
import cv2
import copy
import os
import os.path as path
import imageio
from scipy.spatial.transform import Rotation as R
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import mpl_toolkits.mplot3d as plt3d
from numpy.linalg import inv
from utils import transformations as tfms
from scipy.interpolate import interp1d
import math
import random
import train_network

mask_img = cv2.imread('data/US_mask.png', 0)

frames_folder = '/home/guoh9/tmp/US_vid_frames'
pos_folder = '/home/guoh9/tmp/US_vid_pos'


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

def save_vid_1frame():
    """
    Convert the frames of video to a gif
    """
    project_folder = '/home/guoh9/tmp/US_vid_frames'
    dst_folder = '/home/guoh9/tmp/US_1frame'


    for status in ['train', 'val', 'test']:
        case_list = os.listdir(path.join(project_folder, status))
        case_list.sort()

        for case in case_list:
            case_folder = os.path.join(project_folder, status, case)
            frames_list = os.listdir(case_folder)
            frames_list.sort()
            # print(frames_list)
            # time.sleep(30)

            frame1 = cv2.imread(path.join(case_folder, frames_list[0]), 0)
            cv2.imwrite(path.join(dst_folder, '{}.jpg'.format(case)), frame1)
            print('{}.gif saved'.format(case))
    print('ALL CASES FINISHED!!!')
    time.sleep(30)


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

    """ Transform quaternion to 3 x 3 rotation matrix, get rid of unstable scipy codes"""
    # r_mat = R.from_quat(quaternion).as_matrix()
    # print('r_mat\n{}'.format(r_mat))

    new_quat = np.zeros((4,))
    new_quat[0] = quaternion[-1]
    new_quat[1:] = quaternion[:3]
    r_mat = tfms.quaternion_matrix(quaternion=new_quat)[:3, :3]
    # print('my_mat\n{}'.format(r_mat))

    trans_mat = np.zeros((4, 4))
    trans_mat[:3, :3] = r_mat
    trans_mat[:3, 3] = translation
    trans_mat[3, 3] = 1

    trans_mat = np.dot(cam_cali_mat, trans_mat)
    trans_mat = inv(trans_mat)

    return trans_mat


def params2corner_pts(params, cam_cali_mat, input_img=np.ones((224, 224)), shrink=1):
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

    corner_pts = np.asarray([[-h*(1+shrink)/2, -w*(1-shrink)/2, 0],
                             [-h*(1+shrink)/2, -w*(1+shrink)/2, 0],
                             [-h*(1-shrink)/2, -w*(1+shrink)/2, 0],
                             [-h*(1-shrink)/2, -w*(1-shrink)/2, 0]])

    corner_pts = np.concatenate((corner_pts, np.ones((4, 1))), axis=1)
    corner_pts = np.transpose(corner_pts)

    transformed_pts = []
    for frame_id in range(params.shape[0]):
        trans_mat = params_to_mat44(trans_params=params[frame_id, :],
                                    cam_cali_mat=cam_cali_mat)
        transformed_corner_pts = np.dot(trans_mat, corner_pts)
        # print('transformed_corner_pts shape {}'.format(transformed_corner_pts.shape))
        # print(transformed_corner_pts)

        dist1 = np.linalg.norm(transformed_corner_pts[:3, 0] - transformed_corner_pts[:3, 1]) * shrink
        dist2 = np.linalg.norm(transformed_corner_pts[:3, 1] - transformed_corner_pts[:3, 2]) * shrink
        scale_ratio = (dist2 / input_img.shape[0] + dist1 / input_img.shape[1]) / 2
        transformed_corner_pts = transformed_corner_pts / scale_ratio

        # dist3 = np.linalg.norm(transformed_corner_pts[:3, 2] - transformed_corner_pts[:3, 3])
        # dist4 = np.linalg.norm(transformed_corner_pts[:3, 3] - transformed_corner_pts[:3, 0])
        # print(dist1, dist2, dist3, dist4)

        transformed_corner_pts = np.moveaxis(transformed_corner_pts[:3, :], 0, 1)
        transformed_pts.append(transformed_corner_pts)
    transformed_pts = np.asarray(transformed_pts)
    return transformed_pts


def evaluate_dist(pts1, pts2, resolution=0.2):
    """
    points input formats are frame_num x 4 (corner_points) x 3 (xyz)
    :param pts1:
    :param pts2:
    :param resolution:
    :return: The average Euclidean distance between all points pairs, times 0.2 is mm
    """
    error = np.square(pts1 - pts2)
    error = np.sum(error, axis=2)
    error = np.sqrt(error)
    error = np.mean(error) * resolution
    return error


def final_drift(pts1, pts2, resolution=0.2):
    # print(pts1.shape, pts2.shape)
    center_pt1 = np.mean(pts1, axis=0)
    center_pt2 = np.mean(pts2, axis=0)
    dist = np.linalg.norm(center_pt1 - center_pt2) * resolution
    return dist



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
        self.case_id = case_id
        if 1 <= self.case_id <= 71:
            self.data_part = 'test'
        elif 71 < self.case_id <= 140:
            self.data_part = 'val'
        elif 140 < self.case_id <= 747:
            self.data_part = 'train'
        self.case_frames_path = path.join(frames_folder, self.data_part,
                                          'Case{:04}'.format(self.case_id))
        self.frames_list = os.listdir(self.case_frames_path)
        self.frames_list.sort()

        self.cam_cali_mat = np.loadtxt('/zion/common/data/uronav_data/Case{:04}/'
                                       'Case{:04}_USCalib.txt'.format(self.case_id, self.case_id))

        case_pos_path = path.join(pos_folder, 'Case{:04}.txt'.format(self.case_id))
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
            :param frame_color: color of the initial frame, default to be blue
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
            print('transformed_corner_pts shape {}'.format(transformed_corner_pts.shape))
            time.sleep(30)
            # dst = np.linalg.norm(transformed_corner_pts[:, 0] - transformed_corner_pts[:, 2])
            # print(dst)

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


        for frame_id in range(self.frames_num):
            frame_pos = self.case_pos[frame_id, :]
            frame_color = tuple(self.colors[frame_id, :])
            frame_img = cv2.imread(path.join(self.case_frames_path, '{:04}.jpg'.format(frame_id)), 0)
            frame_img = train_network.data_transform(frame_img)
            plot_frame3d(trans_params=frame_pos, frame_color=frame_color,
                         input_img=frame_img, plot_img=True)
            print('{} frame'.format(frame_id))
        plt.show()


class DofPlot():
    def __init__(self, case_id):
        super(DofPlot, self).__init__()
        self.case_id = case_id
        self.case_frames_path = path.join(frames_folder, 'Case{:04}'.format(self.case_id))
        self.frames_list = os.listdir(self.case_frames_path)
        self.frames_list.sort()

        self.cam_cali_mat = np.loadtxt('/zion/common/data/uronav_data/Case{:04}/'
                                       'Case{:04}_USCalib.txt'.format(self.case_id, self.case_id))

        case_pos_path = path.join(pos_folder, 'Case{:04}.txt'.format(self.case_id))
        self.case_pos = np.loadtxt(case_pos_path)
        print('frames_list {}, case_pos {}'.format(len(self.frames_list), self.case_pos.shape))

        self.frames_num = self.case_pos.shape[0]

        def plot_dof():
            plt.figure()
            colors = ['lightcoral', 'darkorange', 'palegreen',
                      'aqua', 'royalblue', 'violet']
            names = ['tX', 'tY', 'tZ', 'rX', 'rY', 'rZ']
            for dof_id in range(0, self.extracted_dof.shape[1]):
                plt.plot(self.extracted_dof[:, dof_id], color=colors[dof_id],
                         label=names[dof_id])
            plt.legend(loc='upper left')
            # plt.show()
            plot_path = 'figures/dofs/Case{:04}.jpg'.format(self.case_id)
            plt.savefig(plot_path)


        extracted_dof = []
        for frame_id in range(1, self.frames_num):
            this_params = self.case_pos[frame_id, :]
            this_dof = get_6dof_label(trans_params1=self.case_pos[0, :],
                                      trans_params2=this_params,
                                      cam_cali_mat=self.cam_cali_mat,
                                      use_euler=False)
            extracted_dof.append(this_dof)
        self.extracted_dof = np.asarray(extracted_dof)
        plot_dof()
        print('extracted_dof shape {}'.format(self.extracted_dof.shape))

        # for frame_id in range(self.frames_num):
        #     frame_pos = self.case_pos[frame_id, :]
        #     frame_color = tuple(self.colors[frame_id, :])
        #     frame_img = cv2.imread(path.join(self.case_frames_path, '{:04}.jpg'.format(frame_id)), 0)
        #     plot_frame3d(trans_params=frame_pos, frame_color=frame_color,
        #                  input_img=frame_img, plot_img=False)
        #     print('{} frame'.format(frame_id))
        # plt.show()


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
    The angles are in degrees, not euler!
    """
    trans_mat1 = params_to_mat44(trans_params1, cam_cali_mat=cam_cali_mat)
    trans_mat2 = params_to_mat44(trans_params2, cam_cali_mat=cam_cali_mat)

    relative_mat = np.dot(trans_mat2, inv(trans_mat1))

    translations = relative_mat[:3, 3]
    rotations_eulers = np.asarray(tfms.euler_from_matrix(relative_mat))

    rotations_degrees = (rotations_eulers / (2 * math.pi)) * 360

    dof = np.concatenate((translations, rotations_degrees), axis=0)

    return dof


def get_next_pos(trans_params1, dof, cam_cali_mat):
    """
    Given the first frame's Aurora position line and relative 6dof, return second frame's position line
    :param trans_params1: Aurora position line of the first frame
    :param dof: 6 degrees of freedom based on the first frame, rotations should be degrees
    :param cam_cali_mat: Camera calibration matrix of this case
    :return: Aurora position line of the second frame
    """
    trans_mat1 = params_to_mat44(trans_params1, cam_cali_mat=cam_cali_mat)

    """ Transfer degrees to euler """
    dof[3:] = dof[3:] * (2 * math.pi) / 360

    rot_mat = tfms.euler_matrix(dof[5], dof[4], dof[3], 'rzyx')[:3, :3]

    relative_mat = np.identity(4)
    relative_mat[:3, :3] = rot_mat
    relative_mat[:3, 3] = dof[:3]

    next_mat = np.dot(inv(cam_cali_mat), inv(np.dot(relative_mat, trans_mat1)))
    quaternions = tfms.quaternion_from_matrix(next_mat)  # wxyz

    next_params = np.zeros(7)
    next_params[:3] = next_mat[:3, 3]
    next_params[3:6] = quaternions[1:]
    next_params[6] = quaternions[0]
    return next_params


def smooth_array(input_array1d, smooth_deg=10):
    ori_x = np.linspace(0, input_array1d.shape[0]-1, input_array1d.shape[0])
    print('ori_x\n{}'.format(ori_x))
    print('ori_x shape {}'.format(ori_x.shape))
    ori_y = input_array1d

    p = np.polyfit(ori_x, ori_y, deg=smooth_deg)
    f = np.poly1d(p)

    smoothed = f(ori_x)
    # print('input_array1d\n{}'.format(input_array1d))
    # print('smoothed\n{}'.format(smoothed))
    # time.sleep(30)
    return smoothed


def sample_ids(slice_num, neighbour_num, sample_option='skip', random_reverse_prob=0,
               self_prob=0):
    """
    This function gives different sampling strategies.
    :param slice_num: Number of total slices of a case
    :param neighbour_num: How many slices to serve as one input
    :param sample_option: skip range or normally consecutive
    :param random_reverse_prob: probability of applying random reverse, 0 to be no random reverse
    :return:
    """
    skip_ratio = 3

    if sample_option in ['skip', 'skip_random'] and neighbour_num * skip_ratio > slice_num:
        sample_option = 'normal'

    if sample_option == 'skip':
        start_range = slice_num - skip_ratio * neighbour_num
        if start_range == 0:
            start_id = 0
        else:
            start_id = np.random.randint(0, start_range, 1)[0]
        end_id = start_id + skip_ratio * neighbour_num - 1
        range = np.linspace(start_id, end_id, skip_ratio * neighbour_num)
        np.random.shuffle(range)
        select_ids = np.sort(range[:neighbour_num])
    elif sample_option == 'skip_random':
        ''' ending sample ID is randomly chose from latter half '''
        ''' This function creates more varieties of sampling range'''
        start_range = slice_num - skip_ratio * neighbour_num
        if start_range == 0:
            start_id = 0
        else:
            start_id = np.random.randint(0, start_range, 1)[0]
        end_id = start_id + skip_ratio * neighbour_num - 1
        central_id = int((start_id + end_id) / 2)

        sample_end_id_pool = np.linspace(central_id, end_id, end_id - central_id + 1)
        sample_end_id = int(np.random.choice(sample_end_id_pool, 1)[0])

        sample_ratio = np.linspace(0, 1, neighbour_num)
        select_ids = (sample_ratio * (sample_end_id - start_id) + start_id).astype(np.uint64)
    elif sample_option == 'skip_random_fixed':
        start_range = slice_num - skip_ratio * neighbour_num
        if start_range == 0:
            start_id = 0
        else:
            start_id = np.random.randint(0, start_range, 1)[0]
        frame_gap_choices = [0, 1, 2, 3]
        frame_gap_probs = [0, 1, 0, 0]
        frame_gap_random = np.random.choice(frame_gap_choices, 1, p=frame_gap_probs)[0]
        select_ids = np.linspace(start=start_id,
                                 stop=start_id + (neighbour_num - 1) * frame_gap_random,
                                 num=neighbour_num, endpoint=True)
        # print(frame_gap_random)
        # print(select_ids)
        # time.sleep(30)
    else:
        start_range = slice_num - neighbour_num
        if start_range == 0:
            start_id = 0
        else:
            start_id = np.random.randint(0, start_range, 1)[0]
        select_ids = np.linspace(start_id, start_id + neighbour_num - 1, neighbour_num)

    if random.uniform(0, 1) < random_reverse_prob:
        select_ids = np.flip(select_ids)

    if random.uniform(0, 1) < self_prob:
        ''' input the same slice for NS times '''
        slice_id = random.randint(0, slice_num-1)
        select_ids = slice_id * np.ones((neighbour_num,))
        # print(select_ids)

    select_ids = select_ids.astype(np.int64)
    # print('selected ids {}'.format(select_ids))
    # select_ids = np.random.shuffle(select_ids)
    # print('shuffled selected ids {}'.format(select_ids))
    return select_ids


def clean_ids():
    """
    Eliminate weird BK scans from all three portions
    """
    project_folder = '/home/guoh9/tmp/US_vid_frames'
    bk_ids = np.loadtxt('infos/BK.txt')
    clean_case_ids = {'train': [], 'val': [], 'test': []}

    for status in ['train', 'val', 'test']:
        case_list = os.listdir(path.join(project_folder, status))
        case_list.sort()

        for case in case_list:
            case_id = int(case[-4:])
            if case_id not in bk_ids:
                clean_case_ids[status].append(case_id)

        np_id = np.asarray(clean_case_ids[status]).astype(np.int64)
        np.savetxt('infos/{}_ids.txt'.format(status), np_id)

    print('clean cases ids finished')
    time.sleep(30)


def test_avg_dof():
    case_id = 10
    case_pos_np = np.loadtxt('/home/guoh9/tmp/US_vid_pos/Case{:04}.txt'.format(case_id))
    case_calib_mat = np.loadtxt('/zion/common/data/uronav_data/Case{:04}/'
                                'Case{:04}_USCalib.txt'.format(case_id, case_id))

    start_id = 0
    ns = 10

    # print(np.around(case_pos_np, decimals=3))

    all_labels = []
    for id in range(start_id, start_id + ns - 1):
        # print('id {}'.format(id))
        pos1 = case_pos_np[id, :]
        pos2 = case_pos_np[id + 1, :]
        print('{}, {}'.format(id, id + 1))
        label = get_6dof_label(trans_params1=pos1, trans_params2=pos2,
                               cam_cali_mat=case_calib_mat)
        all_labels.append(label)
    all_labels = np.asarray(all_labels)
    print('all_labels shape {}'.format(all_labels.shape))

    sum_labels = np.sum(all_labels, axis=0)

    pos_start = case_pos_np[start_id, :]
    pos_end = case_pos_np[start_id + ns - 1, :]
    print('pos_start\n{}'.format(pos_start))
    print('pos_end\n{}'.format(pos_end))

    pos_end_recon = get_next_pos(trans_params1=pos_start, dof=sum_labels,
                                 cam_cali_mat=case_calib_mat)
    print('pos_end_recon\n{}'.format(pos_end_recon))

    time.sleep(30)
    pos1 = case_pos_np[1, 2:]
    pos2 = case_pos_np[10, 2:]
    label = get_6dof_label(trans_params1=pos1, trans_params2=pos2,
                           cam_cali_mat=case_calib_mat)
    recon_params = get_next_pos(trans_params1=pos1, dof=label,
                                cam_cali_mat=case_calib_mat)
    print('labels\n{}'.format(label))
    print('params2\n{}'.format(pos2))
    print('recon_params\n{}'.format(recon_params))


def center_crop():
    folder = '/home/guoh9/tmp/US_vid_frames/train/Case0347'
    frame_list = os.listdir(folder)
    frame_list.sort()

    for i in frame_list:
        frame_path = path.join(folder, i)
        frame_img = cv2.imread(frame_path, 0)
        crop = train_network.data_transform(input_img=frame_img)
        cv2.imwrite('data/crops/{}.jpg'.format(i), crop)
    print('finished')
    time.sleep(30)


def produce_Aurora(case_id):
    original_pos_path = '/zion/common/shared/uronav_data/test/Case{:04}/Case{:04}_Aurora.pos'.format(case_id, case_id)
    results_pos_path = 'results/pos/Case{:04}_Aurora_result.pos'.format(case_id)
    results_pos = np.loadtxt(results_pos_path)

    # if results_pos.shape[1] == 7:
    #     results_pos = np.concatenate((np.zeros((results_pos.shape[0], 2)), results_pos), axis=1)

    file = open(original_pos_path, 'r')
    file_dst = open('results/results_pos/Case{:04}_Aurora_results.pos'.format(case_id), 'a+')

    lines = file.readlines()
    file_dst.write('{}'.format(lines[0]))

    pos_np = []
    for line_index in range(1, len(lines) - 1):  # exclude the first line and last line
        result_index = line_index - 1
        line = lines[line_index]
        values = line.split()
        values_np = np.asarray(values[1:]).astype(np.float32)
        pos_np.append(values_np)

        for fixed_id in range(3):
            file_dst.write('{} '.format(int(values[fixed_id])))

        for params_id in range(results_pos.shape[1]):
            file_dst.write('{:.6f} '.format(results_pos[result_index, params_id]))
        file_dst.write('\n')
    pos_np = np.asarray(pos_np)
    file_dst.write('{}'.format(lines[-1]))
    file_dst.close()
    print('pos_np.shape {}'.format(pos_np.shape))
    print('results_pos.shape {}'.format(results_pos.shape))
    # time.sleep(30)


def evaluate_correlation(dof1, dof2, abs=False):
    # print(dof1.shape, dof2.shape)
    corrs = []
    for dof_id in range(dof1.shape[1]):
        this_dof1 = dof1[:, dof_id]
        this_dof2 = dof2[:, dof_id]

        cor_coe = np.corrcoef(this_dof1, this_dof2)
        corrs.append(cor_coe[0, 1])
    if abs:
        corr_result = np.mean(np.abs(np.asarray(corrs)))
    else:
        corr_result = np.mean(np.asarray(corrs))
    # time.sleep(30)
    return corr_result


def visualize_attention(case_id, batch_ids, batch_imgs, maps, weights):
    batch_imgs = batch_imgs.data.cpu().numpy()
    maps = maps.data.cpu().numpy()
    print(case_id)
    print(batch_ids)
    print(batch_imgs.shape)
    print(maps.shape)
    print(weights.shape)

    dofs = ['tX', 'tY', 'tZ', 'aX', 'aY', 'aZ']

    for batch_loop in range(len(batch_ids)):
        frame_id = batch_ids[batch_loop]
        frame_map = maps[batch_loop, :, 0, :, :]
        frame_img = batch_imgs[batch_loop, 0, 0, :, :]
        frame_img2 = batch_imgs[batch_loop, 0, -1, :, :]
        diff_img = frame_img2 - frame_img
        # print('frame_id {}, frame_map {}'.format(frame_id, frame_map.shape))

        # dof_atmaps = []
        for dof_id in range(weights.shape[0]):
            dof_weight = weights[dof_id, :]
            dof_weight = np.expand_dims(dof_weight, 1)
            dof_weight = np.expand_dims(dof_weight, 1)

            dof_map = dof_weight * frame_map
            dof_map = np.sum(dof_map, axis=0)
            dof_map = cv2.resize(dof_map, (frame_img.shape[0], frame_img.shape[1]),
                                 interpolation=cv2.INTER_LINEAR)
            # print(dof_weight.shape)
            # print(dof_map.shape)
            plt.imsave('maps/{}_{}_{}.jpg'.format(case_id, frame_id, dofs[dof_id]),
                       dof_map, cmap='jet_r')
            plt.imsave('maps/{}_{}_ad.jpg'.format(case_id, frame_id),
                       diff_img, cmap='jet')
            cv2.imwrite('maps/{}_{}.jpg'.format(case_id, frame_id), frame_img)
            print('Saved {}_{}_{}.jpg'.format(case_id, frame_id, dofs[dof_id]))
            # time.sleep(30)
            # plt.figure()
            # plt.imshow(dof_map, )


            # dof_atmaps.append(dof_map)
    print('batch saved')
    # time.sleep(30)



if __name__ == '__main__':
    # clean_ids()
    #
    # test_ids = np.asarray([8, 12, 15, 43, 54, 55])
    # for id in test_ids:
    #     produce_Aurora(case_id=id)
    #     print('{} finished'.format(id))
    # time.sleep(30)

    # center_crop()
    # test_avg_dof()
    # save_vid_1frame()
    # aurora_path = '/zion/common/data/uronav_data/Case0001/Case0001_Aurora.pos'
    # pos = read_aurora(file_path=aurora_path)
    # print(pos.shape)
    # print(pos)

    # test_us_img = cv2.imread('/zion/guoh9/projects/USFreehandRecon/data/frames/0065.jpg', 0)
    # mask_fan(input_img=test_us_img)
    # save_all_aurora_pos()

    # frame_img = cv2.imread('/home/guoh9/tmp/US_vid_frames/Case0001/0000.jpg', 0)
    # pos = np.loadtxt('/home/guoh9/tmp/US_vid_pos/Case0001.txt')
    # frame_pos = pos[0, :]
    # frame_pos = np.zeros((7,))
    # frame_pos[6] = 1
    # # plot_2d_in_3d(trans_params=frame_pos, frame_color='b', input_img=frame_img)
    #
    # plot_2d_in_3d_test(trans_params1=pos[0, :],
    #                    trans_params2=pos[10, :])

    # visualize_frames(case_id=1)
    # sample_ids(slice_num=78, neighbour_num=10)
    case = VisualizeSequence(case_id=9)
    # case = DofPlot(case_id=370)
    # for i in range(71):
    #     case_plot = DofPlot(case_id=i+1)
    # time.sleep(30)

    case_id = 7
    case_pos_np = np.loadtxt('/home/guoh9/tmp/US_vid_pos/Case{:04}.txt'.format(case_id))
    case_calib_mat = np.loadtxt('/zion/common/data/uronav_data/Case{:04}/'
                                'Case{:04}_USCalib.txt'.format(case_id, case_id))
    pos1 = case_pos_np[1, 2:]
    pos2 = case_pos_np[10, 2:]
    label = get_6dof_label(trans_params1=pos1, trans_params2=pos2,
                           cam_cali_mat=case_calib_mat)
    recon_params = get_next_pos(trans_params1=pos1, dof=label,
                                cam_cali_mat=case_calib_mat)
    print('labels\n{}'.format(label))
    print('params2\n{}'.format(pos2))
    print('recon_params\n{}'.format(recon_params))




