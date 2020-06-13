#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 10:33:28 2017

@author: yan

Evaluate registrations against the "ground truth" using markers
"""

# %%

from utils import mhd_utils as mu
import numpy as np
from numpy import linalg
from os import path
from stl import mesh

# %% Class

class RegistrationEvaluator:

    # %
    def __init__(self, case_path, quiet=True):
        """
        """
        if not quiet:
            print('\n{}'.format('-' * 80))
            print('Data from <{}>'.format(case_path))
        mat_gt = self.load_gt_registration(case_path)

        # print(mat_us2mr)
        fn_stl = 'segmentationrtss.uronav.stl'
        fn_stl_full = path.join(case_path, fn_stl)
        segMesh = mesh.Mesh.from_file(fn_stl_full)
        if not quiet:
            print('Segmentation loaded from <{}>'.format(fn_stl))

        num_triangle = segMesh.points.shape[0]
        self.markers = np.reshape(segMesh.points, (num_triangle * 3, 3))

        fn_mr = 'MRVol.mhd'
        header = mu.read_meta_header(path.join(case_path, fn_mr))
        if not quiet:
            print('MR header loaded from <{}>'.format(fn_mr))

        offset = header['Offset']

        self.markers -= offset

        self.markers_transformed_gt = self.transform_points3d(linalg.inv(mat_gt),
                                                              self.markers)
        if not quiet:
            print('{}\n'.format('-' * 80))

    # %
    def load_UroNav_registration(self, fn_reg_UroNav, fn_mhd):
        """Load UroNav registration matrix from 'coreg.txt'
        """
        # print('*'*100)
        # print('fn_reg_UroNav: {}'.format(fn_reg_UroNav))
        mat_reg = np.loadtxt(fn_reg_UroNav)
        header = mu.read_meta_header(fn_mhd)
        offset = np.asarray(header['Offset'])

        mat_mr2us = np.identity(4)
        mat_mr2us[:3, :3] = mat_reg[1:, 1:]
        mat_mr2us[:3, 3] = mat_reg[1:, 0]
        mat_us2mr_UroNav = linalg.inv(mat_mr2us)
        # print(linalg.inv(mat_us2mr))

        mat_shift = np.identity(4)
        mat_shift[:3, 3] = - offset

        mat_us2mr = mat_shift.dot(mat_us2mr_UroNav)

        return mat_us2mr

    # %
    def computeTRE(self, pointSet0, pointSet1):
        """
        """
        diff = pointSet0[:, :3] - pointSet1[:, :3]
        return np.mean(linalg.norm(diff, axis=1))

    # %
    def createTranslationMatrix(self, t3d):
        """
        """
        mat = np.identity(4)
        mat[:3, 3] = np.asarray(t3d)
        return mat

    # %
    def load_gt_registration(self, folder_path):

        fn_reg = 'coreg.txt'
        fn_reg_refined = 'coreg_refined.txt'

        # By default, load the refined registration
        fn_reg_full = path.join(folder_path, fn_reg_refined)

        if not path.isfile(fn_reg_full):
            fn_reg_full = path.join(folder_path, fn_reg)

        # idx_case = folder_path.find('Case')
        # print('Loading <{}> for {}'.format(
        #        path.basename(fn_reg_full),
        #        folder_path[idx_case:idx_case+8]))
        self.gt_registration = self.load_registration(fn_reg_full)
        # print('gt_registration {}'.format(self.gt_registration))
        return self.gt_registration

    # ----- #
    def get_gt_registration(self):
        return self.gt_registration

    def compute_euclidean_dist(self, affine1, affine2):

        markers_transformed1 = self.transform_points3d(linalg.inv(affine1),
                                                       self.markers)

        markers_transformed2 = self.transform_points3d(linalg.inv(affine2),
                                                       self.markers)

        return self.computeTRE(markers_transformed1, markers_transformed2)

    # %
    def load_registration(self, filename):
        if filename.endswith('coreg.txt'):
            fn_mr_full = path.join(path.dirname(filename), 'MRVol.mhd')
            mat_us2mr = self.load_UroNav_registration(filename, fn_mr_full)
        else:
            mat_us2mr = np.loadtxt(filename)

        return mat_us2mr

    # %
    def transform_points3d(self, trans3d, points3d):
        """
        """
        points_tp = np.ones((4, points3d.shape[0]))
        points_tp[:3, :] = np.transpose(points3d)
        # print(points_tp)

        points_transformed = trans3d.dot(points_tp)
        points_transformed = np.transpose(points_transformed)

        return points_transformed[:, :3]

    # %
    def evaluate_from_file(self, filename_reg):
        mat_trans = self.load_registration(filename_reg)
        markers_transformed1 = self.transform_points3d(linalg.inv(mat_trans),
                                                       self.markers)

        TRE = self.computeTRE(self.markers_transformed_gt, markers_transformed1)
        return TRE

    # %
    def evaluate_transform(self, mat_trans):
        # print(mat_trans)
        # print(self.markers.shape)
        markers_transformed1 = self.transform_points3d(linalg.inv(mat_trans),
                                                       self.markers)

        TRE = self.computeTRE(self.markers_transformed_gt, markers_transformed1)
        return TRE


# %% Load ground truth registration


if __name__ == '__main__':
    np.set_printoptions(suppress=True)

    caseNo = 335

    # folder_root = '/home/yan/Dropbox/Data/UroNav_registration_data'
    folder_root = '/home/data/uronav_data'
    folder_case = 'Case{:04d}'.format(caseNo)
    full_case = path.join(folder_root, folder_case)

    train_folder_root = '/home/yan/Dropbox/temp/training'

    evaluator = RegistrationEvaluator(full_case)

    print(evaluator.get_gt_registration())

    #    for i in range(20):
    #        fn_reg = path.join(train_folder_root, folder_case, 't{:03d}.txt'.format(i))
    #        regerr = evaluator.evaluate(fn_reg)
    #        print('Surface registration error --> {:.4f} mm'.format(regerr))

    tf = [[0.99958654, 0.01898075, -0.0215982, 65.71091839],
          [-0.01895731, 0.99981946, 0.00128912, 81.38942109],
          [0.02161877, -0.00087915, 0.9997659, 46.83255436],
          [0., 0., 0., 1.]]
    regerr = evaluator.evaluate_transform(tf)
    print('Surface registration error --> {:.4f} mm'.format(regerr))