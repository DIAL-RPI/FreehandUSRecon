#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generator to yield resampled volume data for training and validation
"""

# %%

from keras.models import load_model, Model
from matplotlib import pyplot as plt
import numpy as np
import os
from os import path

import random
import SimpleITK as sitk
from stl import mesh

from utils import data_loading_funcs as dlf
from utils import mhd_utils as mu
from utils import reg_evaluator as regev
from utils import volume_resampler_3d as vr
import tensorflow as tf
from utils import registration_reader as rr
import scipy
#from augment_data import augment



# %%


class VolumeDataGenerator(object):
    """Generate volume image for training or validation
    
    #Arguments
    """
    
    def __init__(self,
                 data_folder,
                 case_num_range,
                 case_num_range_2=None,
                 max_registration_error = 20.0):

        self.data_folder = data_folder
        
        cases = []
        
        # Go through all the case
        for caseIdx in range(case_num_range[0], case_num_range[1]+1):
            caseFolder = 'Case{:04d}'.format(caseIdx)
            full_case = path.join(data_folder, caseFolder)
        
            if not path.isdir(full_case):
                continue
            else:
                cases.append(caseIdx)
        
        if case_num_range_2 != None:
            for caseIdx in range(case_num_range_2[0], case_num_range_2[1]+1):
                caseFolder = 'Case{:04d}'.format(caseIdx)
                full_case = path.join(data_folder, caseFolder)
    
                if not path.isdir(full_case):
                    continue
                else:
                    cases.append(caseIdx)
            

        self.good_cases = np.asarray(cases, dtype=np.int32)
        self.num_cases = self.good_cases.size
        
        random.seed()
        
        self.e_t = 0.5
        self.e_rot = 1
        
        self.isMultiGauss = False
        
        
        self.max_error = max_registration_error
        print('VolumeDataGenerator: max_registration_error = {}'.format(self.max_error))
        
        #self.width, self.height, self.depth = 96, 96, 32
    
    
    # ----- #
    
    def get_sample_multi_gauss(self,mean,cov):
        
        return np.random.multivariate_normal(mean,cov)
        
    def get_num_cases(self):
        return self.num_cases
        
        
    # ----- #
    def _get_random_value(self, r, center, hasSign):
		
        randNumber = random.random() * r + center
		
        if hasSign:
            sign = random.random() > 0.5
            if sign == False:
                randNumber *= -1
		
        return randNumber
    
    
    # ----- #
    def get_array_from_itk_matrix(self, itk_mat):
        mat = np.reshape(np.asarray(itk_mat), (3,3))
        return mat
    
        
    # ----- #
    def generate(self, shuffle=True, shape=(96,96,96)):
        """
        """
        currentIdx = 0
        np.random.seed()
        
        (width, height, depth) = shape
        
        print('Shuffle = {}'.format(shuffle))
        
        while True:
            idx = currentIdx % self.num_cases
            currentIdx += 1
            
            # Shuffle cases
            if idx == 0:
                if shuffle:
                    case_array = np.random.permutation(self.good_cases)
                else:
                    case_array = self.good_cases
            
            case_no = case_array[idx]
            sampledFixed, sampledMoving, err, params = self.create_sample(case_no, shape)
            #sampledFixed, sampledMoving, pos_neg, err, params = self.create_sample(450, shape)
            print('Sample generated frome Case{:04d}'.format(case_no))
            
            # Put into 4D array
            sample4D = np.zeros((depth, height, width, 2), dtype=np.ubyte)
            sample4D[:,:,:,0] = sitk.GetArrayFromImage(sampledFixed)
            sample4D[:,:,:,1] = sitk.GetArrayFromImage(sampledMoving)
            
            yield sample4D, err, params
            

    # ----- #
    def generate_batch(self, batch_size=32, shape=(96,96,32)):
        """Used for keras training and validation
        """
        batch_index = 0
        np.random.seed()
        
        (width, height, depth) = shape
        
        while True:
            # Shuffle cases
            if batch_index == 0:
                case_array = np.random.permutation(self.good_cases)

            #current_index = (batch_index * batch_size) % self.num_cases
            current_index = batch_index * batch_size
            
            if (current_index + batch_size) < self.num_cases:
                current_batch_size = batch_size
                batch_index += 1
            else:
                # handle special case where only 1 sample left for the batch
                if (self.num_cases - current_index) > 1:
                    current_batch_size = self.num_cases - current_index
                else:
                    current_batch_size = 2
                    current_index -= 1
                batch_index = 0
            
            batch_samples = np.zeros((current_batch_size, depth, height, width, 2), dtype=np.ubyte)
            #batch_labels = []
            batch_errors = []
            batch_params = []
                
            for k in range(current_batch_size):
                case_no = case_array[k + current_index]
                #print(case_no)
                sampledFixed, sampledMoving, err, params = self.create_sample(case_no, shape)
                
                # Put into 4D array
                sample4D = np.zeros((depth, height, width, 2), dtype=np.ubyte)
                sample4D[:,:,:,0] = sitk.GetArrayFromImage(sampledFixed)
                sample4D[:,:,:,1] = sitk.GetArrayFromImage(sampledMoving)
                
                batch_samples[k, :,:,:,:] = sample4D
                #batch_labels.append(pos_neg)
                batch_errors.append([err])
                batch_params.append(params)
            
            #yield (batch_samples, [np.asarray(batch_errors), np.asarray(batch_params)])
            yield (batch_samples, np.asarray(batch_params))
            #yield (batch_samples, np.asarray(batch_errors))
            
    def generate_batch_classification(self, batch_size=32, shape=(96,96,32)):
        """Used for keras training and validation
        """
        batch_index = 0
        np.random.seed()
        
        (width, height, depth) = shape
        
        while True:
            # Shuffle cases
            if batch_index == 0:
                case_array = np.random.permutation(self.good_cases)

            #current_index = (batch_index * batch_size) % self.num_cases
            current_index = batch_index * batch_size
            
            if (current_index + batch_size) < self.num_cases:
                current_batch_size = batch_size
                batch_index += 1
            else:
                # handle special case where only 1 sample left for the batch
                if (self.num_cases - current_index) > 1:
                    current_batch_size = self.num_cases - current_index
                else:
                    current_batch_size = 2
                    current_index -= 1
                batch_index = 0
            
            batch_samples = np.zeros((current_batch_size, depth, height, width, 4), dtype=np.ubyte)
            #batch_labels = []
            batch_labels = []
            batch_errs = []
                
            for k in range(current_batch_size):
                case_no = case_array[k + current_index]
                #print(case_no)
                sampledFixed_i, sampledFixed_f, sampledMoving_i, sampledMoving_f, label, err1, err2 = self.create_sample_classification(case_no, shape)
                
                # Put into 4D array
                sample4D = np.zeros((depth, height, width, 4), dtype=np.ubyte)
                sample4D[:,:,:,0] = sitk.GetArrayFromImage(sampledFixed_i)
                sample4D[:,:,:,1] = sitk.GetArrayFromImage(sampledMoving_i)
                sample4D[:,:,:,2] = sitk.GetArrayFromImage(sampledFixed_f)
                sample4D[:,:,:,3] = sitk.GetArrayFromImage(sampledMoving_f)
                
                batch_samples[k, :,:,:,:] = sample4D
                #batch_labels.append(pos_neg)
                batch_labels.append(label)
                batch_errs.append([err1, err2])
            
            yield (batch_samples, [np.asarray(batch_labels), np.asarray(batch_errs)])
            
    def generate_batch_NIH(self, batch_size=32, shape=(96,96,32)):
        """Used for keras training and validation
        """
        batch_index = 0
        np.random.seed()
        
        (width, height, depth) = shape
        
        while True:
            # Shuffle cases
            if batch_index == 0:
                case_array = np.random.permutation(self.good_cases)

            #current_index = (batch_index * batch_size) % self.num_cases
            current_index = batch_index * batch_size
            
            if (current_index + batch_size) < self.num_cases:
                current_batch_size = batch_size
                batch_index += 1
            else:
                # handle special case where only 1 sample left for the batch
                if (self.num_cases - current_index) > 1:
                    current_batch_size = self.num_cases - current_index
                else:
                    current_batch_size = 2
                    current_index -= 1
                batch_index = 0
            
            batch_samples = np.zeros((current_batch_size, depth, height, width, 2), dtype=np.ubyte)
            #batch_labels = []
            batch_errors = []
            batch_params = []
            batch_segs = []
            batch_trans = []
            batch_case_nums = []
                
            for k in range(current_batch_size):
                case_no = case_array[k + current_index]
                #print(case_no)
                sampledFixed, sampledMoving, err, params, segMesh, trans = self.create_sample_NIH(case_no, shape)
                # Put into 4D array
                sample4D = np.zeros((depth, height, width, 2), dtype=np.ubyte)
                sample4D[:,:,:,0] = sitk.GetArrayFromImage(sampledFixed)
                sample4D[:,:,:,1] = sitk.GetArrayFromImage(sampledMoving)
                
                batch_samples[k, :,:,:,:] = sample4D
                #batch_labels.append(pos_neg)
                batch_errors.append([err])
                batch_params.append(params)
                batch_segs.append(segMesh)
                batch_trans.append(trans)
                batch_case_nums.append(case_no)
            
            yield (batch_samples, batch_params)
            
    def generate_batch_NIH_transform_prediction(self, batch_size=32, shape=(96,96,32)):
        """Used for keras training and validation
        """
        batch_index = 0
        np.random.seed()
        
        (width, height, depth) = shape
        
        while True:
            # Shuffle cases
            if batch_index == 0:
                case_array = np.random.permutation(self.good_cases)

            #current_index = (batch_index * batch_size) % self.num_cases
            current_index = batch_index * batch_size
            
            if (current_index + batch_size) < self.num_cases:
                current_batch_size = batch_size
                batch_index += 1
            else:
                # handle special case where only 1 sample left for the batch
                if (self.num_cases - current_index) > 1:
                    current_batch_size = self.num_cases - current_index
                else:
                    current_batch_size = 2
                    current_index -= 1
                batch_index = 0
            
            batch_samples = np.zeros((current_batch_size, depth, height, width, 2), dtype=np.ubyte)
            #batch_labels = []
            batch_transforms = []
                
            for k in range(current_batch_size):
                case_no = case_array[k + current_index]
                #print(case_no)
                sampledFixed, sampledMoving, err, params = self.create_sample(case_no, shape)
                # Put into 4D array
                sample4D = np.zeros((depth, height, width, 2), dtype=np.ubyte)
                sample4D[:,:,:,0] = sitk.GetArrayFromImage(sampledFixed)
                sample4D[:,:,:,1] = sitk.GetArrayFromImage(sampledMoving)
                
                batch_samples[k, :,:,:,:] = sample4D
                #batch_labels.append(pos_neg)
                batch_transforms.append(params)
                #batch_errors.append([err])
            
            yield (batch_samples, batch_transforms)
            
    def generate_batch_NIH_transform_prediction_2D_multiview(self, batch_size=32, shape=(224,222,220)):
        """Used for keras training and validation
        """
        batch_index = 0
        np.random.seed()
        
        slice_num = 3
        
        (width, height, depth) = shape
        
        while True:
            # Shuffle cases
            if batch_index == 0:
                case_array = np.random.permutation(self.good_cases)

            #current_index = (batch_index * batch_size) % self.num_cases
            current_index = batch_index * batch_size
            
            if (current_index + batch_size) < self.num_cases:
                current_batch_size = batch_size
                batch_index += 1
            else:
                # handle special case where only 1 sample left for the batch
                if (self.num_cases - current_index) > 1:
                    current_batch_size = self.num_cases - current_index
                else:
                    current_batch_size = 2
                    current_index -= 1
                batch_index = 0
            
            ax_batch_samples = np.zeros((current_batch_size, height, width, 2, slice_num), dtype=np.ubyte)
            sag_batch_samples = np.zeros((current_batch_size, depth, height, 2, slice_num), dtype=np.ubyte)
            cor_batch_samples = np.zeros((current_batch_size, depth, width, 2, slice_num), dtype=np.ubyte)
            #batch_labels = []
            batch_transforms = []
            ax_transforms = []
            sag_transforms = []
            cor_transforms = []
            
            batch_errors = []
            
            batch_segs = []
            
            batch_affines = []
            
            batch_tX = []
            batch_tY = []
            batch_tZ = []
            batch_rotX = []
            batch_rotY = []
            batch_rotZ = []
                
            for k in range(current_batch_size):
                case_no = case_array[k + current_index]
                #print(case_no)
                sampledFixed, sampledMoving, err, params = self.create_sample(case_no, shape)
                # Put into 4D array
                ax_sample = np.zeros((height, width, 2, slice_num), dtype=np.ubyte)
                sag_sample = np.zeros((depth, height, 2, slice_num), dtype=np.ubyte)
                cor_sample = np.zeros((depth, width, 2, slice_num), dtype=np.ubyte)

                MR = sitk.GetArrayFromImage(sampledFixed)
                TRUS = sitk.GetArrayFromImage(sampledMoving)
                
                ax_sample[:,:,0,:] = np.reshape(MR[int(depth/2)-int((slice_num-1)/2):int(depth/2)+int((slice_num)/2)+1,:,:], (height, width, slice_num))
                ax_sample[:,:,1,:] = np.reshape(TRUS[int(depth/2)-int((slice_num-1)/2):int(depth/2)+int((slice_num)/2)+1,:,:], (height, width, slice_num))
                
                sag_sample[:,:,0,:] = np.reshape(MR[:,:,int(width/2)-int((slice_num-1)/2):int(width/2)+int((slice_num)/2)+1], (depth, height, slice_num))
                sag_sample[:,:,1,:] = np.reshape(TRUS[:,:,int(width/2)-int((slice_num-1)/2):int(width/2)+int((slice_num)/2)+1], (depth, height, slice_num))
                
                cor_sample[:,:,0,:] = np.reshape(MR[:,int(height/2)-int((slice_num-1)/2):int(height/2)+int((slice_num)/2)+1,:], (depth, width, slice_num))
                cor_sample[:,:,1,:] = np.reshape(TRUS[:,int(height/2)-int((slice_num-1)/2):int(height/2)+int((slice_num)/2)+1,:], (depth, width, slice_num))
                
                
                
                ax_batch_samples[k, :,:,:,:] = ax_sample
                sag_batch_samples[k, :,:,:,:] = sag_sample
                cor_batch_samples[k, :,:,:,:] = cor_sample
                #batch_labels.append(pos_neg)
                #params = tuple(-1*np.asarray(params))
                batch_transforms.append(params)
                ax_transforms.append([params[0], params[1], params[5]])
                sag_transforms.append([params[1], params[2], params[3]])
                cor_transforms.append([params[0], params[2], params[4]])
                batch_errors.append([err])
                
                batch_tX.append(params[0])
                batch_tY.append(params[1])
                batch_tZ.append(params[2])
                batch_rotX.append(params[3])
                batch_rotY.append(params[4])
                batch_rotZ.append(params[5])
                
                #batch_segs.append(segMesh)
                
                #batch_affines.append(trans)
            
            yield ([ax_batch_samples, sag_batch_samples, cor_batch_samples], [np.asarray(batch_tX),np.asarray(batch_tY),np.asarray(batch_tZ),np.asarray(batch_rotX),np.asarray(batch_rotY),np.asarray(batch_rotZ),np.asarray(batch_transforms)])
            
    def generate_batch_3D_transform_prediction(self, batch_size=32, shape=(96,96,32)):
        """Used for keras training and validation
        """
        batch_index = 0
        np.random.seed()
        
        (width, height, depth) = shape
        
        while True:
            # Shuffle cases
            if batch_index == 0:
                case_array = np.random.permutation(self.good_cases)

            #current_index = (batch_index * batch_size) % self.num_cases
            current_index = batch_index * batch_size
            
            if (current_index + batch_size) < self.num_cases:
                current_batch_size = batch_size
                batch_index += 1
            else:
                # handle special case where only 1 sample left for the batch
                if (self.num_cases - current_index) > 1:
                    current_batch_size = self.num_cases - current_index
                else:
                    current_batch_size = 2
                    current_index -= 1
                batch_index = 0
            
            batch_samples = np.zeros((current_batch_size, depth, height, width, 2), dtype=np.ubyte)
            #batch_labels = []
            batch_transforms = []
                
            for k in range(current_batch_size):
                case_no = case_array[k + current_index]
                #print(case_no)
                sampledFixed, sampledMoving, err, params, segMesh, trans = self.create_sample_NIH(case_no, shape)
                # Put into 4D array
                sample4D = np.zeros((depth, height, width, 2), dtype=np.ubyte)
                sample4D[:,:,:,0] = sitk.GetArrayFromImage(sampledFixed)
                sample4D[:,:,:,1] = sitk.GetArrayFromImage(sampledMoving)
                
                batch_samples[k, :,:,:,:] = sample4D
                #batch_labels.append(pos_neg)
                batch_transforms.append(params)
                #batch_errors.append([err])
            
            yield (batch_samples, batch_transforms)
            
    def generate_batch_US_regression(self, batch_size=32, shape=(96,96,32)):
        """
        """
        batch_index = 0
        np.random.seed()
        
        (width, height, depth) = shape
        
        while True:
            # Shuffle cases
            if batch_index == 0:
                case_array = np.random.permutation(self.good_cases)

            #current_index = (batch_index * batch_size) % self.num_cases
            current_index = batch_index * batch_size
            
            if (current_index + batch_size) < self.num_cases:
                current_batch_size = batch_size
                batch_index += 1
            else:
                # handle special case where only 1 sample left for the batch
                if (self.num_cases - current_index) > 1:
                    current_batch_size = self.num_cases - current_index
                else:
                    current_batch_size = 2
                    current_index -= 1
                batch_index = 0
            
            batch_samples = np.zeros((current_batch_size, depth, height, width, 2), dtype=np.ubyte)
            #batch_labels = []
            batch_params = np.zeros((current_batch_size, 6), dtype=np.float)
                
            for k in range(current_batch_size):
                case_no = case_array[k + current_index]
                #print(case_no)
                sampledFixed, sampledMoving, err, params = self.create_sample(case_no, shape)
                
                # Put into 4D array
                sample4D = np.zeros((depth, height, width, 2), dtype=np.ubyte)
                sample4D[:,:,:,0] = sitk.GetArrayFromImage(sampledFixed)
                sample4D[:,:,:,1] = sitk.GetArrayFromImage(sampledMoving)
                
                batch_samples[k, :,:,:,:] = sample4D
                #batch_labels.append(pos_neg)
                batch_params[k,:] = params
            
            yield (batch_samples, batch_params)
            
    def generate_batch_US_regression_siamese(self, batch_size=32, shape=(96,96,32)):
        """
        """
        batch_index = 0
        np.random.seed()
        
        (width, height, depth) = shape
        
        while True:
            # Shuffle cases
            if batch_index == 0:
                case_array = np.random.permutation(self.good_cases)

            #current_index = (batch_index * batch_size) % self.num_cases
            current_index = batch_index * batch_size
            
            if (current_index + batch_size) < self.num_cases:
                current_batch_size = batch_size
                batch_index += 1
            else:
                # handle special case where only 1 sample left for the batch
                if (self.num_cases - current_index) > 1:
                    current_batch_size = self.num_cases - current_index
                else:
                    current_batch_size = 2
                    current_index -= 1
                batch_index = 0
            
            batch_samples = np.zeros((current_batch_size, depth, height, width, 1), dtype=np.ubyte)
            batch_samples_GT = np.zeros((current_batch_size, depth, height, width, 1), dtype=np.ubyte)
            #batch_labels = []
            batch_params = np.zeros((current_batch_size, 6), dtype=np.float)
                
            for k in range(current_batch_size):
                case_no = case_array[k + current_index]
                #print(case_no)
                sampledFixed, sampledMoving, sampledMovingGT, err, params = self.create_sample_MRUS2US(case_no, shape)
                
                # Put into 4D array
                sample4D = np.zeros((depth, height, width, 1), dtype=np.ubyte)
                sample4D_GT = np.zeros((depth, height, width, 1), dtype=np.ubyte)
                
                sample4D[:,:,:,0] = sitk.GetArrayFromImage(sampledMoving)
                sample4D_GT[:,:,:,0] = sitk.GetArrayFromImage(sampledMovingGT)
                
                batch_samples[k, :,:,:,:] = sample4D
                batch_samples_GT[k, :,:,:,:] = sample4D_GT
                #batch_labels.append(pos_neg)
                batch_params[k,:] = params
            
            yield ([batch_samples, batch_samples_GT], batch_params)
            
    def generate_batch_transformation_regression(self, batch_size=32, shape=(96,96,32)):
        """
        """
        batch_index = 0
        np.random.seed()
        
        (width, height, depth) = shape
        
        while True:
            # Shuffle cases
            if batch_index == 0:
                case_array = np.random.permutation(self.good_cases)

            #current_index = (batch_index * batch_size) % self.num_cases
            current_index = batch_index * batch_size
            
            if (current_index + batch_size) < self.num_cases:
                current_batch_size = batch_size
                batch_index += 1
            else:
                # handle special case where only 1 sample left for the batch
                if (self.num_cases - current_index) > 1:
                    current_batch_size = self.num_cases - current_index
                else:
                    current_batch_size = 2
                    current_index -= 1
                batch_index = 0
            
            batch_samples = np.zeros((current_batch_size, depth, height, width, 2), dtype=np.ubyte)
            #batch_labels = []
            batch_params = np.zeros((current_batch_size, 6), dtype=np.float)
                
            for k in range(current_batch_size):
                case_no = case_array[k + current_index]
                #print(case_no)
                sampledFixed, sampledMoving, sampledMovingGT, err, params = self.create_sample_MRUS2US(case_no, shape)
                
                # Put into 4D array
                sample4D = np.zeros((depth, height, width, 2), dtype=np.ubyte)
                
                sample4D[:,:,:,0] = sitk.GetArrayFromImage(sampledMoving)
                sample4D[:,:,:,1] = sitk.GetArrayFromImage(sampledFixed)
                
                batch_samples[k, :,:,:,:] = sample4D
                #batch_labels.append(pos_neg)
                batch_params[k,:] = params
            
            yield (batch_samples, batch_params)

    def generate_batch_GAN_AE(self, batch_size=32, shape=(96,96,32), MR_TRUS='MR'):
        """Used for keras training and validation
        """
        batch_index = 0
        np.random.seed()
        
        (width, height, depth) = shape
        
        while True:
            # Shuffle cases
            if batch_index == 0:
                case_array = np.random.permutation(self.good_cases)

            #current_index = (batch_index * batch_size) % self.num_cases
            current_index = batch_index * batch_size
            
            if (current_index + batch_size) < self.num_cases:
                current_batch_size = batch_size
                batch_index += 1
            else:
                # handle special case where only 1 sample left for the batch
                if (self.num_cases - current_index) > 1:
                    current_batch_size = self.num_cases - current_index
                else:
                    current_batch_size = 2
                    current_index -= 1
                batch_index = 0
            
            batch_samples = np.zeros((current_batch_size, depth, height, width, 1), dtype=np.ubyte)
            valid = np.ones(current_batch_size,1)
            #batch_labels = []
                
            for k in range(current_batch_size):
                case_no = case_array[k + current_index]
                #print(case_no)
                sampledFixed, sampledMoving, err, params = self.create_sample(case_no, shape)
                
                # Put into 4D array
                sample4D = np.zeros((depth, height, width, 1), dtype=np.ubyte)
                if MR_TRUS == 'MR':
                    sample4D[:,:,:,0] = sitk.GetArrayFromImage(sampledFixed)
                else:
                    sample4D[:,:,:,0] = sitk.GetArrayFromImage(sampledMoving)
                
                batch_samples[k, :,:,:,:] = sample4D
                #batch_labels.append(pos_neg)
            
            yield (batch_samples)
            
    def generate_batch_AIRNet(self, batch_size=32, shape=(96,96,32)):
        
        """Used for keras training and validation
        """
        batch_index = 0
        np.random.seed()
        
        (width, height, depth) = shape
        
        while True:
            # Shuffle cases
            if batch_index == 0:
                case_array = np.random.permutation(self.good_cases)

            #current_index = (batch_index * batch_size) % self.num_cases
            current_index = batch_index * batch_size
            
            if (current_index + batch_size) < self.num_cases:
                current_batch_size = batch_size
                batch_index += 1
            else:
                # handle special case where only 1 sample left for the batch
                if (self.num_cases - current_index) > 1:
                    current_batch_size = self.num_cases - current_index
                else:
                    current_batch_size = 2
                    current_index -= 1
                batch_index = 0
            
            batch_samples = np.zeros((current_batch_size, depth, height, width, 2), dtype=np.ubyte)
            batch_samples_GT = np.zeros((current_batch_size, depth, height, width, 2), dtype=np.ubyte)
            for k in range(current_batch_size):
                case_no = case_array[k + current_index]
                #print(case_no)
                sampledFixed, sampledMoving, sampledMovingGT, err, params = self.create_sample_MRUS2US(case_no, shape)

                sample4D = np.zeros((depth, height, width, 2), dtype=np.ubyte)
                sample4D[:,:,:,0] = sitk.GetArrayFromImage(sampledFixed)
                sample4D[:,:,:,1] = sitk.GetArrayFromImage(sampledMoving)
                
                batch_samples[k, :,:,:,:] = sample4D
                
                sample4D = np.zeros((depth, height, width, 2), dtype=np.ubyte)
                sample4D[:,:,:,0] = sitk.GetArrayFromImage(sampledFixed)
                sample4D[:,:,:,1] = sitk.GetArrayFromImage(sampledMovingGT)
                
                batch_samples_GT[k, :,:,:,:] = sample4D
                
            yield (batch_samples, batch_samples_GT)
                
            
    def generate_batch_2D_AEMRax(self, batch_size=32, shape=(96,96,32)):
        """Used for keras training and validation
        """
        batch_index = 0
        np.random.seed()
        
        (width, height, depth) = shape
        
        while True:
            # Shuffle cases
            if batch_index == 0:
                case_array = np.random.permutation(self.good_cases)

            #current_index = (batch_index * batch_size) % self.num_cases
            current_index = batch_index * batch_size
            
            if (current_index + batch_size) < self.num_cases:
                current_batch_size = batch_size
                batch_index += 1
            else:
                # handle special case where only 1 sample left for the batch
                if (self.num_cases - current_index) > 1:
                    current_batch_size = self.num_cases - current_index
                else:
                    current_batch_size = 2
                    current_index -= 1
                batch_index = 0
            
            batch_samples = np.zeros((current_batch_size, height, width, 1), dtype=np.ubyte)
            for k in range(current_batch_size):
                case_no = case_array[k + current_index]
                #print(case_no)
                sampledFixed, sampledMoving, err, params = self.create_sample(case_no, shape)
                sample4D = np.zeros((height, width, 1), dtype=np.ubyte)
                sample4D[:,:,0] = sitk.GetArrayFromImage(sampledFixed)[random.randint(0,sitk.GetArrayFromImage(sampledFixed).shape[0]-1)]
                
                batch_samples[k,:,:,:] = sample4D
                
            yield (batch_samples, batch_samples)
            
    def generate_batch_2D_AEUSax(self, batch_size=32, shape=(96,96,32)):
        """Used for keras training and validation
        """
        batch_index = 0
        np.random.seed()
        
        (width, height, depth) = shape
        
        while True:
            # Shuffle cases
            if batch_index == 0:
                case_array = np.random.permutation(self.good_cases)

            #current_index = (batch_index * batch_size) % self.num_cases
            current_index = batch_index * batch_size
            
            if (current_index + batch_size) < self.num_cases:
                current_batch_size = batch_size
                batch_index += 1
            else:
                # handle special case where only 1 sample left for the batch
                if (self.num_cases - current_index) > 1:
                    current_batch_size = self.num_cases - current_index
                else:
                    current_batch_size = 2
                    current_index -= 1
                batch_index = 0
            
            batch_samples = np.zeros((current_batch_size, height, width, 1), dtype=np.ubyte)
            for k in range(current_batch_size):
                case_no = case_array[k + current_index]
                #print(case_no)
                sampledFixed, sampledMoving, err, params = self.create_sample(case_no, shape)
                sample4D = np.zeros((height, width, 1), dtype=np.ubyte)
                sample4D[:,:,0] = sitk.GetArrayFromImage(sampledMoving)[random.randint(0,sitk.GetArrayFromImage(sampledMoving).shape[0]-1)]
                
                batch_samples[k,:,:,:] = sample4D
                
            yield (batch_samples, batch_samples)
            
    def generate_batch_2D_MRUS_recon(self, batch_size=32, shape=(96,96,32)):
        """Used for keras training and validation
        """
        batch_index = 0
        np.random.seed()
        
        (width, height, depth) = shape
        
        while True:
            # Shuffle cases
            if batch_index == 0:
                case_array = np.random.permutation(self.good_cases)

            #current_index = (batch_index * batch_size) % self.num_cases
            current_index = batch_index * batch_size
            
            if (current_index + batch_size) < self.num_cases:
                current_batch_size = batch_size
                batch_index += 1
            else:
                # handle special case where only 1 sample left for the batch
                if (self.num_cases - current_index) > 1:
                    current_batch_size = self.num_cases - current_index
                else:
                    current_batch_size = 2
                    current_index -= 1
                batch_index = 0
            
            batch_samples = np.zeros((current_batch_size, height, width, 2), dtype=np.ubyte)
            for k in range(current_batch_size):
                case_no = case_array[k + current_index]
                #print(case_no)
                sampledFixed, sampledMoving, sampledMovingGT, err, params = self.create_sample_MRUS2US(case_no, shape)
                MR = sitk.GetArrayFromImage(sampledFixed)
                US = sitk.GetArrayFromImage(sampledMovingGT)
                idx = random.randint(0,MR.shape[0]-1)
                MR_ax = MR[idx]
                US_ax = US[idx]

                for i in range(US_ax.shape[0]):
                    for j in range(US_ax.shape[1]):
                        if US_ax[i][j] == 0:
                            MR_ax[i][j] = 0

                sample4D = np.zeros((height, width, 2), dtype=np.ubyte)
                sample4D[:,:,0] = MR_ax
                sample4D[:,:,1] = US_ax
                
                batch_samples[k,:,:,:] = sample4D
                
            yield (np.reshape(batch_samples[:,:,:,0],(current_batch_size,height,width,1)), [np.reshape(batch_samples[:,:,:,0],(current_batch_size,height,width,1)), np.reshape(batch_samples[:,:,:,1],(current_batch_size,height,width,1))])

            
    def generate_batch_2D_MRUSax(self, batch_size=32, shape=(96,96,32)):
        """Used for keras training and validation
        """
        batch_index = 0
        np.random.seed()
        
        (width, height, depth) = shape
        
        while True:
            # Shuffle cases
            if batch_index == 0:
                case_array = np.random.permutation(self.good_cases)

            #current_index = (batch_index * batch_size) % self.num_cases
            current_index = batch_index * batch_size
            
            if (current_index + batch_size) < self.num_cases:
                current_batch_size = batch_size
                batch_index += 1
            else:
                # handle special case where only 1 sample left for the batch
                if (self.num_cases - current_index) > 1:
                    current_batch_size = self.num_cases - current_index
                else:
                    current_batch_size = 2
                    current_index -= 1
                batch_index = 0
            
            batch_samples = np.zeros((current_batch_size, height, width, 2), dtype=np.ubyte)
            for k in range(current_batch_size):
                case_no = case_array[k + current_index]
                #print(case_no)
                sampledFixed, sampledMoving, sampledMovingGT, err, params = self.create_sample_MRUS2US(case_no, shape)
                MR = sitk.GetArrayFromImage(sampledFixed)
                US = sitk.GetArrayFromImage(sampledMovingGT)
                idx = random.randint(0,MR.shape[0]-1)
                MR_ax = MR[idx]
                US_ax = US[idx]
                for i in range(US_ax.shape[0]):
                    for j in range(US_ax.shape[1]):
                        if US_ax[i][j] == 0:
                            MR_ax[i][j] = 0
                sample4D = np.zeros((height, width, 2), dtype=np.ubyte)
                sample4D[:,:,0] = MR_ax
                sample4D[:,:,1] = US_ax
                
                batch_samples[k,:,:,:] = sample4D
                
            yield (np.reshape(batch_samples[:,:,:,0],(current_batch_size,height,width,1)), np.reshape(batch_samples[:,:,:,1],(current_batch_size,height,width,1)))
             
            
    def generate_batch_2D_GAN_MR_US(self, batch_size=32, shape=(96,96,32)):
        """Used for keras training and validation
        """
        batch_index = 0
        np.random.seed()
        
        (width, height, depth) = shape
        
        while True:
            # Shuffle cases
            if batch_index == 0:
                case_array = np.random.permutation(self.good_cases)

            #current_index = (batch_index * batch_size) % self.num_cases
            current_index = batch_index * batch_size
            
            if (current_index + batch_size) < self.num_cases:
                current_batch_size = batch_size
                batch_index += 1
            else:
                # handle special case where only 1 sample left for the batch
                if (self.num_cases - current_index) > 1:
                    current_batch_size = self.num_cases - current_index
                else:
                    current_batch_size = 2
                    current_index -= 1
                batch_index = 0
            

            batch_samples = np.zeros((current_batch_size, height, width, 2), dtype=np.ubyte)
            #encoded_samples = np.zeros((current_batch_size, int(height/2), int(width/2), 2), dtype=np.ubyte)
            cond = True
            k = 0
            while cond == True:
                case_no = case_array[k + current_index]
                #print(case_no)
                sampledFixed, sampledMoving, sampledMovingGT, err, params = self.create_sample_MRUS2US(case_no, shape)
                MR = sitk.GetArrayFromImage(sampledFixed)
                US = sitk.GetArrayFromImage(sampledMovingGT)
                
                black_img = np.zeros(shape=(1,96,96,1))
                
                idx = np.random.randint(13,26)
                MR_ax = MR[idx]
                US_ax = US[idx]


                US_ax = np.reshape(US_ax, (1,96,96,1))
                MR_ax = np.reshape(MR_ax, (1,96,96,1))
                
                if np.array_equal(US_ax, black_img) or np.array_equal(MR_ax, black_img):
                    continue
                
                #US_encoded = US_enc.predict(US_ax)
                #MR_encoded = MR_enc.predict(MR_ax)
                sample3D = np.zeros((height, width, 2), dtype=np.ubyte)
                sample3D[:,:,0] = MR_ax[0,:,:,0]
                sample3D[:,:,1] = US_ax[0,:,:,0]
                
                
                batch_samples[k,:,:,:] = sample3D
                
                k += 1
                
                if k == current_batch_size - 1:
                    cond = False
                
                
            yield (np.reshape(batch_samples[:,:,:,0],(current_batch_size,height,width,1)), np.reshape(batch_samples[:,:,:,1],(current_batch_size,height,width,1)))
        
    def generate_batch_2D_GAN_MRUS_GT(self, batch_size=32, shape=(96,96,32)):
        """Used for keras training and validation
        """
        batch_index = 0
        np.random.seed()
        
        (width, height, depth) = shape
        
        while True:
            # Shuffle cases
            if batch_index == 0:
                case_array = np.random.permutation(self.good_cases)

            #current_index = (batch_index * batch_size) % self.num_cases
            current_index = batch_index * batch_size
            
            if (current_index + batch_size) < self.num_cases:
                current_batch_size = batch_size
                batch_index += 1
            else:
                # handle special case where only 1 sample left for the batch
                if (self.num_cases - current_index) > 1:
                    current_batch_size = self.num_cases - current_index
                else:
                    current_batch_size = 2
                    current_index -= 1
                batch_index = 0
            

            batch_samples = np.zeros((current_batch_size, height, width, 2), dtype=np.ubyte)
            US_GT_imgs = np.zeros((current_batch_size, height, width, 1), dtype=np.ubyte)
            #encoded_samples = np.zeros((current_batch_size, int(height/2), int(width/2), 2), dtype=np.ubyte)
            cond = True
            k = 0
            while cond == True:
                case_no = case_array[k + current_index]
                #print(case_no)
                sampledFixed, sampledMoving, sampledMovingGT, err, params = self.create_sample_MRUS2US(case_no, shape)
                MR = sitk.GetArrayFromImage(sampledFixed)
                US = sitk.GetArrayFromImage(sampledMoving)
                US_GT = sitk.GetArrayFromImage(sampledMovingGT)
                
                black_img = np.zeros(shape=(1,96,96,1))
                
                idx = np.random.randint(0,32)
                MR_ax = MR[idx]
                US_ax = US[idx]
                US_GT_ax = US_GT[idx]


                US_ax = np.reshape(US_ax, (1,96,96,1))
                MR_ax = np.reshape(MR_ax, (1,96,96,1))
                US_GT_ax = np.reshape(US_GT_ax, (1,96,96,1))
                
                if np.array_equal(US_ax, black_img) or np.array_equal(MR_ax, black_img) or np.array_equal(US_GT_ax, black_img):
                    continue
                
                #US_encoded = US_enc.predict(US_ax)
                #MR_encoded = MR_enc.predict(MR_ax)
                sample3D = np.zeros((height, width, 2), dtype=np.ubyte)
                sample3D[:,:,0] = MR_ax[0,:,:,0]
                sample3D[:,:,1] = US_ax[0,:,:,0]
                
                
                batch_samples[k,:,:,:] = sample3D
                US_GT_imgs[k,:,:,:] = US_GT_ax[0,:,:,:]
                
                k += 1
                
                if k == current_batch_size - 1:
                    cond = False
                
                
            yield (np.reshape(batch_samples,(current_batch_size,height,width,2)), np.reshape(US_GT_imgs,(current_batch_size,height,width,1)))
        
    def generate_batch_mapping2D(self, batch_size=32, shape=(96,96,32)):
        """Used for keras training and validation
        """
        batch_index = 0
        np.random.seed()
        
        (width, height, depth) = shape
        
        while True:
            # Shuffle cases
            if batch_index == 0:
                case_array = np.random.permutation(self.good_cases)

            #current_index = (batch_index * batch_size) % self.num_cases
            current_index = batch_index * batch_size
            
            if (current_index + batch_size) < self.num_cases:
                current_batch_size = batch_size
                batch_index += 1
            else:
                # handle special case where only 1 sample left for the batch
                if (self.num_cases - current_index) > 1:
                    current_batch_size = self.num_cases - current_index
                else:
                    current_batch_size = 2
                    current_index -= 1
                batch_index = 0
                
            MR_model = '/home/haskig/tmp/MR2Dax_autoencoder.h5'
            US_model = '/home/haskig/tmp/MR2Dax_autoencoder.h5'
            MR_AE = load_model(MR_model)
            US_AE = load_model(US_model)
            MR_enc = Model(inputs=MR_AE.input, 
                          outputs=MR_AE.get_layer(index=18).output)
            US_enc = Model(inputs=US_AE.input, 
                          outputs=US_AE.get_layer(index=18).output)
            
            MR_encs = np.zeros(shape=(current_batch_size, int(height/2), int(width/2), 1), dtype=np.ubyte)
            US_encs = np.zeros(shape=(current_batch_size, int(height/2), int(width/2), 1), dtype=np.ubyte)
            
            for k in range(current_batch_size):
                case_no = case_array[k + current_index]
                #print(case_no)
                sampledFixed, sampledMoving, sampledMovingGT, err, params = self.create_sample_MRUS2US(case_no, shape)
                MR = sitk.GetArrayFromImage(sampledFixed)
                US = sitk.GetArrayFromImage(sampledMovingGT)
                idx = random.randint(0,MR.shape[0]-1)
                MR_ax = MR[idx]
                US_ax = US[idx]

                MR_ax = np.reshape(MR_ax, (1,MR_ax.shape[0],MR_ax.shape[1],1))
                US_ax = np.reshape(US_ax, (1,US_ax.shape[0],US_ax.shape[1],1))
                MR_rep = MR_enc.predict(MR_ax)
                US_rep = US_enc.predict(US_ax)
                
                MR_encs[k,:,:,0] = MR_rep[0,:,:,0]
                US_encs[k,:,:,0] = US_rep[0,:,:,0]
                
            yield (MR_encs, US_encs)               
            
    def generate_batch_mapping(self, batch_size=32, shape=(96,96,32)):
        """Used for keras training and validation
        """
        batch_index = 0
        np.random.seed()
        
        (width, height, depth) = shape
        
        while True:
            # Shuffle cases
            if batch_index == 0:
                case_array = np.random.permutation(self.good_cases)

            #current_index = (batch_index * batch_size) % self.num_cases
            current_index = batch_index * batch_size
            
            if (current_index + batch_size) < self.num_cases:
                current_batch_size = batch_size
                batch_index += 1
            else:
                # handle special case where only 1 sample left for the batch
                if (self.num_cases - current_index) > 1:
                    current_batch_size = self.num_cases - current_index
                else:
                    current_batch_size = 2
                    current_index -= 1
                batch_index = 0
                
            MR_model = '/home/haskig/tmp/MR_autoencoder.h5'
            US_model = '/home/haskig/tmp/US_autoencoder.h5'
            MR_AE = load_model(MR_model)
            US_AE = load_model(US_model)
            MR_enc = Model(inputs=MR_AE.input, 
                          outputs=MR_AE.get_layer(index=11).output)
            US_enc = Model(inputs=US_AE.input, 
                          outputs=US_AE.get_layer(index=11).output)
            
            MR_encs = np.zeros(shape=(current_batch_size, int(depth/2), int(height/4), int(width/4), 1), dtype=np.ubyte)
            US_encs = np.zeros(shape=(current_batch_size, int(depth/2), int(height/4), int(width/4), 1), dtype=np.ubyte)
            
            for k in range(current_batch_size):
                case_no = case_array[k + current_index]
                #print(case_no)
                sampledFixed, sampledMoving, sampledMovingGT, err, params = self.create_sample_MRUS2US(case_no, shape)
                FixedArr = np.reshape(sitk.GetArrayFromImage(sampledFixed),(1,depth,height,width,1))
                MovingArr = np.reshape(sitk.GetArrayFromImage(sampledMovingGT),(1,depth,height,width,1))
                MR_rep = MR_enc.predict(FixedArr)
                US_rep = US_enc.predict(MovingArr)
                
                MR_encs[k,:,:,:,0] = MR_rep[0,:,:,:,0]
                US_encs[k,:,:,:,0] = US_rep[0,:,:,:,0]
                
            yield (MR_encs, US_encs)
                

    def generate_batch_AE(self, batch_size=32, shape=(96,96,32)):
        """Used for keras training and validation
        """
        batch_index = 0
        np.random.seed()
        
        (width, height, depth) = shape
        
        while True:
            # Shuffle cases
            if batch_index == 0:
                case_array = np.random.permutation(self.good_cases)

            #current_index = (batch_index * batch_size) % self.num_cases
            current_index = batch_index * batch_size
            
            if (current_index + batch_size) < self.num_cases:
                current_batch_size = batch_size
                batch_index += 1
            else:
                # handle special case where only 1 sample left for the batch
                if (self.num_cases - current_index) > 1:
                    current_batch_size = self.num_cases - current_index
                else:
                    current_batch_size = 2
                    current_index -= 1
                batch_index = 0
            
            batch_samples = np.zeros((current_batch_size, depth, height, width, 2), dtype=np.ubyte)
                
            for k in range(current_batch_size):
                case_no = case_array[k + current_index]
                #print(case_no)
                sampledFixed, sampledMoving, err, params = self.create_sample(case_no, shape)
                
                # Put into 4D array
                sample4D = np.zeros((depth, height, width, 2), dtype=np.ubyte)
                sample4D[:,:,:,0] = sitk.GetArrayFromImage(sampledFixed)
                sample4D[:,:,:,1] = sitk.GetArrayFromImage(sampledMoving)
                
                batch_samples[k, :,:,:,:] = sample4D
            
            yield (batch_samples, batch_samples)
            
    def generate_batch_MR_AE(self, batch_size=32, shape=(96,96,32)):
        """Used for keras training and validation
        """
        batch_index = 0
        np.random.seed()
        
        (width, height, depth) = shape
        
        while True:
            # Shuffle cases
            if batch_index == 0:
                case_array = np.random.permutation(self.good_cases)

            #current_index = (batch_index * batch_size) % self.num_cases
            current_index = batch_index * batch_size
            
            if (current_index + batch_size) < self.num_cases:
                current_batch_size = batch_size
                batch_index += 1
            else:
                # handle special case where only 1 sample left for the batch
                if (self.num_cases - current_index) > 1:
                    current_batch_size = self.num_cases - current_index
                else:
                    current_batch_size = 2
                    current_index -= 1
                batch_index = 0
            
            batch_samples = np.zeros((current_batch_size, depth, height, width, 1), dtype=np.ubyte)
                
            for k in range(current_batch_size):
                case_no = case_array[k + current_index]
                #print(case_no)
                sampledFixed, sampledMoving, err, params = self.create_sample(case_no, shape)
                
                # Put into 4D array
                sample4D = np.zeros((depth, height, width, 1), dtype=np.ubyte)
                sample4D[:,:,:,0] = sitk.GetArrayFromImage(sampledFixed)
                
                batch_samples[k, :,:,:,:] = sample4D
            
            yield (batch_samples, batch_samples)
            
    def generate_batch_US_AE(self, batch_size=32, shape=(96,96,32)):
        """Used for keras training and validation
        """
        batch_index = 0
        np.random.seed()
        
        (width, height, depth) = shape
        
        while True:
            # Shuffle cases
            if batch_index == 0:
                case_array = np.random.permutation(self.good_cases)

            #current_index = (batch_index * batch_size) % self.num_cases
            current_index = batch_index * batch_size
            
            if (current_index + batch_size) < self.num_cases:
                current_batch_size = batch_size
                batch_index += 1
            else:
                # handle special case where only 1 sample left for the batch
                if (self.num_cases - current_index) > 1:
                    current_batch_size = self.num_cases - current_index
                else:
                    current_batch_size = 2
                    current_index -= 1
                batch_index = 0
            
            batch_samples = np.zeros((current_batch_size, depth, height, width, 1), dtype=np.ubyte)
                
            for k in range(current_batch_size):
                case_no = case_array[k + current_index]
                #print(case_no)
                sampledFixed, sampledMoving, err, params = self.create_sample(case_no, shape)
                
                # Put into 4D array
                sample4D = np.zeros((depth, height, width, 1), dtype=np.ubyte)
                sample4D[:,:,:,0] = sitk.GetArrayFromImage(sampledMoving)
                
                batch_samples[k, :,:,:,:] = sample4D
            
            yield (batch_samples, batch_samples)

            
    def generate_batch_MR2US(self, batch_size=32, shape=(96,96,32)):
        """Used for keras training and validation
        """
        batch_index = 0
        np.random.seed()
        
        (width, height, depth) = shape
        
        while True:
            # Shuffle cases
            if batch_index == 0:
                case_array = np.random.permutation(self.good_cases)

            #current_index = (batch_index * batch_size) % self.num_cases
            current_index = batch_index * batch_size
            
            if (current_index + batch_size) < self.num_cases:
                current_batch_size = batch_size
                batch_index += 1
            else:
                # handle special case where only 1 sample left for the batch
                if (self.num_cases - current_index) > 1:
                    current_batch_size = self.num_cases - current_index
                else:
                    current_batch_size = 2
                    current_index -= 1
                batch_index = 0
                
            batch_samplesMR = np.zeros((current_batch_size, depth, height, width, 1), dtype=np.ubyte)
            batch_samplesUS = np.zeros((current_batch_size, depth, height, width, 1), dtype=np.ubyte)
            for k in range(current_batch_size):
                case_no = case_array[k + current_index]
                #print(case_no)
                sampledFixed, sampledMoving, sampledMovingGT, err, params = self.create_sample_MRUS2US(case_no, shape)
                
                # Put into 4D array
                
                batch_samplesMR[k, :,:,:,0] = sitk.GetArrayFromImage(sampledFixed)
                batch_samplesUS[k, :,:,:,0] = sitk.GetArrayFromImage(sampledMovingGT)
                
            
            yield (batch_samplesMR, {'decoded_MR': batch_samplesMR, 'decoded_US': batch_samplesUS})
            
    def generate_batch_MRUS_US(self, batch_size=32, shape=(96,96,32)):
        """Used for keras training and validation
        """
        batch_index = 0
        np.random.seed()
        
        (width, height, depth) = shape
        
        while True:
            # Shuffle cases
            if batch_index == 0:
                case_array = np.random.permutation(self.good_cases)

            #current_index = (batch_index * batch_size) % self.num_cases
            current_index = batch_index * batch_size
            
            if (current_index + batch_size) < self.num_cases:
                current_batch_size = batch_size
                batch_index += 1
            else:
                # handle special case where only 1 sample left for the batch
                if (self.num_cases - current_index) > 1:
                    current_batch_size = self.num_cases - current_index
                else:
                    current_batch_size = 2
                    current_index -= 1
                batch_index = 0
                
            batch_samples = np.zeros((current_batch_size, depth, height, width, 2), dtype=np.ubyte)
            US = np.zeros((current_batch_size, depth, height, width, 1), dtype=np.ubyte)
                
            for k in range(current_batch_size):
                case_no = case_array[k + current_index]
                #print(case_no)
                sampledFixed, sampledMoving, err, params = self.create_sample(case_no, shape)
                
                # Put into 4D array
                
                batch_samples[k, :,:,:,0] = sitk.GetArrayFromImage(sampledFixed)
                batch_samples[k, :,:,:,1] = sitk.GetArrayFromImage(sampledMoving)
                
                US[k, :,:,:,0] = sitk.GetArrayFromImage(sampledMoving)
                
            
            yield (batch_samples, US)
            
    def generate_batch_US2MR(self, batch_size=32, shape=(96,96,32)):
        """Used for keras training and validation
        """
        batch_index = 0
        np.random.seed()
        
        (width, height, depth) = shape
        
        while True:
            # Shuffle cases
            if batch_index == 0:
                case_array = np.random.permutation(self.good_cases)

            #current_index = (batch_index * batch_size) % self.num_cases
            current_index = batch_index * batch_size
            
            if (current_index + batch_size) < self.num_cases:
                current_batch_size = batch_size
                batch_index += 1
            else:
                # handle special case where only 1 sample left for the batch
                if (self.num_cases - current_index) > 1:
                    current_batch_size = self.num_cases - current_index
                else:
                    current_batch_size = 2
                    current_index -= 1
                batch_index = 0
                
            batch_samplesMR = np.zeros((current_batch_size, depth, height, width, 1), dtype=np.ubyte)
            batch_samplesUS = np.zeros((current_batch_size, depth, height, width, 1), dtype=np.ubyte)
            for k in range(current_batch_size):
                case_no = case_array[k + current_index]
                #print(case_no)
                sampledFixed, sampledMoving, err, params = self.create_sample_MRUS(case_no, shape)
                
                # Put into 4D array
                
                batch_samplesMR[k, :,:,:,0] = sitk.GetArrayFromImage(sampledFixed)
                batch_samplesUS[k, :,:,:,0] = sitk.GetArrayFromImage(sampledMoving)
                
            
            yield (batch_samplesUS, batch_samplesMR)
            
            
    def generate_batch_MRUS_GTreg(self, batch_size=32, shape=(96,96,32)):
        """Used for keras training and validation
        """
        batch_index = 0
        np.random.seed()
        
        (width, height, depth) = shape
        
        while True:
            # Shuffle cases
            if batch_index == 0:
                case_array = np.random.permutation(self.good_cases)

            #current_index = (batch_index * batch_size) % self.num_cases
            current_index = batch_index * batch_size
            
            if (current_index + batch_size) < self.num_cases:
                current_batch_size = batch_size
                batch_index += 1
            else:
                # handle special case where only 1 sample left for the batch
                if (self.num_cases - current_index) > 1:
                    current_batch_size = self.num_cases - current_index
                else:
                    current_batch_size = 2
                    current_index -= 1
                batch_index = 0
                
            batch_samples = np.zeros((current_batch_size, depth, height, width, 2), dtype=np.ubyte)
            GT_US = np.zeros((current_batch_size, depth, height, width, 1), dtype=np.ubyte)
                
            for k in range(current_batch_size):
                case_no = case_array[k + current_index]
                #print(case_no)
                sampledFixed, sampledMoving, sampledMovingGT, err, params = self.create_sample_MRUS2US(case_no, shape)
                
                # Put into 4D array
                
                batch_samples[k, :,:,:,0] = sitk.GetArrayFromImage(sampledFixed)
                batch_samples[k, :,:,:,1] = sitk.GetArrayFromImage(sampledMoving)
                
                GT_US[k, :,:,:,0] = sitk.GetArrayFromImage(sampledMovingGT)
                
            
            yield (batch_samples, GT_US)
    
    def generate_batch_US2MR_GTreg(self, batch_size=32, shape=(96,96,32)):
        """Used for keras training and validation
        """
        batch_index = 0
        np.random.seed()
        
        (width, height, depth) = shape
        
        while True:
            # Shuffle cases
            if batch_index == 0:
                case_array = np.random.permutation(self.good_cases)

            #current_index = (batch_index * batch_size) % self.num_cases
            current_index = batch_index * batch_size
            
            if (current_index + batch_size) < self.num_cases:
                current_batch_size = batch_size
                batch_index += 1
            else:
                # handle special case where only 1 sample left for the batch
                if (self.num_cases - current_index) > 1:
                    current_batch_size = self.num_cases - current_index
                else:
                    current_batch_size = 2
                    current_index -= 1
                batch_index = 0
                
            batch_samples = np.zeros((current_batch_size, depth, height, width, 2), dtype=np.ubyte)
            GT_MR = np.zeros((current_batch_size, depth, height, width, 1), dtype=np.ubyte)
                
            for k in range(current_batch_size):
                case_no = case_array[k + current_index]
                #print(case_no)
                sampledFixed, sampledMoving, sampledMovingGT, err, params = self.create_sample_MRUS2US(case_no, shape)
                
                # Put into 4D array
                
                batch_samples[k, :,:,:,0] = sitk.GetArrayFromImage(sampledFixed)
                batch_samples[k, :,:,:,1] = sitk.GetArrayFromImage(sampledMoving)
                
                GT_MR[k, :,:,:,1] = sitk.GetArrayFromImage(sampledMovingGT)
                
            
            yield (batch_samples, GT_MR)
            
    def generate_batch_MIND(self, batch_size=32, shape=(96,96,32)):
        """Used for keras training and validation
        """
        batch_index = 0
        np.random.seed()
        
        (width, height, depth) = shape
        
        while True:
            # Shuffle cases
            if batch_index == 0:
                case_array = np.random.permutation(self.good_cases)

            #current_index = (batch_index * batch_size) % self.num_cases
            current_index = batch_index * batch_size
            
            if (current_index + batch_size) < self.num_cases:
                current_batch_size = batch_size
                batch_index += 1
            else:
                # handle special case where only 1 sample left for the batch
                if (self.num_cases - current_index) > 1:
                    current_batch_size = self.num_cases - current_index
                else:
                    current_batch_size = 2
                    current_index -= 1
                batch_index = 0
            
            batch_samples = np.zeros((current_batch_size, depth, height, width, 2), dtype=np.ubyte)
            #batch_labels = []
            batch_errors = []
                
            for k in range(current_batch_size):
                case_no = case_array[k + current_index]
                #print(case_no)
                sampledFixed, sampledMoving, err, params = self.create_sample_MIND(case_no, shape)
                
                # Put into 4D array
                sample4D = np.zeros((depth, height, width, 2), dtype=np.ubyte)
                sample4D[:,:,:,0] = sitk.GetArrayFromImage(sampledFixed)
                sample4D[:,:,:,1] = sitk.GetArrayFromImage(sampledMoving)
                
                batch_samples[k, :,:,:,:] = sample4D
                #batch_labels.append(pos_neg)
                batch_errors.append([err])
            
            yield (batch_samples, batch_errors)

            
    def generate_batch_perturbations(self, batch_size=32, shape=(96,96,32)):
        """Used for keras training and validation
        """
        batch_index = 0
        np.random.seed()
        
        (width, height, depth) = shape
        
        while True:
            # Shuffle cases
            if batch_index == 0:
                case_array = np.random.permutation(self.good_cases)

            #current_index = (batch_index * batch_size) % self.num_cases
            current_index = batch_index * batch_size
            
            if (current_index + batch_size) < self.num_cases:
                current_batch_size = batch_size
                batch_index += 1
            else:
                # handle special case where only 1 sample left for the batch
                if (self.num_cases - current_index) > 1:
                    current_batch_size = self.num_cases - current_index
                else:
                    current_batch_size = 2
                    current_index -= 1
                batch_index = 0
            
            batch_samples = np.zeros((current_batch_size, depth, height, width, 2), dtype=np.ubyte)
            #batch_labels = []
            batch_errors = []
                
            for k in range(current_batch_size):
                if k % 2 == 0:
                    case_no = case_array[k + current_index]
                #print(case_no)
                
                    sampledFixed, sampledMoving, err, params, sampledFixed_p, sampledMoving_p, err_p, params_p = self.create_sample_perturbed(case_no, shape)
                
                # Put into 4D array
                sample4D = np.zeros((depth, height, width, 2), dtype=np.ubyte)
                if k % 2 == 0:
                    sample4D[:,:,:,0] = sitk.GetArrayFromImage(sampledFixed)
                    sample4D[:,:,:,1] = sitk.GetArrayFromImage(sampledMoving)
                    batch_errors.append([err])
                else:
                    sample4D[:,:,:,0] = sitk.GetArrayFromImage(sampledFixed_p)
                    sample4D[:,:,:,1] = sitk.GetArrayFromImage(sampledMoving_p)
                    batch_errors.append([err_p])
                batch_samples[k, :,:,:,:] = sample4D
                #batch_labels.append(pos_neg)
                
            if current_batch_size % 2 == 0:
                batch_errors_even = []
                batch_samples_even = np.zeros((int(current_batch_size/2), depth, height, width, 2), dtype=np.ubyte)
                batch_errors_odd = []
                batch_samples_odd = np.zeros((int(current_batch_size/2), depth, height, width, 2), dtype=np.ubyte)
                for i in range(current_batch_size):
                    if i % 2 == 0:
                        batch_samples_even[int(i/2), :,:,:,:] = batch_samples[i, :,:,:,:]
                        batch_errors_even.append([batch_errors[i]])
                    else:
                        batch_samples_odd[int((i-1)/2)] = batch_samples[i, :,:,:,:]
                        batch_errors_odd.append([batch_errors[i]])
                batch_samples_new = np.zeros((current_batch_size, depth, height, width, 2), dtype=np.ubyte)
                batch_errors_new = []
                
                for i in range(current_batch_size):
                    if i < current_batch_size//2:
                        batch_samples_new[i, :,:,:,:] = batch_samples_even[i, :,:,:,:]
                        batch_errors_new.append([batch_errors_even[i]])
                    else:
                        batch_samples_new[i, :,:,:,:] = batch_samples_odd[int(i-current_batch_size/2), :,:,:,:]
                        batch_errors_new.append([batch_errors_odd[int(i-current_batch_size/2)]])
                batch_errors_new = np.reshape(np.asarray(batch_errors_new), (np.asarray(batch_errors_new).shape[0],1))
                batch_errors_new.tolist()
            else:
                raise(ValueError('Batch size must be even integer!'))
                    
            yield (batch_samples_new, batch_errors_new)



    # ----- #
    def generate_batch_with_parameters(self, batch_size=32, shape=(96,96,32)):
        """
        """
        batch_index = 0
        np.random.seed()
        
        (width, height, depth) = shape
        
        while True:
            # Shuffle cases
            if batch_index == 0:
                case_array = np.random.permutation(self.good_cases)

            #current_index = (batch_index * batch_size) % self.num_cases
            current_index = batch_index * batch_size
            
            if (current_index + batch_size) < self.num_cases:
                current_batch_size = batch_size
                batch_index += 1
            else:
                # handle special case where only 1 sample left for the batch
                if (self.num_cases - current_index) > 1:
                    current_batch_size = self.num_cases - current_index
                else:
                    current_batch_size = 2
                    current_index -= 1
                batch_index = 0
            
            batch_samples = np.zeros((current_batch_size, depth, height, width, 2), dtype=np.ubyte)
            #batch_labels = []
            batch_errors = np.zeros((current_batch_size), dtype=np.float)
            batch_params = np.zeros((current_batch_size, 6), dtype=np.float)
                
            for k in range(current_batch_size):
                case_no = case_array[k + current_index]
                #print(case_no)
                sampledFixed, sampledMoving, err, params = self.create_sample(case_no, shape)
                
                # Put into 4D array
                sample4D = np.zeros((depth, height, width, 2), dtype=np.ubyte)
                sample4D[:,:,:,0] = sitk.GetArrayFromImage(sampledFixed)
                sample4D[:,:,:,1] = sitk.GetArrayFromImage(sampledMoving)
                
                batch_samples[k, :,:,:,:] = sample4D
                #batch_labels.append(pos_neg)
                batch_errors[k] = err
                batch_params[k,:] = params
            
            yield (batch_samples, batch_errors, batch_params)
            
    def generate_batch_w_perturbations_parameters(self, batch_size=32, shape=(96,96,32)):
        """Used for keras training and validation
        """
        batch_index = 0
        np.random.seed()
        
        (width, height, depth) = shape
        
        while True:
            # Shuffle cases
            if batch_index == 0:
                case_array = np.random.permutation(self.good_cases)

            #current_index = (batch_index * batch_size) % self.num_cases
            current_index = batch_index * batch_size
            
            if (current_index + batch_size) < self.num_cases:
                current_batch_size = batch_size
                batch_index += 1
            else:
                # handle special case where only 1 sample left for the batch
                if (self.num_cases - current_index) > 1:
                    current_batch_size = self.num_cases - current_index
                else:
                    current_batch_size = 2
                    current_index -= 1
                batch_index = 0
            
            batch_samples = np.zeros((current_batch_size, depth, height, width, 2), dtype=np.ubyte)
            batch_samples_p = np.zeros((current_batch_size, depth, height, width, 2), dtype=np.ubyte)
            #batch_labels = []
            batch_errors = []
            batch_errors_p = []
                
            for k in range(current_batch_size):
                case_no = case_array[k + current_index]
                #print(case_no)
                sampledFixed, sampledMoving, err, params, sampledFixed_p, sampledMoving_p, err_p, params_p = self.create_sample_perturbed(case_no, shape)
                
                # Put into 4D array
                sample4D = np.zeros((depth, height, width, 2), dtype=np.ubyte)
                sample4D[:,:,:,0] = sitk.GetArrayFromImage(sampledFixed)
                sample4D[:,:,:,1] = sitk.GetArrayFromImage(sampledMoving)
                
                sample4D_p = np.zeros((depth, height, width, 2), dtype=np.ubyte)
                sample4D_p[:,:,:,0] = sitk.GetArrayFromImage(sampledFixed_p)
                sample4D_p[:,:,:,1] = sitk.GetArrayFromImage(sampledMoving_p)
                
                batch_samples_p[k, :,:,:,:] = sample4D_p
                batch_samples[k, :,:,:,:] = sample4D
                #batch_labels.append(pos_neg)
                batch_errors.append([err])
                batch_errors_p.append([err_p])
            
            yield (batch_samples, batch_errors, params), (batch_samples_p, batch_errors_p, params_p)


    # ----- #
    def load_mhd_as_sitkImage(self, fn_mhd, pixel_type=None):
        """
        """
        rawImg, header = mu.load_raw_data_with_mhd(fn_mhd)
        if not pixel_type == None:
            rawImg = rawImg.astype(np.float64)
            rawImg *= 255.0 / rawImg.max()
            #rawImg = rawImg.astype(np.uint16)
        
        #Jittering
        #rawImg=augment.jitter(rawImg, std=10)  
        
        img = sitk.GetImageFromArray(rawImg)
        img.SetOrigin(header['Offset'])
        img.SetSpacing(header['ElementSpacing'])
        
        return img
            

    # ----- #
    def create_sample(self, case_index, sample_dim, mat_trans=None):
        caseFolder = 'Case{:04d}'.format(case_index)

        folder = path.join(self.data_folder, caseFolder)
        
        if not path.isdir(folder):
            return None
        
        #
        # Start volume sampling
        #
        fn_fixed = path.join(folder, 'MRVol_adjusted.mhd')
        #fixedImg = sitk.ReadImage(fn_fixed)
        fixedImg = self.load_mhd_as_sitkImage(fn_fixed, sitk.sitkFloat64)
        
        fn_moving = path.join(folder, 'USVol.mhd')
        #movingImg = sitk.ReadImage(fn_moving)
        movingImg = self.load_mhd_as_sitkImage(fn_moving, sitk.sitkFloat64)
        
        # MR prostate segmentation
        fn_stl = path.join(folder, 'segmentationrtss.uronav.stl')
        segMesh = mesh.Mesh.from_file(fn_stl)
        
        def objective_func(parameters):
            
            tX = parameters[0]
            tY = parameters[1]
            tZ = parameters[2]
            
            angleX = parameters[3]
            angleY = parameters[4]
            angleZ = parameters[5]
            
            mat, t = self.create_transform(angleX, angleY, angleZ,
                                       tX, tY, tZ, self.current_reg)
        
            translation = self.current_reg[:3,3]
            
            arrTrans_us2mr = np.identity(4)
            arrTrans_us2mr[:3,:3] = mat
            arrTrans_us2mr[:3, 3] = translation + t

            error_trans = evaluator.evaluate_transform(arrTrans_us2mr)
            
            return error_trans
        
        

        if mat_trans is None:
            # Create registration evaluator
            evaluator = regev.RegistrationEvaluator(folder)
            mat_us2mr = evaluator.get_gt_registration()
            
            #sample_error = random.random() * self.max_error

            """
            signed = False
            rangeAngle = 5.0 # degree
            centerAngle = -0.5 * rangeAngle
            rangeTranslation = 2.0 # mm
            centerTranslation = -0.5 * rangeTranslation
            """
            
            signed = True
                            
            # Get random rotation and translation
            #
            angleX = self._get_random_value(6, 0, signed)
            angleY = self._get_random_value(6, 0, signed)
            angleZ = self._get_random_value(6, 0, signed)
            
            tX = self._get_random_value(5, 0, signed)
            #tX, angleY, angleZ = 0,0,0
            tY = self._get_random_value(5, 0, signed)
            # larger Z translation to simulate real senario
            tZ = self._get_random_value(5, 0, signed)

            # print('Translations: {}, {}, {}'.format(tX, tY, tZ))
            # print('Angles: {}, {}, {}'.format(angleX, angleY, angleZ))
            
            # mat_all, t_all = self.create_transform(angleX, angleY, angleZ,
            #                                        tX, tY, tZ, mat_us2mr)
        
            translation = mat_us2mr[:3,3]
            
            # arrTrans_us2mr = np.identity(4)
            # arrTrans_us2mr[:3,:3] = mat_all
            # arrTrans_us2mr[:3, 3] = translation + t_all
            parameters = np.asarray([tX, tY, tZ, angleX, angleY, angleZ])
            arrTrans_us2mr = dlf.construct_matrix_degree(parameters, mat_us2mr)
            mat_all = arrTrans_us2mr[:3,:3]
            t_all = np.asarray((tX, tY, tZ))

            self.current_reg = arrTrans_us2mr
            
            error_trans = evaluator.evaluate_transform(arrTrans_us2mr)
            #print('****Registration error = {}'.format(error_trans))

            #if error_trans > 0.1:
            #
            # Scaling the trans parameters to approximate uniform distribution
            # of surface registration error
            #
            #error_trans_old = error_trans
            """
            error_scale = sample_error / error_trans
            if error_scale > self.max_error / 2:
                error_scale /= 2.0
            tX *= error_scale
            tY *= error_scale
            tZ *= error_scale
            angleX *= error_scale
            angleY *= error_scale
            angleZ *= error_scale
            """

            # Warning: Replaced by the code segment below on 2019-08-12
            parameters = scipy.optimize.fmin_bfgs(objective_func,
                                                  [0,0,0,0,0,0],
                                                  fprime=None,
                                                  args=(),
                                                  gtol=1e-05,
                                                  norm=np.inf,
                                                  epsilon=1.4901161193847656e-08,
                                                  maxiter=None,
                                                  full_output=0,
                                                  disp=0,
                                                  retall=0,
                                                  callback=None)
            # print('Optimized parameters:')
            # print(parameters)

            # # new code segment by Yan on 2019-08-12
            # params_current = dlf.decompose_matrix(arrTrans_us2mr)
            # params_gt = dlf.decompose_matrix(mat_us2mr)
            # parameters_0 = params_gt - params_current
            # parameters_0[3:] = parameters_0[3:] / np.pi * 180

            # print('Decomposed parameters:')
            # print(parameters_0)
            # print(parameters - parameters_0)
            parameters = - parameters
        else:
            #pos_neg = True
            error_trans = 0.0
            
            mat_all = mat_trans[:3,:3]
            translation = mat_trans[:3,3]
            
            angleX = 0
            angleY = 0
            angleZ = 0
            
            tX = 0
            tY = 0
            tZ = 0
            #
            t_all = np.asarray((tX, tY, tZ))
        
        itkTrans_us2mr = sitk.AffineTransform(3)
        itkTrans_us2mr.SetMatrix(np.reshape(mat_all, (9,)))
        
        itkTrans_us2mr.SetTranslation(translation + t_all)
        
        #
        # Create instance of VolumeResampler
        #
        sampler = vr.VolumeResampler(fixedImg, segMesh, movingImg, itkTrans_us2mr)

        (width, height, depth) = sample_dim
        sampledFixed, sampledMoving = sampler.resample(width, height, depth)

        return sampledFixed, sampledMoving, error_trans, parameters
    

    # ----- #
    def create_transform(self, aX, aY, aZ, tX, tY, tZ, mat_base):
        t_all = np.asarray((tX, tY, tZ))
        
        # Get the transform
        rotX = sitk.VersorTransform((1,0,0), aX / 180.0 * np.pi)
        matX = self.get_array_from_itk_matrix(rotX.GetMatrix())
        #
        rotY = sitk.VersorTransform((0,1,0), aY / 180.0 * np.pi)
        matY = self.get_array_from_itk_matrix(rotY.GetMatrix())
        #
        rotZ = sitk.VersorTransform((0,0,1), aZ / 180.0 * np.pi)
        matZ = self.get_array_from_itk_matrix(rotZ.GetMatrix())
        
        # Apply all the rotations
        #mat_all = matX.dot(matY.dot(matZ.dot(mat_base[:3,:3])))
        # Modified by Yan on 2019-08-12
        mat_all = matZ.dot(matY.dot(matX.dot(mat_base[:3,:3])))
        
        return mat_all, t_all
    
        
    
    


# %%
"""
img_rows, img_cols = 96, 96
depth = 32
# 
img_channels = 2

# mini batch size
mbs = 32

data_folder = '/home/data/uronav_data'
vdg_train = VolumeDataGenerator(data_folder, (71,749), max_registration_error=20) 
trainGen = vdg_train.generate_batch_perturbations(batch_size=mbs, 
                                    shape=(img_cols,img_rows,depth))
for i in range(10):
    GT_labels = next(trainGen)[1]

    y_true_0, y_true_p = tf.split(GT_labels, num_or_size_splits=2)
    print(y_true_p.eval(), y_true_p.eval())
    print(len(GT_labels.eval()))
    print(i+1)

"""


    


if __name__ == '__main__':
    
    ch = input('Do you want to test batch? (Y/N): ')
    if ch[0] in 'yY':
        testBatch = True
    else:
        testBatch = False

    import fuse_image
    
    data_folder = '/home/data/uronav_data'
    
    home_folder = path.expanduser('~')
    log_folder = path.join(home_folder, 'tmp')
    if not path.exists(log_folder):
        os.makedirs(log_folder)
    
    vdg_train = VolumeDataGenerator(data_folder, (141,749), max_registration_error=50)
    print('{} cases for training'.format(vdg_train.get_num_cases()))
    trainGen = vdg_train.generate(shuffle=True, shape=(96,72,48))
    
    vdg_val = VolumeDataGenerator(data_folder, (71,140))
    print('{} cases for validation'.format(vdg_val.get_num_cases()))
    valGen = vdg_val.generate()
    
    batch_trainGen = vdg_train.generate_batch()
    
    if testBatch:
        errors = []
        for i in range(50):
            samples, err = next(batch_trainGen)
            print('batch {} with {} samples'.format(i, len(err)))
            errors.extend(err)
        errors = np.asarray(errors)
        
        n, bins, patches = plt.hist(errors, 40, normed=1, facecolor='green', alpha=0.75)
        plt.savefig(path.join(log_folder, 'error_histogram.png'), dpi=600)
    else:
        while True:
            #
            # Get the next sample
            #
            sample, err, trans_params = next(trainGen)
        
            print(trans_params)
            
            print('error={}'.format(err))
            
            (depth, height, width, ch) = sample.shape
            fvol = sample[:,:,:,0]
            mvol = sample[:,:,:,1]
            
            z = depth >> 1
            ax_mr = fvol[z,:,:].astype(np.uint8)
            ax_us = mvol[z,:,:].astype(np.uint8)
            fusedImg_ax = fuse_image.fuse_images(ax_mr, ax_us)
            plt.figure()
            plt.imshow(fusedImg_ax)
            plt.show()
            
            y = height >> 1
            cor_mr = np.flipud(fvol[:,y,:].astype(np.uint8))
            cor_us = np.flipud(mvol[:,y,:].astype(np.uint8))
            fusedImg_cor = fuse_image.fuse_images(cor_mr, cor_us)
            plt.figure()
            plt.imshow(fusedImg_cor)
            plt.show()
            
            x = width >> 1
            sag_mr = np.transpose(fvol[:,:,x].astype(np.uint8))
            sag_us = np.transpose(mvol[:,:,x].astype(np.uint8))
            fusedImg_sag = fuse_image.fuse_images(sag_mr, sag_us)
            plt.figure()
            plt.imshow(fusedImg_sag)
            plt.show()
            
            ch = input('Do you want to continue? (Y/N): ')
            if ch[0] not in 'yY':
                break


