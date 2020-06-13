#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2017-08

@author: yanrpi

Resample 3D volume image using 2D planes, MPR

"""

# %%

from matplotlib import pyplot as plt
import numpy as np
from os import path
import SimpleITK as sitk

from utils import reg_evaluator as regev

# %%
class Resampler2D:
    
    def __init__(self, fixed_image, moving_image, m2f_transform):
        """
        """
        # Constructor
        self.fixedImg = fixed_image
        self.movingImg = moving_image
        self.trans_us2mr = m2f_transform


    def set_transform(self, m2f_trans):
        """
        """
        self.trans_us2mr = m2f_trans


    def resample(self, view='sag', loc=0.5):
        """
        """
        tx = sitk.AffineTransform(3)
        tx.SetMatrix(np.reshape(self.trans_us2mr[:3,:3], (9,)))
        #origin = np.asarray(fixedImg.GetOrigin())
        
        translation = self.trans_us2mr[:3,3]
        tx.SetTranslation(translation)
        # print('tx\n{}'.format(tx))

        # %
        spacing_fixed = np.asarray(self.fixedImg.GetSpacing())
        
        if 'sag' in view:
            rotTrans = sitk.VersorTransform((0, 1, 0), -np.pi/2)
            #print('Center of rotation = {}'.format(rotTrans.GetCenter()))
            rotTrans.SetCenter(self.fixedImg.GetOrigin())
            #print('Center of rotation = {}'.format(rotTrans.GetCenter()))
        elif 'cor' in view:
            rotTrans = sitk.VersorTransform((1, 0, 0), -np.pi/2)
            rotTrans.SetCenter(self.fixedImg.GetOrigin())
        else:
            rotTrans = None
        #resampleTrans.SetMatrix(rotTrans.GetMatrix())
        
        fixedImgSize = np.asarray(self.fixedImg.GetSize())
        #print(fixedImgSize)
        spacing_new = fixedImgSize / np.array([512., 512., 512.]) * spacing_fixed
        spacing_new[2] = spacing_new[0]
        #print(spacing_new, spacing_fixed)
        
        size_mm = fixedImgSize * spacing_fixed
        
        #spacing_fixed = fixedImg.GetSpacing()
        position_x = size_mm[0] * loc
        position_y = size_mm[1] * loc
        position_z = size_mm[2] * loc
        shift = (512 * spacing_new[2] - size_mm[2]) / 2.0
        
        if 'ax' in view:
            #print('Position z = {}'.format(position_z))
            vpOffset = sitk.TranslationTransform(3, (0, 0, position_z))
        elif 'sag' in view:
            #print('Position x = {}'.format(position_x))
            vpOffset = sitk.TranslationTransform(3, (position_x, 0, -shift))
            #vpOffset = sitk.TranslationTransform(3, (0, 0, 0))
        elif 'cor' in view:
            #print('Position y = {}'.format(position_y))
            vpOffset = sitk.TranslationTransform(3, (0, position_y, size_mm[2] + shift))
            #vpOffset = sitk.TranslationTransform(3, (0, 0, 0))
        
        origin = self.fixedImg.GetOrigin()
        #print('Fixed image origin = {}'.format(origin))
        trans_origin = sitk.TranslationTransform(3, origin)
        
        transMR = sitk.Transform()
        transMR.AddTransform(vpOffset)
        if rotTrans:
            transMR.AddTransform(rotTrans)
        
        #
        # Using the composite transformation 
        # (stack based, first in - last applied).
        # So the last one will be applied first.
        #
        overallTrans = sitk.Transform(tx.GetInverse())
        overallTrans.AddTransform(trans_origin.GetInverse())
        overallTrans.AddTransform(transMR)
        
        # Try resampling        
        viewplane2D = sitk.Image(512, 512, 1, sitk.sitkUInt8)
        viewplane2D.SetSpacing(spacing_new)

        #print(viewplane2D.GetSpacing())
        viewplane2D.SetOrigin(origin)
        
        resampleFilter_us = sitk.ResampleImageFilter()
        resampleFilter_us.SetReferenceImage(viewplane2D)
        resampleFilter_us.SetInterpolator(sitk.sitkLinear)
        resampleFilter_us.SetDefaultPixelValue(0)

        resampleFilter_us.SetTransform(overallTrans)
        
        usImg2D = resampleFilter_us.Execute(self.movingImg)
        
        usImgArray = sitk.GetArrayFromImage(usImg2D)[0,:,:]
        
        # MR resample
        #image2D.SetOrigin(origin)
        resampleFilter_mr = sitk.ResampleImageFilter()
        resampleFilter_mr.SetReferenceImage(viewplane2D)
        resampleFilter_mr.SetInterpolator(sitk.sitkLinear)
        resampleFilter_mr.SetDefaultPixelValue(0)
        #resampler_mr.SetTransform(resampleTrans)
        resampleFilter_mr.SetTransform(transMR)
        
        mrImg2D = resampleFilter_mr.Execute(self.fixedImg)
        mrImgArray = sitk.GetArrayFromImage(mrImg2D)[0,:,:]
        
        return mrImgArray.astype(np.uint8), usImgArray.astype(np.uint8)


# %%
if __name__ == '__main__':

    # % Load images
    
    #folder = '/home/yan/Dropbox/Data/UroNav_registration_data/Case0012'
    folder = '/home/data/uronav_data/Case0012'
    
    fn_fixed = path.join(folder, 'MRVol_adjusted.mhd')
    fixedImg = sitk.ReadImage(fn_fixed)
    #fixedImg = load_mhd_as_sitkImage(fn_fixed)
    # nda = sitk.GetArrayFromImage(image)
    #fixedImg.SetOrigin((0,0,0))
    
    fn_moving = path.join(folder, 'USVol.mhd')
    movingImg = sitk.ReadImage(fn_moving)
    
    fn_resampled = path.join('/home/yan/tmp', 'ResampledUS.mhd')
    
    
    # Load us2mr transform
    
    # All global transformations except translation are of the form:
    #   T(x)=A(x−c)+t+c
    #
    # In ITK speak (when printing your transformation):
    #
    # Matrix: the matrix A
    # Center: the point c
    # Translation: the vector t
    # Offset: t + c − A*c
    
    fn_reg = path.join(folder, 'coreg.txt')
    evaluator = regev.RegistrationEvaluator(folder)
    us2mrReg = evaluator.load_registration(fn_reg)
    
    # ===
    
    resampler = Resampler2D(fixedImg, movingImg, us2mrReg)
    
    views = ['ax', 'sag', 'cor']
    
    import cv2
    import fuse_image
    
    for v in views:
        mr_array, us_array = resampler.resample(view=v, loc=0.5)
    
        fusedImg = fuse_image.fuse_images(mr_array, us_array)
        plt.figure()
        plt.imshow(cv2.cvtColor(fusedImg, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        #plt.imshow(fusedImg)
        
