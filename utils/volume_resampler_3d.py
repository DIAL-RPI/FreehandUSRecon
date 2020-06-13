#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Resample 3D subvolumes according to the prostate segmentation size
"""

# %%

from matplotlib import pyplot as plt
import numpy as np
from os import path
import SimpleITK as sitk
from stl import mesh

from utils import fuse_image
from utils import reg_evaluator as regev

np.set_printoptions(suppress=True)

# %% Volume Resampler Class

class VolumeResampler():
    """
    """

    def __init__(self, fixed_image, mesh_bounds, moving_image,
                 m2f_transform, enlargeRatio=0.3):
        """
        """
        # Constructor
        self.fixedImg = fixed_image
        self.mesh_bounds = mesh_bounds
        self.movingImg = moving_image
        self.trans_us2mr = m2f_transform

        self.set_transform(m2f_transform)

        #
        # Compute prostate size using segmentation
        #
        # num_triangle = self.segMesh.points.shape[0]
        # markers = np.reshape(self.segMesh.points, (num_triangle*3,3))
        # print(markers)

        # Get lower and upper bounds in DICOM space
        # self.lb = np.min(markers, axis=0)
        # self.ub = np.max(markers, axis=0)
        self.lb = self.mesh_bounds[0]
        self.ub = self.mesh_bounds[1]
        # print('lb\n{}'.format(self.lb))
        # print('ub\n{}'.format(self.ub))
        # print(np.median(markers, axis=0))
        # print(self.ub)

        prostateSize = self.ub - self.lb
        # extend in each bound direction
        self.enlargedBound = prostateSize * (1.0 + enlargeRatio * 2)

        self.corner = self.lb - enlargeRatio * prostateSize

        self.lb = self.lb - enlargeRatio * prostateSize
        self.ub = self.ub + enlargeRatio * prostateSize
        
        
    def set_transform(self, m2f_transform):
        """
        """
        if type(m2f_transform) is sitk.AffineTransform:
            self.trans_us2mr = m2f_transform
        else:
            self.mat_us2mr = m2f_transform
            self.trans_us2mr = sitk.AffineTransform(3)
            self.trans_us2mr.SetMatrix(np.reshape(m2f_transform[:3,:3], (9,)))
            translation = m2f_transform[:3,3]
            self.trans_us2mr.SetTranslation(translation)



    def resample(self, width, height, depth):
        """
        """
        #(width, height, depth) = shape_out
        
        #destVol = sitk.Image(width, height, depth, sitk.sitkUInt16)
        destVol = sitk.Image(width, height, depth, sitk.sitkUInt8)
        
        destSpacing = self.enlargedBound / np.asarray((width, height, depth))
        #print('Sampling spacing = {}'.format(destSpacing))
        
        destVol.SetSpacing((destSpacing[0], destSpacing[1], destSpacing[2]))
        
        #offset = self.corner - np.asarray(origin)
        trans_corner = sitk.TranslationTransform(3, self.corner.astype(np.float64))
        
        # %
        
        resampler_mr = sitk.ResampleImageFilter()
        resampler_mr.SetReferenceImage(destVol)
        resampler_mr.SetInterpolator(sitk.sitkLinear)
        resampler_mr.SetDefaultPixelValue(0)
        
        # Apply translation
        resampler_mr.SetTransform(trans_corner)
        
        outFixedImg = resampler_mr.Execute(self.fixedImg)
        
        # %
        
        resampler_us = sitk.ResampleImageFilter()
        resampler_us.SetReferenceImage(destVol)
        resampler_us.SetInterpolator(sitk.sitkLinear)
        resampler_us.SetDefaultPixelValue(0)
        
        origin = self.fixedImg.GetOrigin()
        trans_origin = sitk.TranslationTransform(3, origin)

        trans_com_us = sitk.Transform(self.trans_us2mr.GetInverse())
        trans_com_us.AddTransform(trans_origin.GetInverse())
        trans_com_us.AddTransform(trans_corner)
        
        resampler_us.SetTransform(trans_com_us)
        
        outMovingImg = resampler_us.Execute(self.movingImg)
        
        #return ndaFixed, ndaMoving
        return outFixedImg, outMovingImg


    def resample_fixed_spacing(self, width, height, depth):
        """Resample volume without changing the resolution of the fixed volume
        """        
        #destVol = sitk.Image(width, height, depth, sitk.sitkUInt16)
        destVol = sitk.Image(width, height, depth, sitk.sitkUInt8)
        
        destSpacing = np.asarray(self.fixedImg.GetSpacing())
        #destSpacing = np.asarray([1.61187077, 1.24527003, 4.69228655])
        #print('Sampling spacing = {}'.format(destSpacing))
        
        imgSize = np.asarray(self.fixedImg.GetSize())
        
        destVol.SetSpacing((destSpacing[0], destSpacing[1], destSpacing[2]))
        #destVol.SetSpacing(self.fixedImg.GetSpacing())
        
        origin = self.fixedImg.GetOrigin()
        shift_x = destSpacing[0] * (imgSize[0] - width) / 2.0
        shift_y = destSpacing[1] * (imgSize[1] - height) / 2.0
        offset = np.asarray(origin) + np.asarray([shift_x, shift_y, 0])
        trans_corner = sitk.TranslationTransform(3, offset.astype(np.float64))
        
        # %
        
        resampler_mr = sitk.ResampleImageFilter()
        resampler_mr.SetReferenceImage(destVol)
        resampler_mr.SetInterpolator(sitk.sitkLinear)
        resampler_mr.SetDefaultPixelValue(0)
        
        # Apply translation
        resampler_mr.SetTransform(trans_corner)
        
        outFixedImg = resampler_mr.Execute(self.fixedImg)
        
        # %
        
        resampler_us = sitk.ResampleImageFilter()
        resampler_us.SetReferenceImage(destVol)
        resampler_us.SetInterpolator(sitk.sitkLinear)
        resampler_us.SetDefaultPixelValue(0)
        
        #origin = self.fixedImg.GetOrigin()
        trans_origin = sitk.TranslationTransform(3, origin)

        trans_com_us = sitk.Transform(self.trans_us2mr.GetInverse())
        trans_com_us.AddTransform(trans_origin.GetInverse())
        trans_com_us.AddTransform(trans_corner)
        
        resampler_us.SetTransform(trans_com_us)
        
        outMovingImg = resampler_us.Execute(self.movingImg)
        
        #return ndaFixed, ndaMoving
        return outFixedImg, outMovingImg
    
    
    def determine_bbox(self, img):
        #plt.figure()
        #plt.imshow(img, cmap='gray')
        
        h, w = img.shape
        proj = np.sum(img, axis=1)
        
        th = 255 * 10
        
        for i in range(h):
            if proj[i] > th:
                break
        
        for j in range(h-1, 1, -1):
            if proj[j] > th:
                break
            
        #print(i,j,h)
        return i, j
            
    
    # 
    def resample_overlap(self):
        """Resample the overalp between enlarged MR segmentation, MR volume and US volume
        """
        
        #us_img_size = self.movingImg.GetSize()
        mvol = sitk.GetArrayFromImage(self.movingImg)
        projs = []
        for n in range(3):
            projs.append(np.mean(mvol, axis=n))
        bot, top = self.determine_bbox(projs[0])
        #bot += 20
        #top -= 20
        left, right = self.determine_bbox(np.transpose(projs[1]))
        front, back = self.determine_bbox(projs[2])
        #print('Bounding box:')
        #print(mvol.shape, bot, top, left, right, front, back)
        
        us_img_spacing = self.movingImg.GetSpacing()
        #us_size_mm = np.asarray(us_img_size) * np.asarray(us_img_spacing)
        
        corner_points = np.zeros((8,4))
        cnt = 0
        for i in [left*us_img_spacing[0], right*us_img_spacing[0]]:
            for j in [bot*us_img_spacing[1], top*us_img_spacing[1]]:
                for k in [front*us_img_spacing[2], back*us_img_spacing[2]]:
                    corner_points[cnt, :] = np.asarray([i, j, k, 1.0])
                    cnt += 1
        #print(corner_points)
        
        origin = self.fixedImg.GetOrigin()
        mat_origin = np.identity(4)
        mat_origin[:3, 3] = origin
        
        mat_chain_us2mr = mat_origin.dot(self.mat_us2mr)
        transformed_points = np.zeros((8,4))
        for n in range(8):
            transformed_points[n,:] = mat_chain_us2mr.dot(corner_points[n,:])
        #print(transformed_points)
        
        us_lb = np.min(transformed_points, axis=0)
        us_ub = np.max(transformed_points, axis=0)
        #print('US bounds:')
        #print(us_lb)
        #print(us_ub)
        #print('Segmentation bounds:')
        #print(self.lb)
        #print(self.ub)
        
        mr_lb = np.asarray(origin)
        mr_size_mm = np.asarray(self.fixedImg.GetSpacing()) * np.asarray(self.fixedImg.GetSize())
        mr_ub = mr_lb + mr_size_mm
        
        lb_final = np.max(np.asarray([self.lb, us_lb[:3], mr_lb]), axis=0)
        ub_final = np.min(np.asarray([self.ub, us_ub[:3], mr_ub]), axis=0)
        #print('Final bounds:')
        #print(lb_final)
        #print(ub_final)

        # Compute dest volume size        
        destSpacing = np.asarray(self.fixedImg.GetSpacing())
        destSpacing[2] = max(destSpacing[0:2])
        
        shape = (ub_final - lb_final) / destSpacing + 0.5
        width, height, depth = int(shape[0]), int(shape[1]), int(shape[2])
        #print('resampled image size: {}, {}, {}'.format(width, height, depth))

        destVol = sitk.Image(width, height, depth, sitk.sitkUInt8)
        destVol.SetSpacing((destSpacing[0], destSpacing[1], destSpacing[2]))

        # compute the corner
        #offset = self.corner - np.asarray(origin)
        #trans_corner = sitk.TranslationTransform(3, self.corner.astype(np.float64))
        trans_corner = sitk.TranslationTransform(3, lb_final.astype(np.float64))
        
        # %
        
        resampler_mr = sitk.ResampleImageFilter()
        resampler_mr.SetReferenceImage(destVol)
        resampler_mr.SetInterpolator(sitk.sitkLinear)
        resampler_mr.SetDefaultPixelValue(0)
        
        # Apply translation
        resampler_mr.SetTransform(trans_corner)
        
        outFixedImg = resampler_mr.Execute(self.fixedImg)
        
        # %
        
        resampler_us = sitk.ResampleImageFilter()
        resampler_us.SetReferenceImage(destVol)
        resampler_us.SetInterpolator(sitk.sitkLinear)
        resampler_us.SetDefaultPixelValue(0)
        
        trans_origin = sitk.TranslationTransform(3, origin)

        trans_com_us = sitk.Transform(self.trans_us2mr.GetInverse())
        trans_com_us.AddTransform(trans_origin.GetInverse())
        trans_com_us.AddTransform(trans_corner)
        
        resampler_us.SetTransform(trans_com_us)
        
        outMovingImg = resampler_us.Execute(self.movingImg)
        
        # TODO: Post-resample crop
        
        #return ndaFixed, ndaMoving
        return outFixedImg, outMovingImg


# %%

if __name__=='__main__':
    # Load images
    
    #folder = '/Users/yan/Documents/data/uronav_data/Case0464'
    spacings = []
    for i in range(750):
        folder = '/home/data/uronav_data/Case{:04}'.format(i)
    
        if path.isdir(folder):    
            fn_fixed = path.join(folder, 'MRVol_adjusted.mhd')
            fixedImg = sitk.ReadImage(fn_fixed)
            # nda = sitk.GetArrayFromImage(image)
            origin = fixedImg.GetOrigin()
            #fixedImg.SetOrigin((0,0,0))
            
            fn_moving = path.join(folder, 'USVol.mhd')
            movingImg = sitk.ReadImage(fn_moving)
            #print('Size of moving image: {}'.format(movingImg.GetSize()))
            
            # MR prostate segmentation
            fn_stl = path.join(folder, 'segmentationrtss.uronav.stl')
            segMesh = mesh.Mesh.from_file(fn_stl)
            
            num_triangle = segMesh.points.shape[0]
            markers = np.reshape(segMesh.points, (num_triangle*3,3))
            
            import file_utils as fu
            tmp_dir = fu.get_tmp_dir()
            fn_resampled = path.join(tmp_dir, 'ResampledUS.mhd')
            
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
            #print(fn_reg)
            evaluator = regev.RegistrationEvaluator(folder)
            mat_us2mr = evaluator.load_registration(fn_reg)
            
            trans_us2mr = sitk.AffineTransform(3)
            trans_us2mr.SetMatrix(np.reshape(mat_us2mr[:3,:3], (9,)))
            
            #origin = np.asarray(fixedImg.GetOrigin())
            
            translation = mat_us2mr[:3,3]
            trans_us2mr.SetTranslation(translation)
        
            (width, height, depth) = (96,96,96)
            
            #
            # Create instance of VolumeResampler
            #
            #vr = VolumeResampler(fixedImg, segMesh, movingImg, trans_us2mr)
            vr = VolumeResampler(fixedImg, segMesh, movingImg, mat_us2mr, 0.9)
            #sampledFixed, sampledMoving = vr.resample(width, height, depth)
            _, _, spacing = vr.resample(96,96,32)
            spacings.append(spacing)
            
            """
            sampledFixed, sampledMoving = vr.resample_overlap()
            
            (width, height, depth) = sampledFixed.GetSize()
            #print(width, height, depth)
            
            fvol = sitk.GetArrayFromImage(sampledFixed)
            mvol = sitk.GetArrayFromImage(sampledMoving)
            
            #for i in range(3):
            #    plt.figure()
            #    plt.imshow(np.mean(mvol,axis=i), cmap='gray')
        
            # Display results to verify
            z = depth >> 1
            ax_mr = fvol[z,:,:].astype(np.uint8)
            ax_us = mvol[z,:,:].astype(np.uint8)
            fusedImg_ax = fuse_image.fuse_images(ax_mr, ax_us)
            plt.figure()
            plt.imshow(fusedImg_ax)
            
            y = height >> 1
            cor_mr = np.flipud(fvol[:,y,:].astype(np.uint8))
            cor_us = np.flipud(mvol[:,y,:].astype(np.uint8))
            fusedImg_cor = fuse_image.fuse_images(cor_mr, cor_us)
            plt.figure()
            plt.imshow(fusedImg_cor)
            
            x = width >> 1
            #x=0
            sag_mr = np.transpose(fvol[:,:,x].astype(np.uint8))
            sag_us = np.transpose(mvol[:,:,x].astype(np.uint8))
            fusedImg_sag = fuse_image.fuse_images(sag_mr, sag_us)
            plt.figure()
            plt.imshow(fusedImg_sag)
            #plt.close()
            """
    
        else:
            print('{} does not exist!'.format(folder))



