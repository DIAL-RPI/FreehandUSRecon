#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 11:49:45 2018

@author: haskig
"""

import SimpleITK as sitk
from os import path
from utils import volume_resampler_3d as vr3D
from utils import reg_evaluator as regev
import numpy as np
from stl import mesh
import matplotlib.pyplot as plt
import scipy


class write_resized_imgs():
    def __init__(self, n):
        self.n=n
        self.folder = '/home/haskig/data/uronav_data'
    
    def get_path_US(self, m):
        #n refers to the case number. This will output a path
        folder = '/home/haskig/data/uronav_data'
        return path.join(folder, 'Case{:04}/USVol.mhd'.format(m))
    
    def get_path_MR(self, m):
        #n refers to the case number. This will output a path
        folder = '/home/haskig/data/uronav_data'
        return path.join(folder, 'Case{:04}/MRVol_adjusted.mhd'.format(m))
    
    def get_segmentation(self, m):
        folder = '/home/haskig/data/uronav_data'
        path_seg = path.join(folder, 'Case{:04}/segmentationrtss.uronav.stl'.format(m))
        return mesh.Mesh.from_file(path_seg)
    
    def load_itk(self, enter_path):
        itkimage=sitk.ReadImage(enter_path)
        image = sitk.Cast(itkimage, sitk.sitkUInt8)
        #img = sitk.GetArrayFromImage(image)
        
        return image
    
    def get_reg(self, m):
        folder = '/home/data/uronav_data/Case{:04}'.format(m)
        fn_reg = path.join(folder, 'coreg.txt')
        evaluator = regev.RegistrationEvaluator(folder)
        return evaluator.load_registration(fn_reg)
    
    def write_ITK_metaimage(self, volume, name, order=None):
        """
        Writes a ITK metaimage file, which can be viewed by Paraview.
        See http://www.itk.org/Wiki/ITK/MetaIO/Documentation
        Generates a raw file containing the volume, and an associated mhd
        metaimge file.
        TODO: Order should not be given as an argument, but guessed from the
        layout of the numpy array (possibly modified).
        TODO: Use compressed raw binary to save space. Should be possible, but
        given the lack of documentation it is a pain in the ass.
        Parameters
        ----------
        volume: array to be converted to mhd file
        name : string
            Name of the metaimage file.
        """
        if order is None:
            order = [2, 1, 0]
        assert len(volume.shape) == 3
        #print("* Writing ITK metaimage " + name + "...")
        # Write volume data
        with open(name + ".raw", "wb") as raw_file:
            raw_file.write(bytearray(volume.astype(np.uint8).flatten()))
        # Compute meta data
        if volume.dtype == np.float32:
            typename = 'MET_FLOAT'
        elif volume.dtype == np.uint8:
            typename = 'MET_UCHAR'
        else:
            raise RuntimeError("Incorrect element type: " + volume.dtype)
        # Write meta data
        with open(name + ".mhd", "w") as meta_file:
            basename = path.basename(name)
            meta_file.write("ObjectType = Image\nNDims = 3\n")
            meta_file.write(
                "DimSize = " + str(volume.shape[order[0]]) + " " +
                str(volume.shape[order[1]]) + " " +
                str(volume.shape[order[2]]) + "\n")
            meta_file.write(
                "ElementType = {0}\nElementDataFile = {1}.raw\n".format(
                typename, basename))

  
    
    def write_imgs(self):

        for i in range(1,self.n+1):
            try:
                path_US = self.get_path_US(i)
                path_MR = self.get_path_MR(i)
                seg = self.get_segmentation(i)
                m2f_transform = self.get_reg(i)
                US_vol = self.load_itk(path_US)
                MR_vol = self.load_itk(path_MR)
                vr3d = vr3D.VolumeResampler(MR_vol, seg, 
                                                US_vol, m2f_transform,
                                                enlargeRatio = 0.3)
                MR_vol, US_vol = vr3d.resample(96,96,32)
                MR_vol = sitk.GetArrayFromImage(MR_vol)
                US_vol = sitk.GetArrayFromImage(US_vol)
                US_path_new = path.join(self.folder, 'Case{:04}/USVol_resized'.format(i))
                self.write_ITK_metaimage(US_vol, US_path_new)
                MR_path_new = path.join(self.folder, 'Case{:04}/MRVol_resized'.format(i))
                self.write_ITK_metaimage(MR_vol, MR_path_new)
                
            except (RuntimeError, FileNotFoundError):
                pass

resize = write_resized_imgs(70)
resize.write_imgs() 


idx = 20
folder = '/home/haskig/data/uronav_data'
US = sitk.GetArrayFromImage(sitk.ReadImage(
        path.join(folder, 'Case{:04}/USVol_resized.mhd'.format(idx))))
count = 0
for i in range(70):
    if not path.exists(path.join(folder,'Case{:04}/MRVol_adjusted.mhd'.format(i+1))):
        count += 1
print(count)









