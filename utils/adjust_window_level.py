#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2017-08-27

@author: yanrpi

Automatically adjust window and level
"""

# %%

from os import path

import math
import sys

#from matplotlib import pyplot as plt
sys.path.append('../utils')
# import mhd_utils as mu
from utils import mhd_utils as mu
import numpy as np
import SimpleITK as sitk


# %%

class LUT:
    """Lookup Table - currently supports only gray scale value
    """
    
    def __init__(self, value_range_input, value_range_to_use):
        """
        """
        self.originalMax = max(value_range_input)
        self.originalMin = min(value_range_input)
        diff = self.originalMax - self.originalMin + 1
        significantBits = int(math.ceil(math.log(diff) / math.log(2)))
        
        self.lut = np.zeros(1<<significantBits, dtype=np.uint8)

        self.newMax = max(value_range_to_use)
        self.newMin = min(value_range_to_use)

        self.prepareLUT()
        
        
    def prepareLUT(self):
        """
        """
        output_max = 255.0
        scale = output_max / (self.newMax - self.newMin)
        
        for i in range(self.lut.size):
            lutItem = i + self.originalMin
            if (lutItem <= self.newMin):
                value = 0
            elif (lutItem <= self.newMax):
                value = int(scale * (lutItem - self.newMin) + 0.5)
            else:
                value = 0xff;
		            
            self.lut[i] = value
            
    def getLUT(self):
        return self.lut
    
# %%

def autoAdjustWL(img_in, cut_ratio=0.001):
    # print(img_in)
    pixelVal_max = np.max(img_in)
    pixelVal_min = np.min(img_in)
    pixelVal_cut = 0
    
    numberCut = img_in.size * cut_ratio
    # Compute the cut-off value
    num_bins = int(pixelVal_max + 1)
    hist, be = np.histogram(img_in, range(num_bins))

    for i in range(pixelVal_max, 0, -1):
        if np.sum(hist[i:]) > numberCut:
            pixelVal_cut = i
            break
    pixelVal_cut = max(255, pixelVal_cut)        

    print('Max: {}, Min: {}, Cut off value: {}'.format(pixelVal_max, pixelVal_min, pixelVal_cut))
    
    img_out = np.zeros(img_in.shape, dtype=np.uint8)
    
    table = LUT((pixelVal_min, pixelVal_max), (pixelVal_min, pixelVal_cut))
    lut = table.getLUT()
    #print(lut.shape, lut)
    
    depth, height, width = img_in.shape
    
    for d in range(depth):
        for h in range(height):
            for w in range(width):
                img_out[d,h,w] = lut[img_in[d,h,w] - pixelVal_min]
                
    return img_out


# %%

if __name__=='__main__':
    
#    fn_fixed = path.join(folder, 'MRVol.mhd')
#    (img, header) = mu.load_raw_data_with_mhd(fn_fixed)
#
#    img_adjusted = autoAdjustWL(img)
#    
#    #mu.write_mhd_file('/home/yan/tmp/MRVol_adjusted.mhd', img_adjusted)
#    img_itk = sitk.GetImageFromArray(img_adjusted)
#    img_itk.SetOrigin(header['Offset'])
#    img_itk.SetSpacing(header['ElementSpacing'])
#    fn_out = path.join(folder, 'MRVol_adjusted.mhd')
#    sitk.WriteImage(img_itk, fn_out)

    # Below is data_folder for Dropshot and finn
    #data_folder = '/home/data/uronav_data'
    data_folder = '/Users/yan/Documents/data/uronav_data'
    #
    # Apply to all cases
    #
    for caseIdx in range(460,470):
        caseFolder = 'Case{:04d}'.format(caseIdx)
        
        full_case = path.join(data_folder, caseFolder)
        
        if not path.isdir(full_case):
            continue
        
        fn_in = path.join(full_case, 'MRVol.mhd')
        fn_out = path.join(full_case, 'MRVol_adjusted.mhd')
        
        if path.exists(fn_in):
            #if not path.exists(fn_out):
            print('Processing case {}...'.format(caseIdx))
        
            (img, header) = mu.load_raw_data_with_mhd(fn_in)
        
            img_adjusted = autoAdjustWL(img)
            
            img_itk = sitk.GetImageFromArray(img_adjusted)
            img_itk.SetOrigin(header['Offset'])
            img_itk.SetSpacing(header['ElementSpacing'])
            
            sitk.WriteImage(img_itk, fn_out)
            #else:
            #    print('Case {} has been processed before'.format(caseIdx))
        else:
            print('MR image is not available for case {}!'.format(caseIdx))
