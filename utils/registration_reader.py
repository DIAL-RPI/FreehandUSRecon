#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 17:26:22 2017

@author: yan
"""
# %%


import numpy as np
from numpy import linalg
from utils import mhd_utils as mu
from os import path
import xml.etree.ElementTree as ET


# %%

def load_UroNav_registration(fn_reg_UroNav, fn_mhd):
    """Load UroNav registration matrix from 'coreg.txt'
    """
    mat_reg = np.loadtxt(fn_reg_UroNav)
    header = mu.read_meta_header(fn_mhd)
    offset = np.asarray(header['Offset'])
    
    mat_mr2us = np.identity(4)
    mat_mr2us[:3,:3] = mat_reg[1:,1:]
    mat_mr2us[:3,3] = mat_reg[1:,0]
    mat_us2mr_UroNav = linalg.inv(mat_mr2us)
    #print(linalg.inv(mat_us2mr))
    
    mat_shift = np.identity(4)
    mat_shift[:3,3] = - offset
    
    mat_us2mr = mat_shift.dot(mat_us2mr_UroNav)
    
    return mat_us2mr


# %
def load_gt_registration(folder_path):

    fn_reg = 'coreg.txt'
    fn_reg_refined = 'coreg_refined.txt'
    
    # By default, load the refined registration
    fn_reg_full = path.join(folder_path, fn_reg_refined)
    
    if not path.isfile(fn_reg_full):
        fn_reg_full = path.join(folder_path, fn_reg)

    #idx_case = folder_path.find('Case')
    # print('Loading <{}> for {}'.format(
    #         path.basename(fn_reg_full),
    #         folder_path[idx_case:idx_case+8]))
    gt_registration = load_registration(fn_reg_full)
    
    return gt_registration


# %
def load_registration(filename):
    if filename.endswith('coreg.txt'):
        fn_mr_full = path.join(path.dirname(filename), 'MRVol.mhd')
        if not path.isfile(fn_mr_full):
            return None
        
        mat_us2mr = load_UroNav_registration(filename, fn_mr_full)
    else:
        try:
            mat_us2mr = np.loadtxt(filename)
        except:
            print('Failed to load <{}>'.format(filename))

    return mat_us2mr

# %%

def load_registration_xml(xml_file):
    e = ET.parse(xml_file).getroot()
    mat=[]
    for i in e.iter():
        if i.text != '\n' and i.text != None:
            mat.append(float(i.text))
    
    matrix = np.array(mat)
    matrix.resize(3,4)
    matrix = np.vstack((matrix, [0, 0, 0, 1.0])) 

    return matrix