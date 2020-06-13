#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 17:47:34 2019

@author: labadmin
"""
import nibabel as nib
import matplotlib.pyplot as plt
import itertools
import SimpleITK as sitk
import numpy as np
import scipy
from skimage.measure import regionprops,label
import os
import zipfile
import re
from PIL import Image, ImageDraw
# import dicom
from matplotlib import pyplot as plt
import numpy as np
import pickle
from skimage import draw
from scipy.linalg import norm
import math
import copy
import scipy.ndimage
import SimpleITK as sitk
import time
import nibabel as nib

re_digits = re.compile(r'(\d+)')  


def Data_cheack():
    file_list = os.listdir('./')
#    print(file_list)
    for cid in range(77,204):
      case_id = "PEx{:04d}_00000000".format(cid)+''
      if case_id in file_list:
        print(case_id)
        dicoms_path = case_id+'/dicoms/'
        nifti_path = case_id+'/nifti/'        
        file_level2 = ['adc/', 'highb/', 't2/']
        file_level3 = ['aligned/', 'raw/']  
        
        ###########cheack nill##############
        cheack_space = []
        cheack_size = []
        for fid in file_level2:
            IM_list = os.listdir(nifti_path +fid)
#            print(IM_list)
            img = sitk.ReadImage(nifti_path +fid+IM_list[0])
            original_spacing = img.GetSpacing()
            original_size = img.GetSize()
            resampled_sitk_IM = sitk.GetArrayFromImage(img)
#            print('original_size:',resampled_sitk_IM.shape) 
            if original_spacing not in cheack_space:
                cheack_space.append(original_spacing)
  
            if original_size not in cheack_size:
                cheack_size.append(original_size)
        if len(cheack_space) + len(cheack_size)>2:
            print("!!!!!!!!!!!!!!!!!!!!!!ERROR##############%d"%cid)  
                  
        ###########cheack nill##############
        cheack_num = []
        cheack_size = []
        for fid2 in file_level2[0:2]:
            for fid3 in file_level3:
                IM_list = os.listdir(dicoms_path + fid2 + fid3)
#                print(IM_list)
                file_num = len(IM_list)
                if file_num not in cheack_num:
                    cheack_num.append(file_num)  
                for dic in IM_list:
                    img = sitk.ReadImage(dicoms_path + fid2 + fid3+dic)
                    original_spacing = img.GetSpacing()
                    resampled_sitk_IM = sitk.GetArrayFromImage(img)
                    if resampled_sitk_IM.shape[0] < 0:
                        print("!!!!!!!!!!!!!!!!!!!!!!ERROR##############%d"%cid,dicoms_path + fid2 + fid3)  
        fid2 = file_level2[2]
        IM_list = os.listdir(dicoms_path + fid2)
#        print(IM_list)
        for dic in IM_list:
            img = sitk.ReadImage(dicoms_path + fid2 +dic)
            original_spacing = img.GetSpacing()
            resampled_sitk_IM = sitk.GetArrayFromImage(img)
            if resampled_sitk_IM.shape[0] < 0:
                print("!!!!!!!!!!!!!!!!!!!!!!ERROR##############%d"%cid,dicoms_path + fid2)  
                                

def embedded_numbers(s):  
     pieces = re_digits.split(s)               # 切成数字与非数字  
     pieces[1::2] = map(int, pieces[1::2])     # 将数字部分转成整数  
     return pieces  
def sort_strings_with_embedded_numbers(alist):  
     return sorted(alist, key=embedded_numbers)


re_digits = re.compile(r'(\d+)')  
def embedded_numbers(s):  
     pieces = re_digits.split(s)               # 切成数字与非数字  
     pieces[1::2] = map(int, pieces[1::2])     # 将数字部分转成整数  
     return pieces  
def sort_strings_with_embedded_numbers(alist):  
     return sorted(alist, key=embedded_numbers)
 

#%%
def conver_dicom():
    for id in range(0,81):
        path='/home/kui/prostate_segment/prostate_data_all/pat%d/'%id
        files1 = os.listdir(path)
        print(len(files1))
        files2 = os.listdir(path+files1[0])
        print(files2)
        print(files2[0][-3:])
        if files2[0][-3:]=='voi':
            path2=path+files1[0]+'/'+files2[1]
        else:
            path2=path+files1[0]+'/'+files2[0]
        print(path2)
        files3 = os.listdir(path2)
        path3=path2+'/'+files3[0]   
        files4 =  os.listdir(path3)
        print(files4)
        import shutil
        if len(files4):
            for f in files4:
                shutil.copyfile(path3+'/'+f,path+f)  
                
                
def get_min_indx(name):
    idx=[]
    for li in name:
        if li[2]=='0':
            ID=int(li[1:2])
        else:
            ID=int(li[1:3])
#        print(li)
#        print(ID)
        idx.append(ID)
    return np.min((idx))

def Get_GT_Array(num_file,path,IM_list):
 
    min_id=get_min_indx(IM_list)
    IM_str='I'+str(min_id)+'000000'
    RefDs = dicom.read_file(path+IM_str)
    
    
    
    spacing = map(float, ([RefDs.SliceThickness] + RefDs.PixelSpacing))
    spacing = np.array(list(spacing))
    print(spacing)
    GT_array=np.zeros((len(IM_list),RefDs.Rows,RefDs.Columns),dtype=np.uint8)
#    print(GT_array.shape)
    full_path1="/home/kui/prostate_segment/prostate_data_all/pat00%d_Study_1.zip"%(num_file+1)
    full_path2="/home/kui/prostate_segment/prostate_data_all/pat0%d_Study_1.zip"%(num_file+1)
    
    if num_file+1<10:
        zip_name=full_path1
    else:
        zip_name=full_path2
        
    z = zipfile.ZipFile(zip_name,'r')
    
    ##打印zip文件中的文件列表
    for file_list in z.namelist():
          #print ('File:', file_list)
          if ~file_list.find('.voi'):
             GT_file=file_list
    ##GroundTruth的处理
    content = z.read(GT_file)
    new=content.decode("utf-8")
#    print(new)
    slice_num=[]
    totle_num=[]
    count_x=[]
    count_y=[]
    #寻找slice number
    pattern2=r'[0-9]+\t\t# slice number'
    regex=re.compile(pattern2)
    slice_str=regex.findall(new)
 
    for num in slice_str:
        slice=num.split('\t')
        slice_num.append(int(slice[0]))
#    print('slice_num:',slice_num)
    
    #寻找number of pts in contour
    pattern3=r'[0-9]+\t\t# number of pts in contour <Chain-element-type>1</Chain-element-type>'
    regex=re.compile(pattern3)
    totle_str=regex.findall(new)
    for num in totle_str:
        totle_num.append(int(num.split('\t')[0]))
#    print('totle_num:',totle_num)

#    #提取轮廓点
    pattern1=r'[0-9]*\.?[0-9]+ [0-9]*\.?[0-9]+'
    regex=re.compile(pattern1)
    point_index=regex.findall(new)
    for num in point_index:
        count_x.append(float(num.split(' ')[0]))
        count_y.append(float(num.split(' ')[1]))
        
        #每个轮廓的数目
    temp_num=0
    for Id in range(len(totle_num)):#lentotle_numlen(totle_num)
        num_point=totle_num[Id]
#        print(num_point)
            #依次画出groundtruth    
        X=(np.array(count_x))[temp_num:temp_num+num_point]
        Y=(np.array(count_y))[temp_num:temp_num+num_point]
        temp_num+=num_point
#            RefDs = dicom.read_file('I2100000')
        #对图像进行读取显示
    
        img = np.zeros((RefDs.Rows,RefDs.Columns),dtype=np.uint8)
        rr,cc=draw.polygon(Y,X)
        draw.set_color(img,[rr,cc],[1])
        GT_array[slice_num[Id]-1]=img
   
    print(GT_array.shape)
    return GT_array
    
    


def Get_IM_Array(path,IM_list):
        IM_array=[]
        min_id=get_min_indx(IM_list)
        for ID in range(min_id,len(IM_list)+min_id):#lentotle_numlen(totle_num)
            if ID<10:
                IM_str='I'+str(ID)+'000000'
            else:
                if (ID)%10==0:
                   IM_str='I'+str(ID)+'00001'
                else:
                   IM_str='I'+str(ID)+'00000' 
            
            RefDs = dicom.read_file(path+IM_str)
            img = np.zeros((RefDs.Rows,RefDs.Columns),dtype=RefDs.pixel_array.dtype)
            img=RefDs.pixel_array
            IM_array.append(img)
        IM_array=np.array(IM_array)
        print(IM_array.shape)
#        plt.figure('00001')
#        plt.imshow(IM_array[:,:,20],cmap='binary_r')
        return IM_array   
    
def deal_dicom():
    save_path='./Data/'
    for id in range(1,83):
        if id <10:
            str_id='0'+str(id)
        else:
            str_id=str(id)
        path='/home/kui/Pancreas_Segmentation/Pancreas/PANCREAS_00'+str_id
        file1_name = os.listdir(path)
        path=path+'/'+file1_name[0]
        file1_name = os.listdir(path)  
        path=path+'/'+file1_name[0]
        dicom_names = os.listdir(path)          
        dicom_names = sort_strings_with_embedded_numbers(dicom_names)
#        print((dicom_names))  
        print(len(dicom_names))
        IM_array=[]
        for dicom_name in dicom_names:
            RefDs = dicom.read_file(path+'/'+dicom_name)
#            spacing = map(float, ([RefDs.SliceThickness] + RefDs.PixelSpacing))
#            spacing = np.array(list(spacing))
#            print('spacing',RefDs.PixelSpacing)
            img = np.zeros((RefDs.Rows,RefDs.Columns),dtype=RefDs.pixel_array.dtype)
            img=RefDs.pixel_array
            IM_array.append(img)
        IM_array=np.array(IM_array)
        print(IM_array.shape)        
        np.save(save_path+'Original/IM%d'%id,IM_array)


def read_VOI_information(num_file,GT_array):
 
    
    full_path1="./prostate/pat00%d_Study_1.zip"%(num_file+1)
    full_path2="./prostate/pat0%d_Study_1.zip"%(num_file+1)
    
    if num_file+1<10:
        zip_name=full_path1
    else:
        zip_name=full_path2
        
    z = zipfile.ZipFile(zip_name,'r')

    
    ##打印zip文件中的文件列表
    for file_list in z.namelist():
          #print ('File:', file_list)
          if ~file_list.find('.voi'):
             GT_file=file_list    
    ##GroundTruth的处理
    content = z.read(GT_file)
    new=content.decode("utf-8")    
    
#    print(new)
    slice_num=[]
    totle_num=[]
    count_x=[]
    count_y=[]
    #寻找slice number
    pattern2=r'[0-9]+\t\t# slice number'
    regex=re.compile(pattern2)
    slice_str=regex.findall(new)
 
    for num in slice_str:
        slice=num.split('\t')
        slice_num.append(int(slice[0]))
#    print('slice_num:',slice_num)
    
    #寻找number of pts in contour
    pattern3=r'[0-9]+\t\t# number of pts in contour <Chain-element-type>1</Chain-element-type>'
    regex=re.compile(pattern3)
    totle_str=regex.findall(new)
    for num in totle_str:
        totle_num.append(int(num.split('\t')[0]))
#    print('totle_num:',totle_num)

#    #提取轮廓点
    pattern1=r'[0-9]*\.?[0-9]+ [0-9]*\.?[0-9]+'
    regex=re.compile(pattern1)
    point_index=regex.findall(new)
    for num in point_index:
        count_x.append(float(num.split(' ')[0]))
        count_y.append(float(num.split(' ')[1]))
        
        #每个轮廓的数目
    temp_num=0
    for Id in range(len(totle_num)):#lentotle_numlen(totle_num)
        num_point=totle_num[Id]
#        print(num_point)
            #依次画出groundtruth    
        X=(np.array(count_x))[temp_num:temp_num+num_point]
        Y=(np.array(count_y))[temp_num:temp_num+num_point]
        temp_num+=num_point
#            RefDs = dicom.read_file('I2100000')
        #对图像进行读取显示
    
        img = np.zeros_like(GT_array[0],dtype=np.uint8)
        rr,cc=draw.polygon(Y,X)
        draw.set_color(img,[rr,cc],[1])
        GT_array[slice_num[Id]-1]=img
   
    print(GT_array.shape)
    return GT_array

    
    
def read_prostate_data_from_pat():
    save_path='DATA/'
    for i in range(81):
        path='./prostate/Unpack/pat%d'%i
        file1_name = os.listdir(path)
        path=path+'/'+file1_name[0]
        file1_name = os.listdir(path)  
#        print(path)
        for filename in file1_name:
            if os.path.isdir(path+'/'+filename):
                path=path+'/'+filename  
                file1_name = os.listdir(path)  
                path=path+'/'+file1_name[0]
                print(path)
            else:
                voi_path=path
                print(voi_path)
                
                
    
#        file1_name = os.listdir(path)
#        print(file1_name)
        reader=sitk.ImageSeriesReader()
        print(path)
        dicom_names=reader.GetGDCMSeriesFileNames(path)
        reader.SetFileNames(dicom_names)
        image=reader.Execute()
        image_array=sitk.GetArrayFromImage(image)#z,y,x
        print(image_array.shape)
        
        
        GT_array = read_VOI_information(i,np.zeros_like(image_array))        
        print(GT_array.shape)        
        np.save(save_path+'GT%d.npy'%i,GT_array)
#####################Lood GroundTruth#######################
                
def conver_to_MHD():
    Orpath='/home/labadmin/ProstateSegment/PROMISE12/TestData/'
    Sepath='/home/labadmin/ProstateSegment/PROMISE12/TrainAndTest/Resample2D_resize_224_224_50/Re/OR/'
    save=Sepath+'/MHD/'
    for i in range(30):
        if i<10:
            imname='Case0%d.mhd'%i
        else:
            imname='Case%d.mhd'%i  
        Result=np.load(Sepath+'Case%d.npy'%i)
        print(Result.shape)
        
        img = sitk.ReadImage(Orpath+imname)
        image=(sitk.GetArrayFromImage(img))
        print('or:',image.shape)
        original_spacing = img.GetSpacing()
        print('original_spacing:',original_spacing)
        original_size = img.GetSize()
        print('original_size:',original_size)
        new_img=sitk.GetImageFromArray(Result)
        new_img.SetOrigin(img.GetOrigin())
        new_img.SetSpacing(img.GetSpacing())
        new_img.SetDirection(img.GetDirection())
        sitk.WriteImage(new_img, save+imname)

def Draw_GT(file_path, file_list,GT_array):
    
    ID = 0
    for file_name in file_list:
        ID += 1
        content  = open(file_path + file_name,'r')   
        new = content.read()
    #    new=content.decode("utf-8")    
        
    #    print(new)
        slice_num=[]
        totle_num=[]
        count_x=[]
        count_y=[]
        #寻找slice number
        pattern2=r'[0-9]+\t\t# slice number'
        regex=re.compile(pattern2)
        slice_str=regex.findall(new)
     #   print(slice_str)
     
        for num in slice_str:
            slice=num.split('\t')
            slice_num.append(int(slice[0]))
        print('slice_num:',slice_num)
        
        #寻找number of pts in contour
        pattern3=r'[0-9]+\t\t# number of pts in contour <Chain-element-type>1</Chain-element-type>'
        regex=re.compile(pattern3)
        totle_str=regex.findall(new)
        for num in totle_str:
            totle_num.append(int(num.split('\t')[0]))
        print('totle_num:',totle_num)
    
    #    #提取轮廓点
        pattern1=r'[0-9]*\.?[0-9]+ [0-9]*\.?[0-9]+'
        regex=re.compile(pattern1)
        point_index=regex.findall(new)
        for num in point_index:
            count_x.append(float(num.split(' ')[0]))
            count_y.append(float(num.split(' ')[1]))
            
            #每个轮廓的数目
        temp_num=0
        for Id in range(len(totle_num)):#lentotle_numlen(totle_num)
            num_point=totle_num[Id]
    #        print(num_point)
                #依次画出groundtruth    
            X=(np.array(count_x))[temp_num:temp_num+num_point]
            Y=(np.array(count_y))[temp_num:temp_num+num_point]
            temp_num+=num_point
    #            RefDs = dicom.read_file('I2100000')
            #对图像进行读取显示

            img = np.zeros_like(GT_array[0],dtype=np.uint8)
            rr,cc=draw.polygon(Y,X)
            draw.set_color(img,[rr,cc],[ID])
            GT_array[slice_num[Id]]+=img
        GT_array = np.clip(GT_array,0,ID)

    return GT_array            





def get_nifti_t2_GT():
    save_path = '/home/labadmin/DataProstate/MHDfile/'
    file_list = os.listdir('./')
#    print(file_list)
    for cid in range(201,204):
      case_id = "PEx{:04d}_00000000".format(cid)+''
      if case_id in file_list:
        print(case_id)
        IM_path = case_id+'/nifti/t2/' 
        GT_path = case_id+'/voi/' 
        file_level = ['wp_bt.voi','tz_bt.voi', 'urethra_bt.voi']#,'tz_bt.voi', 'urethra_bt.voi'
        IM_file_name = os.listdir(IM_path)
        img = sitk.ReadImage(IM_path+IM_file_name[0])
        new_img = sitk.GetArrayFromImage(img)
        new_gt = Draw_GT(GT_path, file_level,np.zeros_like(new_img))
        print(new_img.shape)
        print(new_gt)
        new_gt = sitk.GetImageFromArray(new_gt)

#        new_img.SetOrigin(img.GetOrigin())
#        new_img.SetSpacing(img.GetSpacing())
#        new_img.SetDirection(img.GetDirection())
        sitk.WriteImage(img, save_path+case_id+'.mhd')


        new_gt.SetOrigin(img.GetOrigin())
        new_gt.SetSpacing(img.GetSpacing())
        new_gt.SetDirection(img.GetDirection())
        sitk.WriteImage(new_gt, save_path+case_id+'_segmentation.mhd')        
        
#        if original_spacing not in cheack_space:
#            cheack_space.append(original_spacing)
#  
#        if original_size not in cheack_size:
#            cheack_size.append(original_size)    


def read_voi(file_name):

    content = open(file_name, 'r')
    new = content.read()
    # print('new\n{}'.format(new))

    slice_num = []
    totle_num = []
    count_x = []
    count_y = []
    # 寻找slice number
    pattern2 = r'[0-9]+\t\t# slice number'
    regex = re.compile(pattern2)
    slice_str = regex.findall(new)
    #   print(slice_str)

    for num in slice_str:
        slice = num.split('\t')
        slice_num.append(int(slice[0]))

    # 寻找number of pts in contour
    pattern3 = r'[0-9]+\t\t# number of pts in contour <Chain-element-type>1</Chain-element-type>'
    regex = re.compile(pattern3)
    totle_str = regex.findall(new)
    for num in totle_str:
        totle_num.append(int(num.split('\t')[0]))

    #    #提取轮廓点
    pattern1 = r'[0-9]*\.?[0-9]+ [0-9]*\.?[0-9]+'
    regex = re.compile(pattern1)
    point_index = regex.findall(new)
    # print('point index\n{}'.format(len(point_index)))
    # print('point index\n{}'.format(point_index))
    # time.sleep(30)
    for num in point_index:
        count_x.append(float(num.split(' ')[0]))
        count_y.append(float(num.split(' ')[1]))

        # 每个轮廓的数目

    slice_index_container = np.ones((1,))
    for slice_index, pt_num in zip(slice_num, totle_num):
        slice_index_array = np.ones((pt_num,)) * slice_index
        slice_index_container = np.concatenate((slice_index_container, slice_index_array))
    slice_index_container = slice_index_container[1:]
    slice_index_container = np.expand_dims(slice_index_container, axis=1)

    splited_coords = [i.lower().split() for i in point_index]
    splited_coords = np.asarray(splited_coords).astype(np.float32)

    coords_slice = np.concatenate((splited_coords, slice_index_container), axis=1)

    return coords_slice


                
if __name__ == '__main__':
    # get_nifti_t2_GT()
    
#    zip_name = '/home/labadmin/DataProstate/PEx0000_00000000/voi/'
#    z = zipfile.ZipFile(zip_name,'r')
#
#    
#    ##打印zip文件中的文件列表
#    for file_list in z.namelist():
#          #print ('File:', file_list)
#          if ~file_list.find('.voi'):
#             GT_file=file_list     
#

    mr_path = '/zion/guoh9/projects/reg4nih/data_sample/MRI_US_Reg_sample2/' \
              'volume-MRI.nii'
    us_path = '/zion/guoh9/projects/reg4nih/data_sample/MRI_US_Reg_sample2/' \
              'volume-_091029_3D.nii'
    data = nib.load(mr_path)
    img = data.get_data().astype(np.float32)
    img_header = data.header
    print('nii header\n{}'.format(img_header))
    print('nii datatype: {}'.format(img_header['quatern_b']))
    # time.sleep(30)
    img = np.transpose(img, [1, 0, 2])
    for i in range(img.shape[0]):
        img[i] = img[i] / 255
        img[i] = np.clip(img[i], -2., 2.)
    # img = np.moveaxis(img, 0, 2)
    print('img shape: {}'.format(img.shape))
    # sitk_img = Image.fromarray(img)
    # print('sitk_img size\n{}'.format(sitk_img.__sizeof__()))
    # time.sleep(30)

    file_name ='/zion/guoh9/projects/reg4nih/data_sample/MRI_US_Reg_sample2/' \
             'Right mid anterior TZ lesion_2nd session.voi'
    content = open(file_name, 'r')
    new = content.read()
    print('new\n{}'.format(new))

    slice_num = []
    totle_num = []
    count_x = []
    count_y = []
    # 寻找slice number
    pattern2 = r'[0-9]+\t\t# slice number'
    regex = re.compile(pattern2)
    slice_str = regex.findall(new)
    #   print(slice_str)

    for num in slice_str:
        slice = num.split('\t')
        slice_num.append(int(slice[0]))

    # 寻找number of pts in contour
    pattern3 = r'[0-9]+\t\t# number of pts in contour <Chain-element-type>1</Chain-element-type>'
    regex = re.compile(pattern3)
    totle_str = regex.findall(new)
    for num in totle_str:
        totle_num.append(int(num.split('\t')[0]))


    #    #提取轮廓点
    pattern1 = r'[0-9]*\.?[0-9]+ [0-9]*\.?[0-9]+'
    regex = re.compile(pattern1)
    point_index = regex.findall(new)
    # print('point index\n{}'.format(len(point_index)))
    # print('point index\n{}'.format(point_index))
    # time.sleep(30)
    for num in point_index:
        count_x.append(float(num.split(' ')[0]))
        count_y.append(float(num.split(' ')[1]))

        # 每个轮廓的数目
    print(point_index)
    print('totle_num: {}'.format(totle_num))
    print('slice_num: {}'.format(slice_num))
    print('points num: {}'.format(np.sum(totle_num)))
    time.sleep(30)
    temp_num = 0
    for Id in range(len(totle_num)):  # lentotle_numlen(totle_num)
        num_point = totle_num[Id]
        #        print(num_point)
        # 依次画出groundtruth
        X = (np.array(count_x))[temp_num:temp_num + num_point]
        Y = (np.array(count_y))[temp_num:temp_num + num_point]
        temp_num += num_point
        #            RefDs = dicom.read_file('I2100000')
        # 对图像进行读取显示

        img = np.zeros_like(GT_array[0], dtype=np.uint8)
        rr, cc = draw.polygon(Y, X)
        draw.set_color(img, [rr, cc], [ID])
        GT_array[slice_num[Id]] += img

##    content = z.read(GT_file)

 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    








                

    
    
