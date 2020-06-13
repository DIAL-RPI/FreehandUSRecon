# -*- coding: utf-8 -*-
"""
Fuse two images using pseudo color to encode one image and superimposing on the other.
"""

# %%

import cv2
import matplotlib.pyplot as plt
import numpy as np


# %%
def fuse_images(img_ref, img_folat, alpha=0.6):
    mask = (img_folat < 50).astype(np.float) 
    mask[mask < 1] = alpha
    mask_comp = 1.0 - mask
    
    img_color = cv2.applyColorMap(img_folat, cv2.COLORMAP_JET)
    #print(img_color.shape)

    dst = np.zeros((img_folat.shape[0], img_folat.shape[1], 3), dtype=np.uint8)

    for i in range(3):
        dst[:,:,i] = (img_ref * mask + img_color[:,:,i] * mask_comp).astype(np.uint8)

    return dst

# %%

if __name__=='__main__':
    fn_usResampled = "C:\\Temp\\usSection.png";
    fn_mrResampled = "C:\\Temp\\mrSection.png";
    fn_fusedImage = "C:\\Temp\\fusedSection.png";
    
    usImg = cv2.imread(fn_usResampled)
    us_gray = usImg[:,:,0]
    #usImg_color = cv2.merge((us_color, usImg[:,:,3]))
    
    mrImg = cv2.imread(fn_mrResampled)
    mrImg_ch = mrImg[:,:,0]
    #ch_eq = cv2.equalizeHist(mrImg_ch)
    #mrImg_eq = cv2.merge((ch_eq, ch_eq, ch_eq, mrImg[:,:,3]))    
    
    dst = fuse_images(mrImg_ch, us_gray, 0.6)

    cv2.imwrite(fn_fusedImage, dst)
    
    # 
    plt.imshow(dst)

