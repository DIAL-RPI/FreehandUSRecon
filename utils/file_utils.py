#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 22:25:02 2017

@author: yan
"""

# %%

import os
from os import path


# %%


def get_home_dir():
    '''return the home directory of the current user
    '''
    return path.expanduser('~')


def get_tmp_dir():
    '''return full path of '~/tmp'
    '''
    home_folder = get_home_dir()
    tmp_folder = path.join(home_folder, 'tmp')
    
    if not path.exists(tmp_folder):
        os.mkdir(tmp_folder)
        
    return tmp_folder