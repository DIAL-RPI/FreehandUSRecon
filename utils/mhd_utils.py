#!/usr/bin/env python

#
# Original Version: bjian 2008/10/27
# 3-D extension:    PJackson 2013/06/06        
# More datatypes, Multiple Channels, Python 3, ...: Peter Fischer
#
# %%

from __future__ import division, print_function
import os
import numpy as np
import array


# %%

def read_meta_header(filename):
    """Return a dictionary of meta data from meta header file"""
    fileIN = open(filename, "r")
    line = fileIN.readline()

    meta_dict = {}
    tag_set = []
    tag_set.extend(['ObjectType', 'NDims', 'DimSize', 'ElementType', 'ElementDataFile', 'ElementNumberOfChannels'])
    tag_set.extend(['BinaryData', 'BinaryDataByteOrderMSB', 'CompressedData', 'CompressedDataSize'])
    tag_set.extend(['Offset', 'CenterOfRotation', 'AnatomicalOrientation', 'ElementSpacing', 'TransformMatrix'])
    tag_set.extend(['Comment', 'SeriesDescription', 'AcquisitionDate', 'AcquisitionTime', 'StudyDate', 'StudyTime'])

    tag_flag = [False] * len(tag_set)
    while line:
        tags = str.split(line, '=')
        # print(tags[0])
        for i in range(len(tag_set)):
            tag = tag_set[i]
            if (str.strip(tags[0]) == tag) and (not tag_flag[i]):
                # print(tags[1])
                content = str.strip(tags[1])
                if tag in ['ElementSpacing', 'Offset', 'CenterOfRotation', 'TransformMatrix']:
                    meta_dict[tag] = [float(s) for s in content.split()]
                elif tag in ['NDims', 'ElementNumberOfChannels']:
                    meta_dict[tag] = int(content)
                elif tag in ['DimSize']:
                    meta_dict[tag] = [int(s) for s in content.split()]
                elif tag in ['BinaryData', 'BinaryDataByteOrderMSB', 'CompressedData']:
                    if content == "True":
                        meta_dict[tag] = True
                    else:
                        meta_dict[tag] = False
                else:
                    meta_dict[tag] = content
                tag_flag[i] = True
        line = fileIN.readline()
    # print(comment)
    fileIN.close()
    return meta_dict


def load_raw_data_with_mhd(filename):
    meta_dict = read_meta_header(filename)
    dim = int(meta_dict['NDims'])
    if "ElementNumberOfChannels" in meta_dict:
        element_channels = int(meta_dict["ElementNumberOfChannels"])
    else:
        element_channels = 1
    # print(dim)
    # print(meta_dict['ElementType'])
    if meta_dict['ElementType'] == 'MET_FLOAT':
        np_type = np.float32
    elif meta_dict['ElementType'] == 'MET_DOUBLE':
        np_type = np.float64
    elif meta_dict['ElementType'] == 'MET_CHAR':
        np_type = np.byte
    elif meta_dict['ElementType'] == 'MET_UCHAR':
        np_type = np.ubyte
    elif meta_dict['ElementType'] == 'MET_SHORT':
        np_type = np.short
    elif meta_dict['ElementType'] == 'MET_USHORT':
        np_type = np.ushort
    elif meta_dict['ElementType'] == 'MET_INT':
        np_type = np.int32
    elif meta_dict['ElementType'] == 'MET_UINT':
        np_type = np.uint32
    else:
        raise NotImplementedError("ElementType " + meta_dict['ElementType'] + " not understood.")
    arr = list(meta_dict['DimSize'])
    # print(arr)
    volume = np.prod(arr[0:dim - 1])
    # print(volume)
    pwd = os.path.split(filename)[0]
    if pwd:
        data_file = pwd + '/' + meta_dict['ElementDataFile']
    else:
        data_file = meta_dict['ElementDataFile']

    shape = (arr[dim - 1], volume, element_channels)
    with open(data_file,'rb') as fid:
        data = np.fromfile(fid, count=np.prod(shape),dtype = np_type)
    data.shape = shape

    # swap bytes if 'BinaryDataByteOrderMSB' is True
    if meta_dict['BinaryDataByteOrderMSB'] == True:
        data = data.byteswap()
        meta_dict['BinaryDataByteOrderMSB'] = False

    # Begin 3D fix
    arr.reverse()
    if element_channels > 1:
        data = data.reshape(arr + [element_channels])
    else:
        data = data.reshape(arr)
    # End 3D fix
    
    return (data, meta_dict)


def write_meta_header(filename, meta_dict):
    header = ''
    # do not use tags = meta_dict.keys() because the order of tags matters
    tags = ['ObjectType', 'NDims', 'BinaryData',
            'BinaryDataByteOrderMSB', 'CompressedData', 'CompressedDataSize',
            'TransformMatrix', 'Offset', 'CenterOfRotation',
            'AnatomicalOrientation', 'ElementSpacing',
            'DimSize', 'ElementNumberOfChannels', 'ElementType', 'ElementDataFile',
            'Comment', 'SeriesDescription', 'AcquisitionDate',
            'AcquisitionTime', 'StudyDate', 'StudyTime']
    for tag in tags:
        if tag in meta_dict.keys():
            header += '%s = %s\n' % (tag, meta_dict[tag])
    f = open(filename, 'w')
    f.write(header)
    f.close()


def dump_raw_data(filename, data, dsize, element_channels=1):
    """ Write the data into a raw format file. Big endian is always used. """
    data = data.reshape(dsize[0], -1, element_channels)
    rawfile = open(filename, 'wb')
    if data.dtype == np.float32:
        array_string = 'f'
    elif data.dtype == np.double or data.dtype == np.float64:
        array_string = 'd'
    elif data.dtype == np.short:
        array_string = 'h'
    elif data.dtype == np.ushort:
        array_string = 'H'
    elif data.dtype == np.int32:
        array_string = 'i'
    elif data.dtype == np.uint32:
        array_string = 'I'
    else:
        raise NotImplementedError("ElementType " + str(data.dtype) + " not implemented.")
    a = array.array(array_string)
    a.fromlist(list(data.ravel()))
    # if is_little_endian():
    #    a.byteswap()
    a.tofile(rawfile)
    rawfile.close()


def write_mhd_file(mhdfile, data, **meta_dict):
    assert(mhdfile[-4:] == '.mhd')
    meta_dict['ObjectType'] = 'Image'
    meta_dict['BinaryData'] = 'True'
    meta_dict['BinaryDataByteOrderMSB'] = 'False'
    if data.dtype == np.float32:
        meta_dict['ElementType'] = 'MET_FLOAT'
    elif data.dtype == np.double or data.dtype == np.float64:
        meta_dict['ElementType'] = 'MET_DOUBLE'
    elif data.dtype == np.byte:
        meta_dict['ElementType'] = 'MET_CHAR'
    elif data.dtype == np.uint8 or data.dtype == np.ubyte:
        meta_dict['ElementType'] = 'MET_UCHAR'
    elif data.dtype == np.short or data.dtype == np.int16:
        meta_dict['ElementType'] = 'MET_SHORT'
    elif data.dtype == np.ushort or data.dtype == np.uint16:
        meta_dict['ElementType'] = 'MET_USHORT'
    elif data.dtype == np.int32:
        meta_dict['ElementType'] = 'MET_INT'
    elif data.dtype == np.uint32:
        meta_dict['ElementType'] = 'MET_UINT'
    else:
        raise NotImplementedError("ElementType " + str(data.dtype) + " not implemented.")
    dsize = list(data.shape)
    if 'ElementNumberOfChannels' in meta_dict.keys():
        element_channels = int(meta_dict['ElementNumberOfChannels'])
        assert(dsize[-1] == element_channels)
        dsize = dsize[:-1]
    else:
        element_channels = 1
    dsize.reverse()
    meta_dict['NDims'] = str(len(dsize))
    meta_dict['DimSize'] = dsize
    meta_dict['ElementDataFile'] = os.path.split(mhdfile)[1].replace('.mhd',
                                                                     '.raw')

    # Tags that need conversion of list to string
    tags = ['ElementSpacing', 'Offset', 'DimSize', 'CenterOfRotation', 'TransformMatrix']
    for tag in tags:
        if tag in meta_dict.keys() and not isinstance(meta_dict[tag], str):
            meta_dict[tag] = ' '.join([str(i) for i in meta_dict[tag]])
    write_meta_header(mhdfile, meta_dict)

    pwd = os.path.split(mhdfile)[0]
    if pwd:
        data_file = pwd + '/' + meta_dict['ElementDataFile']
    else:
        data_file = meta_dict['ElementDataFile']

    dump_raw_data(data_file, data, dsize, element_channels=element_channels)
