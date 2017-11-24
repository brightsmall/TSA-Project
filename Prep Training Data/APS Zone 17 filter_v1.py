# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 19:35:39 2017

@author: Administrator
"""
#updating process to concatenate as fill strip instead of as channels
#modified to test larger slice sizes

import numpy as np
import pandas as pd
import os
import matplotlib
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
import math
import cv2



def read_header(infile):
    """Read image header (first 512 bytes)
    """
    h = dict()
    fid = open(infile, 'r+b')
    h['filename'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 20))
    h['parent_filename'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 20))
    h['comments1'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 80))
    h['comments2'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 80))
    h['energy_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['config_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['file_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['trans_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['scan_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['data_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['date_modified'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 16))
    h['frequency'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['mat_velocity'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['num_pts'] = np.fromfile(fid, dtype = np.int32, count = 1)
    h['num_polarization_channels'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['spare00'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['adc_min_voltage'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['adc_max_voltage'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['band_width'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['spare01'] = np.fromfile(fid, dtype = np.int16, count = 5)
    h['polarization_type'] = np.fromfile(fid, dtype = np.int16, count = 4)
    h['record_header_size'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['word_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['word_precision'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['min_data_value'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['max_data_value'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['avg_data_value'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['data_scale_factor'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['data_units'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['surf_removal'] = np.fromfile(fid, dtype = np.uint16, count = 1)
    h['edge_weighting'] = np.fromfile(fid, dtype = np.uint16, count = 1)
    h['x_units'] = np.fromfile(fid, dtype = np.uint16, count = 1)
    h['y_units'] = np.fromfile(fid, dtype = np.uint16, count = 1)
    h['z_units'] = np.fromfile(fid, dtype = np.uint16, count = 1)
    h['t_units'] = np.fromfile(fid, dtype = np.uint16, count = 1)
    h['spare02'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['x_return_speed'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['y_return_speed'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['z_return_speed'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['scan_orientation'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['scan_direction'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['data_storage_order'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['scanner_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['x_inc'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['y_inc'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['z_inc'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['t_inc'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['num_x_pts'] = np.fromfile(fid, dtype = np.int32, count = 1)
    h['num_y_pts'] = np.fromfile(fid, dtype = np.int32, count = 1)
    h['num_z_pts'] = np.fromfile(fid, dtype = np.int32, count = 1)
    h['num_t_pts'] = np.fromfile(fid, dtype = np.int32, count = 1)
    h['x_speed'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['y_speed'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['z_speed'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['x_acc'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['y_acc'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['z_acc'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['x_motor_res'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['y_motor_res'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['z_motor_res'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['x_encoder_res'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['y_encoder_res'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['z_encoder_res'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['date_processed'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 8))
    h['time_processed'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 8))
    h['depth_recon'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['x_max_travel'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['y_max_travel'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['elevation_offset_angle'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['roll_offset_angle'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['z_max_travel'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['azimuth_offset_angle'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['adc_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['spare06'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['scanner_radius'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['x_offset'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['y_offset'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['z_offset'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['t_delay'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['range_gate_start'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['range_gate_end'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['ahis_software_version'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['spare_end'] = np.fromfile(fid, dtype = np.float32, count = 10)
    return h


def read_data(infile):
    """Read any of the 4 types of image files, returns a numpy array of the image contents
    """
    extension = os.path.splitext(infile)[1]
    h = read_header(infile)
    nx = int(h['num_x_pts'])
    ny = int(h['num_y_pts'])
    nt = int(h['num_t_pts'])
    fid = open(infile, 'rb')
    fid.seek(512) #skip header
    if extension == '.aps' or extension == '.a3daps':
        if(h['word_type']==7): #float32
            data = np.fromfile(fid, dtype = np.float32, count = nx * ny * nt)
        elif(h['word_type']==4): #uint16
            data = np.fromfile(fid, dtype = np.uint16, count = nx * ny * nt)
        data = data * h['data_scale_factor'] #scaling factor
        data = data.reshape(nx, ny, nt, order='F').copy() #make N-d image
    elif extension == '.a3d':
        if(h['word_type']==7): #float32
            data = np.fromfile(fid, dtype = np.float32, count = nx * ny * nt)
        elif(h['word_type']==4): #uint16
            data = np.fromfile(fid, dtype = np.uint16, count = nx * ny * nt)
        data = data * h['data_scale_factor'] #scaling factor
        data = data.reshape(nx, nt, ny, order='F').copy() #make N-d image
    elif extension == '.ahi':
        data = np.fromfile(fid, dtype = np.float32, count = 2* nx * ny * nt)
        data = data.reshape(2, ny, nx, nt, order='F').copy()
        real = data[0,:,:,:].copy()
        imag = data[1,:,:,:].copy()
    fid.close()
    if extension != '.ahi':
        return data
    else:
        return real, imag
    


def extract_left_side(image_data):
    
   
    image_strip = image_data[:,260:540,0]
    
    for i in range(1,16):
        image_strip = np.concatenate((image_strip,image_data[:,260:540,i]),axis=0)       
    return image_strip

def extract_mirror_side(image_data):
  
    mirror_image = np.empty([512,660,16],dtype='float32')
    mirror_image[:,:,0] = np.flip(image_data[:,:,0],axis=0)
    mirror_image[:,:,1] = np.flip(image_data[:,:,15],axis=0)
    mirror_image[:,:,2] = np.flip(image_data[:,:,14],axis=0)
    mirror_image[:,:,3] = np.flip(image_data[:,:,13],axis=0)
    mirror_image[:,:,4] = np.flip(image_data[:,:,12],axis=0)
    mirror_image[:,:,5] = np.flip(image_data[:,:,11],axis=0)
    mirror_image[:,:,6] = np.flip(image_data[:,:,10],axis=0)
    mirror_image[:,:,7] = np.flip(image_data[:,:,9],axis=0)
    mirror_image[:,:,8] = np.flip(image_data[:,:,8],axis=0)
    mirror_image[:,:,9] = np.flip(image_data[:,:,7],axis=0)
    mirror_image[:,:,10] = np.flip(image_data[:,:,6],axis=0)
    mirror_image[:,:,11] = np.flip(image_data[:,:,5],axis=0)
    mirror_image[:,:,12] = np.flip(image_data[:,:,4],axis=0)
    mirror_image[:,:,13] = np.flip(image_data[:,:,3],axis=0)
    mirror_image[:,:,14] = np.flip(image_data[:,:,2],axis=0)
    mirror_image[:,:,15] = np.flip(image_data[:,:,1],axis=0)
    
    image_strip = mirror_image[:,260:540,0]     
    for i in range(1,16):
        image_strip = np.concatenate((image_strip,mirror_image[:,260:540,i]),axis=0)
    return image_strip

def random_stretch(image_data):
    output_img = np.zeros([512,660,16], dtype = "float32")  
    height_change = np.random.randint(low=-40, high=41,size=1)
    for j in range(16):
        new_height = 660 + height_change
        stretch_img = cv2.resize(image_data[:,:,j],( new_height,512), interpolation = cv2.INTER_CUBIC)
        top_padding = np.zeros([512,40], dtype = "float32") 
        padded_img = np.concatenate((stretch_img,top_padding), axis=1)
        cropped_img = padded_img[:,0:660]
        output_img[:,:,j] = cropped_img
    return output_img                


def random_width(image_data):
    output_img = np.zeros([512,660,16], dtype = "float32")  
    width_change = np.random.randint(low=-20, high=21,size=1)*2
    for j in range(16):
        new_width = 512 + width_change
        stretch_img = cv2.resize(image_data[:,:,j],( 660,new_width), interpolation = cv2.INTER_CUBIC)
        pad_width = int(40 - int(width_change)/2)
        side_padding = np.zeros([pad_width,660], dtype = "float32")
        padded_img = np.concatenate((side_padding,stretch_img,side_padding), axis=0)
        cropped_img = padded_img[40:552,0:660]
        output_img[:,:,j] = cropped_img
    return output_img 




################################################
# create lists with file names and ids


id_list=[]

#write_folder = "D:\\TSA\\stage 1\\aps\\left leg\\"

for file in os.listdir("D:\\TSA\\stage 1\\aps"):
    if file.endswith(".aps"):
        id_list.append(str(file).replace(".aps",""))

# create df with labels

raw_df = pd.read_csv('D:\\TSA\\stage1_labels.csv')
split_labels = raw_df['Id'].str.split('_')
new_id = [x[0] for x in split_labels]
zone = [x[1] for x in split_labels]
raw_df['Id'] = new_id
raw_df['Zone'] = zone
zone17_labels = raw_df[raw_df['Zone']=='Zone17']
zone17_labels.reset_index(inplace=True, drop=True)


del raw_df, split_labels, new_id, zone



#load all data to get max and mean
#group_max = 0
#group_mean = 0
#
#for i in range(len(id_list)):
#    raw_data = read_data("D:\\TSA\\stage 1\\aps\\" + id_list[i] +".aps")
#    current_max = np.max(raw_data)
#    group_max = max([group_max,current_max])
#    group_mean = group_mean + np.mean(raw_data)/len(id_list)
#    
#    
group_max = 0.0019839206
group_mean = 5.416645786519416e-05

#loop through id_list and sort by pos, neg, test


pos_folder =  "D:\\TSA\\stage 1\\aps\\Zone 17\\pos\\"
neg_folder =  "D:\\TSA\\stage 1\\aps\\Zone 17\\neg\\"
test_folder = "D:\\TSA\\stage 1\\aps\\Zone 17\\test\\"

if not os.path.exists(pos_folder):
    os.makedirs(pos_folder)
if not os.path.exists(neg_folder):
    os.makedirs(neg_folder)   
if not os.path.exists(test_folder):
    os.makedirs(test_folder)  
        
# real training data
        
for i,item in enumerate(id_list):
    image_data = read_data("D:\\TSA\\stage 1\\aps\\" + item +".aps")
    image_data = (image_data - group_mean)
    image_data = image_data *2 / (group_max - group_mean)
    
    img_strip = extract_left_side(image_data)
    if zone17_labels['Id'].str.contains(item).any():
        label = int(zone17_labels[zone17_labels['Id']==item]['Probability'])
        if label == 1:
            np.save(pos_folder + item, img_strip)
        else:
            np.save(neg_folder + item, img_strip)
    else:
        np.save(test_folder + item, img_strip)
        
# first augmentation        
        
for i,item in enumerate(id_list):
    image_data = read_data("D:\\TSA\\stage 1\\aps\\" + item +".aps")
    image_data = (image_data - group_mean)
    image_data = image_data *2 / (group_max - group_mean)
    
    image_data = random_stretch(image_data)
    image_data = random_width(image_data)
    
    
    img_strip = extract_left_side(image_data)
    if zone17_labels['Id'].str.contains(item).any():
        label = int(zone17_labels[zone17_labels['Id']==item]['Probability'])
        if label == 1:
            np.save(pos_folder + "aug1_"+ item, img_strip)
        else:
            np.save(neg_folder + "aug1_"+ item, img_strip)
    
# second augmentation                

for i,item in enumerate(id_list):
    image_data = read_data("D:\\TSA\\stage 1\\aps\\" + item +".aps")
    image_data = (image_data - group_mean)
    image_data = image_data *2 / (group_max - group_mean)
    
    image_data = random_stretch(image_data)
    image_data = random_width(image_data)
    
    
    img_strip = extract_left_side(image_data)
    if zone17_labels['Id'].str.contains(item).any():
        label = int(zone17_labels[zone17_labels['Id']==item]['Probability'])
        if label == 1:
            np.save(pos_folder + "aug2_"+ item, img_strip)
        else:
            np.save(neg_folder + "aug2_"+ item, img_strip)       



################################################

# real training data for opposite side
        
test_folder = "D:\\TSA\\stage 1\\aps\\Zone 17\\test_mirror\\"

if not os.path.exists(test_folder):
    os.makedirs(test_folder) 
        
for i,item in enumerate(id_list):
    
    image_data = read_data("D:\\TSA\\stage 1\\aps\\" + item +".aps")
    image_data = (image_data - group_mean)
    image_data = image_data *2 / (group_max - group_mean)
    
    img_strip = extract_mirror_side(image_data)
    
    if zone17_labels['Id'].str.contains(item).any():
        label = int(zone17_labels[zone17_labels['Id']==item]['Probability'])
        if label == 1:
            np.save(pos_folder + item + "_mr", img_strip)
        else:
            np.save(neg_folder + item + "_mr", img_strip)
    #else:
        #np.save(test_folder + item +"_mr", img_strip)

# first augmentation

for i,item in enumerate(id_list):
    image_data = read_data("D:\\TSA\\stage 1\\aps\\" + item +".aps")
    image_data = (image_data - group_mean)
    image_data = image_data *2 / (group_max - group_mean)
    image_data = random_stretch(image_data)
    image_data = random_width(image_data)
    img_strip = extract_mirror_side(image_data)
 
    if zone17_labels['Id'].str.contains(item).any():
        label = int(zone17_labels[zone17_labels['Id']==item]['Probability'])
        if label == 1:
            np.save(pos_folder +"aug1_"+ item + "_mr", img_strip)
        else:
            np.save(neg_folder +"aug1_"+ item + "_mr", img_strip)


# second augmentation
        
for i,item in enumerate(id_list):
    image_data = read_data("D:\\TSA\\stage 1\\aps\\" + item +".aps")
    image_data = (image_data - group_mean)
    image_data = image_data *2 / (group_max - group_mean)
    image_data = random_stretch(image_data)
    image_data = random_width(image_data)
    img_strip = extract_mirror_side(image_data)
    if zone17_labels['Id'].str.contains(item).any():
        label = int(zone17_labels[zone17_labels['Id']==item]['Probability'])
        if label == 1:
            np.save(pos_folder +"aug2_"+ item + "_mr", img_strip)
        else:
            np.save(neg_folder +"aug2_"+ item + "_mr", img_strip)
