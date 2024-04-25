# coding: utf-8
from __future__ import print_function
import os
import time
import random
from PIL import Image
import tensorflow as tf
import numpy as np
from utils import *
from model import *
from glob import glob
from skimage import color,filters
import argparse
import scipy.io as scio

#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

parser = argparse.ArgumentParser(description='')

parser.add_argument('--save_Sharp_full_dir', dest='save_Sharp_full_dir', default='./results/Sharp/full/HRMS/', help='directory for testing outputs')
parser.add_argument('--save_Sharp_reduce_dir', dest='save_Sharp_reduce_dir', default='./results/Sharp/reduce/HRMS/', help='directory for testing outputs')
parser.add_argument('--test_PAN_dir', dest='test_PAN_dir', default='./Dataset/Test/PAN/', help='directory for testing inputs')
parser.add_argument('--test_MS_dir', dest='test_MS_dir', default='./Dataset/Test/MS/', help='directory for testing inputs')
parser.add_argument('--Data_Sensor', dest='Data_Sensor', default='QB', help='Sensors: QB/IKONOS/WV2......')

args = parser.parse_args()
Data_Sensor = args.Data_Sensor


sess = tf.Session()
training = tf.placeholder_with_default(False, shape=(), name='training')
## Reduce test
#input_MS_size_H = 200
#input_MS_size_W = 962
#input_PAN_size_H = 800
#input_PAN_size_W = 3848

## Full test
input_MS_size_H = 200
input_MS_size_W = 200
input_PAN_size_H = 800
input_PAN_size_W = 800

input_MS_Scale_2 = tf.placeholder(tf.float32, [None, None, None, 4], name='input_MS')
input_PAN_Scale_1 = tf.placeholder(tf.float32, [None, None, None, 1], name='input_PAN')


[Input_MS_Scale_3, Input_PAN_Scale_2] = downgrade_images(input_MS_Scale_2, input_PAN_Scale_1, 4, sensor=Data_Sensor,flag_PAN_MTF=1)

Sharpened_MS_Scale_1 = Sharpening_Net(input_MS_Scale_2,input_PAN_Scale_1,input_MS_size_H,input_MS_size_W,training = False)
Sharpened_MS_Scale_2 = Sharpening_Net(Input_MS_Scale_3,Input_PAN_Scale_2,input_MS_size_H/4,input_MS_size_W/4,training = True)


# load pretrained model
var_Sharpening = [var for var in tf.trainable_variables() if 'Sharpening_Net' in var.name]
g_list = tf.global_variables()

saver_Sharpening = tf.train.Saver(var_list=var_Sharpening)


Sharpening_checkpoint_dir ='./checkpoint/QuickBird/Sharpening_Net/'
Sharpening_ckpt_pre=tf.train.get_checkpoint_state(Sharpening_checkpoint_dir)
if Sharpening_ckpt_pre:
    print('loaded '+Sharpening_ckpt_pre.model_checkpoint_path)
    saver_Sharpening.restore(sess,Sharpening_ckpt_pre.model_checkpoint_path)
else:
    print('No Sharpening checkpoint!')


save_Sharp_full_dir = args.save_Sharp_full_dir
if not os.path.isdir(save_Sharp_full_dir):
    os.makedirs(save_Sharp_full_dir)

save_Sharp_reduce_dir = args.save_Sharp_reduce_dir
if not os.path.isdir(save_Sharp_reduce_dir):
    os.makedirs(save_Sharp_reduce_dir)

    
 ###load eval data
eval_PAN_data = []
eval_PAN_img_name =[]
eval_MS_data = []
eval_MS_img_name =[]
eval_PAN_data_name = glob(args.test_PAN_dir+'*')
eval_PAN_data_name.sort()
eval_MS_data_name = glob(args.test_MS_dir+'*')
eval_MS_data_name.sort()

for idx in range(len(eval_PAN_data_name)):
    [_, name]  = os.path.split(eval_PAN_data_name[idx])        
    suffix = name[name.find('.') + 1:]
    name = name[:name.find('.')]
    eval_PAN_img_name.append(name)
    eval_PAN_im = load_images_no_norm(eval_PAN_data_name[idx])
    eval_PAN_im = np.expand_dims(eval_PAN_im,2)
    eval_MS_im = load_images_no_norm(eval_MS_data_name[idx])
    h,w,c = eval_PAN_im.shape
    h_tmp = h%1
    w_tmp = w%1
    eval_PAN_im_resize = eval_PAN_im[0:h-h_tmp, 0:w-w_tmp, :]
    eval_MS_im_resize = eval_MS_im[0:h-h_tmp, 0:w-w_tmp, :]
    eval_PAN_data.append(eval_PAN_im_resize)
    eval_MS_data.append(eval_MS_im_resize)


print("Start evalating!")
start_time = time.time()
for idx in range(len(eval_PAN_data)):
    print(idx)
    name = eval_PAN_img_name[idx]
    input_PAN_im = eval_PAN_data[idx]
    input_MS_im = eval_MS_data[idx]    
    input_PAN_eval = np.expand_dims(input_PAN_im, axis=0)
    input_MS_eval = np.expand_dims(input_MS_im, axis=0)

    [Full_HRMS,Reduce_HRMS] = sess.run([Sharpened_MS_Scale_1,Sharpened_MS_Scale_2], feed_dict={input_MS_Scale_2: input_MS_eval,input_PAN_Scale_1: input_PAN_eval})
    Full_HRMS = np.squeeze(Full_HRMS)*255
    Reduce_HRMS = np.squeeze(Reduce_HRMS)*255
    Full_image_path = os.path.join(save_Sharp_full_dir,str(name)+".mat")
    scio.savemat(Full_image_path, {'I':Full_HRMS})
    Reduce_image_path = os.path.join(save_Sharp_reduce_dir,str(name)+".mat")
    scio.savemat(Reduce_image_path, {'I':Reduce_HRMS})