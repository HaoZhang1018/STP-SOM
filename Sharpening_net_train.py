# coding: utf-8
from __future__ import print_function
import os, time, random
import tensorflow as tf
from PIL import Image
import numpy as np
from utils import *
from model import *
from glob import glob
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=10, help='number of samples in one batch')
parser.add_argument('--MS_patch_size',  dest='MS_patch_size', type=int, default=120, help='MS_patch_size size')
parser.add_argument('--PAN_patch_size', dest='PAN_patch_size', type=int, default=480, help='PAN_patch_size')
parser.add_argument('--train_data_dir', dest='train_data_dir', default='./Dataset/Training/', help='directory for training inputs')
parser.add_argument('--Data_Sensor', dest='Data_Sensor', default='QB', help='Sensors: QB/IKONOS/WV2......')


args = parser.parse_args()

batch_size = args.batch_size
MS_patch_size = args.MS_patch_size
PAN_patch_size = args.PAN_patch_size
Data_Sensor = args.Data_Sensor

sess = tf.Session()

## Input
input_MS_Scale_2 = tf.placeholder(tf.float32, [None, None, None, 4], name='input_MS')
input_PAN_Scale_1 = tf.placeholder(tf.float32, [None, None, None, 1], name='input_PAN') 

[Input_MS_Scale_3, Input_PAN_Scale_2] = downgrade_images(input_MS_Scale_2, input_PAN_Scale_1, 4, sensor=Data_Sensor,flag_PAN_MTF=1)


Sharpened_MS_Scale_1 = Sharpening_Net(input_MS_Scale_2,input_PAN_Scale_1,MS_patch_size,MS_patch_size,training = True)
Sharpened_MS_Scale_2 = Sharpening_Net(Input_MS_Scale_3,Input_PAN_Scale_2,MS_patch_size/4,MS_patch_size/4,training = True)

###LOSS FUNCTION
#Unsupervised frame
[Sharpened_MS_Scale_1_SPAD,Input_PAN_Scale_2] = downgrade_images(Sharpened_MS_Scale_1, input_PAN_Scale_1, 4, sensor=Data_Sensor,flag_PAN_MTF=1) #Spatial Degradation
Sharpened_MS_Scale_1_SPAD_SPED = Spectral_De_Net(Sharpened_MS_Scale_1_SPAD, training = False)  #Spectral Degradation

Uns_SPED_loss = tf.reduce_mean(tf.abs(Sharpened_MS_Scale_1_SPAD_SPED-Input_PAN_Scale_2)) 
UnSup_loss =   1*Uns_SPED_loss


#Supervised frame
Sharpened_MS_Scale_2_SPED = Spectral_De_Net(Sharpened_MS_Scale_2, training = False)  #Spectral Degradation

Sup_loss_refer = tf.reduce_mean(tf.abs(Sharpened_MS_Scale_2-input_MS_Scale_2))
Sup_loss_SPED = tf.reduce_mean(tf.abs(Sharpened_MS_Scale_2_SPED-Input_PAN_Scale_2))
Sup_loss = 1*Sup_loss_refer  +  0.1*Sup_loss_SPED

Sharpening_loss_total =  10*Sup_loss+ 1* UnSup_loss

tf.summary.scalar('Uns_SPED_loss',Uns_SPED_loss)
tf.summary.scalar('Sup_loss_refer',Sup_loss_refer)
tf.summary.scalar('Sup_loss_SPED',Sup_loss_SPED)
tf.summary.scalar('Sharpening_loss_total',Sharpening_loss_total)


## Optimize Configuration
lr = tf.placeholder(tf.float32, name='learning_rate')

optimizer = tf.train.AdamOptimizer(learning_rate=lr, name='AdamOptimizer')
var_SDN = [var for var in tf.trainable_variables() if 'Spectral_De_Net' in var.name]
var_Sharpening = [var for var in tf.trainable_variables() if 'Sharpening_Net' in var.name]
g_list = tf.global_variables()

with tf.name_scope('train_step'):
  train_op_Sharpening  = optimizer.minimize(Sharpening_loss_total, var_list = var_Sharpening)


saver_SDN = tf.train.Saver(var_list=var_SDN,max_to_keep=8000)
saver_Sharpening = tf.train.Saver(var_list=var_Sharpening,max_to_keep=8000)

sess.run(tf.global_variables_initializer())

print("[*] Initialize model successfully...")

with tf.name_scope('image'):
  tf.summary.image('input_MS_Scale_2',tf.expand_dims(input_MS_Scale_2[1,:,:,:],0))
  tf.summary.image('input_PAN_Scale_1',tf.expand_dims(input_PAN_Scale_1[1,:,:,:],0))
  tf.summary.image('Sharpened_MS_Scale_1',tf.expand_dims(Sharpened_MS_Scale_1[1,:,:,:],0))
  tf.summary.image('Input_MS_Scale_3',tf.expand_dims(Input_MS_Scale_3[1,:,:,:],0))
  tf.summary.image('Input_PAN_Scale_2',tf.expand_dims(Input_PAN_Scale_2[1,:,:,:],0))  
  tf.summary.image('Sharpened_MS_Scale_2',tf.expand_dims(Sharpened_MS_Scale_2[1,:,:,:],0))
    
  tf.summary.image('Sharpened_MS_Scale_1_SPAD',tf.expand_dims(Sharpened_MS_Scale_1_SPAD[1,:,:,:],0))  
  tf.summary.image('Sharpened_MS_Scale_1_SPAD_SPED',tf.expand_dims(Sharpened_MS_Scale_1_SPAD_SPED[1,:,:,:],0))    

  
summary_op = tf.summary.merge_all()

train_writer = tf.summary.FileWriter('./log' + '/Sharpening_train',sess.graph,flush_secs=60)


#load data
###train_data
train_MS_data = []
train_PAN_data = []
train_MS_data_names = glob(args.train_data_dir +'/MS/*.tif') 
train_MS_data_names.sort()
train_PAN_data_names = glob(args.train_data_dir +'/PAN/*.tif') 
train_PAN_data_names.sort()
assert len(train_MS_data_names) == len(train_PAN_data_names)
print('[*] Number of training data: %d' % len(train_PAN_data_names))
for idx in range(len(train_MS_data_names)):
    MS_im = load_images_no_norm(train_MS_data_names[idx])    
    train_MS_data.append(MS_im)
    PAN_im = load_images_no_norm(train_PAN_data_names[idx])
    PAN_im = np.expand_dims(PAN_im,2)
    train_PAN_data.append(PAN_im)


epoch = 8000
learning_rate = 0.0001
train_phase = 'Sharpening'
numBatch = len(train_PAN_data) // int(batch_size)


SDN_checkpoint_dir = './checkpoint/SDN/'
if not os.path.isdir(SDN_checkpoint_dir):
    os.makedirs(SDN_checkpoint_dir)
ckpt_SDN=tf.train.get_checkpoint_state(SDN_checkpoint_dir)
if ckpt_SDN:
    print('loaded '+ckpt_SDN.model_checkpoint_path)
    saver_SDN.restore(sess,ckpt_SDN.model_checkpoint_path)
else:
    print('No SDN pretrained model!')



Sharpening_checkpoint_dir = './checkpoint/QuickBird/Sharpening_Net/'
if not os.path.isdir(Sharpening_checkpoint_dir):
    os.makedirs(Sharpening_checkpoint_dir)
ckpt_sharpening=tf.train.get_checkpoint_state(Sharpening_checkpoint_dir)
if ckpt_sharpening:
    print('loaded '+ckpt_sharpening.model_checkpoint_path)
    saver_Sharpening.restore(sess,ckpt_sharpening.model_checkpoint_path)
else:
    print('No Sharpening pretrained model!')


start_step = 0
start_epoch = 0
iter_num = 0

print("[*] Start training for phase %s, with start epoch %d start iter %d : " % (train_phase, start_epoch, iter_num))
start_time = time.time()
image_id = 0
counter = 0
for epoch in range(start_epoch, epoch):
    for batch_id in range(start_step, numBatch):
        batch_input_PAN = np.zeros((batch_size, PAN_patch_size, PAN_patch_size, 1), dtype="float32")
        batch_input_MS = np.zeros((batch_size, MS_patch_size, MS_patch_size, 4), dtype="float32")
        for patch_id in range(batch_size):
            h_ms, w_ms,_ = train_MS_data[image_id].shape
            h_pan, w_pan,_ = train_PAN_data[image_id].shape
            x_ms = random.randint(0, h_ms - MS_patch_size)
            y_ms = random.randint(0, w_ms - MS_patch_size)           
            rand_mode = random.randint(0, 7)
            batch_input_PAN[patch_id, :, :,:] = data_augmentation(train_PAN_data[image_id][4*x_ms : 4*x_ms+PAN_patch_size, 4*y_ms : 4*y_ms+PAN_patch_size,:], rand_mode)
            batch_input_MS[patch_id, :, :, :] = data_augmentation(train_MS_data[image_id][x_ms : x_ms+MS_patch_size, y_ms : y_ms+MS_patch_size, :], rand_mode)
            image_id = (image_id + 1) % len(train_PAN_data)
            if image_id == 0:
                tmp = list(zip(train_PAN_data, train_MS_data))
                random.shuffle(tmp)
                train_PAN_data, train_MS_data  = zip(*tmp)
        _, loss_sharpening,summary_str = sess.run([train_op_Sharpening, Sharpening_loss_total,summary_op], feed_dict={input_MS_Scale_2: batch_input_MS,input_PAN_Scale_1: batch_input_PAN,lr: learning_rate})
        if counter % 10 == 0:
            print("Epoch: [%2d], step: [%2d], loss_sharpening: [%.8f]" \
              % ((epoch+1), iter_num, loss_sharpening))

        train_writer.add_summary(summary_str,iter_num)
        iter_num += 1        
    global_step = epoch+1
    if (epoch+1)%10==0:
      saver_Sharpening.save(sess, Sharpening_checkpoint_dir + 'model.ckpt', global_step=global_step)

print("[*] Finish training for phase %s." % train_phase)
