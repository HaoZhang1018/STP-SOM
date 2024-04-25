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

input_MS_Scale_2 = tf.placeholder(tf.float32, [None, None, None, 4], name='input_MS')
input_PAN_Scale_1 = tf.placeholder(tf.float32, [None, None, None, 1], name='input_PAN') 

[Input_MS_Scale_3, Input_PAN_Scale_2] = downgrade_images(input_MS_Scale_2, input_PAN_Scale_1, 4, sensor=Data_Sensor,flag_PAN_MTF=1)


#MS->PAN->MS
Input_MS_2_PAN = Spectral_De_Net(input_MS_Scale_2, training = True)
Input_MS_2_PAN_2_MS = Spectral_Int_Net(Input_MS_2_PAN, training = True)

#PAN->MS->PAN
Input_PAN_2_MS = Spectral_Int_Net(Input_PAN_Scale_2, training = True)
Input_PAN_2_MS_2_PAN = Spectral_De_Net(Input_PAN_2_MS, training = True)

#Discriminator  MS2PAN


Dis_SDN_prob_True = Discriminator_SDN(Input_PAN_Scale_2,training = True)
Dis_SDN_prob_Fake = Discriminator_SDN(Input_MS_2_PAN,training = True)

#Discriminator  PAN2MS
Dis_SIN_prob_True = Discriminator_SIN(input_MS_Scale_2,training = True)
Dis_SIN_prob_Fake = Discriminator_SIN(Input_PAN_2_MS,training = True)


###LOSS FUNCTION
## SDN discriminator loss
Dis_loss_true_SDN=tf.reduce_mean(tf.abs(Dis_SDN_prob_True-tf.random_uniform(shape=[batch_size,1],minval=0.8,maxval=1.0,dtype=tf.float32)))
Dis_loss_fake_SDN=tf.reduce_mean(tf.abs(Dis_SDN_prob_Fake-tf.random_uniform(shape=[batch_size,1],minval=0.0,maxval=0.2,dtype=tf.float32)))
Dis_loss_SDN_total=Dis_loss_true_SDN+Dis_loss_fake_SDN
tf.summary.scalar('Dis_loss_SDN_total',Dis_loss_SDN_total)

## SIN discriminator loss
Dis_loss_true_SIN=tf.reduce_mean(tf.abs(Dis_SIN_prob_True-tf.random_uniform(shape=[batch_size,1],minval=0.8,maxval=1.0,dtype=tf.float32)))
Dis_loss_fake_SIN=tf.reduce_mean(tf.abs(Dis_SIN_prob_Fake-tf.random_uniform(shape=[batch_size,1],minval=0.0,maxval=0.2,dtype=tf.float32)))
Dis_loss_SIN_total=Dis_loss_true_SIN+Dis_loss_fake_SIN
tf.summary.scalar('Dis_loss_SIN_total',Dis_loss_SIN_total)


SDN_Trans_loss=tf.reduce_mean(tf.abs(Input_MS_2_PAN-Input_PAN_Scale_2)) 
SDN_Recon_loss=tf.reduce_mean(tf.abs(Input_PAN_2_MS_2_PAN-Input_PAN_Scale_2)) 
SDN_G_loss_dis=tf.reduce_mean(tf.abs(Dis_SDN_prob_Fake-tf.random_uniform(shape=[batch_size,1],minval=0.8,maxval=1.0,dtype=tf.float32))) 
SDN_loss_total = 1*SDN_Trans_loss + 0.2*SDN_Recon_loss + 0.1*SDN_G_loss_dis
tf.summary.scalar('SDN_Trans_loss',SDN_Trans_loss)
tf.summary.scalar('SDN_Recon_loss',SDN_Recon_loss)
tf.summary.scalar('SDN_G_loss_dis',SDN_G_loss_dis)
tf.summary.scalar('SDN_loss_total',SDN_loss_total)


SIN_Trans_loss=tf.reduce_mean(tf.abs(Input_PAN_2_MS-input_MS_Scale_2))
SIN_Recon_loss=tf.reduce_mean(tf.abs(Input_MS_2_PAN_2_MS-input_MS_Scale_2)) 
SIN_G_loss_dis=tf.reduce_mean(tf.abs(Dis_SIN_prob_Fake-tf.random_uniform(shape=[batch_size,1],minval=0.8,maxval=1.0,dtype=tf.float32))) 
SIN_loss_total = 1*SIN_Trans_loss + 0.2*SIN_Recon_loss + 0.1*SIN_G_loss_dis
tf.summary.scalar('SIN_Trans_loss',SIN_Trans_loss)
tf.summary.scalar('SIN_Recon_loss',SIN_Recon_loss)
tf.summary.scalar('SIN_G_loss_dis',SIN_G_loss_dis)
tf.summary.scalar('SIN_loss_total',SIN_loss_total)
 
lr = tf.placeholder(tf.float32, name='learning_rate')

optimizer = tf.train.AdamOptimizer(learning_rate=lr, name='AdamOptimizer')
var_SDN = [var for var in tf.trainable_variables() if 'Spectral_De_Net' in var.name]
var_SIN = [var for var in tf.trainable_variables() if 'Spectral_Int_Net' in var.name]
var_DSDN = [var for var in tf.trainable_variables() if 'Discriminator_SDN' in var.name]
var_DSIN = [var for var in tf.trainable_variables() if 'Discriminator_SIN' in var.name]
g_list = tf.global_variables()

with tf.name_scope('train_step'):
  train_op_SDN  = optimizer.minimize(SDN_loss_total, var_list = var_SDN)
  train_op_SIN  = optimizer.minimize(SIN_loss_total, var_list = var_SIN)
  train_op_DSDN = optimizer.minimize(Dis_loss_SDN_total, var_list = var_DSDN)
  train_op_DSIN = optimizer.minimize(Dis_loss_SIN_total, var_list = var_DSIN)    


saver_SDN = tf.train.Saver(var_list=var_SDN,max_to_keep=3000)
saver_SIN = tf.train.Saver(var_list=var_SIN,max_to_keep=3000)
saver_DSDN = tf.train.Saver(var_list=var_DSDN,max_to_keep=3000)
saver_DSIN = tf.train.Saver(var_list=var_DSIN,max_to_keep=3000)
sess.run(tf.global_variables_initializer())

print("[*] Initialize model successfully...")

with tf.name_scope('image'):
  tf.summary.image('input_MS_Scale_2',tf.expand_dims(input_MS_Scale_2[1,:,:,:],0))
  tf.summary.image('Input_PAN_Scale_2',tf.expand_dims(Input_PAN_Scale_2[1,:,:,:],0))
  tf.summary.image('Input_MS_2_PAN',tf.expand_dims(Input_MS_2_PAN[1,:,:,:],0))
  tf.summary.image('Input_PAN_2_MS_2_PAN',tf.expand_dims(Input_PAN_2_MS_2_PAN[1,:,:,:],0))
  tf.summary.image('Input_PAN_2_MS',tf.expand_dims(Input_PAN_2_MS[1,:,:,:],0)) 
  tf.summary.image('Input_MS_2_PAN_2_MS',tf.expand_dims(Input_MS_2_PAN_2_MS[1,:,:,:],0)) 
     
  summary_op = tf.summary.merge_all()

train_writer = tf.summary.FileWriter('./log' + '/Cycle_Trans_train',sess.graph,flush_secs=60)


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
    PAN_im = load_images_no_norm(train_PAN_data_names[idx])
    PAN_im = np.expand_dims(PAN_im,2)    
    train_MS_data.append(MS_im)    
    train_PAN_data.append(PAN_im)



epoch = 3000
learning_rate = 0.0001
train_phase = 'Degradation'
numBatch = len(train_PAN_data) // int(batch_size)


SDN_checkpoint_dir = './checkpoint/QuickBird/SDN/'
if not os.path.isdir(SDN_checkpoint_dir):
    os.makedirs(SDN_checkpoint_dir)
ckpt_SDN=tf.train.get_checkpoint_state(SDN_checkpoint_dir)
if ckpt_SDN:
    print('loaded '+ckpt_SDN.model_checkpoint_path)
    saver_SDN.restore(sess,ckpt_SDN.model_checkpoint_path)
else:
    print('No SDN pretrained model!')


SIN_checkpoint_dir = './checkpoint/QuickBird/SIN/'
if not os.path.isdir(SIN_checkpoint_dir):
    os.makedirs(SIN_checkpoint_dir)
ckpt_SIN=tf.train.get_checkpoint_state(SIN_checkpoint_dir)
if ckpt_SIN:
    print('loaded '+ckpt_SIN.model_checkpoint_path)
    saver_SIN.restore(sess,ckpt_SIN.model_checkpoint_path)
else:
    print('No SIN pretrained model!')

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
        batch_input_PAN_dis = np.zeros((batch_size, MS_patch_size, MS_patch_size, 1), dtype="float32")
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
        for i in range(1):
            _, loss_dis_SDN = sess.run([train_op_DSDN, Dis_loss_SDN_total], feed_dict={input_MS_Scale_2: batch_input_MS,input_PAN_Scale_1: batch_input_PAN, lr: learning_rate})
        _, loss_SDN = sess.run([train_op_SDN, SDN_loss_total], feed_dict={input_MS_Scale_2: batch_input_MS,input_PAN_Scale_1: batch_input_PAN, lr: learning_rate})
        for i in range(1):
            _, loss_dis_SIN = sess.run([train_op_DSIN, Dis_loss_SIN_total], feed_dict={input_MS_Scale_2: batch_input_MS,input_PAN_Scale_1: batch_input_PAN,lr: learning_rate})                    
        _, loss_SIN,summary_str = sess.run([train_op_SIN, SIN_loss_total,summary_op], feed_dict={input_MS_Scale_2: batch_input_MS,input_PAN_Scale_1: batch_input_PAN,lr: learning_rate})          
        if counter % 10 == 0:
            print("Epoch: [%2d], step: [%2d], loss_SDN: [%.8f], loss_SDN_dis: [%.8f], loss_SIN:[%.8f], loss_SIN_dis: [%.8f]" \
              % ((epoch+1), iter_num, loss_SDN,loss_dis_SDN,loss_SIN,loss_dis_SIN))
        train_writer.add_summary(summary_str,iter_num)
        iter_num += 1        
    global_step = epoch+1
    if (epoch+1)%10==0:
      saver_SDN.save(sess, SDN_checkpoint_dir + 'model.ckpt', global_step=global_step)
      saver_SIN.save(sess, SIN_checkpoint_dir + 'model.ckpt', global_step=global_step)

print("[*] Finish training for phase %s." % train_phase)
