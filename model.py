import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python.layers import initializers
from utils import *

def lrelu(x, trainbable=None):
    return tf.maximum(x*0.2,x)
    

def Spectral_De_Net(input, training = True):  
    with tf.variable_scope('Spectral_De_Net', reuse=tf.AUTO_REUSE):
        #print(input.shape)
        input_lp = blur(input, kernlen = 41, nsig = 3, channel=4)
        input_hp = input - input_lp

        ### low_pass domain:  the intensity distribition transfer
        SDN_lp_conv1 = slim.conv2d(input_lp, 16, [3,3],  rate=1, stride=1, padding='SAME', activation_fn=lrelu, scope='SDN_lp_conv1') 
                  
        SDN_lp_conv2 = slim.conv2d(SDN_lp_conv1, 16, [3,3],  rate=1, stride=1, padding='SAME', activation_fn=lrelu, scope='SDN_lp_conv2')                
        SDN_lp_conv3 = slim.conv2d(SDN_lp_conv2, 16, [3,3],  rate=1, stride=1, padding='SAME', activation_fn=lrelu, scope='SDN_lp_conv3')        
        SDN_lp_cat_1=tf.concat([SDN_lp_conv1,SDN_lp_conv3],axis=-1)
        SDN_lp_conv4 = slim.conv2d(SDN_lp_cat_1, 16, [3,3],  rate=1, stride=1, padding='SAME', activation_fn=lrelu, scope='SDN_lp_conv4')
        SDN_lp_conv5 = slim.conv2d(SDN_lp_conv4, 16, [3,3],  rate=1, stride=1, padding='SAME', activation_fn=lrelu, scope='SDN_lp_conv5')                        
        SDN_lp_cat_2=tf.concat([SDN_lp_cat_1,SDN_lp_conv5],axis=-1)        
        SDN_lp_conv6 = slim.conv2d(SDN_lp_cat_2, 16, [3,3],  rate=1, stride=1, padding='SAME', activation_fn=lrelu, scope='SDN_lp_conv6')        
        SDN_lp_conv7 = slim.conv2d(SDN_lp_conv6, 8, [3,3],  rate=1, stride=1, padding='SAME', activation_fn=lrelu, scope='SDN_lp_conv7')
        SDN_lp_conv8 = slim.conv2d(SDN_lp_conv7, 1, [3,3],  rate=1, stride=1, padding='SAME', activation_fn=tf.nn.tanh, scope='SDN_lp_conv8')
        
        ### high_pass domain:  the texture distribition transfer
        SDN_hp_conv1 = slim.conv2d(input_hp, 16, [3,3],  rate=1, stride=1, padding='SAME', activation_fn=lrelu, scope='SDN_hp_conv1')
        
        SDN_hp_conv2 = slim.conv2d(SDN_hp_conv1, 16, [3,3],  rate=1, stride=1, padding='SAME', activation_fn=lrelu, scope='SDN_hp_conv2')
        SDN_hp_conv3 = slim.conv2d(SDN_hp_conv2, 16, [3,3],  rate=1, stride=1, padding='SAME', activation_fn=None, scope='SDN_hp_conv3')
        SDN_hp_conv3_out = lrelu(SDN_hp_conv1+SDN_hp_conv2)        
        SDN_hp_conv4 = slim.conv2d(SDN_hp_conv3_out, 16, [3,3],  rate=1, stride=1, padding='SAME', activation_fn=lrelu, scope='SDN_hp_conv4')       
        SDN_hp_conv5 = slim.conv2d(SDN_hp_conv4, 16, [3,3],  rate=1, stride=1, padding='SAME', activation_fn=None, scope='SDN_hp_conv5')
        SDN_hp_conv5_out = lrelu(SDN_hp_conv3_out+SDN_hp_conv5)         
        SDN_hp_conv6 = slim.conv2d(SDN_hp_conv5_out, 16, [3,3],  rate=1, stride=1, padding='SAME', activation_fn=lrelu, scope='SDN_hp_conv6')            
        SDN_hp_conv7 = slim.conv2d(SDN_hp_conv6, 8, [3,3],  rate=1, stride=1, padding='SAME', activation_fn=lrelu, scope='SDN_hp_conv7')
        SDN_hp_conv8 = slim.conv2d(SDN_hp_conv7, 1, [3,3],  rate=1, stride=1, padding='SAME', activation_fn=tf.nn.sigmoid, scope='SDN_hp_conv8')        
        Trans_PAN_lp = SDN_lp_conv8
        Trans_PAN_hp = SDN_hp_conv8
        Trans_PAN =  Trans_PAN_lp + Trans_PAN_hp
        #print(Trans_PAN.shape)                                                                    
        return Trans_PAN



def Spectral_Int_Net(input, training = True):  
    with tf.variable_scope('Spectral_Int_Net', reuse=tf.AUTO_REUSE):
        #print(input.shape)
        input_lp = blur(input, kernlen = 41, nsig = 3, channel=1)
        input_hp = input - input_lp

        ### low_pass domain:  the intensity distribition transfer
        SIN_lp_conv1 = slim.conv2d(input_lp, 16, [3,3],  rate=1, stride=1, padding='SAME', activation_fn=lrelu, scope='SIN_lp_conv1')
        
        SIN_lp_conv2 = slim.conv2d(SIN_lp_conv1, 16, [3,3],  rate=1, stride=1, padding='SAME', activation_fn=lrelu, scope='SIN_lp_conv2')
        SIN_lp_conv3 = slim.conv2d(SIN_lp_conv2, 16, [3,3],  rate=1, stride=1, padding='SAME', activation_fn=lrelu, scope='SIN_lp_conv3')                
        SIN_lp_cat_1=tf.concat([SIN_lp_conv1,SIN_lp_conv3],axis=-1)
        
        SIN_lp_conv4 = slim.conv2d(SIN_lp_cat_1, 16, [3,3],  rate=1, stride=1, padding='SAME', activation_fn=lrelu, scope='SIN_lp_conv4')                
        SIN_lp_conv5 = slim.conv2d(SIN_lp_conv4, 16, [3,3],  rate=1, stride=1, padding='SAME', activation_fn=lrelu, scope='SIN_lp_conv5')
        SIN_lp_cat_2=tf.concat([SIN_lp_cat_1,SIN_lp_conv5],axis=-1)
        
        SIN_lp_conv6 = slim.conv2d(SIN_lp_cat_2, 16, [3,3],  rate=1, stride=1, padding='SAME', activation_fn=lrelu, scope='SIN_lp_conv6')        
        SIN_lp_conv7 = slim.conv2d(SIN_lp_conv6, 8, [3,3],  rate=1, stride=1, padding='SAME', activation_fn=lrelu, scope='SIN_lp_conv7')
        SIN_lp_conv8 = slim.conv2d(SIN_lp_conv7, 4, [3,3],  rate=1, stride=1, padding='SAME', activation_fn=tf.nn.tanh, scope='SIN_lp_conv8')
        
        ### high_pass domain:  the texture distribition transfer
        SIN_hp_conv1 = slim.conv2d(input_hp, 16, [3,3],  rate=1, stride=1, padding='SAME', activation_fn=lrelu, scope='SIN_hp_conv1')
        
        SIN_hp_conv2 = slim.conv2d(SIN_hp_conv1, 16, [3,3],  rate=1, stride=1, padding='SAME', activation_fn=lrelu, scope='SIN_hp_conv2')
        SIN_hp_conv3 = slim.conv2d(SIN_hp_conv2, 16, [3,3],  rate=1, stride=1, padding='SAME', activation_fn=None, scope='SIN_hp_conv3')
        SIN_hp_conv3_out = lrelu(SIN_hp_conv1+SIN_hp_conv3)
        
        SIN_hp_conv4 = slim.conv2d(SIN_hp_conv3_out, 16, [3,3],  rate=1, stride=1, padding='SAME', activation_fn=lrelu, scope='SIN_hp_conv4')
        SIN_hp_conv5 = slim.conv2d(SIN_hp_conv4, 16, [3,3],  rate=1, stride=1, padding='SAME', activation_fn=None, scope='SIN_hp_conv5')
        SIN_hp_conv5_out = lrelu(SIN_hp_conv3_out+SIN_hp_conv5)        
        
        SIN_hp_conv6 = slim.conv2d(SIN_hp_conv5_out, 16, [3,3],  rate=1, stride=1, padding='SAME', activation_fn=lrelu, scope='SIN_hp_conv6')             
        SIN_hp_conv7 = slim.conv2d(SIN_hp_conv6, 8, [3,3],  rate=1, stride=1, padding='SAME', activation_fn=lrelu, scope='SIN_hp_conv7')
        SIN_hp_conv8 = slim.conv2d(SIN_hp_conv7, 4, [3,3],  rate=1, stride=1, padding='SAME', activation_fn=tf.nn.sigmoid, scope='SIN_hp_conv8')
        
        Trans_MS_lp = SIN_lp_conv8
        Trans_MS_hp = SIN_hp_conv8
        Trans_MS =  Trans_MS_lp + Trans_MS_hp                                                                    
        return  Trans_MS



def Discriminator_SDN(input_PAN,training = True):
    with tf.variable_scope('Discriminator_SDN', reuse=tf.AUTO_REUSE):
        DSDN_conv1=slim.conv2d(input_PAN, 16,[1,1], rate=1, stride=2,padding='SAME',activation_fn=lrelu,scope='DSDN_conv1')
        DSDN_conv2=slim.conv2d(DSDN_conv1,64,[1,1], rate=1, stride=2,padding='SAME',activation_fn=lrelu,scope='DSDN_conv2')
        DSDN_conv3=slim.conv2d(DSDN_conv2,256,[1,1], rate=1, stride=2,padding='SAME',activation_fn=lrelu,scope='DSDN_conv3')
        DSDN_conv4=slim.conv2d(DSDN_conv3,64,[1,1], rate=1, stride=2,padding='SAME',activation_fn=lrelu,scope='DSDN_conv4')
        DSDN_conv5=slim.conv2d(DSDN_conv4,16,[1,1], rate=1, stride=2,padding='SAME',activation_fn=lrelu,scope='DSDN_conv5')
        DSDN_conv6=slim.conv2d(DSDN_conv5,1,[1,1], rate=1, stride=2,padding='SAME',activation_fn=None,scope='DSDN_conv6')
        DSDN_prob = tf.reduce_mean(DSDN_conv6, axis=[1,2])     
        return DSDN_prob

def Discriminator_SIN(input_MS,training = True):
    with tf.variable_scope('Discriminator_SIN', reuse=tf.AUTO_REUSE):
        DSIN_conv1=slim.conv2d(input_MS, 16,[1,1], rate=1, stride=2,padding='SAME',activation_fn=lrelu,scope='DSIN_conv1')
        DSIN_conv2=slim.conv2d(DSIN_conv1,64,[1,1], rate=1, stride=2,padding='SAME',activation_fn=lrelu,scope='DSIN_conv2')
        DSIN_conv3=slim.conv2d(DSIN_conv2,256,[1,1], rate=1, stride=2,padding='SAME',activation_fn=lrelu,scope='DSIN_conv3')
        DSIN_conv4=slim.conv2d(DSIN_conv3,64,[1,1], rate=1, stride=2,padding='SAME',activation_fn=lrelu,scope='DSIN_conv4')
        DSIN_conv5=slim.conv2d(DSIN_conv4,16,[1,1], rate=1, stride=2,padding='SAME',activation_fn=lrelu,scope='DSIN_conv5')
        DSIN_conv6=slim.conv2d(DSIN_conv5,1,[1,1], rate=1, stride=2,padding='SAME',activation_fn=None,scope='DSIN_conv6')
        DSIN_prob = tf.reduce_mean(DSIN_conv6, axis=[1,2])     
        return DSIN_prob

        
def Sharpening_Net(input_MS,input_PAN,input_MS_size_H,input_MS_size_W, training = True):  
    with tf.variable_scope('Sharpening_Net', reuse=tf.AUTO_REUSE):
    ## MS_processing: UP-Sample the MS 
        UP_MS_x2=tf.image.resize_images(images=input_MS, size=[2*input_MS_size_H, 2*input_MS_size_W],method=tf.image.ResizeMethod.BICUBIC,align_corners=True)
        Up_MS_conv1=slim.conv2d(UP_MS_x2, 16, [3,3],  rate=1, stride=1, padding='SAME', activation_fn=lrelu, scope='MS_up_conv1')
        UP_MS_x4=tf.image.resize_images(images=Up_MS_conv1, size=[4*input_MS_size_H, 4*input_MS_size_W],method=tf.image.ResizeMethod.BICUBIC,align_corners=True)
        Up_MS_conv2=slim.conv2d(UP_MS_x4, 16, [3,3],  rate=1, stride=1, padding='SAME', activation_fn=lrelu, scope='MS_up_conv2')        

    ## PAN_processing: Increase the channel of PAN 
        Up_PAN_conv1=slim.conv2d(input_PAN, 16, [3,3],  rate=1, stride=1, padding='SAME', activation_fn=lrelu, scope='PAN_up_conv1')
                
    ## MS feature extract phase I: dense+residual
        MS_1_conv1=slim.conv2d(Up_MS_conv2, 16, [3,3],  rate=1, stride=1, padding='SAME', activation_fn=lrelu, scope='MS_1_conv1')
        MS_1_conv2=slim.conv2d(MS_1_conv1, 16, [3,3],  rate=1, stride=1, padding='SAME', activation_fn=lrelu, scope='MS_1_conv2')
        MS_1_cat_12=tf.concat([MS_1_conv1,MS_1_conv2],axis=-1)
        MS_1_conv3=slim.conv2d(MS_1_cat_12, 16, [3,3],  rate=1, stride=1, padding='SAME', activation_fn=lrelu, scope='MS_1_conv3')
        MS_1_cat_123=tf.concat([MS_1_conv1,MS_1_conv2,MS_1_conv3],axis=-1)
        MS_1_conv4=slim.conv2d(MS_1_cat_123, 16, [3,3],  rate=1, stride=1, padding='SAME', activation_fn=None, scope='MS_1_conv4')
        MS_1_output= lrelu(Up_MS_conv2+MS_1_conv4)

    ## PAN feature extract phase I: dense+residual
        PAN_1_conv1=slim.conv2d(Up_PAN_conv1, 16, [3,3],  rate=1, stride=1, padding='SAME', activation_fn=lrelu, scope='PAN_1_conv1')
        PAN_1_conv2=slim.conv2d(PAN_1_conv1, 16, [3,3],  rate=1, stride=1, padding='SAME', activation_fn=lrelu, scope='PAN_1_conv2')
        PAN_1_cat_12=tf.concat([PAN_1_conv1,PAN_1_conv2],axis=-1)
        PAN_1_conv3=slim.conv2d(PAN_1_cat_12, 16, [3,3],  rate=1, stride=1, padding='SAME', activation_fn=lrelu, scope='PAN_1_conv3')
        PAN_1_cat_123=tf.concat([PAN_1_conv1,PAN_1_conv2,PAN_1_conv3],axis=-1)
        PAN_1_conv4=slim.conv2d(PAN_1_cat_123, 16, [3,3],  rate=1, stride=1, padding='SAME', activation_fn=None, scope='PAN_1_conv4')
        PAN_1_output= lrelu(Up_PAN_conv1+PAN_1_conv4)
        
    ## Unique information extract from phase I of PAN: subtract+multi-scale
        Unique_PAN_1 = PAN_1_output - MS_1_output
        Unique_PAN_1_scale_1x1 = slim.conv2d(Unique_PAN_1, 16, [1,1],  rate=1, stride=1, padding='SAME', activation_fn=lrelu, scope='Unique_scale_1x1')
        Unique_PAN_1_scale_3x3 = slim.conv2d(Unique_PAN_1, 16, [3,3],  rate=1, stride=1, padding='SAME', activation_fn=lrelu, scope='Unique_scale_3x3')
        Unique_PAN_1_scale_5x5 = slim.conv2d(Unique_PAN_1, 16, [5,5],  rate=1, stride=1, padding='SAME', activation_fn=lrelu, scope='Unique_scale_5x5')

    ## Multi-scale unique information injection
        MS_2_injection_cat=tf.concat([MS_1_output,Unique_PAN_1_scale_1x1,Unique_PAN_1_scale_3x3,Unique_PAN_1_scale_5x5],axis=-1)
        MS_2_injection_output=slim.conv2d(MS_2_injection_cat, 16, [3,3],  rate=1, stride=1, padding='SAME', activation_fn=lrelu, scope='MS_2_injection')
        
    ## MS reconstruction phase: dense+residual          
        MS_2_conv1=slim.conv2d(MS_2_injection_output, 16, [3,3],  rate=1, stride=1, padding='SAME', activation_fn=lrelu, scope='MS_2_conv1')
        MS_2_conv2=slim.conv2d(MS_2_conv1, 8, [3,3],  rate=1, stride=1, padding='SAME', activation_fn=lrelu, scope='MS_2_conv2')
        MS_2_conv3=slim.conv2d(MS_2_conv2, 4, [3,3],  rate=1, stride=1, padding='SAME', activation_fn=None, scope='MS_2_conv3')
        MS_2_residual_output= tf.nn.tanh(MS_2_conv3)  
        
        Sharpened_MS = MS_2_residual_output + tf.image.resize_images(images=input_MS, size=[4*input_MS_size_H, 4*input_MS_size_W],method=tf.image.ResizeMethod.BICUBIC,align_corners=True)
                                                              
        return Sharpened_MS           
        
        