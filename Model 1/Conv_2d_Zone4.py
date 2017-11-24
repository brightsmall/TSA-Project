# -*- coding: utf-8 -*-
"""
Created on Sat Jul  8 10:20:40 2017

@author: Administrator
"""

### Adding Batch Norm with Multi GPU Support

### This model is awesome ; trained to 100% accuracy in 2100 steps



# "c:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe"  --loop=1 --query-gpu=memory.used --format=csv memory.used [MiB]

import tensorflow as tf
import numpy as np
import pandas as pd
import os


config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)
#config.gpu_options.allow_growth = True
sess = tf.InteractiveSession(config=config)


######################

# load complete pos list and neg list 

pos_folder =  "D:\\TSA\\stage 1\\aps\\Zone 4\\pos\\"
neg_folder =  "D:\\TSA\\stage 1\\aps\\Zone 4\\neg\\"
test_folder = "D:\\TSA\\stage 1\\aps\\Zone 4\\test\\"


pos_ID_list=[]

for file in os.listdir(pos_folder):
    if file.endswith(".npy"):
        file_id = str(file).split('.')[0]
        pos_ID_list.append(file_id)
        

neg_ID_list=[]

for file in os.listdir(neg_folder):
    if file.endswith(".npy"):
        file_id = str(file).split('.')[0]
        neg_ID_list.append(file_id)
        

test_ID_list=[]

for file in os.listdir(test_folder):
    if file.endswith(".npy"):
        file_id = str(file).split('.')[0]
        test_ID_list.append(file_id)
        

####################################################

# load random batch of training images and corresponding labels
    
batch_size = 16

def get_batch(batch_size):
    array_list=[]
    label_list=[]

    pos_indices1 = np.random.choice(len(pos_ID_list), int(batch_size/4))
    neg_indices1 = np.random.choice(len(neg_ID_list), int(batch_size/4))
    pos_indices2 = np.random.choice(len(pos_ID_list), int(batch_size/4))
    neg_indices2 = np.random.choice(len(neg_ID_list), int(batch_size/4))

#load pos image 

    for idx in pos_indices1:
        img_id = pos_ID_list[idx]
        img_npy_file = pos_folder+img_id + ".npy"
        image = np.load(img_npy_file)
        array_list.append(image)
        label_list.append([0,1])
    
    for idx in neg_indices1:
        img_id = neg_ID_list[idx]
        img_npy_file = neg_folder+img_id + ".npy"
        image = np.load(img_npy_file)
        array_list.append(image)
        label_list.append([1,0])
        
    for idx in pos_indices2:
        img_id = pos_ID_list[idx]
        img_npy_file = pos_folder+img_id + ".npy"
        image = np.load(img_npy_file)
        array_list.append(image)
        label_list.append([0,1])
    
    for idx in neg_indices2:
        img_id = neg_ID_list[idx]
        img_npy_file = neg_folder+img_id + ".npy"
        image = np.load(img_npy_file)
        array_list.append(image)
        label_list.append([1,0])
    
    batch_stack = np.stack(array_list,axis=0) # x's
    

    labels_array = np.asarray(label_list)     # y's
    return batch_stack,labels_array

def get_test_batch():
    array_list=[]
    test_folder = "D:\\TSA\\stage 1\\aps\\Zone 4\\test\\"
   
    for idx in range(len(test_ID_list)):
        img_id = test_ID_list[idx]
        img_npy_file = test_folder+img_id + ".npy"
        image = np.load(img_npy_file)
        array_list.append(image)
        
    batch_stack = np.stack(array_list,axis=0)    
    return batch_stack

def get_test_batch_right_leg():
    array_list=[]
    test_folder = "D:\\TSA\stage 1\\aps\\Zone 2\\test_mirror\\"
   
    for idx in range(len(test_ID_list)):
        img_id = test_ID_list[idx]
        img_npy_file = test_folder+img_id + "_mr.npy"
        image = np.load(img_npy_file)
        array_list.append(image)
        
    batch_stack = np.stack(array_list,axis=0)    
    return batch_stack




def check_train_accuracy():
    total_accuracy = 0
    for i in range(10):
        x_batch , y_batch = get_batch(10)
        train_accuracy = sess.run(accuracy,{x: x_batch, y: y_batch, keep_rate: keep_rate_full,  phase: test_phase})
        total_accuracy += train_accuracy
    total_accuracy = total_accuracy/10
    #print(total_accuracy)
    return total_accuracy




##################

n_classes = 2


def weight_variable_xavier(name,shape):
    initial=tf.contrib.layers.xavier_initializer()
    return tf.get_variable(name, shape=shape, initializer= initial)

def bias_variable(name,shape):
    initial = tf.constant(0.0001, shape=shape)
    return tf.get_variable(name, initializer= initial)

def conv2d_stride1(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def conv2d_stride2(x, W):
    return tf.nn.conv2d(x, W, strides=[1,2,2,1], padding='SAME')

def maxpool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

def maxpool_8x5(x):
    return tf.nn.pool(x,window_shape=[8,5],strides=[8,5],pooling_type='MAX',padding='SAME')

def make_parallel(fn, num_gpus, **kwargs):

    in_splits = {}  # create empty dictionary

    # for each of the tensors in kwargs, create a split and add it to the dictionary
    for k, v in kwargs.items():
        in_splits[k] = tf.split(v, num_gpus)  

    loss_split = [] # create empty list
    correct_split = [] 
    pred_split = [] 
    
    for i in range(num_gpus):
        with tf.device(tf.DeviceSpec(device_type="GPU", device_index=i)):
            # allow for variable reuse on GPUs beyond index 0
            with tf.variable_scope(tf.get_variable_scope(), reuse=i > 0):
                #pass the splits into the function and append results
                loss, correct_prediction, pred = fn(**{k : v[i] for k, v in in_splits.items()})                              
                loss_split.append(loss)
                correct_split.append(correct_prediction)
                pred_split.append(pred)

    return tf.concat(loss_split, axis=0), tf.concat(correct_split, axis=0), tf.concat(pred_split, axis=0)

def model(x, y, keep_rate, phase):
    

    
    x_reshape = tf.reshape(x, shape=[-1, 8192, 280, 1])
    

    # first convolution uses stride 2 to reduce output size
    
    W_conv1 = weight_variable_xavier("w_conv1",[7,7,1,32]) 
    b_conv1 = bias_variable("b_conv1",[32])
    
    conv1 = conv2d_stride2(x_reshape, W_conv1)
    #bn_1 = tf.layers.batch_normalization(conv1,center=True,scale=True,training=phase)
    elu1 = tf.nn.elu(conv1 + b_conv1)
     
    
    # second layer
    
    W_conv2 = weight_variable_xavier("w_conv2",[5,5,32,32])  
    b_conv2 = bias_variable("b_conv2",[32])
    
    conv2 = conv2d_stride1(elu1, W_conv2)
    #bn_2 = tf.layers.batch_normalization(conv2,center=True,scale=True,training=phase)
    elu2 = tf.nn.elu(conv2 + b_conv2 ) 
    conv2_pool = maxpool_2x2(elu2)
    
    # third layer
    
    W_conv3 = weight_variable_xavier("w_conv3",[3,3,32,64])
    b_conv3 = bias_variable("b_conv3", [64])
    
    conv3 = conv2d_stride1(conv2_pool, W_conv3) 
    #bn_3 = tf.layers.batch_normalization(conv3,center=True,scale=True,training=phase)
    elu3 = tf.nn.elu(conv3+b_conv3 ) 
    conv3_pool = maxpool_2x2(elu3)
    
    W_conv4 = weight_variable_xavier("w_conv4",[3,3,64,128])
    b_conv4 = bias_variable("b_conv4",[128])
    conv4 = conv2d_stride1(conv3_pool, W_conv4)
    #bn_4 = tf.layers.batch_normalization(conv4,center=True,scale=True,training=phase)
    elu4 = tf.nn.elu(conv4 + b_conv4 ) 
    
    
    #dropout
    
    conv4_do = tf.nn.dropout(elu4, keep_rate)
    
    # this stacks the 65x35 images and then runs 1x1 convolution to extract features across the images
    # output retains all spatial information (65X35 image size) but condenses information from the features
    # the idea is that this layer acts somewhat like a fully connected layer by bringing together information
    # from across the spatial layers 
    
    conv4_restack = tf.concat((tf.split(conv4_do,16,axis=1)),axis=3)
    
    W_conv5 = weight_variable_xavier("w_conv5",[1,1,2048,64])
    b_conv5 = bias_variable("b_conv5",[64])
    
    conv5 = conv2d_stride1(conv4_restack, W_conv5) 
    
    #bn_5 = tf.layers.batch_normalization(conv5,center=True,scale=True,training=phase)
    elu5 = tf.nn.elu(conv5 + b_conv5)
    # flatten, run dropout, and ccompute final output
    
    conv5_flat = tf.reshape(elu5,[-1, 143360])
    conv5_flat_do = tf.nn.dropout(conv5_flat, keep_rate)
    
    W_out = weight_variable_xavier("w_out",[143360, n_classes])
    b_out = bias_variable("b_out",[n_classes])
    
    output = tf.matmul(conv5_flat_do, W_out) + b_out
                  
    pred = tf.nn.softmax(output) 

    correct_prediction = tf.equal( tf.argmax(output,1), tf.argmax(y,1) )                   
                    
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output)
    
    return loss, correct_prediction, pred
################################

x = tf.placeholder(tf.float32, shape=[None, 8192,280])
y = tf.placeholder(tf.float32, shape=[None,2]) 
keep_rate = tf.placeholder(tf.float32)
phase = tf.placeholder(tf.bool)                
                  
# train model
loss,correct_prediction,pred = make_parallel(model, 2, x=x, y=y, keep_rate=keep_rate, phase=phase)

mean_loss = tf.reduce_mean(loss)
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_op = tf.train.AdamOptimizer(1e-4).minimize(mean_loss,colocate_gradients_with_ops=True)


#correct_prediction = tf.equal(tf.argmax(output,1), tf.argmax(y,1))
#accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

sess.run(tf.global_variables_initializer())

keep_rate_half = np.array([0.5,0.5])
keep_rate_full = np.array([1,1])
training_phase = np.array([1,1])
test_phase= np.array([0,0])


for i in range(2501):
    x_batch , y_batch = get_batch(batch_size)
    
    _, loss_val = sess.run([train_op, loss], {x: x_batch, y: y_batch, keep_rate: keep_rate_half, phase: training_phase} )
       
    if i%10 ==0:
        step_loss = sess.run(mean_loss,{x: x_batch, y: y_batch, keep_rate: keep_rate_full,  phase: test_phase})
        print("step %d, loss %g"%(i,step_loss))

    if i%100 == 0:

        train_accuracy = check_train_accuracy()
        print("step %d, train accuracy %g"%(i,train_accuracy))
        

# save the model

saver = tf.train.Saver()

#save_path = saver.save(sess, "D:\\TSA\\stage 1\\aps\\Saved Models\\Zone 4 model.ckpt")

#last ran for 2500 steps

saver.restore(sess,  "D:\\TSA\\stage 1\\aps\\Saved Models\\Zone 4 model.ckpt")

#################

# score test data for left leg

x_batch = get_test_batch()

get_pred_part1 = sess.run(pred,{x: x_batch[0:20], keep_rate: keep_rate_full,  phase: test_phase})
get_pred_part2 = sess.run(pred,{x: x_batch[20:40], keep_rate: keep_rate_full,  phase: test_phase})
get_pred_part3 = sess.run(pred,{x: x_batch[40:60], keep_rate: keep_rate_full,  phase: test_phase})
get_pred_part4 = sess.run(pred,{x: x_batch[60:80], keep_rate: keep_rate_full,  phase: test_phase})
get_pred_part5 = sess.run(pred,{x: x_batch[80:100], keep_rate: keep_rate_full,  phase: test_phase})


get_pred = np.concatenate((get_pred_part1,get_pred_part2,get_pred_part3,get_pred_part4,get_pred_part5), axis=0)

pred_df = pd.DataFrame(get_pred).sort_index()


pred_df['Id']=test_ID_list
       
#pred_df['X_quad_root'] = pred_df[1]**(1/4)

pred_df.to_csv("D:\\TSA\\stage 1\\zone_4_pred.csv", index=False)

#############


# score test data for right leg

x_batch = get_test_batch_right_leg()

get_pred_part1 = sess.run(pred,{x: x_batch[0:20], keep_rate: keep_rate_full,  phase: test_phase})
get_pred_part2 = sess.run(pred,{x: x_batch[20:40], keep_rate: keep_rate_full,  phase: test_phase})
get_pred_part3 = sess.run(pred,{x: x_batch[40:60], keep_rate: keep_rate_full,  phase: test_phase})
get_pred_part4 = sess.run(pred,{x: x_batch[60:80], keep_rate: keep_rate_full,  phase: test_phase})
get_pred_part5 = sess.run(pred,{x: x_batch[80:100], keep_rate: keep_rate_full,  phase: test_phase})

get_pred = np.concatenate((get_pred_part1,get_pred_part2,get_pred_part3,get_pred_part4,get_pred_part5), axis=0)

pred_df = pd.DataFrame(get_pred).sort_index()


pred_df['Id']=test_ID_list
       
#pred_df['X_quad_root'] = pred_df[1]**(1/4)

pred_df.to_csv("D:\\TSA\\stage 1\\zone_2_pred.csv", index=False)