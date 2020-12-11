#试验二所用代码应该在这里，根据最小帧长和最小分段，将每一类进行分段，然后将每一类分成三段作为train，val,test，这样保证了train，val,test里面所有类的值都是平均的
import os
import pickle
import json
import math
import numpy as np
import librosa
import librosa.core
import librosa.display

import random

Datadir = '../../data/'
out_path = '../../input/M2D_segment_single_data'
fps = 25
num_joint = 23
max_body = 1
min_duration = 229
min_num = 34
from m2d_pre import pre_normalization
from sklearn import preprocessing
from m2d_gen import motion_feature_extract
from sklearn.utils import shuffle
#读取这个骨架的json文件有两个问题是：是否需要旋转，去中心化；每一个json文件都有不同的帧数长度：帧数 * 23 * 3；比2sgcn网络少了人物的维度，而且帧数是不确定的。
#接受数组，产生文件
def genda(data_dirs):
    
    train_dir = []
    test_dir = []
    val_dir = []
    
    seg = min_num // 3#整除3，将其分为三段，train，val,test
    
    train = np.zeros((seg, 3, min_duration, num_joint, max_body), dtype=np.float32)#(3, 3, 1494, 23, 1)
    val = np.zeros((seg, 3, min_duration, num_joint, max_body), dtype=np.float32)
    test = np.zeros((min_num - 2 * seg, 3, min_duration, num_joint, max_body), dtype=np.float32)
    
    temp_num = 0
    train_i = 0
    val_i = 0
    test_i = 0
    
    for i, one in enumerate(data_dirs):
        
        data_path = os.path.join(Datadir,one)
        
        motion_features_pre = motion_feature_extract(data_path, with_centering=True, with_rotate=False)
        motion_features = motion_features_pre.reshape(motion_features_pre.shape[0],-1)
        min_max_scaler = preprocessing.MinMaxScaler().fit_transform(motion_features)
        motion_features = min_max_scaler.reshape(min_max_scaler.shape[0],23,3)
        
        t = motion_features.shape[0] // min_duration
        
        for j in range(t):
            
            temp_num += 1
            
            if temp_num <= seg:
                #train
                temp_features = motion_features[min_duration * j:min_duration * (j + 1),:,:]
                temp_features = np.expand_dims(temp_features,3)
                temp_features = np.transpose(temp_features,[2,0,1,3])

                train[train_i,:,:,:,:] = temp_features
                train_dir.append(one)
                train_i += 1
            elif temp_num > seg and temp_num<= seg * 2:
                #val
                temp_features = motion_features[min_duration * j:min_duration * (j + 1),:,:]
                temp_features = np.expand_dims(temp_features,3)
                temp_features = np.transpose(temp_features,[2,0,1,3])

                val[val_i,:,:,:,:] = temp_features
                val_dir.append(one)
                val_i += 1
            elif temp_num > seg * 2 and temp_num <= seg*3 + 1:
               #test
                temp_features = motion_features[min_duration * j:min_duration * (j + 1),:,:]
                temp_features = np.expand_dims(temp_features,3)
                temp_features = np.transpose(temp_features,[2,0,1,3])

                test[test_i,:,:,:,:] = temp_features
                test_dir.append(one)
                test_i += 1
            else:
                break

    train = pre_normalization(train)#(61, 3, 900, 23, 1)
    val = pre_normalization(val)
    test = pre_normalization(test)
    
    return train,train_dir,val,val_dir,test,test_dir

if __name__ == '__main__':
    All_dirs = os.listdir(Datadir)
    All_dirs.sort()
    C_dirs = []
    R_dirs = []
    T_dirs = []
    W_dirs = []
    for one in All_dirs:
        if one.split('_')[1] == 'C':
            C_dirs.append(one)
        elif one.split('_')[1] == 'R':
            R_dirs.append(one)
        elif one.split('_')[1] == 'T':
            T_dirs.append(one)
        else:
            W_dirs.append(one)
    C_train,C_train_dir,C_val,C_val_dir,C_test,C_test_dir = genda(C_dirs)
    R_train,R_train_dir,R_val,R_val_dir,R_test,R_test_dir = genda(R_dirs)
    T_train,T_train_dir,T_val,T_val_dir,T_test,T_test_dir = genda(T_dirs)
    W_train, W_train_dir, W_val, W_val_dir, W_test, W_test_dir = genda(W_dirs)

    train = np.vstack((C_train,R_train,T_train,W_train))#(3, 3, 1494, 23, 1)
    val = np.vstack((C_val,R_val,T_val,W_val))
    test = np.vstack((C_test,R_test,T_test,W_test))
    train_dir = C_train_dir + R_train_dir + T_train_dir + W_train_dir
    test_dir = C_test_dir + R_test_dir + T_test_dir + W_test_dir
    val_dir = C_val_dir + R_val_dir + T_val_dir + W_val_dir

    seg = min_num // 3#整除为多少段

    train_label = [0] * seg + [1] * seg + [2] * seg + [3] * seg
    val_label = [0] * seg + [1] * seg + [2] * seg + [3] * seg
    test_label = [0] * (seg + 1) + [1] * (seg + 1) + [2] * (seg + 1) + [3] * (seg + 1)#这里+1是因为段数是34，导致不能平分，所以剩余

    train,train_dir,train_label = shuffle(train,train_dir,train_label,random_state = 0)
    val,val_dir,val_label = shuffle(val,val_dir,val_label,random_state = 0)
    test,test_dir,test_label = shuffle(test,test_dir,test_label,random_state = 0)
    np.save('{}/{}_joint.npy'.format(out_path, 'train'), train)
    np.save('{}/{}_joint.npy'.format(out_path,'val'), val)
    np.save('{}/{}_joint.npy'.format(out_path,'test'), test)

    with open('{}/{}_label.pkl'.format(out_path,'train' ), 'wb') as f:
        pickle.dump((train_dir, list(train_label)), f)#保存文件名称和对应的缩写（作为标签）形成pkl文件。
    with open('{}/{}_label.pkl'.format(out_path,'val'), 'wb') as f:
        pickle.dump((val_dir, list(val_label)), f)#保存文件名称和对应的缩写（作为标签）形成pkl文件。
    with open('{}/{}_label.pkl'.format(out_path,'test'), 'wb') as f:
        pickle.dump((test_dir, list(test_label)), f)#保存文件名称和对应的缩写（作为标签）形成pkl文件。