#试验一：数据在进行train，val,test之前打乱，这个文件是将全部分段，然后不是从一个数据中分train，val,test而是从所有数据中进行分割
import os
import pickle
import json
import math
import numpy as np
import random


Datadir = '../../data/'
out_path = '../../input/M2D_segment_frames/'

min_duration = 229
min_num = 34
num_joint = 23
max_body = 1

from m2d_pre import pre_normalization
from sklearn import preprocessing
from m2d_gen import motion_feature_extract
from sklearn.utils import shuffle

# 读取这个骨架的json文件有两个问题是：是否需要旋转，去中心化；每一个json文件都有不同的帧数长度：帧数 * 23 * 3；比2sgcn网络少了人物的维度，而且帧数是不确定的。
# 接受数组，产生文件
def genda(data_dirs):
    data = np.zeros((min_num, 3, min_duration, num_joint, max_body), dtype=np.float32)
    data_dir = []
    data_num = 0
    for i, one in enumerate(data_dirs):
        if data_num >= min_num:
            break

        data_path = os.path.join(Datadir, one)

        motion_features_pre = motion_feature_extract(data_path, with_centering=True, with_rotate=False)
        #正则化，对于每一个数据都正则化
        motion_features = motion_features_pre.reshape(motion_features_pre.shape[0], -1)
        min_max_scaler = preprocessing.MinMaxScaler().fit_transform(motion_features)
        motion_features = min_max_scaler.reshape(min_max_scaler.shape[0], 23, 3)

        t = motion_features.shape[0] // min_duration

        for j in range(t):
            if data_num >= min_num:
                break
            else:
                temp_features = motion_features[min_duration * j:min_duration * (j + 1), :, :]
                temp_features = np.expand_dims(temp_features, 3)
                temp_features = np.transpose(temp_features, [2, 0, 1, 3])

                data[data_num, :, :, :, :] = temp_features
                data_num += 1
                data_dir.append(one)

    data = pre_normalization(data)

    return data,data_dir


if __name__ == '__main__':
    All_dirs = os.listdir(Datadir)
    All_dirs.sort()
    C_dirs = []
    R_dirs = []
    T_dirs = []
    W_dirs = []
    for one in All_dirs:
        if one[0] != 'D':
            continue
        if one.split('_')[1] == 'C':
            C_dirs.append(one)
        elif one.split('_')[1] == 'R':
            R_dirs.append(one)
        elif one.split('_')[1] == 'T':
            T_dirs.append(one)
        else:
            W_dirs.append(one)
    C_data,C_dir = genda(C_dirs)
    R_data,R_dir = genda(R_dirs)
    T_data,T_dir = genda(T_dirs)
    W_data,W_dir = genda(W_dirs)
    data_dir = C_dir + R_dir + T_dir + W_dir
    data = np.vstack((C_data, R_data, T_data, W_data))

    seg = min_num

    label = [0] * seg + [1] * seg + [2] * seg + [3] * seg
    data,label,data_dir = shuffle(data,label,data_dir,random_state = 0)
    s = len(label) // 3
    train,train_label,train_dir = data[:s],label[:s],data_dir[:s]
    val,val_label,val_dir = data[s:s*2],label[s:s*2],data_dir[s:s*2]
    test,test_label,test_dir = data[s*2:],label[s*2:],data_dir[s*2:]

    np.save('{}/{}_joint.npy'.format(out_path, 'train'), train)
    np.save('{}/{}_joint.npy'.format(out_path, 'val'), val)
    np.save('{}/{}_joint.npy'.format(out_path, 'test'), test)

    with open('{}/{}_label.pkl'.format(out_path, 'train'), 'wb') as f:
        pickle.dump((train_dir, list(train_label)), f)  # 保存文件名称和对应的缩写（作为标签）形成pkl文件。
    with open('{}/{}_label.pkl'.format(out_path, 'val'), 'wb') as f:
        pickle.dump((val_dir, list(val_label)), f)  # 保存文件名称和对应的缩写（作为标签）形成pkl文件。
    with open('{}/{}_label.pkl'.format(out_path, 'test'), 'wb') as f:
        pickle.dump((test_dir, list(test_label)), f)  # 保存文件名称和对应的缩写（作为标签）形成pkl文件。