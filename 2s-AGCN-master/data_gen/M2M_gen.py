#试验一：数据在进行train，val,test之前打乱，这个文件是将全部分段，然后不是从一个数据中分train，val,test而是从所有数据中进行分割
import os
import pickle
import json
import math
import numpy as np
import random


Datadir = '../../data/'
out_path = '../../input/M2M_gen'

min_duration = 208
num_joint = 23
max_body = 1

from m2d_pre import pre_normalization
from sklearn import preprocessing
from m2d_gen import motion_feature_extract

# 读取这个骨架的json文件有两个问题是：是否需要旋转，去中心化；每一个json文件都有不同的帧数长度：帧数 * 23 * 3；比2sgcn网络少了人物的维度，而且帧数是不确定的。
# 接受数组，产生文件
def genda(type,data_dirs):
    data = []
    label = []
    data_dir = []
    for i, one in enumerate(data_dirs):
        data_path = os.path.join(Datadir, one)
        motion_features_pre = motion_feature_extract(data_path, with_centering=True, with_rotate=False)
        #正则化，对于每一个数据都正则化
        motion_features = motion_features_pre.reshape(motion_features_pre.shape[0], -1)
        min_max_scaler = preprocessing.MinMaxScaler().fit_transform(motion_features)
        motion_features = min_max_scaler.reshape(min_max_scaler.shape[0], 23, 3)

        t = motion_features.shape[0] // min_duration
        if one.split('_')[1] == 'C':
            temp_label = 0
        elif one.split('_')[1] == 'R':
            temp_label = 1
        else:
            temp_label = 2
        for j in range(t):
            data_dir.append(one)
            label.append(temp_label)
            temp_features = motion_features[min_duration * j:min_duration * (j + 1), :, :]
            temp_features = np.expand_dims(temp_features, 3)
            temp_features = np.transpose(temp_features, [2, 0, 1, 3])
            data.append(temp_features)
    data = np.array(data)
    data = pre_normalization(data)
    np.save('{}/{}_data_joint.npy'.format(out_path, type), data)

    with open('{}/{}_label.pkl'.format(out_path, type), 'wb') as f:
        pickle.dump((data_dir, list(label)), f)  # 保存文件名称和对应的缩写（作为标签）形成pkl文件。

if __name__ == '__main__':
    with open('data_dict_withoutW.json','r') as f:
        data_dict = json.load(f)
    train_dirs = data_dict['train']
    valid_dirs = data_dict['valid']
    test_dirs = data_dict['test']
    train_data_dir = []
    valid_data_dir = []
    test_data_dir = []
    for i in train_dirs:
        train_data_dir.append(i.split('/')[-1])
    for i in valid_dirs:
        valid_data_dir.append(i.split('/')[-1])
    for i in test_dirs:
        test_data_dir.append(i.split('/')[-1])
    genda('train',train_data_dir)
    genda('valid',valid_data_dir)
    genda('test',test_data_dir)