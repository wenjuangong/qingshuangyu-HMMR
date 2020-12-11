#将经过PCA后的motion进行分段，产生标签，生成网络的输入数据，节点部分
import os
import pickle
import json
import math
import numpy as np
from m2d_pre import pre_normalization
import random
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
fps = 25
num_joint = 23
max_body = 1
datadir = '../../data/'
out_path = '../../input/M2D/'

def load_skeleton(skeleton_json):
    with open(skeleton_json, 'r') as f:
        data = json.load(f)
#         print(data['skeletons'])
        return data['length'],data['center'],data['skeletons'] #length是节点的帧数，以第一个文件为例，4300，center是每一帧有一个中心位置，是一个二维列表，4300x3,skeletons是所有帧的节点，4300x23x3,
    pass

def rotate_one_skeleton_by_axis(skeleton, axis, angle):
    delta_x = skeleton[0] - axis[0]
    delta_z = skeleton[2] - axis[2]
    skeleton_new = skeleton
    skeleton_new[0] = delta_x * math.cos(angle) + delta_z * math.sin(angle)
    skeleton_new[2] = -delta_x * math.sin(angle) + delta_z * math.cos(angle)


    return skeleton_new

def rotate_skeleton(frames):

    frames = np.asarray(frames) # 4300 23 3

    for i in range(len(frames)):
        this_frame = frames[i]
        waist_lf = this_frame[16] #左腰子
        waist_rt = this_frame[7]  #右腰子

        axis = this_frame[2] #平衡点

        lf = waist_lf - axis
        rt = waist_rt - axis
        mid = lf+rt

        theta = math.atan2(mid[2], mid[0]) # from x+ axis

        for j in range(len(this_frame)):
            frames[i][j] =  rotate_one_skeleton_by_axis(this_frame[j], axis, theta)
            frames[i][j] =  rotate_one_skeleton_by_axis(this_frame[j], axis, -math.pi/2) # turn to y- axis

    return frames
def motion_feature_extract(data_path, with_rotate, with_centering):
    skeleton_path = os.path.join(data_path, "skeletons.json")
    #print(skeleton_path)
    duration, center, frames = load_skeleton(skeleton_json=skeleton_path)
    center = np.asarray(center)
    #print(center.shape) #4300x3
    frames = np.asarray(frames)
    if with_centering:
        for i in range(len(frames)):
            for j in range(len(frames[i])):
                frames[i][j] -= center[i]
#将节点进行旋转
    if with_rotate:
        frames = rotate_skeleton(frames)
    return frames

    pass
#读取这个骨架的json文件有两个问题是：是否需要旋转，去中心化；每一个json文件都有不同的帧数长度：帧数 * 23 * 3；比2sgcn网络少了人物的维度，而且帧数是不确定的。
#接受数组，产生文件
def genda(data_dirs,All_data,max_interval):
    label = []
    label_dict = {"index":0}
    input_data = []
    for i, one in enumerate(data_dirs):
        data_path = os.path.join(datadir,one)
        motion_features_pre = motion_feature_extract(data_path, with_centering=False, with_rotate=True)
        motion_features = motion_features_pre.reshape(motion_features_pre.shape[0],-1)
        #############################################################
#标准化：
       # scaler = preprocessing.StandardScaler().fit(motion_features)#正态分布，得到的数据在0附近，方差1
    
#归一化，让数据分布到指定的范围
        min_max_scaler = preprocessing.MinMaxScaler().fit_transform(motion_features)
        motion_features = min_max_scaler.reshape(min_max_scaler.shape[0],23,3)

        motion_features = np.expand_dims(motion_features,3) # 在最后面加一个维度作为原始数据中的演员数（帧数，节点数，节点维度，1）
        # motion_features = np.transpose(motion_features,[2,0,1,3])
        motion_features = motion_features.reshape(len(motion_features),-1)#这里将4300，23，3，1变成4300，69
        for j in range(len(All_data[i])):
            if len(All_data[i][j]) < 3:
                continue
            start = All_data[i][j][0]
            end = All_data[i][j][1]
            l = All_data[i][j][2]
            if end - start == max_interval:
                input_data.append(motion_features[int(start * fps):int(end * fps)])
            else:
                interval = max_interval/(end - start)
                index = 0
                temp = np.empty([0,69])
                while index < int(interval):
                    index += 1
                    temp = np.vstack((temp,motion_features[int(start * fps):int(end * fps)]))
                if temp.shape[0] != max_interval * fps:
                    itv = max_interval * fps - temp.shape[0]
                    temp = np.vstack((temp,motion_features[int(start * fps):int(start * fps) + int(itv)]))
                input_data.append(temp)
            if l not in label_dict:
                label_dict[l] = label_dict['index']
                label_dict['index'] += 1
                label.append(label_dict[l])
            else:
                label.append(label_dict[l])
    # 接下来把input转化成正确格式保存
    for i in range(len(input_data)):
        if input_data[i].shape[0] > 50:
            input_data[i] = input_data[i][:50]
        if input_data[i].shape[0] < 50:
            input_data[i] = np.vstack((input_data[i],input_data[i][:50 - input_data[i].__len__()]))
    inputs = np.empty([input_data.__len__(),50,69])
    for i in range(len(input_data)):
        inputs[i] = input_data[i]
    print(inputs.shape)
    inputs = inputs.reshape(-1,50,23,3,1)
    inputs = np.transpose(inputs,[0,3,1,2,4])
    # for i in range(len(label)):
    #     label[i] -= 1#这里是把每一个数据作为标签时要把标签减1，因为是从1开始的，而我们现在每一段一个标签用的是字典的方式求解

    inputs,label = shuffle(inputs,label,random_state = 0)#标签种类为10083
    train_set,test_set,train_label,test_label = train_test_split(inputs,label,test_size=0.4,random_state=0)
    # train_set = inputs[:len(inputs)//3]
    # train_label = label[:len(inputs)//3]
    # val_set = inputs[len(inputs) // 3:2 * len(inputs) // 3]
    # val_label = label[len(inputs)//3:2 * len(inputs)//3]
    # test_set = inputs[2 * len(inputs) // 3:]  # (61, 3, 300, 23, 1)
    # test_label = label[2 * len(inputs) // 3:]
    np.save('{}/{}_data_joint.npy'.format(out_path, 'train'), train_set)
    np.save('{}/{}_data_joint.npy'.format(out_path, 'val'), test_set)
    np.save('{}/{}_data_joint.npy'.format(out_path, 'test'), inputs)
    with open('{}/{}_label.pkl'.format(out_path, 'test'), 'wb') as f:
        pickle.dump(list(label), f)#保存文件名称和对应的缩写（作为标签）形成pkl文件。
    with open('{}/{}_label.pkl'.format(out_path, 'train'), 'wb') as f:
        pickle.dump(list(train_label), f)
    with open('{}/{}_label.pkl'.format(out_path, 'val'), 'wb') as f:
        pickle.dump(list(test_label), f)
    # with open('{}/{}_label.pkl'.format(out_path, 'test'), 'wb') as f:
    #     pickle.dump(list(test_label), f)
if __name__ == '__main__':
    All_data = np.load('../../../data.npy',allow_pickle=True)

    max_interval = 0
    for i in All_data:
        for j in i:
            temp_interval = j[1] - j[0]
            max_interval = max(max_interval,temp_interval)
    All_dirs = os.listdir(datadir)
    All_dirs.sort()
    genda(All_dirs,All_data,max_interval)
