#一个motion和多个music进行匹配增大数据量
datadir = '../data/M2D_seg/'
import os
import pickle
import json
import math
import numpy as np
import librosa
import librosa.core
import librosa.display

import random

out_path = '../data/m2d/music/'
fps = 25
hop_length = 512
resample_rate = hop_length * fps
window_length = hop_length * 2
min_duration = 1472
min_num = 87

from sklearn import preprocessing


class Music:

    '''
    @:param
        path:   music file path
        sr:     music sample rate
        start:  start offset(s) of the music
        end:    end offset(s) of the music
    '''
    def __init__(self, path, sr, start, duration):
        self.path = path
        self.start = start
        self.music_data, self.sr = librosa.core.load(path=path, sr=sr, offset=start,duration=duration) # 210秒
#         librosa.output.write_wav(path=path.replace("mp3","wav"),y=self.music_data, sr=self.sr)

    def extract_features(self):
        # 40ms / frame
        mel_spectrum = librosa.feature.melspectrogram(y=self.music_data, sr=self.sr,n_fft=window_length, hop_length=hop_length)
        _,beats = librosa.beat.beat_track(y=self.music_data, sr = self.sr,hop_length=hop_length)
        mfcc = librosa.feature.mfcc(S=mel_spectrum,n_mfcc=20) # mfcc[3]
        
        acoustic_features = mfcc
        acoustic_features = acoustic_features[:, beats[0]:]

        return acoustic_features.transpose()

        pass
    
def load_start_end_frame_num(config_fp):
    with open(config_fp, 'r') as f:
        data = json.load(f)
        start = data["start_position"]
        end = data["end_position"]
        return start,end
    pass




'''
Get frame len, center array, skeletons array
'''
def load_skeleton(skeleton_json):
    with open(skeleton_json, 'r') as f:
        data = json.load(f)
        return data['length'],data['center'],data['skeletons']
    pass

def audio_feature_extract(data_dir):

    config_path = os.path.join(data_dir, "config.json")
    skeleton_path = os.path.join(data_dir, "skeletons.json")
    music_path = os.path.join(data_dir,"audio.mp3")

    start_frame, end_frame = load_start_end_frame_num(config_fp=config_path) # frame num
    duration,_,_ = load_skeleton(skeleton_json=skeleton_path)

    print("%s %d" % (data_dir,duration))
    music = Music(music_path, sr=resample_rate, start=start_frame / fps, duration=(duration-1) / fps) # 25fps

    acoustic_features = music.extract_features()  # 16 dim

#     np.save(acoustic_features_path, acoustic_features)
    
    return acoustic_features
def genda(data_dirs):
    
    train_dir = []
    test_dir = []
    val_dir = []
    
    seg = min_num // 3#整除3，将其分为三段，train，val,test
    
    train = np.zeros((seg, min_duration, 20), dtype=np.float32)
    val = np.zeros((seg, min_duration, 20), dtype=np.float32)
    test = np.zeros((seg, min_duration, 20), dtype=np.float32)
    
    temp_num = 0
    train_i = 0
    val_i = 0
    test_i = 0
    
    for i, one in enumerate(data_dirs):
        
        data_path = os.path.join(datadir,one)
        
        music_features = audio_feature_extract(data_path)
        #标准化
        music_features = preprocessing.MinMaxScaler().fit_transform(music_features)
        
        t = music_features.shape[0] // min_duration
        
        for j in range(t):
            
            temp_num += 1
            
            if temp_num <= seg:
                #train
                temp_features = music_features[min_duration * j:min_duration * (j + 1),:]
                
                train[train_i,:,:] = temp_features
                train_dir.append(one)
                train_i += 1
            elif temp_num > seg and temp_num<= seg * 2:
                #val
                temp_features = music_features[min_duration * j:min_duration * (j + 1),:]

                val[val_i,:,:] = temp_features
                val_dir.append(one)
                val_i += 1
            elif temp_num > seg * 2 and temp_num <= seg*3:
               #test
                temp_features = music_features[min_duration * j:min_duration * (j + 1),:]

                test[test_i,:,:] = temp_features
                test_dir.append(one)
                test_i += 1
            else:
                break

    return train,train_dir,val,val_dir,test,test_dir

if __name__ == '__main__':
    All_dirs = os.listdir(datadir)
    C_dirs = []
    R_dirs = []
    T_dirs = []

    for one in All_dirs:
        if one.split('_')[1] == 'C':
            C_dirs.append(one)
        elif one.split('_')[1] == 'R':
            R_dirs.append(one)
        elif one.split('_')[1] == 'T':
            T_dirs.append(one)
    C_dirs.sort()
    R_dirs.sort()
    T_dirs.sort()
    
    C_train,C_train_dir,C_val,C_val_dir,C_test,C_test_dir = genda(C_dirs)
    R_train,R_train_dir,R_val,R_val_dir,R_test,R_test_dir = genda(R_dirs)
    T_train,T_train_dir,T_val,T_val_dir,T_test,T_test_dir = genda(T_dirs)
    
    train = np.vstack((C_train,R_train,T_train))#(3, 3, 1494, 23, 1)
    val = np.vstack((C_val,R_val,T_val))
    test = np.vstack((C_test,R_test,T_test))
    train_dir = C_train_dir + R_train_dir + T_train_dir
    test_dir = C_test_dir + R_test_dir + T_test_dir
    val_dir = C_val_dir + R_val_dir + T_val_dir
    
    train_label = [0]*29 + [1] * 29 + [2] * 29
    val_label = [0]*29 + [1] * 29 + [2] * 29
    test_label = [0]*29 + [1] * 29 + [2] * 29
    
    np.save('{}/{}_data.npy'.format(out_path, 'train'), train)
    np.save('{}/{}_data.npy'.format(out_path,'val'), val)
    np.save('{}/{}_data.npy'.format(out_path,'test'), test)

    with open('{}/{}_label.pkl'.format(out_path,'train' ), 'wb') as f:
        pickle.dump((train_dir, list(train_label)), f)#保存文件名称和对应的缩写（作为标签）形成pkl文件。
    with open('{}/{}_label.pkl'.format(out_path,'val'), 'wb') as f:
        pickle.dump((val_dir, list(val_label)), f)#保存文件名称和对应的缩写（作为标签）形成pkl文件。
    with open('{}/{}_label.pkl'.format(out_path,'test'), 'wb') as f:
        pickle.dump((test_dir, list(test_label)), f)#保存文件名称和对应的缩写（作为标签）形成pkl文件。