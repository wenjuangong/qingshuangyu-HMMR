#工具文件，遍历所有数据找出所有段的持续时间和开始节点，找出最小的段和最小的帧长
import librosa
import librosa.core
import librosa.display
import matplotlib.pyplot as plt
import os
import numpy as np
import json
import math

from sklearn.preprocessing import StandardScaler

fps = 25
hop_length = 512
resample_rate = hop_length * fps

data_dir = './data/M2D_seg/'


class Music:

    '''
    @:param
        path:   music file path
        sr:     music sample rate
        start:  start offset(s) of the music
        end:    end offset(s) of the music
    '''
    def __init__(self, name,path, sr, start, duration):
        self.name = name
        self.path = path
        self.start = start
        self.duration = duration
        self.music_data, self.sr = librosa.core.load(path=path, sr=sr, offset=start,duration=duration)

    def extract_features(self):
        _,beats = librosa.beat.beat_track(y=self.music_data, sr = self.sr,hop_length=hop_length)
        restart = self.start + beats[0]/fps
        reduration = self.duration - beats[0]/fps
        
        return reduration * fps
    
        pass
    
def load_start_end_frame_num(config_fp):
    with open(config_fp, 'r') as f:
        data = json.load(f)
        start = data["start_position"]
        end = data["end_position"]
        return start,end
    pass

def load_skeleton(skeleton_json):
    with open(skeleton_json, 'r') as f:
        data = json.load(f)
        return data['length'],data['center'],data['skeletons']
    pass
def audio_feature_extract(data_dir,name):

    config_path = os.path.join(data_dir, "config.json")
    skeleton_path = os.path.join(data_dir, "skeletons.json")
    music_path = os.path.join(data_dir,"audio.mp3")
    start_frame, end_frame = load_start_end_frame_num(config_fp=config_path) # frame num
    duration,_,_ = load_skeleton(skeleton_json=skeleton_path)

    print("%s %d" % (data_dir,duration))
    music = Music(name,music_path, sr=resample_rate, start=start_frame / fps, duration=(duration-1) / fps) # 25fps
    return start_frame,end_frame,music.extract_features()


All_dir = os.listdir(data_dir)

C_dirs = []
R_dirs = []
T_dirs = []

C_dict = {}
R_dict = {}
T_dict = {}

for one in All_dir:
    if one.split('_')[1] == 'C':
        C_dirs.append(one)
    elif one.split('_')[1] == 'R':
        R_dirs.append(one)
    elif one.split('_')[1] == 'T':
        T_dirs.append(one)

for one in C_dirs:       
    one_dir = os.path.join(data_dir, one)
    
    start_frame, end_frame,duration = audio_feature_extract(one_dir,one)
    C_dict[one] = {'start_frame':start_frame,'end_frame':end_frame,'duration':duration}
    
for one in R_dirs:       
    one_dir = os.path.join(data_dir, one)
    
    start_frame, end_frame,duration = audio_feature_extract(one_dir,one)
    R_dict[one] = {'start_frame':start_frame,'end_frame':end_frame,'duration':duration}
    
for one in T_dirs:       
    one_dir = os.path.join(data_dir, one)
    
    start_frame, end_frame,duration = audio_feature_extract(one_dir,one)
    T_dict[one] = {'start_frame':start_frame,'end_frame':end_frame,'duration':duration}
    
min_C_duration = float('inf')
min_C_name = None

sum_C_duration = 0
sum_R_duration = 0
sum_T_duration = 0

for c in C_dict.keys():
    sum_C_duration += C_dict[c]['duration']
for t in T_dict.keys():
    sum_T_duration += T_dict[t]['duration']
    
for r in R_dict.keys():
    sum_R_duration += R_dict[r]['duration']


    
for c in C_dict.keys():
    if C_dict[c]['duration'] < min_C_duration:
        min_C_duration = C_dict[c]['duration']
        min_C_name = c
        
min_R_duration = float('inf')
min_R_name = None

for r in R_dict.keys():
    if R_dict[r]['duration'] < min_R_duration:
        min_R_duration = R_dict[r]['duration']
        min_R_name = r
        
min_T_duration = float('inf')
min_T_name = None

for t in T_dict.keys():
    if T_dict[t]['duration'] < min_T_duration:
        min_T_duration = T_dict[t]['duration']
        min_T_name = t
        
print(min_C_duration,min_C_name)
print(min_R_duration,min_R_name)
print(min_T_duration,min_T_name)
#去除W的最小值212后最小值是1494
min_duration = min(min_C_duration,min_T_duration,min_R_duration)
total_C = 0
total_R = 0
total_T = 0
#total_W = 0
for c in C_dict.keys():
    total_C += C_dict[c]['duration'] // min_duration
for r in R_dict.keys():
    total_R += R_dict[r]['duration'] // min_duration
for t in T_dict.keys():
    total_T += T_dict[t]['duration'] // min_duration
#for w in W_dict.keys():
#    total_W += W_dict[w]['duration'] // min_duration
print(total_C)#83 11
print(total_R)#85 11
print(total_T)#184 24
#print(total_W)#30
min_num = 9#W
#取最小值30，则将每个类别中的音乐分成30段，然后30 / 3 = 10，train，val，test分别有10段
