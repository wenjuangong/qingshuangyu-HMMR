##读取每个motion的长度将motion统一到指定的长度。
#然后看看最小的长度每一个数据能按这种长度分成几段，让每一类中的段数相同
import os
import numpy as np
from m2d_gen import motion_feature_extract
with_centering = True
with_rotate = False
Datadir = '../../../data/'
def count_frames(datadir):
    counts = []
    C_sum = []
    R_sum = []
    T_sum = []
    W_sum = []
    for i,d in enumerate(datadir):
        data_path = os.path.join(Datadir, d)
        temp = motion_feature_extract(data_path, with_centering, with_rotate)
        if d.split('_')[1] == 'C':
            C_sum.append(temp.shape[0])
        elif d.split('_')[1] == 'R':
            R_sum.append(temp.shape[0])
        elif d.split('_')[1] == 'T':
            T_sum.append(temp.shape[0])
        else:
            W_sum.append(temp.shape[0])
        counts.append(temp.shape[0])
    counts_min = min(counts)
    counts_mean = np.mean(counts)
    counts_max = max(counts)
    duration = counts_min#1.先将最小的持续时间设置为所有值得最小值看看
    C_seg = sum(np.array(C_sum) // duration)
    R_seg = sum(np.array(R_sum) // duration)
    T_seg = sum(np.array(T_sum) // duration)
    W_seg = sum(np.array(W_sum) // duration)
    min_seg = min(C_seg,T_seg,W_seg,R_seg)
    print(counts)
    print(min_seg)
if __name__ == '__main__':
    All_dirs = os.listdir(Datadir)
    All_dirs.sort()
    count_frames(All_dirs)