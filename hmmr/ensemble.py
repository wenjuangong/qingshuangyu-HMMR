#To ensemble the results of joints and bones,将joints和bone的试验结果结合起来，这种方式是将最后的结果结合起来，而不是将两种特征叠加起来输入到网络中
import argparse
import pickle

import numpy as np
from tqdm import tqdm
out_path = '../input/M2D_segment_frames/'
parser = argparse.ArgumentParser()
parser.add_argument('--datasets', default='m2d_segment', choices={'kinetics', 'ntu/xsub', 'ntu/xview'},
                    help='the work folder for storing results')
parser.add_argument('--alpha', default=1, help='weighted summation')
arg = parser.parse_args()

dataset = arg.datasets
label = open(out_path + '/test_label.pkl', 'rb')
label = np.array(pickle.load(label))
r1 = open('./work_dir/' + dataset + '/m2d_test_data/epoch1_test_score.pkl', 'rb')
r1 = list(pickle.load(r1).items())
r2 = open('./work_dir/' + dataset + '/m2d_test_bone/epoch1_test_score.pkl', 'rb')
r2 = list(pickle.load(r2).items())
right_num = total_num = right_num_5 = 0
text_path = './ensemble.txt'
f_w = open(text_path,'w')
for i in tqdm(range(len(label[0]))):
    _, l = label[:, i]
    _, r11 = r1[i]
    _, r22 = r2[i]
    r = r11 + r22 * arg.alpha
    rank_5 = r.argsort()[-5:]
    right_num_5 += int(int(l) in rank_5)
    r = np.argmax(r)
    f_w.write(str(l)+','+str(r) + '\n')#真，预测
    right_num += int(r == int(l))
    total_num += 1
acc = right_num / total_num
acc5 = right_num_5 / total_num
print(acc, acc5)
