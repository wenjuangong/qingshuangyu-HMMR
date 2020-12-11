import numpy as np
import json
import os

seq_len = 120
data_dir = "../../data/"

ori_npy = 'acoustic_features.npy'
pre_npy = 'LSTM-AE_rotate_Ortho_Leaky_InputSize_69_Seq_120_Overlap_Threshold_0.400_Reduced_10.npy'

def cos_sim(vector_a,vector_b):
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = vector_a * vector_b.T
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom
    sim = 0.5 + 0.5 * cos
    return np.sum(sim)

if __name__ == '__main__':
    with open('data_dict.json', 'r') as f:
        data_dict = json.load(f)
    test_dirs = data_dict['test']
    data = os.listdir(data_dir)
    All_dict = {}
    count_label = 0
    for i in data:
        All_dict[i] = count_label
        count_label += 1
    ori_data = []
    ori_label = []
    for i in range(len(data)):
        temp_dir = os.path.join(data_dir,data[i],ori_npy)
        temp_data = np.load(temp_dir)
        ori_data.append(temp_data)
        ori_label.append(All_dict[data[i]])

    pre_data = []
    pre_label = []
    for i in range(len(test_dirs)):
        temp_dir = os.path.join(test_dirs[i], pre_npy)
        temp_data = np.load(temp_dir)

        pre_data.append(temp_data)

        pre_label.append(All_dict[test_dirs[i].split('/')[-1]])

    pre = []
    for i in range(len(pre_data)):
        similiar = -float('inf')
        sim_label = None
        for j in range(len(ori_data)):
            temp_sim = cos_sim(pre_data[i],ori_data[j])
            if temp_sim > similiar:
                similiar = temp_sim
                sim_label = ori_label[j]
        pre.append(sim_label)
    count = 0
    for i, j in zip(pre, pre_label):
        if i == j:
            count += 1
    print("test original label: ",pre_label)
    print('\n')
    print("test predict label: ", pre)
    print('\n')
    print('Accuracy: ',count / len(pre))


