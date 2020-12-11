#正则化代码，将空帧用之前的帧长填充
import sys

sys.path.extend(['../'])
from rotation import *
from tqdm import tqdm

def pre_normalization(data, zaxis=[0, 1], xaxis=[8, 4]):
    N, C, T, V, M = data.shape
    s = np.transpose(data, [0, 4, 2, 3, 1])  # N, C, T, V, M  to  N, M, T, V, C (20046, 2, 300, 25, 3)

    print('pad the null frames with the previous frames')
    for i_s, skeleton in enumerate(tqdm(s)):  # pad
        if skeleton.sum() == 0:
            print(i_s, ' has no skeleton')
        for i_p, person in enumerate(skeleton):
            if person.sum() == 0:#person.shape = (300,25,3);
                continue
            if person[0].sum() == 0:#第一帧下25个节点的坐标
                index = (person.sum(-1).sum(-1) != 0)
                tmp = person[index].copy()
                person *= 0
                person[:len(tmp)] = tmp
            for i_f, frame in enumerate(person):#frame.shape = [25,3]
                if frame.sum() == 0:
                    if person[i_f:].sum() == 0:
                        rest = len(person) - i_f
                        num = int(np.ceil(rest / i_f))
                        pad = np.concatenate([person[0:i_f] for _ in range(num)], 0)[:rest]
                        s[i_s, i_p, i_f:] = pad
                        break
                        
                        
    data = np.transpose(s, [0, 4, 2, 3, 1])
    return data#(20046, 3, 300, 25, 2)