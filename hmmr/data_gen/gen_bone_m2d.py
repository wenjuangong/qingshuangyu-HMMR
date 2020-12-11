#这里是产生骨骼边长的数据，用于emerge将节点和骨骼结合起来，但是因为节点的个数和骨骼的个数不同就难办了，应该不能组合把
#先只用关节点的数据进行训练吧
import os
import numpy as np
from numpy.lib.format import open_memmap
import tqdm
out_path = '../../input/M2D_segment_frames/'
m2d = ((3,2),(4,3),(5,4),(6,5),(8,7),(9,8),(10,9),(11,10),(13,12),(14,13),(15,14),(17,16),(18,17),(19,18),
          (20,19),(7,2),(12,2),(16,2))
#下面这个版本是现在的完整画法，上面那个为了使不同节点保持相同的向心性，所以省略了一些节点。
m2d_seg = ((0,1),(2,3),(3,4),(4,5),(5,6),(7,8),(8,9),(9,10),(10,11),(12,13),(13,14),(14,15),(16,17),(17,18),(18,19),
          (19,20),(2,7),(2,12),(2,16),(3,7),(3,12),(7,16),(12,16))

sets = {
    'train', 'val','test'
}
for set in sets:
    data = np.load(out_path + '{}_joint.npy'.format(set))#(20046, 3, 300, 25, 2)
    N, C, T, V, M = data.shape
    fp_sp = open_memmap(out_path + '{}_bone.npy'.format(set),dtype='float32',mode='w+',shape=(N, 3, T, V, M))
#(20046, 3, 300, 25, 2)
    fp_sp[:, :C, :, :, :] = data
    for i,(v1, v2) in enumerate(m2d_seg):
        fp_sp[:, :, :, i, :] = data[:, :, :, v1, :] - data[:, :, :, v2, :]#存储v1的节点到v2的节点的长度（v1- v2）(20046, 3, 300, 25, 2)
    print(fp_sp.shape)
