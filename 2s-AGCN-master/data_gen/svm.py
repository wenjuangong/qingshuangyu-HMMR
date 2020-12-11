#这个文件是将经过网络训练的全连接层前的输出作为motion特征保存起来，同时将motion和music进行svm分类
import numpy as np
import os
from sklearn import svm
datadir = '../data/M2D/music/'
from sklearn.model_selection import train_test_split as ts
from sklearn.cluster import KMeans

if __name__ == '__main__':
    with open(datadir + 'motion_net_train.npy','rb') as f:
        train_motion = np.load(f)
    with open(datadir + 'motion_net_val.npy','rb') as f:
        val_motion = np.load(f)
    with open(datadir + 'motion_net_test.npy','rb') as f:
        test_motion = np.load(f)
    with open(datadir + 'train_data.npy','rb') as f:
        Train_music = np.load(f)
    with open(datadir + 'val_data.npy','rb') as f:
        Val_music = np.load(f)
    with open(datadir + 'test_data.npy','rb') as f:
        Test_music = np.load(f)
        
    ##对music进行聚类的运算
    
#     n_clusters = 256
#     train_music = np.zeros((87,256))
#     for i in range(87):
#         kmean = KMeans(n_clusters = n_clusters)

#         temp = Train_music[i].reshape(-1,1)

#         kmean.fit(temp)

#         center = kmean.cluster_centers_.squeeze()
#         print(center.shape)
#         train_music[i] = center

#     print(train_music.shape)
#     n_clusters = 256
#     val_music = np.zeros((87,256))
#     for i in range(87):
#         kmean = KMeans(n_clusters = n_clusters)

#         temp = Val_music[i].reshape(-1,1)

#         kmean.fit(temp)

#         center = kmean.cluster_centers_.squeeze()
#         print(center.shape)
#         val_music[i] = center

#     print(val_music.shape)
#     n_clusters = 256
#     test_music = np.zeros((87,256))
#     for i in range(87):
#         kmean = KMeans(n_clusters = n_clusters)

#         temp = Test_music[i].reshape(-1,1)

#         kmean.fit(temp)

#         center = kmean.cluster_centers_.squeeze()
#         print(center.shape)
#         test_music[i] = center

#     print(test_music.shape)
    
    train_music = Train_music
    val_music = Val_music
    test_music = Test_music

    print('---------------Motion-------------')
    train_motion = train_motion.reshape(train_motion.shape[0],-1)
    val_motion = val_motion.reshape(val_motion.shape[0],-1)
    test_motion = test_motion.reshape(test_motion.shape[0],-1)
    
    print(train_motion.shape,val_motion.shape,test_motion.shape)
    print('---------------Music---------------')
    print(train_music.shape,val_music.shape,test_music.shape)
    train_music = train_music.reshape(train_music.shape[0],-1)
    val_music = val_music.reshape(val_music.shape[0],-1)
    test_music = test_music.reshape(test_music.shape[0],-1)
    print('---------------After Music---------------')
    print(train_music.shape,val_music.shape,test_music.shape)
    
    
    
    
    data_motion = np.vstack((train_motion,test_motion))
    data_music = np.vstack((train_music,test_music))
    print(data_motion.shape,data_music.shape)
    data = np.hstack((data_motion,data_music))
    print(data.shape)
    label =  ([0]*29 + [1] * 29 + [2] * 29)*2
    X_train,X_test,y_train,y_test = ts(data,label,test_size=0.3)
    #这样设置标签的话就是一个三分类的问题了。
    #如果标签设置为0，1则还需要制造假数据
    clf = svm.SVC(C = 0.8,kernel = 'rbf',gamma = 'auto',decision_function_shape = 'ovr')
    clf.fit(X_train,y_train)
    print(clf.score(X_train,y_train))
    y_train_predict = clf.predict(X_train)
    print(y_train_predict)
    print(clf.score(X_test,y_test))
    y_test_predict = clf.predict(X_test)
    print(y_test_predict)

    
    print('将其分为0，1标签')
    train_data = np.empty([174,train_motion.shape[1] + train_music.shape[1]])
    for i in range(train_motion.shape[0]):
        if i == train_motion.shape[0] - 1:
            temp_T = np.hstack((train_motion[i],train_music[i]))
            temp_F = np.hstack((train_motion[i],train_music[0]))
            np.append(train_data,temp_T)
            np.append(train_data,temp_F)
        else:
            temp_T = np.hstack((train_motion[i],train_music[i]))
            temp_F = np.hstack((train_motion[i],train_music[i + 1]))
            np.append(train_data,temp_T)
            np.append(train_data,temp_F)
    print(train_data.shape)
    
#     test_data = np.empty([174,test_motion.shape[1] + test_music.shape[1]])
#     for i in range(test_motion.shape[0]):
#         if i == test_motion.shape[0] - 1:
#             temp_T = np.hstack((test_motion[i],test_music[i]))
#             temp_F = np.hstack((test_motion[i],test_music[0]))
#             np.append(test_data,temp_T)
#             np.append(test_data,temp_F)
#         else:
#             temp_T = np.hstack((test_motion[i],test_music[i]))
#             temp_F = np.hstack((test_motion[i],test_music[i + 1]))
#             np.append(test_data,temp_T)
#             np.append(test_data,temp_F)
    #不对test数据进行添加妨碍数据的做法。        
    test_data = np.empty([87,test_motion.shape[1] + test_music.shape[1]])
    for i in range(test_motion.shape[0]):
        temp_T = np.hstack((test_motion[i],test_music[i]))

        np.append(test_data,temp_T)
    print(test_data.shape)
    train_label = [0,1] * 87
    test_label = [0] * 87
    #0代表正确，1代表错误
#     data = np.vstack((train_data,test_data))
#     print(data.shape)
    X_train,X_test,y_train,y_test = train_data,test_data,train_label,test_label
    clf = svm.SVC(C = 1,kernel = 'rbf',gamma = 'auto',decision_function_shape = 'ovo',max_iter = -1)
    clf.fit(X_train,y_train)
    print(clf.score(X_train,y_train))
    y_train_predict = clf.predict(X_train)
    print(y_train_predict)
    print(clf.score(X_test,y_test))
    y_test_predict = clf.predict(X_test)
    print(y_test_predict)