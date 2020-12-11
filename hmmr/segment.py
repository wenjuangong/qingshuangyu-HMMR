#将一段motion对应多条music，和将一条music对应多段motion
import os
from shutil import copyfile

datadir = './data/M2D_raw/'
outputdir = './data/M2D_seg/'
if __name__=='__main__':
    All_dirs =os.listdir(datadir)
    #一段motion对应该类别中所有的music
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
    for i in C_dirs:
        music_path = os.path.join(datadir,i,'audio.mp3')
        
        for n,j in enumerate(C_dirs):
#             config_path = os.path.join(datadir,j,'config.json')
            motion_path = os.path.join(datadir,j,'skeletons.json')
    
            config_path = os.path.join(datadir,j,'config.json')
        
            path_name = i + '_' + str(n)
            path = os.path.join(outputdir,path_name)
            if not os.path.exists(path):
                os.makedirs(path)
            motion = os.path.join(path,'skeletons.json')
            music = os.path.join(path,'audio.mp3')
            config = os.path.join(path,'config.json')
            copyfile(motion_path, motion)
            copyfile(music_path,music)
            copyfile(config_path,config)
            
    for i in R_dirs:
        music_path = os.path.join(datadir,i,'audio.mp3')
        
        for n,j in enumerate(R_dirs):
#             config_path = os.path.join(datadir,j,'config.json')
            motion_path = os.path.join(datadir,j,'skeletons.json')
    
            config_path = os.path.join(datadir,j,'config.json')
        
            path_name = i + '_' + str(n)
            path = os.path.join(outputdir,path_name)
            if not os.path.exists(path):
                os.makedirs(path)
            motion = os.path.join(path,'skeletons.json')
            music = os.path.join(path,'audio.mp3')
            config = os.path.join(path,'config.json')
            copyfile(motion_path, motion)
            copyfile(music_path,music)
            copyfile(config_path,config)
            
    for i in T_dirs:
        music_path = os.path.join(datadir,i,'audio.mp3')
        
        for n,j in enumerate(T_dirs):
#             config_path = os.path.join(datadir,j,'config.json')
            motion_path = os.path.join(datadir,j,'skeletons.json')
    
            config_path = os.path.join(datadir,j,'config.json')
        
            path_name = i + '_' + str(n)
            path = os.path.join(outputdir,path_name)
            if not os.path.exists(path):
                os.makedirs(path)
            motion = os.path.join(path,'skeletons.json')
            music = os.path.join(path,'audio.mp3')
            config = os.path.join(path,'config.json')
            copyfile(motion_path, motion)
            copyfile(music_path,music)
            copyfile(config_path,config)
            

