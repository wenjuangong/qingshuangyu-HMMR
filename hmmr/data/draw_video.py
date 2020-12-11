import numpy as np
import cv2
import json
import os
from moviepy.editor import *
CANVAS_SIZE = (900,600,3)
fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
fps = 25

def draw(frames,video_path):
    frames[:,:,0] += CANVAS_SIZE[0]//2#x
    frames[:,:,1] += CANVAS_SIZE[1]//2#y
    video = cv2.VideoWriter(video_path, fourcc, fps, (CANVAS_SIZE[0],CANVAS_SIZE[1]), 1)
    for i in range(len(frames)):
        cvs = np.ones((CANVAS_SIZE[1], CANVAS_SIZE[0], CANVAS_SIZE[2]))
        cvs[:,:,:] = 255
        color = (0,0,0)
        hlcolor = (255,0,0)
        dlcolor = (0,0,255)
        for points in frames[i]:
            cv2.circle(cvs,(int(points[0]),int(points[1])),radius=4,thickness=-1,color=hlcolor)
        frame = frames[i]
        
        cv2.line(cvs, (int(frame[0][0]), int(frame[0][1])), (int(frame[1][0]), int(frame[1][1])), color, 2)
        cv2.line(cvs, (int(frame[2][0]), int(frame[2][1])), (int(frame[3][0]), int(frame[3][1])), color, 2)
        cv2.line(cvs, (int(frame[3][0]), int(frame[3][1])), (int(frame[4][0]), int(frame[4][1])), color, 2)
        cv2.line(cvs, (int(frame[4][0]), int(frame[4][1])), (int(frame[5][0]), int(frame[5][1])), color, 2)
        cv2.line(cvs, (int(frame[5][0]), int(frame[5][1])), (int(frame[6][0]), int(frame[6][1])), color, 2)
        cv2.line(cvs, (int(frame[7][0]), int(frame[7][1])), (int(frame[8][0]), int(frame[8][1])), color, 2)
        cv2.line(cvs, (int(frame[8][0]), int(frame[8][1])), (int(frame[9][0]), int(frame[9][1])), color, 2)
        cv2.line(cvs, (int(frame[9][0]), int(frame[9][1])), (int(frame[10][0]), int(frame[10][1])), color, 2)
        cv2.line(cvs, (int(frame[10][0]), int(frame[10][1])), (int(frame[11][0]), int(frame[11][1])), color, 2)
        cv2.line(cvs, (int(frame[12][0]), int(frame[12][1])), (int(frame[13][0]), int(frame[13][1])), color, 2)
        cv2.line(cvs, (int(frame[13][0]), int(frame[13][1])), (int(frame[14][0]), int(frame[14][1])), color, 2)
        cv2.line(cvs, (int(frame[14][0]), int(frame[14][1])), (int(frame[15][0]), int(frame[15][1])), color, 2)
        cv2.line(cvs, (int(frame[16][0]), int(frame[16][1])), (int(frame[17][0]), int(frame[17][1])), color, 2)
        cv2.line(cvs, (int(frame[17][0]), int(frame[17][1])), (int(frame[18][0]), int(frame[18][1])), color, 2)
        cv2.line(cvs, (int(frame[18][0]), int(frame[18][1])), (int(frame[19][0]), int(frame[19][1])), color, 2)
        cv2.line(cvs, (int(frame[19][0]), int(frame[19][1])), (int(frame[20][0]), int(frame[20][1])), color, 2)
        cv2.line(cvs, (int(frame[2][0]), int(frame[2][1])), (int(frame[7][0]), int(frame[7][1])), color, 2)
        cv2.line(cvs, (int(frame[2][0]), int(frame[2][1])), (int(frame[12][0]), int(frame[12][1])), color, 2)
        cv2.line(cvs, (int(frame[2][0]), int(frame[2][1])), (int(frame[16][0]), int(frame[16][1])), color, 2)
        cv2.line(cvs, (int(frame[3][0]), int(frame[3][1])), (int(frame[7][0]), int(frame[7][1])), color, 2)
        cv2.line(cvs, (int(frame[3][0]), int(frame[3][1])), (int(frame[12][0]), int(frame[12][1])), color, 2)
        cv2.line(cvs, (int(frame[7][0]), int(frame[7][1])), (int(frame[16][0]), int(frame[16][1])), color, 2)
        cv2.line(cvs, (int(frame[12][0]), int(frame[12][1])), (int(frame[16][0]), int(frame[16][1])), color, 2)

        '''
        for j in range(23):
            cv2.putText(cvs,str(j),(int(frame[j][0]),int(frame[j][1])),cv2.FONT_HERSHEY_SIMPLEX,.4, (155, 0, 255), 1, False)
            '''
        #cv2.imshow('canvas',np.flip(cvs,0))
        #cv2.waitKey(0)
        ncvs = np.flip(cvs, 0).copy()
        video.write(np.uint8(ncvs))
    video.release()
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

if __name__ == '__main__':
    data_dir = '../../../../data/'
    data_list = os.listdir(data_dir)
    for i in data_list:
        if i.split('_')[0] != 'DANCE':
            continue
        skeletons = os.path.join(data_dir,i,'skeletons.json')
        video_path = os.path.join(data_dir,i,'temp.mp4')
        with open(skeletons,'r') as fin:
            data = json.load(fin)
        draw(np.array(data['skeletons']),video_path)

        audio = os.path.join(data_dir, i, 'audio.mp3')
        config_path = os.path.join(data_dir,i,'config.json')

        out_path = os.path.join(data_dir,i,'output.mp4')

        start, end = load_start_end_frame_num(config_path)
        gr_duration, _, _ = load_skeleton(skeletons)  # 获取目标节点的动作时长
        audio = AudioFileClip(audio)  # 这里可以debug一下，看一下参数，返回值
        sub = audio.subclip(start / fps, end / fps)
        print('Analyzed the audio, found a period of %.02f seconds' % sub.duration)
        video = VideoFileClip(video_path, audio=False)
        video = video.set_audio(sub)
        video.write_videofile(out_path)
        pass