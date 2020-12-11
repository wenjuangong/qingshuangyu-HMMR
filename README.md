# A Deep Music Recommendation Method based on Human Motion Analysis
This work proposes a deep music recommendation algorithm based on dance motion analysis and a LSTM-AE based method which learns the correspondences between motion and music.
## Dependencies
1.pytorch\
2.librosa\
3.json\
4.numpy\
5.ffmpeg\
6.panda\
7.skelearn\
8.CUDA
## Training & Testing

### LSTM-AE
`cd DWM/run/`\
change the 'is_train' depending on what you want\
`python train.py`
### Motion-Analysis
`cd hmmr/`\
change model.agcn.__init__() in_channels=3\
`python main.py --config ./config/segment_frames/train_joint.yaml`
`python main.py --config ./config/segment_frames/train_bone.yaml`
change model.agcn.__init__() in_channels=6\
`python main.py --config ./config/segment_frames/train_data_sum.yaml`\
test and show accuracy\
change model.agcn.__init__() in_channels=3\
`python main.py --config ./config/segment_frames/test_joint.yaml`
`python main.py --config ./config/segment_frames/test_bone.yaml`
change model.agcn.__init__() in_channels=6\
`python main.py --config ./config/segment_frames/test_data_sum.yaml`\
`python ensemble.py`
## Show result
### LSTM-AE
`cd DWM/run/`\
`python classify.py`\
`python classify_single.py`
### Motion-Analysis
`cd hmmr/`\
`python confusion_matrix.py`
