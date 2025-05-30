# ReadMe

https://github.com/vigil1917/AFF-Net?tab=readme-ov-file

Codes for ICPR2020 paper [Adaptive Feature Fusion Network for Gaze Tracking in Mobile Tablets](https://ieeexplore.ieee.org/abstract/document/9412205/).


4/26
### 学習
自動でleave-one-person-out学習を行う。
```
bash run.sh train.py config.yaml 
```

### テスト
自動でleave-one-person-out学習を行う。
```
bash run.sh test.py config.yaml 
```


savesフォルダにcheckpointとテスト結果あり。




readerで得られるデータ
rects
顔width
顔height
顔右上x
顔右上y
左目width
・・・

output



The project is based on:

```
numpy==1.20.1
opencv-python==4.2.0.34
torch==1.5.0
torchvision==0.6.0
```

As we conduct experiments on re-face-detected [GazeCapture](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Krafka_Eye_Tracking_for_CVPR_2016_paper.html) dataset, the data-loader codes are not universal with original dataset. For details and codes about data processings, SOTA methods and benchmark, please refer to our survey paper ["Appearance-based Gaze Estimation With Deep Learning: A Review and Benchmark"](http://phi-ai.buaa.edu.cn/Gazehub/2D-dataset/).
