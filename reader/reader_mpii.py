import numpy as np
import cv2 
import os
from torch.utils.data import Dataset, DataLoader
import torch
import json

def gazeto2d(gaze):
  yaw = np.arctan2(-gaze[0], -gaze[2])
  pitch = np.arcsin(-gaze[1])
  return np.array([yaw, pitch])

class loader(Dataset): 
  def __init__(self, path, root, header=True): # 軽量化のため、ここでlabelの座標変換を行う必要あり！！
    self.lines = []
    if isinstance(path, list):
      for i in path:
        with open(i) as f:
          line = f.readlines()
          if header: line.pop(0)
          self.lines.extend(line)
    else:
      with open(path) as f:
        self.lines = f.readlines()
        if header: self.lines.pop(0)

    self.root = root
    
    
    self.tvecs = {}
    with open('/home/islabshi/workspace-cloud/koki.murata/datasets/processed/MPII_2d/params.json', 'r') as f:
      data = json.load(f)
      for k in data:
        _tvec = data[k]['tvec']
        _tvec.pop(1)
        _tvec = np.array(_tvec).flatten()
        _tvec = _tvec*np.array([-1,1])   # mpiifacegazeが右上原点のlabelのため、反転
        self.tvecs[k] = _tvec
      

  def __len__(self):
    return len(self.lines)

  def __getitem__(self, idx):
    line = self.lines[idx]
    line = line.strip().split(" ")

    name = line[4]
    point = line[6]
    ratio = line[9]

    face = line[0]
    lefteye = line[1]
    righteye = line[2]
    grid = line[3]
    
    
    width = 1270
    height = 720
    # (左上x,y,右下x,y)
    fc = list(map(int, line[10].split(',')))
    lc = list(map(int, line[11].split(',')))
    rc = list(map(int, line[12].split(',')))


  
    ### ここでカメラ原点となるように変更 →ここでやると時間かかるのでだめ。→ここでやる。いうてそんな時間かからんかも。
    # 座標系は、カメラ中心が原点。被験者から見てxは左に正、yは下に正、zは向かってくる向きに正(スクリーン上は全て0とおく)
    # パーソンを判別("pxx"のstr型)
    person = face.split('/')[0]
    
    
    label = np.array(point.split(",")).astype("float")
    ratio = np.array(ratio.split(",")).astype("float")
    label = label*ratio
    label = label + self.tvecs[person]
    label = label*0.01
    # label = torch.from_numpy(label).type(torch.FloatTensor)

    rimg = cv2.imread(os.path.join(self.root, righteye))
    rimg = cv2.resize(rimg, (112, 112))
    rimg = cv2.cvtColor(rimg, cv2.COLOR_BGR2RGB)
    rimg = rimg / 255
    rimg = rimg.transpose(2, 0, 1)

    limg = cv2.imread(os.path.join(self.root, lefteye))
    limg = cv2.resize(limg, (112, 112))
    limg = cv2.cvtColor(limg, cv2.COLOR_BGR2RGB)
    limg = limg / 255
    limg = limg.transpose(2, 0, 1)
    
    fimg = cv2.imread(os.path.join(self.root, face))
    fimg = cv2.resize(fimg, (224, 224))
    fimg = cv2.cvtColor(fimg, cv2.COLOR_BGR2RGB)
    fimg = fimg / 255
    fimg = fimg.transpose(2, 0, 1)
 
   
    
    rects = [
        (fc[1]-fc[0])/width, (fc[3]-fc[2])/height, fc[0]/width, fc[1]/height,
        (lc[1]-lc[0])/width, (lc[3]-lc[2])/height, lc[0]/width, lc[1]/height,
        (rc[1]-rc[0])/width, (rc[3]-rc[2])/height, rc[0]/width, rc[1]/height
    ]

    return {"faceImg": torch.from_numpy(fimg).type(torch.FloatTensor), 
                "leftEyeImg": torch.from_numpy(limg).type(torch.FloatTensor),
                "rightEyeImg": torch.from_numpy(rimg).type(torch.FloatTensor),
                "rects": torch.from_numpy(np.array(rects)).type(torch.FloatTensor),
                "label": torch.from_numpy(label).type(torch.FloatTensor), 
                "ratio": torch.from_numpy(np.array(ratio)).type(torch.FloatTensor), "frame": line}

def txtload(labelpath, imagepath, batch_size, shuffle=True, num_workers=0, header=True):
  dataset = loader(labelpath, imagepath, header)
  print(f"[Read Data]: {labelpath}")
  print(f"[Read Data]: Total num: {len(dataset)}")
  load = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
  return load


if __name__ == "__main__":
  label = "/home/islabshi/workspace-cloud/koki.murata/datasets/processed/MPII_2d/Label"
  image = "/home/islabshi/workspace-cloud/koki.murata/datasets/processed/MPII_2d/Image"
  trains = os.listdir(label)
  trains = [os.path.join(label, j) for j in trains]
  d = txtload(trains, image, 10)
  print(len(d))
  (data, label) = d.__iter__()
