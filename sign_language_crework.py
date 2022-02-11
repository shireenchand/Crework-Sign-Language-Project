import os
import shutil
import cv2 
import random
import json
import pickle
from torchvision import transforms
from test_i3d import *
from vid import split_by_seconds,get_video_length

os.chdir('/content/Crework-Sign-Language-Project')
vid_length = get_video_length('/content/Crework-Sign-Language-Project/videos/vid.mp4')
split_by_seconds(filename='/content/Crework-Sign-Language-Project/videos/vid.mp4',split_length=3,vcodec='h264',video_length=vid_length)

mode = 'rgb'
num_classes = 2000
save_model = './checkpoints/'

 
## Change to where the videos are located
root = {'word':'videos'}

train_split = 'preprocess/nslt_2000.json'

weights = '/content/Crework-Sign-Language-Project/checkpoints/nslt_2000_065846_0.447803.pt'

shutil.move('/content/Crework-Sign-Language-Project/videos/vid.mp4','/content/vid.mp4')
os.remove('/content/Crework-Sign-Language-Project/videos/t')


complete_dict = dict()
os.chdir('/content/Crework-Sign-Language-Project/videos')
for video in os.listdir():
  if video == ".ipynb_checkpoints":
    continue
  id = video.split('.')[0]
  capture = cv2.VideoCapture(video)
  frameNr = 0 
  while (True):
      success, frame = capture.read() 
      if success:
       frameNr = frameNr+1
      else:
        break 
  
  dictionary = {
      str(id):{
          "subset":"test",
          "action":[random.randint(0,2000),1,frameNr]
      }
  }
  complete_dict.update(dictionary)
json_object = json.dumps(complete_dict, indent = 4)
with open("/content/sample.json", "w") as outfile:
    outfile.write(json_object)
capture.release()


os.chdir('/content/Crework-Sign-Language-Project')
import videotransforms
import numpy as np
from datasets.nslt_dataset import NSLT as Dataset

test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])
dataset = Dataset("/content/sample.json", 'test', root, mode, test_transforms)

datasets = {'test': dataset}
with open('datasets.pkl', 'wb') as f:
    pickle.dump(datasets, f)

with open('datasets.pkl', 'rb') as f:
   datasets = pickle.load(f)

run(mode=mode, root=root, train_split="/content/sample.json", weights=weights, datasets=datasets, num_classes=num_classes)


with open('/content/Crework-Sign-Language-Project/predictions.txt') as f:
  pred = f.readlines()
f.close()

words = [i.split('\n')[0] for i in pred ]

with open('/content/Crework-Sign-Language-Project/preprocess/wlasl_class_list.txt') as f:
      labels = f.readlines()

map = {}
for element in labels:
  first = element.split('\t')
  second = first[1].split('\n')[0]
  map[first[0]] = second


text_words = [map[num] for num in words]
print(text_words)


































