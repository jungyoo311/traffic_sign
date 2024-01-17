import os
from ultralytics import YOLO
import torch
import matplotlib.pyplot as pyplot
import numpy as numpy
import cv2

model_path = os.path.join('.', 'runs', 'detect', 'train2', 'weights', 'last.pt')
model = YOLO(model_path)
img = './data/images/00034.jpg'
video_ref = './driving.mp4'
results = model(video_ref, show = True, verbose = True, conf = 0.8)
cv2.waitKey(0.01)