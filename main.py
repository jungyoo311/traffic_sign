import os
from ultralytics import YOLO
import torch
import matplotlib.pyplot as pyplot
import numpy as numpy
import cv2

model_path = os.path.join('.', 'runs', 'detect', 'train', 'weights', 'last.pt')
model = YOLO(model_path)
img = './data/images/00034.jpg'
results = model(img, show = True, verbose = False, conf = 0.2)
cv2.waitKey(5000)