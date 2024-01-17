import os
from ultralytics import YOLO
import torch
import matplotlib.pyplot as pyplot
import numpy as numpy
import cv2

model = YOLO('yolov8s.pt')
model.train(data = "dataset.yaml", epochs = 1)