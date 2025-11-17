import torch
from PIL import Image
import numpy as np
import cv2
from pathlib import Path

from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes, cv2

weights = 'yolov5s.pt'
device = 'cpu'  


model = DetectMultiBackend(weights, device=device, dnn=False)
stride, names, pt = model.stride, model.names, model.pt

img = cv2.imread("data/images/bus.jpg")
assert img is not None, "이미지 없음"

img0 = img.copy()
img = cv2.resize(img, (640, 640))
img = img[:, :, ::-1].transpose(2,0,1)    # BGR → RGB → CHW
img = np.ascontiguousarray(img)

img = torch.from_numpy(img).float()
img /= 255.0
img = img.unsqueeze(0)

pred = model(img)
pred = non_max_suppression(pred, 0.25, 0.45)

print("Pred:", pred)
