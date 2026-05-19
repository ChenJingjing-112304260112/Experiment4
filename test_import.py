# -*- coding: utf-8 -*-
print("Step 1: Before import")
from ultralytics import YOLO
print("Step 2: After import")
model = YOLO("yolov8n.pt")
print("Step 3: After model load")