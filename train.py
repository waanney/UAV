from ultralytics import YOLOWorld,YOLO

import torch
torch.use_deterministic_algorithms(False)

model = YOLOWorld("/home/linianxin/aaai2026/ultralytics/cfg/models/v8/yolov8x-worldv2.yaml")
model.load('/home/linianxin/aaai2026/models/yolov8x-worldv2.pt')
train_results = model.train(
    data="/home/linianxin/aaai2026/ultralytics/data/VisDrone.yaml",  # path to dataset YAML
    epochs=75,  # number of training epochs
    imgsz=1536,  # training image size
    device=[0,1],
    batch=8
)
