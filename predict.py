
from ultralytics import YOLO,YOLOWorld
import cv2

# 加载模型
model = YOLOWorld("/home/cvlab1003/nianxin/aaai/best.pt")  # 可换成 yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt

# 读取图像
image_path = "/home/cvlab1003/nianxin/aaai/name3.png"
img = cv2.imread(image_path)

# 检测图像
results = model(img)

# 显示检测结果
results[0].show()

# 或保存检测结果图像
results[0].save(filename="output1.png")
