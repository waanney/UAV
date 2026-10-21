from ultralytics import YOLO

# Load a model
# model = YOLO("/home/lihua_zhou/nianxin/ultralytics/runs/detect/0.05-0.45-63.5/weights/best.pt")  # load an official model
# model = YOLO("path/to/best.pt")  # load a custom model


from ultralytics import YOLOWorld
# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Load a model
# model = YOLOWorld("/home/lihua_zhou/nianxin/ultralytics/runs/detect/train7/weights/best.pt")  # load an official model

# metrics=model.val(data='/home/lihua_zhou/nianxin/ultralytics/ultralytics/data/visdrone/VisDrone.yaml',device=[4],batch=2)
# Customize validation settings
# validation_results = model.val(data="/home/lihua_zhou/nianxin/ultralytics/ultralytics/data/visdrone/VisDrone.yaml", imgsz=1536, batch=16, device="0")



#   另外一种方式
# 导入ultralytics的YOLO库
# from ultralytics import YOLO

# 加载模型
model = YOLOWorld('/home/linianxin/aaai2026/runs/detect/train(0.20.6)/weights/best.pt')  # 加载自定义的训练模型

 
if __name__ == '__main__':
    # 对模型进行验证
    metrics = model.val(data='/home/linianxin/aaai2026/ultralytics/data/VisDrone.yaml',device=[0,1],batch=8,conf=0.21)  # 调用val方法进行模型验证，不需要传入参数，数据集和设置已被模型记住

    # 输出不同的性能指标
    print("AP (mAP@0.5:0.95):", metrics.box.map)    # 输出平均精度均值（AP，Average Precision）在IoU阈值从0.5到0.95的范围内的结果
    print("AP@0.5 (mAP@0.5):", metrics.box.map50)  # 输出在IoU=0.5时的平均精度（AP50）
    print("AP@0.75 (mAP@0.75):", metrics.box.map75)  # 输出在IoU=0.75时的平均精度（AP75）
    print("APs per category (mAP@0.5:0.95 per category):", metrics.box.maps)  # 输出每个类别在IoU阈值从0.5到0.95的平均精度的列表