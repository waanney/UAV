import os
from ultralytics import YOLO
import cv2

def detect_images_in_folder(model_path, input_folder, output_folder):
    """
    使用 YOLOv8 模型对文件夹中的所有图片进行目标检测，并保存结果。

    Args:
        model_path (str): 训练好的 YOLOv8 模型路径（.pt 文件）
        input_folder (str): 包含待检测图片的文件夹路径
        output_folder (str): 保存检测结果的文件夹路径
    """
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 加载 YOLOv8 模型
    model = YOLO(model_path)

    # 支持的图片格式
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')

    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(valid_extensions):
            # 构建图片完整路径
            image_path = os.path.join(input_folder, filename)
            
            # 进行目标检测
            results = model(image_path, imgsz=640, conf=0.25, iou=0.45)

            # 获取检测结果的图像（带边界框）
            annotated_image = results[0].plot()  # Ultralytics 提供绘图功能

            # 构建输出路径
            output_path = os.path.join(output_folder, f"detected_{filename}")

            # 保存检测结果
            cv2.imwrite(output_path, annotated_image)
            print(f"已处理并保存：{output_path}")

if __name__ == '__main__':
    # 配置路径
    model_path = "/home/cvlab1003/nianxin/aaai/runs/detect/65.7/weights/best.pt"  # 你的模型路径
    input_folder = "/home/cvlab1003/nianxin/aaai/测试"  # 替换为你的图片文件夹路径
    output_folder = "/home/cvlab1003/nianxin/aaai/result1"  # 替换为输出文件夹路径

    # 执行检测
    detect_images_in_folder(model_path, input_folder, output_folder)