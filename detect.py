import os
from ultralytics import YOLO
import cv2
import numpy as np

from conformal import correctness_prediction_set, correctness_priority, expanded_box, load_calibrator


def annotate_result(result, calibration=None):
    """Draw raw YOLO boxes and optional CP localization/correctness outputs."""
    if calibration and result.boxes is not None and len(result.boxes):
        confidences = result.boxes.conf.cpu().numpy()
        priorities = np.asarray(
            [correctness_priority(conf, float(calibration["qhat_correctness"])) for conf in confidences]
        )
        order = np.argsort(-priorities, kind="stable")
        result.boxes.data = result.boxes.data[order.tolist()]
    annotated_image = result.plot()
    if calibration and result.boxes is not None:
        qhat = float(calibration["qhat"])
        qhat_correctness = float(calibration["qhat_correctness"])
        image_height, image_width = annotated_image.shape[:2]
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        for box, confidence in zip(boxes, confidences):
            x1, y1, x2, y2 = expanded_box(box, qhat)
            x1 = max(0, min(image_width - 1, int(round(x1))))
            y1 = max(0, min(image_height - 1, int(round(y1))))
            x2 = max(0, min(image_width - 1, int(round(x2))))
            y2 = max(0, min(image_height - 1, int(round(y2))))
            candidate_labels = correctness_prediction_set(confidence, qhat_correctness)
            if candidate_labels == {"correct"}:
                label, color = "CP: correct", (0, 200, 0)
            elif candidate_labels == {"incorrect"}:
                label, color = "CP: incorrect", (0, 0, 255)
            elif candidate_labels == {"correct", "incorrect"}:
                label, color = "CP: uncertain", (0, 165, 255)
            else:
                label, color = "CP: empty set", (255, 255, 255)
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                annotated_image,
                label,
                (x1, max(y1 - 6, 14)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
    return annotated_image


def detect_images_in_folder(model_path, input_folder, output_folder, calibration_path=None):
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

    # Load the model and optional split-conformal calibrator.
    model = YOLO(model_path)
    calibration = load_calibrator(calibration_path) if calibration_path else None

    # 支持的图片格式
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')

    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(valid_extensions):
            # 构建图片完整路径
            image_path = os.path.join(input_folder, filename)
            
            # Use the exact candidate-generation settings used to calibrate qhat.
            if calibration:
                results = model(
                    image_path,
                    imgsz=int(calibration["imgsz"]),
                    conf=float(calibration["predict_conf"]),
                    iou=float(calibration["predict_iou"]),
                )
            else:
                results = model(image_path, imgsz=640, conf=0.25, iou=0.45)

            annotated_image = annotate_result(results[0], calibration)

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
    calibration_path = None  # Set to the JSON produced by calibrate_conformal.py to draw CP regions.

    # 执行检测
    detect_images_in_folder(model_path, input_folder, output_folder, calibration_path)
