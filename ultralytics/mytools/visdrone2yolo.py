'''
python ultralytics/ultralytics/visdrone2yolo.py

yolo task=detect mode=train model=ultralytics/ultralytics/cfg/models/v8/yolov8x-world.yaml  pretrained=yolov8x.pt data=datasets/VisDrone.yaml batch=2 epochs=100 imgsz=1536 workers=0 device=0
yolo task=detect mode=train resume model=runs/detect/train19/weights/last.pt data=datasets/VisDrone.yaml batch=2 epochs=60 imgsz=1536 workers=0 device=0 lr0=0.0002
yolo task=detect mode=val model=runs/detect/63.6/weights/best.pt data=datasets/VisDrone.yaml batch=16 epochs=100 imgsz=1536 workers=0 device=0
https://docs.ultralytics.com/zh/models/yolo-world/#key-features
CUDA_VISIBLE_DEVICES=0,1 yolo detect train --data data/coco128.yaml --cfg models/yolov8.yaml --weights '' --epochs 100 --batch-size 32 --device 0,1

yolo task=detect mode=val model=runs/detect/63.6/weights/best.pt data=datasets/VisDrone.yaml batch=16 epochs=100 imgsz=1536 workers=0 device=0,1
yolo task=detect mode=val model=runs/detect/64/weights/best.pt data=datasets/VisDrone.yaml batch=16 imgsz=1536 workers=0 device=0,1

https://blog.csdn.net/bajinzhixing/article/details/134842462

yolo task=detect mode=train model=ultralytics/ultralytics/cfg/models/v8/yolov8s-worldv2.yaml pretrained=yolov8x-worldv2.pt data=datasets/VisDrone.yaml batch=4 epochs=60 lr0=0.0002 imgsz=1536 optimizer=AdamW weight_decay=0.00036 cos_lr=True device=0,1
yolo task=detect mode=train model=ultralytics/ultralytics/cfg/models/v8/yolov8x-relation-world.yaml pretrained=runs/detect/64/weights/best.pt  data=datasets/VisDrone.yaml batch=2 epochs=60 lr0=0.0002 imgsz=1536 optimizer=AdamW weight_decay=0.00036 cos_lr=True device=0
yolo task=detect mode=train model=yolov8s.pt  data=datasets/VisDrone.yaml batch=8 epochs=60 lr0=0.0002 imgsz=1536 optimizer=AdamW weight_decay=0.00036 cos_lr=True device=0,1
yolo task=detect mode=train model=ultralytics/ultralytics/cfg/models/v8/yolov8x-relation-world.yaml pretrained=runs/detect/train91/weights/last.pt  data=datasets/VisDrone.yaml batch=2 epochs=60 lr0=0.0002 imgsz=1536 optimizer=AdamW weight_decay=0.00036 cos_lr=True device=0
yolo task=detect mode=train model=ultralytics/ultralytics/cfg/models/v8/yolov8s-worldv2.yaml data=datasets/VisDrone.yaml batch=4 epochs=60 lr0=0.0002 imgsz=1536 optimizer=AdamW weight_decay=0.00036 cos_lr=True device=0,1

'''

import os
from pathlib import Path
def visdrone2yolo(dir):
    from PIL import Image
    from tqdm import tqdm

    def convert_box(size, box):
        # Convert VisDrone box to YOLO xywh box
        dw = 1. / size[0]
        dh = 1. / size[1]
        return (box[0] + box[2] / 2) * dw, (box[1] + box[3] / 2) * dh, box[2] * dw, box[3] * dh

    (dir / 'labels').mkdir(parents=True, exist_ok=True)  # make labels directory
    pbar = tqdm((dir / 'annotations').glob('*.txt'), desc=f'Converting {dir}')
    for f in pbar:
        img_size = Image.open((dir / 'images' / f.name).with_suffix('.jpg')).size
        lines = []
        with open(f, 'r') as file:  # read annotation.txt
            for row in [x.split(',') for x in file.read().strip().splitlines()]:
                if row[4] == '0':  # VisDrone 'ignored regions' class 0
                    continue
                cls = int(row[5]) - 1  # 类别号-1
                box = convert_box(img_size, tuple(map(int, row[:4])))
                lines.append(f"{cls} {' '.join(f'{x:.6f}' for x in box)}\n")
                with open(str(f).replace(os.sep + 'annotations' + os.sep, os.sep + 'labels' + os.sep), 'w') as fl:
                    fl.writelines(lines)  # write label.txt
dir = Path('/home/lihua_zhou/nianxin/ultralytics/ultralytics/data/visdrone')  # datasets文件夹下Visdrone2019文件夹目录
# Convert
for d in 'VisDrone2019-DET-train', 'VisDrone2019-DET-val', 'VisDrone2019-DET-test-dev':
    visdrone2yolo(dir / d)  # convert VisDrone annotations to YOLO labels

