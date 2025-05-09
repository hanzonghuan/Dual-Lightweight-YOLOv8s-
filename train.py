import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('C:/Users/CAU/Desktop/ultralytics-20240323/ultralytics-main/ultralytics/cfg/models/v8/yolov8-BIFPN-EfficientRepHead.yaml')
    # model.load('yolov8n.pt') # loading pretrain weights
    model.train(data='C:/Users/CAU/Desktop/ultralytics-20240323/ultralytics-main/dataset2/data.yaml',
                cache=False,
                imgsz=640,
                epochs=150,
                batch=32,
                close_mosaic=10,
                workers=8,
                device='0',
                optimizer='Adam', # using SGD
                # resume='', # last.pt path
                # amp=False, # close amp
                # fraction=0.2,
                project='runs/train',
                name='exp',
                )