import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('C:/Users/CAU/Desktop/Fifth/ultralytics-20240323_ceshi/ultralytics-main/runs/prune/yolov8-BIFPN-EfficientRepHead2-finetune4/weights/best.pt') # select your model.pt path
    model.predict(source='C:/Users/CAU/Desktop/Fifth/ultralytics-20240323_ceshi/ultralytics-main/se_t/10C/3',
                  imgsz=640,
                  project='runs/detect',
                  name='exp',
                  save=True,
                  line_width = 2,
                  # conf=0.2,
                  # visualize=True # visualize model features maps
                )