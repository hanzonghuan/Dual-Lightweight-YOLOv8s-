import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':

    model = YOLO("C:/Users/CAU/Desktop/Fifth/ultralytics-20240323_ceshi/ultralytics-main/runs/train/exp10/weights/best.pt")

    model.val(data='dataset2/data.yaml',
              split='test',
              imgsz=640,
              batch=32,
              # rect=False,
              # save_json=True, # if you need to cal coco metrice
              project='runs/val',
              name='exp',
              )