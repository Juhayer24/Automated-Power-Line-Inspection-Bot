# src/detectors/yolo.py
import numpy as np
import torch

# try ultralytics first (yolov8)
try:
    from ultralytics import YOLO
    ULTRALYTICS = True
except Exception:
    ULTRALYTICS = False

class YOLODetector:
    def __init__(self, weights='yolov5s', conf_thresh=0.25, device=None):
        self.conf_thresh = conf_thresh
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device

        if ULTRALYTICS:
            # weights can be 'yolov8n.pt' or a path under models/yolo/
            self.model = YOLO(weights)
            # ultralytics auto-selects device
            self.backend = 'ultralytics'
        else:
            # fallback: torch.hub yolov5
            self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
            self.model.conf = conf_thresh
            self.backend = 'yolov5_hub'

    def predict(self, frame):
        """
        frame: BGR numpy array
        returns: list of dicts {'bbox':(x1,y1,x2,y2), 'score':float, 'class':int}
        """
        boxes = []
        if self.backend == 'ultralytics':
            results = self.model(frame, conf=self.conf_thresh)  # returns a Results list
            r = results[0]
            # r.boxes exists in ultralytics Results; each box has xyxy, conf, cls
            if hasattr(r, 'boxes') and r.boxes is not None:
                for box in r.boxes:
                    xyxy = box.xyxy[0].cpu().numpy() if hasattr(box.xyxy, 'cpu') else np.array(box.xyxy[0])
                    conf = float(box.conf[0]) if hasattr(box, 'conf') else None
                    cls = int(box.cls[0]) if hasattr(box, 'cls') else None
                    boxes.append({'bbox': tuple(map(int, xyxy)), 'score': conf, 'class': cls})
        else:
            results = self.model(frame)
            # results.xyxy[0] is a tensor Nx6 (x1,y1,x2,y2,conf,class)
            det = results.xyxy[0].cpu().numpy()
            for x1,y1,x2,y2,conf,cls in det:
                if conf < self.conf_thresh:
                    continue
                boxes.append({'bbox': (int(x1), int(y1), int(x2), int(y2)), 'score': float(conf), 'class': int(cls)})
        return boxes
