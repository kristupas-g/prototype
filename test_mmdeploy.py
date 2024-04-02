from mmdeploy_runtime import RotatedDetector
import cv2
import numpy as np

img = cv2.imread('/content/dota_demo.jpg')

detector = RotatedDetector(model_path='/workspaces/prototype/ort', device_name='cpu', device_id=0)
det = detector(img)
det