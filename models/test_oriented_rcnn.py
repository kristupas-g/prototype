from oriented_rcnn import Detector
import cv2

img = cv2.imread("/workspaces/prototype/demo.jpg")

detector = Detector("/workspaces/prototype/deployment_files/2x_75_onnx")

output = detector.run(img)