from oriented_rcnn import Detector
import cv2

img = cv2.imread("/workspaces/prototype/test/demo.jpg")

detector = Detector("/workspaces/prototype/deployment_files/oriented_rcnn")

output = detector.run(img)