from rsinet import SuperResolver
import cv2

img = cv2.imread("/workspaces/prototype/demo.jpg", 1)

super_resolver = SuperResolver(
    "/workspaces/prototype/deployment_files/sr/model.onnx"
)

output = super_resolver.run(img)