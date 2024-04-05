from mmdeploy_runtime import RotatedDetector

class Detector:
    def __init__(self, model_path, device='cpu'):
        self.model_path = model_path
        self.device = device

        self.detector = RotatedDetector(
            model_path = self.model_path,
            device_name = self.device,
            device_id = 0
        )

    def run(self, img):
        return self.detector(img)