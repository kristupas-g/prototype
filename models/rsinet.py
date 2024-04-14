import onnxruntime as ort
import numpy as np
import cv2

class SuperResolver:
    def __init__(self, model_path, device='cpu'):
        self.model_path = model_path
        self.device = device

        self.ort_session = ort.InferenceSession(self.model_path)

    def run(self, img):
        input = self.__preprocess_input(img)

        inputs = { self.ort_session.get_inputs()[0].name: input }
        print(f"Running model with input shape {input.shape}")
        output = self.ort_session.run(None, inputs)[0]

        return self.__preprocess_output(output)

    def __preprocess_input(self, input):
        input = np.array(input).astype(np.float32)
        input = self.__ensure_bgr(input)
        input = input.transpose((2, 0, 1))
        input = np.expand_dims(input, axis = 0)
        return input

    def __preprocess_output(self, output):
        output = np.clip(output[0], 0, 255).transpose((1, 2, 0)).astype(np.uint8)
        return cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

    def __ensure_bgr(self, image):
        if len(image.shape) == 3 and image.shape[2] == 3:
            if np.all(image[:, :, 0] == image[:, :, 1]) == False:
                return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image
