import io
import time
import logging
from models.oriented_rcnn import Detector
from models.rsinet import SuperResolver

import numpy as np
from PIL import Image

class ShipDetector:
    def __init__(self, image, sr_scale=1, optimized=False):
        self.onnx_file_subdir = "/workspaces/prototype/deployment_files_optimized" if optimized else "/workspaces/prototype/deployment_files"
        self.logger = logging.getLogger(__name__)
        self.detector = self.__build_detector()

        self.image = image.copy()
        self.image = image.convert("RGB")

        self.scale = sr_scale

    @classmethod
    def run(cls, image, sr_scale=1, optimized=False):
        start_time = time.time()
        self = cls(image, sr_scale, optimized)
        self.logger.info("Initialized ShipDetector")

        image = self.__get_resolved_image()

        image_with_dets = self.__perform_detection(image)

        self.logger.info(f"Total processing time: {time.time() - start_time:.2f} seconds")
        return image_with_dets

    def __get_resolved_image(self):
        if self.scale < 2:
            return self.image

        super_resolver = SuperResolver(
            model_path=f"{self.onnx_file_subdir}/sr/{self.scale}x.onnx"
        )
        start_time = time.time()
        resolved_image = super_resolver.run(self.image)
        self.logger.info(f"Super-resolution completed in {time.time() - start_time:.2f} seconds")
        
        return resolved_image

    def __build_detector(self):
        return Detector(model_path=f"{self.onnx_file_subdir}/2x_75_onnx")

    def __perform_detection(self, image):
        start_time = time.time()
        detector = self.__build_detector()
        
        bboxes, classes = detector.run(image)
        img_with_results = detector.visualize_results(image, bboxes, classes)
        self.logger.info(f"Object detection completed in {time.time() - start_time:.2f} seconds")
        
        return img_with_results
