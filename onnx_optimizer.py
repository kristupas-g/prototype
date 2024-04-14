import os
import shutil
import logging
import onnx
from onnxoptimizer import optimize

class ONNXOptimizer:
    def __init__(self, source_directory, destination_directory):
        self.source_directory = source_directory
        self.destination_directory = destination_directory
        self.copy_and_create_structure()

    def copy_and_create_structure(self):
        if os.path.exists(self.destination_directory):
            shutil.rmtree(self.destination_directory)
        shutil.copytree(self.source_directory, self.destination_directory)

    def optimize_model(self, model_path, optimized_model_path):
        logging.info(f"Optimizing model: {model_path}")
        model = onnx.load(model_path)
        optimized_model = optimize(model)
        onnx.save(optimized_model, optimized_model_path)

    @classmethod
    def optimize(cls, source_dir):
        dest_dir = f"{source_dir}_optimized"
        if os.path.exists(dest_dir):
            return

        self = cls(source_dir, dest_dir)

        for root, _, files in os.walk(self.destination_directory):
            for file in files:
                if file.endswith('.onnx'):
                    model_path = os.path.join(root, file)
                    self.optimize_model(model_path, model_path)

