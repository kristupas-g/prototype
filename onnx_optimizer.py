import os
import onnx
from onnxoptimizer import optimize

class ONNXOptimizer:
    def __init__(self, directory):
        self.directory = directory

    def optimize_model(self, model_path, optimized_model_path):
        print(f"Optimizing model: {model_path}")

        model = onnx.load(model_path)
        optimized_model = optimize(model)
        onnx.save(optimized_model, optimized_model_path)

    @classmethod
    def process_directory(cls, directory):
        self = cls(directory)

        for root, _, files in os.walk(self.directory):
            for file in files:
                if file.endswith('.onnx') and not file.endswith('_optimized.onnx'):
                    original_model_path = os.path.join(root, file)
                    optimized_model_path = os.path.join(root, file[:-5] + '_optimized.onnx')
                    print(original_model_path)
                    print(optimized_model_path)
                    if not os.path.exists(optimized_model_path):
                        # self.optimize_model(original_model_path, optimized_model_path)

# Usage:
# optimizer = ONNXOptimizer('/path/to/directory')
# optimizer.process_directory()
