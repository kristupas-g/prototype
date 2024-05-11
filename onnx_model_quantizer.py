model_path = '/workspaces/prototype/deployment_files/2x_original.onnx'
import onnx
from onnxconverter_common import float16

model = onnx.load(model_path)
model_fp16 = float16.convert_float_to_float16(model)
onnx.save(model_fp16, "/workspaces/prototype/deployment_files/sr/2x.onnx")
