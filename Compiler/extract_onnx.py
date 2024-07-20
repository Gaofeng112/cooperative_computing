import extract_onnx_lib
# import torch
# import onnx
# import re
print("python executed")
extract_onnx_lib.split_onnx('npuCutInstruction.txt','npu')
extract_onnx_lib.split_onnx('cpuCutInstruction.txt','cpu')
