import extract_onnx_lib
import torch
import onnx
import re
print("python executed")
# extract_onnx_lib.split_onnx('npuCutInstruction.txt','npu')
# extract_onnx_lib.split_onnx('cpuCutInstruction.txt','cpu')
#extract_onnx_lib.split_onnx('cpuCutInstructionlast.txt','cpu2')

extract_onnx_lib.split_onnx_ios('subgraphs_ios.txt')
#extract_onnx_lib.sort('instr_original.txt','instr_modified.txt')

# onnx.checker.check_model('net/unet_32_sim_v2.onnx')
# input_path='net/unet_32_sim_v2.onnx'
# output_path='subgraphs_1009'+'/'+'subgraph45'+'.onnx'
# input_names=['/down_blocks.0/attentions.0/transformer_blocks.0/norm2/Sub_output_0']
# output_names=['/down_blocks.0/attentions.0/transformer_blocks.0/norm2/Sqrt_output_0']
# onnx.utils.extract_model(input_path, output_path, input_names, output_names)

