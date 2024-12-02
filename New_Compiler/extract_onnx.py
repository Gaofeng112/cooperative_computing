import extract_onnx_lib
import torch
import onnx
import re
print("python executed")
# extract_onnx_lib.split_onnx('npuCutInstruction.txt','npu')
# extract_onnx_lib.split_onnx('cpuCutInstruction.txt','cpu')
#extract_onnx_lib.split_onnx('cpuCutInstructionlast.txt','cpu2')

#extract_onnx_lib.split_onnx_ios('subgraphs_ios.txt','net/unet_32_sim_v2.onnx')
extract_onnx_lib.split_onnx_ios('subgraphs_ios.txt','net/vision_model_simplify.onnx')
#extract_onnx_lib.sort('instr_original.txt','instr_modified.txt')
# model = onnx.load('net/generation_model_simplify1.onnx')
# onnx.checker.check_model('net/generation_model_simplify1.onnx')
# for output in model.graph.output:
#     model.graph.value_info.append(output)
# onnx.save(model, 'net/generation_model_simplify1.onnx')
# input_path='net/generation_model_simplify1.onnx'
# output_path='test_subgraph'+'.onnx'
# input_names=['onnx::MatMul_1080','/language_model/model/decoder/layers.0/self_attn/Add_output_0']
# output_names=['/language_model/model/decoder/layers.0/self_attn/MatMul_1_output_0']
# # input_names=['/language_model/model/decoder/layers.3/encoder_attn/Reshape_1_output_0']
# # output_names=['onnx::MatMul_1688']
# onnx.utils.extract_model(input_path, output_path, input_names, output_names)

