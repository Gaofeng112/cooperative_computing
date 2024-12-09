
import onnx
import re
print("python executed")
# extract_onnx_lib.split_onnx('npuCutInstruction.txt','npu')
# extract_onnx_lib.split_onnx('cpuCutInstruction.txt','cpu')
#extract_onnx_lib.split_onnx('cpuCutInstructionlast.txt','cpu2')

# extract_onnx_lib.split_onnx_ios('subgraphs_ios.txt')


# input_path='diffusion_model_cpu_89/cpu89_0.onnx'
# output_path='diffusion_model_cpu_89/cpu89_0_matmul.onnx'
# input_names=['StatefulPartitionedCall/model/bk_tiny_lcmu_net_model/spatial_transformer_7/basic_transformer_block_7/cross_attention_15/dense_88/Tensordot/Reshape:0']
# output_names=['StatefulPartitionedCall/model/bk_tiny_lcmu_net_model/spatial_transformer_7/basic_transformer_block_7/cross_attention_15/dense_88/Tensordot/MatMul:0']
# onnx.utils.extract_model(input_path, output_path, input_names, output_names)

# input_path='diffusion_model_cpu_89/cpu89_0.onnx'
# output_path='diffusion_model_cpu_89/cpu89_0_reshape.onnx'
# input_names=['StatefulPartitionedCall/model/bk_tiny_lcmu_net_model/spatial_transformer_7/basic_transformer_block_7/cross_attention_15/dense_88/Tensordot/MatMul:0']
# output_names=['StatefulPartitionedCall/model/bk_tiny_lcmu_net_model/spatial_transformer_7/basic_transformer_block_7/cross_attention_15/dense_88/Tensordot:0']
# onnx.utils.extract_model(input_path, output_path, input_names, output_names)

# input_path='diffusion_model_cpu_89/cpu89_0.onnx'
# output_path='diffusion_model_cpu_89/cpu89_0_add.onnx'
# input_names=['StatefulPartitionedCall/model/bk_tiny_lcmu_net_model/spatial_transformer_7/basic_transformer_block_7/cross_attention_15/dense_88/Tensordot:0']
# output_names=['StatefulPartitionedCall/model/bk_tiny_lcmu_net_model/spatial_transformer_7/basic_transformer_block_7/cross_attention_15/dense_88/BiasAdd:0']
# onnx.utils.extract_model(input_path, output_path, input_names, output_names)
input_path='../../unet_32_sim_v2.onnx'
output_path='../lib_onnx_10.11/npusubgraph0.onnx'
input_names=['sample']
output_names=['/conv_in/Conv_output_0']
onnx.utils.extract_model(input_path, output_path, input_names, output_names)

input_path='../../unet_32_sim_v2.onnx'
output_path='../lib_onnx_10.11/cpusubgraph1.onnx'
input_names=['/conv_in/Conv_output_0','timestep','encoder_hidden_states']
output_names=['/up_blocks.2/attentions.1/proj_out/Conv_output_0','/up_blocks.2/resnets.1/Add_1_output_0']
onnx.utils.extract_model(input_path, output_path, input_names, output_names)

input_path='../../unet_32_sim_v2.onnx'
output_path='../lib_onnx_10.11/cpusubgraph2.onnx'
input_names=['/up_blocks.2/attentions.1/proj_out/Conv_output_0','/up_blocks.2/resnets.1/Add_1_output_0']
output_names=['3089']
onnx.utils.extract_model(input_path, output_path, input_names, output_names)


