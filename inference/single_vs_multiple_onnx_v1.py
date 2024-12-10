''''
推理单个onnx模型和拆分成多个onnx模型的结果对比
使用封装的类,带有PCA
'''
import onnxruntime as ort  
import numpy as np        
from model_inference import * 
import os 

def compare_results(output_one, output_two):
    """
    对两个输出进行比较并计算均方误差(MSE)
    """
    for i, (one, two) in enumerate(zip(output_one, output_two)):
        mse = np.mean((np.array(one) - np.array(two))**2)
        print(f"输出的均方误差 (MSE): {mse}")

# 定义单个ONNX模型路径和拆分后的子图模型路径
single_onnx_model_path = './unet_32_sim_v2.onnx'
model_path = './1108_subgraphs'
subgraphsiostxt_path = './1108_subgraphs/subgraphs_ios.txt'
endwithconv_path='./1108_subgraphs/end_with_conv.txt'
output_dir = "./test_output"
os.makedirs(output_dir, exist_ok=True)

# 初始化ModelInference实例，准备进行推理
model_inference = ModelInference(model_path, subgraphsiostxt_path)
model_pca_inference = PcaInference(model_path, subgraphsiostxt_path, endwithconv_path, output_dir)

# 准备初始输入数据用于推理
initial_input_data = {
    "sample": np.random.rand(1, 4, 32, 32).astype(np.float32),
    "timestep": np.random.rand(1,).astype(np.float32),
    "encoder_hidden_states": np.random.rand(1, 77, 768).astype(np.float32),
}

# 使用单个ONNX模型执行推理
output_single = ModelInference.infer_single_onnx_model(single_onnx_model_path, initial_input_data)

# 使用拆分后的多个子图模型执行推理
output_multiple = model_inference.inference(initial_input_data)
print("比较单个ONNX模型与多个子图模型的推理结果...")
compare_results(output_single, output_multiple)

# 使用PCA进行推理
output_pca = model_pca_inference.inference(initial_input_data, 2)
print("比较多个子图模型与PCA推理结果...")
compare_results(output_multiple, output_pca)