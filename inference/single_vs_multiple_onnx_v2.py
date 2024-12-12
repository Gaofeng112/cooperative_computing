''''
推理单个onnx模型和拆分成多个onnx模型的结果对比
带有PCA,使用了新的封装类model_inference_multiple_output
支持动态输入
'''
import onnxruntime as ort  
import numpy as np        
from model_inference_multiple_output import * 
import os 

def compare_results(output_single, output_multiple):
    """
    比较两个推理结果字典中相同名称输出的均方误差(MSE)。
    确保每个输出名称只被处理一次。
    """
    all_keys = set(output_single.keys()).union(set(output_multiple.keys()))
    for key in sorted(all_keys):
        if key in output_single and key in output_multiple:
            single_output = np.array(output_single[key])
            multiple_output = np.array(output_multiple[key])
            mse = np.mean((single_output - multiple_output) ** 2)
            print(f"Output '{key}' MSE: {mse}")
        else:
            print(f"Output '{key}' is missing in one of the result sets.")

def prepare_initial_input_data(onnx_model_path, default_input_data):
    """
    准备用于推理的初始输入数据。

    参数:
        onnx_model_path (str): ONNX 模型文件路径。
        default_input_data (dict): 包含默认输入数据的字典。

    返回:
        dict: 包含用户指定或默认形状和类型的输入数据。
    """
    # 加载单个ONNX模型以获取输入信息
    session = ort.InferenceSession(onnx_model_path)
    input_info = {input.name: input.shape for input in session.get_inputs()}

    initial_input_data = {}
    dtype_map = {'f': np.float32, 'i': np.int64}

    for input_name, shape in input_info.items():
        print(f"输入名称: {input_name}, 默认shape: {shape}")

        # 让用户输入自定义的形状或使用默认值
        custom_shape_str = input(f"请输入输入 '{input_name}' 的新形状（用逗号分隔的整数），或者按回车键使用默认值：")

        # 提示用户输入数据类型，默认采用模型定义中的类型
        print("f代表float32， i代表int64")
        custom_dtype_str = input(f"请输入输入 '{input_name}' 的数据类型，或者按回车键使用默认值：")

        if not custom_shape_str:
            # 使用默认的形状
            new_shape = default_input_data[input_name].shape
        else:
            try:
                # 将用户输入的字符串转换为整数列表
                new_shape = [int(dim) for dim in custom_shape_str.split(',')]
            except ValueError:
                print("输入无效，请确保您输入的是逗号分隔的整数。")
                continue

        if not custom_dtype_str:
            # 使用默认的数据类型
            dtype = default_input_data[input_name].dtype
        else:
            dtype = dtype_map.get(custom_dtype_str.strip(), None)
            if dtype is None:
                print("输入无效的数据类型，请输入 'f' 或 'i'。")
                continue

        input_data = np.random.rand(*new_shape).astype(dtype)
        initial_input_data[input_name] = input_data

    return initial_input_data
# 定义单个ONNX模型路径和拆分后的子图模型路径
single_onnx_model_path = './generation_model_simplify.onnx'
model_path = './1211generation_subgraphs'
subgraphsiostxt_path = './1211generation_subgraphs/subgraphs_ios.txt'

# 初始化ModelInference实例，准备进行推理
model_inference = ModelInference(model_path, subgraphsiostxt_path)

# 默认输入数据字典
default_input_data = {
    "attention_mask": np.random.rand(1, 3).astype(np.float32),
    "inputs_embeds": np.random.rand(1, 3, 768).astype(np.float32),
    "decoder_input_ids": np.random.rand(1, 1000).astype(np.int64),
    "curr_gen_idx": np.random.rand(1).astype(np.int64),
}

initial_input_data = prepare_initial_input_data(single_onnx_model_path, default_input_data)
# 使用单个ONNX模型执行推理
output_single = ModelInference.infer_single_onnx_model(single_onnx_model_path, initial_input_data)
# 获取单个模型的所有输出名称
output_names_list = list(output_single.keys())

# 使用拆分后的多个子图模型执行推理
output_multiple = model_inference.inference(initial_input_data,output_names_list)

print("比较单个ONNX模型与多个子图模型的推理结果...")
compare_results(output_single, output_multiple)
