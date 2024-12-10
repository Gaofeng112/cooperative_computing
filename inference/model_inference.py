from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity
from skimage.io import imread
import onnxruntime as ort
import numpy as np
import pandas as pd
import torch
import onnx
import pdb
import re
import os

from quant import quant_conv_forward_save_output

class ModelInference:
    """
    这个类用来推理多个onnx模型

    输入
        model_path: 模型路径
        subgraphsiostxt_path: 模型图结构txt文件路径
        model_input_data: 模型初始数据

    输出
        outputs[0]: 模型推理结果

    说明：
        其中,subgraphsiostxt_path为模型图结构txt文件,用于获取输入输出节点名称,model_path是多个onnx文件
    路径,load_sessions函数会根据subgraphsiostxt_pat中order顺序,对model_path中的onnx模型进行排序,并对
    排序完完的onnx模型进行推理,把sesessions数据返回到self.sessions,把排序后的序列sorted_file_paths返回
    到self.sorted_file_paths,最后根据输入的初始数据initial_input_data,对sessions进行推理,返回推理结果.
    """
    def __init__(self, model_path, subgraphsiostxt_path):
        
        self.model_path = model_path
        self.subgraphsiostxt_path = subgraphsiostxt_path
        self.sessions, self.sorted_file_paths = self.load_sessions()

    def load_sessions(self):
        with open(self.subgraphsiostxt_path, 'r') as file:
            content = file.read()
        subgraph_order_map = {}
        matches = re.findall(r'(\w+)subgraph(\d+): order(\d+)', content)

        for match in matches:
            subgraph_type, subgraph_number, order = match
            # lower_subgraph_type = subgraph_type.lower()
            file_path = os.path.join(self.model_path, f"{subgraph_type}subgraph{subgraph_number}.onnx")
            if int(order) in subgraph_order_map:
                subgraph_order_map[int(order)].append(file_path)
            else:
                subgraph_order_map[int(order)] = [file_path]

        sorted_file_paths = []
        for order in sorted(subgraph_order_map.keys()):
            sorted_file_paths.extend(subgraph_order_map[order])

        sessions = [ort.InferenceSession(model) for model in sorted_file_paths]
        return sessions, sorted_file_paths
    def inference(self, initial_input_data):
        input_data = initial_input_data
        for i, (session,model_file) in enumerate(zip(self.sessions, self.sorted_file_paths)):

            input_names = [inp.name for inp in session.get_inputs()]
            model_input_data = {name: input_data[name] for name in input_names}
            outputs = session.run(None, model_input_data)
            output_names = [out.name for out in session.get_outputs()]

            if i < len(self.sessions) - 1:
                for output, output_name in zip(outputs, output_names):
                    input_data[output_name] = output
        return outputs[0]
    
    def infer_single_onnx_model(model_file, input_data):
        session = ort.InferenceSession(model_file)
        outputs = session.run(None, input_data)
        return outputs[0]



class PcaInference:
    """
    该类使用了PCA进行压缩,推理多个ONNX模型

    输入：
        model_path: onnx模型路径
        subgraphsiostxt_path: 模型图结构txt文件路径
        endwithconv_path: 记录了以卷积结尾的onnx的txt文件路径
        initial_input_data: 初始输入数据
        num: 推理次数,根据次数给出的模型名称
        output_dir: 推理结果保存根目录

    输出：
        outputs: 推理结果

    说明：
        中间会生成一个result_pt,用来保存中间结果,该目录不生成不影响实验结果。
        result文件夹会保存卷积层输出的结果,用来计算压缩率。所有结果会保存在output_dir文件夹中。
    """
    def __init__(self, model_path, subgraphsiostxt_path, endwithconv_path, output_dir):
        self.model_path = model_path
        self.subgraphsiostxt_path = subgraphsiostxt_path
        self.endwithconv_path = endwithconv_path
        self.output_dir = output_dir
        (
            self.sessions, 
            self.conv_output_layer_map, 
            self.sorted_file_paths, 
        ) = self.load_sessions()
    
    def load_sessions(self):
        with open(self.subgraphsiostxt_path, 'r') as file:
            content = file.read()
        subgraph_order_map = {}
        matches = re.findall(r'(\w+)subgraph(\d+): order(\d+)', content)

        for match in matches:
            subgraph_type, subgraph_number, order = match
            # pdb.set_trace()
            # lower_subgraph_type = subgraph_type.lower()
            file_path = os.path.join(self.model_path, f"{subgraph_type}subgraph{subgraph_number}.onnx")
            if int(order) in subgraph_order_map:
                subgraph_order_map[int(order)].append(file_path)
            else:
                subgraph_order_map[int(order)] = [file_path]

        sorted_file_paths = []
        for order in sorted(subgraph_order_map.keys()):
            sorted_file_paths.extend(subgraph_order_map[order])

        sessions = []
        conv_output_layer_map = {}
        # onnx_dict = self.load_onnx_dict()
        for model_file in sorted_file_paths:
            session = ort.InferenceSession(model_file)
            sessions.append(session)
            
            conv_outputs = {}
            if self.onnx_end_conv(model_file):
                model = onnx.load(model_file)
                for idx, node in enumerate(model.graph.node):
                    if node.op_type == 'Conv':
                        for output_name in node.output:
                            if output_name not in conv_outputs:
                                conv_outputs[output_name] = idx + 1
                conv_output_layer_map[model_file] = conv_outputs

        return sessions, conv_output_layer_map, sorted_file_paths
    def load_onnx_dict(self):
        onnx_dict = []
        with open(self.endwithconv_path, 'r') as file:
            content = file.read()
            numbers = re.findall(r'\b\d+\b', content)
            for number in numbers:
                onnx_path = os.path.join(self.model_path, f"NPUsubgraph{number}.onnx")
                onnx_dict.append(onnx_path)
        return onnx_dict
    def onnx_end_conv(self, model_file):
        for onnx in self.load_onnx_dict():
            if onnx == model_file:
                return True
        return False
    

    def check_and_convert_inputs(self,model_input_data):
        for key, value in model_input_data.items():
            if isinstance(value, torch.Tensor):
                model_input_data[key] = value.numpy()  # 转换为 NumPy 数组
            elif not isinstance(value, np.ndarray):
                raise TypeError(f"Input data for '{key}' is not a NumPy array. Got type: {type(value)}")
        return model_input_data

    def decomp(self,compressed_tensor, ru, rbits, num_bits=8):
        decompressed_tensor = torch.dequantize(compressed_tensor)
        decompressed_tensor = decompressed_tensor.numpy()
        if not isinstance(decompressed_tensor, np.ndarray):
            raise TypeError("The decompressed tensor is not a NumPy array.")
        return decompressed_tensor
    
    def inference(self, initial_input_data, num):
        input_data = initial_input_data
        aux_data = {}  # 用于存储 ru 和 rbits
        record_model_name = None # 用于记录上一个模型的名称

        for i, (session, model_file) in enumerate(zip(self.sessions, self.sorted_file_paths)):
            input_names = [inp.name for inp in session.get_inputs()]
            
            if self.onnx_end_conv(record_model_name):
                for name in input_names:
                    if name in input_data and name in aux_data:
                        compressed_tensor = input_data[name]
                        ru, rbits = aux_data[name]
                        decompressed_tensor = self.decomp(compressed_tensor, ru, rbits)
                        input_data[name] = decompressed_tensor
            
            model_input_data = {name: input_data[name] for name in input_names}
            self.check_and_convert_inputs(model_input_data)
            outputs = session.run(None, model_input_data)
            output_names = [out.name for out in session.get_outputs()]
            conv_outputs = self.conv_output_layer_map.get(model_file, {})  # 获取卷积输出，如果没有则为空字典

            for output_name, output in zip(output_names, outputs):
                if output_name in conv_outputs:
                    output_tensor = torch.tensor(output)
                    layer = conv_outputs[output_name]
                    output_tensor = quant_conv_forward_save_output(output_tensor, layer, count=1, bit=8, i=num, output_dir=self.output_dir)
                    input_data[output_name] = output_tensor
                else:
                    input_data[output_name] = output  # 不是卷积输出，直接保存
            record_model_name = model_file  # 记录上一个model_file
    
        return outputs[0]



class ImageMetricsEvaluator:
    """
    用于评估图像质量，包括 MSE, PSNR 和 SSIM。
    
    输入：
        original_dir (str): 原始图像所在目录
        generated_dir (str): 生成图像所在目录
        compression_dir (str): 压缩信息文本文件所在目录
    输出：
        output_file (str): 输出文件路径(Excel)
    """
    def __init__(self, original_dir, generated_dir, compression_dir, output_file):

        self.original_dir = original_dir
        self.generated_dir = generated_dir
        self.compression_dir = compression_dir
        self.output_file = output_file

    def calculate_image_metrics(self, original_image_path, generated_image_path):
        """计算给定原始和生成图像之间的 MSE, PSNR 和 SSIM。"""
        original_image = imread(original_image_path)
        generated_image = imread(generated_image_path)

        if original_image.shape != generated_image.shape:
            raise ValueError('两个图像的尺寸必须相同')

        mse = mean_squared_error(original_image, generated_image)
        psnr = peak_signal_noise_ratio(original_image, generated_image)

        min_dim = min(original_image.shape[:2])
        win_size = min(7, min_dim)
        if win_size % 2 == 0:
            win_size -= 1
        if win_size < 3:
            win_size = 3

        ssim = structural_similarity(original_image, generated_image, multichannel=True, win_size=win_size, channel_axis=-1)

        return mse, psnr, ssim
    
    def calculate_compression_rate(self, file_path):
        """从指定的文本文件中读取并计算平均压缩率。"""
        with open(file_path) as f:
            lines = f.readlines()
            rate_all = sum(float(line.split(',')[0]) * float(line.split(',')[1]) for line in lines)
            all_ = sum(float(line.split(',')[1]) for line in lines)
            return rate_all / all_ if all_ != 0 else None

    def find_matching_compression_file(self, image_name):
        """根据图像文件名查找对应的压缩信息文件。"""
        base_name, _ = os.path.splitext(image_name)
        number = re.search(r'_(\d+)', base_name)
        if number:
            number = number.group(1)
            compression_files = [f for f in os.listdir(self.compression_dir) if f.startswith(f'result_{number}') and f.endswith('.txt')]
            if compression_files:
                return os.path.join(self.compression_dir, compression_files[0])
        return None    
    def compare_images_in_directories(self):
        """比较两个目录中的所有图像，并保存结果至 Excel 文件。"""
        def sort_key(filename):
            parts = filename.split('_')
            try:
                return int(parts[1].split('.')[0]) if len(parts) > 1 else 0
            except (ValueError, IndexError):
                print(f"Warning: Could not parse number from filename {filename}")
                return 0

        original_images = sorted([f for f in os.listdir(self.original_dir) if f.endswith('.png')], key=sort_key)
        generated_images = sorted([f for f in os.listdir(self.generated_dir) if f.endswith('.png')], key=sort_key)

        results = []

        for orig_img_name, gen_img_name in zip(original_images, generated_images):
            orig_img_path = os.path.join(self.original_dir, orig_img_name)
            gen_img_path = os.path.join(self.generated_dir, gen_img_name)

            try:
                mse, psnr, ssim = self.calculate_image_metrics(orig_img_path, gen_img_path)
                # 尝试找到与图像名称对应的压缩信息文件
                compression_file_path = self.find_matching_compression_file(orig_img_name)
                compression_rate = self.calculate_compression_rate( compression_file_path) if compression_file_path else None
                # pdb.set_trace()
                results.append({
                    'Original Image': orig_img_name,
                    'Generated Image': gen_img_name,
                    'MSE': mse,
                    'PSNR': psnr,
                    'SSIM': ssim,
                    'Compression Rate': compression_rate
                })
            except Exception as e:
                print(f"Error processing images {orig_img_name} and {gen_img_name}: {e}")

        df = pd.DataFrame(results)

        output_dir = os.path.dirname(self.output_file)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        try:
            df.to_excel(self.output_file, index=False)
            print(f'Results have been saved to {self.output_file}')
        except PermissionError:
            print(f"Permission denied: Unable to write to {self.output_file}. Please check file permissions or close the file if it is open in another program.")
        except Exception as e:
            print(f"An error occurred while saving the results: {e}")
 