"""
加载多个onnx模型,使用压缩解压，解决numpy格式问题
自动化,压缩率内部
"""
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
import numpy as np
import math
import multiprocessing
import os
import random
import numpy as np
import tensorflow as tf
from clip_tokenizer import SimpleTokenizer
from lcm_scheduler import LCMScheduler
from PIL import Image
from tqdm import tqdm
import re
import time
import pdb
import argparse
import onnx
import torch
import torch.nn as nn
# from get_middle_data_upload.quant import comp
from quant import *

parser = argparse.ArgumentParser(description="Generate images using ONNX models.")
parser.add_argument("--prompt", type=str, required=True, help="Prompt for generating the image")
parser.add_argument("--output", type=str, required=True, help="Output filename for the generated image")
parser.add_argument("--num", type=int, required=True, help="the number of images to generate")
args = parser.parse_args()

# 读取文件
with open('./1108_subgraphs/subgraphs_ios.txt', 'r') as file:
    content = file.read()
subgraph_order_map = {}
matches = re.findall(r'(\w+)subgraph(\d+): order(\d+)', content)

for match in matches:
    subgraph_type, subgraph_number, order = match
    lower_subgraph_type = subgraph_type.lower()
    file_path = f"./1108_subgraphs/{lower_subgraph_type}subgraph{subgraph_number}.onnx"
    if int(order) in subgraph_order_map:
        subgraph_order_map[int(order)].append(file_path)
    else:
        subgraph_order_map[int(order)] = [file_path]
sorted_file_paths = []
for order in sorted(subgraph_order_map.keys()):
    sorted_file_paths.extend(subgraph_order_map[order])
model_files = sorted_file_paths

#读取以卷积结尾的文档
onnx_dict = []
with open('./1108_subgraphs/end_with_conv.txt', 'r') as file:
    content = file.read()
    numbers = re.findall(r'\b\d+\b', content)
    for number in numbers:
        onnx_path = f"./1108_subgraphs/npusubgraph{number}.onnx"
        onnx_dict.append(onnx_path)

def onnx_end_conv(model_file):
    for onnx in onnx_dict:
        if model_file == onnx:
            return True
    return False

# 加载模型并记录卷积结尾的输出张量的层数
def load_models_and_record_layers(model_files):
    sessions = []
    conv_output_layer_map = {}
    for model_file in model_files:
        model = onnx.load(model_file)
        session = ort.InferenceSession(model_file)
        sessions.append(session)
        
        # 记录卷积结尾的输出及其层数
        conv_outputs = {}
        if onnx_end_conv(model_file):
            for idx, node in enumerate(model.graph.node):
                if node.op_type == 'Conv':
                    for output_name in node.output:
                        if output_name not in conv_outputs:
                            conv_outputs[output_name] = idx + 1  # 从1开始计数
            conv_output_layer_map[model_file] = conv_outputs
    
    return sessions, conv_output_layer_map

# 检查并转换输入数据为 NumPy 数组
def check_and_convert_inputs(model_input_data):
    for key, value in model_input_data.items():
        if isinstance(value, torch.Tensor):
            model_input_data[key] = value.numpy()  # 转换为 NumPy 数组
        elif not isinstance(value, np.ndarray):
            raise TypeError(f"Input data for '{key}' is not a NumPy array. Got type: {type(value)}")

device_name = 'cpu'
providers = ['CPUExecutionProvider']

# prompt="an astronaut riding a horse"
# prompt="DSLR photograph of an astronaut riding a horse"
# prompt="An island on the sea"
prompt=args.prompt
negative_prompt=None
batch_size=1
num_steps=4
unconditional_guidance_scale=7.5
temperature=1
seed=42
input_image=None
input_mask=None
input_image_strength=0.5
tokenizer=SimpleTokenizer()

def decomp(compressed_tensor, ru, rbits, num_bits=8):
    decompressed_tensor = torch.dequantize(compressed_tensor)
    # 将 PyTorch 张量转换回 NumPy 数组
    decompressed_tensor = decompressed_tensor.numpy()
    # 检查是否为 NumPy 数组
    if not isinstance(decompressed_tensor, np.ndarray):
        raise TypeError("The decompressed tensor is not a NumPy array.")
    return decompressed_tensor
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(42)

path_to_saved_models="./fp32"
scheduler = LCMScheduler(beta_start=0.00085, beta_end=0.0120, beta_schedule="scaled_linear", prediction_type="epsilon")
scheduler.set_timesteps(num_steps, 50)
text_encoder = tf.lite.Interpreter(model_path= os.path.join(path_to_saved_models,"converted_text_encoder.tflite"),num_threads=multiprocessing.cpu_count())
text_encoder.allocate_tensors()

# Get input and output tensors.
input_details_text_encoder = text_encoder.get_input_details()
output_details_text_encoder = text_encoder.get_output_details()


# diffusion_model = onnxruntime.InferenceSession(input_onnx_path, providers=providers)
decoder = tf.lite.Interpreter(model_path=os.path.join(path_to_saved_models,"converted_decoder.tflite"),num_threads=multiprocessing.cpu_count())
decoder.allocate_tensors()
input_details_decoder = decoder.get_input_details()
output_details_decoder = decoder.get_output_details()

inputs = tokenizer.encode(prompt)
# assert len(inputs) < 77, "Prompt is too long (should be < 77 tokens)"
phrase = inputs + [49407] * (77 - len(inputs))
phrase = np.array(phrase)[None].astype("int32")
phrase = np.repeat(phrase, batch_size, axis=0)
# Encode prompt tokens (and their positions) into a "context vector"
pos_ids = np.array(list(range(77)))[None].astype("int32")
pos_ids = np.repeat(pos_ids, batch_size, axis=0)
# context = model.text_encoder.predict_on_batch([phrase, pos_ids])
# print(f"context shape {context.shape}")
text_encoder.set_tensor(input_details_text_encoder[0]['index'], phrase)
text_encoder.set_tensor(input_details_text_encoder[1]['index'], pos_ids)
text_encoder.invoke()
context = text_encoder.get_tensor(output_details_text_encoder[0]['index'])

img_height=256
img_width=256   
n_h = img_height // 8
n_w = img_width // 8
latent = tf.random.normal((batch_size, n_h, n_w, 4), seed=seed)

def timestep_embedding(timesteps, dim=320, max_period=10000):
    half = dim // 2
    freqs = np.exp(
        -math.log(max_period) * np.arange(0, half, dtype="float32") / half
    )
    args = np.array(timesteps) * freqs
    embedding = np.concatenate([np.cos(args), np.sin(args)])
    return embedding.reshape(1, -1).astype(np.float32)

def get_guidance_scale_embedding(w, embedding_dim=512):
    w = w * 1000.0

    half_dim = embedding_dim // 2
    emb = np.exp(np.arange(half_dim) * -np.log(10000.0) / (half_dim - 1))
    emb = np.expand_dims(w, axis=-1) * np.expand_dims(emb, axis=0)
    emb = np.concatenate([np.sin(emb), np.cos(emb)], axis=-1)

    return emb

guidance_scale_embedding = get_guidance_scale_embedding(unconditional_guidance_scale, 256).astype(np.float32)
   
timesteps = scheduler.timesteps
progbar = tqdm(enumerate(timesteps))
for index, timestep in progbar:
    progbar.set_description(f"{index:3d} {timestep:3d}")
    initial_input_data = {
        "sample": np.array(latent, dtype=np.float32).transpose(0, 3, 1, 2),
        "timestep": np.array([float(timestep)], dtype=np.float32),
        "encoder_hidden_states": np.array(context, dtype=np.float32),
    }

    input_data = initial_input_data
    aux_data = {}  # 用于存储 ru 和 rbits
    sessions, conv_output_layer_map = load_models_and_record_layers(model_files)
    record_model_name = None

    for i, (session, model_file) in enumerate(zip(sessions, model_files)):
        input_names = [inp.name for inp in session.get_inputs()]
        # 如果上一个模型是以卷积结尾，则解压
        if  onnx_end_conv(record_model_name):
            for name in input_names:
                if name in input_data and name in aux_data:
                    compressed_tensor = input_data[name]
                    ru, rbits = aux_data[name]
                    decompressed_tensor = decomp(compressed_tensor, ru, rbits)
                    input_data[name] = decompressed_tensor
        
        model_input_data = {name: input_data[name] for name in input_names}
        check_and_convert_inputs(model_input_data)
        outputs = session.run(None, model_input_data)
        output_names = [out.name for out in session.get_outputs()]
        conv_outputs = conv_output_layer_map.get(model_file, {})  # 获取卷积输出，如果没有则为空字典

        for output_name, output in zip(output_names, outputs):
                if output_name in conv_outputs:
                    # print(f"Processing conv output: {output_name}")
                    # pdb.set_trace()
                    output_tensor = torch.tensor(output)
                    layer = conv_outputs[output_name]
                    output_tensor = quant_conv_forward_save_output(output_tensor,layer,count=1, bit=8, i=args.num)
                    input_data[output_name] = output_tensor
                else:
                    input_data[output_name] = output  # 不是卷积输出，直接保存
        record_model_name = model_file#记录上一个model_file
    
    # final_output = {name: output for name, output in zip(output_names, outputs)}
    e_t_hf = outputs[0]
    latent_hf = np.transpose(latent, (0, 3, 1, 2))
    output_latent = scheduler.step(e_t_hf, index, timestep, latent_hf)
    latent = np.transpose(output_latent[0], (0, 2, 3, 1))

denoised = output_latent[1]
denoised = np.transpose(denoised, (0, 2, 3, 1))
decoder.set_tensor(input_details_decoder[0]['index'], denoised)
decoder.invoke()
decoded = decoder.get_tensor(output_details_decoder[0]['index'])
# decoded = model.decoder.predict_on_batch(latent)
decoded = ((decoded + 1) / 2) * 255
img=np.clip(decoded, 0, 255).astype("uint8")
image = Image.fromarray(img[0])
image.save(args.output)
