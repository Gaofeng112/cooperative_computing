"""
加载多个onnx模型,使用压缩解压,解决numpy格式问题
自动化,压缩率,带有封装类,设置了根目录
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
from model_inference import PcaInference

parser = argparse.ArgumentParser(description="Generate images using ONNX models.")
parser.add_argument("--prompt", type=str, required=True, help="Prompt for generating the image")
parser.add_argument("--output", type=str, required=True, help="image name")
parser.add_argument("--num", type=int, required=True, help="the number of images to generate")
args = parser.parse_args()

model_path='./1108_subgraphs'
subgraphsiostxt_path='./1108_subgraphs/subgraphs_ios.txt'
endwithconv_path='./1108_subgraphs/end_with_conv.txt'

output_dir = "./output"
os.makedirs(output_dir, exist_ok=True)

model_inference = PcaInference(model_path, subgraphsiostxt_path, endwithconv_path, output_dir)

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


    e_t_hf = model_inference.inference(initial_input_data, args.num)
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

iamges_path = output_dir + "/pca"
os.makedirs(iamges_path, exist_ok=True)
image.save(os.path.join(iamges_path, args.output))

# evaluator = ImageMetricsEvaluator(
#         original_dir=output_dir + "origin",
#         generated_dir=iamges_path,
#         compression_dir=output_dir + "results",  # 假设压缩信息文本文件存储在此目录下
#         output_file=output_dir + "comp.xlsx"
#     )
# evaluator.compare_images_in_directories()