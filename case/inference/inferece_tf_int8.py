import numpy as np
import math
import multiprocessing
import os
import random
import tensorflow as tf
from clip_tokenizer import SimpleTokenizer
from lcm_scheduler import LCMScheduler
from PIL import Image
from tqdm import tqdm
import re

def parse_model_info(file_path):
    model_info = {}
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    model_name = None
    for line in lines:
        line = line.strip()
        if line.startswith('Model:'):
            model_name = line.split(':')[1].strip()
            model_info[model_name] = {'Inputs': {}, 'Outputs': {}}
        elif line.startswith('Inputs:'):
            current_section = 'Inputs'
        elif line.startswith('Outputs:'):
            current_section = 'Outputs'
        elif line and model_name:
            match = re.match(r'TFLite:\s*(.*) -> ONNX:\s*(.*)', line)
            if match and model_name:
                tflite_name, onnx_name = match.groups()
                model_info[model_name][current_section][tflite_name] = onnx_name

    return model_info

def print_model_info(model_info):
    for model, info in model_info.items():
        print(f"Model: {model}")
        print("Inputs:")
        for tflite, onnx in info['Inputs'].items():
            print(f"  TFLite: {tflite} -> ONNX: {onnx}")
        print("Outputs:")
        for tflite, onnx in info['Outputs'].items():
            print(f"  TFLite: {tflite} -> ONNX: {onnx}")
        print()

file_path = 'modif.txt'
model_info = parse_model_info(file_path)
print_model_info(model_info)

def get_onnx_input_name(model_info, model_name, tflite_input_name):
    if model_name not in model_info:
        raise ValueError(f"Model '{model_name}' not found in model_info.")
    
    inputs = model_info[model_name].get('Inputs', {})
    if tflite_input_name not in inputs:
        raise ValueError(f"TFLite input name '{tflite_input_name}' not found for model '{model_name}'.")
    
    onnx_name = inputs[tflite_input_name]
    return onnx_name

def get_onnx_output_name(model_info, model_name, tflite_output_name):
    if model_name not in model_info:
        raise ValueError(f"Model '{model_name}' not found in model_info.")
    
    outpus = model_info[model_name].get('Outputs', {})
    if tflite_output_name not in outpus:
        raise ValueError(f"TFLite outpus name '{tflite_output_name}' not found for model '{model_name}'.")
    
    onnx_name = outpus[tflite_output_name]
    return onnx_name

def convert_input_to_onnx_names(model_info, model_name, tflite_indices, onnx_input_name):
    for tflite_name, index in tflite_indices.items():
        try:
            onnx_name = get_onnx_input_name(model_info, model_name, tflite_name)
            onnx_input_name.append(onnx_name)
        except ValueError as e:
            print(e)

def convert_output_to_onnx_names(model_info, model_name, tflite_indices, onnx_input_name):
    for tflite_name, index in tflite_indices.items():
        try:
            onnx_name = get_onnx_output_name(model_info, model_name, tflite_name)
            onnx_input_name.append(onnx_name)
        except ValueError as e:
            print(e)

flie = open('subgraphs_ios.txt','r')
content = flie.read()
subgraph_order_map = {}
matches = re.findall(r'(\w+)subgraph(\d+): order(\d+)', content)

for match in matches:
    subgraph_type, subgraph_number, order = match
    lower_subgraph_type = subgraph_type.lower()
    file_path = f"../../Tools/onnx2tflite/tflite/{lower_subgraph_type}subgraph{subgraph_number}.tflite"
    subgraph_order_map[int(order)] = file_path

sorted_file_paths = [subgraph_order_map[order] for order in sorted(subgraph_order_map.keys())]

model_files = []
for model_file in sorted_file_paths:
    model_files.append(model_file)

print(model_files)

interpreters = [tf.lite.Interpreter(model_path=model) for model in model_files]

device_name = 'cpu'
providers = ['CPUExecutionProvider']

prompt="An island on the sea"
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

path_to_saved_models="./int8"

scheduler = LCMScheduler(beta_start=0.00085, beta_end=0.0120, beta_schedule="scaled_linear", prediction_type="epsilon")
scheduler.set_timesteps(num_steps, 50)
text_encoder = tf.lite.Interpreter(model_path= os.path.join(path_to_saved_models,"converted_text_encoder.tflite"))
text_encoder.allocate_tensors()
input_details_text_encoder = text_encoder.get_input_details()
output_details_text_encoder = text_encoder.get_output_details()

decoder = tf.lite.Interpreter(model_path=os.path.join(path_to_saved_models,"converted_decoder.tflite"))
decoder.allocate_tensors()
input_details_decoder = decoder.get_input_details()
output_details_decoder = decoder.get_output_details()

import time

start = time.time()

inputs = tokenizer.encode(prompt)
phrase = inputs + [49407] * (77 - len(inputs))
phrase = np.array(phrase)[None].astype("int32")
phrase = np.repeat(phrase, batch_size, axis=0)
pos_ids = np.array(list(range(77)))[None].astype("int32")
pos_ids = np.repeat(pos_ids, batch_size, axis=0)
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
for index_, timestep in progbar:
    progbar.set_description(f"{index_:3d} {timestep:3d}")

    initial_input_data = {
        "sample": np.array(latent, dtype=np.float32),
        "timestep": np.array([float(timestep)], dtype=np.float32),
        "encoder_hidden_states": np.array(context, dtype=np.float32)
    }
    input_data = initial_input_data

    for i, (interpreter, model_file) in enumerate(zip(interpreters, model_files)):
        name_without_ext =''
        match = re.search(r'/([^/]+)\.tflite$', model_file)
        if match:
            name_without_ext = match.group(1)

        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        input_indices = {tensor['name']: i for i, tensor in enumerate(input_details)}
        output_indices = {tensor['name']: i for i, tensor in enumerate(output_details)}

        onnx_input_name = []
        onnx_output_name = []

        convert_input_to_onnx_names(model_info, name_without_ext, input_indices, onnx_input_name)
        convert_output_to_onnx_names(model_info, name_without_ext, output_indices, onnx_output_name)

        model_input_data = {name: input_data[name] for name in onnx_input_name}

        for name, (_, index) in zip(onnx_input_name, input_indices.items()):
            shape1 = input_details[index]['shape']
            shape2 = model_input_data[name].shape

            def convert_to_nhwc(data):
                return np.transpose(data, (0, 2, 3, 1))

            def convert_to_nchw(data):
                return np.transpose(data, (0, 3, 1, 2))

            def is_nchw(shape):
                return len(shape) == 4 and (shape[2] == shape[3])

            def is_nhwc(shape):
                return len(shape) == 4 and (shape[1] == shape[2])

            def determine_format(shape):

                if len(shape) == 4:
                    if (shape[2] == shape[3]):
                        return 'NCHW'
                    elif (shape[1] == shape[2]):
                        return 'NHWC'
                return None

            target_format = determine_format(shape1)

            if tuple(shape1) == tuple(shape2):
                model_input_data_convert = model_input_data[name]
            else:
                if target_format == 'NHWC' and is_nchw(shape2):
                    model_input_data_convert = convert_to_nhwc(model_input_data[name])
                elif target_format == 'NCHW' and is_nhwc(shape2):
                    model_input_data_convert = convert_to_nchw(model_input_data[name])

            interpreter.set_tensor(input_details[index]['index'], model_input_data_convert)

        interpreter.invoke()

        outputs = {name: interpreter.get_tensor(output_details[index]['index']) for name, index in output_indices.items()}

        if i < len(interpreters) - 1:
            for k, (name, output) in enumerate(outputs.items()):
                input_data[onnx_output_name[k]] = output

    for name, output in outputs.items():
        e_t = output
    e_t_hf = np.transpose(e_t, (0, 3, 1, 2))
    latent_hf = np.transpose(latent, (0, 3, 1, 2))
    output_latent = scheduler.step(e_t_hf, index_, timestep, latent_hf)
    latent = np.transpose(output_latent[0], (0, 2, 3, 1))

denoised = output_latent[1]
denoised = np.transpose(denoised, (0, 2, 3, 1))
decoder.set_tensor(input_details_decoder[0]['index'], denoised)
decoder.invoke()
decoded = decoder.get_tensor(output_details_decoder[0]['index'])
decoded = ((decoded + 1) / 2) * 255
img=np.clip(decoded, 0, 255).astype("uint8")
image = Image.fromarray(img[0])
image.save("test.png")
end = time.time()

print("The time of execution of above program is :",
      (end-start) * 10**3, "ms")
