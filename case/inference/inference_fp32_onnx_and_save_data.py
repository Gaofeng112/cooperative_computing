import onnxruntime as ort
import numpy as np
import multiprocessing
import os
import random
import tensorflow as tf
from clip_tokenizer import SimpleTokenizer
from lcm_scheduler import LCMScheduler
from PIL import Image
from tqdm import tqdm
import re

# 需要把以下路径换成实际的
flie = open('subgraphs_ios.txt','r')
content = flie.read()
subgraph_order_map = {}
matches = re.findall(r'(\w+)subgraph(\d+): order(\d+)', content)

for match in matches:
    subgraph_type, subgraph_number, order = match
    lower_subgraph_type = subgraph_type.lower()
    # 需要把以下路径换成实际的
    file_path = f"./sg/{lower_subgraph_type}subgraph{subgraph_number}.onnx"
    subgraph_order_map[int(order)] = file_path

sorted_file_paths = [subgraph_order_map[order] for order in sorted(subgraph_order_map.keys())]

model_files = []
for model_file in sorted_file_paths:
    model_files.append(model_file)

sessions = [ort.InferenceSession(model) for model in model_files]

device_name = 'cpu'
providers = ['CPUExecutionProvider']

#需要保存多少个prompt的中间数据，那就在这里写多少个prompt
prompts=["An island on the sea.",
        "DSLR photograph of an astronaut riding a horse.",
        "Three zebras in a field near bushes.",
        "A brown horse standing next to a woman in front of a house.",
        "A herd of cattle grazing on a lush green field.",
        "A cat laying on clothes that are in a suitcase.",
        "a group of sheep standing next to each other on a field",
        "a couple of young kids are sitting together",
        "a plate of donuts with a person in the background"]
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
# 需要把以下路径换成实际的
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
for id, prompt in enumerate(prompts):
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
    
    timesteps = scheduler.timesteps
    progbar = tqdm(enumerate(timesteps))
    for index, timestep in progbar:
        progbar.set_description(f"{index:3d} {timestep:3d}")

        initial_input_data = {
            "sample": np.array(latent, dtype=np.float32).transpose(0, 3, 1, 2),
            "timestep": np.array([float(timestep)], dtype=np.float32),
            "encoder_hidden_states": np.array(context, dtype=np.float32)
        }
        input_data = initial_input_data

        for i, session in enumerate(sessions):
            input_names = [inp.name for inp in session.get_inputs()]

            model_input_data = {name: input_data[name] for name in input_names}
            
            outputs = session.run(None, model_input_data)

            output_names = [out.name for out in session.get_outputs()]
            
            if i < len(sessions) - 1:
                for output, output_name in zip(outputs, output_names):
                    input_data[output_name] = output

        #每一个onnx的输出都保存在bin_files里
        output_dir = "./bin_files"
        os.makedirs(output_dir, exist_ok=True)

        for key, data in initial_input_data.items():
            safe_key = key.replace('/', '_')
            filename = f"Prompt_{id}_{safe_key}_{index}.bin"
            filepath = os.path.join(output_dir, filename)
            data.tofile(filepath)

        e_t_hf = outputs[0]
        latent_hf = np.transpose(latent, (0, 3, 1, 2))
        output_latent = scheduler.step(e_t_hf, index, timestep, latent_hf)

        latent = np.transpose(output_latent[0], (0, 2, 3, 1))

    denoised = output_latent[1]
    denoised = np.transpose(denoised, (0, 2, 3, 1))
    decoder.set_tensor(input_details_decoder[0]['index'], denoised)
    decoder.invoke()
    decoded = decoder.get_tensor(output_details_decoder[0]['index'])
    decoded = ((decoded + 1) / 2) * 255
    img=np.clip(decoded, 0, 255).astype("uint8")
    image = Image.fromarray(img[0])
    png_name = f"Prompt_{id}.png"
    image.save(png_name)

end = time.time()

print("The time of execution of above program is :",
      (end-start) * 10**3, "ms")