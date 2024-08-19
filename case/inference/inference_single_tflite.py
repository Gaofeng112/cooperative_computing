
import argparse
import sys

import math
import multiprocessing
import os
import random
from PIL import Image
import numpy as np
import tensorflow as tf
from lcm_scheduler import LCMScheduler
from clip_tokenizer import SimpleTokenizer
from constants import (
    _ALPHAS_CUMPROD,
    _UNCONDITIONAL_TOKENS,
    PYTORCH_CKPT_MAPPING,
)
from tqdm import tqdm
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(42)

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-p', '--prompt', type=str, default="An island on the sea.", help="Prompt for the model")
parser.add_argument('-m', '--model-dir', type=str, default="./int8", help="Path to the model directory")
parser.add_argument('-o', '--output', type=str, default="output.png", help="Output file name")
args = parser.parse_args()

path_to_saved_models=args.model_dir

scheduler = LCMScheduler(beta_start=0.00085, beta_end=0.0120, beta_schedule="scaled_linear", prediction_type="epsilon")
scheduler.set_timesteps(4, 50)
text_encoder = tf.lite.Interpreter(model_path= os.path.join(path_to_saved_models,"converted_text_encoder.tflite"))
text_encoder.allocate_tensors()
input_details_text_encoder = text_encoder.get_input_details()
output_details_text_encoder = text_encoder.get_output_details()
diffusion_model = tf.lite.Interpreter(model_path=os.path.join(path_to_saved_models,"converted_diffusion_model_new.tflite"))
diffusion_model.allocate_tensors()
input_details_diffusion = diffusion_model.get_input_details()
output_details_diffusion =diffusion_model.get_output_details()
decoder = tf.lite.Interpreter(model_path=os.path.join(path_to_saved_models,"converted_decoder.tflite"))
decoder.allocate_tensors()
input_details_decoder = decoder.get_input_details()
output_details_decoder = decoder.get_output_details()

import time

start = time.time()
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
        
input_image_tensor = None

unconditional_tokens=_UNCONDITIONAL_TOKENS
unconditional_tokens = np.array(unconditional_tokens)[None].astype("int32")
unconditional_tokens = np.repeat(unconditional_tokens, batch_size, axis=0)

text_encoder.set_tensor(input_details_text_encoder[0]['index'], unconditional_tokens)
text_encoder.set_tensor(input_details_text_encoder[1]['index'], pos_ids)
text_encoder.invoke()
unconditional_context = text_encoder.get_tensor(output_details_text_encoder[0]['index'])

timesteps = np.arange(1, 1000, 1000 // num_steps)
input_img_noise_t = timesteps[ int(len(timesteps)*input_image_strength) ]
img_height=256
img_width=256   
def get_starting_parameters(timesteps, batch_size, seed,  input_image=None, input_img_noise_t=None):
    n_h = img_height // 8
    n_w = img_width // 8
    alphas = [_ALPHAS_CUMPROD[t] for t in timesteps]
    alphas_prev = [1.0] + alphas[:-1]
    if input_image is None:
        latent = tf.random.normal((batch_size, n_h, n_w, 4), seed=seed)
    else:
        latent = encoder(input_image)
        latent = tf.repeat(latent , batch_size , axis=0)
        latent = add_noise(latent, input_img_noise_t)
    return latent, alphas, alphas_prev
latent, alphas, alphas_prev = get_starting_parameters(
            timesteps, batch_size, seed , input_image=input_image_tensor, input_img_noise_t=input_img_noise_t
        )

tf.keras.mixed_precision.global_policy().name== 'mixed_float16'
dtype = tf.float32
if tf.keras.mixed_precision.global_policy().name == 'mixed_float16':
    dtype = tf.float16
def timestep_embedding(timesteps, dim=320, max_period=10000):
    half = dim // 2
    freqs = np.exp(
        -math.log(max_period) * np.arange(0, half, dtype="float32") / half
    )
    args = np.array(timesteps) * freqs
    embedding = np.concatenate([np.cos(args), np.sin(args)])
    return tf.convert_to_tensor(embedding.reshape(1, -1),dtype=dtype)

import numpy as np

def get_guidance_scale_embedding(w, embedding_dim=512):
    w = w * 1000.0

    half_dim = embedding_dim // 2
    emb = np.log(10000.0) / (half_dim - 1)
    emb = np.exp(np.arange(half_dim) * -emb)
    emb = np.expand_dims(w, axis=-1) * np.expand_dims(emb, axis=0)
    emb = np.concatenate([np.sin(emb), np.cos(emb)], axis=-1)
    
    if embedding_dim % 2 == 1:  # zero pad
        emb = np.pad(emb, ((0, 0), (0, 1)), mode='constant')
    
    return emb

def get_x_prev_and_pred_x0(x, e_t, index, a_t, a_prev, temperature, seed):
    sigma_t = 0
    sqrt_one_minus_at = math.sqrt(1 - a_t)
    pred_x0 = (x - sqrt_one_minus_at * e_t) / math.sqrt(a_t)
    dir_xt = math.sqrt(1.0 - a_prev - sigma_t**2) * e_t
    noise = sigma_t * tf.random.normal(x.shape, seed=seed) * temperature
    x_prev = math.sqrt(a_prev) * pred_x0 + dir_xt
    return x_prev, pred_x0


timesteps = scheduler.timesteps
progbar = tqdm(enumerate(timesteps))

for index, timestep in progbar:
    i_start_time = time.time()
    progbar.set_description(f"{index:3d} {timestep:3d}")

    diffusion_model.set_tensor(input_details_diffusion[0]['index'], latent)
    diffusion_model.set_tensor(input_details_diffusion[2]['index'], timestep_embedding([timestep]))
    diffusion_model.set_tensor(input_details_diffusion[1]['index'], context)

    diffusion_model.invoke()

    e_t = diffusion_model.get_tensor(output_details_diffusion[0]['index'])
    e_t_hf = np.transpose(e_t, (0, 3, 1, 2))
    index_hf = index
    timestep_hf = timestep
    latent_hf = np.transpose(latent, (0, 3, 1, 2))
    output_latent = scheduler.step(e_t_hf, index_hf, timestep_hf, latent_hf)
    latent = np.transpose(output_latent[0], (0, 2, 3, 1))
    i_end_time = time.time()

    print("unet is :",
        (i_end_time - i_start_time) * 10**3, "ms")

denoised = output_latent[1]
denoised_image = np.transpose(denoised, (0, 2, 3, 1))
decoder.set_tensor(input_details_decoder[0]['index'], denoised_image)
decoder.invoke()
decoded = decoder.get_tensor(output_details_decoder[0]['index'])

decoded = ((decoded + 1) / 2) * 255
img=np.clip(decoded, 0, 255).astype("uint8")
image = Image.fromarray(img[0])
image.save("test.png")
end = time.time()

print("The time of execution of above program is :",
      (end-start) * 10**3, "ms")