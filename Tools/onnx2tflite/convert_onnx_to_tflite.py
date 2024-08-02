import os
from onnx2tflite import onnx_converter

def convert_all_onnx_models(input_dir, output_dir):
    for file_name in os.listdir(input_dir):
        if file_name.endswith('.onnx'):
            onnx_model_path = os.path.join(input_dir, file_name)
            print(onnx_model_path)
            output_path = os.path.join(output_dir, file_name.replace('.onnx', '.tflite'))
            res = onnx_converter(
                onnx_model_path=onnx_model_path,
                need_simplify=True,
                output_path=output_dir,
                target_formats=['tflite'],
                weight_quant= True
            )
            print(f'Converted {onnx_model_path} to {output_path}')


os.makedirs('./tflite', exist_ok=True)

input_directory = '/your/path/to/onnx'
output_directory = './tflite'

convert_all_onnx_models(input_directory, output_directory)
