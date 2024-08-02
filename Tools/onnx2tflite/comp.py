import os
import onnx
import tensorflow as tf

def get_onnx_io_shapes(onnx_file):
    model = onnx.load(onnx_file)
    input_shapes = {i.name: tuple(d.dim_value for d in i.type.tensor_type.shape.dim) for i in model.graph.input}
    output_shapes = {o.name: tuple(d.dim_value for d in o.type.tensor_type.shape.dim) for o in model.graph.output}
    return input_shapes, output_shapes

def get_tflite_io_shapes(tflite_file):
    interpreter = tf.lite.Interpreter(model_path=tflite_file)
    input_shapes = {interpreter.get_input_details()[i]['name']: list(interpreter.get_input_details()[i]['shape']) for i in range(len(interpreter.get_input_details()))}
    output_shapes = {interpreter.get_output_details()[i]['name']: list(interpreter.get_output_details()[i]['shape']) for i in range(len(interpreter.get_output_details()))}
    return input_shapes, output_shapes

def compare_shapes_and_generate_dict(folder_path):
    onnx_files = [f for f in os.listdir(folder_path) if f.endswith('.onnx')]
    tflite_files = [f for f in os.listdir(folder_path) if f.endswith('.tflite')]

    name_dict = {}


    def shape_product(shape):
        product = 1
        for dim in shape:
            product *= dim
        return product

    for onnx_file in onnx_files:
        base_name = os.path.splitext(onnx_file)[0]
        tflite_file = f"{base_name}.tflite"
        if tflite_file not in tflite_files:
            continue

        onnx_path = os.path.join(folder_path, onnx_file)
        tflite_path = os.path.join(folder_path, tflite_file)

        onnx_inputs, onnx_outputs = get_onnx_io_shapes(onnx_path)
        tflite_inputs, tflite_outputs = get_tflite_io_shapes(tflite_path)

        input_name_map = {}
        for tflite_name, tflite_shape in tflite_inputs.items():
            tflite_shape = tuple(tflite_shape)
            onnx_keys_to_remove = []
            for onnx_name, onnx_shape in onnx_inputs.items():
                onnx_shape = tuple(onnx_shape)
                if tflite_shape == onnx_shape or (shape_product(tflite_shape) == shape_product(onnx_shape) and len(tflite_shape) == len(onnx_shape)):
                    input_name_map[tflite_name] = onnx_name
                    onnx_keys_to_remove.append(onnx_name)
                    break
            for key in onnx_keys_to_remove:
                del onnx_inputs[key]

        output_name_map = {}
        for tflite_name, tflite_shape in tflite_outputs.items():
            tflite_shape = tuple(tflite_shape)
            onnx_keys_to_remove = []
            for onnx_name, onnx_shape in onnx_outputs.items():
                onnx_shape = tuple(onnx_shape)
                if tflite_shape == onnx_shape or (shape_product(tflite_shape) == shape_product(onnx_shape) and len(tflite_shape) == len(onnx_shape)):
                    output_name_map[tflite_name] = onnx_name
                    onnx_keys_to_remove.append(onnx_name)
                    break
            for key in onnx_keys_to_remove:
                del onnx_outputs[key]

        if input_name_map or output_name_map:
            name_dict[base_name] = {
                'inputs': input_name_map,
                'outputs': output_name_map
            }

    return name_dict

def save_to_txt(data, output_file):
    with open(output_file, 'w') as file:
        for model_name, io_map in data.items():
            file.write(f"Model: {model_name}\n")
            file.write("Inputs:\n")
            for tflite_name, onnx_name in io_map['inputs'].items():
                file.write(f"  TFLite: {tflite_name} -> ONNX: {onnx_name}\n")
            file.write("Outputs:\n")
            for tflite_name, onnx_name in io_map['outputs'].items():
                file.write(f"  TFLite: {tflite_name} -> ONNX: {onnx_name}\n")
            file.write("\n")

if __name__ == "__main__":
    folder_path = './tflite'
    output_file = 'io_mapping.txt'
    
    name_dict = compare_shapes_and_generate_dict(folder_path)
    save_to_txt(name_dict, output_file)

    print(f"Results have been saved to {output_file}")
