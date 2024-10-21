import os
import numpy as np

def parse_line(line):
    input_shape = None
    input_name = None
    calibrate_dataset_name = None

    # 检测--input-shape
    if "--input-shape" in line:
        input_shape_start = line.find("--input-shape") + len("--input-shape") + 2
        input_shape_end = line.find('"', input_shape_start)
        input_shape = line[input_shape_start:input_shape_end]

    # 检测--input-name
    if "--input-name" in line:
        input_name_start = line.find("--input-name") + len("--input-name") + 2
        input_name_end = line.find('"', input_name_start)
        input_name = line[input_name_start:input_name_end]

    # 检测--calibrate-dataset
    if "--calibrate-dataset" in line:
        calibrate_dataset_start = line.find("--calibrate-dataset") + len("--calibrate-dataset") + 1
        calibrate_dataset_end = line.find('.', calibrate_dataset_start)
        calibrate_dataset_name = line[calibrate_dataset_start:calibrate_dataset_end]

    return input_shape, input_name, calibrate_dataset_name

def process_input_shape(input_shape):
    shapes = input_shape.split(';')
    processed_shapes = []
    for shape in shapes:
        shape = shape.strip()
        shape_parts = shape.split()
        shape_parts = [int(part) for part in shape_parts]
        processed_shapes.append(shape_parts)
    return processed_shapes

def generate_data(input_shapes, input_names):
    data = {}
    for i, (input_shape, input_name) in enumerate(zip(input_shapes, input_names)):
        processed_shapes = process_input_shape(input_shape)
        for j, shape in enumerate(processed_shapes):
            name = input_name.split(';')[j]
            print(name)
            data[name] = np.ones(shape, dtype=np.float32)
    return data

def main(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        if not os.path.exists("data"):
            os.makedirs("data")
        for i, line in enumerate(lines):
            input_shape, input_name, calibrate_dataset_name = parse_line(line)
            # print(input_name)
            # print(input_shape)
            # print(calibrate_dataset_name)
            if input_shape:
                data = generate_data([input_shape], [input_name])
                np.savez(os.path.join("data", f"{calibrate_dataset_name}.npz"), **data)

if __name__ == "__main__":
    filename = "../npuCutInstruction.txt"  # 请将文件路径替换为实际的txt文件路径
    main(filename)