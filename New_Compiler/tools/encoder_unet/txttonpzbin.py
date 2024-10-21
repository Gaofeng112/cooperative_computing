import numpy as np
import re

def parse_shape_from_filename(filename):
    pattern = re.compile(r'(npu|cpu)_Subgraphs_\d+.bin_(output\d+)_(\d+)_(\d+)_(\d+)\.txt')
    match = pattern.match(filename)
    if match:
        shape = tuple(map(int, match.groups()[2:]))
        return shape
    else:
        pattern = re.compile(r'(npu|cpu)_Subgraphs_\d+.bin_(output\d+)_(\d+)_(\d+)_(\d+)_(\d+)\.txt')
        match = pattern.match(filename)
        if match:
            shape = tuple(map(int, match.groups()[2:]))
            return shape
        else:
            pattern = re.compile(r'(npu|cpu)_Subgraphs_\d+.bin_(output\d+)_(\d+)_(\d+)\.txt')
            match = pattern.match(filename)
            if match:
                shape = tuple(map(int, match.groups()[2:]))
                return shape
            else:
                raise ValueError("Filename does not match the expected pattern")

# def instance_normalization(data_array):
#     # Assuming data_array shape is [batch_size, channels, length]
#     mean = data_array.mean(axis=(1, 2), keepdims=True)
#     std = data_array.std(axis=(1, 2), keepdims=True)
#     normalized_array = (data_array - mean) / (std + 1e-5)  # Adding epsilon to avoid division by zero
#     return normalized_array

class InstanceNormalization:
    def __init__(self, channels):
        self.gamma = np.ones((1, channels, 1), dtype=np.float32)  # 初始化缩放参数
        self.beta = np.zeros((1, channels, 1), dtype=np.float32)  # 初始化偏移参数

    def __call__(self, data_array):
        # 假设 data_array 的形状是 [batch_size, channels, length]
        mean = data_array.mean(axis=(1, 2), keepdims=True)
        std = data_array.std(axis=(1, 2), keepdims=True)
        normalized_array = (data_array - mean) / (std + 1e-5)  # 加上一个小的 epsilon 防止除以零
        normalized_array = normalized_array * self.gamma + self.beta  # 应用缩放和偏移
        return normalized_array

def txt_to_npz_and_bin(txt_filename, npz_filename, bin_filename, data_name, normalize=False):
    shape = parse_shape_from_filename(txt_filename)
    txt_filename = './txt/' + txt_filename

    # Read the data from the txt file
    with open(txt_filename, 'r') as file:
        data = [float(line.strip()) for line in file]

    # Convert the list to a NumPy array and reshape it
    data_array = np.array(data, dtype=np.float32).reshape(shape)

    # Apply Instance Normalization if the option is enabled
    if normalize:
        instance_norm = InstanceNormalization(32)
        data_array = instance_norm(data_array)

    # Create the dictionary with the specified key
    data = {data_name: data_array}

    # Save the dictionary to a .npz file
    np.savez(npz_filename, **data)

    # # Save the data array to a .bin file
    # data_array.tofile(bin_filename)

def txt_to_npz_and_bin_with_timestep(txt_filename, npz_filename, bin_filename, data_name):
    shape = parse_shape_from_filename(txt_filename)
    txt_filename = './txt/' + txt_filename

    # Read the data from the txt file
    with open(txt_filename, 'r') as file:
        data = [float(line.strip()) for line in file]

    # Convert the list to a NumPy array and reshape it
    data_array = np.array(data, dtype=np.float32).reshape(shape)

    t = np.ones([1], dtype=np.float32)
    t[0] = 256
    # Create the dictionary with the specified key
    data = {data_name: data_array, "timestep": t}

    # Save the dictionary to a .npz file
    np.savez(npz_filename, **data)

    # # Save the data array to a .bin file
    # data_array.tofile(bin_filename)
    # t.tofile('./data/timestep.bin')

def two_txt_to_npz_and_bin(txt_filename_1, txt_filename_2, npz_filename, bin_filename, data_name_1, data_name_2, normalize=False):
    shape = parse_shape_from_filename(txt_filename_1)
    txt_filename_1 = './txt/' + txt_filename_1

    # Read the data from the txt file
    with open(txt_filename_1, 'r') as file:
        data = [float(line.strip()) for line in file]

    # Convert the list to a NumPy array and reshape it
    data_array = np.array(data, dtype=np.float32).reshape(shape)

    # Apply Instance Normalization if the option is enabled
    if normalize:
        instance_norm = InstanceNormalization(32)
        data_array = instance_norm(data_array)

    shape_2 = parse_shape_from_filename(txt_filename_2)
    txt_filename_2 = './txt/' + txt_filename_2

    # Read the data from the txt file
    with open(txt_filename_2, 'r') as file:
        data = [float(line.strip()) for line in file]

    # Convert the list to a NumPy array and reshape it
    data_array_2 = np.array(data, dtype=np.float32).reshape(shape_2)

    # Create the dictionary with the specified key
    data = {data_name_1: data_array, data_name_2: data_array_2}

    # Save the dictionary to a .npz file
    np.savez(npz_filename, **data)

    # # Save the data array to a .bin file
    # data_array.tofile(bin_filename)

def two_txt_to_npz_and_bin_2(txt_filename_1, txt_filename_2, npz_filename, bin_filename, data_name_1, data_name_2, normalize=False):
    shape = parse_shape_from_filename(txt_filename_1)
    txt_filename_1 = './txt/' + txt_filename_1

    # Read the data from the txt file
    with open(txt_filename_1, 'r') as file:
        data = [float(line.strip()) for line in file]

    # Convert the list to a NumPy array and reshape it
    data_array = np.array(data, dtype=np.float32).reshape(shape)

    # Apply Instance Normalization if the option is enabled
    if normalize:
        instance_norm = InstanceNormalization(32)
        data_array = instance_norm(data_array)

    shape_2 = parse_shape_from_filename(txt_filename_2)
    txt_filename_2 = './txt/' + txt_filename_2

    # Read the data from the txt file
    with open(txt_filename_2, 'r') as file:
        data = [float(line.strip()) for line in file]

    # Convert the list to a NumPy array and reshape it
    data_array_2 = np.array(data, dtype=np.float32).reshape(shape_2)

    # Create the dictionary with the specified key
    data = {data_name_2: data_array_2, data_name_1: data_array}

    # Save the dictionary to a .npz file
    np.savez(npz_filename, **data)

    # # Save the data array to a .bin file
    # data_array.tofile(bin_filename)

def txt_to_npz_and_bin_with_encoder_hidden_states(txt_filename, npz_filename, bin_filename, data_name):
    shape = parse_shape_from_filename(txt_filename)
    txt_filename = './txt/' + txt_filename

    # Read the data from the txt file
    with open(txt_filename, 'r') as file:
        data = [float(line.strip()) for line in file]

    # Convert the list to a NumPy array and reshape it
    data_array = np.array(data, dtype=np.float32).reshape(shape)

    # Define the path to your binary file
    binary_file_path = 'encoder_hidden_states.bin'
    # Read the binary file into a NumPy array
    data_array_encoder = np.fromfile(binary_file_path, dtype=np.float32)  # or another dtype if necessary
    # Reshape the array to the desired shape
    desired_shape = (1, 77, 768)
    data_array_encoder = data_array_encoder.reshape(desired_shape)

    # Create the dictionary with the specified key
    data = {"encoder_hidden_states": data_array_encoder, data_name: data_array}

    # Save the dictionary to a .npz file
    np.savez(npz_filename, **data)

    # # Save the data array to a .bin file
    # data_array.tofile(bin_filename)
    # t.tofile('./data/timestep.bin')

def concat_and_InstanceNormalization(txt_filename_1, txt_filename_2):
    shape = parse_shape_from_filename(txt_filename_1)
    txt_filename_1 = './txt/' + txt_filename_1

    # Read the data from the txt file
    with open(txt_filename_1, 'r') as file:
        data = [float(line.strip()) for line in file]

    # Convert the list to a NumPy array and reshape it
    data_array = np.array(data, dtype=np.float32).reshape(shape)

    shape_2 = parse_shape_from_filename(txt_filename_2)
    txt_filename_2 = './txt/' + txt_filename_2

    # Read the data from the txt file
    with open(txt_filename_2, 'r') as file:
        data = [float(line.strip()) for line in file]

    # Convert the list to a NumPy array and reshape it
    data_array_2 = np.array(data, dtype=np.float32).reshape(shape_2)

    # Concatenate the two arrays
    concat_data_array = np.concatenate((data_array, data_array_2), axis=1)

    # Reshape the concatenated array
    # Calculate the third dimension by dividing the total number of elements by 32
    total_elements = concat_data_array.size
    if total_elements % 32 != 0:
        raise ValueError("The total number of elements is not divisible by 32.")
    third_dimension = total_elements // 32

    reshaped_array = concat_data_array.reshape(1, 32, third_dimension)

    instance_norm = InstanceNormalization(32)
    reshape_data_array = instance_norm(reshaped_array)

    return concat_data_array,reshape_data_array

def add_and_InstanceNormalization(txt_filename_1, txt_filename_2):
    shape = parse_shape_from_filename(txt_filename_1)
    txt_filename_1 = './txt/' + txt_filename_1

    # Read the data from the txt file
    with open(txt_filename_1, 'r') as file:
        data = [float(line.strip()) for line in file]

    # Convert the list to a NumPy array and reshape it
    data_array = np.array(data, dtype=np.float32).reshape(shape)

    shape_2 = parse_shape_from_filename(txt_filename_2)
    txt_filename_2 = './txt/' + txt_filename_2

    # Read the data from the txt file
    with open(txt_filename_2, 'r') as file:
        data = [float(line.strip()) for line in file]

    # Convert the list to a NumPy array and reshape it
    data_array_2 = np.array(data, dtype=np.float32).reshape(shape_2)

    # Ensure broadcasting is possible and add the arrays
    if data_array.shape[1] != data_array_2.shape[1] or data_array_2.shape[2:] != (1, 1):
        raise ValueError("Shapes are not compatible for broadcasting")

    # Add the two arrays together
    add_data_array = data_array + data_array_2

    # Reshape the concatenated array
    # Calculate the third dimension by dividing the total number of elements by 32
    total_elements = add_data_array.size
    if total_elements % 32 != 0:
        raise ValueError("The total number of elements is not divisible by 32.")
    third_dimension = total_elements // 32

    reshaped_array = add_data_array.reshape(1, 32, third_dimension)

    instance_norm = InstanceNormalization(32)
    final_data_array = instance_norm(reshaped_array)

    return final_data_array

def one_txt_one_data_to_npz_and_bin(data_array, txt_filename_2, npz_filename, bin_filename, data_name_1, data_name_2):
    shape_2 = parse_shape_from_filename(txt_filename_2)
    txt_filename_2 = './txt/' + txt_filename_2

    # Read the data from the txt file
    with open(txt_filename_2, 'r') as file:
        data = [float(line.strip()) for line in file]

    # Convert the list to a NumPy array and reshape it
    data_array_2 = np.array(data, dtype=np.float32).reshape(shape_2)

    # Create the dictionary with the specified key
    data = {data_name_1: data_array, data_name_2: data_array_2}

    # Save the dictionary to a .npz file
    np.savez(npz_filename, **data)

    # # Save the data array to a .bin file
    # data_array.tofile(bin_filename)

def one_txt_one_data_to_npz_and_bin_2(txt_filename_1, data_array_2, npz_filename, bin_filename, data_name_1, data_name_2, normalize=False):
    shape = parse_shape_from_filename(txt_filename_1)
    txt_filename_1 = './txt/' + txt_filename_1

    # Read the data from the txt file
    with open(txt_filename_1, 'r') as file:
        data = [float(line.strip()) for line in file]

    # Convert the list to a NumPy array and reshape it
    data_array = np.array(data, dtype=np.float32).reshape(shape)

    # Apply Instance Normalization if the option is enabled
    if normalize:
        instance_norm = InstanceNormalization(32)
        data_array = instance_norm(data_array)

    # Create the dictionary with the specified key
    data = {data_name_1: data_array, data_name_2: data_array_2}

    # Save the dictionary to a .npz file
    np.savez(npz_filename, **data)

    # # Save the data array to a .bin file
    # data_array.tofile(bin_filename)

def one_data_to_npz_and_bin(data_array, npz_filename, bin_filename, data_name_1):
    # Create the dictionary with the specified key
    data = {data_name_1: data_array}

    # Save the dictionary to a .npz file
    np.savez(npz_filename, **data)

    # # Save the data array to a .bin file
    # data_array.tofile(bin_filename)

def two_data_to_npz_and_bin(data_array, data_array_2, npz_filename, bin_filename, data_name_1, data_name_2):
    # Create the dictionary with the specified key
    data = {data_name_1: data_array, data_name_2: data_array_2}

    # Save the dictionary to a .npz file
    np.savez(npz_filename, **data)

    # # Save the data array to a .bin file
    # data_array.tofile(bin_filename)

#npu_1
txt_filename = 'npu_Subgraphs_0.bin_output0_1_32_10240.txt'  # or 'npu_0_output0_1_32_10240.txt'
npz_filename = './data/npu_Subgraphs_1.npz'
bin_filename = 'npu_Subgraphs_1.bin'
data_name = "/down_blocks.0/resnets.0/norm1/InstanceNormalization_output_0"
txt_to_npz_and_bin(txt_filename, npz_filename, bin_filename, data_name, normalize=True)  # Set normalize to True or False as needed

#cpu_0
txt_filename = 'npu_Subgraphs_1.bin_output0_1_320_32_32.txt'
npz_filename = './data/cpu_Subgraphs_0.npz'
bin_filename = 'cpu_Subgraphs_0.bin'
data_name = "/down_blocks.0/resnets.0/conv1/Conv_output_0"
txt_to_npz_and_bin_with_timestep(txt_filename, npz_filename, bin_filename, data_name)  # Set normalize to True or False as needed

#npu_2
txt_filename_1 = 'cpu_Subgraphs_0.bin_output0_1_32_10240.txt'
txt_filename_2 = 'npu_Subgraphs_0.bin_output1_1_320_32_32.txt'
npz_filename = './data/npu_Subgraphs_2.npz'
bin_filename = ''
data_name_1 = "/down_blocks.0/resnets.0/norm2/InstanceNormalization_output_0"
data_name_2 = "/conv_in/Conv_output_0"
two_txt_to_npz_and_bin(txt_filename_1, txt_filename_2, npz_filename, bin_filename, data_name_1, data_name_2, normalize=True)

#npu_3
txt_filename = 'npu_Subgraphs_2.bin_output1_1_32_10240.txt'
npz_filename = './data/npu_Subgraphs_3.npz'
bin_filename = ''
data_name = "/down_blocks.0/attentions.0/norm/InstanceNormalization_output_0"
txt_to_npz_and_bin(txt_filename, npz_filename, bin_filename, data_name, normalize=True)  # Set normalize to True or False as needed

#cpu_1
txt_filename = 'npu_Subgraphs_3.bin_output0_1_320_32_32.txt'
npz_filename = './data/cpu_Subgraphs_1.npz'
bin_filename = ''
data_name = "/down_blocks.0/attentions.0/proj_in/Conv_output_0"
txt_to_npz_and_bin_with_encoder_hidden_states(txt_filename, npz_filename, bin_filename, data_name)

#npu_4
txt_filename_1 = 'cpu_Subgraphs_1.bin_output0_1_320_32_32.txt'
txt_filename_2 = 'npu_Subgraphs_2.bin_output0_1_320_32_32.txt'
npz_filename = './data/npu_Subgraphs_4.npz'
bin_filename = ''
data_name_1 = "/down_blocks.0/attentions.0/Transpose_1_output_0"
data_name_2 = "/down_blocks.0/resnets.0/Add_1_output_0"
two_txt_to_npz_and_bin(txt_filename_1, txt_filename_2, npz_filename, bin_filename, data_name_1, data_name_2)

#npu_5
txt_filename_1 = 'npu_Subgraphs_4.bin_output2_1_32_2560.txt'
txt_filename_2 = 'cpu_Subgraphs_0.bin_output1_1_1280.txt'
npz_filename = './data/npu_Subgraphs_5.npz'
bin_filename = ''
data_name_1 = "/down_blocks.1/resnets.0/norm1/InstanceNormalization_output_0"
data_name_2 = "/down_blocks.0/resnets.0/act_1/Mul_output_0"
two_txt_to_npz_and_bin(txt_filename_1, txt_filename_2, npz_filename, bin_filename, data_name_1, data_name_2, normalize=True)

#npu_6
txt_filename_1 = 'npu_Subgraphs_5.bin_output0_1_32_5120.txt'
txt_filename_2 = 'npu_Subgraphs_4.bin_output3_1_640_16_16.txt'
npz_filename = './data/npu_Subgraphs_6.npz'
bin_filename = ''
data_name_1 = "/down_blocks.1/resnets.0/norm2/InstanceNormalization_output_0"
data_name_2 = "/down_blocks.1/resnets.0/conv_shortcut/Conv_output_0"
two_txt_to_npz_and_bin_2(txt_filename_1, txt_filename_2, npz_filename, bin_filename, data_name_1, data_name_2, normalize=True)

#npu_7
txt_filename = 'npu_Subgraphs_6.bin_output0_1_32_5120.txt'
npz_filename = './data/npu_Subgraphs_7.npz'
bin_filename = ''
data_name = "/down_blocks.1/attentions.0/norm/InstanceNormalization_output_0"
txt_to_npz_and_bin(txt_filename, npz_filename, bin_filename, data_name, normalize=True) 

#cpu_2
txt_filename = 'npu_Subgraphs_7.bin_output0_1_640_16_16.txt'
npz_filename = './data/cpu_Subgraphs_2.npz'
bin_filename = ''
data_name = "/down_blocks.1/attentions.0/proj_in/Conv_output_0"
txt_to_npz_and_bin_with_encoder_hidden_states(txt_filename, npz_filename, bin_filename, data_name)

#npu_8
txt_filename_1 = 'cpu_Subgraphs_2.bin_output0_1_640_16_16.txt'
txt_filename_2 = 'npu_Subgraphs_6.bin_output1_1_640_16_16.txt'
npz_filename = './data/npu_Subgraphs_8.npz'
bin_filename = ''
data_name_1 = "/down_blocks.1/attentions.0/Transpose_1_output_0"
data_name_2 = "/down_blocks.1/resnets.0/Add_1_output_0"
two_txt_to_npz_and_bin(txt_filename_1, txt_filename_2, npz_filename, bin_filename, data_name_1, data_name_2)

#npu_9
txt_filename_1 = 'npu_Subgraphs_8.bin_output2_1_32_1280.txt'
txt_filename_2 = 'cpu_Subgraphs_0.bin_output1_1_1280.txt'
npz_filename = './data/npu_Subgraphs_9.npz'
bin_filename = ''
data_name_1 = "/down_blocks.2/resnets.0/norm1/InstanceNormalization_output_0"
data_name_2 = "/down_blocks.0/resnets.0/act_1/Mul_output_0"
two_txt_to_npz_and_bin(txt_filename_1, txt_filename_2, npz_filename, bin_filename, data_name_1, data_name_2, normalize=True)

#npu_10
txt_filename_1 = 'npu_Subgraphs_9.bin_output0_1_32_2560.txt'
txt_filename_2 = 'npu_Subgraphs_8.bin_output1_1_640_8_8.txt'
npz_filename = './data/npu_Subgraphs_10.npz'
bin_filename = ''
data_name_1 = "/down_blocks.2/resnets.0/norm2/InstanceNormalization_output_0"
data_name_2 = "/down_blocks.1/downsamplers.0/conv/Conv_output_0"
two_txt_to_npz_and_bin_2(txt_filename_1, txt_filename_2, npz_filename, bin_filename, data_name_1, data_name_2, normalize=True)

#npu_11
txt_filename = 'npu_Subgraphs_10.bin_output0_1_32_2560.txt'
npz_filename = './data/npu_Subgraphs_11.npz'
bin_filename = ''
data_name = "/down_blocks.2/attentions.0/norm/InstanceNormalization_output_0"
txt_to_npz_and_bin(txt_filename, npz_filename, bin_filename, data_name, normalize=True) 

#cpu_3
txt_filename = 'npu_Subgraphs_11.bin_output0_1_1280_8_8.txt'
npz_filename = './data/cpu_Subgraphs_3.npz'
bin_filename = ''
data_name = "/down_blocks.2/attentions.0/proj_in/Conv_output_0"
txt_to_npz_and_bin_with_encoder_hidden_states(txt_filename, npz_filename, bin_filename, data_name)

#npu_12
txt_filename_1 = 'cpu_Subgraphs_3.bin_output0_1_1280_8_8.txt'
txt_filename_2 = 'npu_Subgraphs_10.bin_output1_1_1280_8_8.txt'
npz_filename = './data/npu_Subgraphs_12.npz'
bin_filename = ''
data_name_1 = "/down_blocks.2/attentions.0/Transpose_1_output_0"
data_name_2 = "/down_blocks.2/resnets.0/Add_1_output_0"
two_txt_to_npz_and_bin(txt_filename_1, txt_filename_2, npz_filename, bin_filename, data_name_1, data_name_2)

#npu_13
txt_filename_1 = 'npu_Subgraphs_12.bin_output0_1_32_5120.txt'
txt_filename_2 = 'cpu_Subgraphs_0.bin_output1_1_1280.txt'
npz_filename = './data/npu_Subgraphs_13.npz'
bin_filename = ''
data_name_1 = "/up_blocks.0/resnets.0/norm1/InstanceNormalization_output_0"
data_name_2 = "/down_blocks.0/resnets.0/act_1/Mul_output_0"
two_txt_to_npz_and_bin(txt_filename_1, txt_filename_2, npz_filename, bin_filename, data_name_1, data_name_2, normalize=True)

#npu_14
txt_filename_1 = 'npu_Subgraphs_13.bin_output0_1_32_2560.txt'
txt_filename_2 = 'npu_Subgraphs_12.bin_output1_1_1280_8_8.txt'
npz_filename = './data/npu_Subgraphs_14.npz'
bin_filename = ''
data_name_1 = "/up_blocks.0/resnets.0/norm2/InstanceNormalization_output_0"
data_name_2 = "/up_blocks.0/resnets.0/conv_shortcut/Conv_output_0"
two_txt_to_npz_and_bin_2(txt_filename_1, txt_filename_2, npz_filename, bin_filename, data_name_1, data_name_2, normalize=True)

#npu_15
txt_filename = 'npu_Subgraphs_14.bin_output0_1_32_2560.txt'
npz_filename = './data/npu_Subgraphs_15.npz'
bin_filename = ''
data_name = "/up_blocks.0/attentions.0/norm/InstanceNormalization_output_0"
txt_to_npz_and_bin(txt_filename, npz_filename, bin_filename, data_name, normalize=True)

#cpu_4
txt_filename = 'npu_Subgraphs_15.bin_output0_1_1280_8_8.txt'
npz_filename = './data/cpu_Subgraphs_4.npz'
bin_filename = ''
data_name = "/up_blocks.0/attentions.0/proj_in/Conv_output_0"
txt_to_npz_and_bin_with_encoder_hidden_states(txt_filename, npz_filename, bin_filename, data_name)

#npu_16
txt_filename_1 = 'cpu_Subgraphs_4.bin_output0_1_1280_8_8.txt'
txt_filename_2 = 'npu_Subgraphs_14.bin_output1_1_1280_8_8.txt'
npz_filename = './data/npu_Subgraphs_16.npz'
bin_filename = ''
data_name_1 = "/up_blocks.0/attentions.0/Transpose_1_output_0"
data_name_2 = "/up_blocks.0/resnets.0/Add_1_output_0"
two_txt_to_npz_and_bin(txt_filename_1, txt_filename_2, npz_filename, bin_filename, data_name_1, data_name_2)


txt_filename_1 = 'npu_Subgraphs_16.bin_output0_1_1280_8_8.txt'
txt_filename_2 = 'npu_Subgraphs_8.bin_output1_1_640_8_8.txt'
concat_data_array, instanceNormalization_data_array = concat_and_InstanceNormalization(txt_filename_1, txt_filename_2)

print("连接后的数据形状:", concat_data_array.shape)
print("归一化后的数据形状:", instanceNormalization_data_array.shape)

#npu_17
txt_filename = 'cpu_Subgraphs_0.bin_output1_1_1280.txt'
npz_filename = './data/npu_Subgraphs_17.npz'
bin_filename = ''
data_name_1 = "/up_blocks.0/resnets.1/norm1/InstanceNormalization_output_0"
data_name_2 = "/down_blocks.0/resnets.0/act_1/Mul_output_0"
one_txt_one_data_to_npz_and_bin(instanceNormalization_data_array, txt_filename, npz_filename, bin_filename, data_name_1, data_name_2)

#npu_18
txt_filename = 'npu_Subgraphs_17.bin_output0_1_32_2560.txt'
npz_filename = './data/npu_Subgraphs_18.npz'
bin_filename = ''
data_name_1 = "/up_blocks.0/resnets.1/norm2/InstanceNormalization_output_0"
data_name_2 = "/up_blocks.0/Concat_1_output_0"
one_txt_one_data_to_npz_and_bin_2(txt_filename, concat_data_array, npz_filename, bin_filename, data_name_1, data_name_2, normalize=True)

#npu_19
txt_filename = 'npu_Subgraphs_18.bin_output0_1_32_2560.txt'
npz_filename = './data/npu_Subgraphs_19.npz'
bin_filename = ''
data_name = "/up_blocks.0/attentions.1/norm/InstanceNormalization_output_0"
txt_to_npz_and_bin(txt_filename, npz_filename, bin_filename, data_name, normalize=True)

#cpu_5
txt_filename = 'npu_Subgraphs_19.bin_output0_1_1280_8_8.txt'
npz_filename = './data/cpu_Subgraphs_5.npz'
bin_filename = ''
data_name = "/up_blocks.0/attentions.1/proj_in/Conv_output_0"
txt_to_npz_and_bin_with_encoder_hidden_states(txt_filename, npz_filename, bin_filename, data_name)

#npu_20
txt_filename_1 = 'cpu_Subgraphs_5.bin_output0_1_1280_8_8.txt'
txt_filename_2 = 'npu_Subgraphs_18.bin_output1_1_1280_8_8.txt'
npz_filename = './data/npu_Subgraphs_20.npz'
bin_filename = ''
data_name_1 = "/up_blocks.0/attentions.1/Transpose_1_output_0"
data_name_2 = "/up_blocks.0/resnets.1/Add_1_output_0"
two_txt_to_npz_and_bin(txt_filename_1, txt_filename_2, npz_filename, bin_filename, data_name_1, data_name_2)

#npu_21
txt_filename = 'cpu_Subgraphs_0.bin_output1_1_1280.txt'
npz_filename = './data/npu_Subgraphs_21.npz'
bin_filename = ''
data_name = "/down_blocks.0/resnets.0/act_1/Mul_output_0"
txt_to_npz_and_bin(txt_filename, npz_filename, bin_filename, data_name) 

txt_filename_1 = 'npu_Subgraphs_20.bin_output0_1_1280_16_16.txt'
txt_filename_2 = 'npu_Subgraphs_8.bin_output0_1_640_16_16.txt'
concat_data_array, instanceNormalization_data_array = concat_and_InstanceNormalization(txt_filename_1, txt_filename_2)

print("连接后的数据形状:", concat_data_array.shape)
print("归一化后的数据形状:", instanceNormalization_data_array.shape)

#npu_22
npz_filename = './data/npu_Subgraphs_22.npz'
bin_filename = ''
data_name_1 = "/up_blocks.1/resnets.0/norm1/InstanceNormalization_output_0"
one_data_to_npz_and_bin(instanceNormalization_data_array, npz_filename, bin_filename, data_name_1)

txt_filename_1 = 'npu_Subgraphs_22.bin_output0_1_640_16_16.txt'
txt_filename_2 = 'npu_Subgraphs_21.bin_output0_1_640_1_1.txt'
instanceNormalization_data_array = add_and_InstanceNormalization(txt_filename_1, txt_filename_2)

#npu_23
npz_filename = './data/npu_Subgraphs_23.npz'
bin_filename = ''
data_name_1 = "/up_blocks.1/Concat_output_0"
data_name_2 = "/up_blocks.1/resnets.0/norm2/InstanceNormalization_output_0"
two_data_to_npz_and_bin(concat_data_array, instanceNormalization_data_array, npz_filename, bin_filename, data_name_1, data_name_2)

#npu_24
txt_filename = 'npu_Subgraphs_23.bin_output0_1_32_5120.txt'
npz_filename = './data/npu_Subgraphs_24.npz'
bin_filename = ''
data_name = "/up_blocks.1/attentions.0/norm/InstanceNormalization_output_0"
txt_to_npz_and_bin(txt_filename, npz_filename, bin_filename, data_name, normalize=True)

#cpu_6
txt_filename = 'npu_Subgraphs_24.bin_output0_1_640_16_16.txt'
npz_filename = './data/cpu_Subgraphs_6.npz'
bin_filename = ''
data_name = "/up_blocks.1/attentions.0/proj_in/Conv_output_0"
txt_to_npz_and_bin_with_encoder_hidden_states(txt_filename, npz_filename, bin_filename, data_name)

#npu_25
txt_filename_1 = 'cpu_Subgraphs_6.bin_output0_1_640_16_16.txt'
txt_filename_2 = 'npu_Subgraphs_23.bin_output1_1_640_16_16.txt'
npz_filename = './data/npu_Subgraphs_25.npz'
bin_filename = ''
data_name_1 = "/up_blocks.1/attentions.0/Transpose_1_output_0"
data_name_2 = "/up_blocks.1/resnets.0/Add_1_output_0"
two_txt_to_npz_and_bin(txt_filename_1, txt_filename_2, npz_filename, bin_filename, data_name_1, data_name_2)

#npu_26
txt_filename = 'cpu_Subgraphs_0.bin_output1_1_1280.txt'
npz_filename = './data/npu_Subgraphs_26.npz'
bin_filename = ''
data_name = "/down_blocks.0/resnets.0/act_1/Mul_output_0"
txt_to_npz_and_bin(txt_filename, npz_filename, bin_filename, data_name) 


txt_filename_1 = 'npu_Subgraphs_25.bin_output0_1_640_16_16.txt'
txt_filename_2 = 'npu_Subgraphs_4.bin_output1_1_320_16_16.txt'
concat_data_array, instanceNormalization_data_array = concat_and_InstanceNormalization(txt_filename_1, txt_filename_2)

print("连接后的数据形状:", concat_data_array.shape)
print("归一化后的数据形状:", instanceNormalization_data_array.shape)

#npu_27
npz_filename = './data/npu_Subgraphs_27.npz'
bin_filename = ''
data_name_1 = "/up_blocks.1/resnets.1/norm1/InstanceNormalization_output_0"
one_data_to_npz_and_bin(instanceNormalization_data_array, npz_filename, bin_filename, data_name_1)

txt_filename_1 = 'npu_Subgraphs_27.bin_output0_1_640_16_16.txt'
txt_filename_2 = 'npu_Subgraphs_26.bin_output0_1_640_1_1.txt'
instanceNormalization_data_array = add_and_InstanceNormalization(txt_filename_1, txt_filename_2)

#npu_28
npz_filename = './data/npu_Subgraphs_28.npz'
bin_filename = ''
data_name_1 = "/up_blocks.1/Concat_1_output_0"
data_name_2 = "/up_blocks.1/resnets.1/norm2/InstanceNormalization_output_0"
two_data_to_npz_and_bin(concat_data_array, instanceNormalization_data_array, npz_filename, bin_filename, data_name_1, data_name_2)

#npu_29
txt_filename = 'npu_Subgraphs_28.bin_output0_1_32_5120.txt'
npz_filename = './data/npu_Subgraphs_29.npz'
bin_filename = ''
data_name = "/up_blocks.1/attentions.1/norm/InstanceNormalization_output_0"
txt_to_npz_and_bin(txt_filename, npz_filename, bin_filename, data_name, normalize=True)

#cpu_7
txt_filename = 'npu_Subgraphs_29.bin_output0_1_640_16_16.txt'
npz_filename = './data/cpu_Subgraphs_7.npz'
bin_filename = ''
data_name = "/up_blocks.1/attentions.1/proj_in/Conv_output_0"
txt_to_npz_and_bin_with_encoder_hidden_states(txt_filename, npz_filename, bin_filename, data_name)

#npu_30
txt_filename_1 = 'cpu_Subgraphs_7.bin_output0_1_640_16_16.txt'
txt_filename_2 = 'npu_Subgraphs_28.bin_output1_1_640_16_16.txt'
npz_filename = './data/npu_Subgraphs_30.npz'
bin_filename = ''
data_name_1 = "/up_blocks.1/attentions.1/Transpose_1_output_0"
data_name_2 = "/up_blocks.1/resnets.1/Add_1_output_0"
two_txt_to_npz_and_bin(txt_filename_1, txt_filename_2, npz_filename, bin_filename, data_name_1, data_name_2)

#npu_31
txt_filename = 'cpu_Subgraphs_0.bin_output1_1_1280.txt'
npz_filename = './data/npu_Subgraphs_31.npz'
bin_filename = ''
data_name = "/down_blocks.0/resnets.0/act_1/Mul_output_0"
txt_to_npz_and_bin(txt_filename, npz_filename, bin_filename, data_name) 

txt_filename_1 = 'npu_Subgraphs_30.bin_output0_1_640_32_32.txt'
txt_filename_2 = 'npu_Subgraphs_4.bin_output0_1_320_32_32.txt'
concat_data_array, instanceNormalization_data_array = concat_and_InstanceNormalization(txt_filename_1, txt_filename_2)

print("连接后的数据形状:", concat_data_array.shape)
print("归一化后的数据形状:", instanceNormalization_data_array.shape)

#npu_32
npz_filename = './data/npu_Subgraphs_32.npz'
bin_filename = ''
data_name_1 = "/up_blocks.2/resnets.0/norm1/InstanceNormalization_output_0"
one_data_to_npz_and_bin(instanceNormalization_data_array, npz_filename, bin_filename, data_name_1)

txt_filename_1 = 'npu_Subgraphs_32.bin_output0_1_320_32_32.txt'
txt_filename_2 = 'npu_Subgraphs_31.bin_output0_1_320_1_1.txt'
instanceNormalization_data_array = add_and_InstanceNormalization(txt_filename_1, txt_filename_2)

#npu_33
npz_filename = './data/npu_Subgraphs_33.npz'
bin_filename = ''
data_name_1 = "/up_blocks.2/Concat_output_0"
data_name_2 = "/up_blocks.2/resnets.0/norm2/InstanceNormalization_output_0"
two_data_to_npz_and_bin(concat_data_array, instanceNormalization_data_array, npz_filename, bin_filename, data_name_1, data_name_2)

#npu_34
txt_filename = 'npu_Subgraphs_33.bin_output0_1_32_10240.txt'
npz_filename = './data/npu_Subgraphs_34.npz'
bin_filename = ''
data_name = "/up_blocks.2/attentions.0/norm/InstanceNormalization_output_0"
txt_to_npz_and_bin(txt_filename, npz_filename, bin_filename, data_name, normalize=True)

#cpu_8
txt_filename = 'npu_Subgraphs_34.bin_output0_1_320_32_32.txt'
npz_filename = './data/cpu_Subgraphs_8.npz'
bin_filename = ''
data_name = "/up_blocks.2/attentions.0/proj_in/Conv_output_0"
txt_to_npz_and_bin_with_encoder_hidden_states(txt_filename, npz_filename, bin_filename, data_name)

#npu_35
txt_filename_1 = 'cpu_Subgraphs_8.bin_output0_1_320_32_32.txt'
txt_filename_2 = 'npu_Subgraphs_33.bin_output1_1_320_32_32.txt'
npz_filename = './data/npu_Subgraphs_35.npz'
bin_filename = ''
data_name_1 = "/up_blocks.2/attentions.0/Transpose_1_output_0"
data_name_2 = "/up_blocks.2/resnets.0/Add_1_output_0"
two_txt_to_npz_and_bin(txt_filename_1, txt_filename_2, npz_filename, bin_filename, data_name_1, data_name_2)

#npu_36
txt_filename = 'cpu_Subgraphs_0.bin_output1_1_1280.txt'
npz_filename = './data/npu_Subgraphs_36.npz'
bin_filename = ''
data_name = "/down_blocks.0/resnets.0/act_1/Mul_output_0"
txt_to_npz_and_bin(txt_filename, npz_filename, bin_filename, data_name) 

txt_filename_1 = 'npu_Subgraphs_35.bin_output0_1_320_32_32.txt'
txt_filename_2 = 'npu_Subgraphs_0.bin_output1_1_320_32_32.txt'
concat_data_array, instanceNormalization_data_array = concat_and_InstanceNormalization(txt_filename_1, txt_filename_2)

print("连接后的数据形状:", concat_data_array.shape)
print("归一化后的数据形状:", instanceNormalization_data_array.shape)

#npu_37
npz_filename = './data/npu_Subgraphs_37.npz'
bin_filename = ''
data_name_1 = "/up_blocks.2/resnets.1/norm1/InstanceNormalization_output_0"
one_data_to_npz_and_bin(instanceNormalization_data_array, npz_filename, bin_filename, data_name_1)

txt_filename_1 = 'npu_Subgraphs_37.bin_output0_1_320_32_32.txt'
txt_filename_2 = 'npu_Subgraphs_36.bin_output0_1_320_1_1.txt'
instanceNormalization_data_array = add_and_InstanceNormalization(txt_filename_1, txt_filename_2)

#npu_38
npz_filename = './data/npu_Subgraphs_38.npz'
bin_filename = ''
data_name_1 = "/up_blocks.2/Concat_1_output_0"
data_name_2 = "/up_blocks.2/resnets.1/norm2/InstanceNormalization_output_0"
two_data_to_npz_and_bin(concat_data_array, instanceNormalization_data_array, npz_filename, bin_filename, data_name_1, data_name_2)

#npu_39
txt_filename = 'npu_Subgraphs_38.bin_output0_1_32_10240.txt'
npz_filename = './data/npu_Subgraphs_39.npz'
bin_filename = ''
data_name = "/up_blocks.2/attentions.1/norm/InstanceNormalization_output_0"
txt_to_npz_and_bin(txt_filename, npz_filename, bin_filename, data_name, normalize=True)

#cpu_9
txt_filename = 'npu_Subgraphs_39.bin_output0_1_320_32_32.txt'
npz_filename = './data/cpu_Subgraphs_9.npz'
bin_filename = ''
data_name = "/up_blocks.2/attentions.1/proj_in/Conv_output_0"
txt_to_npz_and_bin_with_encoder_hidden_states(txt_filename, npz_filename, bin_filename, data_name)

#npu_40
txt_filename_1 = 'cpu_Subgraphs_9.bin_output0_1_320_32_32.txt'
txt_filename_2 = 'npu_Subgraphs_38.bin_output1_1_320_32_32.txt'
npz_filename = './data/npu_Subgraphs_40.npz'
bin_filename = ''
data_name_1 = "/up_blocks.2/attentions.1/Transpose_1_output_0"
data_name_2 = "/up_blocks.2/resnets.1/Add_1_output_0"
two_txt_to_npz_and_bin(txt_filename_1, txt_filename_2, npz_filename, bin_filename, data_name_1, data_name_2)

#npu_41
txt_filename = 'npu_Subgraphs_40.bin_output0_1_32_10240.txt'
npz_filename = './data/npu_Subgraphs_41.npz'
bin_filename = ''
data_name = "/conv_norm_out/InstanceNormalization_output_0"
txt_to_npz_and_bin(txt_filename, npz_filename, bin_filename, data_name, normalize=True)  # Set normalize to True or False as needed