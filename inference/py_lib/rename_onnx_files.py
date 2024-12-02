import os
import re

def rename_files(directory):
    # 定义正则表达式模式
    cpu_pattern = re.compile(r'^CPUsubgraph(\d{2,3})\.onnx$')

    # 遍历指定目录下的所有文件
    for filename in os.listdir(directory):
        if filename.endswith('.onnx'):
            match = cpu_pattern.match(filename)
            if match:
                number = int(match.group(1))
                
                if 0 <= number <= 105:
                    # 构造新的文件名
                    new_filename = f'CPUsubgraph{number}.onnx'
                    # 重命名文件
                    old_path = os.path.join(directory, filename)
                    new_path = os.path.join(directory, new_filename)
                    os.rename(old_path, new_path)
                    print(f'Renamed: {filename} -> {new_filename}')
                else:
                    print(f'Skipped: {filename} (Reason: Number not in range 0-105)')
            else:
                print(f'Skipped: {filename} (Reason: Does not match CPUsubgraphXX.onnx pattern)')

if __name__ == '__main__':
    directory = '../1016_subgraphs'  # 目标目录
    rename_files(directory)


# def rename_files(directory):
#     for filename in os.listdir(directory):
#         if filename.endswith(".onnx"):
#             old_name = filename
#             name_parts = filename.split("_")

#             # 提取子图编号部分
#             subgraph_index = name_parts[1].split(".")[0]

#             # 删除子图编号部分的前导零
#             while subgraph_index.startswith('0') and len(subgraph_index) > 1:
#                 subgraph_index = subgraph_index[1:]

#             name_parts[1] = subgraph_index + ".onnx"
#             new_name = "_".join(name_parts)
#             old_path = os.path.join(directory, old_name)
#             new_path = os.path.join(directory, new_name)

#             # 检查新文件名是否存在
#             if not os.path.exists(new_path):
#                 os.rename(old_path, new_path)
#                 print(f"Renamed {old_name} to {new_name}")

# directory_to_process = "../1016_subgraphs"
# rename_files(directory_to_process)

# import os
# import re

# def rename_files(directory):
#     # 定义正则表达式模式
#     cpu_pattern = re.compile(r'^CPUsubgraph(\d{2,3})\.onnx$')

#     # 遍历指定目录下的所有文件
#     for filename in os.listdir(directory):
#         if filename.endswith('.onnx'):
#             match = cpu_pattern.match(filename)
#             if match:
#                 number = int(match.group(1))
#                 # 检查序号是否在57到84之间
#                 if 57 <= number <= 84:
#                     # 构造新的文件名
#                     new_filename = f'NPUsubgraph{number}.onnx'
#                     # 重命名文件
#                     old_path = os.path.join(directory, filename)
#                     new_path = os.path.join(directory, new_filename)
#                     os.rename(old_path, new_path)
#                     print(f'Renamed: {filename} -> {new_filename}')
#                 else:
#                     print(f'Skipped: {filename} (Reason: Number not in range 57-84)')
#             else:
#                 print(f'Skipped: {filename} (Reason: Does not match CPUsubgraphXX.onnx pattern)')

# if __name__ == '__main__':
#     directory = '../1016_subgraphs'  # 目标目录
#     rename_files(directory)