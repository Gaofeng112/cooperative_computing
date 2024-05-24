import onnx
import numpy as np

# 加载 ONNX 模型
model_path = "test.onnx"
onnx_model = onnx.load(model_path)

# 定义目标节点的名称和输入名称
target_node_name = "Concat_24"
input_name = "vit_180"

# 找到目标节点
target_node = None
for node in onnx_model.graph.node:
    if node.name == target_node_name:
        target_node = node
        break

if target_node is None:
    print("Target node not found.")
    exit()

# 找到目标节点的输入索引
input_index = -1
for i, input in enumerate(target_node.input):
    if input == input_name:
        input_index = i
        break

if input_index == -1:
    print("Input not found in target node.")
    exit()

# 获取输入张量的维度和数据类型
input_type = onnx_model.graph.input[input_index].type
input_shape = [dim.dim_value for dim in input_type.tensor_type.shape.dim]

# 获取目标节点输入的值
input_values = None
for initializer in onnx_model.graph.initializer:
    if initializer.name == input_name:
        input_values = np.frombuffer(initializer.raw_data, dtype=np.float32)
        break

if input_values is None:
    print("Input values not found in initializers.")
    exit()

# 将输入值保存为二进制文件
output_path = "vit_180.bin"
with open(output_path, "wb") as f:
    f.write(input_values.tobytes())

print("Input values saved to:", output_path)
