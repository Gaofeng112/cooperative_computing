import tensorflow as tf
from tensorflow.lite.python.schema_py_generated import Model
from tensorflow.lite.python import schema_py_generated as schema_fb
from lib.graph import Node, MyGraph
from lib.Device.device import Device
import lib.partion as partion

tflite_model_path = '/mnt/d/docs/github/NPU_CPU_COCOMPUTE_GR/Compiler/net/converted_diffusion_model.tflite'
strategy = "SPLIT_NPU_STRUCTURE_FIRST"

def BuiltinCodeToName(code):
    """Converts a builtin op code enum to a readable name."""
    for name, value in schema_fb.BuiltinOperator.__dict__.items():
        if value == code:
            return name
    return "UNKNOWN"

with open(tflite_model_path, 'rb') as f:
    buf = f.read()

interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

# 获取所有张量的详细信息
tensor_details = interpreter.get_tensor_details()

# 创建张量名称到形状的映射
tensor_name_to_shape = {tensor['name']: tensor['shape'] for tensor in tensor_details}

# 解析模型
model = Model.GetRootAsModel(buf, 0)
subgraph = model.Subgraphs(0)

# 创建图
my_graph = MyGraph()
device = Device()
all_input_name = []
all_output_name = []

# 获取所有操作节点的信息
for i in range(subgraph.OperatorsLength()):
    op = subgraph.Operators(i)
    op_code_index = op.OpcodeIndex()
    op_code = model.OperatorCodes(op_code_index).BuiltinCode()

    # 使用映射来获取操作类型
    op_type = BuiltinCodeToName(op_code)

    # 获取操作的输入和输出张量名称及形状
    inputs = []
    for j in range(op.InputsLength()):
        tensor_index = op.Inputs(j)
        print(tensor_index)
        if tensor_index != -1:
            tensor_name = subgraph.Tensors(tensor_index).Name().decode('utf-8') if subgraph.Tensors(tensor_index).Name() else "Unnamed"
            tensor_shape = tensor_name_to_shape.get(tensor_name, "Unknown Shape")
        else:
            tensor_name = "Unnamed" + str(tensor_index)
            print(tensor_name)
            tensor_shape = "Unknown Shape"
        inputs.append({'name': tensor_name, 'shape': tensor_shape, 'index': tensor_index})
        all_input_name.append(tensor_name)

    outputs = []
    for j in range(op.OutputsLength()):
        tensor_index = op.Outputs(j)
        tensor_name = subgraph.Tensors(tensor_index).Name().decode('utf-8') if subgraph.Tensors(tensor_index).Name() else "Unnamed"
        tensor_shape = tensor_name_to_shape.get(tensor_name, "Unknown Shape")
        outputs.append({'name': tensor_name, 'shape': tensor_shape, 'index': tensor_index})
        all_output_name.append(tensor_name)

    node = Node(i, op_type, inputs, outputs)
    my_graph.add_node(node)

print(my_graph)

all_input_name = [name for name in all_input_name if name in all_output_name]

for input_detail in interpreter.get_input_details():
    all_input_name.append(input_detail['name'])

entire_graph_output_name = []
for output_detail in interpreter.get_output_details():
    entire_graph_output_name.append(output_detail['name'])

Subgraphs = partion.find_and_print_structures(my_graph, device, strategy)
otherSubgraphs = partion.find_other_subgraphs(my_graph, Subgraphs)

subgarphs_input = partion.process_subgraph_input(Subgraphs, all_input_name)
othersubgarphs_input = partion.process_subgraph_input(otherSubgraphs, all_input_name)

partion.process_subgraph_output(Subgraphs, otherSubgraphs, subgarphs_input + othersubgarphs_input, entire_graph_output_name)

print(len(otherSubgraphs))
for i in range(len(otherSubgraphs)):
    print("----------------------------",i)
    print(otherSubgraphs[i])
    if i==3:
        break

print(len(Subgraphs))
for i in range(len(Subgraphs)):
    print("----------------------------",i)
    print(Subgraphs[i])

partion.order_subgraph(Subgraphs, otherSubgraphs, strategy)