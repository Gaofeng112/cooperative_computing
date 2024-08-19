import json
import numpy as np
import tensorflow as tf
from tensorflow.lite.python.schema_py_generated import Model
from tensorflow.lite.python import schema_py_generated as schema_fb
import sys
import os

sys.path.append(os.path.abspath("net"))
import partition_main

def extract_json(input_path,output_path = None,node_indices = None, inputs_ = None, outputs_ = None):
    with open(input_path) as f:
        net = json.load(f)
    tensor_visited_flag = np.zeros(len(net["subgraphs"][0]["tensors"]), dtype= np.int32)
    tensor_new_index = np.zeros(len(net["subgraphs"][0]["tensors"]), dtype= np.int32)
    subgraph = {}
    subgraph_nodes = []
    subgraph_tensors = []
    subgraph_buffers = []
    count = 0
    for node_index in node_indices:
        node = net["subgraphs"][0]["operators"][node_index]
        updated_node_inputs = []
        for input_index in node["inputs"]:
            if input_index < 0:
                continue
            if(tensor_visited_flag[input_index] == 0):
                subgraph_tensors.append(net["subgraphs"][0]["tensors"][input_index])
                tensor_new_index[input_index] = len(subgraph_tensors) - 1
                tensor_visited_flag[input_index] = 1
            updated_node_inputs.append(int(tensor_new_index[input_index]))
        node["inputs"] = updated_node_inputs

        updated_node_outputs = []
        for output_index in node["outputs"]:
            if(tensor_visited_flag[output_index] == 0):
                subgraph_tensors.append(net["subgraphs"][0]["tensors"][output_index])
                tensor_new_index[output_index] = len(subgraph_tensors) - 1
                tensor_visited_flag[output_index] = 1
            updated_node_outputs.append(int(tensor_new_index[output_index]))
        node["outputs"] = updated_node_outputs
        subgraph_nodes.append(node)
        count = count + 1
    
    buffer_visited_flag = np.zeros(len(net["buffers"]), dtype= np.int32)
    buffer_new_index = np.zeros(len(net["buffers"]), dtype= np.int32)
    for tensor_element in subgraph_tensors:
        if(buffer_visited_flag[tensor_element["buffer"]] == 0):
            subgraph_buffers.append(net["buffers"][tensor_element["buffer"]])
            buffer_new_index[tensor_element["buffer"]] = len(subgraph_buffers) - 1
            buffer_visited_flag[tensor_element["buffer"]] = 1
        tensor_element["buffer"] = int(buffer_new_index[tensor_element["buffer"]])

    subgraph["tensors"] = subgraph_tensors
    inputs = inputs_
    outputs = outputs_
    for i in range(len(inputs)):
        inputs[i] = int(tensor_new_index[inputs_[i]])
    for i in range(len(outputs)):
        outputs[i] = int(tensor_new_index[outputs_[i]])

    subgraph["inputs"] = inputs
    subgraph["outputs"] = outputs
    subgraph["operators"] = subgraph_nodes

    metadata = {}
    metadata["name"] = "subgraph"
    metadata["buffer"] = len(subgraph_buffers)

    output_net = {}
    output_net["version"] = 3
    output_net["operator_codes"] = net["operator_codes"]

    output_net["subgraphs"] = []
    output_net["subgraphs"].append(subgraph)

    output_net["description"] = net["description"]
    output_net["buffers"] = subgraph_buffers
    output_net["metadata"] = []
    output_net["metadata"].append(metadata)
    output_net["signature_defs"] = []
    with open(output_path, 'w') as file:
        json.dump(output_net, file, indent = 4)

for i in range(len(partition_main.Subgraphs)):
    subgraph_name = 'workdir/subgraphs/npusubgraph'+str(i)+'.json'
    node_index = []
    for j in range(len(partition_main.Subgraphs[i].nodes)):
        node_index.append(partition_main.Subgraphs[i].nodes[j].index)
    input = []
    for j in range(len(partition_main.Subgraphs[i].graphinput)):
        input.append(partition_main.Subgraphs[i].graphinput[j]['index'])
    output = []
    for j in range(len(partition_main.Subgraphs[i].graphoutput)):
        output.append(partition_main.Subgraphs[i].graphoutput[j]['index'])
    extract_json('workdir/converted_diffusion_model.json',subgraph_name, node_index, input, output)

for i in range(len(partition_main.otherSubgraphs)):
    subgraph_name = 'workdir/subgraphs/cpusubgraph'+str(i)+'.json'
    node_index = []
    for j in range(len(partition_main.otherSubgraphs[i].nodes)):
        node_index.append(partition_main.otherSubgraphs[i].nodes[j].index)
    input = []
    for j in range(len(partition_main.otherSubgraphs[i].graphinput)):
        input.append(partition_main.otherSubgraphs[i].graphinput[j]['index'])
    output = []
    for j in range(len(partition_main.otherSubgraphs[i].graphoutput)):
        output.append(partition_main.otherSubgraphs[i].graphoutput[j]['index'])
    extract_json('workdir/converted_diffusion_model.json',subgraph_name, node_index, input, output)

print("Extract json done!")