import re
import onnx
import os

def rename_node_io(file_path):
    model = onnx.load(file_path)
    graph = model.graph
    for inputs in graph.input :
        inputs.name = re.sub(r'[/.]','',inputs.name)
    for outputs in graph.output :
        outputs.name = re.sub(r'[/.]','',outputs.name)
    for value_infos in graph.value_info :
        value_infos.name = re.sub(r'[/.]','',value_infos.name)
    for initializers in graph.initializer :
        initializers.name = re.sub(r'[/.]','',initializers.name)
    for node in graph.node:
        node.name = re.sub(r'[/.]','',node.name)
        for i in range(len(node.input)):
            node.input[i] = re.sub(r'[/.]','',node.input[i])
        for i in range(len(node.output)):
            node.output[i] = re.sub(r'[/.]','',node.output[i])
    return model

def rename_subgraph_node_ios(in_file_path,out_file_path):
    file_names = os.listdir(in_file_path)
    for filename in file_names:
        filename_=in_file_path+'/'+filename
        model=rename_node_io(filename_)
        output_file_path = out_file_path+'/'+filename
        onnx.save(model, output_file_path)
        print(f'Modified model saved to {output_file_path}')

directory_path = 'path/to/your/models'
output_path = 'output_path/to/your/models'
rename_subgraph_node_ios(directory_path, output_path)
