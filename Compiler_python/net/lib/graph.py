class Node:
    def __init__(self, index, op_type, inputs, outputs):
        self.index = index
        self.op_type = op_type
        self.inputs = inputs
        self.outputs = outputs

class MyGraph:
    def __init__(self):
        self.nodes = []
        self.graphinput = []
        self.graphoutput = []

    def add_node(self, node):
        self.nodes.append(node)

    def add_graphinput(self, tensor_name, tensor_shape, tensor_index):
        self.graphinput.append({'name': tensor_name, 'shape': tensor_shape, 'index': tensor_index})

    def add_graphoutput(self, tensor_name, tensor_shape, tensor_index):
        self.graphoutput.append({'name': tensor_name, 'shape': tensor_shape, 'index': tensor_index})

    def __str__(self):
        output = "Graph Inputs:\n"
        output += '\n'.join(f"  {i['name']} (Shape: {i['shape']}, Index: {i['index']})" for i in self.graphinput)
        
        output += "\nGraph Outputs:\n"
        output += '\n'.join(f"  {o['name']} (Shape: {o['shape']}, Index: {o['index']})" for o in self.graphoutput)
        
        output += "\nNodes:\n"
        for node in self.nodes:
            inputs_str = ', '.join(f"{i['name']} (Shape: {i['shape']}, Index: {i['index']})" for i in node.inputs)
            outputs_str = ', '.join(f"{o['name']} (Shape: {o['shape']}, Index: {o['index']})" for o in node.outputs)
            output += f"Node {node.index}:\n"
            output += f"  Operation Type: {node.op_type}\n"
            output += f"  Inputs: {inputs_str}\n"
            output += f"  Outputs: {outputs_str}\n"
        return output