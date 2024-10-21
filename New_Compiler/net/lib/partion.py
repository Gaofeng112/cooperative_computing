from .graph import Node, MyGraph
from .Device.device import Device

Subgraphs = []

def get_enabled_structures(support_op, structure):
    enable_structure = []

    for ops in structure:
        enabled = all(op in support_op for op in ops)
        if enabled:
            enable_structure.append(ops)
    
    return enable_structure

def check_and_print_structure(graph, start_node_index, structure):
    find_flag = False
    next_node_index = start_node_index

    for seq in structure:
        # print(seq)
        structure_index = 0
        current_node_index = start_node_index
        structure_start_node_index = start_node_index

        while current_node_index < len(graph.nodes):
            node = graph.nodes[current_node_index]
            if structure_index >= len(seq):
                # Create subgraph and add nodes to it
                subgraph = MyGraph()
                for i in range(structure_start_node_index, current_node_index):
                    subgraph.add_node(graph.nodes[i])

                Subgraphs.append(subgraph)
                find_flag = True
                next_node_index = current_node_index - 1
                break
            # print(node.op_type,seq[structure_index])
            if node.op_type == seq[structure_index]:
                structure_index += 1
                if structure_index == 1:
                    structure_start_node_index = current_node_index
            else:
                break

            current_node_index += 1

            if structure_index >= len(seq) and current_node_index == len(graph.nodes):
                # Create subgraph and add nodes to it
                subgraph = MyGraph()
                for i in range(structure_start_node_index, current_node_index):
                    subgraph.add_node(graph.nodes[i])

                Subgraphs.append(subgraph)
                find_flag = True
                next_node_index = current_node_index - 1
                break

        if find_flag:
            break

    return next_node_index

def find_and_print_structures(subgraph, device, strategy):
    if strategy == "SPLIT_CPU_STRUCTURE_FIRST":
        enable_cpu_structure = get_enabled_structures(device.get_cpu_support_op(), device.get_cpu_structure())
        
        while i < len(subgraph.nodes):
            # print(i,subgraph.nodes[i].op_type)
            k = check_and_print_structure(subgraph, i, enable_cpu_structure)
            i = k + 1

        return Subgraphs
    elif strategy == "SPLIT_NPU_STRUCTURE_FIRST":
        enable_npu_structure = get_enabled_structures(device.get_npu_support_op(), device.get_npu_structure())

        i = 0
        while i < len(subgraph.nodes):
            # print(i,subgraph.nodes[i].op_type)
            k = check_and_print_structure(subgraph, i, enable_npu_structure)
            i = k + 1

        # print(len(Subgraphs))
        # for i in range(len(Subgraphs)):
        #     print("----------------------------",i)
        #     print(Subgraphs[i])
        return Subgraphs

def find_other_subgraphs(original_graph, known_subgraphs):
# def find_other_subgraphs(original_graph: MyGraph, known_subgraphs: List[MyGraph]) -> List[MyGraph]:
    other_subgraphs = []
    known_subgraph_node_names = set()

    # Collect node names from known subgraphs
    for subgraph in known_subgraphs:
        # print("-------------------")
        for node in subgraph.nodes:
            # print(node.index)
            known_subgraph_node_names.add(node.index)  # Assuming 'name' is the index in the node

    # print(original_graph_node_names)
    start_index = 0
    end_index = -1

    for i, node in enumerate(original_graph.nodes):
        if node.index not in known_subgraph_node_names:
            end_index = i
        else:
            if end_index >= start_index:
                new_subgraph = MyGraph()
                for j in range(start_index, end_index + 1):
                    new_subgraph.add_node(original_graph.nodes[j])
                other_subgraphs.append(new_subgraph)

            start_index = i + 1

    if start_index < len(original_graph.nodes):
        last_subgraph = MyGraph()
        for j in range(start_index, len(original_graph.nodes)):
            last_subgraph.add_node(original_graph.nodes[j])
        other_subgraphs.append(last_subgraph)

    return other_subgraphs

def process_subgraph(subgraph, all_input_name):
    # Collect all outputs
    all_outputs = {o['name'] for node in subgraph.nodes for o in node.outputs}
    all_inputs = {i['name'] for node in subgraph.nodes for i in node.inputs}

    # Collect inputs that are not outputs but are in all_input_name
    for node in subgraph.nodes:
        for input in node.inputs:
            if input['name'] in all_input_name and input['name'] not in all_outputs:
                subgraph.add_graphinput(input['name'], input['shape'], input['index'])

    added_inputs = set()
    unique_graphinputs = []
    for input in subgraph.graphinput:
        # 创建一个不可变的表示形式来用于集合
        if input['name'] not in added_inputs:
            added_inputs.add(input['name'])
            unique_graphinputs.append(input)

    subgraph.graphinput = unique_graphinputs

    # Collect outputs from subgraph nodes
    for node in subgraph.nodes:
        for output in node.outputs:
            if output['name'] not in all_inputs:
                subgraph.add_graphoutput(output['name'], output['shape'], output['index'])

    added_outputs = set()
    unique_graphoutputs = []
    for output in subgraph.graphoutput:
        # 创建一个不可变的表示形式来用于集合
        if output['name'] not in added_outputs:
            added_outputs.add(output['name'])
            unique_graphoutputs.append(output)

    subgraph.graphoutput = unique_graphoutputs

def process_subgraph_input(Subgraphs, all_input_name):
    all_subgraph_input = []
    for subgraph in Subgraphs:
        # Collect all outputs
        all_outputs = {o['name'] for node in subgraph.nodes for o in node.outputs}

        # Collect inputs that are not outputs but are in all_input_name
        for node in subgraph.nodes:
            for input in node.inputs:
                if input['name'] in all_input_name and input['name'] not in all_outputs:
                    subgraph.add_graphinput(input['name'], input['shape'], input['index'])

        added_inputs = set()
        unique_graphinputs = []
        for input in subgraph.graphinput:
            # 创建一个不可变的表示形式来用于集合
            if input['name'] not in added_inputs:
                added_inputs.add(input['name'])
                unique_graphinputs.append(input)

        subgraph.graphinput = unique_graphinputs
        all_subgraph_input.extend(subgraph.graphinput)
    return all_subgraph_input

def process_subgraph_output(Subgraphs, otherSubgraphs, all_subgraph_input, entire_graph_output_name):
    for subgraph in Subgraphs:
        for node in subgraph.nodes:
            # 遍历 subgraph 的 nodes.outputs 列表
            for output in node.outputs:
                # 遍历 all_subgraph_input 查找匹配的项
                for input_item in all_subgraph_input:
                    # 如果 output 的 name 在 all_subgraph_input 的 name 中，则添加 graph output
                    if output['name'] == input_item['name']:
                        subgraph.add_graphoutput(output['name'], output['shape'], output['index'])
                        break  # 假设一个 output 对应一个 input，找到匹配后可以退出循环
                for o in entire_graph_output_name:
                    if o == output['name']:
                        subgraph.add_graphoutput(output['name'], output['shape'], output['index'])
                        break  # 假设一个 output 对应一个 input，找到匹配后可以退出循环

        added_outputs = set()
        unique_graphoutputs = []
        for output in subgraph.graphoutput:
            # 创建一个不可变的表示形式来用于集合
            if output['name'] not in added_outputs:
                added_outputs.add(output['name'])
                unique_graphoutputs.append(output)

        subgraph.graphoutput = unique_graphoutputs

    for subgraph in otherSubgraphs:
        for node in subgraph.nodes:
            # 遍历 subgraph 的 nodes.outputs 列表
            for output in node.outputs:
                # 遍历 all_subgraph_input 查找匹配的项
                for input_item in all_subgraph_input:
                    # 如果 output 的 name 在 all_subgraph_input 的 name 中，则添加 graph output
                    if output['name'] == input_item['name']:
                        subgraph.add_graphoutput(output['name'], output['shape'], output['index'])
                        break  # 假设一个 output 对应一个 input，找到匹配后可以退出循环
                for o in entire_graph_output_name:
                    if o == output['name']:
                        subgraph.add_graphoutput(output['name'], output['shape'], output['index'])
                        break  # 假设一个 output 对应一个 input，找到匹配后可以退出循环

        added_outputs = set()
        unique_graphoutputs = []
        for output in subgraph.graphoutput:
            # 创建一个不可变的表示形式来用于集合
            if output['name'] not in added_outputs:
                added_outputs.add(output['name'])
                unique_graphoutputs.append(output)

        subgraph.graphoutput = unique_graphoutputs

def order_subgraph(Subgraphs, otherSubgraphs, strategy):
    subgraphs_size = len(Subgraphs)
    othersubgraph_size = len(otherSubgraphs)
    totalsubgraph = Subgraphs + otherSubgraphs

    order_subgraphs = [0] * len(totalsubgraph)
    issort_subgraphs = [0] * len(totalsubgraph)
    predecessors_subgraphs = [[] for _ in range(len(totalsubgraph))]
    successors_subgraphs = [[] for _ in range(len(totalsubgraph))]

    finished_flag = False
    sort_count = 0
    while not finished_flag:
        finished_flag = True
        if sort_count == 0:
            for i in range(len(totalsubgraph)):
                find_flag = False
                for g_input in totalsubgraph[i].graphinput:
                    for j in range(len(totalsubgraph)):
                        if any(g_input == item for item in totalsubgraph[j].graphoutput):
                            find_flag = True
                            break
                    if not find_flag:
                        break
                if not find_flag:
                    order_subgraphs[i] = 0
                    issort_subgraphs[i] = 1
                else:
                    order_subgraphs[i] = 1
                    issort_subgraphs[i] = 0
                    finished_flag = False
        else:
            for i in range(len(totalsubgraph)):
                find_flag = False
                predecessors = []
                
                if issort_subgraphs[i] == 1:
                    continue
                
                for g_input in totalsubgraph[i].graphinput:
                    for j in range(len(totalsubgraph)):
                        if any(g_input == item for item in totalsubgraph[j].graphoutput):
                            if issort_subgraphs[j] == 0:
                                find_flag = True
                                break
                            if j not in predecessors:
                                predecessors.append(j)
                    if find_flag:
                        break
                
                if not find_flag:
                    order_subgraphs[i] = sort_count
                    predecessors_subgraphs[i].extend(predecessors)
                else:
                    order_subgraphs[i] = sort_count + 1
                    issort_subgraphs[i] = 0
                    finished_flag = False
                
                if i == len(totalsubgraph) - 1:
                    for j in range(len(totalsubgraph)):
                        if order_subgraphs[j] == sort_count:
                            issort_subgraphs[j] = 1
        
        sort_count += 1

    for i in range(len(totalsubgraph)):
        for j in range(len(totalsubgraph)):
            if i in predecessors_subgraphs[j]:
                successors_subgraphs[i].append(j)

    if strategy == 'SPILTE_CPU_STRUCTURE_FIRST':
        sub1_type = 'CPU'
        sub2_type = 'NPU'
    else:
        sub1_type = 'NPU'
        sub2_type = 'CPU'

    file_name = "subgraphs_ios.txt"

    # Open the file for writing
    with open(file_name, 'w') as outfile:
        for i in range(len(totalsubgraph)):
            # Determine subgraph type and index
            subgraph_type = sub2_type if i >= subgraphs_size else sub1_type
            subgraph_index = i - subgraphs_size if i >= subgraphs_size else i
            
            # Write subgraph details to the file
            outfile.write(f"{subgraph_type}subgraph{subgraph_index}: order{order_subgraphs[i]}\n")
            outfile.write("--input-name ")
            
            # Print to console
            print(f"{subgraph_type}subgraph{subgraph_index}: order{order_subgraphs[i]}")
            print("Inputs:", end=' ')
            
            # Write and print input details
            for element in totalsubgraph[i].graphinput:
                print(f"{element['name']}; size:", end=' ')
                outfile.write(f"{element['name']}; ")
                outfile.write(" ".join(map(str, element['shape'])))
                outfile.write(" ")
                outfile.write("index: ")
                outfile.write(f"{element['index']}; ")
                print(" ".join(map(str, element['shape'])), end=' ')
            
            print()  # Newline for inputs
            outfile.write("\n--output-name ")
            
            # Print output details
            print("Outputs:", end=' ')
            for element in totalsubgraph[i].graphoutput:
                print(f"{element['name']}; size:", end=' ')
                outfile.write(f"{element['name']}; ")
                outfile.write(" ".join(map(str, element['shape'])))
                outfile.write(" ")
                outfile.write("index: ")
                outfile.write(f"{element['index']}; ")
                print(" ".join(map(str, element['shape'])), end=' ')
            
            print()  # Newline for outputs
            outfile.write("\n")
            
            # Write predecessors
            outfile.write(f"The predecessors of {subgraph_type}subgraph{subgraph_index}: ")
            for element in predecessors_subgraphs[i]:
                pred_type = sub2_type if element >= subgraphs_size else sub1_type
                pred_index = element - subgraphs_size if element >= subgraphs_size else element
                outfile.write(f"{pred_type}subgraph{pred_index}; ")
            outfile.write("\n")
            
            # Write successors
            outfile.write(f"The successors of {subgraph_type}subgraph{subgraph_index}: ")

            for element in successors_subgraphs[i]:
                succ_type = sub2_type if element >= subgraphs_size else sub1_type
                succ_index = element - subgraphs_size if element >= subgraphs_size else element
                outfile.write(f"{succ_type}subgraph{succ_index}; ")
            outfile.write("\n")

    print("Data written to", file_name)
