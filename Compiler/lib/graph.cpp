#include "graph.h"

std::unordered_set<NodeTensor> getInitializer(const onnx::GraphProto& graph) {
    std::unordered_set<NodeTensor> initializerNames;
    for (const auto& initializer : graph.initializer()) {
		NodeTensor nt;
		nt.name = initializer.name();
		std::vector<int64_t> shape;
		for (const auto& dim : initializer.dims()) {
			shape.push_back(dim);
		}
		nt.shape = shape;
        initializerNames.insert(nt);
    }
    return initializerNames;
}

std::unordered_set<NodeTensor> getIOvalue(const onnx::GraphProto& graph) {
	std::unordered_set<NodeTensor> IOvalue;
	for (const auto& value_info : graph.value_info()) {
		NodeTensor nt;
		nt.name = value_info.name();
		std::cout << nt.name << std::endl;
		std::vector<int64_t> shape;
		for (const auto& dim : value_info.type().tensor_type().shape().dim()) {
			std::cout << "dim.dim_value():" << dim.dim_value() << std::endl;
			shape.push_back(dim.dim_value());
		}
		nt.shape = shape;
        IOvalue.insert(nt);
	}
	for (auto value_info : graph.input()) {
		NodeTensor nt;
		nt.name = value_info.name();
		std::cout << nt.name << std::endl;
		std::vector<int64_t> shape;
		for (const auto& dim : value_info.type().tensor_type().shape().dim()) {
			shape.push_back(dim.dim_value());
		}
		nt.shape = shape;
        IOvalue.insert(nt);
	}
	for (auto value_info : graph.output()) {
		NodeTensor nt;
		nt.name = value_info.name();
		std::cout << nt.name << std::endl;
		std::vector<int64_t> shape;
		for (const auto& dim : value_info.type().tensor_type().shape().dim()) {
			shape.push_back(dim.dim_value());
		}
		nt.shape = shape;
        IOvalue.insert(nt);
	}
	return IOvalue;
}

std::unordered_set<NodeTensor>::const_iterator isInputFromInitializer(const std::string& name, const std::unordered_set<NodeTensor>& tensors) {
    return std::find_if(tensors.begin(), tensors.end(), [&](const NodeTensor& tensor) { return tensor.name == name; });
}

void determineGraphInputOutput(const onnx::GraphProto& g, const std::unordered_set<NodeTensor>& initializerNames,
    						   std::unordered_set<NodeTensor> &graphInputs, std::unordered_set<NodeTensor> &graphOutputs) {
    std::unordered_set<std::string> allnodeInputs;
    std::unordered_set<std::string> allnodeOutputs;

    for (const auto& node : g.node()) {
        const auto& inputs = node.input();
        const auto& outputs = node.output();

        for (const auto& input : inputs) {
            allnodeInputs.insert(input);
        }

        for (const auto& output : outputs) {
            allnodeOutputs.insert(output);
        }
    }

    for (const auto& node : g.node()) {
        const auto& inputs = node.input();
        const auto& outputs = node.output();

        for (const auto& input : inputs) {
            if (std::find(allnodeOutputs.begin(), allnodeOutputs.end(), input) == allnodeOutputs.end()) {
				auto iter = isInputFromInitializer(input, initializerNames);
				if (iter != initializerNames.end()) {
					graphInputs.insert(*iter);
					std::cout << "Found input tensor with name " << input << " in the set." << std::endl;
				}
            }
        }

        for (const auto& output : outputs) {
            if (std::find(allnodeInputs.begin(), allnodeInputs.end(), output) == allnodeInputs.end()) {
				auto iter = isInputFromInitializer(output, initializerNames);
				if (iter != initializerNames.end()) {
					graphOutputs.insert(*iter);
					std::cout << "Found output tensor with name " << output << " in the set." << std::endl;
				}
            }
        }
    }
}

std::string findInputNode(const onnx::GraphProto &g, const std::string& outputTensorName) {
	std::string node_name = "";
	for (const auto& node : g.node()) {
		for (const auto& output : node.output()) {
			if (output == outputTensorName) {
				node_name = node.name();
			}
		}
	}
    return node_name;
}

std::unordered_set<std::string> collectNodeNames(const onnx::GraphProto& graph) {
    std::unordered_set<std::string> nodeNames;
    for (const auto& node : graph.node()) {
        nodeNames.insert(node.name());
    }
    return nodeNames;
}

int canMerge(int subgraph_id, const std::unordered_set<std::string>& subgraphinputNodeNames,
			  const std::vector<std::unordered_set<std::string>>& AllSubgraphsNodeNames) {
    if (subgraphinputNodeNames.empty()) return -1;
	if (AllSubgraphsNodeNames.empty()) return -1;
	for (size_t i = 0; i < AllSubgraphsNodeNames.size(); i++) {
		if (i == subgraph_id) {
			continue;
		}
		bool allFound = true;
        for (const auto& nodeName : subgraphinputNodeNames) {
            if (AllSubgraphsNodeNames[i].find(nodeName) == AllSubgraphsNodeNames[i].end()) {
                allFound = false;
                break;
            }
        }
		if (allFound) return i;
	}
    return -1;
}

int canMerge(int subgraph_id, const std::vector<std::unordered_set<std::string>>& AllSubgraphsInputNodeNames,
			 const std::unordered_set<std::string>& subgraphNodeNames) {
    if (AllSubgraphsInputNodeNames.empty()) return -1;
	if (subgraphNodeNames.empty()) return -1;
	for (size_t i = 0; i < AllSubgraphsInputNodeNames.size(); i++) {
		if (i == subgraph_id) {
			continue;
		}
        for (const auto& InputnodeName : AllSubgraphsInputNodeNames[i]) {
            if (subgraphNodeNames.find(InputnodeName) != subgraphNodeNames.end()) {
				return i;
            }
        }
	}
    return -1;
}

void mergeGraphs(onnx::GraphProto& targetGraph, onnx::GraphProto& sourceGraph) {
    for (const auto& node : sourceGraph.node()) {
        *targetGraph.add_node() = node;
    }
}

NodeIOSize FindNodeIOSize(std::string nodeName, const std::unordered_map<std::string, NodeIOSize> &nodeSizes) {
	NodeIOSize ioSize;
	auto it = nodeSizes.find(nodeName);
	if (it != nodeSizes.end()) {
        ioSize = it->second;
        std::cout << "Node: " << nodeName << std::endl;
		std::cout << "Input sizes:" << std::endl;
		for (const auto& input : ioSize.inputSizes) {
			std::cout << "  [ ";
			for (const auto& size : input) {
				std::cout << size << " ";
			}
			std::cout << "]" << std::endl;
		}

		std::cout << "Output sizes:" << std::endl;
		for (const auto& output : ioSize.outputSizes) {
			std::cout << "  [ ";
			for (const auto& size : output) {
				std::cout << size << " ";
			}
			std::cout << "]" << std::endl;
		}
    } else {
        std::cout << "Node with name " << nodeName << " not found." << std::endl;
		exit(0);
	}
	return ioSize;
}

void getAllnodeName(const onnx::GraphProto &g) {

    std::ofstream outFile("./allNodeName.txt");
    if (!outFile.is_open()) {
        std::cerr << "Error opening file." << std::endl;
        exit(0);
    }

	for (int i = 0; i < g.node_size(); ++i) {
		outFile << "node " << i << ":" << g.node(i).name() << std::endl;
	}
}

std::unordered_map<std::string, NodeIOSize> Graph::getNodeIOSizes(const onnx::GraphProto& graph) {
    std::unordered_map<std::string, NodeIOSize> nodeSizes;
 	std::unordered_set<std::string> nodeTypes;
	std::cout << "node size:" << graph.node_size() << std::endl;

    for (const auto& node : graph.node()) {
		std::cout << "node name:" << node.name() << std::endl;
        std::string nodeType = node.op_type();

        nodeTypes.insert(nodeType);

        NodeIOSize ioSize;

        for (const auto& input : node.input()) {
			int flag_c = 1;
			std::vector<int64_t> shape_;
			for (const auto& value_info : graph.value_info()) {
				if (value_info.name() == input) {
		    		for (const auto& dim : value_info.type().tensor_type().shape().dim()) {
		    			shape_.push_back(dim.dim_value());
		    		}
					flag_c = 0;
					break;
		    	}
		  	}
			if (flag_c == 1) {
				for (auto value_info : graph.input()) {
					if (value_info.name() == input) {
						for (const auto& dim : value_info.type().tensor_type().shape().dim()) {
							shape_.push_back(dim.dim_value());
						}
						flag_c = 0;
						break;
					}
				}
			}
			if (flag_c == 1) {
				for (auto value_info : graph.initializer()) {
					if (value_info.name() == input) {
						const onnx::TensorProto& tensor_proto = value_info;
						for (const auto& dim : tensor_proto.dims()) {
							shape_.push_back(dim);
						}
						flag_c = 0;
						break;
					}
				}
			}
			if (flag_c == 1) {
				for (auto value_info : graph.output()) {
					if (value_info.name() == input) {
						for (const auto& dim : value_info.type().tensor_type().shape().dim()) {
							shape_.push_back(dim.dim_value());
						}
						flag_c = 0;
						break;
					}
				}
			}
            ioSize.inputSizes.push_back(shape_);
        }

        for (const auto& output : node.output()) {
			int flag_c = 1;
			std::vector<int64_t> shape_;
			for (const auto& value_info : graph.value_info()) {
				if (value_info.name() == output) {
		    		for (const auto& dim : value_info.type().tensor_type().shape().dim()) {
		    			shape_.push_back(dim.dim_value());
		    		}
					flag_c = 0;
					break;
		    	}
		  	}
			if (flag_c == 1) {
				for (auto value_info : graph.input()) {
					if (value_info.name() == output) {
						for (const auto& dim : value_info.type().tensor_type().shape().dim()) {
							shape_.push_back(dim.dim_value());
						}
						flag_c = 0;
						break;
					}
				}
			}
			if (flag_c == 1) {
				for (auto value_info : graph.initializer()) {
					if (value_info.name() == output) {
						const onnx::TensorProto& tensor_proto = value_info;
						for (const auto& dim : tensor_proto.dims()) {
							shape_.push_back(dim);
						}
						flag_c = 0;
						break;
					}
				}
			}
			if (flag_c == 1) {
				for (auto value_info : graph.output()) {
					if (value_info.name() == output) {
						for (const auto& dim : value_info.type().tensor_type().shape().dim()) {
							shape_.push_back(dim.dim_value());
						}
						flag_c = 0;
						break;
					}
				}
			}
            ioSize.outputSizes.push_back(shape_);
        }

        nodeSizes[node.name()] = ioSize;
    }

    std::cout << "Node Types:" << std::endl;
    for (const auto& type : nodeTypes) {
        std::cout << type << std::endl;
    }

    return nodeSizes;
}

onnx::GraphProto Graph::GetGraphFromOnnx(std::string &path) {
	std::ifstream input(path, std::ios::ate | std::ios::binary);
	onnx::ModelProto model;
	std::streamsize size = input.tellg();
	input.seekg(0, std::ios::beg);

	std::vector<char> buffer(size);
	input.read(buffer.data(), size); 
	model.ParseFromArray(buffer.data(), size);
	return model.graph();
}
