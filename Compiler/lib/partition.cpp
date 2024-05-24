#include "partition.h"

std::vector<std::vector<std::string>> getEnabledStructures(const std::vector<std::string>& support_op, const std::vector<std::vector<std::string>>& structure) {
    std::vector<std::vector<std::string>> enable_structure;

    for (const auto& ops : structure) {
        bool enabled = true;
        for (const std::string& op : ops) {
            if (std::find(support_op.begin(), support_op.end(), op) == support_op.end()) {
                enabled = false;
                break;
            }
        }
        if (enabled) {
            enable_structure.push_back(ops);
        }
    }
    
    return enable_structure;
}

std::vector<onnx::GraphProto> Subgraphs;

int checkAndPrintStructure(const onnx::GraphProto& graph, int startNodeIndex, const std::vector<std::vector<std::string>>& structure) {
    int find_flag = 0;
    int nextNodeIndex = startNodeIndex;
    for (const auto& seq : structure) {
        size_t structureIndex = 0;
        int currentNodeIndex = startNodeIndex;
        int structurestartNodeIndex = 0;
        while (currentNodeIndex < graph.node_size()) {
            const auto& node = graph.node(currentNodeIndex);
            if (structureIndex >= seq.size()) {
                onnx::GraphProto subgraph;
                for (int i = structurestartNodeIndex; i < currentNodeIndex; ++i) {
                    *subgraph.add_node() = graph.node(i);
                }
                Subgraphs.push_back(subgraph);
                find_flag = 1;
                nextNodeIndex = currentNodeIndex - 1;
                break;
            }
            if (node.op_type() == seq[structureIndex]) {
                structureIndex++;
                if (structureIndex == 1) {
                    structurestartNodeIndex = currentNodeIndex;
                }
            } else {
                break;
            }
            currentNodeIndex++;
        }
        if (find_flag) {
            break;
        }
    }
    return nextNodeIndex;
}

void findAndPrintStructures(const onnx::GraphProto &g, Device& d, PartitionStrategy strategy) {
    switch(strategy) {
        case SPILTE_CPU_STRUCTURE_FIRST:{
            std::vector<std::vector<std::string>> enable_cpu_structure = getEnabledStructures(d.getCPUSupportOp(), d.getCPUStructure());

            for (int i = 0; i < g.node_size(); ++i) {
                i = checkAndPrintStructure(g, i, enable_cpu_structure);
            }
            break;
        }
        case SPILTE_NPU_STRUCTURE_FIRST:{
            std::vector<std::vector<std::string>> enable_npu_structure = getEnabledStructures(d.getNPUSupportOp(), d.getNPUStructure());

            for (int i = 0; i < g.node_size(); ++i) {
                i = checkAndPrintStructure(g, i, enable_npu_structure);
            }
            break;
        }
        default:
            break;
    }
}

std::vector<onnx::GraphProto> findOtherSubgraphs(const onnx::GraphProto& originalGraph, 
                                                 const std::vector<onnx::GraphProto>& knownSubgraphs) {
    std::vector<onnx::GraphProto> otherSubgraphs;
    std::set<std::string> knownSubgraphNodeNames;

    for (const auto& subgraph : knownSubgraphs) {
        for (const auto& node : subgraph.node()) {
            knownSubgraphNodeNames.insert(node.name());
        }
    }

    std::set<std::string> originalGraphNodeNames;
    for (const auto& node : originalGraph.node()) {
        originalGraphNodeNames.insert(node.name());
    }

    int startIndex = 0;
    int endIndex = -1;
    for (int i = 0; i < originalGraph.node_size(); ++i) {
        if (knownSubgraphNodeNames.find(originalGraph.node(i).name()) == knownSubgraphNodeNames.end()) {
            endIndex = i;
        } else {
            if (endIndex >= startIndex) {
                onnx::GraphProto newSubgraph;
                for (int j = startIndex; j <= endIndex; ++j) {
                    *newSubgraph.add_node() = originalGraph.node(j);
                }
                otherSubgraphs.push_back(newSubgraph);
            }

            startIndex = i + 1;
        }
    }

    if (startIndex < originalGraph.node_size()) {
        onnx::GraphProto lastSubgraph;
        for (int j = startIndex; j < originalGraph.node_size(); ++j) {
            *lastSubgraph.add_node() = originalGraph.node(j);
        }
        otherSubgraphs.push_back(lastSubgraph);
    }

    return otherSubgraphs;
}

void Partition::PartitionGraph(const onnx::GraphProto &g, Device& d, PartitionStrategy strategy, const std::unordered_map<std::string, NodeIOSize> &node_io_size) {
    std::unordered_set<NodeTensor> initializerNames = getInitializer(g);
    std::unordered_set<NodeTensor> IOvalueNames = getIOvalue(g);

    findAndPrintStructures(g, d, strategy);
    int node_sum = 0;
    std::ofstream outFile("./subgraphs_1.txt");
    if (!outFile.is_open()) {
        std::cerr << "Error opening file." << std::endl;
        exit(0);
    }
    int id = 0;
    for (const auto& vec : Subgraphs) {
        outFile << " subgraph" << id << ":";
        for (const auto& node : vec.node()) {
            outFile << node.name() << " ";
        }
        id++;
        outFile << std::endl;
        node_sum += vec.node_size();
    }

    auto otherSubgraphs = findOtherSubgraphs(g, Subgraphs);
    std::ofstream outFile_2("./subgraphs_2.txt");
    if (!outFile_2.is_open()) {
        std::cerr << "Error opening file." << std::endl;
        exit(0);
    }
    std::cout << "before:" << std::endl;
    for (const auto& vec : otherSubgraphs) {
        outFile_2 << " subgraph" << id << ":";
        for (const auto& node : vec.node()) {
            outFile_2 << node.name() << " ";
        }
        id++;
        outFile_2 << std::endl;
        node_sum += vec.node_size();
    }

    std::vector<std::unordered_set<NodeTensor>> subgraphs_1_inputs;
    std::vector<std::unordered_set<NodeTensor>> subgraphs_1_outputs;
    std::vector<std::unordered_set<std::string>> subgraphs_1_input_nodes;
    std::vector<std::unordered_set<std::string>> subgraphs_1_nodes;
    for (const auto& sg : Subgraphs) {
        std::unordered_set<NodeTensor> graphInputs;
        std::unordered_set<NodeTensor> graphOutputs;
        determineGraphInputOutput(sg, IOvalueNames, graphInputs, graphOutputs);
        subgraphs_1_inputs.push_back(graphInputs);
        subgraphs_1_outputs.push_back(graphOutputs);

        std::unordered_set<std::string> graphInputsNodes;
        for (const auto& input : graphInputs) {
            auto nodename = findInputNode(g, input.name);
            if (nodename != "") {
                graphInputsNodes.insert(nodename);
            }
        }
        subgraphs_1_input_nodes.push_back(graphInputsNodes);
        subgraphs_1_nodes.push_back(collectNodeNames(sg));
    }

    std::vector<std::unordered_set<NodeTensor>> subgraphs_2_inputs;
    std::vector<std::unordered_set<NodeTensor>> subgraphs_2_outputs;
    std::vector<std::unordered_set<std::string>> subgraphs_2_input_nodes;
    std::vector<std::unordered_set<std::string>> subgraphs_2_nodes;
    for (const auto& sg : otherSubgraphs) {
        std::unordered_set<NodeTensor> graphInputs;
        std::unordered_set<NodeTensor> graphOutputs;
        determineGraphInputOutput(sg, IOvalueNames, graphInputs, graphOutputs);
        subgraphs_2_inputs.push_back(graphInputs);
        subgraphs_2_outputs.push_back(graphOutputs);

        std::unordered_set<std::string> graphInputsNodes;
        for (const auto& input : graphInputs) {
            auto nodename = findInputNode(g, input.name);
            if (nodename != "") {
                graphInputsNodes.insert(nodename);
            }
        }
        subgraphs_2_input_nodes.push_back(graphInputsNodes);
        subgraphs_2_nodes.push_back(collectNodeNames(sg));
    }

    for (size_t i = 0; i < otherSubgraphs.size(); ++i) {
        if (subgraphs_2_input_nodes[i].empty()) {
            int mergeIndex = canMerge(i, subgraphs_2_input_nodes, subgraphs_2_nodes[i]);
            if (mergeIndex != -1) {
                std::cout << "Merge possible for graphs " << i << " and " << mergeIndex << std::endl;
                if (i < mergeIndex) {
                    mergeGraphs(otherSubgraphs[i], otherSubgraphs[mergeIndex]);
                    otherSubgraphs.erase(otherSubgraphs.begin() + mergeIndex);
                } else {
                    mergeGraphs(otherSubgraphs[mergeIndex], otherSubgraphs[i]);
                    otherSubgraphs.erase(otherSubgraphs.begin() + i);
                }

                if (mergeIndex < i) {
                    i--;
                }
            }
        }
    }

    std::ofstream outFile_3("./subgraphs_3.txt");
    if (!outFile_3.is_open()) {
        std::cerr << "Error opening file." << std::endl;
        exit(0);
    }

    for (const auto& vec : otherSubgraphs) {
        outFile_3 << " subgraph" << id << ":";
        for (const auto& node : vec.node()) {
            outFile_3 << node.name() << " ";
        }
        id++;
        outFile_3 << std::endl;
    }

    std::cout << "graph node size:" << g.node_size() << std::endl;
    std::cout << "sub node size:" << node_sum << std::endl;

    for (const auto& tensor : IOvalueNames) {
        std::cout << "Name: " << tensor.name << ", Shape: [";
        for (size_t i = 0; i < tensor.shape.size(); ++i) {
            std::cout << tensor.shape[i];
            if (i < tensor.shape.size() - 1) {
                std::cout << ", ";
            }
        }
        std::cout << "]" << std::endl;
    }

    switch (d.getType()) {
        case DeviceType::Licheepi:{
            if (strategy == SPILTE_CPU_STRUCTURE_FIRST) {
                d.GenerateCutInstruction(Subgraphs, "cpu", subgraphs_1_inputs, subgraphs_1_outputs);
                d.GenerateCutInstruction(otherSubgraphs, "npu", subgraphs_2_inputs, subgraphs_2_outputs);
            } else if (strategy == SPILTE_NPU_STRUCTURE_FIRST) {
                d.GenerateCutInstruction(Subgraphs, "npu", subgraphs_1_inputs, subgraphs_1_outputs);
                d.GenerateCutInstruction(otherSubgraphs, "cpu", subgraphs_2_inputs, subgraphs_2_outputs);
            }
            break;
        }
        default:
            std::cout << "Unknown device type" << std::endl;
            exit(0);
    }
}
