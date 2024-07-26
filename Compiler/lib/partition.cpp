#include "partition.h"
#include <algorithm>

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

    std::vector<std::unordered_set<std::string>> subgraphs_2_input_nodes_;
    std::vector<std::unordered_set<std::string>> subgraphs_2_nodes_;
    for (const auto& sg : otherSubgraphs) {
        std::unordered_set<NodeTensor> graphInputs;
        determineGraphInput(sg, IOvalueNames, graphInputs);

        std::unordered_set<std::string> graphInputsNodes;
        for (const auto& input : graphInputs) {
            auto nodename = findInputNode(g, input.name);
            if (nodename != "") {
                graphInputsNodes.insert(nodename);
            }
        }
        subgraphs_2_input_nodes_.push_back(graphInputsNodes);
        subgraphs_2_nodes_.push_back(collectNodeNames(sg));
    }

    for (size_t i = 0; i < otherSubgraphs.size(); ++i) {
        if (subgraphs_2_input_nodes_[i].empty()) {
            int mergeIndex = canMerge(i, subgraphs_2_input_nodes_, subgraphs_2_nodes_[i]);
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

    std::vector<std::unordered_set<NodeTensor>> subgraphs_1_inputs;
    std::vector<std::unordered_set<std::string>> subgraphs_1_input_nodes;
    std::vector<std::unordered_set<std::string>> subgraphs_1_nodes;
    for (const auto& sg : Subgraphs) {
        std::unordered_set<NodeTensor> graphInputs;
        determineGraphInput(sg, IOvalueNames, graphInputs);
        subgraphs_1_inputs.push_back(graphInputs);

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
    std::vector<std::unordered_set<std::string>> subgraphs_2_input_nodes;
    std::vector<std::unordered_set<std::string>> subgraphs_2_nodes;
    for (const auto& sg : otherSubgraphs) {
        std::unordered_set<NodeTensor> graphInputs;
        determineGraphInput(sg, IOvalueNames, graphInputs);
        subgraphs_2_inputs.push_back(graphInputs);

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

    std::vector<std::unordered_set<NodeTensor>> subgraphs_1_outputs;
    for (const auto& sg : Subgraphs) {
        std::unordered_set<NodeTensor> graphOutputs;
        determineGraphOutput(g, sg, subgraphs_1_inputs, subgraphs_2_inputs, graphOutputs);
        subgraphs_1_outputs.push_back(graphOutputs);
    }
    std::vector<std::unordered_set<NodeTensor>> subgraphs_2_outputs;
    for (const auto& sg : otherSubgraphs) {
        std::unordered_set<NodeTensor> graphOutputs;
        determineGraphOutput(g, sg, subgraphs_1_inputs, subgraphs_2_inputs, graphOutputs);
        subgraphs_2_outputs.push_back(graphOutputs);
    }



    std::vector<std::unordered_set<NodeTensor>> graphs_inputs;
    graphs_inputs.insert(graphs_inputs.end(),subgraphs_1_inputs.begin(),subgraphs_1_inputs.end());
    graphs_inputs.insert(graphs_inputs.end(),subgraphs_2_inputs.begin(),subgraphs_2_inputs.end());
    std::vector<std::unordered_set<NodeTensor>> graphs_outputs;
    graphs_outputs.insert(graphs_outputs.end(),subgraphs_1_outputs.begin(),subgraphs_1_outputs.end());
    graphs_outputs.insert(graphs_outputs.end(),subgraphs_2_outputs.begin(),subgraphs_2_outputs.end());
    std::vector<int> order_Subgraphs(graphs_inputs.size());
    std::vector<int> issort_Subgraphs(graphs_inputs.size());
    std::vector<std::vector<int>> predecessors_Subgraphs(graphs_inputs.size());
    std::vector<std::vector<int>> successors_Subgraphs(graphs_inputs.size());
    int sub1_size=subgraphs_1_inputs.size();
    int sub2_size=subgraphs_2_inputs.size();
    int finished_flag=0;int sort_count=0;
    while(!finished_flag) 
    {
        finished_flag=1;
        if(sort_count==0)
        {
            for(int i=0; i<graphs_inputs.size();i++)
            {
                int find_flag=0;
                for(const auto& g_input : graphs_inputs[i])
                {
                    for(int j=0; j< graphs_outputs.size();j++)
                    {
                        if(graphs_outputs[j].find(g_input)!=graphs_outputs[j].end())
                        {
                        find_flag=1;
                        break;
                        }
                    }
                    if(find_flag){break;}
                }
                if(!find_flag)
                {
                    order_Subgraphs[i]=0;
                    issort_Subgraphs[i]=1;
                }
                else {order_Subgraphs[i]=1;issort_Subgraphs[i]=0;finished_flag=0;}
            }
        }
        else
        {
            for(int i=0; i<graphs_inputs.size();i++)
            {
                int find_flag=0;
                std::vector<int> predecessors;
                if(issort_Subgraphs[i]==1){continue;}
                for(const auto& g_input : graphs_inputs[i])
                {
                    
                    for(int j=0; j< graphs_outputs.size();j++)
                    {
                        if((graphs_outputs[j].find(g_input)!=graphs_outputs[j].end()))
                        {
                            if((issort_Subgraphs[j]==0))
                            {
                                find_flag=1;
                                break;
                            }
                            predecessors.push_back(j);
                        }
                         
                    }
                    if(find_flag){break;}
                }
                if(!find_flag)
                {
                    order_Subgraphs[i]=sort_count;
                    predecessors_Subgraphs[i].insert(predecessors_Subgraphs[i].end(),predecessors.begin(),predecessors.end());
                }
                else {order_Subgraphs[i]=sort_count+1;issort_Subgraphs[i]=0;finished_flag=0;}
                if(i==graphs_inputs.size()-1)
                {
                    for(int j=0; j<graphs_inputs.size();j++)
                    {
                        if(order_Subgraphs[j]==sort_count)
                        {
                            issort_Subgraphs[j]=1;
                        }
                    }
                }
            }
        }
        
        sort_count++;
    }

    for(int i=0;i<graphs_inputs.size();i++)
    {
        for(int j=0;j<graphs_inputs.size();j++)
        {
            if(find(predecessors_Subgraphs[j].begin(),predecessors_Subgraphs[j].end(),i)!=predecessors_Subgraphs[j].end())
            {
                successors_Subgraphs[i].push_back(j);
            }
        }
    }
    char* sub1_type,*sub2_type;
    if(strategy==SPILTE_CPU_STRUCTURE_FIRST)
    {
        sub1_type="CPU";
        sub2_type="NPU";
    }
    else{
        sub1_type="NPU";
        sub2_type="CPU";
    }
    std::cout <<  " order"<<std::endl;
    for(auto element : order_Subgraphs)
    {
        std::cout << element << " ";
    }
    std::cout<<std::endl;

    std::string file_name = "subgraphs_ios.txt";
    std::ofstream outfile1(file_name);
    if (!outfile1.is_open()) {
        std::cerr << "Error opening file." << std::endl;
        exit(0);
    }

    for(int i=0;i<graphs_inputs.size();i++)
    {
        outfile1 << (i>=sub1_size?sub2_type:sub1_type)<<"subgraph"<<(i>=sub1_size?(i-sub1_size):i)<<": order"<<order_Subgraphs[i]<<std::endl;
        outfile1 <<"--input-name ";
        std::cout << (i>=sub1_size?sub2_type:sub1_type)<<"subgraph"<<(i>=sub1_size?(i-sub1_size):i)<<": order"<<order_Subgraphs[i]<<std::endl;
        std::cout << "Inputs:";
        for(auto element :  graphs_inputs[i])
        {
            std::cout<<element.name<<"; size:";
            for(auto Size : element.shape)
            {std::cout<<Size<<" ";}
            outfile1<<element.name<<";";
        }
        std::cout << std::endl;
        std::cout << "Outputs:";
        outfile1<<"--output-name ";
        for(auto element :  graphs_outputs[i])
        {
            std::cout<<element.name<<"; size:";
            for(auto Size : element.shape)
            {std::cout<<Size<<" ";}
            outfile1<<element.name<<";";
        }
        outfile1<<std::endl;
        std::cout << std::endl;
        std::cout <<  " The predecessors of "<<  (i>=sub1_size?sub2_type:sub1_type)<<"subgraph"<<(i>=sub1_size?(i-sub1_size):i)<<": ";
        for(auto element : predecessors_Subgraphs[i])
        {
            std::cout <<  (element>=sub1_size?sub2_type:sub1_type)<<"subgraph"<<(element>=sub1_size?(element-sub1_size):element) <<"; ";
        }
            std::cout <<std::endl;
        std::cout <<  " The successors of "<<  (i>=sub1_size?sub2_type:sub1_type)<<"subgraph"<<(i>=sub1_size?(i-sub1_size):i)<<": ";
        for(auto element : successors_Subgraphs[i])
        {
             std::cout <<  (element>=sub1_size?sub2_type:sub1_type)<<"subgraph"<<(element>=sub1_size?(element-sub1_size):element) <<"; ";
        }
            std::cout <<std::endl;
    }
    outfile1.close();

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
