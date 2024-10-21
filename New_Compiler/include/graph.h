#ifndef GRAPH_H
#define GRAPH_H

#include "onnx.pb.h"
#include <iostream>
#include <fstream>
#include <unordered_map>
#include <functional>
// 结构体用于保存输入和输出大小
struct NodeIOSize {
    std::vector<std::vector<int64_t>> inputSizes;
    std::vector<std::vector<int64_t>> outputSizes;
};

struct NodeTensor {
    std::string name;
    std::vector<int64_t> shape;

    // Default constructor
    NodeTensor() = default;

    // Constructor with parameters
    NodeTensor(const std::string& n, const std::vector<int64_t>& s) : name(n), shape(s) {}

    // Equality comparison operator
    bool operator==(const NodeTensor& other) const {
        return name == other.name && shape == other.shape;
    }
};

namespace std {
    template <>
    struct hash<NodeTensor> {
        size_t operator()(const NodeTensor& tensor) const {
            size_t hashValue = hash<string>()(tensor.name);
            for (auto& val : tensor.shape) {
                hashValue ^= hash<int64_t>()(val) + 0x9e3779b9 + (hashValue << 6) + (hashValue >> 2);
            }
            return hashValue;
        }
    };
}

std::unordered_set<NodeTensor> getInitializer(const onnx::GraphProto& graph);
std::unordered_set<NodeTensor> getIOvalue(const onnx::GraphProto& graph);
std::unordered_set<NodeTensor> getOutvalue(const onnx::GraphProto& graph);
void determineGraphInputOutput(const onnx::GraphProto& g, const std::unordered_set<NodeTensor>& initializerNames,
    						   std::unordered_set<NodeTensor> &graphInputs, std::unordered_set<NodeTensor> &graphOutputs);
void determineGraphInput(const onnx::GraphProto& g, const std::unordered_set<NodeTensor>& initializerNames,
    						   std::unordered_set<NodeTensor> &graphInputs);
// void determineGraphOutput(const onnx::GraphProto& g, std::vector<std::unordered_set<NodeTensor>> &allgraphInputs_1,
// 						  std::vector<std::unordered_set<NodeTensor>> &allgraphInputs_2, std::unordered_set<NodeTensor> &graphOutputs);
void determineGraphOutput(const onnx::GraphProto& originalGraph, const onnx::GraphProto& g, std::vector<std::unordered_set<NodeTensor>> &allgraphInputs_1,
						std::vector<std::unordered_set<NodeTensor>> &allgraphInputs_2, std::unordered_set<NodeTensor> &graphOutputs);
std::string findInputNode(const onnx::GraphProto &g, const std::string& outputTensorName);
std::unordered_set<std::string> collectNodeNames(const onnx::GraphProto& graph);
int canMerge(int subgraph_id, const std::unordered_set<std::string>& subgraphinputNodeNames,
			  const std::vector<std::unordered_set<std::string>>& AllinputNodeNames);
int canMerge(int subgraph_id, const std::vector<std::unordered_set<std::string>>& AllSubgraphsInputNodeNames,
			 const std::unordered_set<std::string>& subgraphNodeNames);
void mergeGraphs(onnx::GraphProto& targetGraph, onnx::GraphProto& sourceGraph);

NodeIOSize FindNodeIOSize(std::string nodeName, const std::unordered_map<std::string, NodeIOSize> &nodeSizes);
void getAllnodeName(const onnx::GraphProto &g);

class Graph {
private:
    /* data */
public:
    Graph() {}
    ~Graph() {}

    std::unordered_map<std::string, NodeIOSize> getNodeIOSizes(const onnx::GraphProto& graph);
    onnx::GraphProto GetGraphFromOnnx(std::string &path);

};
/////////7.22
struct graph_adjacency_node
{
    //std::vector<int> input_node_index;
    std::vector<int> output_node_index;
    int rank;
    std::string name;
    int index;
};
// std::vector<graph_adjacency_node> get_adjancency_list(const onnx::GraphProto &g, int* visited);
// void DFS(const onnx::GraphProto &g,onnx::GraphProto &subgraph, 
// 		int* visited, const onnx::NodeProto& start_node,
// 		int node_index,std::vector<graph_adjacency_node>& adjacency_list,
// 		const std::vector<std::string>& support_op);
// std::vector<onnx::GraphProto> determine_subgraphs(const onnx::GraphProto& g, Device& d, int* visited, 
// 												std::vector<graph_adjacency_node>& adjacency_list,PartitionStrategy strategy);
/////////////end
#endif
