#ifndef GRAPH_H
#define GRAPH_H

#include "onnx.pb.h"
#include <iostream>
#include <fstream>
#include <unordered_map>
#include <functional>

struct NodeIOSize {
    std::vector<std::vector<int64_t>> inputSizes;
    std::vector<std::vector<int64_t>> outputSizes;
};

struct NodeTensor {
    std::string name;
    std::vector<int64_t> shape;

    NodeTensor() = default;

    NodeTensor(const std::string& n, const std::vector<int64_t>& s) : name(n), shape(s) {}

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

void determineGraphInputOutput(const onnx::GraphProto& g, const std::unordered_set<NodeTensor>& initializerNames,
    						   std::unordered_set<NodeTensor> &graphInputs, std::unordered_set<NodeTensor> &graphOutputs);
void determineGraphInput(const onnx::GraphProto& g, const std::unordered_set<NodeTensor>& initializerNames,
    						   std::unordered_set<NodeTensor> &graphInputs);
void determineGraphOutput(const onnx::GraphProto& g, std::vector<std::unordered_set<NodeTensor>> &allgraphInputs_1,
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

public:
    Graph() {}
    ~Graph() {}

    std::unordered_map<std::string, NodeIOSize> getNodeIOSizes(const onnx::GraphProto& graph);
    onnx::GraphProto GetGraphFromOnnx(std::string &path);

};

#endif
