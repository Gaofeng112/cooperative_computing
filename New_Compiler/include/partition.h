#ifndef PARTITION_H
#define PARTITION_H

#include "onnx.pb.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include "Device.h"
#include "graph.h"

// 定义一个枚举类型表示分区策略
enum PartitionStrategy {
    SPILTE_CPU_STRUCTURE_FIRST,
    SPILTE_NPU_STRUCTURE_FIRST,
    AUTOMATIC_SEARCH
};

class Partition {
private:
    /* data */
public:
    Partition() {}
    ~Partition() {}
    void PartitionGraph(const onnx::GraphProto &g, Device& d, PartitionStrategy strategy, const std::unordered_map<std::string, NodeIOSize> &node_io_size);
};

int DetermineStructure(const onnx::GraphProto& graph, Device &d,PartitionStrategy strategy);
#endif
