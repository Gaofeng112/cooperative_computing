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

enum PartitionStrategy {
    SPILTE_CPU_STRUCTURE_FIRST,
    SPILTE_NPU_STRUCTURE_FIRST,
    AUTOMATIC_SEARCH
};

class Partition {
private:

public:
    Partition() {}
    ~Partition() {}
    void PartitionGraph(const onnx::GraphProto &g, Device& d, PartitionStrategy strategy, const std::unordered_map<std::string, NodeIOSize> &node_io_size);
};

#endif
