#ifndef DEVICE_H
#define DEVICE_H

#include <vector>
#include <string>
#include "onnx.pb.h"
#include "graph.h"

enum class DeviceType { Licheepi };

class Device {
private:

public:
    Device() {}
    ~Device() {}

    virtual DeviceType getType() const = 0;

    virtual std::vector<std::vector<std::string>> getCPUStructure() const = 0;
    virtual std::vector<std::vector<std::string>> getNPUStructure() const = 0;
    virtual std::vector<std::string> getNPUSupportOp() const = 0;
    virtual std::vector<std::string> getCPUSupportOp() const = 0;

    virtual std::vector<std::string> getNPUPreferOp() {
        return {"Conv", "Transpose", "ReduceMean"};
    }

    virtual void GenerateCutInstruction(std::vector<onnx::GraphProto>& Subgraphs, std::string device,
                                        std::vector<std::unordered_set<NodeTensor>> &subgraphs_inputs, std::vector<std::unordered_set<NodeTensor>> &subgraphs_outputs) {};
};

#endif
