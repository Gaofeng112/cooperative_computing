#ifndef DEVICE_H
#define DEVICE_H

#include <vector>
#include <string>
#include "onnx.pb.h"
#include "graph.h"

enum class DeviceType { Licheepi };

class Device {
private:
    /* data */
public:
    Device(/* args */) {}
    ~Device() {}

    virtual DeviceType getType() const = 0; // 返回子类的类型

    // 虚函数，用于获取 cpu_structure 成员变量
    virtual std::vector<std::vector<std::string>> getCPUStructure() const = 0;
    // 虚函数，用于获取 npu_structure 成员变量
    virtual std::vector<std::vector<std::string>> getNPUStructure() const = 0;
    // 虚函数，用于获取 NPU_SupportOp 成员变量
    virtual std::vector<std::string> getNPUSupportOp() const = 0;
    // 虚函数，用于获取 CPU_SupportOp 成员变量
    virtual std::vector<std::string> getCPUSupportOp() const = 0;

    virtual std::vector<std::string> getNPUPreferOp() {
        return {"Conv", "Transpose", "ReduceMean"};
    }

    // 父类的虚函数，子类根据情况自行修改
    virtual void GenerateCutInstruction(std::vector<onnx::GraphProto>& Subgraphs, std::string device,
                                        std::vector<std::unordered_set<NodeTensor>> &subgraphs_inputs, std::vector<std::unordered_set<NodeTensor>> &subgraphs_outputs) {};
};

#endif
