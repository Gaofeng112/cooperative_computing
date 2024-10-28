#ifndef LICHEEPI_H
#define LICHEEPI_H

#include <iostream>
#include <fstream>
#include <string>
#include "Device.h"
#include "graph.h"

class Licheepi : public Device{
private:
    std::string onnxFile;
public:
    Licheepi(/* args */) : Device() {}
    ~Licheepi() {}

    DeviceType getType() const override {
        return DeviceType::Licheepi;
    }
    std::vector<std::vector<std::string>> getCPUStructure() const override {
        return {
            {"Concat"},
            {"Sub", "Pow", "ReduceMean", "Add", "Sqrt", "Div"},
            {"Transpose", "Gather", "Gather", "Gather", "Transpose", "MatMul", "Mul", "Softmax", "MatMul"}
        };
    }
    std::vector<std::vector<std::string>> getNPUStructure() const override {
        return {
            {"Reshape","Transpose","Reshape"},
            {"Reshape","Sigmoid","Mul","Transpose","Conv","Add","Transpose"},
            {"Reshape","Transpose","Conv","Transpose","Reshape"},
            {"Reshape","Conv","Transpose"},
            {"Reshape","Add","Add","Reshape","Transpose","Conv","Add"},
            {"Conv"}
        };
    }
    void GenerateCutInstruction(std::vector<onnx::GraphProto> &Subgraphs, std::string device,
                                std::vector<std::unordered_set<NodeTensor>> &subgraphs_inputs, std::vector<std::unordered_set<NodeTensor>> &subgraphs_outputs) override;

    void updateOnnxFile(std::string &path) {
        onnxFile = path;
    }

    std::string getOnnxFile() {
        return onnxFile;
    }
};


#endif
