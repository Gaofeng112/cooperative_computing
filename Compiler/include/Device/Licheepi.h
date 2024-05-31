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
    Licheepi() {}
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
            {"Reshape", "Mul", "Add", "Sigmoid", "Mul", "Conv", "Conv", "Add", "Reshape"},
            {"Reshape", "Mul", "Add", "Sigmoid", "Mul", "Conv", "Conv", "Add"},
            {"Reshape", "Mul", "Add", "Sigmoid", "Mul", "Conv", "Reshape"},
            {"Reshape", "Mul", "Add", "Sigmoid", "Mul", "Conv"},
            {"Mul", "Add", "Conv", "Reshape"},
            {"Mul", "Add", "Conv"},
            {"Conv", "Reshape"},
            {"Conv"}
        };
    }

    std::vector<std::string> getNPUSupportOp() const override {
        return {"Conv", "Reshape", "Transpose", "Add", "ReduceMean", "Sub", "Div", "Mul", "Gemm", "Sigmoid"};
    }

    std::vector<std::string> getCPUSupportOp() const override {
        return {"Sub", "Pow", "ReduceMean", "Add", "Sqrt", "Div","Transpose", "Gather", "MatMul", "Mul", "Softmax", "Erf", "Gemm", "Conv", "Reshape",
                "Sin", "Where", "ConstantOfShape", "Cast", "Sigmoid", "Cos", "Expand", "Slice", "Unsqueeze"};
    }

    std::vector<std::string> getNPUPreferOp() override {
        return {"Conv", "Gemm"};
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
