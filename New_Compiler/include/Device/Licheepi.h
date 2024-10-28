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

    // 重写虚函数，返回特定的 cpu_structure , pow + reducemean + add + sqrt + div
    std::vector<std::vector<std::string>> getCPUStructure() const override {
        return {
            {"Concat"},
            {"Sub", "Pow", "ReduceMean", "Add", "Sqrt", "Div"},
            {"Transpose", "Gather", "Gather", "Gather", "Transpose", "MatMul", "Mul", "Softmax", "MatMul"}
        };
    }

    // 重写虚函数，返回特定的 npu_structure
    std::vector<std::vector<std::string>> getNPUStructure() const override {
        return {
            // //////7.19加
            // {"Reshape"}
            //diffussion
            {"Reshape","Transpose","Reshape"},
            // {"Add","Sigmoid","Mul"},
            // {"Reshape","ReduceMean","Sub","Mul","ReduceMean","Add"},
            //{"Mul","Mul","Sub","Mul","Add","Reshape"},
            // {"Mul","Mul","Sub","Mul","Add"},
            {"Reshape","Sigmoid","Mul","Transpose","Conv","Add","Transpose"},
            {"Reshape","Transpose","Conv","Transpose","Reshape"},
            {"Reshape","Conv","Transpose"},
            // {"Mul","Add","Mul","Mul","Mul"},
            // //{"Reshape","Add","Add","Reshape","Transpose","Conv","Add","Conv","Transpose","Transpose","Conv"},
            {"Reshape","Add","Add","Reshape","Transpose","Conv","Add"},
            // {"Reshape","Add","Add"},
            {"Conv"}
            ////{"Reshape","Add","Add","Reshape","Transpose","Conv","Add","Transpose"}
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
