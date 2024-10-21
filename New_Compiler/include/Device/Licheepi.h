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
    Licheepi(/* args */) {}
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
            // {"Sigmoid","Mul","Gemm"},
            // {"Div","Mul","Add"},
            // // {"Reshape","Transpose","Reshape","Reshape","Transpose","Transpose","Mul","Mul"},
            // {"Add","Add","ReduceMean","Sub"},
            // //{"Add","Mul","Mul","Mul"},
            //原来的
            // {"Reshape", "Mul", "Add", "Sigmoid", "Mul", "Conv", "Conv", "Add", "Reshape"},
            // {"Reshape", "Mul", "Add", "Sigmoid", "Mul", "Conv", "Conv", "Add"},
            // {"Reshape", "Mul", "Add", "Sigmoid", "Mul", "Conv", "Reshape"},
            // {"Reshape", "Mul", "Add", "Sigmoid", "Mul", "Conv"},
            // {"Mul", "Add", "Conv", "Reshape"},
            // {"Mul", "Add", "Conv"},
            // {"Conv", "Reshape"},
            // {"Conv"}//,
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

    // 重写纯虚函数，提供 NPU 支持的操作
    std::vector<std::string> getNPUSupportOp() const override {
        return {"Conv", "Reshape", "Transpose", "Add", "ReduceMean", "Sub", "Div", "Mul", "Gemm", "Sigmoid"};
    }

    // 重写纯虚函数，提供 CPU 支持的操作
    std::vector<std::string> getCPUSupportOp() const override {
        return {"Sub", "Pow", "ReduceMean", "Add", "Sqrt", "Div","Transpose", "Gather", "MatMul", "Mul", "Softmax", "Erf", "Gemm", "Conv", "Reshape",
                "Sin", "Where", "ConstantOfShape", "Cast", "Sigmoid", "Cos", "Expand", "Slice", "Unsqueeze"};
    }

    std::vector<std::string> getNPUPreferOp() override {  //包含NPU only算子
        // return {"Conv", "Transpose", "ReduceMean", "Gemm"};
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
