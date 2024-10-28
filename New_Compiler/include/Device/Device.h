#ifndef DEVICE_H
#define DEVICE_H

#include <vector>
#include <string>
#include <iostream>  
#include <fstream> 
#include "onnx.pb.h"
#include "graph.h"
#include "json.h"
enum class DeviceType { Licheepi };

class Device {
private:
    /* data */
public:
    Device(/* args */) {
        NPUPreferOp = {"Conv", "Gemm"};
        CPUSupportOp ={"Sub", "Pow", "ReduceMean", "Add", "Sqrt", "Div","Transpose", "Gather", "MatMul", "Mul", "Softmax", "Erf", "Gemm", "Conv", "Reshape",
                 "Sin", "Where", "ConstantOfShape", "Cast", "Sigmoid", "Cos", "Expand", "Slice", "Unsqueeze"};
        NPUSupportOp = {"Conv", "Reshape", "Transpose", "Add", "ReduceMean", "Sub", "Div", "Mul", "Gemm", "Sigmoid"};
        max_subgraph_size = 5;
    }
    ~Device() {}
    std::vector<std::string> NPUPreferOp;
    std::vector<std::string> CPUSupportOp;
    std::vector<std::string> NPUSupportOp;
    int max_subgraph_size;
    virtual DeviceType getType() const = 0; // 返回子类的类型

    // 虚函数，用于获取 cpu_structure 成员变量
    virtual std::vector<std::vector<std::string>> getCPUStructure() const = 0;
    // 虚函数，用于获取 npu_structure 成员变量
    virtual std::vector<std::vector<std::string>> getNPUStructure() const = 0;
    // 虚函数，用于获取 NPU_SupportOp 成员变量
    std::vector<std::string> getNPUSupportOp() { 
        return NPUSupportOp;
    }
    // 虚函数，用于获取 CPU_SupportOp 成员变量
    std::vector<std::string> getCPUSupportOp() {
        return CPUSupportOp;
    }

    std::vector<std::string> getNPUPreferOp() {
        return NPUPreferOp;
    }

    // 父类的虚函数，子类根据情况自行修改
    virtual void GenerateCutInstruction(std::vector<onnx::GraphProto>& Subgraphs, std::string device,
                                        std::vector<std::unordered_set<NodeTensor>> &subgraphs_inputs, std::vector<std::unordered_set<NodeTensor>> &subgraphs_outputs) {};
    void GetDeviceJson(std::string json_path)
    {
        Json::Reader reader;
        Json::Value root;
    
        //从文件中读取，保证当前文件有demo.json文件  
        std::ifstream in(json_path, std::ios::binary);
        if (!in.is_open())
        {
            std::cout << "Error opening file\n";
            return;
        }
        if(reader.parse(in, root))
        {
            int max_subgraph_size_json = root["max_subgraph_size"].asInt();
            max_subgraph_size = max_subgraph_size_json;
            for (unsigned int i = 0; i < root["performance_data"].size(); i++)
            {
                if(root["performance_data"][i]["CPU_time"].asInt() > root["performance_data"][i]["NPU_time"].asInt())
                {
                    NPUPreferOp.push_back(root["performance_data"][i]["name"].asString());
                }
                // else
                // {
                //     CPUPreferOp.push_back(root["performance_data"][i]["name"].asString());
                // }
            }
            for(int i = 0; i < root["NPU_supported_ops"].size(); i++)
            {
                if(std::find(NPUSupportOp.begin(), NPUSupportOp.end(), root["NPU_supported_ops"][i].asString())== NPUSupportOp.end())
                {
                    NPUSupportOp.push_back(root["NPU_supported_ops"][i].asString());
                }
            }
            for(int i = 0; i < root["CPU_supported_ops"].size(); i++)
            {
                if(std::find(CPUSupportOp.begin(), CPUSupportOp.end(), root["CPU_supported_ops"][i].asString())== CPUSupportOp.end())
                {
                    CPUSupportOp.push_back(root["CPU_supported_ops"][i].asString());
                }
            }
        }
        in.close();
    }


};


#endif
