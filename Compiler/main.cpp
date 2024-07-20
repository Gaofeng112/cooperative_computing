#include <iostream>
#include <string>
#include <cstdlib>
#include "graph.h"
#include "partition.h"
#include "Licheepi.h"

int main(int argc, char* argv[]) {
    std::string onnxFile;
    if (argc > 1) {
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg.substr(0, 7) == "--onnx=") {
                onnxFile = arg.substr(7);
                std::cout << "ONNX file: " << onnxFile << std::endl;
            } 
        }
        if (onnxFile.empty()) {
            std::cout << "No ONNX file provided." << std::endl;
            return -1;
        }
    } else {
        printf("Please set valide args: ./CoComputingCompiler --onnx=xxx.onnx\n");
        return -1;
    }

    Graph graph;
    auto g = graph.GetGraphFromOnnx(onnxFile);
    std::unordered_map<std::string, NodeIOSize> node_io_size;
    Partition p;
    Licheepi lpi4a;
    lpi4a.updateOnnxFile(onnxFile);
    p.PartitionGraph(g, lpi4a, PartitionStrategy::SPILTE_NPU_STRUCTURE_FIRST, node_io_size);

    int result;
    //根据当前代码的结果切
    result = std::system("python3 extract_onnx.py");

    if (result == 0) {
        std::cout << "Python extract_onnx ran successfully." << std::endl;
    } else {
        std::cout << "Python extract_onnx failed to run." << std::endl;
        return 1;
    }

    //根据当前runcut文件切
    result = std::system("python3 extract_runcut_onnx.py");

    if (result == 0) {
        std::cout << "Python extract_runcut_onnx ran successfully." << std::endl;
    } else {
        std::cout << "Python extract_runcut_onnx failed to run." << std::endl;
        return 1;
    }

    return 0;
}