#include <iostream>
#include <string>
#include "graph.h"
#include "partition.h"
#include "Licheepi.h"
#include "/usr/include/python3.10/Python.h"
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
    lpi4a.GetDeviceJson("config.json");
    p.PartitionGraph(g, lpi4a, PartitionStrategy::SPILTE_NPU_STRUCTURE_FIRST, node_io_size);

    Py_Initialize();
    if (!Py_IsInitialized()) {
		std::cout << "python init fail" << std::endl;
		return 0;
	}
    PyObject *pModule,*pModule1;
    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.path.append('.')");
    Py_Finalize();

    return 0;
}
