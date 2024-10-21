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
    // getAllnodeName(g);
    // std::unordered_map<std::string, NodeIOSize> node_io_size = graph.getNodeIOSizes(g);
    std::unordered_map<std::string, NodeIOSize> node_io_size;
    Partition p;
    Licheepi lpi4a;
    lpi4a.updateOnnxFile(onnxFile);
    //DetermineStructure(g, lpi4a,PartitionStrategy::SPILTE_NPU_STRUCTURE_FIRST);
    p.PartitionGraph(g, lpi4a, PartitionStrategy::SPILTE_NPU_STRUCTURE_FIRST, node_io_size);

    Py_Initialize();
    if (!Py_IsInitialized()) {
		std::cout << "python init fail" << std::endl;
		return 0;
	}
    PyObject *pModule,*pModule1;
    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.path.append('.')");
    // pModule = PyImport_ImportModule("extract_onnx");
    // if (pModule == NULL) {
	// 	std::cout << "module not found" << std::endl;
	// 	return 1;
	// }
    // pModule1 = PyImport_ImportModule("extract_runcut_onnx");
    // if (pModule1 == NULL) {
	// 	std::cout << "module not found" << std::endl;
	// 	return 1;
	// }
    Py_Finalize();

    return 0;
}
