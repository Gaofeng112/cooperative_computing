#include "Licheepi.h"

// void Licheepi::GenerateCutInstruction(std::vector<onnx::GraphProto> &Subgraphs, std::string device,
//                                       std::vector<std::unordered_set<NodeTensor>> &subgraphs_inputs, std::vector<std::unordered_set<NodeTensor>> &subgraphs_outputs) {
//     std::cout << "Generate Cut Instruction for Licheepi" << std::endl;
//     // 打开文件
//     std::string file_name = device + "CutInstruction.txt";
//     std::ofstream outFile(file_name);
//     if (!outFile.is_open()) {
//         std::cerr << "Error opening file." << std::endl;
//         exit(0);
//     }
//     for (size_t i = 0; i < Subgraphs.size(); i++) {
//         // 默认参数
//         std::string modelFile = onnxFile;
//         std::string dataScaleDiv = "255";
//         std::string board;
//         if (device == "npu") {
//             board = "th1520";
//         } else {
//             board = "c920";
//         }
//         std::string postprocess = "save_and_top5";

//         std::unordered_set<NodeTensor> graphInputs = subgraphs_inputs[i];
//         std::unordered_set<NodeTensor> graphOutputs = subgraphs_outputs[i];

//         std::string inputName = "\"";
//         for (const auto& input : graphInputs) {
//             inputName = inputName + input.name + ";";
//         }
//         // 删除最后一个分号
//         if (!inputName.empty() && inputName.back() == ';') {
//             inputName.pop_back();
//         }
//         inputName = inputName + "\"";
//         std::string outputName = "\"";
//         for (const auto& output : graphOutputs) {
//             outputName = outputName + output.name + ";";
//         }
//         // 删除最后一个分号
//         if (!outputName.empty() && outputName.back() == ';') {
//             outputName.pop_back();
//         }
//         outputName = outputName + "\"";

//         std::string inputShape = "\"";
//         for (const auto& input : graphInputs) {
//             for (const auto& dim : input.shape) {
//                 inputShape = inputShape + std::to_string(dim) + " ";
//             }
//             // 删除最后一个空格
//             if (!inputShape.empty() && inputShape.back() == ' ') {
//                 inputShape.pop_back();
//             }
//             inputShape = inputShape + ";";
//         }
//         // 删除最后一个分号
//         if (!inputShape.empty() && inputShape.back() == ';') {
//             inputShape.pop_back();
//         }
//         inputShape = inputShape + "\"";

//         std::string calibrateDataset = device + "_Subgraphs_" + std::to_string(i) + ".npz";
//         std::string quantizationScheme = "int8_asym";

//         // 写入配置信息
//         outFile << "hhb -C";
//         outFile << " --model-file " << modelFile;
//         outFile << " --data-scale-div " << dataScaleDiv;
//         outFile << " --board " << board;
//         outFile << " --postprocess " << postprocess;
//         outFile << " --input-name " << inputName;
//         outFile << " --output-name " << outputName;
//         outFile << " --input-shape " << inputShape;
//         outFile << " --calibrate-dataset " << calibrateDataset;
//         outFile << " --quantization-scheme " << quantizationScheme;
//         outFile << std::endl;
//     }

//     // 关闭文件
//     outFile.close();

//     std::cout << "Configurations written to config.txt successfully." << std::endl;
// }
