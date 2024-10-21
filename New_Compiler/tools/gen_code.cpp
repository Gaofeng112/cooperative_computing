#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>

int countSemicolons(const std::string& str) {
    int count = 0;

    for (char c : str) {
        if (c == ';') {
            count++;
        }
    }

    return count;
}

std::vector<std::vector<int>> parseShapes(const std::string& str) {
    std::vector<std::vector<int>> shapes;
    std::istringstream iss(str);

    std::string token;
    while (std::getline(iss, token, ';')) {
        std::vector<int> shape;
        std::istringstream iss2(token);

        int value;
        while (iss2 >> value) {
            shape.push_back(value);
        }

        shapes.push_back(shape);
    }

    return shapes;
}

int main() {
    std::ifstream file_in("../npuCutInstruction.txt"); // 打开文件
    if (!file_in.is_open()) {
        std::cerr << "Error opening file." << std::endl;
        return 1;
    }

    std::string line;
    int lineCount = 0;

    std::vector<int> input_num;
    std::vector<int> output_num;
    std::vector<std::vector<std::vector<int>>> allshapes;
    // 逐行读取文件
    while (std::getline(file_in, line)) {
        lineCount++;
        std::string inputName, outputName;

        // 找到--input-name 和 --output-name 的位置
        size_t inputPos = line.find("--input-name");
        size_t outputPos = line.find("--output-name");
        size_t inputshapePos = line.find("--input-shape");

        if (inputPos != std::string::npos && outputPos != std::string::npos && inputshapePos != std::string::npos) {
            // 提取引号里的内容
            size_t inputQuoteStart = line.find('"', inputPos);
            size_t inputQuoteEnd = line.find('"', inputQuoteStart + 1);
            size_t outputQuoteStart = line.find('"', outputPos);
            size_t outputQuoteEnd = line.find('"', outputQuoteStart + 1);
            size_t inputshapeQuoteStart = line.find('"', inputshapePos);
            size_t inputshapeQuoteEnd = line.find('"', inputshapeQuoteStart + 1);


            if (inputQuoteStart != std::string::npos && inputQuoteEnd != std::string::npos &&
                outputQuoteStart != std::string::npos && outputQuoteEnd != std::string::npos &&
                inputshapeQuoteStart != std::string::npos && inputshapeQuoteEnd != std::string::npos) {
                inputName = line.substr(inputQuoteStart + 1, inputQuoteEnd - inputQuoteStart - 1);
                outputName = line.substr(outputQuoteStart + 1, outputQuoteEnd - outputQuoteStart - 1);

                // 统计引号里的分号数量
                int inputSemicolonCount = countSemicolons(inputName);
                int outputSemicolonCount = countSemicolons(outputName);
                input_num.push_back(inputSemicolonCount + 1);
                output_num.push_back(outputSemicolonCount + 1);

                std::string shapeString = line.substr(inputshapeQuoteStart + 1, inputshapeQuoteEnd - inputshapeQuoteStart - 1);
                std::vector<std::vector<int>> shapes = parseShapes(shapeString);
                allshapes.push_back(shapes);

                std::cout << "Input name: " << inputName << ", semicolon count: " << inputSemicolonCount << std::endl;
                std::cout << "Output name: " << outputName << ", semicolon count: " << outputSemicolonCount << std::endl;
            }
        }
    }

    // 关闭文件
    file_in.close();
    // 你想要写入到.c文件的代码字符串
    std::string cCode;
    for (size_t i = 0; i < lineCount; i++) {
        cCode += R"(void *csinn_npu_Subgraphs_)" + std::to_string(i) + R"((char *params);)" + "\n";
        cCode += R"(void csinn_update_input_and_run_npu_Subgraphs_)" + std::to_string(i) + R"((struct csinn_tensor **input_tensors , void *sess);)" + "\n";
    }

    cCode += "\n\n\n\n";
    for (size_t i = 0; i < lineCount; i++) {
        cCode += R"(/******************npu_Subgraphs_)" + std::to_string(i) + R"(**************************/)" + "\n";
        cCode += "int npu_Subgraphs_" + std::to_string(i) + "_input_num = "+ std::to_string(input_num[i]) + ";\n";
        cCode += "int npu_Subgraphs_" + std::to_string(i) + "_output_num = "+ std::to_string(output_num[i]) + ";\n";
        cCode += "float *npu_Subgraphs_" + std::to_string(i) + "_inputf[npu_Subgraphs_" + std::to_string(i) + "_input_num];\n";
        cCode += "int8_t *npu_Subgraphs_" + std::to_string(i) + "_input[npu_Subgraphs_" + std::to_string(i) + "_input_num];\n";
        cCode += "\nvoid *sess_npu_Subgraphs_"+ std::to_string(i) + " = create_Subgraphs(argv[根据实际填], csinn_npu_Subgraphs_" + std::to_string(i) + ");\n";
        cCode += "\nint input_size_npu_Subgraphs_" + std::to_string(i) + "[] = { };\n";
        cCode += "void *input_aligned_npu_Subgraphs_" + std::to_string(i) + "[npu_Subgraphs_" + std::to_string(i) + "_input_num];\n";
        cCode += "for (i = 0; i < npu_Subgraphs_" + std::to_string(i) + "_input_num; i++) {\n";
        cCode += "    input_size_npu_Subgraphs_" + std::to_string(i) + "[i] = csinn_tensor_byte_size(((struct csinn_session *)sess_npu_Subgraphs_" + std::to_string(i) + ")->input[i]);\n";
        cCode += "    input_aligned_npu_Subgraphs_" + std::to_string(i) + "[i] = shl_mem_alloc_aligned(input_size_npu_Subgraphs_" + std::to_string(i) + "[i], 0);\n";
        cCode += "}\n";

        cCode += "struct csinn_tensor* input_tensors_npu_Subgraphs_" + std::to_string(i) + "[npu_Subgraphs_" + std::to_string(i) + "_input_num];\n";
        for (size_t j = 0; j < input_num[i]; j++) {
            cCode += "input_tensors_npu_Subgraphs_" + std::to_string(i) + "[" + std::to_string(j) + "] = csinn_alloc_tensor(NULL);\n";
            cCode += "input_tensors_npu_Subgraphs_" + std::to_string(i) + "[" + std::to_string(j) + "]->dim_count = " + std::to_string(allshapes[i][j].size()) + ";\n";
            for (size_t k = 0; k < allshapes[i][j].size(); k++) {
                cCode += "input_tensors_npu_Subgraphs_" + std::to_string(i) + "[" + std::to_string(j) + "]->dim[" + std::to_string(k) + "] = " + std::to_string(allshapes[i][j][k]) + ";\n";
            }
        }

        cCode += "struct csinn_tensor* output_tensors_npu_Subgraphs_" + std::to_string(i) + "[npu_Subgraphs_" + std::to_string(i) + "_output_num];\n";
        for (size_t j = 0; j < output_num[i]; j++) {
            cCode += "output_tensors_npu_Subgraphs_" + std::to_string(i) + "[" + std::to_string(j) + "] = csinn_alloc_tensor(NULL);\n";
            cCode += "output_tensors_npu_Subgraphs_" + std::to_string(i) + "[" + std::to_string(j) + "]->dim_count = 根据实际情况;\n";
            cCode += "output_tensors_npu_Subgraphs_" + std::to_string(i) + "[" + std::to_string(j) + "]->dim[0] = 根据实际情况;\n";
            cCode += "output_tensors_npu_Subgraphs_" + std::to_string(i) + "[" + std::to_string(j) + "]->dim[1] = 根据实际情况;\n";
            cCode += "output_tensors_npu_Subgraphs_" + std::to_string(i) + "[" + std::to_string(j) + "]->dim[2] = 根据实际情况;\n";
            cCode += "output_tensors_npu_Subgraphs_" + std::to_string(i) + "[" + std::to_string(j) + "]->dim[3] = 根据实际情况;\n";
        }

        cCode += "\n";
    }

    // 打开.c文件
    std::ofstream file("output.c");

    // 将代码字符串写入到文件中
    file << cCode;

    // 关闭文件
    file.close();

    std::cout << "C file generated successfully." << std::endl;

    return 0;
}
