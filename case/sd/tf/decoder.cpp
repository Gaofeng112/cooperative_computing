#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <memory>
#include <cstring>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>
#include <opencv2/opencv.hpp> // Include OpenCV for image processing

std::vector<float> readInputFile(const std::string& filename) {
    std::ifstream file(filename);
    std::vector<float> input_data;
    float value;

    while (file >> value) {
        input_data.push_back(value);
    }

    return input_data;
}

std::vector<float> rearrangeInput(const std::vector<float>& input) {
    std::vector<float> rearranged(1 * 32 * 32 * 4);
    for (int i = 0; i < 32; ++i) {
        for (int j = 0; j < 32; ++j) {
            for (int k = 0; k < 4; ++k) {
                rearranged[i * 32 * 4 + j * 4 + k] = input[k * 32 * 32 + i * 32 + j];
            }
        }
    }
    return rearranged;
}

int main() {
    auto start = std::chrono::high_resolution_clock::now();

    // Read and rearrange input data
    std::vector<float> input_data = readInputFile("sample.bin_output0_1_4_32_32.txt");
    std::vector<float> rearranged_data = rearrangeInput(input_data);

    // Load TFLite model
    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile("converted_decoder.tflite");
    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> interpreter;

    tflite::InterpreterBuilder(*model, resolver)(&interpreter);
    interpreter->AllocateTensors();

    // Get input tensor and set data
    float* input_tensor = interpreter->typed_input_tensor<float>(0);
    std::memcpy(input_tensor, rearranged_data.data(), rearranged_data.size() * sizeof(float));

    // Run model
    if (interpreter->Invoke() != kTfLiteOk) {
        std::cerr << "Failed to invoke tflite model" << std::endl;
        return -1;
    }

    // Get output tensor
    float* output_tensor = interpreter->typed_output_tensor<float>(0);
    int output_size = interpreter->tensor(interpreter->outputs()[0])->bytes / sizeof(float);

    // Post-process the output tensor
    std::vector<float> decoded(output_tensor, output_tensor + output_size);
std::cout << "output_size:" << output_size << std::endl;
    // Assuming the output shape is [1, height, width, channels], reshape to match image format
    int height = 256; // Update this based on your model's output
    int width = 256;  // Update this based on your model's output
    int channels = 3; // Update this based on your model's output

    cv::Mat img(height, width, CV_32FC(channels), decoded.data());

    // Normalize and convert to 8-bit image
    img = ((img + 1.0) / 2.0) * 255.0;
    cv::Mat img_uint8;
    img.convertTo(img_uint8, CV_8UC(channels));
    cv::cvtColor(img_uint8, img_uint8, cv::COLOR_RGBA2BGRA);

    // Save image
    cv::imwrite("test.png", img_uint8);

    // Get end time
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate duration and convert to milliseconds
    std::chrono::duration<double, std::milli> duration = end - start;

    // Output duration
    std::cout << "Elapsed time: " << duration.count() << " ms" << std::endl;
    return 0;
}
