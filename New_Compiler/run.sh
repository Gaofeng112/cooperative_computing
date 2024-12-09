rm -rf build
mkdir build
cd build
cmake -G "Unix Makefiles" ../  #可加选项-DENABLE_TEST_OPTION
cmake --build . -- -j8
cd ..
find . -type f -name "*.txt" ! -name "CMakeLists.txt" -exec rm {} +
#./CoComputingCompiler --onnx=./net/vit_large_simplify.onnx
#./CoComputingCompiler --onnx=./net/vision_model_simplify.onnx
./CoComputingCompiler --onnx=./net/generation_model_simplify.onnx
#./CoComputingCompiler --onnx=./net/unet_32_sim_v2.onnx
#python3 extract_onnx.py