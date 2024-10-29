rm -rf build
mkdir build
cd build
cmake -G "Unix Makefiles" ../  #可加选项-DENABLE_TEST_OPTION
cmake --build . -- -j8
cd ..
# ./CoComputingCompiler --onnx=./net/vit_1.onnx
# ./CoComputingCompiler --onnx=./net/model.onnx
./CoComputingCompiler --onnx=./net/unet_32_sim_v2.onnx
# ./CoComputingCompiler --onnx=./net/unet_32_sim_v2.onnx
python3 extract_onnx.py