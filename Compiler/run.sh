rm -rf build
mkdir build
cd build
cmake -G "Unix Makefiles" ../
cmake --build . -- -j8
cd ..
./CoComputingCompiler --onnx=xxx.onnx