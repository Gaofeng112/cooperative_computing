rm -rf build
mkdir build
cd build
cmake -G "Unix Makefiles" ../
cmake --build . -- -j8
cd ..
./CoComputingCompiler --onnx=xxx.onnx
python3 extract_onnx.py
python3 rename_onnx.py