rm -rf build
mkdir build
cd build
cmake ../
cmake --build . -- -j1
cp encoder ../../
