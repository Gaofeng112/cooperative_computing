1.将tf文件夹内的两个cpp，利用tflite api生成encoder和decoder两个可执行文件，并放在最外层文件夹
2.从以下地址下unet部分的onnx。https://box.nju.edu.cn/f/dcd9717c2dc54917a4ed/
3.运行data_gen.py
4.修改runcut.sh中的onnx的位置，运行runcut.sh脚本
5.运行gen_exe.sh
6.将整个文件夹复制到licheepi
7.在licheepi下运行run.sh