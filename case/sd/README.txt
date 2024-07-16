1.进入tf文件夹，运行run.sh
2.从以下地址下unet部分的onnx。https://box.nju.edu.cn/f/dcd9717c2dc54917a4ed/,从以下地址下载encoder和decoder的tflite:https://box.nju.edu.cn/d/25f88c9c3a20445983c1/
3.运行data_gen.py
4.修改runcut.sh中的onnx的位置，运行runcut.sh脚本
5.运行gen_exe.sh
6.将整个文件夹复制到licheepi
7.在licheepi下运行run.sh