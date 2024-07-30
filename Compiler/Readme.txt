NPU不支持batch_matmul所以非两个变量的matmul维度需要转成没有batch的

onnx api切图
使用onnx的api做切图的部分已经集成到代码里，具体体现在extract_onnx_lib.py,extract_onnx.py,extract_runcut_onnx.py
调用extract_onnx.py之间使用c++的分析结果切图
调用extract_runcut_onnx.py可读取runcut文件然后根据里面的结果切图


7.19 update
避免python c api的环境问题，直接使用system运行python 
run.sh里./CoComputingCompiler --onnx=xxx.onnx 需要自己更新网络名称
extract_onnx_lib.py里写有要拆分的网络的路径,input_path ='../case/sd/unet_32_sim.onnx'目前是直接写在python里，需要修改
extract_runcut_onnx.py是用来针对runcut文件的一个切分，需要在这个python里直接指定runcut文件名:split_runcut_onnx('../case/sd/runcut.sh'),和onnx路径:input_path ='../case/sd/unet_32_sim.onnx'


7.26 update
main函数里的python根据当前系统环境补全路径
修复最后一个图无法切出来的问题
在case文件夹里新增使用onnxruntime的fp32 case和fp32+int8的case
详见case文件夹 readme

7.30 update
同时新增case/inference/requirements.txt里面包含了相关库的版本。
新增rename_onnx.py用来将input和output中的./消除
休要修改的地方如下：
rename_onnx.py的第33行，directory_path = 'path/to/your/models'，这里的路径需要替换
rename_onnx.py的第34行，output_path = 'output_path/to/your/models'，这里的路径需要替换
同时修改main函数，现在不在c++里使用python，而是在run.sh里
