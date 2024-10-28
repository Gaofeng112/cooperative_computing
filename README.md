# cooperative_computing
This is riscv cpu + npu co-computing

Compiler folder is the graph cutting code.
Example folder is test case in licheepi.
Tools folder has shape_inference.py


8.19 update
在case/inference/里新增inference_mulit_tflite.py和inference_single_tflite.py
inference_single_tflite.py是运行新的单个unet tflite文件的
新增inference_mulit_tflite.py是运行新的多个unet tflite文件的
两个文件的运行时间只在数据中间数据传输过程有差别
inference_mulit_tflite.py和inference_single_tflite.py中的tflite的文件地址均需要修改
完整的新的tflite的下载地址https://box.nju.edu.cn/f/9a148fc831564a2f8273/

新增Compiler_python，里面是对tflite进行分析拆分的代码，里面有详细的readme

<<<<<<< Updated upstream
新增New_Compiler，里面是新的切图算法
=======
8.26 update
在inference/里新增inference_fp32_onnx.py
该文件是运行多个onnx文件
>>>>>>> Stashed changes
