# cooperative_computing
This is riscv cpu + npu co-computing

Compiler folder is the graph cutting code.
Example folder is test case in licheepi.
Tools folder has shape_inference.py


##8.19 update
在case/inference/里新增inference_mulit_tflite.py和inference_single_tflite.py
inference_single_tflite.py是运行新的单个unet tflite文件的
新增inference_mulit_tflite.py是运行新的多个unet tflite文件的
两个文件的运行时间只在数据中间数据传输过程有差别
inference_mulit_tflite.py和inference_single_tflite.py中的tflite的文件地址均需要修改
完整的新的tflite的下载地址https://box.nju.edu.cn/f/9a148fc831564a2f8273/

新增Compiler_python，里面是对tflite进行分析拆分的代码，里面有详细的readme

新增New_Compiler，里面是新的切图算法

##11.18 update
已加入在config.json中大致调整子图文件大小上限的接口，可以通过改变hardware_limits中的max_subgraph_size实现，其单位为kb。

若要进行全CPU子图边界条件测试，请将config.json文件中的NPU_supported_ops和performance_data清空。

若要进行全NPU子图边界条件测试，请将include/Device/Device.h的Device类构造函数中NPUPreferOp、NPUSupportOp的初始化代码替换为下方注释的部分（**注意**：全NPU子图测试需要确保图中所有算子都在NPUPreferOp和NPUSupportOp中，而注释掉的部分仅为测试中出现过的算子，故若测试用onnx文件中存在其中未包含的算子，这些算子将被切分为CPU子图，故为确保全部子图都为NPU子图，需要手动将这些算子添加到NPUPreferOp中）

程序中所有循环都有退出机制，不会出现死循环，故若运行时间较长请耐心等待数分钟。