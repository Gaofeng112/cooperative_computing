7.26 update
新增inference文件夹,
里面的inferece_fp32.py是全fp32推理 
inferece_fp32_int8.py是fp32+量化后的int8推理

使用方式：
运行Comiler，将compiler生成的subgraphs_ios.txt拷贝到inference文件夹，
在inference文件夹中新建sg文件夹,fp32文件夹
从https://box.nju.edu.cn/d/32b1e54c935646149474/下载模型放到fp32
将compiler中的sg/subgraphs_cpu和sg/subgraphs_npu文件夹里的所有onnx子图放到当前inference/sg
运行inferece_fp32.py或inferece_fp32_int8.py

7.30 update
int8的encoder，decoder如下：
https://box.nju.edu.cn/d/26c330148aeb4f5bae30/
下载模型放到inference/int8文件夹
修改inferece_fp32_int8.py和inferece_fp32.py中的
path_to_saved_models="./fp32"修改为path_to_saved_models="./int8"
则可以使用int8的encoder，decoder


8.19 update
在case/inference/里新增inference_mulit_tflite.py和inference_single_tflite.py
inference_single_tflite.py是运行新的单个unet tflite文件的
新增inference_mulit_tflite.py是运行新的多个unet tflite文件的
两个文件的运行时间只在数据中间数据传输过程有差别
inference_mulit_tflite.py和inference_single_tflite.py中的tflite的文件地址均需要修改
完整的新的tflite的下载地址https://box.nju.edu.cn/f/9a148fc831564a2f8273/
