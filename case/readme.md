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