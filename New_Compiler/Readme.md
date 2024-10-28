# 使用python完成tflite格式子图切割的步骤

**运行环境：ubuntu22.04**

所需指令已被集成到脚本中，具体位置为New\_Compiler/run.sh。若要执行操作，只需将tflite文件添加到项目文件夹中，确保路径正确并执行如下指令即可<br>

    cd New_Compiler
    ./run.sh
    
切图策略由config.json定制，通过main函数中调用的GetDeviceJson函数读取，该文件格式按文件夹中示例所示，其中项目各自意义如下：

**"NPU/CPU_supported_ops"**：获取NPU/CPU支持的算子

**performance_data**：算子在NPU/CPU上各自执行时间（若不支持，时间为-1），若在NPU上执行更快，则标记为NPUPreferOp

**"max_subgraphs"**：子图最多包含节点数量

**"max_subgraph_size"**：子图数量最大值（暂未支持）
