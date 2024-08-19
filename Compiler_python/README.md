# 使用python完成tflite格式子图切割的步骤

**运行环境：ubuntu22.04**

所需指令已被集成到脚本中，具体位置为./run.sh。若要执行操作，只需将tflite文件添加到项目文件夹中，确保路径正确并执行如下指令即可<br>

    cd Compiler_python
    ./run.sh
    
**注：以下操作默认tflite文件路径为./net/converted\_diffusion\_model.tflite，如与实际位置不符，则需要更改run.sh中第一条指令的文件路径和/net/partition\_main.py中的tflite\_model\_path变量。**

run.sh中的指令具体作用：
------

`./flatc -t --strict-json --defaults-json -o workdir ./schema.fbs -- net/converted_diffusion_model.tflite`

将tflite格式的网络结构转换为json格式,tflite文件的路径为net/converted\_diffusion\_model.tflite，转换后的json文件将存储至workdir文件夹

`python3 extract_json_lib.py`

执行extract\_json\_lib.py，此文件将调用net/partition\_main.py获取具体切图方法，并将此前转换得到的json文件按规则切分为若干npu/cpu子图，格式依然为.json，储存在workdir/subgraphs文件夹中

	for i in {0..96}
	do
	  ./flatc -o workdir -b ./schema.fbs workdir/subgraphs/cpusubgraph"$i".json
	  ./flatc -o workdir -b ./schema.fbs workdir/subgraphs/npusubgraph"$i".json
	done
	for i in {97..123}
	do
	  ./flatc -o workdir -b ./schema.fbs workdir/subgraphs/npusubgraph"$i".json
	done
将生成的json文件转换为tflite格式，储存在workdir文件夹中。

示例用 https://box.nju.edu.cn/f/9a148fc831564a2f8273/ 里网络可切分为97个cpu子图和124个npu子图，如改变网络或切图规则，可运行net/partition\_main.py，依照生成的subgraphs_ios.txt获取子图数量，对run.sh中的循环次数进行更改
