# #!/bin/bash
# 运行1000次普通推理脚本
#源代码已经生成了所需要的的目录

prompts_file="prompts.txt"
mapfile -t prompts < "$prompts_file"

# 检查提示数量是否足够
if [ ${#prompts[@]} -lt 1000 ]; then
    echo "Error: Not enough prompts provided. Please add more prompts."
    exit 1
fi

for i in {1..1000}; do
    prompt=${prompts[$((i-1))]}
    echo "Generating image $i with prompt: $prompt"

    image_name="origin_$i.png"
    python3 inference_onnx.py --prompt "$prompt" --output "$image_name"
    
    sleep 0.1  
done

# #!/bin/bash
# 运行1000次pca脚本
#源代码已经生成了所需要的的目录

# prompts_file="prompts.txt"
# mapfile -t prompts < "$prompts_file"

# # 检查提示数量是否足够
# if [ ${#prompts[@]} -lt 1000 ]; then
#     echo "Error: Not enough prompts provided. Please add more prompts."
#     exit 1
# fi

# for i in {1..1000}; do
#     prompt=${prompts[$((i-1))]}
#     echo "Generating image $i with prompt: $prompt"

#     image_name="pca_$i.png"
#     python3 inference_pca.py --prompt "$prompt" --output "$image_name" --num "$i"
    
#     sleep 0.1  
# done


#原始代码备份，不用看
# # #!/bin/bash
# # 运行1000次脚本
# # 定义输出目录
# OUTPUT_DIR="images_pca_1000"
# # OUTPUT_DIR="images_oringin_1000"


# mkdir -p "$OUTPUT_DIR"
# # 读取提示文件并按行存储到数组中
# prompts_file="prompts.txt"
# mapfile -t prompts < "$prompts_file"

# # 检查提示数量是否足够
# if [ ${#prompts[@]} -lt 1000 ]; then
#     echo "Error: Not enough prompts provided. Please add more prompts."
#     exit 1
# fi

# # 循环调用Python脚本1000次
# for i in {1..1000}; do
#     # 选择第i个提示
#     prompt=${prompts[$((i-1))]}
    
#     # 打印当前提示
#     echo "Generating image $i with prompt: $prompt"

#     #压缩
#     OUTPUT_FILE="$OUTPUT_DIR/pca_$i.png"
#     python3 inference_onnx_pca_auto_v2.py --prompt "$prompt" --output "$OUTPUT_FILE" --num "$i"
    
#     #非压缩
#     # OUTPUT_FILE="$OUTPUT_DIR/origin_$i.png"
#     # python3 inference_fp32_onnx_final.py --prompt "$prompt" --output "$OUTPUT_FILE"
    
#     # 可选：在每次迭代后添加短暂的延迟
#     sleep 0.1  # 等待0.1秒
# done
