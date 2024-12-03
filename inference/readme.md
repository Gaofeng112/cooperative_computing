#步骤1，保证当前目录有必要的文件
#quant.py pca代码，包含压缩函数，提供计算压缩率的必要数据
#inference_fp32_onnx_auto.py是推理多个onnx
#inference_onnx_pca_auto.py是带有pca压缩的推理代码
#prompts.txt是储存了coco数据集中前1000个prompt


#步骤2，生成1000个prompt推理后的图片
./gen_images.sh
#原理：脚本包含了，运行1000次inference_fp32_onnx_auto.py，每次使用一个prompt，
#生成图片保存到了images_oringin_1000文件夹中


#生成1000张pca压缩后的图片
./gen_images.sh
#把gen_images.sh脚本中pca部分的代码取消注释，把非pca压缩部分注释，
#即可实现生成1000pca压缩后的图片，生成图片保存在images_pca_1000文件夹中
#同时在./result文件还生成了1000个result_i.txt文件，代码自动生成（每个txt保存了每个卷积输出）


#步骤3，计算PSNR值
python iamges_MPS.py
#在py_lib下，在iamges_MPS.py填入images_oringin_1000文件夹和images_pca_1000文件夹的路径，然后运行iamges_MPS.py
#即可得到压缩后与压缩前图片的PSNR值,数据保存在image_quality_metris.xlsx文件中


#步骤5，计算压缩率
python comprate.py
#在py_lib下，运行comprate，作用是卷输出累加求平均
#压缩率数据保存在compression_rates.xlsx文件中，使用excel公式就可以得到平均压缩率

#步骤6，计算kid
./compare.sh
#运行compare.sh，就会得到kid值
#或者直接在命令行输入
fidelity --gpu 0 --kid --input1 "./images_origin_1000" --input2 "./images_pca_1000"

### 总结
这个过程是实现了pca压缩，并通过脚本生成了1000张图，比较图片的PSNR值，压缩率以及kid值
