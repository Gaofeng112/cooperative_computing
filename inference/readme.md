#步骤1，首先将pca代码保存到当前目录中
quant.py

#我们有两个py文件，inference_fp32_onnx_final.py和inference_onnx_pca_auto_v2.py
#inference_fp32_onnx_final.py是推理多个onnx
inference_onnx_pca_auto_v2.py是带有pca压缩的推理代码

#步骤2，首先生成带有1000prompt的prompt.txt文件，使用gen_prompt.sh脚本
./gen_prompt.sh

#步骤3，运行1000次inference_fp32_onnx_final.py，恶魔，每次使用一个prompt，使用gen_images.sh脚本
#生成图片保存到了images_oringin文件夹中
./gen_images.sh

#把gen_images.sh脚本中pca部分的代码取消注释，即可实现生成1000pca压缩后的图片
#生成图片保存在imahes_pca文件夹中
#在result文件还生成了1000个result_i.txt文件
./gen_images.sh

#步骤4，在py_lib下，在iamges_MPS.py填入images_oringin文件夹和imahes_pca文件夹的路径，然后运行iamges_MPS.py
#即可得到压缩后与压缩前图片的PSNR值
python iamges_MPS.py

#步骤5，在py_lib下，运行comprate_v1，就会在compression_rates输出压缩率
python comprate_v1.py

#步骤6，运行compare.sh，就会得到fid值
### 总结

这个过程是实现了pca压缩，并通过脚本生成了1000张图，比较图片的PSNR值，压缩率以及fid值
