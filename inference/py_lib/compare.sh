# fidelity --gpu 1 --fid --input1 ./result_16bits --input2 ./result_fp
# fidelity --gpu 1 --fid --input1 ./result_8bits --input2 ./result_fp
fidelity --gpu 1 --fid --input1 ../images_oringin_1000 --input2 ../images_pca_1000
# fidelity --gpu 1 --fid --input1 ./result_comp --input2 ./result_fp
# 结果 fid 小 (10%)      is 大