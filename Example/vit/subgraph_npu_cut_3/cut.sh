export PATH=/tools/Xuantie-900-gcc-linux-5.10.4-glibc-x86_64-V2.6.1-light.1/bin/:$PATH
rm -rf .pkl_memoize_py3/
rm -rf hhb_out_0/
rm -rf hhb_out_1/
rm -rf hhb_out_2/
python3 data_gen.py
python3 get_concat_input.py
# concat之前的用NPU
hhb -C --model-file test.onnx --data-scale-div 255 --board th1520 --postprocess save_and_top5 --input-name "vit_input" --output-name "vit_165" --input-shape "1 3 224 224" --calibrate-dataset persian_cat.jpg --quantization-scheme "int8_asym"
mv hhb_out/ hhb_out_0/
# concat之后的CPU
hhb -C --model-file test.onnx --data-scale-div 255 --board c920 --postprocess save_and_top5 --input-name "vit_181" --output-name "vit_1215" --input-shape "1 197 768" --calibrate-dataset vit_181.npz --quantization-scheme "float16"
mv hhb_out/ hhb_out_1/
# 最后的mul开始也是NPU
hhb -C --model-file test.onnx --data-scale-div 255 --board th1520 --postprocess save_and_top5 --input-name "vit_1215" --output-name "vit_output" --input-shape "1 768" --calibrate-dataset vit_1215.npz --quantization-scheme "int8_asym"
mv hhb_out/ hhb_out_2/