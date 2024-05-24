export PATH=/tools/Xuantie-900-gcc-linux-5.10.4-glibc-x86_64-V2.6.1-light.1/bin/:$PATH
rm -rf hhb_out/ 
rm -rf hhb_out_0/
# concat之前的用NPU
hhb -C --model-file test.onnx --data-scale-div 255 --board th1520 --postprocess save_and_top5 --input-name "vit_input" --output-name "vit_165" --input-shape "1 3 224 224" --calibrate-dataset persian_cat.jpg --quantization-scheme "int8_asym"
mv hhb_out/ hhb_out_0/
python3 data_gen.py
# concat之后的
hhb -C --model-file test.onnx --data-scale-div 255 --board c920 --postprocess save_and_top5 --input-name "vit_181" --output-name "vit_output" --input-shape "1 197 768" --calibrate-dataset vit_181.npz --quantization-scheme "float16"