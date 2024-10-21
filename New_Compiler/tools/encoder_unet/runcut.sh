export PATH=/tools/Xuantie-900-gcc-linux-5.10.4-glibc-x86_64-V2.6.1-light.1/bin/:$PATH
# hhb -D --model-file ../../net/unet_32_sim.onnx --data-scale-div 255 --board th1520 --postprocess save_and_top5 --input-name "sample" --output-name "/down_blocks.0/resnets.0/norm1/Reshape_output_0;/conv_in/Conv_output_0" --input-shape "1 4 32 32" --calibrate-dataset ./data/sample.npz --quantization-scheme int8_asym
# mv hhb_out/ npu_Subgraphs_0_out/
# hhb -D --model-file ../../net/unet_32_sim.onnx --data-scale-div 255 --board th1520 --postprocess save_and_top5 --input-name "/down_blocks.0/resnets.0/norm1/InstanceNormalization_output_0" --output-name "/down_blocks.0/resnets.0/conv1/Conv_output_0" --input-shape "1 32 10240" --calibrate-dataset ./data/npu_Subgraphs_1.npz --quantization-scheme int8_asym
# mv hhb_out/ npu_Subgraphs_1_out/
# hhb -D --model-file ../../net/unet_32_sim.onnx --data-scale-div 255 --board c920 --postprocess save_and_top5 --input-name "/down_blocks.0/resnets.0/conv1/Conv_output_0;timestep" --output-name "/down_blocks.0/resnets.0/norm2/Reshape_output_0;/down_blocks.0/resnets.0/act_1/Mul_output_0" --input-shape "1 320 32 32;1" --calibrate-dataset ./data/cpu_Subgraphs_0.npz --quantization-scheme int8_asym
# mv hhb_out/ cpu_Subgraphs_0_out/
# hhb -D --model-file ../../net/unet_32_sim.onnx --data-scale-div 255 --board th1520 --postprocess save_and_top5 --input-name "/down_blocks.0/resnets.0/norm2/InstanceNormalization_output_0;/conv_in/Conv_output_0" --output-name "/down_blocks.0/resnets.0/Add_1_output_0;/down_blocks.0/attentions.0/norm/Reshape_output_0" --input-shape "1 32 10240;1 320 32 32" --calibrate-dataset ./data/npu_Subgraphs_2.npz --quantization-scheme int8_asym
# mv hhb_out/ npu_Subgraphs_2_out/
# hhb -D --model-file ../../net/unet_32_sim.onnx --data-scale-div 255 --board th1520 --postprocess save_and_top5 --input-name "/down_blocks.0/attentions.0/norm/InstanceNormalization_output_0" --output-name "/down_blocks.0/attentions.0/proj_in/Conv_output_0" --input-shape "1 32 10240" --calibrate-dataset ./data/npu_Subgraphs_3.npz --quantization-scheme int8_asym
# mv hhb_out/ npu_Subgraphs_3_out/
# hhb -D --model-file ../../net/unet_32_sim.onnx --data-scale-div 255 --board c920 --postprocess save_and_top5 --input-name "encoder_hidden_states;/down_blocks.0/attentions.0/proj_in/Conv_output_0" --output-name "/down_blocks.0/attentions.0/Transpose_1_output_0" --input-shape "1 77 768;1 320 32 32" --calibrate-dataset ./data/cpu_Subgraphs_1.npz --quantization-scheme int8_asym
# mv hhb_out/ cpu_Subgraphs_1_out/
# hhb -D --model-file ../../net/unet_32_sim.onnx --data-scale-div 255 --board th1520 --postprocess save_and_top5 --input-name "/down_blocks.0/attentions.0/Transpose_1_output_0;/down_blocks.0/resnets.0/Add_1_output_0" --output-name "/down_blocks.0/attentions.0/Add_output_0;/down_blocks.0/downsamplers.0/conv/Conv_output_0;/down_blocks.1/resnets.0/norm1/Reshape_output_0;/down_blocks.1/resnets.0/conv_shortcut/Conv_output_0" --input-shape "1 320 32 32;1 320 32 32" --calibrate-dataset ./data/npu_Subgraphs_4.npz --quantization-scheme int8_asym
# mv hhb_out/ npu_Subgraphs_4_out/
# hhb -D --model-file ../../net/unet_32_sim.onnx --data-scale-div 255 --board th1520 --postprocess save_and_top5 --input-name "/down_blocks.1/resnets.0/norm1/InstanceNormalization_output_0;/down_blocks.0/resnets.0/act_1/Mul_output_0" --output-name "/down_blocks.1/resnets.0/norm2/Reshape_output_0" --input-shape "1 32 2560;1 1280" --calibrate-dataset ./data/npu_Subgraphs_5.npz --quantization-scheme int8_asym
# mv hhb_out/ npu_Subgraphs_5_out/
# hhb -D --model-file ../../net/unet_32_sim.onnx --data-scale-div 255 --board th1520 --postprocess save_and_top5 --input-name "/down_blocks.1/resnets.0/conv_shortcut/Conv_output_0;/down_blocks.1/resnets.0/norm2/InstanceNormalization_output_0" --output-name "/down_blocks.1/attentions.0/norm/Reshape_output_0;/down_blocks.1/resnets.0/Add_1_output_0" --input-shape "1 640 16 16;1 32 5120" --calibrate-dataset ./data/npu_Subgraphs_6.npz --quantization-scheme int8_asym
# mv hhb_out/ npu_Subgraphs_6_out/
# hhb -D --model-file ../../net/unet_32_sim.onnx --data-scale-div 255 --board th1520 --postprocess save_and_top5 --input-name "/down_blocks.1/attentions.0/norm/InstanceNormalization_output_0" --output-name "/down_blocks.1/attentions.0/proj_in/Conv_output_0" --input-shape "1 32 5120" --calibrate-dataset ./data/npu_Subgraphs_7.npz --quantization-scheme int8_asym
# mv hhb_out/ npu_Subgraphs_7_out/
# hhb -D --model-file ../../net/unet_32_sim.onnx --data-scale-div 255 --board c920 --postprocess save_and_top5 --input-name "encoder_hidden_states;/down_blocks.1/attentions.0/proj_in/Conv_output_0" --output-name "/down_blocks.1/attentions.0/Transpose_1_output_0" --input-shape "1 77 768;1 640 16 16" --calibrate-dataset ./data/cpu_Subgraphs_2.npz --quantization-scheme int8_asym
# mv hhb_out/ cpu_Subgraphs_2_out/
# hhb -D --model-file ../../net/unet_32_sim.onnx --data-scale-div 255 --board th1520 --postprocess save_and_top5 --input-name "/down_blocks.1/attentions.0/Transpose_1_output_0;/down_blocks.1/resnets.0/Add_1_output_0" --output-name "/down_blocks.1/attentions.0/Add_output_0;/down_blocks.1/downsamplers.0/conv/Conv_output_0;/down_blocks.2/resnets.0/norm1/Reshape_output_0" --input-shape "1 640 16 16;1 640 16 16" --calibrate-dataset ./data/npu_Subgraphs_8.npz --quantization-scheme int8_asym
# mv hhb_out/ npu_Subgraphs_8_out/
# hhb -D --model-file ../../net/unet_32_sim.onnx --data-scale-div 255 --board th1520 --postprocess save_and_top5 --input-name "/down_blocks.2/resnets.0/norm1/InstanceNormalization_output_0;/down_blocks.0/resnets.0/act_1/Mul_output_0" --output-name "/down_blocks.2/resnets.0/norm2/Reshape_output_0" --input-shape "1 32 1280;1 1280" --calibrate-dataset ./data/npu_Subgraphs_9.npz --quantization-scheme int8_asym
# mv hhb_out/ npu_Subgraphs_9_out/
# hhb -D --model-file ../../net/unet_32_sim.onnx --data-scale-div 255 --board th1520 --postprocess save_and_top5 --input-name "/down_blocks.1/downsamplers.0/conv/Conv_output_0;/down_blocks.2/resnets.0/norm2/InstanceNormalization_output_0" --output-name "/down_blocks.2/attentions.0/norm/Reshape_output_0;/down_blocks.2/resnets.0/Add_1_output_0" --input-shape "1 640 8 8;1 32 2560" --calibrate-dataset ./data/npu_Subgraphs_10.npz --quantization-scheme int8_asym
# mv hhb_out/ npu_Subgraphs_10_out/
# hhb -D --model-file ../../net/unet_32_sim.onnx --data-scale-div 255 --board th1520 --postprocess save_and_top5 --input-name "/down_blocks.2/attentions.0/norm/InstanceNormalization_output_0" --output-name "/down_blocks.2/attentions.0/proj_in/Conv_output_0" --input-shape "1 32 2560" --calibrate-dataset ./data/npu_Subgraphs_11.npz --quantization-scheme int8_asym
# mv hhb_out/ npu_Subgraphs_11_out/
# hhb -D --model-file ../../net/unet_32_sim.onnx --data-scale-div 255 --board c920 --postprocess save_and_top5 --input-name "encoder_hidden_states;/down_blocks.2/attentions.0/proj_in/Conv_output_0" --output-name "/down_blocks.2/attentions.0/Transpose_1_output_0" --input-shape "1 77 768;1 1280 8 8" --calibrate-dataset ./data/cpu_Subgraphs_3.npz --quantization-scheme int8_asym
# mv hhb_out/ cpu_Subgraphs_3_out/
# hhb -D --model-file ../../net/unet_32_sim.onnx --data-scale-div 255 --board th1520 --postprocess save_and_top5 --input-name "/down_blocks.2/attentions.0/Transpose_1_output_0;/down_blocks.2/resnets.0/Add_1_output_0" --output-name "/up_blocks.0/resnets.0/norm1/Reshape_output_0;/up_blocks.0/resnets.0/conv_shortcut/Conv_output_0" --input-shape "1 1280 8 8;1 1280 8 8" --calibrate-dataset ./data/npu_Subgraphs_12.npz --quantization-scheme int8_asym
# mv hhb_out/ npu_Subgraphs_12_out/
# hhb -D --model-file ../../net/unet_32_sim.onnx --data-scale-div 255 --board th1520 --postprocess save_and_top5 --input-name "/up_blocks.0/resnets.0/norm1/InstanceNormalization_output_0;/down_blocks.0/resnets.0/act_1/Mul_output_0" --output-name "/up_blocks.0/resnets.0/norm2/Reshape_output_0" --input-shape "1 32 5120;1 1280" --calibrate-dataset ./data/npu_Subgraphs_13.npz --quantization-scheme int8_asym
# mv hhb_out/ npu_Subgraphs_13_out/
# hhb -D --model-file ../../net/unet_32_sim.onnx --data-scale-div 255 --board th1520 --postprocess save_and_top5 --input-name "/up_blocks.0/resnets.0/conv_shortcut/Conv_output_0;/up_blocks.0/resnets.0/norm2/InstanceNormalization_output_0" --output-name "/up_blocks.0/attentions.0/norm/Reshape_output_0;/up_blocks.0/resnets.0/Add_1_output_0" --input-shape "1 1280 8 8;1 32 2560" --calibrate-dataset ./data/npu_Subgraphs_14.npz --quantization-scheme int8_asym
# mv hhb_out/ npu_Subgraphs_14_out/
# hhb -D --model-file ../../net/unet_32_sim.onnx --data-scale-div 255 --board th1520 --postprocess save_and_top5 --input-name "/up_blocks.0/attentions.0/norm/InstanceNormalization_output_0" --output-name "/up_blocks.0/attentions.0/proj_in/Conv_output_0" --input-shape "1 32 2560" --calibrate-dataset ./data/npu_Subgraphs_15.npz --quantization-scheme int8_asym
# mv hhb_out/ npu_Subgraphs_15_out/
# hhb -D --model-file ../../net/unet_32_sim.onnx --data-scale-div 255 --board c920 --postprocess save_and_top5 --input-name "encoder_hidden_states;/up_blocks.0/attentions.0/proj_in/Conv_output_0" --output-name "/up_blocks.0/attentions.0/Transpose_1_output_0" --input-shape "1 77 768;1 1280 8 8" --calibrate-dataset ./data/cpu_Subgraphs_4.npz --quantization-scheme int8_asym
# mv hhb_out/ cpu_Subgraphs_4_out/
# hhb -D --model-file ../../net/unet_32_sim.onnx --data-scale-div 255 --board th1520 --postprocess save_and_top5 --input-name "/up_blocks.0/attentions.0/Transpose_1_output_0;/up_blocks.0/resnets.0/Add_1_output_0" --output-name "/up_blocks.0/attentions.0/Add_output_0" --input-shape "1 1280 8 8;1 1280 8 8" --calibrate-dataset ./data/npu_Subgraphs_16.npz --quantization-scheme int8_asym
# mv hhb_out/ npu_Subgraphs_16_out/
# hhb -D --model-file ../../net/unet_32_sim.onnx --data-scale-div 255 --board th1520 --postprocess save_and_top5 --input-name "/up_blocks.0/resnets.1/norm1/InstanceNormalization_output_0;/down_blocks.0/resnets.0/act_1/Mul_output_0" --output-name "/up_blocks.0/resnets.1/norm2/Reshape_output_0" --input-shape "1 32 3840;1 1280" --calibrate-dataset ./data/npu_Subgraphs_17.npz --quantization-scheme int8_asym
# mv hhb_out/ npu_Subgraphs_17_out/
# hhb -D --model-file ../../net/unet_32_sim.onnx --data-scale-div 255 --board th1520 --postprocess save_and_top5 --input-name "/up_blocks.0/resnets.1/norm2/InstanceNormalization_output_0;/up_blocks.0/Concat_1_output_0" --output-name "/up_blocks.0/attentions.1/norm/Reshape_output_0;/up_blocks.0/resnets.1/Add_1_output_0" --input-shape "1 32 2560;1 1920 8 8" --calibrate-dataset ./data/npu_Subgraphs_18.npz --quantization-scheme int8_asym
# mv hhb_out/ npu_Subgraphs_18_out/
# hhb -D --model-file ../../net/unet_32_sim.onnx --data-scale-div 255 --board th1520 --postprocess save_and_top5 --input-name "/up_blocks.0/attentions.1/norm/InstanceNormalization_output_0" --output-name "/up_blocks.0/attentions.1/proj_in/Conv_output_0" --input-shape "1 32 2560" --calibrate-dataset ./data/npu_Subgraphs_19.npz --quantization-scheme int8_asym
# mv hhb_out/ npu_Subgraphs_19_out/
# hhb -D --model-file ../../net/unet_32_sim.onnx --data-scale-div 255 --board c920 --postprocess save_and_top5 --input-name "encoder_hidden_states;/up_blocks.0/attentions.1/proj_in/Conv_output_0" --output-name "/up_blocks.0/attentions.1/Transpose_1_output_0" --input-shape "1 77 768;1 1280 8 8" --calibrate-dataset ./data/cpu_Subgraphs_5.npz --quantization-scheme int8_asym
# mv hhb_out/ cpu_Subgraphs_5_out/
# hhb -D --model-file ../../net/unet_32_sim.onnx --data-scale-div 255 --board th1520 --postprocess save_and_top5 --input-name "/up_blocks.0/attentions.1/Transpose_1_output_0;/up_blocks.0/resnets.1/Add_1_output_0" --output-name "/up_blocks.0/upsamplers.0/conv/Conv_output_0" --input-shape "1 1280 8 8;1 1280 8 8" --calibrate-dataset ./data/npu_Subgraphs_20.npz --quantization-scheme int8_asym
# mv hhb_out/ npu_Subgraphs_20_out/
# hhb -D --model-file ../../net/unet_32_sim.onnx --data-scale-div 255 --board th1520 --postprocess save_and_top5 --input-name "/down_blocks.0/resnets.0/act_1/Mul_output_0" --output-name "/up_blocks.1/resnets.0/Unsqueeze_1_output_0" --input-shape "1 1280" --calibrate-dataset ./data/npu_Subgraphs_21.npz --quantization-scheme int8_asym
# mv hhb_out/ npu_Subgraphs_21_out/
# hhb -D --model-file ../../net/unet_32_sim.onnx --data-scale-div 255 --board th1520 --postprocess save_and_top5 --input-name "/up_blocks.1/resnets.0/norm1/InstanceNormalization_output_0" --output-name "/up_blocks.1/resnets.0/conv1/Conv_output_0" --input-shape "1 32 15360" --calibrate-dataset ./data/npu_Subgraphs_22.npz --quantization-scheme int8_asym
# mv hhb_out/ npu_Subgraphs_22_out/
# hhb -D --model-file ../../net/unet_32_sim.onnx --data-scale-div 255 --board th1520 --postprocess save_and_top5 --input-name "/up_blocks.1/Concat_output_0;/up_blocks.1/resnets.0/norm2/InstanceNormalization_output_0" --output-name "/up_blocks.1/attentions.0/norm/Reshape_output_0;/up_blocks.1/resnets.0/Add_1_output_0" --input-shape "1 1920 16 16;1 32 5120" --calibrate-dataset ./data/npu_Subgraphs_23.npz --quantization-scheme int8_asym
# mv hhb_out/ npu_Subgraphs_23_out/
# hhb -D --model-file ../../net/unet_32_sim.onnx --data-scale-div 255 --board th1520 --postprocess save_and_top5 --input-name "/up_blocks.1/attentions.0/norm/InstanceNormalization_output_0" --output-name "/up_blocks.1/attentions.0/proj_in/Conv_output_0" --input-shape "1 32 5120" --calibrate-dataset ./data/npu_Subgraphs_24.npz --quantization-scheme int8_asym
# mv hhb_out/ npu_Subgraphs_24_out/
# hhb -D --model-file ../../net/unet_32_sim.onnx --data-scale-div 255 --board c920 --postprocess save_and_top5 --input-name "encoder_hidden_states;/up_blocks.1/attentions.0/proj_in/Conv_output_0" --output-name "/up_blocks.1/attentions.0/Transpose_1_output_0" --input-shape "1 77 768;1 640 16 16" --calibrate-dataset ./data/cpu_Subgraphs_6.npz --quantization-scheme int8_asym
# mv hhb_out/ cpu_Subgraphs_6_out/
# hhb -D --model-file ../../net/unet_32_sim.onnx --data-scale-div 255 --board th1520 --postprocess save_and_top5 --input-name "/up_blocks.1/attentions.0/Transpose_1_output_0;/up_blocks.1/resnets.0/Add_1_output_0" --output-name "/up_blocks.1/attentions.0/Add_output_0" --input-shape "1 640 16 16;1 640 16 16" --calibrate-dataset ./data/npu_Subgraphs_25.npz --quantization-scheme int8_asym
# mv hhb_out/ npu_Subgraphs_25_out/
# hhb -D --model-file ../../net/unet_32_sim.onnx --data-scale-div 255 --board th1520 --postprocess save_and_top5 --input-name "/down_blocks.0/resnets.0/act_1/Mul_output_0" --output-name "/up_blocks.1/resnets.1/Unsqueeze_1_output_0" --input-shape "1 1280" --calibrate-dataset ./data/npu_Subgraphs_26.npz --quantization-scheme int8_asym
# mv hhb_out/ npu_Subgraphs_26_out/
# hhb -D --model-file ../../net/unet_32_sim.onnx --data-scale-div 255 --board th1520 --postprocess save_and_top5 --input-name "/up_blocks.1/resnets.1/norm1/InstanceNormalization_output_0" --output-name "/up_blocks.1/resnets.1/conv1/Conv_output_0" --input-shape "1 32 7680" --calibrate-dataset ./data/npu_Subgraphs_27.npz --quantization-scheme int8_asym
# mv hhb_out/ npu_Subgraphs_27_out/
# hhb -D --model-file ../../net/unet_32_sim.onnx --data-scale-div 255 --board th1520 --postprocess save_and_top5 --input-name "/up_blocks.1/Concat_1_output_0;/up_blocks.1/resnets.1/norm2/InstanceNormalization_output_0" --output-name "/up_blocks.1/attentions.1/norm/Reshape_output_0;/up_blocks.1/resnets.1/Add_1_output_0" --input-shape "1 960 16 16;1 32 5120" --calibrate-dataset ./data/npu_Subgraphs_28.npz --quantization-scheme int8_asym
# mv hhb_out/ npu_Subgraphs_28_out/
# hhb -D --model-file ../../net/unet_32_sim.onnx --data-scale-div 255 --board th1520 --postprocess save_and_top5 --input-name "/up_blocks.1/attentions.1/norm/InstanceNormalization_output_0" --output-name "/up_blocks.1/attentions.1/proj_in/Conv_output_0" --input-shape "1 32 5120" --calibrate-dataset ./data/npu_Subgraphs_29.npz --quantization-scheme int8_asym
# mv hhb_out/ npu_Subgraphs_29_out/
# hhb -D --model-file ../../net/unet_32_sim.onnx --data-scale-div 255 --board c920 --postprocess save_and_top5 --input-name "encoder_hidden_states;/up_blocks.1/attentions.1/proj_in/Conv_output_0" --output-name "/up_blocks.1/attentions.1/Transpose_1_output_0" --input-shape "1 77 768;1 640 16 16" --calibrate-dataset ./data/cpu_Subgraphs_7.npz --quantization-scheme int8_asym
# mv hhb_out/ cpu_Subgraphs_7_out/
# hhb -D --model-file ../../net/unet_32_sim.onnx --data-scale-div 255 --board th1520 --postprocess save_and_top5 --input-name "/up_blocks.1/attentions.1/Transpose_1_output_0;/up_blocks.1/resnets.1/Add_1_output_0" --output-name "/up_blocks.1/upsamplers.0/conv/Conv_output_0" --input-shape "1 640 16 16;1 640 16 16" --calibrate-dataset ./data/npu_Subgraphs_30.npz --quantization-scheme int8_asym
# mv hhb_out/ npu_Subgraphs_30_out/
# hhb -D --model-file ../../net/unet_32_sim.onnx --data-scale-div 255 --board th1520 --postprocess save_and_top5 --input-name "/down_blocks.0/resnets.0/act_1/Mul_output_0" --output-name "/up_blocks.2/resnets.0/Unsqueeze_1_output_0" --input-shape "1 1280" --calibrate-dataset ./data/npu_Subgraphs_31.npz --quantization-scheme int8_asym
# mv hhb_out/ npu_Subgraphs_31_out/
# hhb -D --model-file ../../net/unet_32_sim.onnx --data-scale-div 255 --board th1520 --postprocess save_and_top5 --input-name "/up_blocks.2/resnets.0/norm1/InstanceNormalization_output_0" --output-name "/up_blocks.2/resnets.0/conv1/Conv_output_0" --input-shape "1 32 30720" --calibrate-dataset ./data/npu_Subgraphs_32.npz --quantization-scheme int8_asym
# mv hhb_out/ npu_Subgraphs_32_out/
# hhb -D --model-file ../../net/unet_32_sim.onnx --data-scale-div 255 --board th1520 --postprocess save_and_top5 --input-name "/up_blocks.2/Concat_output_0;/up_blocks.2/resnets.0/norm2/InstanceNormalization_output_0" --output-name "/up_blocks.2/attentions.0/norm/Reshape_output_0;/up_blocks.2/resnets.0/Add_1_output_0" --input-shape "1 960 32 32;1 32 10240" --calibrate-dataset ./data/npu_Subgraphs_33.npz --quantization-scheme int8_asym
# mv hhb_out/ npu_Subgraphs_33_out/
# hhb -D --model-file ../../net/unet_32_sim.onnx --data-scale-div 255 --board th1520 --postprocess save_and_top5 --input-name "/up_blocks.2/attentions.0/norm/InstanceNormalization_output_0" --output-name "/up_blocks.2/attentions.0/proj_in/Conv_output_0" --input-shape "1 32 10240" --calibrate-dataset ./data/npu_Subgraphs_34.npz --quantization-scheme int8_asym
# mv hhb_out/ npu_Subgraphs_34_out/
# hhb -D --model-file ../../net/unet_32_sim.onnx --data-scale-div 255 --board c920 --postprocess save_and_top5 --input-name "encoder_hidden_states;/up_blocks.2/attentions.0/proj_in/Conv_output_0" --output-name "/up_blocks.2/attentions.0/Transpose_1_output_0" --input-shape "1 77 768;1 320 32 32" --calibrate-dataset ./data/cpu_Subgraphs_8.npz --quantization-scheme int8_asym
# mv hhb_out/ cpu_Subgraphs_8_out/
# hhb -D --model-file ../../net/unet_32_sim.onnx --data-scale-div 255 --board th1520 --postprocess save_and_top5 --input-name "/up_blocks.2/attentions.0/Transpose_1_output_0;/up_blocks.2/resnets.0/Add_1_output_0" --output-name "/up_blocks.2/attentions.0/Add_output_0" --input-shape "1 320 32 32;1 320 32 32" --calibrate-dataset ./data/npu_Subgraphs_35.npz --quantization-scheme int8_asym
# mv hhb_out/ npu_Subgraphs_35_out/
# hhb -D --model-file ../../net/unet_32_sim.onnx --data-scale-div 255 --board th1520 --postprocess save_and_top5 --input-name "/down_blocks.0/resnets.0/act_1/Mul_output_0" --output-name "/up_blocks.2/resnets.1/Unsqueeze_1_output_0" --input-shape "1 1280" --calibrate-dataset ./data/npu_Subgraphs_36.npz --quantization-scheme int8_asym
# mv hhb_out/ npu_Subgraphs_36_out/
# hhb -D --model-file ../../net/unet_32_sim.onnx --data-scale-div 255 --board th1520 --postprocess save_and_top5 --input-name "/up_blocks.2/resnets.1/norm1/InstanceNormalization_output_0" --output-name "/up_blocks.2/resnets.1/conv1/Conv_output_0" --input-shape "1 32 20480" --calibrate-dataset ./data/npu_Subgraphs_37.npz --quantization-scheme int8_asym
# mv hhb_out/ npu_Subgraphs_37_out/
# hhb -D --model-file ../../net/unet_32_sim.onnx --data-scale-div 255 --board th1520 --postprocess save_and_top5 --input-name "/up_blocks.2/Concat_1_output_0;/up_blocks.2/resnets.1/norm2/InstanceNormalization_output_0" --output-name "/up_blocks.2/attentions.1/norm/Reshape_output_0;/up_blocks.2/resnets.1/Add_1_output_0" --input-shape "1 640 32 32;1 32 10240" --calibrate-dataset ./data/npu_Subgraphs_38.npz --quantization-scheme int8_asym
# mv hhb_out/ npu_Subgraphs_38_out/
# hhb -D --model-file ../../net/unet_32_sim.onnx --data-scale-div 255 --board th1520 --postprocess save_and_top5 --input-name "/up_blocks.2/attentions.1/norm/InstanceNormalization_output_0" --output-name "/up_blocks.2/attentions.1/proj_in/Conv_output_0" --input-shape "1 32 10240" --calibrate-dataset ./data/npu_Subgraphs_39.npz --quantization-scheme int8_asym
# mv hhb_out/ npu_Subgraphs_39_out/
# hhb -D --model-file ../../net/unet_32_sim.onnx --data-scale-div 255 --board c920 --postprocess save_and_top5 --input-name "encoder_hidden_states;/up_blocks.2/attentions.1/proj_in/Conv_output_0" --output-name "/up_blocks.2/attentions.1/Transpose_1_output_0" --input-shape "1 77 768;1 320 32 32" --calibrate-dataset ./data/cpu_Subgraphs_9.npz --quantization-scheme int8_asym
# mv hhb_out/ cpu_Subgraphs_9_out/
# hhb -D --model-file ../../net/unet_32_sim.onnx --data-scale-div 255 --board th1520 --postprocess save_and_top5 --input-name "/up_blocks.2/attentions.1/Transpose_1_output_0;/up_blocks.2/resnets.1/Add_1_output_0" --output-name "/conv_norm_out/Reshape_output_0" --input-shape "1 320 32 32;1 320 32 32" --calibrate-dataset ./data/npu_Subgraphs_40.npz --quantization-scheme int8_asym
# mv hhb_out/ npu_Subgraphs_40_out/
# hhb -D --model-file ../../net/unet_32_sim.onnx --data-scale-div 255 --board th1520 --postprocess save_and_top5 --input-name "/conv_norm_out/InstanceNormalization_output_0" --output-name "3089" --input-shape "1 32 10240" --calibrate-dataset ./data/npu_Subgraphs_41.npz --quantization-scheme int8_asym
# mv hhb_out/ npu_Subgraphs_41_out/
# python3 modify_model_c.py