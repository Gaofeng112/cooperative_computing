export PATH=/tools/Xuantie-900-gcc-linux-5.10.4-glibc-x86_64-V2.6.1-light.1/bin/:$PATH

# npu_subgraph_0 + npu_subgraph_1 + cpu_subgraph_0 + npu_subgraph_2 + npu_subgraph_3 + cpu_subgraph_1 + npu_subgraph_4 + npu_subgraph_5 + npu_subgraph_6 + npu_subgraph_7 + cpu_subgraph_2 + npu_subgraph_8 + npu_subgraph_9 + npu_subgraph_10 + npu_subgraph_11 + cpu_subgraph_3 + npu_subgraph_12 + npu_subgraph_13 + npu_subgraph_14 + npu_subgraph_15 + cpu_subgraph_4 + npu_subgraph_16 + npu_subgraph_17 + npu_subgraph_18 + npu_subgraph_19 + cpu_subgraph_5 + npu_subgraph_20 + npu_subgraph_21 + npu_subgraph_22 + npu_subgraph_23 + npu_subgraph_24 + cpu_subgraph_6 + npu_subgraph_25 + npu_subgraph_26 + npu_subgraph_27 + npu_subgraph_28 + npu_subgraph_29 + cpu_subgraph_7 + npu_subgraph_30 + npu_subgraph_31 + npu_subgraph_32 + npu_subgraph_33 + npu_subgraph_34 + cpu_subgraph_8 + npu_subgraph_35 + npu_subgraph_36 + npu_subgraph_37 + npu_subgraph_38 + npu_subgraph_39 + cpu_subgraph_9 + npu_subgraph_40 + npu_subgraph_41
riscv64-unknown-linux-gnu-gcc main_final_with_schedule.c -o main_example npu_Subgraphs_0_out/io.c npu_Subgraphs_0_out/model.c npu_Subgraphs_1_out/model.c cpu_Subgraphs_0_out/model.c npu_Subgraphs_2_out/model.c npu_Subgraphs_3_out/model.c cpu_Subgraphs_1_out/model.c npu_Subgraphs_4_out/model.c npu_Subgraphs_5_out/model.c npu_Subgraphs_6_out/model.c npu_Subgraphs_7_out/model.c cpu_Subgraphs_2_out/model.c npu_Subgraphs_8_out/model.c npu_Subgraphs_9_out/model.c npu_Subgraphs_10_out/model.c npu_Subgraphs_11_out/model.c cpu_Subgraphs_3_out/model.c npu_Subgraphs_12_out/model.c npu_Subgraphs_13_out/model.c npu_Subgraphs_14_out/model.c npu_Subgraphs_15_out/model.c cpu_Subgraphs_4_out/model.c npu_Subgraphs_16_out/model.c npu_Subgraphs_17_out/model.c npu_Subgraphs_18_out/model.c npu_Subgraphs_19_out/model.c cpu_Subgraphs_5_out/model.c npu_Subgraphs_20_out/model.c npu_Subgraphs_21_out/model.c npu_Subgraphs_22_out/model.c npu_Subgraphs_23_out/model.c npu_Subgraphs_24_out/model.c cpu_Subgraphs_6_out/model.c npu_Subgraphs_25_out/model.c npu_Subgraphs_26_out/model.c npu_Subgraphs_27_out/model.c npu_Subgraphs_28_out/model.c npu_Subgraphs_29_out/model.c cpu_Subgraphs_7_out/model.c npu_Subgraphs_30_out/model.c npu_Subgraphs_31_out/model.c npu_Subgraphs_32_out/model.c npu_Subgraphs_33_out/model.c npu_Subgraphs_34_out/model.c cpu_Subgraphs_8_out/model.c npu_Subgraphs_35_out/model.c npu_Subgraphs_36_out/model.c npu_Subgraphs_37_out/model.c npu_Subgraphs_38_out/model.c npu_Subgraphs_39_out/model.c cpu_Subgraphs_9_out/model.c npu_Subgraphs_40_out/model.c npu_Subgraphs_41_out/model.c\
            -Wl,--gc-sections -O2 -g -mabi=lp64d -lm \
            -I npu_Subgraphs_0_out/ -L /usr/local/lib/python3.8/dist-packages/hhb/install_nn2/th1520/lib/ \
            -I npu_Subgraphs_1_out/ -L /usr/local/lib/python3.8/dist-packages/hhb/install_nn2/th1520/lib/ \
            -I cpu_Subgraphs_0_out/ -L /usr/local/lib/python3.8/dist-packages/hhb/install_nn2/c920/lib/ \
            -I npu_Subgraphs_2_out/ -L /usr/local/lib/python3.8/dist-packages/hhb/install_nn2/th1520/lib/ \
            -I npu_Subgraphs_3_out/ -L /usr/local/lib/python3.8/dist-packages/hhb/install_nn2/th1520/lib/ \
            -I cpu_Subgraphs_1_out/ -L /usr/local/lib/python3.8/dist-packages/hhb/install_nn2/c920/lib/ \
            -I npu_Subgraphs_4_out/ -L /usr/local/lib/python3.8/dist-packages/hhb/install_nn2/th1520/lib/ \
            -I npu_Subgraphs_5_out/ -L /usr/local/lib/python3.8/dist-packages/hhb/install_nn2/th1520/lib/ \
            -I npu_Subgraphs_6_out/ -L /usr/local/lib/python3.8/dist-packages/hhb/install_nn2/th1520/lib/ \
            -I npu_Subgraphs_7_out/ -L /usr/local/lib/python3.8/dist-packages/hhb/install_nn2/th1520/lib/ \
            -I cpu_Subgraphs_2_out/ -L /usr/local/lib/python3.8/dist-packages/hhb/install_nn2/c920/lib/ \
            -I npu_Subgraphs_8_out/ -L /usr/local/lib/python3.8/dist-packages/hhb/install_nn2/th1520/lib/ \
            -I npu_Subgraphs_9_out/ -L /usr/local/lib/python3.8/dist-packages/hhb/install_nn2/th1520/lib/ \
            -I npu_Subgraphs_10_out/ -L /usr/local/lib/python3.8/dist-packages/hhb/install_nn2/th1520/lib/ \
            -I npu_Subgraphs_11_out/ -L /usr/local/lib/python3.8/dist-packages/hhb/install_nn2/th1520/lib/ \
            -I cpu_Subgraphs_3_out/ -L /usr/local/lib/python3.8/dist-packages/hhb/install_nn2/c920/lib/ \
            -I npu_Subgraphs_12_out/ -L /usr/local/lib/python3.8/dist-packages/hhb/install_nn2/th1520/lib/ \
            -I npu_Subgraphs_13_out/ -L /usr/local/lib/python3.8/dist-packages/hhb/install_nn2/th1520/lib/ \
            -I npu_Subgraphs_14_out/ -L /usr/local/lib/python3.8/dist-packages/hhb/install_nn2/th1520/lib/ \
            -I npu_Subgraphs_15_out/ -L /usr/local/lib/python3.8/dist-packages/hhb/install_nn2/th1520/lib/ \
            -I cpu_Subgraphs_4_out/ -L /usr/local/lib/python3.8/dist-packages/hhb/install_nn2/c920/lib/ \
            -I npu_Subgraphs_16_out/ -L /usr/local/lib/python3.8/dist-packages/hhb/install_nn2/th1520/lib/ \
            -I npu_Subgraphs_17_out/ -L /usr/local/lib/python3.8/dist-packages/hhb/install_nn2/th1520/lib/ \
            -I npu_Subgraphs_18_out/ -L /usr/local/lib/python3.8/dist-packages/hhb/install_nn2/th1520/lib/ \
            -I npu_Subgraphs_19_out/ -L /usr/local/lib/python3.8/dist-packages/hhb/install_nn2/th1520/lib/ \
            -I cpu_Subgraphs_5_out/ -L /usr/local/lib/python3.8/dist-packages/hhb/install_nn2/c920/lib/ \
            -I npu_Subgraphs_20_out/ -L /usr/local/lib/python3.8/dist-packages/hhb/install_nn2/th1520/lib/ \
            -I npu_Subgraphs_21_out/ -L /usr/local/lib/python3.8/dist-packages/hhb/install_nn2/th1520/lib/ \
            -I npu_Subgraphs_22_out/ -L /usr/local/lib/python3.8/dist-packages/hhb/install_nn2/th1520/lib/ \
            -I npu_Subgraphs_23_out/ -L /usr/local/lib/python3.8/dist-packages/hhb/install_nn2/th1520/lib/ \
            -I npu_Subgraphs_24_out/ -L /usr/local/lib/python3.8/dist-packages/hhb/install_nn2/th1520/lib/ \
            -I cpu_Subgraphs_6_out/ -L /usr/local/lib/python3.8/dist-packages/hhb/install_nn2/c920/lib/ \
            -I npu_Subgraphs_25_out/ -L /usr/local/lib/python3.8/dist-packages/hhb/install_nn2/th1520/lib/ \
            -I npu_Subgraphs_26_out/ -L /usr/local/lib/python3.8/dist-packages/hhb/install_nn2/th1520/lib/ \
            -I npu_Subgraphs_27_out/ -L /usr/local/lib/python3.8/dist-packages/hhb/install_nn2/th1520/lib/ \
            -I npu_Subgraphs_28_out/ -L /usr/local/lib/python3.8/dist-packages/hhb/install_nn2/th1520/lib/ \
            -I npu_Subgraphs_29_out/ -L /usr/local/lib/python3.8/dist-packages/hhb/install_nn2/th1520/lib/ \
            -I cpu_Subgraphs_7_out/ -L /usr/local/lib/python3.8/dist-packages/hhb/install_nn2/c920/lib/ \
            -I npu_Subgraphs_30_out/ -L /usr/local/lib/python3.8/dist-packages/hhb/install_nn2/th1520/lib/ \
            -I npu_Subgraphs_31_out/ -L /usr/local/lib/python3.8/dist-packages/hhb/install_nn2/th1520/lib/ \
            -I npu_Subgraphs_32_out/ -L /usr/local/lib/python3.8/dist-packages/hhb/install_nn2/th1520/lib/ \
            -I npu_Subgraphs_33_out/ -L /usr/local/lib/python3.8/dist-packages/hhb/install_nn2/th1520/lib/ \
            -I npu_Subgraphs_34_out/ -L /usr/local/lib/python3.8/dist-packages/hhb/install_nn2/th1520/lib/ \
            -I cpu_Subgraphs_8_out/ -L /usr/local/lib/python3.8/dist-packages/hhb/install_nn2/c920/lib/ \
            -I npu_Subgraphs_35_out/ -L /usr/local/lib/python3.8/dist-packages/hhb/install_nn2/th1520/lib/ \
            -I npu_Subgraphs_36_out/ -L /usr/local/lib/python3.8/dist-packages/hhb/install_nn2/th1520/lib/ \
            -I npu_Subgraphs_37_out/ -L /usr/local/lib/python3.8/dist-packages/hhb/install_nn2/th1520/lib/ \
            -I npu_Subgraphs_38_out/ -L /usr/local/lib/python3.8/dist-packages/hhb/install_nn2/th1520/lib/ \
            -I npu_Subgraphs_39_out/ -L /usr/local/lib/python3.8/dist-packages/hhb/install_nn2/th1520/lib/ \
            -I cpu_Subgraphs_9_out/ -L /usr/local/lib/python3.8/dist-packages/hhb/install_nn2/c920/lib/ \
            -I npu_Subgraphs_40_out/ -L /usr/local/lib/python3.8/dist-packages/hhb/install_nn2/th1520/lib/ \
            -I npu_Subgraphs_41_out/ -L /usr/local/lib/python3.8/dist-packages/hhb/install_nn2/th1520/lib/ \
            -lshl -L /usr/local/lib/python3.8/dist-packages/hhb/prebuilt/decode/install/lib/rv -L /usr/local/lib/python3.8/dist-packages/hhb/prebuilt/runtime/riscv_linux \
            -lprebuilt_runtime -ljpeg -lpng -lz -lstdc++ -lm -I /usr/local/lib/python3.8/dist-packages/hhb/install_nn2/th1520/include/ -I /usr/local/lib/python3.8/dist-packages/hhb/install_nn2/c920/include/ \
            -mabi=lp64d -march=rv64gcv0p7_zfh_xtheadc -Wl,-unresolved-symbols=ignore-in-shared-libs

# # scp -r ../case0/ sipeed@192.168.1.18:/home/sipeed/wq_work/unet/coocompute/
# # scp -r main_example sipeed@192.168.1.18:/home/sipeed/wq_work/unet/coocompute/case0