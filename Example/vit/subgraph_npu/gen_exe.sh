export PATH=/tools/Xuantie-900-gcc-linux-5.10.4-glibc-x86_64-V2.6.1-light.1/bin/:$PATH
riscv64-unknown-linux-gnu-gcc main.c -o main_example hhb_out/io.c hhb_out_0/model.c hhb_out/model.c -Wl,--gc-sections -O2 -g -mabi=lp64d -I hhb_out_0/ -L /usr/local/lib/python3.8/dist-packages/hhb/install_nn2/th1520/lib/ -I hhb_out/ -L /usr/local/lib/python3.8/dist-packages/hhb/install_nn2/c920/lib/ -lshl -L /usr/local/lib/python3.8/dist-packages/hhb/prebuilt/decode/install/lib/rv -L /usr/local/lib/python3.8/dist-packages/hhb/prebuilt/runtime/riscv_linux -lprebuilt_runtime -ljpeg -lpng -lz -lstdc++ -lm -I /usr/local/lib/python3.8/dist-packages/hhb/install_nn2/th1520/include/ -I /usr/local/lib/python3.8/dist-packages/hhb/install_nn2/c920/include/ -mabi=lp64d -march=rv64gcv0p7_zfh_xtheadc -Wl,-unresolved-symbols=ignore-in-shared-libs
scp -r main_example sipeed@192.168.1.18:/home/sipeed/wq_work/vit/subgraph_npu/