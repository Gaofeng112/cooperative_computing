import extract_onnx_lib
import torch
import onnx
import re
def split_runcut_onnx(instrfile):
    f1=open(instrfile,"r")
    lines=f1.readlines()
    count=0
    cpu_count=0
    npu_count=0
    for line in lines:
        if(line.find("th1520")!=-1):
            type='npu'
            count=npu_count
        elif(line.find("c920")!=-1):
            type='cpu'
            count=cpu_count
        else:
            continue
        input_names, output_names = extract_onnx_lib.splitinstruction(line)
        input_path ='net/unet_32_sim.onnx'
        output_path ='runcut_subgraphs_'+type+'/'+type+'subgraph'+str(count)+'.onnx'
        if type=='cpu':
            cpu_count=cpu_count+1
        else:
            npu_count=npu_count+1
        if((input_names!=['']) and (output_names!=[''])):
            onnx.utils.extract_model(input_path, output_path, input_names, output_names)
    f1.close()
split_runcut_onnx('tools/encoder_unet/runcut.sh')