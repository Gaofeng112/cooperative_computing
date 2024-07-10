import torch
import onnx
import re
def splitinstruction(instr):
    iolist=re.split('--input-name \"|\" --output-name \"|\" --input-shape \"',instr)
    del iolist[0]
    del iolist[-1]
    in_=iolist[0].split(';')
    out_=iolist[1].split(';')
    return in_,out_

def split_onnx(instrfile,type):
    f1=open(instrfile,"r")
    lines=f1.readlines()
    count=0
    for line in lines:
        input_names, output_names = splitinstruction(line)
        input_path ='net/unet_32_sim.onnx'
        output_path ='subgraphs_'+type+'/'+type+'subgraph'+str(count)+'.onnx'
        count=count+1
        if((input_names!=['']) and (output_names!=[''])):
            onnx.utils.extract_model(input_path, output_path, input_names, output_names)
    f1.close()
