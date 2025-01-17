import gc
import os
import time
from types import MethodType

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from tqdm import tqdm
from sklearn.cluster import KMeans
from torch.nn import functional as F
import numpy as np
import pdb
def quant_transmartix(x, bits=8):
    if(x.max() == x.min()):
        return x, 0
    n = 2 ** (bits - 1) - 1
    act_scale = (x.max() - x.min()) / 2 / n
    zero_point = (x.min() + x.max()) / 2
    aint = ((x - zero_point) / act_scale).round().clamp(-n - 1, n)
    xq = aint * act_scale + zero_point
    return xq, aint

def quant_transmartix1(x, bits=8):
    if(x.max() == x.min()):
        return x, 0
    n = 2 ** (bits - 1) - 1
    act_scale = (x.max() - x.min()) / 2 / n
    zero_point = (x.min() + x.max()) / 2
    aint = ((x - zero_point) / act_scale).round().clamp(-n - 1, n)
    return aint, act_scale, zero_point

def get_projection_matrix(im, eigenVar,num_bits=8):
    # covariance matrix
    cov = torch.matmul(im, im.t()) / im.shape[1]
    # svd
    u, s, _ = torch.svd(cov)
    u,_ = quant_transmartix(u,16)
    return u, s

# def comp(x, rate, fname, count, transu, inb, num_bits, layer):
#     if(len(x.shape) == 2):
#         B,C = x.shape
#         x_reshape = x
#     elif(len(x.shape) == 3):
#         B,C,H = x.shape
#         x_reshape = x.permute(1,0,2).reshape(C,-1)
#     elif(len(x.shape) == 4):
#         B, C, H, W = x.shape  
#         x_reshape = x.permute(1, 0, 2, 3).reshape(C, -1)  # 重塑输入张量为 [C, B*H*W] 形状
#     else:
#         raise NotImplementedError
#     if count == 1:
#         u, s = get_projection_matrix(x_reshape, rate, num_bits)
#         # print(f'Layer: {layer} Rate Init: {len(u.t())/len(u)} {len(u.t())} {len(u)} | Size: {x.numel()}')
#         x_trans = torch.matmul(u.t(), x_reshape)
#         x_trans, x_trans_int = quant_transmartix(x_trans, num_bits)    
#         x_return = torch.matmul(u, x_trans)
#         x_return, x_return_int = quant_transmartix(x_return, num_bits)
#         ru = u  # 保存投影矩阵 u
#         rbits = None  # 量化位宽暂无变化
#     else:
#         x_trans = torch.matmul(transu.t(), x_reshape)
#         # 对变换后的张量进行量化，并获取量化尺度和零点
#         x_trans_int, act_scale, zero_point = quant_transmartix1(x_trans, num_bits)
#         # 扩展 inb 以匹配 x_trans_int 的形状
#         inb_expend = inb[:, :, None].repeat(1, 1, H * W)
#         # 创建裁剪掩码
#         mask_clip_max = torch.where(x_trans_int > inb_expend[0])
#         mask_clip_min = torch.where(x_trans_int < inb_expend[1])
#         # 应用裁剪
#         x_trans_int[mask_clip_max] = inb_expend[0][mask_clip_max]
#         x_trans_int[mask_clip_min] = inb_expend[1][mask_clip_min]
#         # 反量化回浮点数
#         x_trans = x_trans_int * act_scale + zero_point
#         # 使用投影矩阵 transu 对变换后的张量进行逆变换
#         x_return = torch.matmul(transu, x_trans)
#         # 再次对逆变换后的张量进行量化
#         x_return, x_return_int = quant_transmartix(x_return, num_bits)
#         ru = None  # 无需保存投影矩阵
#         rbits = None  # 量化位宽暂无变化

#     # 保存投影矩阵 ru
#     torch.save(ru, f'result_ru/{layer}.pt')
#     # torch.save(rbits, f'result_rbits/{layer}.pt')
#     # 重塑 x_return 回原始形状 [B, C, H, W]
#     if len(x.shape) == 2:
#         x_return = x_return
#     elif len(x.shape) == 3:
#         x_return = x_return.reshape(C, B, H).permute(1, 0, 2)
#     elif len(x.shape) == 4:
#         x_return = x_return.reshape(C, B, H, W).permute(1, 0, 2, 3)
#     # x_return = x_return.reshape(C, B, H, W).permute(1, 0, 2, 3)
#     return x_return, ru, rbits
def comp(x, rate,fname, count,transu,inb, num_bits,layer):
    if(len(x.shape) == 2):
        B,C = x.shape
        x_reshape = x
    elif(len(x.shape) == 3):
        B,C,H = x.shape
        x_reshape = x.permute(1,0,2).reshape(C,-1)
    elif(len(x.shape) == 4):
        B, C, H, W = x.shape  
        x_reshape = x.permute(1, 0, 2, 3).reshape(C, -1)  # 重塑输入张量为 [C, B*H*W] 形状
    else:
        raise NotImplementedError
    if(count==1):
        
        u,s = get_projection_matrix(x_reshape, rate, num_bits)
        # print(f'Rate Init: {len(u.t())/len(u)} {len(u.t())} {len(u)} | Size: {x.numel()}')
        
        x_trans = torch.matmul(u.t(), x_reshape)
        x_trans, x_trans_int = quant_transmartix(x_trans,num_bits)    
    
        channel_max = x_trans_int.max(-1)[0].reshape(1,-1)
        channel_min = x_trans_int.min(-1)[0].reshape(1,-1)

        channel_dif = channel_max-channel_min
        channel_dif[torch.where(channel_dif==0)]=1
        bits = torch.ceil(torch.log2(channel_dif))

        max_min = torch.cat([channel_max,channel_min],dim=0)
        
        x_return = torch.matmul(u, x_trans)
        # print(x_return)
        x_return, x_return_int = quant_transmartix(x_return,num_bits)
        ru = u
        rbits=max_min
        # rbits=None
    elif(count<=100):
        x_trans = torch.matmul(transu.t(), x_reshape)
        x_trans,x_trans_int = quant_transmartix(x_trans,num_bits)

        channel_max = x_trans_int.max(-1)[0].reshape(1,-1)
        channel_min = x_trans_int.min(-1)[0].reshape(1,-1)
        max_min = torch.cat([channel_max,channel_min],dim=0)

        x_return = torch.matmul(transu, x_trans)
        # print(x_return.size())
        x_return,x_return_int = quant_transmartix(x_return,num_bits)
        ru = None
        # rbits = None
        rbits=max_min
    else:
        x_trans = torch.matmul(transu.t(), x_reshape)
        x_trans_int,act_scale,zero_point = quant_transmartix1(x_trans,num_bits)
        

        # print(C, inb.shape,inb.mean())
        inb_expend = inb[:,:,None].repeat(1,1,H*W)
        mask_clip_max = torch.where(x_trans_int>inb_expend[0])
        mask_clip_min = torch.where(x_trans_int<inb_expend[1])
        x_trans_int[mask_clip_max]=inb_expend[0][mask_clip_max]
        x_trans_int[mask_clip_min]=inb_expend[1][mask_clip_min]

        x_trans = x_trans_int * act_scale + zero_point
        channel_max = x_trans_int.max(-1)[0].reshape(1,-1)
        channel_min = x_trans_int.min(-1)[0].reshape(1,-1)        
        max_min = torch.cat([channel_max,channel_min],dim=0)


        x_return = torch.matmul(transu, x_trans)
        print(x_return.size())
        x_return,x_return_int = quant_transmartix(x_return,num_bits)
        ru = None
        rbits=max_min


    # 保存投影矩阵 ru
    torch.save(ru, f'result_ru/{layer}.pt')
    # torch.save(rbits, f'result_rbits/{layer}.pt')
    # 重塑 x_return 回原始形状 [B, C, H, W]
    if len(x.shape) == 2:
        x_return = x_return
    elif len(x.shape) == 3:
        x_return = x_return.reshape(C, B, H).permute(1, 0, 2)
    elif len(x.shape) == 4:
        x_return = x_return.reshape(C, B, H, W).permute(1, 0, 2, 3)

    return x_return, ru, rbits
# def comp(x, rate,fname, count,transu,inb, num_bits,layer):
#     B,C,H,W = x.shape
#     x_reshape = x.permute(1,0,2,3).reshape(C,-1)
#     if(count==1):
        
#         u,s = get_projection_matrix(x_reshape, rate, num_bits)
#         print(f'Rate Init: {len(u.t())/len(u)} {len(u.t())} {len(u)} | Size: {x.numel()}')
        
#         x_trans = torch.matmul(u.t(), x_reshape)
#         x_trans, x_trans_int = quant_transmartix(x_trans,num_bits)    
    
#         # channel_max = x_trans_int.max(-1)[0].reshape(1,-1)
#         # channel_min = x_trans_int.min(-1)[0].reshape(1,-1)

#         # channel_dif = channel_max-channel_min
#         # channel_dif[torch.where(channel_dif==0)]=1
#         # bits = torch.ceil(torch.log2(channel_dif))

#         # max_min = torch.cat([channel_max,channel_min],dim=0)
        
#         x_return = torch.matmul(u, x_trans)
#         # print(x_return)
#         x_return, x_return_int = quant_transmartix(x_return,num_bits)
#         ru = u
#         # rbits=max_min
#         rbits=None
#     elif(count<=100):
#         x_trans = torch.matmul(transu.t(), x_reshape)
#         x_trans,x_trans_int = quant_transmartix(x_trans,num_bits)

#         # channel_max = x_trans_int.max(-1)[0].reshape(1,-1)
#         # channel_min = x_trans_int.min(-1)[0].reshape(1,-1)
#         # max_min = torch.cat([channel_max,channel_min],dim=0)

#         x_return = torch.matmul(transu, x_trans)
#         # print(x_return.size())
#         x_return,x_return_int = quant_transmartix(x_return,num_bits)
#         ru = None
#         rbits = None
#         # rbits=max_min
#     else:
#         x_trans = torch.matmul(transu.t(), x_reshape)
#         x_trans_int,act_scale,zero_point = quant_transmartix1(x_trans,num_bits)
        

#         # # print(C, inb.shape,inb.mean())
#         # inb_expend = inb[:,:,None].repeat(1,1,H*W)
#         # mask_clip_max = torch.where(x_trans_int>inb_expend[0])
#         # mask_clip_min = torch.where(x_trans_int<inb_expend[1])
#         # x_trans_int[mask_clip_max]=inb_expend[0][mask_clip_max]
#         # x_trans_int[mask_clip_min]=inb_expend[1][mask_clip_min]

#         # x_trans = x_trans_int * act_scale + zero_point
#         # channel_max = x_trans_int.max(-1)[0].reshape(1,-1)
#         # channel_min = x_trans_int.min(-1)[0].reshape(1,-1)        
#         # max_min = torch.cat([channel_max,channel_min],dim=0)


#         x_return = torch.matmul(transu, x_trans)
#         print(x_return.size())
#         x_return,x_return_int = quant_transmartix(x_return,num_bits)
#         ru = None
#         rbits = None
#         # rbits=max_min


#     x_return = x_return.reshape(C,B,H,W).permute(1,0,2,3)
#     # print(x_reshape.size())
#     # return x_reshape
#     return x_return,ru,rbits


def quant_activation(x, bit, act_scale, zero_point=0):
    n = 2 ** (bit - 1) - 1
    aint = ((x - zero_point) / act_scale).round().clamp(-n - 1, n)
    xq = aint * act_scale + zero_point
    return xq



def quant_linear_weight(w, bit, mode="channel_wise", symmetric=True):
    if mode == "channel_wise" and symmetric:
        n = 2 ** (bit - 1) - 1
        scale_channel_wise = w.abs().max(dim=1, keepdim=True)[0] / n
        wint = (w / scale_channel_wise).round().clamp(-n - 1, n)
        wq = wint * scale_channel_wise
    else:
        n = 2 ** (bit - 1) - 1
        scale_tensor_wise = w.abs().max() / n
        wint = (w / scale_tensor_wise).round().clamp(-n - 1, n)
        wq = wint * scale_tensor_wise
    return wq


def quant_conv_weight(w, bit, mode="channel_wise", symmetric=True):
    if mode == "channel_wise" and symmetric:
        n = 2 ** (bit - 1) - 1
        scale_channel_wise = (
            w.view(w.shape[0], -1).abs().max(dim=-1, keepdim=True)[0] / n
        )
        scale_channel_wise = scale_channel_wise.view(w.shape[0], 1, 1, 1)
        wint = (w / scale_channel_wise).round().clamp(-n - 1, n)
        wq = wint * scale_channel_wise
    else:
        raise NotImplementedError
    return wq

def quant_conv_forward_save_output( x, layer, count, bit, i ):

    x, xq_int=quant_transmartix(x,bit)
    os.makedirs('middle_data', exist_ok=True)
    output_tensor, ru, rb = comp(
                        x=x,
                        rate=0.999999,
                        fname=f"call_{layer}",
                        count=1,
                        transu=None,
                        inb=None,
                        num_bits=8,
                        layer=layer  
                        )
    # pdb.set_trace()
    if(count==1):
        u = ru
        rb= rb

    B,C,H,W=x.shape
    Max = rb[0:200:2]
    Min = rb[1:200:2]
    # print(Max.shape,Min.shape)
    channel_max = Max.max(0)[0].reshape(1,-1)
    channel_min = Min.min(0)[0].reshape(1,-1)
    # pdb.set_trace()
#0.285
    mask_neg_max = torch.where(channel_max<0)
    channel_max[mask_neg_max] = -1 * channel_max[mask_neg_max]
    mask_zero_max = torch.where(channel_max==0)
    channel_max[mask_zero_max] = 1
    channel_max_log = torch.log2(channel_max)
    # channel_max_return = 2 ** torch.floor(channel_max_log)

    condition = channel_max_log - torch.floor(channel_max_log) <= 0.55
    channel_max_return = torch.where(condition,   
                                    2 ** torch.floor(channel_max_log),   
                                    2 ** torch.ceil(channel_max_log))

    mask_neg_min = torch.where(channel_min<0)
    channel_min[mask_neg_min] = -1 * channel_min[mask_neg_min]
    mask_zero_min = torch.where(channel_min==0)
    channel_min[mask_zero_min] = 1
    channel_min_log = torch.log2(channel_min)
    # channel_min_return = 2 ** torch.floor(channel_min_log)

    condition = channel_min_log - torch.floor(channel_min_log) <= 0.6
    channel_min_return = torch.where(condition,   
                                    2 ** torch.floor(channel_min_log),   
                                    2 ** torch.ceil(channel_min_log)) 

    channel_min_return[mask_neg_min] = -1 * channel_min_return[mask_neg_min]
    rb = torch.cat([channel_max_return,channel_min_return],dim=0)
    filename = f'./result/result_{i}.txt'
    with open(filename, 'a') as f:
        f.write(f'{(channel_max_return-channel_min_return).mean()/2**bit},{x.numel()}\n')
    # zq = _conv_forward(xq, weight,  bias)

    return output_tensor
# def quant_conv_forward_save_output(self, x, layer):
#     self.count += 1
#     # # save zq to pt file
#     fname=f"{self.layer}-{self.own_name}"
#     os.makedirs('middle_data', exist_ok=True)
#     # os.makedirs(fname, exist_ok=True)
#     # check the num of pt files
#     # num = len(os.listdir(fname))
#     # fname_in = f'{fname}/{num+1}-{x.shape}.pt'
    
#     xq = x.clip(-self.clip_value, self.clip_value)

#     xq, xq_int=quant_transmartix(xq,self.bit)

#     if(self.comp and self.layer>0): 
#         # xq,ru,rb=comp(xq,0.999999,fname,self.count,self.u,self.rb,self.bit,self.layer)
#         output_tensor, ru, rb = comp(
#                             x=output_tensor,
#                             rate=0.999999,
#                             fname=f"call_{layer}",
#                             count=1,
#                             transu=None,
#                             inb=None,
#                             num_bits=8,
#                             layer=layer  
#                         )
#         if(self.count<=100):
#             if(self.count==1):
#                 self.u = ru
#                 # self.rb = rb

#                 self.rb= rb
#             # else:
#             #     self.rb=torch.cat([self.rb,rb],dim=0)

#         if(self.count==100):
#             os.makedirs(f'middle_data/Count{self.count}', exist_ok=True)
#             B,C,H,W=x.shape
#             Max = self.rb[0:200:2]
#             Min = self.rb[1:200:2]
#             # print(Max.shape,Min.shape)
#             channel_max = Max.max(0)[0].reshape(1,-1)
#             channel_min = Min.min(0)[0].reshape(1,-1)

#         #0.285
#             mask_neg_max = torch.where(channel_max<0)
#             channel_max[mask_neg_max] = -1 * channel_max[mask_neg_max]
#             mask_zero_max = torch.where(channel_max==0)
#             channel_max[mask_zero_max] = 1
#             channel_max_log = torch.log2(channel_max)
#             # channel_max_return = 2 ** torch.floor(channel_max_log)

#             condition = channel_max_log - torch.floor(channel_max_log) <= 0.55
#             channel_max_return = torch.where(condition,   
#                                             2 ** torch.floor(channel_max_log),   
#                                             2 ** torch.ceil(channel_max_log))

#             # count_test1 = 0
#             # condition = channel_max_log - torch.floor(channel_max_log) <= 0.6
#             # # print(condition)
#             # count_test1 = torch.sum(condition).item()  # 使用 .item() 将结果转换为 Python 整数  
            
#             # # 计算比例  
#             # proportion = count_test1 / condition.numel()
#             # # print(count_test1)
#             # # print(len(condition)) 
#             # # print(proportion)
#             # # 根据比例选择操作  
#             # if proportion >= 0.7:  
#             #     channel_max_return = 2 ** torch.floor(channel_max_log)
#             # else:
#             #     channel_max_return = torch.where(condition,   
#             #                                 2 ** torch.floor(channel_max_log),   
#             #                                 2 ** torch.ceil(channel_max_log))
                
#             # channel_max_return[mask_neg_max] = -1 * channel_max_return[mask_neg_max]
#             # print(channel_max_return)

#             mask_neg_min = torch.where(channel_min<0)
#             channel_min[mask_neg_min] = -1 * channel_min[mask_neg_min]
#             mask_zero_min = torch.where(channel_min==0)
#             channel_min[mask_zero_min] = 1
#             channel_min_log = torch.log2(channel_min)
#             # channel_min_return = 2 ** torch.floor(channel_min_log)

#             condition = channel_min_log - torch.floor(channel_min_log) <= 0.6
#             channel_min_return = torch.where(condition,   
#                                             2 ** torch.floor(channel_min_log),   
#                                             2 ** torch.ceil(channel_min_log)) 

#             # count_test2 = 0
#             # condition = channel_min_log - torch.floor(channel_min_log) <= 0.55
#             # count_test2 = torch.sum(condition).item()  # 使用 .item() 将结果转换为 Python 整数  
  
#             # proportion = count_test2 / condition.numel()
#             # if proportion >= 0.8:  
#             #     channel_min_return = 2 ** torch.floor(channel_min_log)
#             # else:
#             #     channel_min_return = torch.where(condition,   
#             #                                 2 ** torch.floor(channel_min_log),   
#             #                                 2 ** torch.ceil(channel_min_log))
                
#             channel_min_return[mask_neg_min] = -1 * channel_min_return[mask_neg_min]
#             # print(channel_min_return)

            

#             self.rb = torch.cat([channel_max_return,channel_min_return],dim=0)
#             # # self.rb = torch.cat([channel_max,channel_min],dim=0)
#             f=open('result.txt','a')
#             f.write(f'{(channel_max_return-channel_min_return).mean()/2**self.bit},{x.numel()}\n')
#             # # f.write(f'{(channel_max-channel_min).mean()/2**self.bit},{x.numel()}\n')
#             # # torch.save(self.rb, f'result_rb/{self.layer}.pt')
#             # np.savetxt(f'result_rb_txt/{self.layer}.txt',self.rb.cpu(),fmt="%d")
#             # import matplotlib.pyplot as plt
#             # fig, ax = plt.subplots() 
#             # xx=np.arange(0, C, 1)
#             # for i in range(len(Max)):
#             #     Y0 = np.array(Max[i].cpu())
#             #     Y1 = np.array(Min[i].cpu())
#             #     ax.plot(xx,Y0,label=f'Max_{i}')
#             #     ax.plot(xx,Y1,label=f'MIN_{i}')
#             #     print(f'XCJDEBUG SAVEFIG {fname}')
#             # ax.legend()
#             # plt.savefig(f'middle_data/Count{self.count}/{fname}.jpg')




#     zq = self._conv_forward(xq, self.weight,  self.bias)
#     return zq


def quant_conv_forward_optimization(self, x):
    if self.enable_calib_act_min_max:
        z_target = self._conv_forward(x, self.weight, None)
        xmax = x.abs().max()
        best_scale = None
        best_mse = 1e10
        range_num = 200
        pbar = tqdm(range(range_num), desc=self.own_name)
        for ii in pbar:
            xq = x.clip(
                -xmax * (1 / range_num * (range_num - ii)),
                xmax * (1 / range_num * (range_num - ii)),
            )
            zero_point = (xq.max() + xq.min()) / 2
            act_scale = (xq.max() - xq.min()) / 2 / (2 ** (self.bit - 1) - 1)
            xq = quant_activation(
                xq, bit=self.bit, act_scale=act_scale, zero_point=zero_point
            )
            zq = self._conv_forward(xq, self.weight, None)
            mse = ((z_target - zq) ** 2).mean().item()
            if mse < best_mse:
                best_mse = mse
                best_scale = act_scale
                best_zero_point = zero_point
                best_clip_value = xmax * (1 / range_num * (range_num - ii))
            pbar.set_postfix(
                dict(
                    best_mse=f"{best_mse:.1e}",
                    best_scale=best_scale.data.item(),
                    xmax=xmax.data.item(),
                    best_clip_value=best_clip_value.data.item(),
                )
            )
        assert best_scale is not None
        del z_target
        del xq
        del zq
        del mse
        gc.collect()
        torch.cuda.empty_cache()
        self.act_scale = best_scale
        self.zero_point = best_zero_point
        self.clip_value = best_clip_value
        self.enable_calib_act_min_max = False
    if self.act_scale is None:
        self.zero_point = (x.max() + x.min()) / 2
        self.act_scale = (x.max() - x.min()) / 2 / (2 ** (self.bit - 1) - 1)
    x = quant_activation(
        x, bit=self.bit, act_scale=self.act_scale, zero_point=self.zero_point
    )
    return self._conv_forward(x, self.weight, self.bias)


def quant_linear_forward_save_output(self, x):
    # # save zq to pt file
    # fname=f"middle_data/{self.layer}-{self.own_name}"
    # os.makedirs(fname, exist_ok=True)
    # # check the num of pt files
    # num = len(os.listdir(fname))
    # fname_in = f'{fname}/{num+1}-{x.shape}.pt'

    xq = x.clip(-self.clip_value, self.clip_value)
    # xq=x
    act_scale = (xq.max() - xq.min()) / 2 / (2 ** (self.bit - 1) - 1)
    zero_point = (xq.min() + xq.max()) / 2
    xq = quant_activation(xq, bit=self.bit, act_scale=act_scale, zero_point=zero_point)
    zq = F.linear(xq, self.weight,self.bias)
    return zq


def quant_linear_forward_optimization(self, x):
    if self.enable_calib_act_min_max:
        z_target = F.linear(x, self.weight)
        xmax = x.abs().max()
        best_scale = None
        best_mse = 1e5
        range_num = 200
        pbar = tqdm(range(range_num), desc=self.own_name)
        for ii in pbar:
            xq = x.clip(
                -xmax * (1 / range_num * (range_num - ii)),
                xmax * (1 / range_num * (range_num - ii)),
            )
            act_scale = (xq.max() - xq.min()) / 2 / (2 ** (self.bit - 1) - 1)
            zero_point = (xq.min() + xq.max()) / 2
            xq = quant_activation(
                xq, bit=self.bit, act_scale=act_scale, zero_point=zero_point
            )
            zq = F.linear(xq, self.weight)
            mse = ((z_target - zq) ** 2).mean()
            if mse < best_mse:
                best_mse = mse
                best_scale = act_scale
                best_zero = zero_point
                best_clip_value = xmax * (1 / range_num * (range_num - ii))
            pbar.set_postfix(
                dict(
                    best_mse=best_mse.data.item(),
                    best_scale=best_scale.data.item(),
                    xmax=xmax.data.item(),
                    best_clip_value=best_clip_value.data.item(),
                )
            )
        assert best_scale is not None
        del z_target
        del xq
        del zq
        del mse
        gc.collect()
        torch.cuda.empty_cache()
        self.act_scale = best_scale
        self.zero_point = best_zero
        self.clip_value = best_clip_value
        self.enable_calib_act_min_max = False

    if self.act_scale is None:
        self.zero_point = (x.max() + x.min()) / 2
        self.act_scale = (x.max() - x.min()) / 2 / (2 ** (self.bit - 1) - 1)
    x = quant_activation(
        x, bit=self.bit, act_scale=self.act_scale, zero_point=self.zero_point
    )

    if self.enable_calib_weight_min_max:
        z_target = F.linear(x, self.fp_weight)
        best_mse = 1e5
        range_num = 200
        wmax = self.fp_weight.abs().max()
        pbar = tqdm(range(range_num), desc=self.own_name)
        for ii in pbar:
            w_clip = wmax * (1 / range_num * (range_num - ii))
            wq = quant_linear_weight(
                self.fp_weight.clip(-w_clip, w_clip),
                self.bit,
                mode="tensor_wise",
                symmetric=True,
            )
            zq = F.linear(x, wq)
            mse = ((z_target - zq) ** 2).mean()
            if mse < best_mse:
                best_mse = mse
                best_w_clip = w_clip
            pbar.set_postfix(
                dict(
                    best_mse=best_mse.data.item(),
                    best_w_clip=best_w_clip.data.item(),
                )
            )
            # time.sleep(1)
        self.weight.data = quant_linear_weight(
            self.fp_weight.clip(-best_w_clip, best_w_clip),
            self.bit,
            mode="tensor_wise",
            symmetric=True,
        )
        self.enable_calib_weight_min_max = False

    return F.linear(x, self.weight, self.bias)


def fast_quant(
    model,
    comp,
    bit=8,
    fp=False,
    enable_calib_act_min_max=False,
    enable_calib_weight_min_max=False,
    optimization=False,
    load_min_max_from_json=False,
    min_max_dict=None,
):
    if fp:
        return model
    layer = 0
    convlayer=0
    for name, module in tqdm(model.named_modules(), desc="Quantize weights"):
        module.own_name = name
        if isinstance(module, nn.Linear):
            module.bit = bit
            w = module.weight.data.clone()
            # module.fp_weight = module.weight.clone().cuda()
            wq = quant_linear_weight(w, bit, mode="tensor_wise", symmetric=True)
            module.weight.data = wq.data
            module.act_scale = None
            if optimization:
                module.forward = MethodType(quant_linear_forward_optimization, module)
            elif load_min_max_from_json:
                module.clip_value = torch.tensor(min_max_dict[name], device="cuda")
                module.forward = MethodType(quant_linear_forward_save_output, module)
            module.enable_calib_act_min_max = enable_calib_act_min_max
            module.enable_calib_weight_min_max = enable_calib_weight_min_max
            module.layer = layer
            layer += 1
        if isinstance(module, nn.Conv2d):
            module.layer = layer
            module.convlayer = convlayer
            # module.param = lines[convlayer].split(',')
            # print(module.param)
            convlayer+=1
            layer += 1
            module.count = 0
            module.u  = 0
            module.rb = 0
            module.comp = comp
            

            module.bit = bit
            w = module.weight.data.clone()
            wq = quant_conv_weight(w, bit, mode="channel_wise", symmetric=True)
            module.weight.data = wq.data
            module.act_scale = None
            if optimization:
                module.forward = MethodType(quant_conv_forward_optimization, module)
            elif load_min_max_from_json:
                module.clip_value = torch.tensor(min_max_dict[name], device="cuda")
                module.forward = MethodType(quant_conv_forward_save_output, module)
            module.enable_calib_act_min_max = enable_calib_act_min_max
            # module.__repr__ = MethodType(lambda self: f'QConv2d(bit={self.bit}, {self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, bias={self.bias is not None})', module)

    return model
