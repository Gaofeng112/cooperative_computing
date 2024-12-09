"""
  Description: 计算mse psnr ssim  
"""
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity
from skimage.io import imread
import numpy as np
import pandas as pd

# 读取图像
original_image = imread('../images/onnx_V3.png')
generated_image = imread('../images/test_onnx_pca.png')

# 确保图像大小相同
if original_image.shape != generated_image.shape:
    raise ValueError('两个图像的尺寸必须相同')

# 打印图像的形状
# print(f"Original image shape: {original_image.shape}")
# print(f"Generated image shape: {generated_image.shape}")

# 计算MSE
mse = mean_squared_error(original_image, generated_image)
print(f'MSE: {mse}')

# 计算PSNR
psnr = peak_signal_noise_ratio(original_image, generated_image)
print(f'PSNR: {psnr}')

# 计算SSIM，指定 multichannel=True 并设置 win_size
min_dim = min(original_image.shape[:2])  # 获取图像的最小维度
win_size = min(7, min_dim)  # 确保 win_size 不超过图像的最小维度

# 确保 win_size 是奇数
if win_size % 2 == 0:
    win_size -= 1

# 检查 win_size 是否有效
if win_size < 3:
    win_size = 3  # 确保 win_size 至少为3

# 显式设置 channel_axis 参数
ssim = structural_similarity(original_image, generated_image, multichannel=True, win_size=win_size, channel_axis=-1)
print(f'SSIM: {ssim}')
# 创建一个DataFrame来存储数据
data = {
    'Metric': ['MSE', 'PSNR', 'SSIM'],
    'Value': [mse, psnr, ssim]
}
df = pd.DataFrame(data)

# 将DataFrame写入Excel文件
output_file = '../image_quality_metrics.xlsx'
df.to_excel(output_file, index=False)

print(f'Results have been saved to {output_file}')