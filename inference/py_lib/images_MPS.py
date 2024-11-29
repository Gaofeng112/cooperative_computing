from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity
from skimage.io import imread
import numpy as np
import pandas as pd
import os
import re

def calculate_image_metrics(original_image_path, generated_image_path):
    original_image = imread(original_image_path)
    generated_image = imread(generated_image_path)

    # 确保图像大小相同
    if original_image.shape != generated_image.shape:
        raise ValueError('两个图像的尺寸必须相同')

    # 计算MSE
    mse = mean_squared_error(original_image, generated_image)

    # 计算PSNR
    psnr = peak_signal_noise_ratio(original_image, generated_image)

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

    return mse, psnr, ssim

def compare_images_in_directories(original_dir, generated_dir, output_file):
    # 使用正则表达式提取文件名中的数字部分

    def sort_key(filename):
        parts = filename.split('_')
        if len(parts) > 1:
            try:
                return int(parts[1].split('.')[0])
            except ValueError:
                print(f"Warning: Could not parse number from filename {filename}")
                return 0  # 返回一个默认值，防止排序失败
        else:
            print(f"Warning: Filename {filename} does not contain an underscore")
            return 0  # 返回一个默认值，防止排序失败

    original_images = sorted([f for f in os.listdir(original_dir) if f.endswith('.png')], key=sort_key)
    generated_images = sorted([f for f in os.listdir(generated_dir) if f.endswith('.png')], key=sort_key)

    results = []

    for orig_img_name, gen_img_name in zip(original_images, generated_images):
        orig_img_path = os.path.join(original_dir, orig_img_name)
        gen_img_path = os.path.join(generated_dir, gen_img_name)

        try:
            mse, psnr, ssim = calculate_image_metrics(orig_img_path, gen_img_path)
            results.append({
                'Original Image': orig_img_name,
                'Generated Image': gen_img_name,
                'MSE': mse,
                'PSNR': psnr,
                'SSIM': ssim
            })
        except Exception as e:
            print(f"Error processing images {orig_img_name} and {gen_img_name}: {e}")

    df = pd.DataFrame(results)
    
    # 检查输出文件路径是否存在
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        df.to_excel(output_file, index=False)
        print(f'Results have been saved to {output_file}')
    except PermissionError:
        print(f"Permission denied: Unable to write to {output_file}. Please check file permissions or close the file if it is open in another program.")
    except Exception as e:
        print(f"An error occurred while saving the results: {e}")

if __name__ == "__main__":
    original_dir = '../images_origin'
    generated_dir = '../images_pca'
    output_file = '../image_quality_metrics.xlsx'

    compare_images_in_directories(original_dir, generated_dir, output_file)