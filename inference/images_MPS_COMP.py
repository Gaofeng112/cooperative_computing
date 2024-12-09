from model_inference import ImageMetricsEvaluator
import os


evaluator = ImageMetricsEvaluator(
        original_dir="./output/origin",
        generated_dir="./output/pca",
        compression_dir="./output/result",  # 假设压缩信息文本文件存储在此目录下
        output_file= "./output/Image_MPS_COMP.xlsx"
    )
evaluator.compare_images_in_directories()