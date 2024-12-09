import os
import shutil

# 原始图片目录
original_dir = '../images_origin'

# 新的目标目录
new_dir = '../images_origin_new'

# 创建新目录（如果不存在）
os.makedirs(new_dir, exist_ok=True)

# 遍历原始目录中的所有文件
for filename in os.listdir(original_dir):
    if filename.startswith("origin") and filename.endswith(".png"):
        # 提取数字部分
        num_part = filename[len("origin"):].strip()
        
        # 构建新的文件名
        new_filename = f"origin_{num_part}"
        
        # 完整的新文件路径
        old_path = os.path.join(original_dir, filename)
        new_path = os.path.join(new_dir, new_filename)
        
        # 复制文件到新位置
        shutil.copy(old_path, new_path)

print("Files renamed and copied successfully!")