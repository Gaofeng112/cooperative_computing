import re

def process_file(input_file, output_file):
    with open(input_file, 'r') as file:
        lines = file.readlines()

    new_lines = []
    for line in lines:
        # 查找 CPUsubgraph 后面的数字
        match = re.search(r'CPUsubgraph(\d+)', line)
        if match:
            # 提取数字并加 57
            num = int(match.group(1))
            new_num = num + 57
            # 替换原数字
            line = re.sub(r'CPUsubgraph\d+', f'CPUsubgraph{new_num}', line)
        new_lines.append(line)

    with open(output_file, 'w') as file:
        file.writelines(new_lines)

# 指定输入和输出文件
input_file = '../1016_subgraphs/subgraphs1016_ios1.txt'
output_file = '../1016_subgraphs/subgraphs1016_ios2.txt'

process_file(input_file, output_file)

print(f"Processed file saved as {output_file}")