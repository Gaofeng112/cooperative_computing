import os
import pandas as pd

def calculate_compression_rate(file_path):
    f=open(file_path)
    lines=f.readlines()
    rate_all=0
    all=0
    for line in lines:
        rate_all+=float(line.split(',')[0])*float(line.split(',')[1])
        all += float(line.split(',')[1])
    print(rate_all/all)
    return rate_all/all


def write_to_excel(results, output_file):
    df = pd.DataFrame(results, columns=['File', 'Compression Rate'])
    df.to_excel(output_file, index=False)

def main():
    # 假设所有结果文件都在同一个目录下
    directory = '../result'
    results = []

    # 遍历目录中的所有 txt 文件
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory, filename)
            compression_rate = calculate_compression_rate(file_path)
            if compression_rate is not None:
                results.append((filename, compression_rate))

    # 将结果写入 Excel 文件
    output_file = 'compression_rates.xlsx'
    write_to_excel(results, output_file)

if __name__ == '__main__':
    main()