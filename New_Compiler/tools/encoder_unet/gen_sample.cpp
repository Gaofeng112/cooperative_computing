#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// 生成高斯随机数的函数
float generate_gaussian_noise(float mean, float stddev) {
    static int haveSpare = 0;
    static double rand1, rand2;

    if (haveSpare) {
        haveSpare = 0;
        return mean + stddev * sqrt(rand1) * sin(rand2);
    }

    haveSpare = 1;

    rand1 = rand() / ((double) RAND_MAX);
    if (rand1 < 1e-100) rand1 = 1e-100;
    rand1 = -2 * log(rand1);
    rand2 = (rand() / ((double) RAND_MAX)) * M_PI * 2;

    return mean + stddev * sqrt(rand1) * cos(rand2);
}

int main() {
    // 数组的大小
    int size = 1 * 4 * 32 * 32;

    // 分配内存
    float *data = (float *)malloc(size * sizeof(float));
    if (data == NULL) {
        fprintf(stderr, "内存分配失败\n");
        return 1;
    }

    // 初始化随机数种子
    srand(time(NULL));

    // 使用高斯随机噪声填充数组
    float mean = 0.0;
    float stddev = 1.0;
    for (int i = 0; i < size; ++i) {
        data[i] = generate_gaussian_noise(mean, stddev);
    }

    // 将数据写入二进制文件
    FILE *file = fopen("sample.bin", "wb");
    if (file == NULL) {
        fprintf(stderr, "文件打开失败\n");
        free(data);
        return 1;
    }

    size_t written = fwrite(data, sizeof(float), size, file);
    if (written != size) {
        fprintf(stderr, "文件写入失败\n");
        fclose(file);
        free(data);
        return 1;
    }

    fclose(file);

    // 输出部分数据进行检查
    for (int i = 0; i < 10; ++i) {
        printf("data[%d] = %f\n", i, data[i]);
    }

    // 释放内存
    free(data);

    return 0;
}
