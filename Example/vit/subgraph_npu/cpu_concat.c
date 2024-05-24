#include <stdio.h>
#include <stdlib.h>

void concatArrays(float *arr1, float *arr2, int depth, int rows1, int rows2, int cols, float *result) {
    int total_rows = rows1 + rows2;
    for (int i = 0; i < depth; i++) {
        for (int j = 0; j < rows1; j++) {
            for (int k = 0; k < cols; k++) {
                result[i * total_rows * cols + j * cols + k] = arr1[i * rows1 * cols + j * cols + k];
            }
        }
        for (int j = 0; j < rows2; j++) {
            for (int k = 0; k < cols; k++) {
                result[i * total_rows * cols + (rows1 + j) * cols + k] = arr2[i * rows2 * cols + j * cols + k];
            }
        }
    }
}

void printArray(float *arr, int depth, int rows, int cols) {
    for (int i = 0; i < depth; i++) {
        printf("Depth %d:\n", i);
        for (int j = 0; j < rows; j++) {
            for (int k = 0; k < cols; k++) {
                printf("%.2f ", arr[i * rows * cols + j * cols + k]);
            }
            printf("\n");
        }
    }
}

int main() {
    FILE *file = fopen("vit_180.bin", "rb"); // 打开二进制文件
    if (file == NULL) {
        printf("Error opening file!\n");
        return 1;
    }

    int depth = 1; // 数组深度
    int rows1 = 196; // 第一个数组的行数
    int rows2 = 1; // 第二个数组的行数
    int cols = 768; // 数组的列数

    // 读取二进制文件中的浮点数组 arr2
    float arr2[1][1][768];
    fread(arr2, sizeof(float), depth * rows2 * cols, file);

    // 打印 arr2 的内容
    printf("Array arr2:\n");
    printArray((float *)arr2, depth, rows2, cols);

    fclose(file);

    // 创建 arr1 数组并填充假数据
    float *arr1 = (float *)malloc(depth * rows1 * cols * sizeof(float));
    // 这里省略了从文件中读取 arr1 的过程，你需要根据实际情况进行填充

    float result[1][197][768]; // 连接后的数组大小

    concatArrays(arr1, (float *)arr2, depth, rows1, rows2, cols, (float *)result);

    printf("\nConcatenated array:\n");
    printArray((float *)result, depth, 197, 768);

    // 释放 arr1 的内存
    free(arr1);

    return 0;
}
