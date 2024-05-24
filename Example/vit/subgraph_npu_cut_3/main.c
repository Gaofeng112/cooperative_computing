/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/* auto generate by HHB_VERSION "2.4.5" */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <libgen.h>
#include <unistd.h>
#include "io.h"
#include "shl_ref.h"
#include "process.h"
#include "shl_c920.h"

#define MIN(x, y)           ((x) < (y) ? (x) : (y))
#define FILE_LENGTH         1028
#define SHAPE_LENGHT        128
#define FILE_PREFIX_LENGTH  (1028 - 2 * 128)

void *csinn_0(char *params);
void csinn_update_input_and_run_0(struct csinn_tensor **input_tensors , void *sess);
void *csinn_1(char *params);
void csinn_update_input_and_run_1(struct csinn_tensor **input_tensors , void *sess);
void *csinn_2(char *params);
void csinn_update_input_and_run_2(struct csinn_tensor **input_tensors , void *sess);
#define csinn_nbg(...) NULL

int input_size[] = {1 * 3 * 224 * 224, };
int sess2_input_size[] = {1 * 178, };
const char model_name[] = "network";

#define RESIZE_HEIGHT       224
#define RESIZE_WIDTH        224
#define CROP_HEGHT          224
#define CROP_WIDTH          224
#define R_MEAN              0.0
#define G_MEAN              0.0
#define B_MEAN              0.0
#define SCALE               0.00392156862745098

/*
 * Preprocess function
 */
void preprocess(struct image_data *img, int is_rgb, int to_bgr)
{
    uint32_t new_height, new_width;
    uint32_t min_side;
    if (is_rgb) {
        im2rgb(img);
    }
    if (RESIZE_WIDTH == 0) {
        min_side = MIN(img->shape[0], img->shape[1]);
        new_height = (uint32_t) (img->shape[0] * (((float)RESIZE_HEIGHT) / (float)min_side));
        new_width = (uint32_t) (img->shape[1] * (((float)RESIZE_HEIGHT) / (float)min_side));
        imresize(img, new_height, new_width);
    } else {
        imresize(img, RESIZE_HEIGHT, RESIZE_WIDTH);
    }
    data_crop(img, CROP_HEGHT, CROP_WIDTH);
    sub_mean(img, R_MEAN, G_MEAN, B_MEAN);
    data_scale(img, SCALE);
    if(to_bgr) {
        imrgb2bgr(img);
    }
    imhwc2chw(img);
}

static void print_tensor_info(struct csinn_tensor *t) {
    printf("\n=== tensor info ===\n");
    printf("shape: ");
    for (int j = 0; j < t->dim_count; j++) {
        printf("%d ", t->dim[j]);
    }
    printf("\n");
    if (t->dtype == CSINN_DTYPE_UINT8) {
        printf("scale: %f\n", t->qinfo->scale);
        printf("zero point: %d\n", t->qinfo->zero_point);
    }
    printf("data pointer: %p\n", t->data);
}


/*
 * Postprocess function
 */
static void postprocess_npu(void *sess, const char *filename_prefix) {
    int output_num, input_num;
    struct csinn_tensor *input = csinn_alloc_tensor(NULL);
    struct csinn_tensor *output = csinn_alloc_tensor(NULL);

    input_num = csinn_get_input_number(sess);
    for (int i = 0; i < input_num; i++) {
        input->data = NULL;
        csinn_get_input(i, input, sess);
        print_tensor_info(input);
        
    }

    output_num = csinn_get_output_number(sess);
    for (int i = 0; i < output_num; i++) {
        output->data = NULL;
        csinn_get_output(i, output, sess);
        print_tensor_info(output);

        struct csinn_tensor *foutput = shl_ref_tensor_transform_f32(output);
        shl_show_top5(foutput, sess);
        char filename[FILE_LENGTH] = {0};
        char shape[SHAPE_LENGHT] = {0};
        shape2string(output->dim, output->dim_count, shape, SHAPE_LENGHT);
        snprintf(filename, FILE_LENGTH, "%s_output%u_%s.txt", filename_prefix, i, shape);
        int output_size = csinn_tensor_size(foutput);
        save_data_to_file(filename, (float*)foutput->data, output_size);

        shl_ref_tensor_transform_free_f32(foutput);

    }
    csinn_free_tensor(input);
    csinn_free_tensor(output);
}

static void postprocess_cpu(void *sess, const char *filename_prefix) {
    int output_num, input_num;
    struct csinn_tensor *input = csinn_alloc_tensor(NULL);
    struct csinn_tensor *output = csinn_alloc_tensor(NULL);

    input_num = csinn_get_input_number(sess);
    for (int i = 0; i < input_num; i++) {
        input->data = NULL;
        csinn_get_input(i, input, sess);
        print_tensor_info(input);
    }
    output_num = csinn_get_output_number(sess);
    for (int i = 0; i < output_num; i++) {
        output->data = NULL;
        csinn_get_output(i, output, sess);
        print_tensor_info(output);

        struct csinn_tensor *foutput = shl_ref_tensor_transform_f32(output);
        shl_show_top5(foutput, sess);

        #if 1
        char filename[FILE_PREFIX_LENGTH] = {0};
        char shape[SHAPE_LENGHT] = {0};
        shape2string(output->dim, output->dim_count, shape, SHAPE_LENGHT);
        snprintf(filename, FILE_PREFIX_LENGTH, "%s_output%u_%s.txt", filename_prefix, i, shape);
        int output_size = csinn_tensor_size(foutput);
        float* fdata = shl_c920_output_to_f32_dtype(i, output->data, sess);
        save_data_to_file(filename, fdata, output_size);
        shl_mem_free(fdata);
        #endif

        shl_ref_tensor_transform_free_f32(foutput);
        if (!output->is_const) {
            shl_mem_free(output->data);
        }
    }
    csinn_free_tensor(input);
    csinn_free_tensor(output);
}

static void get_top5(float *buf, uint32_t size, float *prob, uint32_t *cls)
{
    uint32_t i, j, k;

    memset(prob, 0xfe, sizeof(float) * 5);
    memset(cls, 0xff, sizeof(uint32_t) * 5);

    for (j = 0; j < 5; j++)
    {
        for (i = 0; i < size; i++)
        {
        for (k = 0; k < 5; k++)
        {
            if (i == cls[k])
            {
            break;
            }
        }

        if (k != 5)
        {
            continue;
        }

        if (buf[i] > prob[j])
        {
            prob[j] = buf[i];
            cls[j] = i;
        }
        }
    }
}

static float* get_data_from_file(const char* filename, uint32_t size) {
    uint32_t j;
    float fval = 0.0;
    float* buffer = NULL;
    FILE* fp = fopen(filename, "rb");
    if (fp == NULL) {
        printf("Invalid input file: %s\n", filename);
        return NULL;
    }

    buffer = (float *)malloc(size * sizeof(float));
    if (buffer == NULL) {
        printf("Malloc fail\n");
        return NULL;
    }
    for (j = 0; j < size; j++) {
        if (fscanf(fp, "%f ", &fval) != 1) {
        printf("Invalid input size\n");
        return NULL;
        } else {
        buffer[j] = fval;
        }
    }

    fclose(fp);
    return buffer;
}

static void load_result_and_show_lable(const char *filename_prefix)
{
    uint32_t i = 0, size = 1000;
    uint32_t cls[5];
    float prob[5];
    char filename[FILE_PREFIX_LENGTH] = {0};
    snprintf(filename, FILE_PREFIX_LENGTH, "%s_output0_%u_%u.txt", filename_prefix, 1, 1000);
    float* output_data = get_data_from_file(filename, 1000);

    get_top5(output_data, size, prob, cls);

    FILE *file = fopen("synset.txt", "r");
    if (file == NULL) {
        printf("Error opening synset.txt\n");
        return;
    }

    char line[256]; // 假设每行最大长度为256
    char *labels[1000]; // 假设最多有1000个标签
    int index = 0;
    while (fgets(line, sizeof(line), file)) {
        line[strcspn(line, "\n")] = '\0'; // 去除换行符
        labels[index] = strdup(line); // 使用strdup动态分配内存保存标签
        index++;
    }
    fclose(file);

    printf(" ********** probability top5: ********** \n");
    size = size > 5 ? 5 : size;
    for (i = 0; i < size; i++) {
        printf("%s\n", labels[cls[i]]);
        free(labels[cls[i]]); // 释放每个标签的内存
    }
}

void *create_graph_0(char *params_path) {
    int binary_size;
    char *params = get_binary_from_file(params_path, &binary_size);
    if (params == NULL) {
        return NULL;
    }

    char *suffix = params_path + (strlen(params_path) - 7);
    if (strcmp(suffix, ".params") == 0) {
        // create general graph
        return csinn_0(params);
    }

    suffix = params_path + (strlen(params_path) - 3);
    if (strcmp(suffix, ".bm") == 0) {
        struct shl_bm_sections *section = (struct shl_bm_sections *)(params + 4128);
        if (section->graph_offset) {
            return csinn_import_binary_model(params);
        } else {
            return csinn_0(params + section->params_offset * 4096);
        }
    } else {
        return NULL;
    }
}

void *create_graph_1(char *params_path) {
    char *params = get_binary_from_file(params_path, NULL);
    if (params == NULL) {
        return NULL;
    }

    char *suffix = params_path + (strlen(params_path) - 7);
    if (strcmp(suffix, ".params") == 0) {
        // create general graph
        return csinn_1(params);
    }

    suffix = params_path + (strlen(params_path) - 3);
    if (strcmp(suffix, ".bm") == 0) {
        struct shl_bm_sections *section = (struct shl_bm_sections *)(params + 4128);
        if (section->graph_offset) {
            return csinn_import_binary_model(params);
        } else {
            return csinn_1(params + section->params_offset * 4096);
        }
    } else {
        return NULL;
    }
}

void *create_graph_2(char *params_path) {
    int binary_size;
    char *params = get_binary_from_file(params_path, &binary_size);
    if (params == NULL) {
        return NULL;
    }

    char *suffix = params_path + (strlen(params_path) - 7);
    if (strcmp(suffix, ".params") == 0) {
        // create general graph
        return csinn_2(params);
    }

    suffix = params_path + (strlen(params_path) - 3);
    if (strcmp(suffix, ".bm") == 0) {
        struct shl_bm_sections *section = (struct shl_bm_sections *)(params + 4128);
        if (section->graph_offset) {
            return csinn_import_binary_model(params);
        } else {
            return csinn_2(params + section->params_offset * 4096);
        }
    } else {
        return NULL;
    }
}

void get_npu_output(struct csinn_tensor **out, struct csinn_session *sess)
{
    int output_num;
    output_num = csinn_get_output_number(sess);

    struct csinn_tensor *output;
    for (int i = 0; i < output_num; i++)
    {
        output = csinn_alloc_tensor(NULL);
        output->data = NULL;
        csinn_get_output(i, output, sess);
        print_tensor_info(output);

        out[i] = output;
        // printf("output type:%d", output->dtype);
        struct csinn_tensor *ret = csinn_alloc_tensor(NULL);
        csinn_tensor_copy(ret, output);
        if (ret->qinfo != NULL)
        {
            shl_mem_free(ret->qinfo);
            ret->qinfo = NULL;
        }
        ret->quant_channel = 0;
        ret->dtype = CSINN_DTYPE_FLOAT32;
        ret->data = shl_c920_output_to_f32_dtype(i, output->data, sess);
        out[i] = ret;
    }

    csinn_free_tensor(output);
}

void get_cpu_output(struct csinn_tensor **out, struct csinn_session *sess)
{
    int output_num;
    output_num = csinn_get_output_number(sess);

    struct csinn_tensor *output;
    for (int i = 0; i < output_num; i++)
    {
        output = csinn_alloc_tensor(NULL);
        output->data = NULL;
        csinn_get_output(i, output, sess);
        print_tensor_info(output);

        out[i] = output;
        // printf("output type:%d", output->dtype);
        struct csinn_tensor *ret = csinn_alloc_tensor(NULL);
        csinn_tensor_copy(ret, output);
        if (ret->qinfo != NULL)
        {
            shl_mem_free(ret->qinfo);
            ret->qinfo = NULL;
        }
        ret->quant_channel = 0;
        ret->dtype = CSINN_DTYPE_FLOAT32;
        ret->data = shl_c920_output_to_f32_dtype(i, output->data, sess);
        out[i] = ret;
    }

    csinn_free_tensor(output);
}

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

int main(int argc, char **argv) {
    char **data_path = NULL;
    int input_num = 1;
    int output_num = 1;
    int input_group_num = 1;
    int i;

    int sess0_output_num = 1;
    int sess1_iutput_num = 1;
    int sess1_output_num = 1;
    int sess2_iutput_num = 1;

    if (argc < (5 + input_num)) { //exe 两个hhb.bm 一个concat用的bin文件
        printf("Please set valide args: ./model.elf hhb0.bm hhb1.bm xx.bin"
                "[data1 data2 ...]|[.txt]\n");
        return -1;
    } else {
        data_path = argv + 5;
        input_group_num = (argc - 5) / input_num;
    }

    // 将 argv[4] 存储在一个字符串中
    char *filename = argv[4];

    void *sess0 = create_graph_0(argv[1]);
    void *sess1 = create_graph_1(argv[2]);
    void *sess2 = create_graph_2(argv[3]);

    struct csinn_tensor* input_tensors[input_num];
    input_tensors[0] = csinn_alloc_tensor(NULL);
    input_tensors[0]->dim_count = 4;
    input_tensors[0]->dim[0] = 1;
    input_tensors[0]->dim[1] = 3;
    input_tensors[0]->dim[2] = 224;
    input_tensors[0]->dim[3] = 224;

    struct csinn_tensor *sess0_output[sess0_output_num];
    sess0_output[0] = csinn_alloc_tensor(NULL);
    sess0_output[0]->dim_count = 3;
    sess0_output[0]->dim[0] = 1;
    sess0_output[0]->dim[1] = 196;
    sess0_output[0]->dim[2] = 768;

    struct csinn_tensor *sess1_input[sess1_iutput_num];
    sess1_input[0] = csinn_alloc_tensor(NULL);
    sess1_input[0]->dim_count = 3;
    sess1_input[0]->dim[0] = 1;
    sess1_input[0]->dim[1] = 197;
    sess1_input[0]->dim[2] = 768;

    struct csinn_tensor *sess1_output[sess1_output_num];
    sess1_output[0] = csinn_alloc_tensor(NULL);
    sess1_output[0]->dim_count = 2;
    sess1_output[0]->dim[0] = 1;
    sess1_output[0]->dim[1] = 768;

    struct csinn_tensor *sess2_input[sess2_iutput_num];
    sess2_input[0] = csinn_alloc_tensor(NULL);
    sess2_input[0]->dim_count = 2;
    sess2_input[0]->dim[0] = 1;
    sess2_input[0]->dim[1] = 1000;

    float *sess0_inputf[input_num];
    int8_t *sess0_inputi[input_num];
    void *sess0_input_aligned[input_num];

    float *sess1_inputf[sess1_iutput_num];

    float *sess2_inputf[sess2_iutput_num];
    int8_t *sess2_inputi[sess2_iutput_num];
    void *sess2_input_aligned[input_num];

    for (i = 0; i < input_num; i++)
    {
        input_size[i] = csinn_tensor_byte_size(((struct csinn_session *)sess0)->input[i]);
        sess0_input_aligned[i] = shl_mem_alloc_aligned(input_size[i], 0);
    }

    for (i = 0; i < sess2_iutput_num; i++)
    {
        sess2_input_size[i] = csinn_tensor_byte_size(((struct csinn_session *)sess2)->input[i]);
        sess2_input_aligned[i] = shl_mem_alloc_aligned(sess2_input_size[i], 0);
    }


    char filename_prefix[FILE_PREFIX_LENGTH] = {0};

    uint64_t start_time, end_time;
    for (i = 0; i < input_group_num; i++) {
        /* set input */
        for (int j = 0; j < input_num; j++) {
            int input_len = csinn_tensor_size(((struct csinn_session *)sess0)->input[j]);
            struct image_data *img = get_input_data(data_path[i * input_num + j], input_len);
            if (get_file_type(data_path[i * input_num + j]) == FILE_PNG || get_file_type(data_path[i * input_num + j]) == FILE_JPEG) {
                preprocess(img, 1, 0);
            }
            sess0_inputf[j] = img->data;
            free_image_data(img);

            sess0_inputi[j] = shl_ref_f32_to_input_dtype(j, sess0_inputf[j], sess0);
        }
        memcpy(sess0_input_aligned[0], sess0_inputi[0], input_size[0]);
        input_tensors[0]->data = sess0_input_aligned[0];

        start_time = shl_get_timespec();
        csinn_update_input_and_run_0(input_tensors, sess0);
        // 需要读取那个vit_180.bin然后和第一部分的输出进行concat计算
        get_npu_output(sess0_output, sess0);

        FILE *file = fopen(filename, "rb"); // 打开二进制文件
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

        // // 打印 arr2 的内容
        // printf("Array arr2:\n");
        // printArray((float *)arr2, depth, rows2, cols);

        fclose(file);

        float result[1][197][768]; // 连接后的数组大小
        concatArrays(sess0_output[0]->data, (float *)arr2, depth, rows1, rows2, cols, (float *)result);

        // printf("\nConcatenated array:\n");
        // printArray((float *)result, depth, 197, 768);

        // 开始第二部分
        for (int j = 0; j < sess1_iutput_num; j++) {
            sess1_inputf[j] = (float *)result;
            sess1_input[j]->data = shl_c920_f32_to_input_dtype(j, sess1_inputf[j], sess1);
        }
        csinn_update_input_and_run_1(sess1_input, sess1);

        get_cpu_output(sess1_output,sess1);

// // 打开二进制文件以写入
// FILE *output_file = fopen("vit_1215.bin", "wb");
// if (output_file == NULL) {
//     perror("Error opening output file");
//     return 1;
// }

// // 将数据写入二进制文件
// fwrite(sess1_output[0]->data, sizeof(float), 768, output_file);

// // 关闭二进制文件
// fclose(output_file);

        // 开始第三部分
        for (int j = 0; j < sess2_iutput_num; j++) {
            sess2_inputf[j] = (float *)sess1_output[j]->data;
            sess2_inputi[j] = shl_ref_f32_to_input_dtype(j, sess2_inputf[j], sess2);
        }
        memcpy(sess2_input_aligned[0], sess2_inputi[0], sess2_input_size[0]);
        sess2_input[0]->data = sess2_input_aligned[0];
        csinn_update_input_and_run_2(sess2_input, sess2);

        end_time = shl_get_timespec();
        printf("Run graph execution time: %.5fms, FPS=%.2f\n", ((float)(end_time-start_time))/1000000,
                    1000000000.0/((float)(end_time-start_time)));

        snprintf(filename_prefix, FILE_PREFIX_LENGTH, "%s", basename(data_path[i * input_num]));
        postprocess_npu(sess2, filename_prefix);
        load_result_and_show_lable(filename_prefix);
        // postprocess_cpu(sess1, filename_prefix);

        for (int j = 0; j < input_num; j++) {
            shl_mem_free(sess0_inputf[j]);
            shl_mem_free(sess0_inputi[j]);
        }
        // for (int j = 0; j < sess1_iutput_num; j++) {
        //     shl_mem_free(sess1_inputf[j]);
        // }
        for (int j = 0; j < sess2_iutput_num; j++) {
            shl_mem_free(sess2_inputf[j]);
            shl_mem_free(sess2_inputi[j]);
        }
    }

    for (int j = 0; j < input_num; j++) {
        csinn_free_tensor(input_tensors[j]);
        shl_mem_free(sess0_input_aligned[j]);
    }

    for(int j = 0; j<sess0_output_num; j++){
        csinn_free_tensor(sess0_output[j]);
        csinn_free_tensor(sess1_input[j]);
    }

    for(int j = 0; j<sess1_output_num; j++){
        csinn_free_tensor(sess1_output[j]);
        csinn_free_tensor(sess2_input[j]);
    }

    csinn_session_deinit(sess0);
    csinn_free_session(sess0);

    csinn_session_deinit(sess1);
    csinn_free_session(sess1);

    csinn_session_deinit(sess2);
    csinn_free_session(sess2);

    return 0;
}

