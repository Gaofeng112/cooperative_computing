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
#include <math.h>
#include "io.h"
#include "shl_ref.h"

#define MIN(x, y)           ((x) < (y) ? (x) : (y))
#define FILE_LENGTH         1028
#define SHAPE_LENGHT        128
#define FILE_PREFIX_LENGTH  (1028 - 2 * 128)

void *csinn_npu_Subgraphs_0(char *params_base);
void csinn_update_input_and_run_npu_Subgraphs_0(struct csinn_tensor **input_tensors , void *sess);
void *csinn_npu_Subgraphs_1(char *params_base);
void csinn_update_input_and_run_npu_Subgraphs_1(struct csinn_tensor **input_tensors , void *sess);
void *csinn_npu_Subgraphs_2(char *params_base);
void csinn_update_input_and_run_npu_Subgraphs_2(struct csinn_tensor **input_tensors , void *sess);
void *csinn_npu_Subgraphs_3(char *params_base);
void csinn_update_input_and_run_npu_Subgraphs_3(struct csinn_tensor **input_tensors , void *sess);
void *csinn_npu_Subgraphs_4(char *params_base);
void csinn_update_input_and_run_npu_Subgraphs_4(struct csinn_tensor **input_tensors , void *sess);
void *csinn_npu_Subgraphs_5(char *params_base);
void csinn_update_input_and_run_npu_Subgraphs_5(struct csinn_tensor **input_tensors , void *sess);
void *csinn_npu_Subgraphs_6(char *params_base);
void csinn_update_input_and_run_npu_Subgraphs_6(struct csinn_tensor **input_tensors , void *sess);
void *csinn_npu_Subgraphs_7(char *params_base);
void csinn_update_input_and_run_npu_Subgraphs_7(struct csinn_tensor **input_tensors , void *sess);
void *csinn_npu_Subgraphs_8(char *params_base);
void csinn_update_input_and_run_npu_Subgraphs_8(struct csinn_tensor **input_tensors , void *sess);
void *csinn_npu_Subgraphs_9(char *params_base);
void csinn_update_input_and_run_npu_Subgraphs_9(struct csinn_tensor **input_tensors , void *sess);
void *csinn_npu_Subgraphs_10(char *params_base);
void csinn_update_input_and_run_npu_Subgraphs_10(struct csinn_tensor **input_tensors , void *sess);
void *csinn_npu_Subgraphs_11(char *params_base);
void csinn_update_input_and_run_npu_Subgraphs_11(struct csinn_tensor **input_tensors , void *sess);
void *csinn_npu_Subgraphs_12(char *params_base);
void csinn_update_input_and_run_npu_Subgraphs_12(struct csinn_tensor **input_tensors , void *sess);
void *csinn_npu_Subgraphs_13(char *params_base);
void csinn_update_input_and_run_npu_Subgraphs_13(struct csinn_tensor **input_tensors , void *sess);
void *csinn_npu_Subgraphs_14(char *params_base);
void csinn_update_input_and_run_npu_Subgraphs_14(struct csinn_tensor **input_tensors , void *sess);
void *csinn_npu_Subgraphs_15(char *params_base);
void csinn_update_input_and_run_npu_Subgraphs_15(struct csinn_tensor **input_tensors , void *sess);
void *csinn_npu_Subgraphs_16(char *params_base);
void csinn_update_input_and_run_npu_Subgraphs_16(struct csinn_tensor **input_tensors , void *sess);
void *csinn_npu_Subgraphs_17(char *params_base);
void csinn_update_input_and_run_npu_Subgraphs_17(struct csinn_tensor **input_tensors , void *sess);
void *csinn_npu_Subgraphs_18(char *params_base);
void csinn_update_input_and_run_npu_Subgraphs_18(struct csinn_tensor **input_tensors , void *sess);
void *csinn_npu_Subgraphs_19(char *params_base);
void csinn_update_input_and_run_npu_Subgraphs_19(struct csinn_tensor **input_tensors , void *sess);
void *csinn_npu_Subgraphs_20(char *params_base);
void csinn_update_input_and_run_npu_Subgraphs_20(struct csinn_tensor **input_tensors , void *sess);
void *csinn_npu_Subgraphs_21(char *params_base);
void csinn_update_input_and_run_npu_Subgraphs_21(struct csinn_tensor **input_tensors , void *sess);
void *csinn_npu_Subgraphs_22(char *params_base);
void csinn_update_input_and_run_npu_Subgraphs_22(struct csinn_tensor **input_tensors , void *sess);
void *csinn_npu_Subgraphs_23(char *params_base);
void csinn_update_input_and_run_npu_Subgraphs_23(struct csinn_tensor **input_tensors , void *sess);
void *csinn_npu_Subgraphs_24(char *params_base);
void csinn_update_input_and_run_npu_Subgraphs_24(struct csinn_tensor **input_tensors , void *sess);
void *csinn_npu_Subgraphs_25(char *params_base);
void csinn_update_input_and_run_npu_Subgraphs_25(struct csinn_tensor **input_tensors , void *sess);
void *csinn_npu_Subgraphs_26(char *params_base);
void csinn_update_input_and_run_npu_Subgraphs_26(struct csinn_tensor **input_tensors , void *sess);
void *csinn_npu_Subgraphs_27(char *params_base);
void csinn_update_input_and_run_npu_Subgraphs_27(struct csinn_tensor **input_tensors , void *sess);
void *csinn_npu_Subgraphs_28(char *params_base);
void csinn_update_input_and_run_npu_Subgraphs_28(struct csinn_tensor **input_tensors , void *sess);
void *csinn_npu_Subgraphs_29(char *params_base);
void csinn_update_input_and_run_npu_Subgraphs_29(struct csinn_tensor **input_tensors , void *sess);
void *csinn_npu_Subgraphs_30(char *params_base);
void csinn_update_input_and_run_npu_Subgraphs_30(struct csinn_tensor **input_tensors , void *sess);
void *csinn_npu_Subgraphs_31(char *params_base);
void csinn_update_input_and_run_npu_Subgraphs_31(struct csinn_tensor **input_tensors , void *sess);
void *csinn_npu_Subgraphs_32(char *params_base);
void csinn_update_input_and_run_npu_Subgraphs_32(struct csinn_tensor **input_tensors , void *sess);
void *csinn_npu_Subgraphs_33(char *params_base);
void csinn_update_input_and_run_npu_Subgraphs_33(struct csinn_tensor **input_tensors , void *sess);
void *csinn_npu_Subgraphs_34(char *params_base);
void csinn_update_input_and_run_npu_Subgraphs_34(struct csinn_tensor **input_tensors , void *sess);
void *csinn_npu_Subgraphs_35(char *params_base);
void csinn_update_input_and_run_npu_Subgraphs_35(struct csinn_tensor **input_tensors , void *sess);
void *csinn_npu_Subgraphs_36(char *params_base);
void csinn_update_input_and_run_npu_Subgraphs_36(struct csinn_tensor **input_tensors , void *sess);
void *csinn_npu_Subgraphs_37(char *params_base);
void csinn_update_input_and_run_npu_Subgraphs_37(struct csinn_tensor **input_tensors , void *sess);
void *csinn_npu_Subgraphs_38(char *params_base);
void csinn_update_input_and_run_npu_Subgraphs_38(struct csinn_tensor **input_tensors , void *sess);
void *csinn_npu_Subgraphs_39(char *params_base);
void csinn_update_input_and_run_npu_Subgraphs_39(struct csinn_tensor **input_tensors , void *sess);
void *csinn_npu_Subgraphs_40(char *params_base);
void csinn_update_input_and_run_npu_Subgraphs_40(struct csinn_tensor **input_tensors , void *sess);
void *csinn_npu_Subgraphs_41(char *params_base);
void csinn_update_input_and_run_npu_Subgraphs_41(struct csinn_tensor **input_tensors , void *sess);

void *csinn_cpu_Subgraphs_0(char *params_base);
void csinn_update_input_and_run_cpu_Subgraphs_0(struct csinn_tensor **input_tensors , void *sess);
void *csinn_cpu_Subgraphs_1(char *params_base);
void csinn_update_input_and_run_cpu_Subgraphs_1(struct csinn_tensor **input_tensors , void *sess);
void *csinn_cpu_Subgraphs_2(char *params_base);
void csinn_update_input_and_run_cpu_Subgraphs_2(struct csinn_tensor **input_tensors , void *sess);
void *csinn_cpu_Subgraphs_3(char *params_base);
void csinn_update_input_and_run_cpu_Subgraphs_3(struct csinn_tensor **input_tensors , void *sess);
void *csinn_cpu_Subgraphs_4(char *params_base);
void csinn_update_input_and_run_cpu_Subgraphs_4(struct csinn_tensor **input_tensors , void *sess);
void *csinn_cpu_Subgraphs_5(char *params_base);
void csinn_update_input_and_run_cpu_Subgraphs_5(struct csinn_tensor **input_tensors , void *sess);
void *csinn_cpu_Subgraphs_6(char *params_base);
void csinn_update_input_and_run_cpu_Subgraphs_6(struct csinn_tensor **input_tensors , void *sess);
void *csinn_cpu_Subgraphs_7(char *params_base);
void csinn_update_input_and_run_cpu_Subgraphs_7(struct csinn_tensor **input_tensors , void *sess);
void *csinn_cpu_Subgraphs_8(char *params_base);
void csinn_update_input_and_run_cpu_Subgraphs_8(struct csinn_tensor **input_tensors , void *sess);
void *csinn_cpu_Subgraphs_9(char *params_base);
void csinn_update_input_and_run_cpu_Subgraphs_9(struct csinn_tensor **input_tensors , void *sess);


#define csinn_nbg(...) NULL

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
static void postprocess(void *sess, const char *filename_prefix) {
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

typedef void *(*SubgraphsCreator)(char *);

void *create_Subgraphs(char *params_path, SubgraphsCreator creator_function) {
    int binary_size;
    char *params = get_binary_from_file(params_path, &binary_size);
    if (params == NULL) {
        return NULL;
    }

    char *suffix = params_path + (strlen(params_path) - 7);
    if (strcmp(suffix, ".params") == 0) {
        // create general graph
        return creator_function(params);
    }

    suffix = params_path + (strlen(params_path) - 3);
    if (strcmp(suffix, ".bm") == 0) {
        struct shl_bm_sections *section = (struct shl_bm_sections *)(params + 4128);
        if (section->graph_offset) {
            return csinn_import_binary_model(params);
        } else {
            return creator_function(params + section->params_offset * 4096);
        }
    } else {
        return NULL;
    }
}

void get_output(struct csinn_tensor **out, struct csinn_session *sess) {
    int output_num;
    output_num = csinn_get_output_number(sess);

    struct csinn_tensor *output;
    for (int i = 0; i < output_num; i++)
    {
        output = csinn_alloc_tensor(NULL);
        output->data = NULL;
        csinn_get_output(i, output, sess);
        // print_tensor_info(output);

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
        // printf("out[i]:\r\n");
        // print_tensor_info(out[i]);
    }

    csinn_free_tensor(output);
}

void instance_normalization(float *input, float *scale, float *B, int height, int channels, int width, float *output) {
    // 遍历每个通道
    for (int h = 0; h < height; ++h) {
        for (int c = 0; c < channels; ++c) {
            float mean = 0.0;
            float variance = 0.0;

            // 计算每个通道的均值和方差
            for (int w = 0; w < width; ++w) {
                int index = h * channels * width + c * width + w;
                mean += input[index];
            }
            mean /= width;

            for (int w = 0; w < width; ++w) {
                int index = h * channels * width + c * width + w;
                variance += pow(input[index] - mean, 2);
            }
            variance /= width;

            // 计算每个像素的Instance Normalization并应用scale和B
            for (int w = 0; w < width; ++w) {
                int index = h * channels * width + c * width + w;
                float normalized_value = (input[index] - mean) / sqrt(variance + 1e-5);
                output[index] = normalized_value * scale[c] + B[c];
            }
        }
    }
}

void multiply4DMatrix(float *mat1, float *mat2, int *result, int dim1, int dim2, int dim3_a, int dim4_a, int dim3_b, int dim4_b) {
    printf("start multiply4DMatrix\r\n");
    // 执行矩阵乘法
    for(int i = 0; i < dim1; i++) {
        for(int j = 0; j < dim2; j++) {
            for(int k = 0; k < dim3_a; k++) {
                for(int l = 0; l < dim4_b; l++) {
                    result[i * dim2 * dim3_a * dim4_b + j * dim3_a * dim4_b + k * dim4_b + l] = 0;
                    for(int m = 0; m < dim4_a; m++) {
                        result[i * dim2 * dim3_a * dim4_b + j * dim3_a * dim4_b + k * dim4_b + l] +=
                            mat1[i * dim2 * dim3_a * dim4_a + j * dim3_a * dim4_a + k * dim4_a + m] *
                            mat2[i * dim2 * dim3_b * dim4_b + j * dim3_b * dim4_b + m * dim4_b + l];
                    }
                }
            }
        }
    }
}

void concat(float* array1, int batch_size, int channels1, int height, int width, float* array2, int channels2, float* result, int new_channels) {
    int i, j, k, l;

    // 复制第一个数组的元素到结果数组
    for (i = 0; i < batch_size; i++) {
        for (j = 0; j < channels1; j++) {
            for (k = 0; k < height; k++) {
                for (l = 0; l < width; l++) {
                    result[((i * new_channels + j) * height + k) * width + l] = array1[((i * channels1 + j) * height + k) * width + l];
                }
            }
        }
    }

    // 复制第二个数组的元素到结果数组
    for (i = 0; i < batch_size; i++) {
        for (j = 0; j < channels2; j++) {
            for (k = 0; k < height; k++) {
                for (l = 0; l < width; l++) {
                    result[((i * new_channels + channels1 + j) * height + k) * width + l] = array2[((i * channels2 + j) * height + k) * width + l];
                }
            }
        }
    }
}

void reshape(float *input, float *output, int dim1, int dim2, int dim3, int dim4, int new_dim2, int new_dim3) {
    int i, j, k, l;
    for (i = 0; i < dim1; ++i) {
        for (j = 0; j < dim2; ++j) {
            for (k = 0; k < dim3; ++k) {
                for (l = 0; l < dim4; ++l) {
                    int old_index = i * (dim2 * dim3 * dim4) + j * (dim3 * dim4) + k * dim4 + l;
                    int new_index = i * (new_dim2 * new_dim3) + (j * dim3 + k) * dim4 + l;
                    output[new_index] = input[old_index];
                }
            }
        }
    }
}

// 函数来计算多维数组在一维数组中的索引
inline int get_index(int b, int c, int h, int w, int channels, int height, int width) {
    return ((b * channels + c) * height + h) * width + w;
}

// ADD操作函数
void add_arrays(float *array1, float *array2, float *result, int batch, int channels, int height, int width) {
    for (int b = 0; b < batch; ++b) {
        for (int c = 0; c < channels; ++c) {
            float value = array2[get_index(b, c, 0, 0, channels, 1, 1)];
            for (int h = 0; h < height; ++h) {
                for (int w = 0; w < width; ++w) {
                    int idx = get_index(b, c, h, w, channels, height, width);
                    result[idx] = array1[idx] + value;
                }
            }
        }
    }
}

typedef struct {
    int num_train_timesteps;
    float* betas;
    float* alphas;
    float* alphas_cumprod;
    float final_alpha_cumprod;
    float init_noise_sigma;
    int64_t* timesteps;
    int num_inference_steps;
    const char* prediction_type;
    float sigma_data;
} LCMScheduler;

void initialize_scheduler(LCMScheduler* scheduler, int num_train_timesteps, float beta_start, float beta_end, int set_alpha_to_one, const char* prediction_type) {
    scheduler->prediction_type = prediction_type;
    scheduler->num_train_timesteps = num_train_timesteps;

    // 分配内存
    scheduler->betas = (float*)malloc(num_train_timesteps * sizeof(float));
    scheduler->alphas = (float*)malloc(num_train_timesteps * sizeof(float));
    scheduler->alphas_cumprod = (float*)malloc(num_train_timesteps * sizeof(float));
    scheduler->timesteps = (int64_t*)malloc(num_train_timesteps * sizeof(int64_t));

    //求平方根
    float sqrt_s = sqrt(beta_start);
    float sqrt_e = sqrt(beta_end);
    float temp = sqrt_e - sqrt_s;
    // printf("temp = %f\n",  temp);

    // 初始化betas
    for (int i = 0; i < num_train_timesteps; i++) {
        float t = (float)(i + 1) / num_train_timesteps;
        scheduler->betas[i] = powf(sqrt_s + temp * t, 2);
        // printf("t[%d] = %.8f\n", i, sqrt_s + temp * t);
        // printf("betas[%d] = %.10f\n", i, scheduler->betas[i]);
    }

    // 初始化alphas
    for (int i = 0; i < num_train_timesteps; i++) {
        scheduler->alphas[i] = 1.0f - scheduler->betas[i];
    }


    // 初始化alphas_cumprod
    scheduler->alphas_cumprod[0] = scheduler->alphas[0];
    for (int i = 1; i < num_train_timesteps; i++) {
        scheduler->alphas_cumprod[i] = scheduler->alphas_cumprod[i - 1] * scheduler->alphas[i];
    }

    // 设置final_alpha_cumprod
    if (set_alpha_to_one) {
        scheduler->final_alpha_cumprod = 1.0f;
    } else {
        scheduler->final_alpha_cumprod = scheduler->alphas_cumprod[0];
    }

    // 初始化timesteps
    for (int i = 0; i < num_train_timesteps; i++) {
        scheduler->timesteps[i] = num_train_timesteps - 1 - i;
    }

    scheduler->init_noise_sigma = 1.0f;
    scheduler->num_inference_steps = 0;
    scheduler->sigma_data = 0.5; // Default: 0.5
}

void set_timesteps(LCMScheduler* scheduler, int num_inference_steps, int lcm_origin_steps) {
    scheduler->num_inference_steps = num_inference_steps;

    int c = scheduler->num_train_timesteps / lcm_origin_steps;
    int* lcm_origin_timesteps = (int*)malloc(lcm_origin_steps * sizeof(int));
    for (int i = 0; i < lcm_origin_steps; i++) {
        lcm_origin_timesteps[i] = (i + 1) * c - 1;
        // printf("lcm_origin_timesteps[%d] = %d\n", i, lcm_origin_timesteps[i]);
    }

    int skipping_step = lcm_origin_steps / num_inference_steps;
    // printf("skipping_step = %d\n",  skipping_step);
    for (int i = 0; i < num_inference_steps; i++) {
        scheduler->timesteps[i] = lcm_origin_timesteps[lcm_origin_steps - 1 - i * skipping_step];
        // printf("scheduler->timesteps[%d] = %ld\n", i, scheduler->timesteps[i]);
    }

    free(lcm_origin_timesteps);
}

void get_scalings_for_boundary_condition_discrete(float t, float* c_skip, float* c_out, float sigma_data) {
    *c_skip = sigma_data * sigma_data / (powf(t / 0.1f, 2) + sigma_data * sigma_data);
    // printf("c_skip:%.12f\n",sigma_data * sigma_data / (powf(t / 0.1f, 2) + sigma_data * sigma_data));
    *c_out = (t / 0.1f) / sqrtf(powf(t / 0.1f, 2) + sigma_data * sigma_data);
    // printf("c_out:%.14f\n",(t / 0.1f) / sqrtf(powf(t / 0.1f, 2) + sigma_data * sigma_data));
}

float generate_normal_random() {
    float u = ((float) rand() / RAND_MAX);
    float v = ((float) rand() / RAND_MAX);
    return sqrtf(-2.0f * logf(u)) * cosf(2.0f * M_PI * v);
}

void step(LCMScheduler* scheduler, float* model_output, int timeindex, int timestep, float* sample, int model_output_size) {
    int prev_timeindex = timeindex + 1;
    int prev_timestep = (prev_timeindex < scheduler->num_inference_steps) ? scheduler->timesteps[prev_timeindex] : timestep;

    float alpha_prod_t = scheduler->alphas_cumprod[timestep];
    float alpha_prod_t_prev = (prev_timestep >= 0) ? scheduler->alphas_cumprod[prev_timestep] : scheduler->final_alpha_cumprod;

    float beta_prod_t = 1.0f - alpha_prod_t;
    float beta_prod_t_prev = 1.0f - alpha_prod_t_prev;

    float c_skip, c_out;
    get_scalings_for_boundary_condition_discrete((float)timestep, &c_skip, &c_out, scheduler->sigma_data);

    float* pred_x0 = (float*)malloc(model_output_size * sizeof(float));
    for (int i = 0; i < model_output_size; i++) {
        pred_x0[i] = (sample[i] - sqrtf(beta_prod_t) * model_output[i]) / sqrtf(alpha_prod_t);
    }

    float* denoised = (float*)malloc(model_output_size * sizeof(float));
    for (int i = 0; i < model_output_size; i++) {
        denoised[i] = c_out * pred_x0[i] + c_skip * sample[i];
    }

    float* noise = (float*)malloc(model_output_size * sizeof(float));
    for (int i = 0; i < model_output_size; i++) {
        // noise[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        noise[i] = generate_normal_random();
    }

    for (int i = 0; i < model_output_size; i++) {
        sample[i] = sqrtf(alpha_prod_t_prev) * denoised[i] + sqrtf(beta_prod_t_prev) * noise[i];
    }

    free(pred_x0);
    free(denoised);
    free(noise);
}

int main(int argc, char **argv) {
    char **data_path = NULL;
    int input_num = 3;
    int output_num = 2;
    int input_group_num = 1;
    int i;
    float *inputf[input_num];

    int argc_num = 40 + 14;
    if (argc == argc_num && get_file_type(argv[argc_num - 1]) == FILE_TXT) {
        data_path = read_string_from_file(argv[argc_num - 2], &input_group_num);
        input_group_num /= input_num;
    } else if (argc >= ((argc_num - 1) + input_num)) {
        data_path = argv + (argc_num - 1);
        input_group_num = (argc - (argc_num - 1)) / input_num;
    } else {
        printf("Please set valide args: ./model.elf hhb.bm "
                "[data1 data2 ...]|[.txt]\n");
        return -1;
    }

    /******************npu_Subgraphs_0**************************/
    int npu_Subgraphs_0_input_num = 1;
    int npu_Subgraphs_0_output_num = 2;
    int8_t *npu_Subgraphs_0_input[npu_Subgraphs_0_input_num];

    void *sess_npu_Subgraphs_0 = create_Subgraphs(argv[1], csinn_npu_Subgraphs_0);

    int input_size_npu_Subgraphs_0[] = {1 * 4 * 32 * 32, };
    void *input_aligned_npu_Subgraphs_0[1];
    for (i = 0; i < 1; i++) {
        input_size_npu_Subgraphs_0[i] = csinn_tensor_byte_size(((struct csinn_session *)sess_npu_Subgraphs_0)->input[i]);
        input_aligned_npu_Subgraphs_0[i] = shl_mem_alloc_aligned(input_size_npu_Subgraphs_0[i], 0);
    }

    struct csinn_tensor* input_tensors_npu_Subgraphs_0[npu_Subgraphs_0_input_num];
    input_tensors_npu_Subgraphs_0[0] = csinn_alloc_tensor(NULL);
    input_tensors_npu_Subgraphs_0[0]->dim_count = 4;
    input_tensors_npu_Subgraphs_0[0]->dim[0] = 1;
    input_tensors_npu_Subgraphs_0[0]->dim[1] = 4;
    input_tensors_npu_Subgraphs_0[0]->dim[2] = 32;
    input_tensors_npu_Subgraphs_0[0]->dim[3] = 32;

    struct csinn_tensor *output_tensors_npu_Subgraphs_0[npu_Subgraphs_0_output_num];
    output_tensors_npu_Subgraphs_0[0] = csinn_alloc_tensor(NULL);
    output_tensors_npu_Subgraphs_0[0]->dim_count = 3;
    output_tensors_npu_Subgraphs_0[0]->dim[0] = 1;
    output_tensors_npu_Subgraphs_0[0]->dim[1] = 32;
    output_tensors_npu_Subgraphs_0[0]->dim[2] = 10240;

    output_tensors_npu_Subgraphs_0[1] = csinn_alloc_tensor(NULL);
    output_tensors_npu_Subgraphs_0[1]->dim_count = 4;
    output_tensors_npu_Subgraphs_0[1]->dim[0] = 1;
    output_tensors_npu_Subgraphs_0[1]->dim[1] = 320;
    output_tensors_npu_Subgraphs_0[1]->dim[2] = 32;
    output_tensors_npu_Subgraphs_0[1]->dim[3] = 32;

   /******************npu_Subgraphs_1**************************/
    int npu_Subgraphs_1_input_num = 1;
    int npu_Subgraphs_1_output_num = 1;
    float *npu_Subgraphs_1_inputf[npu_Subgraphs_1_input_num];
    int8_t *npu_Subgraphs_1_input[npu_Subgraphs_1_input_num];

    void *sess_npu_Subgraphs_1 = create_Subgraphs(argv[2], csinn_npu_Subgraphs_1);

    int input_size_npu_Subgraphs_1[] = {1 * 32 * 10240, };
    void *input_aligned_npu_Subgraphs_1[1];
    for (i = 0; i < npu_Subgraphs_1_input_num; i++) {
        input_size_npu_Subgraphs_1[i] = csinn_tensor_byte_size(((struct csinn_session *)sess_npu_Subgraphs_1)->input[i]);
        input_aligned_npu_Subgraphs_1[i] = shl_mem_alloc_aligned(input_size_npu_Subgraphs_1[i], 0);
    }

    struct csinn_tensor* input_tensors_npu_Subgraphs_1[npu_Subgraphs_1_input_num];
    input_tensors_npu_Subgraphs_1[0] = csinn_alloc_tensor(NULL);
    input_tensors_npu_Subgraphs_1[0]->dim_count = 3;
    input_tensors_npu_Subgraphs_1[0]->dim[0] = 1;
    input_tensors_npu_Subgraphs_1[0]->dim[1] = 32;
    input_tensors_npu_Subgraphs_1[0]->dim[2] = 10240;

    struct csinn_tensor* output_tensors_npu_Subgraphs_1[npu_Subgraphs_1_output_num];
    output_tensors_npu_Subgraphs_1[0] = csinn_alloc_tensor(NULL);
    output_tensors_npu_Subgraphs_1[0]->dim_count = 4;
    output_tensors_npu_Subgraphs_1[0]->dim[0] = 1;
    output_tensors_npu_Subgraphs_1[0]->dim[1] = 320;
    output_tensors_npu_Subgraphs_1[0]->dim[2] = 32;
    output_tensors_npu_Subgraphs_1[0]->dim[3] = 32;

    /******************cpu_Subgraphs_0**************************/
    int cpu_Subgraphs_0_input_num = 2;
    int cpu_Subgraphs_0_output_num = 2;
    float *cpu_Subgraphs_0_inputf[cpu_Subgraphs_0_input_num];

    void *sess_cpu_Subgraphs_0 = create_Subgraphs(argv[3], csinn_cpu_Subgraphs_0);

    struct csinn_tensor* input_tensors_cpu_Subgraphs_0[cpu_Subgraphs_0_input_num];
    input_tensors_cpu_Subgraphs_0[0] = csinn_alloc_tensor(NULL);
    input_tensors_cpu_Subgraphs_0[0]->dim_count = 4;
    input_tensors_cpu_Subgraphs_0[0]->dim[0] = 1;
    input_tensors_cpu_Subgraphs_0[0]->dim[1] = 320;
    input_tensors_cpu_Subgraphs_0[0]->dim[2] = 32;
    input_tensors_cpu_Subgraphs_0[0]->dim[3] = 32;
    input_tensors_cpu_Subgraphs_0[1] = csinn_alloc_tensor(NULL);
    input_tensors_cpu_Subgraphs_0[1]->dim_count = 1;
    input_tensors_cpu_Subgraphs_0[1]->dim[0] = 1;

    struct csinn_tensor* output_tensors_cpu_Subgraphs_0[cpu_Subgraphs_0_output_num];
    output_tensors_cpu_Subgraphs_0[0] = csinn_alloc_tensor(NULL);
    output_tensors_cpu_Subgraphs_0[0]->dim_count = 3;
    output_tensors_cpu_Subgraphs_0[0]->dim[0] = 1;
    output_tensors_cpu_Subgraphs_0[0]->dim[1] = 32;
    output_tensors_cpu_Subgraphs_0[0]->dim[2] = 10240;
    output_tensors_cpu_Subgraphs_0[1] = csinn_alloc_tensor(NULL);
    output_tensors_cpu_Subgraphs_0[1]->dim_count = 2;
    output_tensors_cpu_Subgraphs_0[1]->dim[0] = 1;
    output_tensors_cpu_Subgraphs_0[1]->dim[1] = 1280;

    /******************npu_Subgraphs_2**************************/
    int npu_Subgraphs_2_input_num = 2;
    int npu_Subgraphs_2_output_num = 2;
    float *npu_Subgraphs_2_inputf[npu_Subgraphs_2_input_num];
    int8_t *npu_Subgraphs_2_input[npu_Subgraphs_2_input_num];

    void *sess_npu_Subgraphs_2 = create_Subgraphs(argv[4], csinn_npu_Subgraphs_2);

    int input_size_npu_Subgraphs_2[] = {1 * 320 * 32 * 32, 1 * 32 * 10240, };
    void *input_aligned_npu_Subgraphs_2[npu_Subgraphs_2_input_num];
    for (i = 0; i < npu_Subgraphs_2_input_num; i++) {
        input_size_npu_Subgraphs_2[i] = csinn_tensor_byte_size(((struct csinn_session *)sess_npu_Subgraphs_2)->input[i]);
        input_aligned_npu_Subgraphs_2[i] = shl_mem_alloc_aligned(input_size_npu_Subgraphs_2[i], 0);
    }
    struct csinn_tensor* input_tensors_npu_Subgraphs_2[npu_Subgraphs_2_input_num];
    input_tensors_npu_Subgraphs_2[0] = csinn_alloc_tensor(NULL);
    input_tensors_npu_Subgraphs_2[0]->dim_count = 4;
    input_tensors_npu_Subgraphs_2[0]->dim[0] = 1;
    input_tensors_npu_Subgraphs_2[0]->dim[1] = 320;
    input_tensors_npu_Subgraphs_2[0]->dim[2] = 32;
    input_tensors_npu_Subgraphs_2[0]->dim[3] = 32;
    input_tensors_npu_Subgraphs_2[1] = csinn_alloc_tensor(NULL);
    input_tensors_npu_Subgraphs_2[1]->dim_count = 3;
    input_tensors_npu_Subgraphs_2[1]->dim[0] = 1;
    input_tensors_npu_Subgraphs_2[1]->dim[1] = 32;
    input_tensors_npu_Subgraphs_2[1]->dim[2] = 10240;
    struct csinn_tensor* output_tensors_npu_Subgraphs_2[npu_Subgraphs_2_output_num];
    output_tensors_npu_Subgraphs_2[0] = csinn_alloc_tensor(NULL);
    output_tensors_npu_Subgraphs_2[0]->dim_count = 4;
    output_tensors_npu_Subgraphs_2[0]->dim[0] = 1;
    output_tensors_npu_Subgraphs_2[0]->dim[1] = 320;
    output_tensors_npu_Subgraphs_2[0]->dim[2] = 32;
    output_tensors_npu_Subgraphs_2[0]->dim[3] = 32;
    output_tensors_npu_Subgraphs_2[1] = csinn_alloc_tensor(NULL);
    output_tensors_npu_Subgraphs_2[1]->dim_count = 3;
    output_tensors_npu_Subgraphs_2[1]->dim[0] = 1;
    output_tensors_npu_Subgraphs_2[1]->dim[1] = 32;
    output_tensors_npu_Subgraphs_2[1]->dim[2] = 10240;

    /******************npu_Subgraphs_3**************************/
    int npu_Subgraphs_3_input_num = 1;
    int npu_Subgraphs_3_output_num = 1;
    float *npu_Subgraphs_3_inputf[npu_Subgraphs_3_input_num];
    int8_t *npu_Subgraphs_3_input[npu_Subgraphs_3_input_num];

    void *sess_npu_Subgraphs_3 = create_Subgraphs(argv[5], csinn_npu_Subgraphs_3);

    int input_size_npu_Subgraphs_3[] = {1 * 32 * 10240, };
    void *input_aligned_npu_Subgraphs_3[npu_Subgraphs_3_input_num];
    for (i = 0; i < npu_Subgraphs_3_input_num; i++) {
        input_size_npu_Subgraphs_3[i] = csinn_tensor_byte_size(((struct csinn_session *)sess_npu_Subgraphs_3)->input[i]);
        input_aligned_npu_Subgraphs_3[i] = shl_mem_alloc_aligned(input_size_npu_Subgraphs_3[i], 0);
    }
    struct csinn_tensor* input_tensors_npu_Subgraphs_3[npu_Subgraphs_3_input_num];
    input_tensors_npu_Subgraphs_3[0] = csinn_alloc_tensor(NULL);
    input_tensors_npu_Subgraphs_3[0]->dim_count = 3;
    input_tensors_npu_Subgraphs_3[0]->dim[0] = 1;
    input_tensors_npu_Subgraphs_3[0]->dim[1] = 32;
    input_tensors_npu_Subgraphs_3[0]->dim[2] = 10240;
    struct csinn_tensor* output_tensors_npu_Subgraphs_3[npu_Subgraphs_3_output_num];
    output_tensors_npu_Subgraphs_3[0] = csinn_alloc_tensor(NULL);
    output_tensors_npu_Subgraphs_3[0]->dim_count = 4;
    output_tensors_npu_Subgraphs_3[0]->dim[0] = 1;
    output_tensors_npu_Subgraphs_3[0]->dim[1] = 320;
    output_tensors_npu_Subgraphs_3[0]->dim[2] = 32;
    output_tensors_npu_Subgraphs_3[0]->dim[3] = 32;

    /******************cpu_Subgraphs_1**************************/
    int cpu_Subgraphs_1_input_num = 2;
    int cpu_Subgraphs_1_output_num = 1;
    float *cpu_Subgraphs_1_inputf[cpu_Subgraphs_1_input_num];

    void *sess_cpu_Subgraphs_1 = create_Subgraphs(argv[6], csinn_cpu_Subgraphs_1);

    struct csinn_tensor* input_tensors_cpu_Subgraphs_1[cpu_Subgraphs_1_input_num];
    input_tensors_cpu_Subgraphs_1[0] = csinn_alloc_tensor(NULL);
    input_tensors_cpu_Subgraphs_1[0]->dim_count = 4;
    input_tensors_cpu_Subgraphs_1[0]->dim[0] = 1;
    input_tensors_cpu_Subgraphs_1[0]->dim[1] = 320;
    input_tensors_cpu_Subgraphs_1[0]->dim[2] = 32;
    input_tensors_cpu_Subgraphs_1[0]->dim[3] = 32;
    input_tensors_cpu_Subgraphs_1[1] = csinn_alloc_tensor(NULL);
    input_tensors_cpu_Subgraphs_1[1]->dim_count = 3;
    input_tensors_cpu_Subgraphs_1[1]->dim[0] = 1;
    input_tensors_cpu_Subgraphs_1[1]->dim[1] = 77;
    input_tensors_cpu_Subgraphs_1[1]->dim[2] = 768;
    struct csinn_tensor* output_tensors_cpu_Subgraphs_1[cpu_Subgraphs_1_output_num];
    output_tensors_cpu_Subgraphs_1[0] = csinn_alloc_tensor(NULL);
    output_tensors_cpu_Subgraphs_1[0]->dim_count = 4;
    output_tensors_cpu_Subgraphs_1[0]->dim[0] = 1;
    output_tensors_cpu_Subgraphs_1[0]->dim[1] = 320;
    output_tensors_cpu_Subgraphs_1[0]->dim[2] = 32;
    output_tensors_cpu_Subgraphs_1[0]->dim[3] = 32;

    /******************npu_Subgraphs_4**************************/
    int npu_Subgraphs_4_input_num = 2;
    int npu_Subgraphs_4_output_num = 4;
    float *npu_Subgraphs_4_inputf[npu_Subgraphs_4_input_num];
    int8_t *npu_Subgraphs_4_input[npu_Subgraphs_4_input_num];

    void *sess_npu_Subgraphs_4 = create_Subgraphs(argv[7], csinn_npu_Subgraphs_4);

    int input_size_npu_Subgraphs_4[] = {1 * 320 * 32 * 32, 1 * 320 * 32 * 32, };
    void *input_aligned_npu_Subgraphs_4[npu_Subgraphs_4_input_num];
    for (i = 0; i < npu_Subgraphs_4_input_num; i++) {
        input_size_npu_Subgraphs_4[i] = csinn_tensor_byte_size(((struct csinn_session *)sess_npu_Subgraphs_4)->input[i]);
        input_aligned_npu_Subgraphs_4[i] = shl_mem_alloc_aligned(input_size_npu_Subgraphs_4[i], 0);
    }
    struct csinn_tensor* input_tensors_npu_Subgraphs_4[npu_Subgraphs_4_input_num];
    //Transpose_1_output_0
    input_tensors_npu_Subgraphs_4[0] = csinn_alloc_tensor(NULL);
    input_tensors_npu_Subgraphs_4[0]->dim_count = 4;
    input_tensors_npu_Subgraphs_4[0]->dim[0] = 1;
    input_tensors_npu_Subgraphs_4[0]->dim[1] = 320;
    input_tensors_npu_Subgraphs_4[0]->dim[2] = 32;
    input_tensors_npu_Subgraphs_4[0]->dim[3] = 32;
    //Add_1_output_0
    input_tensors_npu_Subgraphs_4[1] = csinn_alloc_tensor(NULL);
    input_tensors_npu_Subgraphs_4[1]->dim_count = 4;
    input_tensors_npu_Subgraphs_4[1]->dim[0] = 1;
    input_tensors_npu_Subgraphs_4[1]->dim[1] = 320;
    input_tensors_npu_Subgraphs_4[1]->dim[2] = 32;
    input_tensors_npu_Subgraphs_4[1]->dim[3] = 32;
    struct csinn_tensor* output_tensors_npu_Subgraphs_4[npu_Subgraphs_4_output_num];
    output_tensors_npu_Subgraphs_4[0] = csinn_alloc_tensor(NULL);
    output_tensors_npu_Subgraphs_4[0]->dim_count = 4;
    output_tensors_npu_Subgraphs_4[0]->dim[0] = 1;
    output_tensors_npu_Subgraphs_4[0]->dim[1] = 320;
    output_tensors_npu_Subgraphs_4[0]->dim[2] = 32;
    output_tensors_npu_Subgraphs_4[0]->dim[3] = 32;
    output_tensors_npu_Subgraphs_4[1] = csinn_alloc_tensor(NULL);
    output_tensors_npu_Subgraphs_4[1]->dim_count = 4;
    output_tensors_npu_Subgraphs_4[1]->dim[0] = 1;
    output_tensors_npu_Subgraphs_4[1]->dim[1] = 320;
    output_tensors_npu_Subgraphs_4[1]->dim[2] = 16;
    output_tensors_npu_Subgraphs_4[1]->dim[3] = 16;
    output_tensors_npu_Subgraphs_4[2] = csinn_alloc_tensor(NULL);
    output_tensors_npu_Subgraphs_4[2]->dim_count = 3;
    output_tensors_npu_Subgraphs_4[2]->dim[0] = 1;
    output_tensors_npu_Subgraphs_4[2]->dim[1] = 32;
    output_tensors_npu_Subgraphs_4[2]->dim[2] = 2560;
    output_tensors_npu_Subgraphs_4[3] = csinn_alloc_tensor(NULL);
    output_tensors_npu_Subgraphs_4[3]->dim_count = 4;
    output_tensors_npu_Subgraphs_4[3]->dim[0] = 1;
    output_tensors_npu_Subgraphs_4[3]->dim[1] = 640;
    output_tensors_npu_Subgraphs_4[3]->dim[2] = 16;
    output_tensors_npu_Subgraphs_4[3]->dim[3] = 16;

    /******************npu_Subgraphs_5**************************/
    int npu_Subgraphs_5_input_num = 2;
    int npu_Subgraphs_5_output_num = 1;
    float *npu_Subgraphs_5_inputf[npu_Subgraphs_5_input_num];
    int8_t *npu_Subgraphs_5_input[npu_Subgraphs_5_input_num];

    void *sess_npu_Subgraphs_5 = create_Subgraphs(argv[8], csinn_npu_Subgraphs_5);

    int input_size_npu_Subgraphs_5[] = {1 * 32 * 2560, 1 * 1280, };
    void *input_aligned_npu_Subgraphs_5[npu_Subgraphs_5_input_num];
    for (i = 0; i < npu_Subgraphs_5_input_num; i++) {
        input_size_npu_Subgraphs_5[i] = csinn_tensor_byte_size(((struct csinn_session *)sess_npu_Subgraphs_5)->input[i]);
        input_aligned_npu_Subgraphs_5[i] = shl_mem_alloc_aligned(input_size_npu_Subgraphs_5[i], 0);
    }
    struct csinn_tensor* input_tensors_npu_Subgraphs_5[npu_Subgraphs_5_input_num];
    input_tensors_npu_Subgraphs_5[0] = csinn_alloc_tensor(NULL);
    input_tensors_npu_Subgraphs_5[0]->dim_count = 3;
    input_tensors_npu_Subgraphs_5[0]->dim[0] = 1;
    input_tensors_npu_Subgraphs_5[0]->dim[1] = 32;
    input_tensors_npu_Subgraphs_5[0]->dim[2] = 2560;
    input_tensors_npu_Subgraphs_5[1] = csinn_alloc_tensor(NULL);
    input_tensors_npu_Subgraphs_5[1]->dim_count = 2;
    input_tensors_npu_Subgraphs_5[1]->dim[0] = 1;
    input_tensors_npu_Subgraphs_5[1]->dim[1] = 1280;
    struct csinn_tensor* output_tensors_npu_Subgraphs_5[npu_Subgraphs_5_output_num];
    output_tensors_npu_Subgraphs_5[0] = csinn_alloc_tensor(NULL);
    output_tensors_npu_Subgraphs_5[0]->dim_count = 3;
    output_tensors_npu_Subgraphs_5[0]->dim[0] = 1;
    output_tensors_npu_Subgraphs_5[0]->dim[1] = 32;
    output_tensors_npu_Subgraphs_5[0]->dim[2] = 5120;

    /******************npu_Subgraphs_6**************************/
    int npu_Subgraphs_6_input_num = 2;
    int npu_Subgraphs_6_output_num = 2;
    float *npu_Subgraphs_6_inputf[npu_Subgraphs_6_input_num];
    int8_t *npu_Subgraphs_6_input[npu_Subgraphs_6_input_num];

    void *sess_npu_Subgraphs_6 = create_Subgraphs(argv[9], csinn_npu_Subgraphs_6);

    int input_size_npu_Subgraphs_6[] = {1 * 640 * 16 * 16, 1 * 32 * 5120, };
    void *input_aligned_npu_Subgraphs_6[npu_Subgraphs_6_input_num];
    for (i = 0; i < npu_Subgraphs_6_input_num; i++) {
        input_size_npu_Subgraphs_6[i] = csinn_tensor_byte_size(((struct csinn_session *)sess_npu_Subgraphs_6)->input[i]);
        input_aligned_npu_Subgraphs_6[i] = shl_mem_alloc_aligned(input_size_npu_Subgraphs_6[i], 0);
    }
    struct csinn_tensor* input_tensors_npu_Subgraphs_6[npu_Subgraphs_6_input_num];
    input_tensors_npu_Subgraphs_6[0] = csinn_alloc_tensor(NULL);
    input_tensors_npu_Subgraphs_6[0]->dim_count = 4;
    input_tensors_npu_Subgraphs_6[0]->dim[0] = 1;
    input_tensors_npu_Subgraphs_6[0]->dim[1] = 640;
    input_tensors_npu_Subgraphs_6[0]->dim[2] = 16;
    input_tensors_npu_Subgraphs_6[0]->dim[3] = 16;
    input_tensors_npu_Subgraphs_6[1] = csinn_alloc_tensor(NULL);
    input_tensors_npu_Subgraphs_6[1]->dim_count = 3;
    input_tensors_npu_Subgraphs_6[1]->dim[0] = 1;
    input_tensors_npu_Subgraphs_6[1]->dim[1] = 32;
    input_tensors_npu_Subgraphs_6[1]->dim[2] = 5120;
    struct csinn_tensor* output_tensors_npu_Subgraphs_6[npu_Subgraphs_6_output_num];
    output_tensors_npu_Subgraphs_6[0] = csinn_alloc_tensor(NULL);
    output_tensors_npu_Subgraphs_6[0]->dim_count = 3;
    output_tensors_npu_Subgraphs_6[0]->dim[0] = 1;
    output_tensors_npu_Subgraphs_6[0]->dim[1] = 32;
    output_tensors_npu_Subgraphs_6[0]->dim[2] = 5120;
    output_tensors_npu_Subgraphs_6[1] = csinn_alloc_tensor(NULL);
    output_tensors_npu_Subgraphs_6[1]->dim_count = 4;
    output_tensors_npu_Subgraphs_6[1]->dim[0] = 1;
    output_tensors_npu_Subgraphs_6[1]->dim[1] = 640;
    output_tensors_npu_Subgraphs_6[1]->dim[2] = 16;
    output_tensors_npu_Subgraphs_6[1]->dim[3] = 16;

    /******************npu_Subgraphs_7**************************/
    int npu_Subgraphs_7_input_num = 1;
    int npu_Subgraphs_7_output_num = 1;
    float *npu_Subgraphs_7_inputf[npu_Subgraphs_7_input_num];
    int8_t *npu_Subgraphs_7_input[npu_Subgraphs_7_input_num];

    void *sess_npu_Subgraphs_7 = create_Subgraphs(argv[10], csinn_npu_Subgraphs_7);

    int input_size_npu_Subgraphs_7[] = {1 * 32 * 5120, };
    void *input_aligned_npu_Subgraphs_7[npu_Subgraphs_7_input_num];
    for (i = 0; i < npu_Subgraphs_7_input_num; i++) {
        input_size_npu_Subgraphs_7[i] = csinn_tensor_byte_size(((struct csinn_session *)sess_npu_Subgraphs_7)->input[i]);
        input_aligned_npu_Subgraphs_7[i] = shl_mem_alloc_aligned(input_size_npu_Subgraphs_7[i], 0);
    }
    struct csinn_tensor* input_tensors_npu_Subgraphs_7[npu_Subgraphs_7_input_num];
    input_tensors_npu_Subgraphs_7[0] = csinn_alloc_tensor(NULL);
    input_tensors_npu_Subgraphs_7[0]->dim_count = 3;
    input_tensors_npu_Subgraphs_7[0]->dim[0] = 1;
    input_tensors_npu_Subgraphs_7[0]->dim[1] = 32;
    input_tensors_npu_Subgraphs_7[0]->dim[2] = 5120;
    struct csinn_tensor* output_tensors_npu_Subgraphs_7[npu_Subgraphs_7_output_num];
    output_tensors_npu_Subgraphs_7[0] = csinn_alloc_tensor(NULL);
    output_tensors_npu_Subgraphs_7[0]->dim_count = 4;
    output_tensors_npu_Subgraphs_7[0]->dim[0] = 1;
    output_tensors_npu_Subgraphs_7[0]->dim[1] = 640;
    output_tensors_npu_Subgraphs_7[0]->dim[2] = 16;
    output_tensors_npu_Subgraphs_7[0]->dim[3] = 16;

    /******************cpu_Subgraphs_2**************************/
    int cpu_Subgraphs_2_input_num = 2;
    int cpu_Subgraphs_2_output_num = 1;
    float *cpu_Subgraphs_2_inputf[cpu_Subgraphs_2_input_num];

    void *sess_cpu_Subgraphs_2 = create_Subgraphs(argv[11], csinn_cpu_Subgraphs_2);

    struct csinn_tensor* input_tensors_cpu_Subgraphs_2[cpu_Subgraphs_2_input_num];
    input_tensors_cpu_Subgraphs_2[0] = csinn_alloc_tensor(NULL);
    input_tensors_cpu_Subgraphs_2[0]->dim_count = 4;
    input_tensors_cpu_Subgraphs_2[0]->dim[0] = 1;
    input_tensors_cpu_Subgraphs_2[0]->dim[1] = 640;
    input_tensors_cpu_Subgraphs_2[0]->dim[2] = 16;
    input_tensors_cpu_Subgraphs_2[0]->dim[3] = 16;
    input_tensors_cpu_Subgraphs_2[1] = csinn_alloc_tensor(NULL);
    input_tensors_cpu_Subgraphs_2[1]->dim_count = 3;
    input_tensors_cpu_Subgraphs_2[1]->dim[0] = 1;
    input_tensors_cpu_Subgraphs_2[1]->dim[1] = 77;
    input_tensors_cpu_Subgraphs_2[1]->dim[2] = 768;
    struct csinn_tensor* output_tensors_cpu_Subgraphs_2[cpu_Subgraphs_2_output_num];
    output_tensors_cpu_Subgraphs_2[0] = csinn_alloc_tensor(NULL);
    output_tensors_cpu_Subgraphs_2[0]->dim_count = 4;
    output_tensors_cpu_Subgraphs_2[0]->dim[0] = 1;
    output_tensors_cpu_Subgraphs_2[0]->dim[1] = 640;
    output_tensors_cpu_Subgraphs_2[0]->dim[2] = 16;
    output_tensors_cpu_Subgraphs_2[0]->dim[3] = 16;

    /******************npu_Subgraphs_8**************************/
    int npu_Subgraphs_8_input_num = 2;
    int npu_Subgraphs_8_output_num = 3;
    float *npu_Subgraphs_8_inputf[npu_Subgraphs_8_input_num];
    int8_t *npu_Subgraphs_8_input[npu_Subgraphs_8_input_num];

    void *sess_npu_Subgraphs_8 = create_Subgraphs(argv[12], csinn_npu_Subgraphs_8);

    int input_size_npu_Subgraphs_8[] = {1 * 640 * 16 * 16, 1 * 640 * 16 * 16, };
    void *input_aligned_npu_Subgraphs_8[npu_Subgraphs_8_input_num];
    for (i = 0; i < npu_Subgraphs_8_input_num; i++) {
        input_size_npu_Subgraphs_8[i] = csinn_tensor_byte_size(((struct csinn_session *)sess_npu_Subgraphs_8)->input[i]);
        input_aligned_npu_Subgraphs_8[i] = shl_mem_alloc_aligned(input_size_npu_Subgraphs_8[i], 0);
    }
    //Transpose_1_output_0
    struct csinn_tensor* input_tensors_npu_Subgraphs_8[npu_Subgraphs_8_input_num];
    input_tensors_npu_Subgraphs_8[0] = csinn_alloc_tensor(NULL);
    input_tensors_npu_Subgraphs_8[0]->dim_count = 4;
    input_tensors_npu_Subgraphs_8[0]->dim[0] = 1;
    input_tensors_npu_Subgraphs_8[0]->dim[1] = 640;
    input_tensors_npu_Subgraphs_8[0]->dim[2] = 16;
    input_tensors_npu_Subgraphs_8[0]->dim[3] = 16;
    input_tensors_npu_Subgraphs_8[1] = csinn_alloc_tensor(NULL);
    input_tensors_npu_Subgraphs_8[1]->dim_count = 4;
    input_tensors_npu_Subgraphs_8[1]->dim[0] = 1;
    input_tensors_npu_Subgraphs_8[1]->dim[1] = 640;
    input_tensors_npu_Subgraphs_8[1]->dim[2] = 16;
    input_tensors_npu_Subgraphs_8[1]->dim[3] = 16;
    //0_Add_1_output_0
    struct csinn_tensor* output_tensors_npu_Subgraphs_8[npu_Subgraphs_8_output_num];
    output_tensors_npu_Subgraphs_8[0] = csinn_alloc_tensor(NULL);
    output_tensors_npu_Subgraphs_8[0]->dim_count = 4;
    output_tensors_npu_Subgraphs_8[0]->dim[0] = 1;
    output_tensors_npu_Subgraphs_8[0]->dim[1] = 640;
    output_tensors_npu_Subgraphs_8[0]->dim[2] = 16;
    output_tensors_npu_Subgraphs_8[0]->dim[3] = 16;
    output_tensors_npu_Subgraphs_8[1] = csinn_alloc_tensor(NULL);
    output_tensors_npu_Subgraphs_8[1]->dim_count = 4;
    output_tensors_npu_Subgraphs_8[1]->dim[0] = 1;
    output_tensors_npu_Subgraphs_8[1]->dim[1] = 640;
    output_tensors_npu_Subgraphs_8[1]->dim[2] = 8;
    output_tensors_npu_Subgraphs_8[1]->dim[3] = 8;
    output_tensors_npu_Subgraphs_8[2] = csinn_alloc_tensor(NULL);
    output_tensors_npu_Subgraphs_8[2]->dim_count = 3;
    output_tensors_npu_Subgraphs_8[2]->dim[0] = 1;
    output_tensors_npu_Subgraphs_8[2]->dim[1] = 32;
    output_tensors_npu_Subgraphs_8[2]->dim[2] = 1280;

    /******************npu_Subgraphs_9**************************/
    int npu_Subgraphs_9_input_num = 2;
    int npu_Subgraphs_9_output_num = 1;
    float *npu_Subgraphs_9_inputf[npu_Subgraphs_9_input_num];
    int8_t *npu_Subgraphs_9_input[npu_Subgraphs_9_input_num];

    void *sess_npu_Subgraphs_9 = create_Subgraphs(argv[13], csinn_npu_Subgraphs_9);

    int input_size_npu_Subgraphs_9[] = {1 * 32 * 1280, 1 * 1280, };
    void *input_aligned_npu_Subgraphs_9[npu_Subgraphs_9_input_num];
    for (i = 0; i < npu_Subgraphs_9_input_num; i++) {
        input_size_npu_Subgraphs_9[i] = csinn_tensor_byte_size(((struct csinn_session *)sess_npu_Subgraphs_9)->input[i]);
        input_aligned_npu_Subgraphs_9[i] = shl_mem_alloc_aligned(input_size_npu_Subgraphs_9[i], 0);
    }
    struct csinn_tensor* input_tensors_npu_Subgraphs_9[npu_Subgraphs_9_input_num];
    input_tensors_npu_Subgraphs_9[0] = csinn_alloc_tensor(NULL);
    input_tensors_npu_Subgraphs_9[0]->dim_count = 3;
    input_tensors_npu_Subgraphs_9[0]->dim[0] = 1;
    input_tensors_npu_Subgraphs_9[0]->dim[1] = 32;
    input_tensors_npu_Subgraphs_9[0]->dim[2] = 1280;
    input_tensors_npu_Subgraphs_9[1] = csinn_alloc_tensor(NULL);
    input_tensors_npu_Subgraphs_9[1]->dim_count = 2;
    input_tensors_npu_Subgraphs_9[1]->dim[0] = 1;
    input_tensors_npu_Subgraphs_9[1]->dim[1] = 1280;
    struct csinn_tensor* output_tensors_npu_Subgraphs_9[npu_Subgraphs_9_output_num];
    output_tensors_npu_Subgraphs_9[0] = csinn_alloc_tensor(NULL);
    output_tensors_npu_Subgraphs_9[0]->dim_count = 3;
    output_tensors_npu_Subgraphs_9[0]->dim[0] = 1;
    output_tensors_npu_Subgraphs_9[0]->dim[1] = 32;
    output_tensors_npu_Subgraphs_9[0]->dim[2] = 2560;

    /******************npu_Subgraphs_10**************************/
    int npu_Subgraphs_10_input_num = 2;
    int npu_Subgraphs_10_output_num = 2;
    float *npu_Subgraphs_10_inputf[npu_Subgraphs_10_input_num];
    int8_t *npu_Subgraphs_10_input[npu_Subgraphs_10_input_num];

    void *sess_npu_Subgraphs_10 = create_Subgraphs(argv[14], csinn_npu_Subgraphs_10);

    int input_size_npu_Subgraphs_10[] = {1 * 640 * 8 * 8, 1 * 32 * 2560, };
    void *input_aligned_npu_Subgraphs_10[npu_Subgraphs_10_input_num];
    for (i = 0; i < npu_Subgraphs_10_input_num; i++) {
        input_size_npu_Subgraphs_10[i] = csinn_tensor_byte_size(((struct csinn_session *)sess_npu_Subgraphs_10)->input[i]);
        input_aligned_npu_Subgraphs_10[i] = shl_mem_alloc_aligned(input_size_npu_Subgraphs_10[i], 0);
    }
    struct csinn_tensor* input_tensors_npu_Subgraphs_10[npu_Subgraphs_10_input_num];
    input_tensors_npu_Subgraphs_10[0] = csinn_alloc_tensor(NULL);
    input_tensors_npu_Subgraphs_10[0]->dim_count = 4;
    input_tensors_npu_Subgraphs_10[0]->dim[0] = 1;
    input_tensors_npu_Subgraphs_10[0]->dim[1] = 640;
    input_tensors_npu_Subgraphs_10[0]->dim[2] = 8;
    input_tensors_npu_Subgraphs_10[0]->dim[3] = 8;
    input_tensors_npu_Subgraphs_10[1] = csinn_alloc_tensor(NULL);
    input_tensors_npu_Subgraphs_10[1]->dim_count = 3;
    input_tensors_npu_Subgraphs_10[1]->dim[0] = 1;
    input_tensors_npu_Subgraphs_10[1]->dim[1] = 32;
    input_tensors_npu_Subgraphs_10[1]->dim[2] = 2560;
    struct csinn_tensor* output_tensors_npu_Subgraphs_10[npu_Subgraphs_10_output_num];
    output_tensors_npu_Subgraphs_10[0] = csinn_alloc_tensor(NULL);
    output_tensors_npu_Subgraphs_10[0]->dim_count = 3;
    output_tensors_npu_Subgraphs_10[0]->dim[0] = 1;
    output_tensors_npu_Subgraphs_10[0]->dim[1] = 32;
    output_tensors_npu_Subgraphs_10[0]->dim[2] = 2560;
    output_tensors_npu_Subgraphs_10[1] = csinn_alloc_tensor(NULL);
    output_tensors_npu_Subgraphs_10[1]->dim_count = 4;
    output_tensors_npu_Subgraphs_10[1]->dim[0] = 1;
    output_tensors_npu_Subgraphs_10[1]->dim[1] = 1280;
    output_tensors_npu_Subgraphs_10[1]->dim[2] = 8;
    output_tensors_npu_Subgraphs_10[1]->dim[3] = 8;

    /******************npu_Subgraphs_11**************************/
    int npu_Subgraphs_11_input_num = 1;
    int npu_Subgraphs_11_output_num = 1;
    float *npu_Subgraphs_11_inputf[npu_Subgraphs_11_input_num];
    int8_t *npu_Subgraphs_11_input[npu_Subgraphs_11_input_num];

    void *sess_npu_Subgraphs_11 = create_Subgraphs(argv[15], csinn_npu_Subgraphs_11);

    int input_size_npu_Subgraphs_11[] = {1 * 32 * 2560, };
    void *input_aligned_npu_Subgraphs_11[npu_Subgraphs_11_input_num];
    for (i = 0; i < npu_Subgraphs_11_input_num; i++) {
        input_size_npu_Subgraphs_11[i] = csinn_tensor_byte_size(((struct csinn_session *)sess_npu_Subgraphs_11)->input[i]);
        input_aligned_npu_Subgraphs_11[i] = shl_mem_alloc_aligned(input_size_npu_Subgraphs_11[i], 0);
    }
    struct csinn_tensor* input_tensors_npu_Subgraphs_11[npu_Subgraphs_11_input_num];
    input_tensors_npu_Subgraphs_11[0] = csinn_alloc_tensor(NULL);
    input_tensors_npu_Subgraphs_11[0]->dim_count = 3;
    input_tensors_npu_Subgraphs_11[0]->dim[0] = 1;
    input_tensors_npu_Subgraphs_11[0]->dim[1] = 32;
    input_tensors_npu_Subgraphs_11[0]->dim[2] = 2560;
    struct csinn_tensor* output_tensors_npu_Subgraphs_11[npu_Subgraphs_11_output_num];
    output_tensors_npu_Subgraphs_11[0] = csinn_alloc_tensor(NULL);
    output_tensors_npu_Subgraphs_11[0]->dim_count = 4;
    output_tensors_npu_Subgraphs_11[0]->dim[0] = 1;
    output_tensors_npu_Subgraphs_11[0]->dim[1] = 1280;
    output_tensors_npu_Subgraphs_11[0]->dim[2] = 8;
    output_tensors_npu_Subgraphs_11[0]->dim[3] = 8;

    /******************cpu_Subgraphs_3**************************/
    int cpu_Subgraphs_3_input_num = 2;
    int cpu_Subgraphs_3_output_num = 1;
    float *cpu_Subgraphs_3_inputf[cpu_Subgraphs_3_input_num];

    void *sess_cpu_Subgraphs_3 = create_Subgraphs(argv[16], csinn_cpu_Subgraphs_3);

    struct csinn_tensor* input_tensors_cpu_Subgraphs_3[cpu_Subgraphs_3_input_num];
    input_tensors_cpu_Subgraphs_3[0] = csinn_alloc_tensor(NULL);
    input_tensors_cpu_Subgraphs_3[0]->dim_count = 4;
    input_tensors_cpu_Subgraphs_3[0]->dim[0] = 1;
    input_tensors_cpu_Subgraphs_3[0]->dim[1] = 1280;
    input_tensors_cpu_Subgraphs_3[0]->dim[2] = 8;
    input_tensors_cpu_Subgraphs_3[0]->dim[3] = 8;
    input_tensors_cpu_Subgraphs_3[1] = csinn_alloc_tensor(NULL);
    input_tensors_cpu_Subgraphs_3[1]->dim_count = 3;
    input_tensors_cpu_Subgraphs_3[1]->dim[0] = 1;
    input_tensors_cpu_Subgraphs_3[1]->dim[1] = 77;
    input_tensors_cpu_Subgraphs_3[1]->dim[2] = 768;
    struct csinn_tensor* output_tensors_cpu_Subgraphs_3[cpu_Subgraphs_3_output_num];
    output_tensors_cpu_Subgraphs_3[0] = csinn_alloc_tensor(NULL);
    output_tensors_cpu_Subgraphs_3[0]->dim_count = 4;
    output_tensors_cpu_Subgraphs_3[0]->dim[0] = 1;
    output_tensors_cpu_Subgraphs_3[0]->dim[1] = 1280;
    output_tensors_cpu_Subgraphs_3[0]->dim[2] = 8;
    output_tensors_cpu_Subgraphs_3[0]->dim[3] = 8;

    /******************npu_Subgraphs_12**************************/
    int npu_Subgraphs_12_input_num = 2;
    int npu_Subgraphs_12_output_num = 2;
    float *npu_Subgraphs_12_inputf[npu_Subgraphs_12_input_num];
    int8_t *npu_Subgraphs_12_input[npu_Subgraphs_12_input_num];

    void *sess_npu_Subgraphs_12 = create_Subgraphs(argv[17], csinn_npu_Subgraphs_12);

    int input_size_npu_Subgraphs_12[] = {1 * 1280 * 8 * 8, 1 * 1280 * 8 * 8, };
    void *input_aligned_npu_Subgraphs_12[npu_Subgraphs_12_input_num];
    for (i = 0; i < npu_Subgraphs_12_input_num; i++) {
        input_size_npu_Subgraphs_12[i] = csinn_tensor_byte_size(((struct csinn_session *)sess_npu_Subgraphs_12)->input[i]);
        input_aligned_npu_Subgraphs_12[i] = shl_mem_alloc_aligned(input_size_npu_Subgraphs_12[i], 0);
    }
    struct csinn_tensor* input_tensors_npu_Subgraphs_12[npu_Subgraphs_12_input_num];
    //ranspose_1_output_0
    input_tensors_npu_Subgraphs_12[0] = csinn_alloc_tensor(NULL);
    input_tensors_npu_Subgraphs_12[0]->dim_count = 4;
    input_tensors_npu_Subgraphs_12[0]->dim[0] = 1;
    input_tensors_npu_Subgraphs_12[0]->dim[1] = 1280;
    input_tensors_npu_Subgraphs_12[0]->dim[2] = 8;
    input_tensors_npu_Subgraphs_12[0]->dim[3] = 8;
    //Add_1_output_0
    input_tensors_npu_Subgraphs_12[1] = csinn_alloc_tensor(NULL);
    input_tensors_npu_Subgraphs_12[1]->dim_count = 4;
    input_tensors_npu_Subgraphs_12[1]->dim[0] = 1;
    input_tensors_npu_Subgraphs_12[1]->dim[1] = 1280;
    input_tensors_npu_Subgraphs_12[1]->dim[2] = 8;
    input_tensors_npu_Subgraphs_12[1]->dim[3] = 8;
    struct csinn_tensor* output_tensors_npu_Subgraphs_12[npu_Subgraphs_12_output_num];
    output_tensors_npu_Subgraphs_12[0] = csinn_alloc_tensor(NULL);
    output_tensors_npu_Subgraphs_12[0]->dim_count = 3;
    output_tensors_npu_Subgraphs_12[0]->dim[0] = 1;
    output_tensors_npu_Subgraphs_12[0]->dim[1] = 32;
    output_tensors_npu_Subgraphs_12[0]->dim[2] = 5120;
    output_tensors_npu_Subgraphs_12[1] = csinn_alloc_tensor(NULL);
    output_tensors_npu_Subgraphs_12[1]->dim_count = 4;
    output_tensors_npu_Subgraphs_12[1]->dim[0] = 1;
    output_tensors_npu_Subgraphs_12[1]->dim[1] = 1280;
    output_tensors_npu_Subgraphs_12[1]->dim[2] = 8;
    output_tensors_npu_Subgraphs_12[1]->dim[3] = 8;

    /******************npu_Subgraphs_13**************************/
    int npu_Subgraphs_13_input_num = 2;
    int npu_Subgraphs_13_output_num = 1;
    float *npu_Subgraphs_13_inputf[npu_Subgraphs_13_input_num];
    int8_t *npu_Subgraphs_13_input[npu_Subgraphs_13_input_num];

    void *sess_npu_Subgraphs_13 = create_Subgraphs(argv[18], csinn_npu_Subgraphs_13);

    int input_size_npu_Subgraphs_13[] = {1 * 32 * 5120, 1 * 1280, };
    void *input_aligned_npu_Subgraphs_13[npu_Subgraphs_13_input_num];
    for (i = 0; i < npu_Subgraphs_13_input_num; i++) {
        input_size_npu_Subgraphs_13[i] = csinn_tensor_byte_size(((struct csinn_session *)sess_npu_Subgraphs_13)->input[i]);
        input_aligned_npu_Subgraphs_13[i] = shl_mem_alloc_aligned(input_size_npu_Subgraphs_13[i], 0);
    }
    struct csinn_tensor* input_tensors_npu_Subgraphs_13[npu_Subgraphs_13_input_num];
    input_tensors_npu_Subgraphs_13[0] = csinn_alloc_tensor(NULL);
    input_tensors_npu_Subgraphs_13[0]->dim_count = 3;
    input_tensors_npu_Subgraphs_13[0]->dim[0] = 1;
    input_tensors_npu_Subgraphs_13[0]->dim[1] = 32;
    input_tensors_npu_Subgraphs_13[0]->dim[2] = 5120;
    input_tensors_npu_Subgraphs_13[1] = csinn_alloc_tensor(NULL);
    input_tensors_npu_Subgraphs_13[1]->dim_count = 2;
    input_tensors_npu_Subgraphs_13[1]->dim[0] = 1;
    input_tensors_npu_Subgraphs_13[1]->dim[1] = 1280;
    struct csinn_tensor* output_tensors_npu_Subgraphs_13[npu_Subgraphs_13_output_num];
    output_tensors_npu_Subgraphs_13[0] = csinn_alloc_tensor(NULL);
    output_tensors_npu_Subgraphs_13[0]->dim_count = 3;
    output_tensors_npu_Subgraphs_13[0]->dim[0] = 1;
    output_tensors_npu_Subgraphs_13[0]->dim[1] = 32;
    output_tensors_npu_Subgraphs_13[0]->dim[2] = 2560;

    /******************npu_Subgraphs_14**************************/
    int npu_Subgraphs_14_input_num = 2;
    int npu_Subgraphs_14_output_num = 2;
    float *npu_Subgraphs_14_inputf[npu_Subgraphs_14_input_num];
    int8_t *npu_Subgraphs_14_input[npu_Subgraphs_14_input_num];

    void *sess_npu_Subgraphs_14 = create_Subgraphs(argv[19], csinn_npu_Subgraphs_14);

    int input_size_npu_Subgraphs_14[] = {1 * 1280 * 8 * 8, 1 * 32 * 2560, };
    void *input_aligned_npu_Subgraphs_14[npu_Subgraphs_14_input_num];
    for (i = 0; i < npu_Subgraphs_14_input_num; i++) {
        input_size_npu_Subgraphs_14[i] = csinn_tensor_byte_size(((struct csinn_session *)sess_npu_Subgraphs_14)->input[i]);
        input_aligned_npu_Subgraphs_14[i] = shl_mem_alloc_aligned(input_size_npu_Subgraphs_14[i], 0);
    }
    struct csinn_tensor* input_tensors_npu_Subgraphs_14[npu_Subgraphs_14_input_num];
    input_tensors_npu_Subgraphs_14[0] = csinn_alloc_tensor(NULL);
    input_tensors_npu_Subgraphs_14[0]->dim_count = 4;
    input_tensors_npu_Subgraphs_14[0]->dim[0] = 1;
    input_tensors_npu_Subgraphs_14[0]->dim[1] = 1280;
    input_tensors_npu_Subgraphs_14[0]->dim[2] = 8;
    input_tensors_npu_Subgraphs_14[0]->dim[3] = 8;
    input_tensors_npu_Subgraphs_14[1] = csinn_alloc_tensor(NULL);
    input_tensors_npu_Subgraphs_14[1]->dim_count = 3;
    input_tensors_npu_Subgraphs_14[1]->dim[0] = 1;
    input_tensors_npu_Subgraphs_14[1]->dim[1] = 32;
    input_tensors_npu_Subgraphs_14[1]->dim[2] = 2560;
    struct csinn_tensor* output_tensors_npu_Subgraphs_14[npu_Subgraphs_14_output_num];
    output_tensors_npu_Subgraphs_14[0] = csinn_alloc_tensor(NULL);
    output_tensors_npu_Subgraphs_14[0]->dim_count = 3;
    output_tensors_npu_Subgraphs_14[0]->dim[0] = 1;
    output_tensors_npu_Subgraphs_14[0]->dim[1] = 32;
    output_tensors_npu_Subgraphs_14[0]->dim[2] = 2560;
    output_tensors_npu_Subgraphs_14[1] = csinn_alloc_tensor(NULL);
    output_tensors_npu_Subgraphs_14[1]->dim_count = 4;
    output_tensors_npu_Subgraphs_14[1]->dim[0] = 1;
    output_tensors_npu_Subgraphs_14[1]->dim[1] = 1280;
    output_tensors_npu_Subgraphs_14[1]->dim[2] = 8;
    output_tensors_npu_Subgraphs_14[1]->dim[3] = 8;

    /******************npu_Subgraphs_15**************************/
    int npu_Subgraphs_15_input_num = 1;
    int npu_Subgraphs_15_output_num = 1;
    float *npu_Subgraphs_15_inputf[npu_Subgraphs_15_input_num];
    int8_t *npu_Subgraphs_15_input[npu_Subgraphs_15_input_num];

    void *sess_npu_Subgraphs_15 = create_Subgraphs(argv[20], csinn_npu_Subgraphs_15);

    int input_size_npu_Subgraphs_15[] = {1 * 32 * 2560, };
    void *input_aligned_npu_Subgraphs_15[npu_Subgraphs_15_input_num];
    for (i = 0; i < npu_Subgraphs_15_input_num; i++) {
        input_size_npu_Subgraphs_15[i] = csinn_tensor_byte_size(((struct csinn_session *)sess_npu_Subgraphs_15)->input[i]);
        input_aligned_npu_Subgraphs_15[i] = shl_mem_alloc_aligned(input_size_npu_Subgraphs_15[i], 0);
    }
    struct csinn_tensor* input_tensors_npu_Subgraphs_15[npu_Subgraphs_15_input_num];
    input_tensors_npu_Subgraphs_15[0] = csinn_alloc_tensor(NULL);
    input_tensors_npu_Subgraphs_15[0]->dim_count = 3;
    input_tensors_npu_Subgraphs_15[0]->dim[0] = 1;
    input_tensors_npu_Subgraphs_15[0]->dim[1] = 32;
    input_tensors_npu_Subgraphs_15[0]->dim[2] = 2560;
    struct csinn_tensor* output_tensors_npu_Subgraphs_15[npu_Subgraphs_15_output_num];
    output_tensors_npu_Subgraphs_15[0] = csinn_alloc_tensor(NULL);
    output_tensors_npu_Subgraphs_15[0]->dim_count = 4;
    output_tensors_npu_Subgraphs_15[0]->dim[0] = 1;
    output_tensors_npu_Subgraphs_15[0]->dim[1] = 1280;
    output_tensors_npu_Subgraphs_15[0]->dim[2] = 8;
    output_tensors_npu_Subgraphs_15[0]->dim[3] = 8;

    /******************cpu_Subgraphs_4**************************/
    int cpu_Subgraphs_4_input_num = 2;
    int cpu_Subgraphs_4_output_num = 1;
    float *cpu_Subgraphs_4_inputf[cpu_Subgraphs_4_input_num];

    void *sess_cpu_Subgraphs_4 = create_Subgraphs(argv[21], csinn_cpu_Subgraphs_4);

    struct csinn_tensor* input_tensors_cpu_Subgraphs_4[cpu_Subgraphs_4_input_num];
    input_tensors_cpu_Subgraphs_4[0] = csinn_alloc_tensor(NULL);
    input_tensors_cpu_Subgraphs_4[0]->dim_count = 4;
    input_tensors_cpu_Subgraphs_4[0]->dim[0] = 1;
    input_tensors_cpu_Subgraphs_4[0]->dim[1] = 1280;
    input_tensors_cpu_Subgraphs_4[0]->dim[2] = 8;
    input_tensors_cpu_Subgraphs_4[0]->dim[3] = 8;
    input_tensors_cpu_Subgraphs_4[1] = csinn_alloc_tensor(NULL);
    input_tensors_cpu_Subgraphs_4[1]->dim_count = 3;
    input_tensors_cpu_Subgraphs_4[1]->dim[0] = 1;
    input_tensors_cpu_Subgraphs_4[1]->dim[1] = 77;
    input_tensors_cpu_Subgraphs_4[1]->dim[2] = 768;
    struct csinn_tensor* output_tensors_cpu_Subgraphs_4[cpu_Subgraphs_4_output_num];
    output_tensors_cpu_Subgraphs_4[0] = csinn_alloc_tensor(NULL);
    output_tensors_cpu_Subgraphs_4[0]->dim_count = 4;
    output_tensors_cpu_Subgraphs_4[0]->dim[0] = 1;
    output_tensors_cpu_Subgraphs_4[0]->dim[1] = 1280;
    output_tensors_cpu_Subgraphs_4[0]->dim[2] = 8;
    output_tensors_cpu_Subgraphs_4[0]->dim[3] = 8;

    /******************npu_Subgraphs_16**************************/
    int npu_Subgraphs_16_input_num = 2;
    int npu_Subgraphs_16_output_num = 1;
    float *npu_Subgraphs_16_inputf[npu_Subgraphs_16_input_num];
    int8_t *npu_Subgraphs_16_input[npu_Subgraphs_16_input_num];

    void *sess_npu_Subgraphs_16 = create_Subgraphs(argv[22], csinn_npu_Subgraphs_16);

    int input_size_npu_Subgraphs_16[] = {1 * 1280 * 8 * 8, 1 * 1280 * 8 * 8, };
    void *input_aligned_npu_Subgraphs_16[npu_Subgraphs_16_input_num];
    for (i = 0; i < npu_Subgraphs_16_input_num; i++) {
        input_size_npu_Subgraphs_16[i] = csinn_tensor_byte_size(((struct csinn_session *)sess_npu_Subgraphs_16)->input[i]);
        input_aligned_npu_Subgraphs_16[i] = shl_mem_alloc_aligned(input_size_npu_Subgraphs_16[i], 0);
    }
    struct csinn_tensor* input_tensors_npu_Subgraphs_16[npu_Subgraphs_16_input_num];
    input_tensors_npu_Subgraphs_16[0] = csinn_alloc_tensor(NULL);
    input_tensors_npu_Subgraphs_16[0]->dim_count = 4;
    input_tensors_npu_Subgraphs_16[0]->dim[0] = 1;
    input_tensors_npu_Subgraphs_16[0]->dim[1] = 1280;
    input_tensors_npu_Subgraphs_16[0]->dim[2] = 8;
    input_tensors_npu_Subgraphs_16[0]->dim[3] = 8;
    input_tensors_npu_Subgraphs_16[1] = csinn_alloc_tensor(NULL);
    input_tensors_npu_Subgraphs_16[1]->dim_count = 4;
    input_tensors_npu_Subgraphs_16[1]->dim[0] = 1;
    input_tensors_npu_Subgraphs_16[1]->dim[1] = 1280;
    input_tensors_npu_Subgraphs_16[1]->dim[2] = 8;
    input_tensors_npu_Subgraphs_16[1]->dim[3] = 8;
    struct csinn_tensor* output_tensors_npu_Subgraphs_16[npu_Subgraphs_16_output_num];
    output_tensors_npu_Subgraphs_16[0] = csinn_alloc_tensor(NULL);
    output_tensors_npu_Subgraphs_16[0]->dim_count = 4;
    output_tensors_npu_Subgraphs_16[0]->dim[0] = 1;
    output_tensors_npu_Subgraphs_16[0]->dim[1] = 1280;
    output_tensors_npu_Subgraphs_16[0]->dim[2] = 8;
    output_tensors_npu_Subgraphs_16[0]->dim[3] = 8;

    /******************npu_Subgraphs_17**************************/
    int npu_Subgraphs_17_input_num = 2;
    int npu_Subgraphs_17_output_num = 1;
    float *npu_Subgraphs_17_inputf[npu_Subgraphs_17_input_num];
    int8_t *npu_Subgraphs_17_input[npu_Subgraphs_17_input_num];

    void *sess_npu_Subgraphs_17 = create_Subgraphs(argv[23], csinn_npu_Subgraphs_17);

    int input_size_npu_Subgraphs_17[] = {1 * 32 * 3840, 1 * 1280, };
    void *input_aligned_npu_Subgraphs_17[npu_Subgraphs_17_input_num];
    for (i = 0; i < npu_Subgraphs_17_input_num; i++) {
        input_size_npu_Subgraphs_17[i] = csinn_tensor_byte_size(((struct csinn_session *)sess_npu_Subgraphs_17)->input[i]);
        input_aligned_npu_Subgraphs_17[i] = shl_mem_alloc_aligned(input_size_npu_Subgraphs_17[i], 0);
    }
    struct csinn_tensor* input_tensors_npu_Subgraphs_17[npu_Subgraphs_17_input_num];
    input_tensors_npu_Subgraphs_17[0] = csinn_alloc_tensor(NULL);
    input_tensors_npu_Subgraphs_17[0]->dim_count = 3;
    input_tensors_npu_Subgraphs_17[0]->dim[0] = 1;
    input_tensors_npu_Subgraphs_17[0]->dim[1] = 32;
    input_tensors_npu_Subgraphs_17[0]->dim[2] = 3840;
    input_tensors_npu_Subgraphs_17[1] = csinn_alloc_tensor(NULL);
    input_tensors_npu_Subgraphs_17[1]->dim_count = 2;
    input_tensors_npu_Subgraphs_17[1]->dim[0] = 1;
    input_tensors_npu_Subgraphs_17[1]->dim[1] = 1280;
    struct csinn_tensor* output_tensors_npu_Subgraphs_17[npu_Subgraphs_17_output_num];
    output_tensors_npu_Subgraphs_17[0] = csinn_alloc_tensor(NULL);
    output_tensors_npu_Subgraphs_17[0]->dim_count = 3;
    output_tensors_npu_Subgraphs_17[0]->dim[0] = 1;
    output_tensors_npu_Subgraphs_17[0]->dim[1] = 32;
    output_tensors_npu_Subgraphs_17[0]->dim[2] = 2560;

    /******************npu_Subgraphs_18**************************/
    int npu_Subgraphs_18_input_num = 2;
    int npu_Subgraphs_18_output_num = 1;
    float *npu_Subgraphs_18_inputf[npu_Subgraphs_18_input_num];
    int8_t *npu_Subgraphs_18_input[npu_Subgraphs_18_input_num];

    void *sess_npu_Subgraphs_18 = create_Subgraphs(argv[24], csinn_npu_Subgraphs_18);

    int input_size_npu_Subgraphs_18[] = {1 * 1920 * 8 * 8, 1 * 32 * 2560, };
    void *input_aligned_npu_Subgraphs_18[npu_Subgraphs_18_input_num];
    for (i = 0; i < npu_Subgraphs_18_input_num; i++) {
        input_size_npu_Subgraphs_18[i] = csinn_tensor_byte_size(((struct csinn_session *)sess_npu_Subgraphs_18)->input[i]);
        input_aligned_npu_Subgraphs_18[i] = shl_mem_alloc_aligned(input_size_npu_Subgraphs_18[i], 0);
    }
    struct csinn_tensor* input_tensors_npu_Subgraphs_18[npu_Subgraphs_18_input_num];
    input_tensors_npu_Subgraphs_18[0] = csinn_alloc_tensor(NULL);
    input_tensors_npu_Subgraphs_18[0]->dim_count = 4;
    input_tensors_npu_Subgraphs_18[0]->dim[0] = 1;
    input_tensors_npu_Subgraphs_18[0]->dim[1] = 1920;
    input_tensors_npu_Subgraphs_18[0]->dim[2] = 8;
    input_tensors_npu_Subgraphs_18[0]->dim[3] = 8;
    input_tensors_npu_Subgraphs_18[1] = csinn_alloc_tensor(NULL);
    input_tensors_npu_Subgraphs_18[1]->dim_count = 3;
    input_tensors_npu_Subgraphs_18[1]->dim[0] = 1;
    input_tensors_npu_Subgraphs_18[1]->dim[1] = 32;
    input_tensors_npu_Subgraphs_18[1]->dim[2] = 2560;
    struct csinn_tensor* output_tensors_npu_Subgraphs_18[npu_Subgraphs_18_output_num];
    output_tensors_npu_Subgraphs_18[0] = csinn_alloc_tensor(NULL);
    output_tensors_npu_Subgraphs_18[0]->dim_count = 3;
    output_tensors_npu_Subgraphs_18[0]->dim[0] = 1;
    output_tensors_npu_Subgraphs_18[0]->dim[1] = 32;
    output_tensors_npu_Subgraphs_18[0]->dim[2] = 2560;
    output_tensors_npu_Subgraphs_18[1] = csinn_alloc_tensor(NULL);
    output_tensors_npu_Subgraphs_18[1]->dim_count = 4;
    output_tensors_npu_Subgraphs_18[1]->dim[0] = 1;
    output_tensors_npu_Subgraphs_18[1]->dim[1] = 1280;
    output_tensors_npu_Subgraphs_18[1]->dim[2] = 8;
    output_tensors_npu_Subgraphs_18[1]->dim[3] = 8;

    /******************npu_Subgraphs_19**************************/
    int npu_Subgraphs_19_input_num = 1;
    int npu_Subgraphs_19_output_num = 1;
    float *npu_Subgraphs_19_inputf[npu_Subgraphs_19_input_num];
    int8_t *npu_Subgraphs_19_input[npu_Subgraphs_19_input_num];

    void *sess_npu_Subgraphs_19 = create_Subgraphs(argv[25], csinn_npu_Subgraphs_19);

    int input_size_npu_Subgraphs_19[] = {1 * 32 * 2560, };
    void *input_aligned_npu_Subgraphs_19[npu_Subgraphs_19_input_num];
    for (i = 0; i < npu_Subgraphs_19_input_num; i++) {
        input_size_npu_Subgraphs_19[i] = csinn_tensor_byte_size(((struct csinn_session *)sess_npu_Subgraphs_19)->input[i]);
        input_aligned_npu_Subgraphs_19[i] = shl_mem_alloc_aligned(input_size_npu_Subgraphs_19[i], 0);
    }
    struct csinn_tensor* input_tensors_npu_Subgraphs_19[npu_Subgraphs_19_input_num];
    input_tensors_npu_Subgraphs_19[0] = csinn_alloc_tensor(NULL);
    input_tensors_npu_Subgraphs_19[0]->dim_count = 3;
    input_tensors_npu_Subgraphs_19[0]->dim[0] = 1;
    input_tensors_npu_Subgraphs_19[0]->dim[1] = 32;
    input_tensors_npu_Subgraphs_19[0]->dim[2] = 2560;
    struct csinn_tensor* output_tensors_npu_Subgraphs_19[npu_Subgraphs_19_output_num];
    output_tensors_npu_Subgraphs_19[0] = csinn_alloc_tensor(NULL);
    output_tensors_npu_Subgraphs_19[0]->dim_count = 4;
    output_tensors_npu_Subgraphs_19[0]->dim[0] = 1;
    output_tensors_npu_Subgraphs_19[0]->dim[1] = 1280;
    output_tensors_npu_Subgraphs_19[0]->dim[2] = 8;
    output_tensors_npu_Subgraphs_19[0]->dim[3] = 8;

    /******************cpu_Subgraphs_5**************************/
    int cpu_Subgraphs_5_input_num = 2;
    int cpu_Subgraphs_5_output_num = 1;
    float *cpu_Subgraphs_5_inputf[cpu_Subgraphs_5_input_num];

    void *sess_cpu_Subgraphs_5 = create_Subgraphs(argv[26], csinn_cpu_Subgraphs_5);

    struct csinn_tensor* input_tensors_cpu_Subgraphs_5[cpu_Subgraphs_5_input_num];
    input_tensors_cpu_Subgraphs_5[0] = csinn_alloc_tensor(NULL);
    input_tensors_cpu_Subgraphs_5[0]->dim_count = 4;
    input_tensors_cpu_Subgraphs_5[0]->dim[0] = 1;
    input_tensors_cpu_Subgraphs_5[0]->dim[1] = 1280;
    input_tensors_cpu_Subgraphs_5[0]->dim[2] = 8;
    input_tensors_cpu_Subgraphs_5[0]->dim[3] = 8;
    input_tensors_cpu_Subgraphs_5[1] = csinn_alloc_tensor(NULL);
    input_tensors_cpu_Subgraphs_5[1]->dim_count = 3;
    input_tensors_cpu_Subgraphs_5[1]->dim[0] = 1;
    input_tensors_cpu_Subgraphs_5[1]->dim[1] = 77;
    input_tensors_cpu_Subgraphs_5[1]->dim[2] = 768;
    struct csinn_tensor* output_tensors_cpu_Subgraphs_5[cpu_Subgraphs_5_output_num];
    output_tensors_cpu_Subgraphs_5[0] = csinn_alloc_tensor(NULL);
    output_tensors_cpu_Subgraphs_5[0]->dim_count = 4;
    output_tensors_cpu_Subgraphs_5[0]->dim[0] = 1;
    output_tensors_cpu_Subgraphs_5[0]->dim[1] = 1280;
    output_tensors_cpu_Subgraphs_5[0]->dim[2] = 8;
    output_tensors_cpu_Subgraphs_5[0]->dim[3] = 8;

    /******************npu_Subgraphs_20**************************/
    int npu_Subgraphs_20_input_num = 2;
    int npu_Subgraphs_20_output_num = 1;
    float *npu_Subgraphs_20_inputf[npu_Subgraphs_20_input_num];
    int8_t *npu_Subgraphs_20_input[npu_Subgraphs_20_input_num];

    void *sess_npu_Subgraphs_20 = create_Subgraphs(argv[27], csinn_npu_Subgraphs_20);

    int input_size_npu_Subgraphs_20[] = {1 * 1280 * 8 * 8, 1 * 1280 * 8 * 8, };
    void *input_aligned_npu_Subgraphs_20[npu_Subgraphs_20_input_num];
    for (i = 0; i < npu_Subgraphs_20_input_num; i++) {
        input_size_npu_Subgraphs_20[i] = csinn_tensor_byte_size(((struct csinn_session *)sess_npu_Subgraphs_20)->input[i]);
        input_aligned_npu_Subgraphs_20[i] = shl_mem_alloc_aligned(input_size_npu_Subgraphs_20[i], 0);
    }
    struct csinn_tensor* input_tensors_npu_Subgraphs_20[npu_Subgraphs_20_input_num];
    input_tensors_npu_Subgraphs_20[0] = csinn_alloc_tensor(NULL);
    input_tensors_npu_Subgraphs_20[0]->dim_count = 4;
    input_tensors_npu_Subgraphs_20[0]->dim[0] = 1;
    input_tensors_npu_Subgraphs_20[0]->dim[1] = 1280;
    input_tensors_npu_Subgraphs_20[0]->dim[2] = 8;
    input_tensors_npu_Subgraphs_20[0]->dim[3] = 8;
    input_tensors_npu_Subgraphs_20[1] = csinn_alloc_tensor(NULL);
    input_tensors_npu_Subgraphs_20[1]->dim_count = 4;
    input_tensors_npu_Subgraphs_20[1]->dim[0] = 1;
    input_tensors_npu_Subgraphs_20[1]->dim[1] = 1280;
    input_tensors_npu_Subgraphs_20[1]->dim[2] = 8;
    input_tensors_npu_Subgraphs_20[1]->dim[3] = 8;
    struct csinn_tensor* output_tensors_npu_Subgraphs_20[npu_Subgraphs_20_output_num];
    output_tensors_npu_Subgraphs_20[0] = csinn_alloc_tensor(NULL);
    output_tensors_npu_Subgraphs_20[0]->dim_count = 4;
    output_tensors_npu_Subgraphs_20[0]->dim[0] = 1;
    output_tensors_npu_Subgraphs_20[0]->dim[1] = 1280;
    output_tensors_npu_Subgraphs_20[0]->dim[2] = 16;
    output_tensors_npu_Subgraphs_20[0]->dim[3] = 16;

    /******************npu_Subgraphs_21**************************/
    int npu_Subgraphs_21_input_num = 1;
    int npu_Subgraphs_21_output_num = 1;
    float *npu_Subgraphs_21_inputf[npu_Subgraphs_21_input_num];
    int8_t *npu_Subgraphs_21_input[npu_Subgraphs_21_input_num];

    void *sess_npu_Subgraphs_21 = create_Subgraphs(argv[28], csinn_npu_Subgraphs_21);

    int input_size_npu_Subgraphs_21[] = {1 * 1280, };
    void *input_aligned_npu_Subgraphs_21[npu_Subgraphs_21_input_num];
    for (i = 0; i < npu_Subgraphs_21_input_num; i++) {
        input_size_npu_Subgraphs_21[i] = csinn_tensor_byte_size(((struct csinn_session *)sess_npu_Subgraphs_21)->input[i]);
        input_aligned_npu_Subgraphs_21[i] = shl_mem_alloc_aligned(input_size_npu_Subgraphs_21[i], 0);
    }
    struct csinn_tensor* input_tensors_npu_Subgraphs_21[npu_Subgraphs_21_input_num];
    input_tensors_npu_Subgraphs_21[0] = csinn_alloc_tensor(NULL);
    input_tensors_npu_Subgraphs_21[0]->dim_count = 2;
    input_tensors_npu_Subgraphs_21[0]->dim[0] = 1;
    input_tensors_npu_Subgraphs_21[0]->dim[1] = 1280;
    struct csinn_tensor* output_tensors_npu_Subgraphs_21[npu_Subgraphs_21_output_num];
    output_tensors_npu_Subgraphs_21[0] = csinn_alloc_tensor(NULL);
    output_tensors_npu_Subgraphs_21[0]->dim_count = 4;
    output_tensors_npu_Subgraphs_21[0]->dim[0] = 1;
    output_tensors_npu_Subgraphs_21[0]->dim[1] = 640;
    output_tensors_npu_Subgraphs_21[0]->dim[2] = 1;
    output_tensors_npu_Subgraphs_21[0]->dim[3] = 1;

    /******************npu_Subgraphs_22**************************/
    int npu_Subgraphs_22_input_num = 1;
    int npu_Subgraphs_22_output_num = 1;
    float *npu_Subgraphs_22_inputf[npu_Subgraphs_22_input_num];
    int8_t *npu_Subgraphs_22_input[npu_Subgraphs_22_input_num];

    void *sess_npu_Subgraphs_22 = create_Subgraphs(argv[29], csinn_npu_Subgraphs_22);

    int input_size_npu_Subgraphs_22[] = {1 * 32 * 15360, };
    void *input_aligned_npu_Subgraphs_22[npu_Subgraphs_22_input_num];
    for (i = 0; i < npu_Subgraphs_22_input_num; i++) {
        input_size_npu_Subgraphs_22[i] = csinn_tensor_byte_size(((struct csinn_session *)sess_npu_Subgraphs_22)->input[i]);
        input_aligned_npu_Subgraphs_22[i] = shl_mem_alloc_aligned(input_size_npu_Subgraphs_22[i], 0);
    }
    struct csinn_tensor* input_tensors_npu_Subgraphs_22[npu_Subgraphs_22_input_num];
    input_tensors_npu_Subgraphs_22[0] = csinn_alloc_tensor(NULL);
    input_tensors_npu_Subgraphs_22[0]->dim_count = 3;
    input_tensors_npu_Subgraphs_22[0]->dim[0] = 1;
    input_tensors_npu_Subgraphs_22[0]->dim[1] = 32;
    input_tensors_npu_Subgraphs_22[0]->dim[2] = 15360;
    struct csinn_tensor* output_tensors_npu_Subgraphs_22[npu_Subgraphs_22_output_num];
    output_tensors_npu_Subgraphs_22[0] = csinn_alloc_tensor(NULL);
    output_tensors_npu_Subgraphs_22[0]->dim_count = 3;
    output_tensors_npu_Subgraphs_22[0]->dim[0] = 1;
    output_tensors_npu_Subgraphs_22[0]->dim[1] = 640;
    output_tensors_npu_Subgraphs_22[0]->dim[2] = 16;
    output_tensors_npu_Subgraphs_22[0]->dim[3] = 16;

    /******************npu_Subgraphs_23**************************/
    int npu_Subgraphs_23_input_num = 2;
    int npu_Subgraphs_23_output_num = 2;
    float *npu_Subgraphs_23_inputf[npu_Subgraphs_23_input_num];
    int8_t *npu_Subgraphs_23_input[npu_Subgraphs_23_input_num];

    void *sess_npu_Subgraphs_23 = create_Subgraphs(argv[30], csinn_npu_Subgraphs_23);

    int input_size_npu_Subgraphs_23[] = {1 * 1920 * 16 * 16, 1 * 32 * 5120, };
    void *input_aligned_npu_Subgraphs_23[npu_Subgraphs_23_input_num];
    for (i = 0; i < npu_Subgraphs_23_input_num; i++) {
        input_size_npu_Subgraphs_23[i] = csinn_tensor_byte_size(((struct csinn_session *)sess_npu_Subgraphs_23)->input[i]);
        input_aligned_npu_Subgraphs_23[i] = shl_mem_alloc_aligned(input_size_npu_Subgraphs_23[i], 0);
    }
    struct csinn_tensor* input_tensors_npu_Subgraphs_23[npu_Subgraphs_23_input_num];
    input_tensors_npu_Subgraphs_23[0] = csinn_alloc_tensor(NULL);
    input_tensors_npu_Subgraphs_23[0]->dim_count = 4;
    input_tensors_npu_Subgraphs_23[0]->dim[0] = 1;
    input_tensors_npu_Subgraphs_23[0]->dim[1] = 1920;
    input_tensors_npu_Subgraphs_23[0]->dim[2] = 16;
    input_tensors_npu_Subgraphs_23[0]->dim[3] = 16;
    input_tensors_npu_Subgraphs_23[1] = csinn_alloc_tensor(NULL);
    input_tensors_npu_Subgraphs_23[1]->dim_count = 3;
    input_tensors_npu_Subgraphs_23[1]->dim[0] = 1;
    input_tensors_npu_Subgraphs_23[1]->dim[1] = 32;
    input_tensors_npu_Subgraphs_23[1]->dim[2] = 5120;
    struct csinn_tensor* output_tensors_npu_Subgraphs_23[npu_Subgraphs_23_output_num];
    output_tensors_npu_Subgraphs_23[0] = csinn_alloc_tensor(NULL);
    output_tensors_npu_Subgraphs_23[0]->dim_count = 3;
    output_tensors_npu_Subgraphs_23[0]->dim[0] = 1;
    output_tensors_npu_Subgraphs_23[0]->dim[1] = 32;
    output_tensors_npu_Subgraphs_23[0]->dim[2] = 5120;
    output_tensors_npu_Subgraphs_23[1] = csinn_alloc_tensor(NULL);
    output_tensors_npu_Subgraphs_23[1]->dim_count = 4;
    output_tensors_npu_Subgraphs_23[1]->dim[0] = 1;
    output_tensors_npu_Subgraphs_23[1]->dim[1] = 640;
    output_tensors_npu_Subgraphs_23[1]->dim[2] = 16;
    output_tensors_npu_Subgraphs_23[1]->dim[3] = 16;

    /******************npu_Subgraphs_24**************************/
    int npu_Subgraphs_24_input_num = 1;
    int npu_Subgraphs_24_output_num = 1;
    float *npu_Subgraphs_24_inputf[npu_Subgraphs_24_input_num];
    int8_t *npu_Subgraphs_24_input[npu_Subgraphs_24_input_num];

    void *sess_npu_Subgraphs_24 = create_Subgraphs(argv[31], csinn_npu_Subgraphs_24);

    int input_size_npu_Subgraphs_24[] = {1 * 32 * 5120, };
    void *input_aligned_npu_Subgraphs_24[npu_Subgraphs_24_input_num];
    for (i = 0; i < npu_Subgraphs_24_input_num; i++) {
        input_size_npu_Subgraphs_24[i] = csinn_tensor_byte_size(((struct csinn_session *)sess_npu_Subgraphs_24)->input[i]);
        input_aligned_npu_Subgraphs_24[i] = shl_mem_alloc_aligned(input_size_npu_Subgraphs_24[i], 0);
    }
    struct csinn_tensor* input_tensors_npu_Subgraphs_24[npu_Subgraphs_24_input_num];
    input_tensors_npu_Subgraphs_24[0] = csinn_alloc_tensor(NULL);
    input_tensors_npu_Subgraphs_24[0]->dim_count = 3;
    input_tensors_npu_Subgraphs_24[0]->dim[0] = 1;
    input_tensors_npu_Subgraphs_24[0]->dim[1] = 32;
    input_tensors_npu_Subgraphs_24[0]->dim[2] = 5120;
    struct csinn_tensor* output_tensors_npu_Subgraphs_24[npu_Subgraphs_24_output_num];
    output_tensors_npu_Subgraphs_24[0] = csinn_alloc_tensor(NULL);
    output_tensors_npu_Subgraphs_24[0]->dim_count = 4;
    output_tensors_npu_Subgraphs_24[0]->dim[0] = 1;
    output_tensors_npu_Subgraphs_24[0]->dim[1] = 640;
    output_tensors_npu_Subgraphs_24[0]->dim[2] = 16;
    output_tensors_npu_Subgraphs_24[0]->dim[3] = 16;


    /******************cpu_Subgraphs_6**************************/
    int cpu_Subgraphs_6_input_num = 2;
    int cpu_Subgraphs_6_output_num = 1;
    float *cpu_Subgraphs_6_inputf[cpu_Subgraphs_6_input_num];

    void *sess_cpu_Subgraphs_6 = create_Subgraphs(argv[32], csinn_cpu_Subgraphs_6);

    struct csinn_tensor* input_tensors_cpu_Subgraphs_6[cpu_Subgraphs_6_input_num];
    input_tensors_cpu_Subgraphs_6[0] = csinn_alloc_tensor(NULL);
    input_tensors_cpu_Subgraphs_6[0]->dim_count = 4;
    input_tensors_cpu_Subgraphs_6[0]->dim[0] = 1;
    input_tensors_cpu_Subgraphs_6[0]->dim[1] = 640;
    input_tensors_cpu_Subgraphs_6[0]->dim[2] = 16;
    input_tensors_cpu_Subgraphs_6[0]->dim[3] = 16;
    input_tensors_cpu_Subgraphs_6[1] = csinn_alloc_tensor(NULL);
    input_tensors_cpu_Subgraphs_6[1]->dim_count = 3;
    input_tensors_cpu_Subgraphs_6[1]->dim[0] = 1;
    input_tensors_cpu_Subgraphs_6[1]->dim[1] = 77;
    input_tensors_cpu_Subgraphs_6[1]->dim[2] = 768;
    struct csinn_tensor* output_tensors_cpu_Subgraphs_6[cpu_Subgraphs_6_output_num];
    output_tensors_cpu_Subgraphs_6[0] = csinn_alloc_tensor(NULL);
    output_tensors_cpu_Subgraphs_6[0]->dim_count = 4;
    output_tensors_cpu_Subgraphs_6[0]->dim[0] = 1;
    output_tensors_cpu_Subgraphs_6[0]->dim[1] = 640;
    output_tensors_cpu_Subgraphs_6[0]->dim[2] = 16;
    output_tensors_cpu_Subgraphs_6[0]->dim[3] = 16;


    /******************npu_Subgraphs_25**************************/
    int npu_Subgraphs_25_input_num = 2;
    int npu_Subgraphs_25_output_num = 1;
    float *npu_Subgraphs_25_inputf[npu_Subgraphs_25_input_num];
    int8_t *npu_Subgraphs_25_input[npu_Subgraphs_25_input_num];

    void *sess_npu_Subgraphs_25 = create_Subgraphs(argv[33], csinn_npu_Subgraphs_25);

    int input_size_npu_Subgraphs_25[] = {1 * 640 * 16 * 16, 1 * 640 * 16 * 16, };
    void *input_aligned_npu_Subgraphs_25[npu_Subgraphs_25_input_num];
    for (i = 0; i < npu_Subgraphs_25_input_num; i++) {
        input_size_npu_Subgraphs_25[i] = csinn_tensor_byte_size(((struct csinn_session *)sess_npu_Subgraphs_25)->input[i]);
        input_aligned_npu_Subgraphs_25[i] = shl_mem_alloc_aligned(input_size_npu_Subgraphs_25[i], 0);
    }
    struct csinn_tensor* input_tensors_npu_Subgraphs_25[npu_Subgraphs_25_input_num];
    input_tensors_npu_Subgraphs_25[0] = csinn_alloc_tensor(NULL);
    input_tensors_npu_Subgraphs_25[0]->dim_count = 4;
    input_tensors_npu_Subgraphs_25[0]->dim[0] = 1;
    input_tensors_npu_Subgraphs_25[0]->dim[1] = 640;
    input_tensors_npu_Subgraphs_25[0]->dim[2] = 16;
    input_tensors_npu_Subgraphs_25[0]->dim[3] = 16;
    input_tensors_npu_Subgraphs_25[1] = csinn_alloc_tensor(NULL);
    input_tensors_npu_Subgraphs_25[1]->dim_count = 4;
    input_tensors_npu_Subgraphs_25[1]->dim[0] = 1;
    input_tensors_npu_Subgraphs_25[1]->dim[1] = 640;
    input_tensors_npu_Subgraphs_25[1]->dim[2] = 16;
    input_tensors_npu_Subgraphs_25[1]->dim[3] = 16;
    struct csinn_tensor* output_tensors_npu_Subgraphs_25[npu_Subgraphs_25_output_num];
    output_tensors_npu_Subgraphs_25[0] = csinn_alloc_tensor(NULL);
    output_tensors_npu_Subgraphs_25[0]->dim_count = 4;
    output_tensors_npu_Subgraphs_25[0]->dim[0] = 1;
    output_tensors_npu_Subgraphs_25[0]->dim[1] = 640;
    output_tensors_npu_Subgraphs_25[0]->dim[2] = 16;
    output_tensors_npu_Subgraphs_25[0]->dim[3] = 16;

    /******************npu_Subgraphs_26**************************/
    int npu_Subgraphs_26_input_num = 1;
    int npu_Subgraphs_26_output_num = 1;
    float *npu_Subgraphs_26_inputf[npu_Subgraphs_26_input_num];
    int8_t *npu_Subgraphs_26_input[npu_Subgraphs_26_input_num];

    void *sess_npu_Subgraphs_26 = create_Subgraphs(argv[34], csinn_npu_Subgraphs_26);

    int input_size_npu_Subgraphs_26[] = {1 * 1280, };
    void *input_aligned_npu_Subgraphs_26[npu_Subgraphs_26_input_num];
    for (i = 0; i < npu_Subgraphs_26_input_num; i++) {
        input_size_npu_Subgraphs_26[i] = csinn_tensor_byte_size(((struct csinn_session *)sess_npu_Subgraphs_26)->input[i]);
        input_aligned_npu_Subgraphs_26[i] = shl_mem_alloc_aligned(input_size_npu_Subgraphs_26[i], 0);
    }
    struct csinn_tensor* input_tensors_npu_Subgraphs_26[npu_Subgraphs_26_input_num];
    input_tensors_npu_Subgraphs_26[0] = csinn_alloc_tensor(NULL);
    input_tensors_npu_Subgraphs_26[0]->dim_count = 2;
    input_tensors_npu_Subgraphs_26[0]->dim[0] = 1;
    input_tensors_npu_Subgraphs_26[0]->dim[1] = 1280;
    struct csinn_tensor* output_tensors_npu_Subgraphs_26[npu_Subgraphs_26_output_num];
    output_tensors_npu_Subgraphs_26[0] = csinn_alloc_tensor(NULL);
    output_tensors_npu_Subgraphs_26[0]->dim_count = 4;
    output_tensors_npu_Subgraphs_26[0]->dim[0] = 1;
    output_tensors_npu_Subgraphs_26[0]->dim[1] = 640;
    output_tensors_npu_Subgraphs_26[0]->dim[2] = 1;
    output_tensors_npu_Subgraphs_26[0]->dim[3] = 1;

    /******************npu_Subgraphs_27**************************/
    int npu_Subgraphs_27_input_num = 1;
    int npu_Subgraphs_27_output_num = 1;
    float *npu_Subgraphs_27_inputf[npu_Subgraphs_27_input_num];
    int8_t *npu_Subgraphs_27_input[npu_Subgraphs_27_input_num];

    void *sess_npu_Subgraphs_27 = create_Subgraphs(argv[35], csinn_npu_Subgraphs_27);

    int input_size_npu_Subgraphs_27[] = {1 * 32 * 7680, };
    void *input_aligned_npu_Subgraphs_27[npu_Subgraphs_27_input_num];
    for (i = 0; i < npu_Subgraphs_27_input_num; i++) {
        input_size_npu_Subgraphs_27[i] = csinn_tensor_byte_size(((struct csinn_session *)sess_npu_Subgraphs_27)->input[i]);
        input_aligned_npu_Subgraphs_27[i] = shl_mem_alloc_aligned(input_size_npu_Subgraphs_27[i], 0);
    }
    struct csinn_tensor* input_tensors_npu_Subgraphs_27[npu_Subgraphs_27_input_num];
    input_tensors_npu_Subgraphs_27[0] = csinn_alloc_tensor(NULL);
    input_tensors_npu_Subgraphs_27[0]->dim_count = 3;
    input_tensors_npu_Subgraphs_27[0]->dim[0] = 1;
    input_tensors_npu_Subgraphs_27[0]->dim[1] = 32;
    input_tensors_npu_Subgraphs_27[0]->dim[2] = 7680;
    struct csinn_tensor* output_tensors_npu_Subgraphs_27[npu_Subgraphs_27_output_num];
    output_tensors_npu_Subgraphs_27[0] = csinn_alloc_tensor(NULL);
    output_tensors_npu_Subgraphs_27[0]->dim_count = 3;
    output_tensors_npu_Subgraphs_27[0]->dim[0] = 1;
    output_tensors_npu_Subgraphs_27[0]->dim[1] = 640;
    output_tensors_npu_Subgraphs_27[0]->dim[2] = 16;
    output_tensors_npu_Subgraphs_27[0]->dim[3] = 16;

    /******************npu_Subgraphs_28**************************/
    int npu_Subgraphs_28_input_num = 2;
    int npu_Subgraphs_28_output_num = 2;
    float *npu_Subgraphs_28_inputf[npu_Subgraphs_28_input_num];
    int8_t *npu_Subgraphs_28_input[npu_Subgraphs_28_input_num];

    void *sess_npu_Subgraphs_28 = create_Subgraphs(argv[36], csinn_npu_Subgraphs_28);

    int input_size_npu_Subgraphs_28[] = {1 * 960 * 16 * 16, 1 * 32 * 5120, };
    void *input_aligned_npu_Subgraphs_28[npu_Subgraphs_28_input_num];
    for (i = 0; i < npu_Subgraphs_28_input_num; i++) {
        input_size_npu_Subgraphs_28[i] = csinn_tensor_byte_size(((struct csinn_session *)sess_npu_Subgraphs_28)->input[i]);
        input_aligned_npu_Subgraphs_28[i] = shl_mem_alloc_aligned(input_size_npu_Subgraphs_28[i], 0);
    }
    struct csinn_tensor* input_tensors_npu_Subgraphs_28[npu_Subgraphs_28_input_num];
    input_tensors_npu_Subgraphs_28[0] = csinn_alloc_tensor(NULL);
    input_tensors_npu_Subgraphs_28[0]->dim_count = 4;
    input_tensors_npu_Subgraphs_28[0]->dim[0] = 1;
    input_tensors_npu_Subgraphs_28[0]->dim[1] = 960;
    input_tensors_npu_Subgraphs_28[0]->dim[2] = 16;
    input_tensors_npu_Subgraphs_28[0]->dim[3] = 16;
    input_tensors_npu_Subgraphs_28[1] = csinn_alloc_tensor(NULL);
    input_tensors_npu_Subgraphs_28[1]->dim_count = 3;
    input_tensors_npu_Subgraphs_28[1]->dim[0] = 1;
    input_tensors_npu_Subgraphs_28[1]->dim[1] = 32;
    input_tensors_npu_Subgraphs_28[1]->dim[2] = 5120;
    struct csinn_tensor* output_tensors_npu_Subgraphs_28[npu_Subgraphs_28_output_num];
    output_tensors_npu_Subgraphs_28[0] = csinn_alloc_tensor(NULL);
    output_tensors_npu_Subgraphs_28[0]->dim_count = 3;
    output_tensors_npu_Subgraphs_28[0]->dim[0] = 1;
    output_tensors_npu_Subgraphs_28[0]->dim[1] = 32;
    output_tensors_npu_Subgraphs_28[0]->dim[2] = 5120;
    output_tensors_npu_Subgraphs_28[1] = csinn_alloc_tensor(NULL);
    output_tensors_npu_Subgraphs_28[1]->dim_count = 4;
    output_tensors_npu_Subgraphs_28[1]->dim[0] = 1;
    output_tensors_npu_Subgraphs_28[1]->dim[1] = 640;
    output_tensors_npu_Subgraphs_28[1]->dim[2] = 16;
    output_tensors_npu_Subgraphs_28[1]->dim[3] = 16;

    /******************npu_Subgraphs_29**************************/
    int npu_Subgraphs_29_input_num = 1;
    int npu_Subgraphs_29_output_num = 1;
    float *npu_Subgraphs_29_inputf[npu_Subgraphs_29_input_num];
    int8_t *npu_Subgraphs_29_input[npu_Subgraphs_29_input_num];

    void *sess_npu_Subgraphs_29 = create_Subgraphs(argv[37], csinn_npu_Subgraphs_29);

    int input_size_npu_Subgraphs_29[] = {1 * 32 * 5120, };
    void *input_aligned_npu_Subgraphs_29[npu_Subgraphs_29_input_num];
    for (i = 0; i < npu_Subgraphs_29_input_num; i++) {
        input_size_npu_Subgraphs_29[i] = csinn_tensor_byte_size(((struct csinn_session *)sess_npu_Subgraphs_29)->input[i]);
        input_aligned_npu_Subgraphs_29[i] = shl_mem_alloc_aligned(input_size_npu_Subgraphs_29[i], 0);
    }
    struct csinn_tensor* input_tensors_npu_Subgraphs_29[npu_Subgraphs_29_input_num];
    input_tensors_npu_Subgraphs_29[0] = csinn_alloc_tensor(NULL);
    input_tensors_npu_Subgraphs_29[0]->dim_count = 3;
    input_tensors_npu_Subgraphs_29[0]->dim[0] = 1;
    input_tensors_npu_Subgraphs_29[0]->dim[1] = 32;
    input_tensors_npu_Subgraphs_29[0]->dim[2] = 5120;
    struct csinn_tensor* output_tensors_npu_Subgraphs_29[npu_Subgraphs_29_output_num];
    output_tensors_npu_Subgraphs_29[0] = csinn_alloc_tensor(NULL);
    output_tensors_npu_Subgraphs_29[0]->dim_count = 4;
    output_tensors_npu_Subgraphs_29[0]->dim[0] = 1;
    output_tensors_npu_Subgraphs_29[0]->dim[1] = 640;
    output_tensors_npu_Subgraphs_29[0]->dim[2] = 16;
    output_tensors_npu_Subgraphs_29[0]->dim[3] = 16;


    /******************cpu_Subgraphs_7**************************/
    int cpu_Subgraphs_7_input_num = 2;
    int cpu_Subgraphs_7_output_num = 1;
    float *cpu_Subgraphs_7_inputf[cpu_Subgraphs_7_input_num];

    void *sess_cpu_Subgraphs_7 = create_Subgraphs(argv[38], csinn_cpu_Subgraphs_7);

    struct csinn_tensor* input_tensors_cpu_Subgraphs_7[cpu_Subgraphs_7_input_num];
    input_tensors_cpu_Subgraphs_7[0] = csinn_alloc_tensor(NULL);
    input_tensors_cpu_Subgraphs_7[0]->dim_count = 4;
    input_tensors_cpu_Subgraphs_7[0]->dim[0] = 1;
    input_tensors_cpu_Subgraphs_7[0]->dim[1] = 640;
    input_tensors_cpu_Subgraphs_7[0]->dim[2] = 16;
    input_tensors_cpu_Subgraphs_7[0]->dim[3] = 16;
    input_tensors_cpu_Subgraphs_7[1] = csinn_alloc_tensor(NULL);
    input_tensors_cpu_Subgraphs_7[1]->dim_count = 3;
    input_tensors_cpu_Subgraphs_7[1]->dim[0] = 1;
    input_tensors_cpu_Subgraphs_7[1]->dim[1] = 77;
    input_tensors_cpu_Subgraphs_7[1]->dim[2] = 768;
    struct csinn_tensor* output_tensors_cpu_Subgraphs_7[cpu_Subgraphs_7_output_num];
    output_tensors_cpu_Subgraphs_7[0] = csinn_alloc_tensor(NULL);
    output_tensors_cpu_Subgraphs_7[0]->dim_count = 4;
    output_tensors_cpu_Subgraphs_7[0]->dim[0] = 1;
    output_tensors_cpu_Subgraphs_7[0]->dim[1] = 640;
    output_tensors_cpu_Subgraphs_7[0]->dim[2] = 16;
    output_tensors_cpu_Subgraphs_7[0]->dim[3] = 16;

    /******************npu_Subgraphs_30**************************/
    int npu_Subgraphs_30_input_num = 2;
    int npu_Subgraphs_30_output_num = 1;
    float *npu_Subgraphs_30_inputf[npu_Subgraphs_30_input_num];
    int8_t *npu_Subgraphs_30_input[npu_Subgraphs_30_input_num];

    void *sess_npu_Subgraphs_30 = create_Subgraphs(argv[39], csinn_npu_Subgraphs_30);

    int input_size_npu_Subgraphs_30[] = {1 * 640 * 16 * 16, 1 * 640 * 16 * 16, };
    void *input_aligned_npu_Subgraphs_30[npu_Subgraphs_30_input_num];
    for (i = 0; i < npu_Subgraphs_30_input_num; i++) {
        input_size_npu_Subgraphs_30[i] = csinn_tensor_byte_size(((struct csinn_session *)sess_npu_Subgraphs_30)->input[i]);
        input_aligned_npu_Subgraphs_30[i] = shl_mem_alloc_aligned(input_size_npu_Subgraphs_30[i], 0);
    }
    struct csinn_tensor* input_tensors_npu_Subgraphs_30[npu_Subgraphs_30_input_num];
    input_tensors_npu_Subgraphs_30[0] = csinn_alloc_tensor(NULL);
    input_tensors_npu_Subgraphs_30[0]->dim_count = 4;
    input_tensors_npu_Subgraphs_30[0]->dim[0] = 1;
    input_tensors_npu_Subgraphs_30[0]->dim[1] = 640;
    input_tensors_npu_Subgraphs_30[0]->dim[2] = 16;
    input_tensors_npu_Subgraphs_30[0]->dim[3] = 16;
    input_tensors_npu_Subgraphs_30[1] = csinn_alloc_tensor(NULL);
    input_tensors_npu_Subgraphs_30[1]->dim_count = 4;
    input_tensors_npu_Subgraphs_30[1]->dim[0] = 1;
    input_tensors_npu_Subgraphs_30[1]->dim[1] = 640;
    input_tensors_npu_Subgraphs_30[1]->dim[2] = 16;
    input_tensors_npu_Subgraphs_30[1]->dim[3] = 16;
    struct csinn_tensor* output_tensors_npu_Subgraphs_30[npu_Subgraphs_30_output_num];
    output_tensors_npu_Subgraphs_30[0] = csinn_alloc_tensor(NULL);
    output_tensors_npu_Subgraphs_30[0]->dim_count = 4;
    output_tensors_npu_Subgraphs_30[0]->dim[0] = 1;
    output_tensors_npu_Subgraphs_30[0]->dim[1] = 640;
    output_tensors_npu_Subgraphs_30[0]->dim[2] = 32;
    output_tensors_npu_Subgraphs_30[0]->dim[3] = 32;

    /******************npu_Subgraphs_31**************************/
    int npu_Subgraphs_31_input_num = 1;
    int npu_Subgraphs_31_output_num = 1;
    float *npu_Subgraphs_31_inputf[npu_Subgraphs_31_input_num];
    int8_t *npu_Subgraphs_31_input[npu_Subgraphs_31_input_num];

    void *sess_npu_Subgraphs_31 = create_Subgraphs(argv[40], csinn_npu_Subgraphs_31);

    int input_size_npu_Subgraphs_31[] = {1 * 1280, };
    void *input_aligned_npu_Subgraphs_31[npu_Subgraphs_31_input_num];
    for (i = 0; i < npu_Subgraphs_31_input_num; i++) {
        input_size_npu_Subgraphs_31[i] = csinn_tensor_byte_size(((struct csinn_session *)sess_npu_Subgraphs_31)->input[i]);
        input_aligned_npu_Subgraphs_31[i] = shl_mem_alloc_aligned(input_size_npu_Subgraphs_31[i], 0);
    }
    struct csinn_tensor* input_tensors_npu_Subgraphs_31[npu_Subgraphs_31_input_num];
    input_tensors_npu_Subgraphs_31[0] = csinn_alloc_tensor(NULL);
    input_tensors_npu_Subgraphs_31[0]->dim_count = 2;
    input_tensors_npu_Subgraphs_31[0]->dim[0] = 1;
    input_tensors_npu_Subgraphs_31[0]->dim[1] = 1280;
    struct csinn_tensor* output_tensors_npu_Subgraphs_31[npu_Subgraphs_31_output_num];
    output_tensors_npu_Subgraphs_31[0] = csinn_alloc_tensor(NULL);
    output_tensors_npu_Subgraphs_31[0]->dim_count = 4;
    output_tensors_npu_Subgraphs_31[0]->dim[0] = 1;
    output_tensors_npu_Subgraphs_31[0]->dim[1] = 320;
    output_tensors_npu_Subgraphs_31[0]->dim[2] = 1;
    output_tensors_npu_Subgraphs_31[0]->dim[3] = 1;

    /******************npu_Subgraphs_32**************************/
    int npu_Subgraphs_32_input_num = 1;
    int npu_Subgraphs_32_output_num = 1;
    float *npu_Subgraphs_32_inputf[npu_Subgraphs_32_input_num];
    int8_t *npu_Subgraphs_32_input[npu_Subgraphs_32_input_num];

    void *sess_npu_Subgraphs_32 = create_Subgraphs(argv[41], csinn_npu_Subgraphs_32);

    int input_size_npu_Subgraphs_32[] = {1 * 32 * 30720, };
    void *input_aligned_npu_Subgraphs_32[npu_Subgraphs_32_input_num];
    for (i = 0; i < npu_Subgraphs_32_input_num; i++) {
        input_size_npu_Subgraphs_32[i] = csinn_tensor_byte_size(((struct csinn_session *)sess_npu_Subgraphs_32)->input[i]);
        input_aligned_npu_Subgraphs_32[i] = shl_mem_alloc_aligned(input_size_npu_Subgraphs_32[i], 0);
    }
    struct csinn_tensor* input_tensors_npu_Subgraphs_32[npu_Subgraphs_32_input_num];
    input_tensors_npu_Subgraphs_32[0] = csinn_alloc_tensor(NULL);
    input_tensors_npu_Subgraphs_32[0]->dim_count = 3;
    input_tensors_npu_Subgraphs_32[0]->dim[0] = 1;
    input_tensors_npu_Subgraphs_32[0]->dim[1] = 32;
    input_tensors_npu_Subgraphs_32[0]->dim[2] = 30720;
    struct csinn_tensor* output_tensors_npu_Subgraphs_32[npu_Subgraphs_32_output_num];
    output_tensors_npu_Subgraphs_32[0] = csinn_alloc_tensor(NULL);
    output_tensors_npu_Subgraphs_32[0]->dim_count = 3;
    output_tensors_npu_Subgraphs_32[0]->dim[0] = 1;
    output_tensors_npu_Subgraphs_32[0]->dim[1] = 320;
    output_tensors_npu_Subgraphs_32[0]->dim[2] = 32;
    output_tensors_npu_Subgraphs_32[0]->dim[3] = 32;

    /******************npu_Subgraphs_33**************************/
    int npu_Subgraphs_33_input_num = 2;
    int npu_Subgraphs_33_output_num = 2;
    float *npu_Subgraphs_33_inputf[npu_Subgraphs_33_input_num];
    int8_t *npu_Subgraphs_33_input[npu_Subgraphs_33_input_num];

    void *sess_npu_Subgraphs_33 = create_Subgraphs(argv[42], csinn_npu_Subgraphs_33);

    int input_size_npu_Subgraphs_33[] = {1 * 960 * 32 * 32, 1 * 32 * 10240, };
    void *input_aligned_npu_Subgraphs_33[npu_Subgraphs_33_input_num];
    for (i = 0; i < npu_Subgraphs_33_input_num; i++) {
        input_size_npu_Subgraphs_33[i] = csinn_tensor_byte_size(((struct csinn_session *)sess_npu_Subgraphs_33)->input[i]);
        input_aligned_npu_Subgraphs_33[i] = shl_mem_alloc_aligned(input_size_npu_Subgraphs_33[i], 0);
    }
    struct csinn_tensor* input_tensors_npu_Subgraphs_33[npu_Subgraphs_33_input_num];
    input_tensors_npu_Subgraphs_33[0] = csinn_alloc_tensor(NULL);
    input_tensors_npu_Subgraphs_33[0]->dim_count = 4;
    input_tensors_npu_Subgraphs_33[0]->dim[0] = 1;
    input_tensors_npu_Subgraphs_33[0]->dim[1] = 960;
    input_tensors_npu_Subgraphs_33[0]->dim[2] = 32;
    input_tensors_npu_Subgraphs_33[0]->dim[3] = 32;
    input_tensors_npu_Subgraphs_33[1] = csinn_alloc_tensor(NULL);
    input_tensors_npu_Subgraphs_33[1]->dim_count = 3;
    input_tensors_npu_Subgraphs_33[1]->dim[0] = 1;
    input_tensors_npu_Subgraphs_33[1]->dim[1] = 32;
    input_tensors_npu_Subgraphs_33[1]->dim[2] = 10240;
    struct csinn_tensor* output_tensors_npu_Subgraphs_33[npu_Subgraphs_33_output_num];
    output_tensors_npu_Subgraphs_33[0] = csinn_alloc_tensor(NULL);
    output_tensors_npu_Subgraphs_33[0]->dim_count = 3;
    output_tensors_npu_Subgraphs_33[0]->dim[0] = 1;
    output_tensors_npu_Subgraphs_33[0]->dim[1] = 32;
    output_tensors_npu_Subgraphs_33[0]->dim[2] = 10240;
    output_tensors_npu_Subgraphs_33[1] = csinn_alloc_tensor(NULL);
    output_tensors_npu_Subgraphs_33[1]->dim_count = 4;
    output_tensors_npu_Subgraphs_33[1]->dim[0] = 1;
    output_tensors_npu_Subgraphs_33[1]->dim[1] = 320;
    output_tensors_npu_Subgraphs_33[1]->dim[2] = 32;
    output_tensors_npu_Subgraphs_33[1]->dim[3] = 32;

    /******************npu_Subgraphs_34**************************/
    int npu_Subgraphs_34_input_num = 1;
    int npu_Subgraphs_34_output_num = 1;
    float *npu_Subgraphs_34_inputf[npu_Subgraphs_34_input_num];
    int8_t *npu_Subgraphs_34_input[npu_Subgraphs_34_input_num];

    void *sess_npu_Subgraphs_34 = create_Subgraphs(argv[43], csinn_npu_Subgraphs_34);

    int input_size_npu_Subgraphs_34[] = {1 * 32 * 10240, };
    void *input_aligned_npu_Subgraphs_34[npu_Subgraphs_34_input_num];
    for (i = 0; i < npu_Subgraphs_34_input_num; i++) {
        input_size_npu_Subgraphs_34[i] = csinn_tensor_byte_size(((struct csinn_session *)sess_npu_Subgraphs_34)->input[i]);
        input_aligned_npu_Subgraphs_34[i] = shl_mem_alloc_aligned(input_size_npu_Subgraphs_34[i], 0);
    }
    struct csinn_tensor* input_tensors_npu_Subgraphs_34[npu_Subgraphs_34_input_num];
    input_tensors_npu_Subgraphs_34[0] = csinn_alloc_tensor(NULL);
    input_tensors_npu_Subgraphs_34[0]->dim_count = 3;
    input_tensors_npu_Subgraphs_34[0]->dim[0] = 1;
    input_tensors_npu_Subgraphs_34[0]->dim[1] = 32;
    input_tensors_npu_Subgraphs_34[0]->dim[2] = 10240;
    struct csinn_tensor* output_tensors_npu_Subgraphs_34[npu_Subgraphs_34_output_num];
    output_tensors_npu_Subgraphs_34[0] = csinn_alloc_tensor(NULL);
    output_tensors_npu_Subgraphs_34[0]->dim_count = 4;
    output_tensors_npu_Subgraphs_34[0]->dim[0] = 1;
    output_tensors_npu_Subgraphs_34[0]->dim[1] = 320;
    output_tensors_npu_Subgraphs_34[0]->dim[2] = 32;
    output_tensors_npu_Subgraphs_34[0]->dim[3] = 32;


    /******************cpu_Subgraphs_8**************************/
    int cpu_Subgraphs_8_input_num = 2;
    int cpu_Subgraphs_8_output_num = 1;
    float *cpu_Subgraphs_8_inputf[cpu_Subgraphs_8_input_num];

    void *sess_cpu_Subgraphs_8 = create_Subgraphs(argv[44], csinn_cpu_Subgraphs_8);

    struct csinn_tensor* input_tensors_cpu_Subgraphs_8[cpu_Subgraphs_8_input_num];
    input_tensors_cpu_Subgraphs_8[0] = csinn_alloc_tensor(NULL);
    input_tensors_cpu_Subgraphs_8[0]->dim_count = 4;
    input_tensors_cpu_Subgraphs_8[0]->dim[0] = 1;
    input_tensors_cpu_Subgraphs_8[0]->dim[1] = 320;
    input_tensors_cpu_Subgraphs_8[0]->dim[2] = 32;
    input_tensors_cpu_Subgraphs_8[0]->dim[3] = 32;
    input_tensors_cpu_Subgraphs_8[1] = csinn_alloc_tensor(NULL);
    input_tensors_cpu_Subgraphs_8[1]->dim_count = 3;
    input_tensors_cpu_Subgraphs_8[1]->dim[0] = 1;
    input_tensors_cpu_Subgraphs_8[1]->dim[1] = 77;
    input_tensors_cpu_Subgraphs_8[1]->dim[2] = 768;
    struct csinn_tensor* output_tensors_cpu_Subgraphs_8[cpu_Subgraphs_8_output_num];
    output_tensors_cpu_Subgraphs_8[0] = csinn_alloc_tensor(NULL);
    output_tensors_cpu_Subgraphs_8[0]->dim_count = 4;
    output_tensors_cpu_Subgraphs_8[0]->dim[0] = 1;
    output_tensors_cpu_Subgraphs_8[0]->dim[1] = 320;
    output_tensors_cpu_Subgraphs_8[0]->dim[2] = 32;
    output_tensors_cpu_Subgraphs_8[0]->dim[3] = 32;

    /******************npu_Subgraphs_35**************************/
    int npu_Subgraphs_35_input_num = 2;
    int npu_Subgraphs_35_output_num = 1;
    float *npu_Subgraphs_35_inputf[npu_Subgraphs_35_input_num];
    int8_t *npu_Subgraphs_35_input[npu_Subgraphs_35_input_num];

    void *sess_npu_Subgraphs_35 = create_Subgraphs(argv[45], csinn_npu_Subgraphs_35);

    int input_size_npu_Subgraphs_35[] = {1 * 320 * 32 * 32, 1 * 320 * 32 * 32, };
    void *input_aligned_npu_Subgraphs_35[npu_Subgraphs_35_input_num];
    for (i = 0; i < npu_Subgraphs_35_input_num; i++) {
        input_size_npu_Subgraphs_35[i] = csinn_tensor_byte_size(((struct csinn_session *)sess_npu_Subgraphs_35)->input[i]);
        input_aligned_npu_Subgraphs_35[i] = shl_mem_alloc_aligned(input_size_npu_Subgraphs_35[i], 0);
    }
    struct csinn_tensor* input_tensors_npu_Subgraphs_35[npu_Subgraphs_35_input_num];
    input_tensors_npu_Subgraphs_35[0] = csinn_alloc_tensor(NULL);
    input_tensors_npu_Subgraphs_35[0]->dim_count = 4;
    input_tensors_npu_Subgraphs_35[0]->dim[0] = 1;
    input_tensors_npu_Subgraphs_35[0]->dim[1] = 320;
    input_tensors_npu_Subgraphs_35[0]->dim[2] = 32;
    input_tensors_npu_Subgraphs_35[0]->dim[3] = 32;
    input_tensors_npu_Subgraphs_35[1] = csinn_alloc_tensor(NULL);
    input_tensors_npu_Subgraphs_35[1]->dim_count = 4;
    input_tensors_npu_Subgraphs_35[1]->dim[0] = 1;
    input_tensors_npu_Subgraphs_35[1]->dim[1] = 320;
    input_tensors_npu_Subgraphs_35[1]->dim[2] = 32;
    input_tensors_npu_Subgraphs_35[1]->dim[3] = 32;
    struct csinn_tensor* output_tensors_npu_Subgraphs_35[npu_Subgraphs_35_output_num];
    output_tensors_npu_Subgraphs_35[0] = csinn_alloc_tensor(NULL);
    output_tensors_npu_Subgraphs_35[0]->dim_count = 4;
    output_tensors_npu_Subgraphs_35[0]->dim[0] = 1;
    output_tensors_npu_Subgraphs_35[0]->dim[1] = 320;
    output_tensors_npu_Subgraphs_35[0]->dim[2] = 32;
    output_tensors_npu_Subgraphs_35[0]->dim[3] = 32;

    /******************npu_Subgraphs_36**************************/
    int npu_Subgraphs_36_input_num = 1;
    int npu_Subgraphs_36_output_num = 1;
    float *npu_Subgraphs_36_inputf[npu_Subgraphs_36_input_num];
    int8_t *npu_Subgraphs_36_input[npu_Subgraphs_36_input_num];

    void *sess_npu_Subgraphs_36 = create_Subgraphs(argv[46], csinn_npu_Subgraphs_36);

    int input_size_npu_Subgraphs_36[] = {1 * 1280, };
    void *input_aligned_npu_Subgraphs_36[npu_Subgraphs_36_input_num];
    for (i = 0; i < npu_Subgraphs_36_input_num; i++) {
        input_size_npu_Subgraphs_36[i] = csinn_tensor_byte_size(((struct csinn_session *)sess_npu_Subgraphs_36)->input[i]);
        input_aligned_npu_Subgraphs_36[i] = shl_mem_alloc_aligned(input_size_npu_Subgraphs_36[i], 0);
    }
    struct csinn_tensor* input_tensors_npu_Subgraphs_36[npu_Subgraphs_36_input_num];
    input_tensors_npu_Subgraphs_36[0] = csinn_alloc_tensor(NULL);
    input_tensors_npu_Subgraphs_36[0]->dim_count = 2;
    input_tensors_npu_Subgraphs_36[0]->dim[0] = 1;
    input_tensors_npu_Subgraphs_36[0]->dim[1] = 1280;
    struct csinn_tensor* output_tensors_npu_Subgraphs_36[npu_Subgraphs_36_output_num];
    output_tensors_npu_Subgraphs_36[0] = csinn_alloc_tensor(NULL);
    output_tensors_npu_Subgraphs_36[0]->dim_count = 4;
    output_tensors_npu_Subgraphs_36[0]->dim[0] = 1;
    output_tensors_npu_Subgraphs_36[0]->dim[1] = 320;
    output_tensors_npu_Subgraphs_36[0]->dim[2] = 1;
    output_tensors_npu_Subgraphs_36[0]->dim[3] = 1;

    /******************npu_Subgraphs_37**************************/
    int npu_Subgraphs_37_input_num = 1;
    int npu_Subgraphs_37_output_num = 1;
    float *npu_Subgraphs_37_inputf[npu_Subgraphs_37_input_num];
    int8_t *npu_Subgraphs_37_input[npu_Subgraphs_37_input_num];

    void *sess_npu_Subgraphs_37 = create_Subgraphs(argv[47], csinn_npu_Subgraphs_37);

    int input_size_npu_Subgraphs_37[] = {1 * 32 * 20480, };
    void *input_aligned_npu_Subgraphs_37[npu_Subgraphs_37_input_num];
    for (i = 0; i < npu_Subgraphs_37_input_num; i++) {
        input_size_npu_Subgraphs_37[i] = csinn_tensor_byte_size(((struct csinn_session *)sess_npu_Subgraphs_37)->input[i]);
        input_aligned_npu_Subgraphs_37[i] = shl_mem_alloc_aligned(input_size_npu_Subgraphs_37[i], 0);
    }
    struct csinn_tensor* input_tensors_npu_Subgraphs_37[npu_Subgraphs_37_input_num];
    input_tensors_npu_Subgraphs_37[0] = csinn_alloc_tensor(NULL);
    input_tensors_npu_Subgraphs_37[0]->dim_count = 3;
    input_tensors_npu_Subgraphs_37[0]->dim[0] = 1;
    input_tensors_npu_Subgraphs_37[0]->dim[1] = 32;
    input_tensors_npu_Subgraphs_37[0]->dim[2] = 20480;
    struct csinn_tensor* output_tensors_npu_Subgraphs_37[npu_Subgraphs_37_output_num];
    output_tensors_npu_Subgraphs_37[0] = csinn_alloc_tensor(NULL);
    output_tensors_npu_Subgraphs_37[0]->dim_count = 3;
    output_tensors_npu_Subgraphs_37[0]->dim[0] = 1;
    output_tensors_npu_Subgraphs_37[0]->dim[1] = 320;
    output_tensors_npu_Subgraphs_37[0]->dim[2] = 32;
    output_tensors_npu_Subgraphs_37[0]->dim[3] = 32;

    /******************npu_Subgraphs_38**************************/
    int npu_Subgraphs_38_input_num = 2;
    int npu_Subgraphs_38_output_num = 2;
    float *npu_Subgraphs_38_inputf[npu_Subgraphs_38_input_num];
    int8_t *npu_Subgraphs_38_input[npu_Subgraphs_38_input_num];

    void *sess_npu_Subgraphs_38 = create_Subgraphs(argv[48], csinn_npu_Subgraphs_38);

    int input_size_npu_Subgraphs_38[] = {1 * 640 * 64 * 64, 1 * 32 * 40960, };
    void *input_aligned_npu_Subgraphs_38[npu_Subgraphs_38_input_num];
    for (i = 0; i < npu_Subgraphs_38_input_num; i++) {
        input_size_npu_Subgraphs_38[i] = csinn_tensor_byte_size(((struct csinn_session *)sess_npu_Subgraphs_38)->input[i]);
        input_aligned_npu_Subgraphs_38[i] = shl_mem_alloc_aligned(input_size_npu_Subgraphs_38[i], 0);
    }
    struct csinn_tensor* input_tensors_npu_Subgraphs_38[npu_Subgraphs_38_input_num];
    input_tensors_npu_Subgraphs_38[0] = csinn_alloc_tensor(NULL);
    input_tensors_npu_Subgraphs_38[0]->dim_count = 4;
    input_tensors_npu_Subgraphs_38[0]->dim[0] = 1;
    input_tensors_npu_Subgraphs_38[0]->dim[1] = 640;
    input_tensors_npu_Subgraphs_38[0]->dim[2] = 32;
    input_tensors_npu_Subgraphs_38[0]->dim[3] = 32;
    input_tensors_npu_Subgraphs_38[1] = csinn_alloc_tensor(NULL);
    input_tensors_npu_Subgraphs_38[1]->dim_count = 3;
    input_tensors_npu_Subgraphs_38[1]->dim[0] = 1;
    input_tensors_npu_Subgraphs_38[1]->dim[1] = 32;
    input_tensors_npu_Subgraphs_38[1]->dim[2] = 10240;
    struct csinn_tensor* output_tensors_npu_Subgraphs_38[npu_Subgraphs_38_output_num];
    output_tensors_npu_Subgraphs_38[0] = csinn_alloc_tensor(NULL);
    output_tensors_npu_Subgraphs_38[0]->dim_count = 3;
    output_tensors_npu_Subgraphs_38[0]->dim[0] = 1;
    output_tensors_npu_Subgraphs_38[0]->dim[1] = 32;
    output_tensors_npu_Subgraphs_38[0]->dim[2] = 10240;
    output_tensors_npu_Subgraphs_38[1] = csinn_alloc_tensor(NULL);
    output_tensors_npu_Subgraphs_38[1]->dim_count = 4;
    output_tensors_npu_Subgraphs_38[1]->dim[0] = 1;
    output_tensors_npu_Subgraphs_38[1]->dim[1] = 320;
    output_tensors_npu_Subgraphs_38[1]->dim[2] = 32;
    output_tensors_npu_Subgraphs_38[1]->dim[3] = 32;

    /******************npu_Subgraphs_39**************************/
    int npu_Subgraphs_39_input_num = 1;
    int npu_Subgraphs_39_output_num = 1;
    float *npu_Subgraphs_39_inputf[npu_Subgraphs_39_input_num];
    int8_t *npu_Subgraphs_39_input[npu_Subgraphs_39_input_num];

    void *sess_npu_Subgraphs_39 = create_Subgraphs(argv[49], csinn_npu_Subgraphs_39);

    int input_size_npu_Subgraphs_39[] = {1 * 32 * 10240, };
    void *input_aligned_npu_Subgraphs_39[npu_Subgraphs_39_input_num];
    for (i = 0; i < npu_Subgraphs_39_input_num; i++) {
        input_size_npu_Subgraphs_39[i] = csinn_tensor_byte_size(((struct csinn_session *)sess_npu_Subgraphs_39)->input[i]);
        input_aligned_npu_Subgraphs_39[i] = shl_mem_alloc_aligned(input_size_npu_Subgraphs_39[i], 0);
    }
    struct csinn_tensor* input_tensors_npu_Subgraphs_39[npu_Subgraphs_39_input_num];
    input_tensors_npu_Subgraphs_39[0] = csinn_alloc_tensor(NULL);
    input_tensors_npu_Subgraphs_39[0]->dim_count = 3;
    input_tensors_npu_Subgraphs_39[0]->dim[0] = 1;
    input_tensors_npu_Subgraphs_39[0]->dim[1] = 32;
    input_tensors_npu_Subgraphs_39[0]->dim[2] = 10240;
    struct csinn_tensor* output_tensors_npu_Subgraphs_39[npu_Subgraphs_39_output_num];
    output_tensors_npu_Subgraphs_39[0] = csinn_alloc_tensor(NULL);
    output_tensors_npu_Subgraphs_39[0]->dim_count = 4;
    output_tensors_npu_Subgraphs_39[0]->dim[0] = 1;
    output_tensors_npu_Subgraphs_39[0]->dim[1] = 320;
    output_tensors_npu_Subgraphs_39[0]->dim[2] = 32;
    output_tensors_npu_Subgraphs_39[0]->dim[3] = 32;


    /******************cpu_Subgraphs_9**************************/
    int cpu_Subgraphs_9_input_num = 2;
    int cpu_Subgraphs_9_output_num = 1;
    float *cpu_Subgraphs_9_inputf[cpu_Subgraphs_9_input_num];

    void *sess_cpu_Subgraphs_9 = create_Subgraphs(argv[50], csinn_cpu_Subgraphs_9);

    struct csinn_tensor* input_tensors_cpu_Subgraphs_9[cpu_Subgraphs_9_input_num];
    input_tensors_cpu_Subgraphs_9[0] = csinn_alloc_tensor(NULL);
    input_tensors_cpu_Subgraphs_9[0]->dim_count = 4;
    input_tensors_cpu_Subgraphs_9[0]->dim[0] = 1;
    input_tensors_cpu_Subgraphs_9[0]->dim[1] = 320;
    input_tensors_cpu_Subgraphs_9[0]->dim[2] = 32;
    input_tensors_cpu_Subgraphs_9[0]->dim[3] = 32;
    input_tensors_cpu_Subgraphs_9[1] = csinn_alloc_tensor(NULL);
    input_tensors_cpu_Subgraphs_9[1]->dim_count = 3;
    input_tensors_cpu_Subgraphs_9[1]->dim[0] = 1;
    input_tensors_cpu_Subgraphs_9[1]->dim[1] = 77;
    input_tensors_cpu_Subgraphs_9[1]->dim[2] = 768;
    struct csinn_tensor* output_tensors_cpu_Subgraphs_9[cpu_Subgraphs_9_output_num];
    output_tensors_cpu_Subgraphs_9[0] = csinn_alloc_tensor(NULL);
    output_tensors_cpu_Subgraphs_9[0]->dim_count = 4;
    output_tensors_cpu_Subgraphs_9[0]->dim[0] = 1;
    output_tensors_cpu_Subgraphs_9[0]->dim[1] = 320;
    output_tensors_cpu_Subgraphs_9[0]->dim[2] = 32;
    output_tensors_cpu_Subgraphs_9[0]->dim[3] = 32;

    /******************npu_Subgraphs_40**************************/
    int npu_Subgraphs_40_input_num = 2;
    int npu_Subgraphs_40_output_num = 1;
    float *npu_Subgraphs_40_inputf[npu_Subgraphs_40_input_num];
    int8_t *npu_Subgraphs_40_input[npu_Subgraphs_40_input_num];

    void *sess_npu_Subgraphs_40 = create_Subgraphs(argv[51], csinn_npu_Subgraphs_40);

    int input_size_npu_Subgraphs_40[] = {1 * 320 * 32 * 32, 1 * 320 * 32 * 32, };
    void *input_aligned_npu_Subgraphs_40[npu_Subgraphs_40_input_num];
    for (i = 0; i < npu_Subgraphs_40_input_num; i++) {
        input_size_npu_Subgraphs_40[i] = csinn_tensor_byte_size(((struct csinn_session *)sess_npu_Subgraphs_40)->input[i]);
        input_aligned_npu_Subgraphs_40[i] = shl_mem_alloc_aligned(input_size_npu_Subgraphs_40[i], 0);
    }
    struct csinn_tensor* input_tensors_npu_Subgraphs_40[npu_Subgraphs_40_input_num];
    input_tensors_npu_Subgraphs_40[0] = csinn_alloc_tensor(NULL);
    input_tensors_npu_Subgraphs_40[0]->dim_count = 4;
    input_tensors_npu_Subgraphs_40[0]->dim[0] = 1;
    input_tensors_npu_Subgraphs_40[0]->dim[1] = 320;
    input_tensors_npu_Subgraphs_40[0]->dim[2] = 32;
    input_tensors_npu_Subgraphs_40[0]->dim[3] = 32;
    input_tensors_npu_Subgraphs_40[1] = csinn_alloc_tensor(NULL);
    input_tensors_npu_Subgraphs_40[1]->dim_count = 4;
    input_tensors_npu_Subgraphs_40[1]->dim[0] = 1;
    input_tensors_npu_Subgraphs_40[1]->dim[1] = 320;
    input_tensors_npu_Subgraphs_40[1]->dim[2] = 32;
    input_tensors_npu_Subgraphs_40[1]->dim[3] = 32;
    struct csinn_tensor* output_tensors_npu_Subgraphs_40[npu_Subgraphs_40_output_num];
    output_tensors_npu_Subgraphs_40[0] = csinn_alloc_tensor(NULL);
    output_tensors_npu_Subgraphs_40[0]->dim_count = 3;
    output_tensors_npu_Subgraphs_40[0]->dim[0] = 1;
    output_tensors_npu_Subgraphs_40[0]->dim[1] = 32;
    output_tensors_npu_Subgraphs_40[0]->dim[2] = 10240;

    /******************npu_Subgraphs_41**************************/
    int npu_Subgraphs_41_input_num = 1;
    int npu_Subgraphs_41_output_num = 1;
    float *npu_Subgraphs_41_inputf[npu_Subgraphs_41_input_num];
    int8_t *npu_Subgraphs_41_input[npu_Subgraphs_41_input_num];

    void *sess_npu_Subgraphs_41 = create_Subgraphs(argv[52], csinn_npu_Subgraphs_41);

    int input_size_npu_Subgraphs_41[] = {1 * 32 * 10240, };
    void *input_aligned_npu_Subgraphs_41[npu_Subgraphs_41_input_num];
    for (i = 0; i < npu_Subgraphs_41_input_num; i++) {
        input_size_npu_Subgraphs_41[i] = csinn_tensor_byte_size(((struct csinn_session *)sess_npu_Subgraphs_41)->input[i]);
        input_aligned_npu_Subgraphs_41[i] = shl_mem_alloc_aligned(input_size_npu_Subgraphs_41[i], 0);
    }
    struct csinn_tensor* input_tensors_npu_Subgraphs_41[npu_Subgraphs_41_input_num];
    input_tensors_npu_Subgraphs_41[0] = csinn_alloc_tensor(NULL);
    input_tensors_npu_Subgraphs_41[0]->dim_count = 3;
    input_tensors_npu_Subgraphs_41[0]->dim[0] = 32;
    input_tensors_npu_Subgraphs_41[0]->dim[1] = 1280;
    input_tensors_npu_Subgraphs_41[0]->dim[2] = 10240;
    struct csinn_tensor* output_tensors_npu_Subgraphs_41[npu_Subgraphs_41_output_num];
    output_tensors_npu_Subgraphs_41[0] = csinn_alloc_tensor(NULL);
    output_tensors_npu_Subgraphs_41[0]->dim_count = 4;
    output_tensors_npu_Subgraphs_41[0]->dim[0] = 1;
    output_tensors_npu_Subgraphs_41[0]->dim[1] = 4;
    output_tensors_npu_Subgraphs_41[0]->dim[2] = 32;
    output_tensors_npu_Subgraphs_41[0]->dim[3] = 32;


    char filename_prefix[FILE_PREFIX_LENGTH] = {0};

    int result = system("./encoder");
    
    if (result == -1) {
        perror("system");
        return 1;
    }


    // 设置随机种子
    srand(42);
    LCMScheduler scheduler;
    int num_train_timesteps = 1000;
    float beta_start = 0.00085;
    float beta_end = 0.0120;
    int set_alpha_to_one = 1;
    const char* prediction_type = "epsilon";
    
    initialize_scheduler(&scheduler, num_train_timesteps, beta_start, beta_end, set_alpha_to_one, prediction_type);

    int num_inference_steps = 4;
    int lcm_origin_steps = 50;
    set_timesteps(&scheduler, num_inference_steps, lcm_origin_steps);

    float *latent = (float *)malloc(1 * 4 * 32 * 32 * sizeof(float));

    int BATCH_SIZE = 1;
    int CHANNELS = 4;
    int N_H = 32;
    int N_W = 32;

    // 填充数组
    for (int b = 0; b < BATCH_SIZE; ++b) {
        for (int c = 0; c < CHANNELS; ++c) {
            for (int h = 0; h < N_H; ++h) {
                for (int w = 0; w < N_W; ++w) {
                    latent[b * CHANNELS * N_H * N_W + c * N_H * N_W + h * N_W + w] = generate_normal_random();
                }
            }
        }
    }

    uint64_t start_time, end_time;
    for (int ii = 0; ii < 4; ii++) {
        /* set input */
        for (int j = 0; j < input_num; j++) {
            if (get_file_type(data_path[j]) != FILE_BIN) {
                printf("Please input binary files, since you compiled the model without preprocess.\n");
                return -1;
            }
            inputf[j] = (float*)get_binary_from_file(data_path[j], NULL);
            if (j == 0) {
                // npu_Subgraphs_0_input[j] = shl_ref_f32_to_input_dtype(j, inputf[j], sess_npu_Subgraphs_0);
                npu_Subgraphs_0_input[j] = shl_ref_f32_to_input_dtype(j, latent, sess_npu_Subgraphs_0);
            } else if (j == 1) {
                input_tensors_cpu_Subgraphs_0[1]->data = shl_c920_f32_to_input_dtype(1, inputf[j], sess_cpu_Subgraphs_0);
            } else if (j == 2) {
                input_tensors_cpu_Subgraphs_1[1]->data = shl_c920_f32_to_input_dtype(1, inputf[j], sess_cpu_Subgraphs_1);
                input_tensors_cpu_Subgraphs_2[1]->data = shl_c920_f32_to_input_dtype(1, inputf[j], sess_cpu_Subgraphs_2);
                input_tensors_cpu_Subgraphs_3[1]->data = shl_c920_f32_to_input_dtype(1, inputf[j], sess_cpu_Subgraphs_3);
                input_tensors_cpu_Subgraphs_4[1]->data = shl_c920_f32_to_input_dtype(1, inputf[j], sess_cpu_Subgraphs_4);
                input_tensors_cpu_Subgraphs_5[1]->data = shl_c920_f32_to_input_dtype(1, inputf[j], sess_cpu_Subgraphs_5);
                input_tensors_cpu_Subgraphs_6[1]->data = shl_c920_f32_to_input_dtype(1, inputf[j], sess_cpu_Subgraphs_6);
                input_tensors_cpu_Subgraphs_7[1]->data = shl_c920_f32_to_input_dtype(1, inputf[j], sess_cpu_Subgraphs_7);
                input_tensors_cpu_Subgraphs_8[1]->data = shl_c920_f32_to_input_dtype(1, inputf[j], sess_cpu_Subgraphs_8);
                input_tensors_cpu_Subgraphs_9[1]->data = shl_c920_f32_to_input_dtype(1, inputf[j], sess_cpu_Subgraphs_9);
            }
        }
        memcpy(input_aligned_npu_Subgraphs_0[0], npu_Subgraphs_0_input[0], input_size_npu_Subgraphs_0[0]);
        input_tensors_npu_Subgraphs_0[0]->data = input_aligned_npu_Subgraphs_0[0];

        start_time = shl_get_timespec();
        // npu_Subgraphs_0
        csinn_update_input_and_run_npu_Subgraphs_0(input_tensors_npu_Subgraphs_0, sess_npu_Subgraphs_0);
        //获取输出
        get_output(output_tensors_npu_Subgraphs_0, sess_npu_Subgraphs_0);
        int channels = 32;
        int height = 1;
        int width = 10240;
        float *scale = (float *)malloc(channels * sizeof(float));
        float *B = (float *)malloc(channels * sizeof(float));
        float *instance_normalization_output = (float *)malloc(height * channels * width * sizeof(float));
        for (int i = 0; i < channels; ++i) {
            scale[i] = 1;
            B[i] = 0;
        }
        instance_normalization(output_tensors_npu_Subgraphs_0[0]->data, scale, B, channels, height, width, instance_normalization_output);

        // npu_Subgraphs_1
        for (int j = 0; j < npu_Subgraphs_1_input_num; j++) {
            npu_Subgraphs_1_inputf[j] = instance_normalization_output;
            npu_Subgraphs_1_input[j] = shl_ref_f32_to_input_dtype(j, npu_Subgraphs_1_inputf[j], sess_npu_Subgraphs_1);
        }
        memcpy(input_aligned_npu_Subgraphs_1[0], npu_Subgraphs_1_input[0], input_size_npu_Subgraphs_1[0]);
        input_tensors_npu_Subgraphs_1[0]->data = input_aligned_npu_Subgraphs_1[0];
        csinn_update_input_and_run_npu_Subgraphs_1(input_tensors_npu_Subgraphs_1, sess_npu_Subgraphs_1);

        // cpu_Subgraphs_0
        get_output(output_tensors_npu_Subgraphs_1, sess_npu_Subgraphs_1);
        cpu_Subgraphs_0_inputf[0] = output_tensors_npu_Subgraphs_1[0]->data;
        input_tensors_cpu_Subgraphs_0[0]->data = shl_c920_f32_to_input_dtype(0, cpu_Subgraphs_0_inputf[0], sess_cpu_Subgraphs_0);
        csinn_update_input_and_run_cpu_Subgraphs_0(input_tensors_cpu_Subgraphs_0, sess_cpu_Subgraphs_0);
        get_output(output_tensors_cpu_Subgraphs_0, sess_cpu_Subgraphs_0);
        float *instance_normalization_output_2 = (float *)malloc(height * channels * width * sizeof(float));
        instance_normalization(output_tensors_cpu_Subgraphs_0[0]->data, scale, B, channels, height, width, instance_normalization_output_2);

        // npu_Subgraphs_2
        for (int j = 0; j < npu_Subgraphs_2_input_num; j++) {
            if (j == 1) {
                npu_Subgraphs_2_inputf[j] = instance_normalization_output_2;
                npu_Subgraphs_2_input[j] = shl_ref_f32_to_input_dtype(j, npu_Subgraphs_2_inputf[j], sess_npu_Subgraphs_2);
            } else if (j == 0) {
                npu_Subgraphs_2_inputf[j] = output_tensors_npu_Subgraphs_0[1]->data;
                npu_Subgraphs_2_input[j] = shl_ref_f32_to_input_dtype(j, npu_Subgraphs_2_inputf[j], sess_npu_Subgraphs_2);
            }
        }
        memcpy(input_aligned_npu_Subgraphs_2[0], npu_Subgraphs_2_input[0], input_size_npu_Subgraphs_2[0]);
        memcpy(input_aligned_npu_Subgraphs_2[1], npu_Subgraphs_2_input[1], input_size_npu_Subgraphs_2[1]);
        input_tensors_npu_Subgraphs_2[0]->data = input_aligned_npu_Subgraphs_2[0];
        input_tensors_npu_Subgraphs_2[1]->data = input_aligned_npu_Subgraphs_2[1];
        csinn_update_input_and_run_npu_Subgraphs_2(input_tensors_npu_Subgraphs_2, sess_npu_Subgraphs_2);
        get_output(output_tensors_npu_Subgraphs_2, sess_npu_Subgraphs_2);
        float *instance_normalization_output_3 = (float *)malloc(height * channels * width * sizeof(float));
        instance_normalization(output_tensors_npu_Subgraphs_2[1]->data, scale, B, channels, height, width, instance_normalization_output_3);

        // npu_Subgraphs_3
        npu_Subgraphs_3_inputf[0] = instance_normalization_output_3;
        npu_Subgraphs_3_input[0] = shl_ref_f32_to_input_dtype(0, npu_Subgraphs_3_inputf[0], sess_npu_Subgraphs_3);
        memcpy(input_aligned_npu_Subgraphs_3[0], npu_Subgraphs_3_input[0], input_size_npu_Subgraphs_3[0]);
        input_tensors_npu_Subgraphs_3[0]->data = input_aligned_npu_Subgraphs_3[0];
        csinn_update_input_and_run_npu_Subgraphs_3(input_tensors_npu_Subgraphs_3, sess_npu_Subgraphs_3);

        // cpu_Subgraphs_1
        get_output(output_tensors_npu_Subgraphs_3, sess_npu_Subgraphs_3);
        cpu_Subgraphs_1_inputf[0] = output_tensors_npu_Subgraphs_3[0]->data;
        input_tensors_cpu_Subgraphs_1[0]->data = shl_c920_f32_to_input_dtype(0, cpu_Subgraphs_1_inputf[0], sess_cpu_Subgraphs_1);
        csinn_update_input_and_run_cpu_Subgraphs_1(input_tensors_cpu_Subgraphs_1, sess_cpu_Subgraphs_1);

        // npu_Subgraphs_4
        get_output(output_tensors_cpu_Subgraphs_1, sess_cpu_Subgraphs_1);
        for (int j = 0; j < npu_Subgraphs_4_input_num; j++) {
            if (j == 1) {
                npu_Subgraphs_4_inputf[j] = output_tensors_npu_Subgraphs_2[0]->data;
                npu_Subgraphs_4_input[j] = shl_ref_f32_to_input_dtype(j, npu_Subgraphs_4_inputf[j], sess_npu_Subgraphs_4);
            } else if (j == 0) {
                npu_Subgraphs_4_inputf[j] = output_tensors_cpu_Subgraphs_1[0]->data;
                npu_Subgraphs_4_input[j] = shl_ref_f32_to_input_dtype(j, npu_Subgraphs_4_inputf[j], sess_npu_Subgraphs_4);
            }
        }
        memcpy(input_aligned_npu_Subgraphs_4[0], npu_Subgraphs_4_input[0], input_size_npu_Subgraphs_4[0]);
        memcpy(input_aligned_npu_Subgraphs_4[1], npu_Subgraphs_4_input[1], input_size_npu_Subgraphs_4[1]);
        input_tensors_npu_Subgraphs_4[0]->data = input_aligned_npu_Subgraphs_4[0];
        input_tensors_npu_Subgraphs_4[1]->data = input_aligned_npu_Subgraphs_4[1];
        csinn_update_input_and_run_npu_Subgraphs_4(input_tensors_npu_Subgraphs_4, sess_npu_Subgraphs_4);

        // npu_Subgraphs_5
        width = 2560;
// printf("npu_Subgraphs_5\n");
        get_output(output_tensors_npu_Subgraphs_4, sess_npu_Subgraphs_4);
        float *instance_normalization_output_4 = (float *)malloc(height * channels * width * sizeof(float));
        instance_normalization(output_tensors_npu_Subgraphs_4[2]->data, scale, B, channels, height, width, instance_normalization_output_4);
        npu_Subgraphs_5_inputf[0] = instance_normalization_output_4;
        npu_Subgraphs_5_input[0] = shl_ref_f32_to_input_dtype(0, npu_Subgraphs_5_inputf[0], sess_npu_Subgraphs_5);
        memcpy(input_aligned_npu_Subgraphs_5[0], npu_Subgraphs_5_input[0], input_size_npu_Subgraphs_5[0]);
        input_tensors_npu_Subgraphs_5[0]->data = input_aligned_npu_Subgraphs_5[0];

        npu_Subgraphs_5_inputf[1] = output_tensors_cpu_Subgraphs_0[1]->data;
        npu_Subgraphs_5_input[1] = shl_ref_f32_to_input_dtype(1, npu_Subgraphs_5_inputf[1], sess_npu_Subgraphs_5);
        memcpy(input_aligned_npu_Subgraphs_5[1], npu_Subgraphs_5_input[1], input_size_npu_Subgraphs_5[1]);
        input_tensors_npu_Subgraphs_5[1]->data = input_aligned_npu_Subgraphs_5[1];

        csinn_update_input_and_run_npu_Subgraphs_5(input_tensors_npu_Subgraphs_5, sess_npu_Subgraphs_5);

        // npu_Subgraphs_6
        width = 5120;
        get_output(output_tensors_npu_Subgraphs_5, sess_npu_Subgraphs_5);
        float *instance_normalization_output_5 = (float *)malloc(height * channels * width * sizeof(float));
        instance_normalization(output_tensors_npu_Subgraphs_5[0]->data, scale, B, channels, height, width, instance_normalization_output_5);
        npu_Subgraphs_6_inputf[1] = instance_normalization_output_5;
        npu_Subgraphs_6_input[1] = shl_ref_f32_to_input_dtype(1, npu_Subgraphs_6_inputf[1], sess_npu_Subgraphs_6);
        memcpy(input_aligned_npu_Subgraphs_6[1], npu_Subgraphs_6_input[1], input_size_npu_Subgraphs_6[1]);
        input_tensors_npu_Subgraphs_6[1]->data = input_aligned_npu_Subgraphs_6[1];

        npu_Subgraphs_6_inputf[0] = output_tensors_npu_Subgraphs_4[3]->data;
        npu_Subgraphs_6_input[0] = shl_ref_f32_to_input_dtype(0, npu_Subgraphs_6_inputf[0], sess_npu_Subgraphs_6);
        memcpy(input_aligned_npu_Subgraphs_6[0], npu_Subgraphs_6_input[0], input_size_npu_Subgraphs_6[0]);
        input_tensors_npu_Subgraphs_6[0]->data = input_aligned_npu_Subgraphs_6[0];

        csinn_update_input_and_run_npu_Subgraphs_6(input_tensors_npu_Subgraphs_6, sess_npu_Subgraphs_6);

        width = 5120;
        get_output(output_tensors_npu_Subgraphs_6, sess_npu_Subgraphs_6);
        float *instance_normalization_output_6 = (float *)malloc(height * channels * width * sizeof(float));
        instance_normalization(output_tensors_npu_Subgraphs_6[0]->data, scale, B, channels, height, width, instance_normalization_output_6);

        // npu_Subgraphs_7
        npu_Subgraphs_7_inputf[0] = instance_normalization_output_6;
        npu_Subgraphs_7_input[0] = shl_ref_f32_to_input_dtype(0, npu_Subgraphs_7_inputf[0], sess_npu_Subgraphs_7);
        memcpy(input_aligned_npu_Subgraphs_7[0], npu_Subgraphs_7_input[0], input_size_npu_Subgraphs_7[0]);
        input_tensors_npu_Subgraphs_7[0]->data = input_aligned_npu_Subgraphs_7[0];

        csinn_update_input_and_run_npu_Subgraphs_7(input_tensors_npu_Subgraphs_7, sess_npu_Subgraphs_7);
        get_output(output_tensors_npu_Subgraphs_7, sess_npu_Subgraphs_7);

        // cpu_Subgraphs_2
        cpu_Subgraphs_2_inputf[0] = output_tensors_npu_Subgraphs_7[0]->data;
        input_tensors_cpu_Subgraphs_2[0]->data = shl_c920_f32_to_input_dtype(0, cpu_Subgraphs_2_inputf[0], sess_cpu_Subgraphs_2);
        csinn_update_input_and_run_cpu_Subgraphs_2(input_tensors_cpu_Subgraphs_2, sess_cpu_Subgraphs_2);
        get_output(output_tensors_cpu_Subgraphs_2, sess_cpu_Subgraphs_2);

        // npu_Subgraphs_8
        npu_Subgraphs_8_inputf[0] = output_tensors_cpu_Subgraphs_2[0]->data;
        npu_Subgraphs_8_input[0] = shl_ref_f32_to_input_dtype(0, npu_Subgraphs_8_inputf[0], sess_npu_Subgraphs_8);
        memcpy(input_aligned_npu_Subgraphs_8[0], npu_Subgraphs_8_input[0], input_size_npu_Subgraphs_8[0]);
        input_tensors_npu_Subgraphs_8[0]->data = input_aligned_npu_Subgraphs_8[0];

        npu_Subgraphs_8_inputf[1] = output_tensors_npu_Subgraphs_6[1]->data;
        npu_Subgraphs_8_input[1] = shl_ref_f32_to_input_dtype(1, npu_Subgraphs_8_inputf[1], sess_npu_Subgraphs_8);
        memcpy(input_aligned_npu_Subgraphs_8[1], npu_Subgraphs_8_input[1], input_size_npu_Subgraphs_8[1]);
        input_tensors_npu_Subgraphs_8[1]->data = input_aligned_npu_Subgraphs_8[1];

        csinn_update_input_and_run_npu_Subgraphs_8(input_tensors_npu_Subgraphs_8, sess_npu_Subgraphs_8);
        
        width = 1280;
        get_output(output_tensors_npu_Subgraphs_8, sess_npu_Subgraphs_8);
        float *instance_normalization_output_7 = (float *)malloc(height * channels * width * sizeof(float));
        instance_normalization(output_tensors_npu_Subgraphs_8[2]->data, scale, B, channels, height, width, instance_normalization_output_7);
        
        // npu_Subgraphs_9
        npu_Subgraphs_9_inputf[0] = instance_normalization_output_7;
        npu_Subgraphs_9_input[0] = shl_ref_f32_to_input_dtype(0, npu_Subgraphs_9_inputf[0], sess_npu_Subgraphs_9);
        memcpy(input_aligned_npu_Subgraphs_9[0], npu_Subgraphs_9_input[0], input_size_npu_Subgraphs_9[0]);
        input_tensors_npu_Subgraphs_9[0]->data = input_aligned_npu_Subgraphs_9[0];

        npu_Subgraphs_9_inputf[1] = output_tensors_cpu_Subgraphs_0[1]->data;
        npu_Subgraphs_9_input[1] = shl_ref_f32_to_input_dtype(1, npu_Subgraphs_9_inputf[1], sess_npu_Subgraphs_9);
        memcpy(input_aligned_npu_Subgraphs_9[1], npu_Subgraphs_9_input[1], input_size_npu_Subgraphs_9[1]);
        input_tensors_npu_Subgraphs_9[1]->data = input_aligned_npu_Subgraphs_9[1];

        csinn_update_input_and_run_npu_Subgraphs_9(input_tensors_npu_Subgraphs_9, sess_npu_Subgraphs_9);

        width = 2560;
        get_output(output_tensors_npu_Subgraphs_9, sess_npu_Subgraphs_9);
        float *instance_normalization_output_8 = (float *)malloc(height * channels * width * sizeof(float));
        instance_normalization(output_tensors_npu_Subgraphs_9[0]->data, scale, B, channels, height, width, instance_normalization_output_8);

        // npu_Subgraphs_10
        npu_Subgraphs_10_inputf[0] = output_tensors_npu_Subgraphs_8[1]->data;
        npu_Subgraphs_10_input[0] = shl_ref_f32_to_input_dtype(0, npu_Subgraphs_10_inputf[0], sess_npu_Subgraphs_10);
        memcpy(input_aligned_npu_Subgraphs_10[0], npu_Subgraphs_10_input[0], input_size_npu_Subgraphs_10[0]);
        input_tensors_npu_Subgraphs_10[0]->data = input_aligned_npu_Subgraphs_10[0];

        npu_Subgraphs_10_inputf[1] = instance_normalization_output_7;
        npu_Subgraphs_10_input[1] = shl_ref_f32_to_input_dtype(1, npu_Subgraphs_10_inputf[1], sess_npu_Subgraphs_10);
        memcpy(input_aligned_npu_Subgraphs_10[1], npu_Subgraphs_10_input[1], input_size_npu_Subgraphs_10[1]);
        input_tensors_npu_Subgraphs_10[1]->data = input_aligned_npu_Subgraphs_10[1];

        csinn_update_input_and_run_npu_Subgraphs_10(input_tensors_npu_Subgraphs_10, sess_npu_Subgraphs_10);

        width = 2560;
        get_output(output_tensors_npu_Subgraphs_10, sess_npu_Subgraphs_10);
        float *instance_normalization_output_9 = (float *)malloc(height * channels * width * sizeof(float));
        instance_normalization(output_tensors_npu_Subgraphs_10[0]->data, scale, B, channels, height, width, instance_normalization_output_9);

        // npu_Subgraphs_11
        npu_Subgraphs_11_inputf[0] = instance_normalization_output_9;
        npu_Subgraphs_11_input[0] = shl_ref_f32_to_input_dtype(0, npu_Subgraphs_11_inputf[0], sess_npu_Subgraphs_11);
        memcpy(input_aligned_npu_Subgraphs_11[0], npu_Subgraphs_11_input[0], input_size_npu_Subgraphs_11[0]);
        input_tensors_npu_Subgraphs_11[0]->data = input_aligned_npu_Subgraphs_11[0];

        csinn_update_input_and_run_npu_Subgraphs_11(input_tensors_npu_Subgraphs_11, sess_npu_Subgraphs_11);
        get_output(output_tensors_npu_Subgraphs_11, sess_npu_Subgraphs_11);

        // cpu_Subgraphs_3
        cpu_Subgraphs_3_inputf[0] = output_tensors_npu_Subgraphs_11[0]->data;
        input_tensors_cpu_Subgraphs_3[0]->data = shl_c920_f32_to_input_dtype(0, cpu_Subgraphs_3_inputf[0], sess_cpu_Subgraphs_3);
        csinn_update_input_and_run_cpu_Subgraphs_3(input_tensors_cpu_Subgraphs_3, sess_cpu_Subgraphs_3);
        get_output(output_tensors_cpu_Subgraphs_3, sess_cpu_Subgraphs_3);

        // npu_Subgraphs_12
        npu_Subgraphs_12_inputf[0] = output_tensors_cpu_Subgraphs_3[0]->data;
        npu_Subgraphs_12_input[0] = shl_ref_f32_to_input_dtype(0, npu_Subgraphs_12_inputf[0], sess_npu_Subgraphs_12);
        memcpy(input_aligned_npu_Subgraphs_12[0], npu_Subgraphs_12_input[0], input_size_npu_Subgraphs_12[0]);
        input_tensors_npu_Subgraphs_12[0]->data = input_aligned_npu_Subgraphs_12[0];

        npu_Subgraphs_12_inputf[1] = output_tensors_npu_Subgraphs_10[1]->data;
        npu_Subgraphs_12_input[1] = shl_ref_f32_to_input_dtype(1, npu_Subgraphs_12_inputf[1], sess_npu_Subgraphs_12);
        memcpy(input_aligned_npu_Subgraphs_12[1], npu_Subgraphs_12_input[1], input_size_npu_Subgraphs_12[1]);
        input_tensors_npu_Subgraphs_12[1]->data = input_aligned_npu_Subgraphs_12[1];

        csinn_update_input_and_run_npu_Subgraphs_12(input_tensors_npu_Subgraphs_12, sess_npu_Subgraphs_12);

        width = 5120;
        get_output(output_tensors_npu_Subgraphs_12, sess_npu_Subgraphs_12);
        float *instance_normalization_output_10 = (float *)malloc(height * channels * width * sizeof(float));
        instance_normalization(output_tensors_npu_Subgraphs_12[0]->data, scale, B, channels, height, width, instance_normalization_output_10);

        // npu_Subgraphs_13
        npu_Subgraphs_13_inputf[0] = instance_normalization_output_10;
        npu_Subgraphs_13_input[0] = shl_ref_f32_to_input_dtype(0, npu_Subgraphs_13_inputf[0], sess_npu_Subgraphs_13);
        memcpy(input_aligned_npu_Subgraphs_13[0], npu_Subgraphs_13_input[0], input_size_npu_Subgraphs_13[0]);
        input_tensors_npu_Subgraphs_13[0]->data = input_aligned_npu_Subgraphs_13[0];

        npu_Subgraphs_13_inputf[1] = output_tensors_cpu_Subgraphs_0[1]->data;
        npu_Subgraphs_13_input[1] = shl_ref_f32_to_input_dtype(1, npu_Subgraphs_13_inputf[1], sess_npu_Subgraphs_13);
        memcpy(input_aligned_npu_Subgraphs_13[1], npu_Subgraphs_13_input[1], input_size_npu_Subgraphs_13[1]);
        input_tensors_npu_Subgraphs_13[1]->data = input_aligned_npu_Subgraphs_13[1];

        csinn_update_input_and_run_npu_Subgraphs_13(input_tensors_npu_Subgraphs_13, sess_npu_Subgraphs_13);

        width = 2560;
        get_output(output_tensors_npu_Subgraphs_13, sess_npu_Subgraphs_13);
        float *instance_normalization_output_11 = (float *)malloc(height * channels * width * sizeof(float));
        instance_normalization(output_tensors_npu_Subgraphs_13[0]->data, scale, B, channels, height, width, instance_normalization_output_11);

        // npu_Subgraphs_14
        npu_Subgraphs_14_inputf[0] = output_tensors_npu_Subgraphs_12[1]->data;
        npu_Subgraphs_14_input[0] = shl_ref_f32_to_input_dtype(0, npu_Subgraphs_14_inputf[0], sess_npu_Subgraphs_14);
        memcpy(input_aligned_npu_Subgraphs_14[0], npu_Subgraphs_14_input[0], input_size_npu_Subgraphs_14[0]);
        input_tensors_npu_Subgraphs_14[0]->data = input_aligned_npu_Subgraphs_14[0];

        npu_Subgraphs_14_inputf[1] = instance_normalization_output_11;
        npu_Subgraphs_14_input[1] = shl_ref_f32_to_input_dtype(1, npu_Subgraphs_14_inputf[1], sess_npu_Subgraphs_14);
        memcpy(input_aligned_npu_Subgraphs_14[1], npu_Subgraphs_14_input[1], input_size_npu_Subgraphs_14[1]);
        input_tensors_npu_Subgraphs_14[1]->data = input_aligned_npu_Subgraphs_14[1];

        csinn_update_input_and_run_npu_Subgraphs_14(input_tensors_npu_Subgraphs_14, sess_npu_Subgraphs_14);

        width = 2560;
        get_output(output_tensors_npu_Subgraphs_14, sess_npu_Subgraphs_14);
        float *instance_normalization_output_12 = (float *)malloc(height * channels * width * sizeof(float));
        instance_normalization(output_tensors_npu_Subgraphs_14[0]->data, scale, B, channels, height, width, instance_normalization_output_12);

// printf("-----------npu_Subgraphs_15------------\r\n");
        // npu_Subgraphs_15
        npu_Subgraphs_15_inputf[0] = instance_normalization_output_12;
        npu_Subgraphs_15_input[0] = shl_ref_f32_to_input_dtype(0, npu_Subgraphs_15_inputf[0], sess_npu_Subgraphs_15);
        memcpy(input_aligned_npu_Subgraphs_15[0], npu_Subgraphs_15_input[0], input_size_npu_Subgraphs_15[0]);
        input_tensors_npu_Subgraphs_15[0]->data = input_aligned_npu_Subgraphs_15[0];

        csinn_update_input_and_run_npu_Subgraphs_15(input_tensors_npu_Subgraphs_15, sess_npu_Subgraphs_15);
        get_output(output_tensors_npu_Subgraphs_15, sess_npu_Subgraphs_15);

        // cpu_Subgraphs_4
        cpu_Subgraphs_4_inputf[0] = output_tensors_npu_Subgraphs_15[0]->data;
        input_tensors_cpu_Subgraphs_4[0]->data = shl_c920_f32_to_input_dtype(0, cpu_Subgraphs_4_inputf[0], sess_cpu_Subgraphs_4);
        csinn_update_input_and_run_cpu_Subgraphs_4(input_tensors_cpu_Subgraphs_4, sess_cpu_Subgraphs_4);
        get_output(output_tensors_cpu_Subgraphs_4, sess_cpu_Subgraphs_4);

// printf("-----------npu_Subgraphs_16------------\r\n");
        // npu_Subgraphs_16
        npu_Subgraphs_16_inputf[0] = output_tensors_cpu_Subgraphs_4[0]->data;
        npu_Subgraphs_16_input[0] = shl_ref_f32_to_input_dtype(0, npu_Subgraphs_16_inputf[0], sess_npu_Subgraphs_16);
        memcpy(input_aligned_npu_Subgraphs_16[0], npu_Subgraphs_16_input[0], input_size_npu_Subgraphs_16[0]);
        input_tensors_npu_Subgraphs_16[0]->data = input_aligned_npu_Subgraphs_16[0];

        npu_Subgraphs_16_inputf[1] = output_tensors_npu_Subgraphs_14[1]->data;
        npu_Subgraphs_16_input[1] = shl_ref_f32_to_input_dtype(1, npu_Subgraphs_16_inputf[1], sess_npu_Subgraphs_16);
        memcpy(input_aligned_npu_Subgraphs_16[1], npu_Subgraphs_16_input[1], input_size_npu_Subgraphs_16[1]);
        input_tensors_npu_Subgraphs_16[1]->data = input_aligned_npu_Subgraphs_16[1];

        csinn_update_input_and_run_npu_Subgraphs_16(input_tensors_npu_Subgraphs_16, sess_npu_Subgraphs_16);

        get_output(output_tensors_npu_Subgraphs_16, sess_npu_Subgraphs_16);

        // 参数定义
        float* concat_result_1 = (float*)malloc(1 * 1920 * 8 * 8 * sizeof(float));
        concat(output_tensors_npu_Subgraphs_16[0]->data, 1, 1280, 8, 8, output_tensors_npu_Subgraphs_8[1]->data, 640, concat_result_1, 1920);
        float *reshape_output_1 = (float *)malloc(1 * 32 * 3840 * sizeof(float));
        reshape(concat_result_1, reshape_output_1, 1, 1920, 8, 8, 32, 3840);

        width = 3840;
        float *instance_normalization_output_13 = (float *)malloc(height * channels * width * sizeof(float));
        instance_normalization(reshape_output_1, scale, B, channels, height, width, instance_normalization_output_13);

// printf("-----------npu_Subgraphs_17------------\r\n");
        // npu_Subgraphs_17
        npu_Subgraphs_17_inputf[0] = instance_normalization_output_13;
        npu_Subgraphs_17_input[0] = shl_ref_f32_to_input_dtype(0, npu_Subgraphs_17_inputf[0], sess_npu_Subgraphs_17);
        memcpy(input_aligned_npu_Subgraphs_17[0], npu_Subgraphs_17_input[0], input_size_npu_Subgraphs_17[0]);
        input_tensors_npu_Subgraphs_17[0]->data = input_aligned_npu_Subgraphs_17[0];

        npu_Subgraphs_17_inputf[1] = output_tensors_cpu_Subgraphs_0[1]->data;
        npu_Subgraphs_17_input[1] = shl_ref_f32_to_input_dtype(1, npu_Subgraphs_17_inputf[1], sess_npu_Subgraphs_17);
        memcpy(input_aligned_npu_Subgraphs_17[1], npu_Subgraphs_17_input[1], input_size_npu_Subgraphs_17[1]);
        input_tensors_npu_Subgraphs_17[1]->data = input_aligned_npu_Subgraphs_17[1];

        csinn_update_input_and_run_npu_Subgraphs_17(input_tensors_npu_Subgraphs_17, sess_npu_Subgraphs_17);

        width = 2560;
        get_output(output_tensors_npu_Subgraphs_17, sess_npu_Subgraphs_17);
        float *instance_normalization_output_14 = (float *)malloc(height * channels * width * sizeof(float));
        instance_normalization(output_tensors_npu_Subgraphs_17[0]->data, scale, B, channels, height, width, instance_normalization_output_14);

// printf("-----------npu_Subgraphs_18------------\r\n");
        // npu_Subgraphs_18
        npu_Subgraphs_18_inputf[0] = concat_result_1;
        npu_Subgraphs_18_input[0] = shl_ref_f32_to_input_dtype(0, npu_Subgraphs_18_inputf[0], sess_npu_Subgraphs_18);
        memcpy(input_aligned_npu_Subgraphs_18[0], npu_Subgraphs_18_input[0], input_size_npu_Subgraphs_18[0]);
        input_tensors_npu_Subgraphs_18[0]->data = input_aligned_npu_Subgraphs_18[0];

        npu_Subgraphs_18_inputf[1] = instance_normalization_output_14;
        npu_Subgraphs_18_input[1] = shl_ref_f32_to_input_dtype(1, npu_Subgraphs_18_inputf[1], sess_npu_Subgraphs_18);
        memcpy(input_aligned_npu_Subgraphs_18[1], npu_Subgraphs_18_input[1], input_size_npu_Subgraphs_18[1]);
        input_tensors_npu_Subgraphs_18[1]->data = input_aligned_npu_Subgraphs_18[1];

        csinn_update_input_and_run_npu_Subgraphs_18(input_tensors_npu_Subgraphs_18, sess_npu_Subgraphs_18);

        width = 2560;
        get_output(output_tensors_npu_Subgraphs_18, sess_npu_Subgraphs_18);
        float *instance_normalization_output_15 = (float *)malloc(height * channels * width * sizeof(float));
        instance_normalization(output_tensors_npu_Subgraphs_18[0]->data, scale, B, channels, height, width, instance_normalization_output_15);

// printf("-----------npu_Subgraphs_19------------\r\n");
        // npu_Subgraphs_19
        npu_Subgraphs_19_inputf[0] = instance_normalization_output_15;
        npu_Subgraphs_19_input[0] = shl_ref_f32_to_input_dtype(0, npu_Subgraphs_19_inputf[0], sess_npu_Subgraphs_19);
        memcpy(input_aligned_npu_Subgraphs_19[0], npu_Subgraphs_19_input[0], input_size_npu_Subgraphs_19[0]);
        input_tensors_npu_Subgraphs_19[0]->data = input_aligned_npu_Subgraphs_19[0];

        csinn_update_input_and_run_npu_Subgraphs_19(input_tensors_npu_Subgraphs_19, sess_npu_Subgraphs_19);
        get_output(output_tensors_npu_Subgraphs_19, sess_npu_Subgraphs_19);

// printf("-----------cpu_Subgraphs_5------------\r\n");
        // cpu_Subgraphs_5
        cpu_Subgraphs_5_inputf[0] = output_tensors_npu_Subgraphs_19[0]->data;
        input_tensors_cpu_Subgraphs_5[0]->data = shl_c920_f32_to_input_dtype(0, cpu_Subgraphs_5_inputf[0], sess_cpu_Subgraphs_5);
        csinn_update_input_and_run_cpu_Subgraphs_5(input_tensors_cpu_Subgraphs_5, sess_cpu_Subgraphs_5);
        get_output(output_tensors_cpu_Subgraphs_5, sess_cpu_Subgraphs_5);

// printf("-----------npu_Subgraphs_20------------\r\n");
        // npu_Subgraphs_20
        npu_Subgraphs_20_inputf[0] = output_tensors_cpu_Subgraphs_5[0]->data;
        npu_Subgraphs_20_input[0] = shl_ref_f32_to_input_dtype(0, npu_Subgraphs_20_inputf[0], sess_npu_Subgraphs_20);
        memcpy(input_aligned_npu_Subgraphs_20[0], npu_Subgraphs_20_input[0], input_size_npu_Subgraphs_20[0]);
        input_tensors_npu_Subgraphs_20[0]->data = input_aligned_npu_Subgraphs_20[0];

        npu_Subgraphs_20_inputf[1] = output_tensors_npu_Subgraphs_18[1]->data;
        npu_Subgraphs_20_input[1] = shl_ref_f32_to_input_dtype(1, npu_Subgraphs_20_inputf[1], sess_npu_Subgraphs_20);
        memcpy(input_aligned_npu_Subgraphs_20[1], npu_Subgraphs_20_input[1], input_size_npu_Subgraphs_20[1]);
        input_tensors_npu_Subgraphs_20[1]->data = input_aligned_npu_Subgraphs_20[1];

        csinn_update_input_and_run_npu_Subgraphs_20(input_tensors_npu_Subgraphs_20, sess_npu_Subgraphs_20);

        get_output(output_tensors_npu_Subgraphs_20, sess_npu_Subgraphs_20);

        // 参数定义
        float* concat_result_2 = (float*)malloc(1 * 1920 * 16 * 16 * sizeof(float));
        concat(output_tensors_npu_Subgraphs_20[0]->data, 1, 1280, 16, 16, output_tensors_npu_Subgraphs_8[0]->data, 640, concat_result_2, 1920);
        float *reshape_output_2 = (float *)malloc(1 * 32 * 15360 * sizeof(float));
        reshape(concat_result_2, reshape_output_2, 1, 1920, 16, 16, 32, 15360);

        width = 15360;
        float *instance_normalization_output_16 = (float *)malloc(height * channels * width * sizeof(float));
        instance_normalization(reshape_output_2, scale, B, channels, height, width, instance_normalization_output_16);

// printf("-----------npu_Subgraphs_21------------\r\n");
        // npu_Subgraphs_21
        npu_Subgraphs_21_inputf[0] = output_tensors_cpu_Subgraphs_0[1]->data;
        npu_Subgraphs_21_input[0] = shl_ref_f32_to_input_dtype(0, npu_Subgraphs_21_inputf[0], sess_npu_Subgraphs_21);
        memcpy(input_aligned_npu_Subgraphs_21[0], npu_Subgraphs_21_input[0], input_size_npu_Subgraphs_21[0]);
        input_tensors_npu_Subgraphs_21[0]->data = input_aligned_npu_Subgraphs_21[0];

        csinn_update_input_and_run_npu_Subgraphs_21(input_tensors_npu_Subgraphs_21, sess_npu_Subgraphs_21);
        get_output(output_tensors_npu_Subgraphs_21, sess_npu_Subgraphs_21);

// printf("-----------npu_Subgraphs_22------------\r\n");
        // npu_Subgraphs_22
        npu_Subgraphs_22_inputf[0] = instance_normalization_output_16;
        npu_Subgraphs_22_input[0] = shl_ref_f32_to_input_dtype(0, npu_Subgraphs_22_inputf[0], sess_npu_Subgraphs_22);
        memcpy(input_aligned_npu_Subgraphs_22[0], npu_Subgraphs_22_input[0], input_size_npu_Subgraphs_22[0]);
        input_tensors_npu_Subgraphs_22[0]->data = input_aligned_npu_Subgraphs_22[0];

        csinn_update_input_and_run_npu_Subgraphs_22(input_tensors_npu_Subgraphs_22, sess_npu_Subgraphs_22);
        get_output(output_tensors_npu_Subgraphs_22, sess_npu_Subgraphs_22);

        float *add_result_1 = (float *)malloc(1 * 640 * 16 * 16 * sizeof(float));
        add_arrays(output_tensors_npu_Subgraphs_22[0]->data, output_tensors_npu_Subgraphs_21[0]->data, add_result_1, 1, 640, 16, 16);
        float *reshape_output_3 = (float *)malloc(1 * 32 * 5120 * sizeof(float));
        reshape(add_result_1, reshape_output_3, 1, 640, 16, 16, 32, 5120);

        width = 5120;
        float *instance_normalization_output_17 = (float *)malloc(height * channels * width * sizeof(float));
        instance_normalization(reshape_output_3, scale, B, channels, height, width, instance_normalization_output_17);

// printf("-----------npu_Subgraphs_23------------\r\n");
        // npu_Subgraphs_23
        npu_Subgraphs_23_inputf[0] = concat_result_2;
        npu_Subgraphs_23_input[0] = shl_ref_f32_to_input_dtype(0, npu_Subgraphs_23_inputf[0], sess_npu_Subgraphs_23);
        memcpy(input_aligned_npu_Subgraphs_23[0], npu_Subgraphs_23_input[0], input_size_npu_Subgraphs_23[0]);
        input_tensors_npu_Subgraphs_23[0]->data = input_aligned_npu_Subgraphs_23[0];

        npu_Subgraphs_23_inputf[1] = instance_normalization_output_17;
        npu_Subgraphs_23_input[1] = shl_ref_f32_to_input_dtype(1, npu_Subgraphs_23_inputf[1], sess_npu_Subgraphs_23);
        memcpy(input_aligned_npu_Subgraphs_23[1], npu_Subgraphs_23_input[1], input_size_npu_Subgraphs_23[1]);
        input_tensors_npu_Subgraphs_23[1]->data = input_aligned_npu_Subgraphs_23[1];

        csinn_update_input_and_run_npu_Subgraphs_23(input_tensors_npu_Subgraphs_23, sess_npu_Subgraphs_23);

        width = 5120;
        get_output(output_tensors_npu_Subgraphs_23, sess_npu_Subgraphs_23);
        float *instance_normalization_output_18 = (float *)malloc(height * channels * width * sizeof(float));
        instance_normalization(output_tensors_npu_Subgraphs_23[0]->data, scale, B, channels, height, width, instance_normalization_output_18);

// printf("-----------npu_Subgraphs_24------------\r\n");
        // npu_Subgraphs_24
        npu_Subgraphs_24_inputf[0] = instance_normalization_output_18;
        npu_Subgraphs_24_input[0] = shl_ref_f32_to_input_dtype(0, npu_Subgraphs_24_inputf[0], sess_npu_Subgraphs_24);
        memcpy(input_aligned_npu_Subgraphs_24[0], npu_Subgraphs_24_input[0], input_size_npu_Subgraphs_24[0]);
        input_tensors_npu_Subgraphs_24[0]->data = input_aligned_npu_Subgraphs_24[0];

        csinn_update_input_and_run_npu_Subgraphs_24(input_tensors_npu_Subgraphs_24, sess_npu_Subgraphs_24);
        get_output(output_tensors_npu_Subgraphs_24, sess_npu_Subgraphs_24);

// printf("-----------cpu_Subgraphs_6------------\r\n");
        // cpu_Subgraphs_6
        cpu_Subgraphs_6_inputf[0] = output_tensors_npu_Subgraphs_24[0]->data;
        input_tensors_cpu_Subgraphs_6[0]->data = shl_c920_f32_to_input_dtype(0, cpu_Subgraphs_6_inputf[0], sess_cpu_Subgraphs_6);
        csinn_update_input_and_run_cpu_Subgraphs_6(input_tensors_cpu_Subgraphs_6, sess_cpu_Subgraphs_6);
        get_output(output_tensors_cpu_Subgraphs_6, sess_cpu_Subgraphs_6);

// printf("-----------npu_Subgraphs_25------------\r\n");
        // npu_Subgraphs_25
        npu_Subgraphs_25_inputf[0] = output_tensors_cpu_Subgraphs_6[0]->data;
        npu_Subgraphs_25_input[0] = shl_ref_f32_to_input_dtype(0, npu_Subgraphs_25_inputf[0], sess_npu_Subgraphs_25);
        memcpy(input_aligned_npu_Subgraphs_25[0], npu_Subgraphs_25_input[0], input_size_npu_Subgraphs_25[0]);
        input_tensors_npu_Subgraphs_25[0]->data = input_aligned_npu_Subgraphs_25[0];

        npu_Subgraphs_25_inputf[1] = output_tensors_npu_Subgraphs_23[1]->data;
        npu_Subgraphs_25_input[1] = shl_ref_f32_to_input_dtype(1, npu_Subgraphs_25_inputf[1], sess_npu_Subgraphs_25);
        memcpy(input_aligned_npu_Subgraphs_25[1], npu_Subgraphs_25_input[1], input_size_npu_Subgraphs_25[1]);
        input_tensors_npu_Subgraphs_25[1]->data = input_aligned_npu_Subgraphs_25[1];

        csinn_update_input_and_run_npu_Subgraphs_25(input_tensors_npu_Subgraphs_25, sess_npu_Subgraphs_25);

        get_output(output_tensors_npu_Subgraphs_25, sess_npu_Subgraphs_25);

        // 参数定义
        float* concat_result_3 = (float*)malloc(1 * 960 * 16 * 16 * sizeof(float));
        concat(output_tensors_npu_Subgraphs_25[0]->data, 1, 640, 16, 16, output_tensors_npu_Subgraphs_4[1]->data, 320, concat_result_3, 960);
        float *reshape_output_4 = (float *)malloc(1 * 32 * 7680 * sizeof(float));
        reshape(concat_result_3, reshape_output_4, 1, 960, 16, 16, 32, 7680);

        width = 7680;
        float *instance_normalization_output_19 = (float *)malloc(height * channels * width * sizeof(float));
        instance_normalization(reshape_output_4, scale, B, channels, height, width, instance_normalization_output_19);

// printf("-----------npu_Subgraphs_26------------\r\n");
        // npu_Subgraphs_26
        npu_Subgraphs_26_inputf[0] = output_tensors_cpu_Subgraphs_0[1]->data;
        npu_Subgraphs_26_input[0] = shl_ref_f32_to_input_dtype(0, npu_Subgraphs_26_inputf[0], sess_npu_Subgraphs_26);
        memcpy(input_aligned_npu_Subgraphs_26[0], npu_Subgraphs_26_input[0], input_size_npu_Subgraphs_26[0]);
        input_tensors_npu_Subgraphs_26[0]->data = input_aligned_npu_Subgraphs_26[0];

        csinn_update_input_and_run_npu_Subgraphs_26(input_tensors_npu_Subgraphs_26, sess_npu_Subgraphs_26);
        get_output(output_tensors_npu_Subgraphs_26, sess_npu_Subgraphs_26);

// printf("-----------npu_Subgraphs_27------------\r\n");
        // npu_Subgraphs_27
        npu_Subgraphs_27_inputf[0] = instance_normalization_output_19;
        npu_Subgraphs_27_input[0] = shl_ref_f32_to_input_dtype(0, npu_Subgraphs_27_inputf[0], sess_npu_Subgraphs_27);
        memcpy(input_aligned_npu_Subgraphs_27[0], npu_Subgraphs_27_input[0], input_size_npu_Subgraphs_27[0]);
        input_tensors_npu_Subgraphs_27[0]->data = input_aligned_npu_Subgraphs_27[0];

        csinn_update_input_and_run_npu_Subgraphs_27(input_tensors_npu_Subgraphs_27, sess_npu_Subgraphs_27);
        get_output(output_tensors_npu_Subgraphs_27, sess_npu_Subgraphs_27);

        float *add_result_2 = (float *)malloc(1 * 640 * 16 * 16 * sizeof(float));
        add_arrays(output_tensors_npu_Subgraphs_27[0]->data, output_tensors_npu_Subgraphs_26[0]->data, add_result_2, 1, 640, 16, 16);
        float *reshape_output_5 = (float *)malloc(1 * 32 * 5120 * sizeof(float));
        reshape(add_result_2, reshape_output_5, 1, 640, 16, 16, 32, 5120);

        width = 5120;
        float *instance_normalization_output_20 = (float *)malloc(height * channels * width * sizeof(float));
        instance_normalization(reshape_output_5, scale, B, channels, height, width, instance_normalization_output_20);

// printf("-----------npu_Subgraphs_28------------\r\n");
        // npu_Subgraphs_28
        npu_Subgraphs_28_inputf[0] = concat_result_3;
        npu_Subgraphs_28_input[0] = shl_ref_f32_to_input_dtype(0, npu_Subgraphs_28_inputf[0], sess_npu_Subgraphs_28);
        memcpy(input_aligned_npu_Subgraphs_28[0], npu_Subgraphs_28_input[0], input_size_npu_Subgraphs_28[0]);
        input_tensors_npu_Subgraphs_28[0]->data = input_aligned_npu_Subgraphs_28[0];

        npu_Subgraphs_28_inputf[1] = instance_normalization_output_20;
        npu_Subgraphs_28_input[1] = shl_ref_f32_to_input_dtype(1, npu_Subgraphs_28_inputf[1], sess_npu_Subgraphs_28);
        memcpy(input_aligned_npu_Subgraphs_28[1], npu_Subgraphs_28_input[1], input_size_npu_Subgraphs_28[1]);
        input_tensors_npu_Subgraphs_28[1]->data = input_aligned_npu_Subgraphs_28[1];

        csinn_update_input_and_run_npu_Subgraphs_28(input_tensors_npu_Subgraphs_28, sess_npu_Subgraphs_28);

        width = 5120;
        get_output(output_tensors_npu_Subgraphs_28, sess_npu_Subgraphs_28);
        float *instance_normalization_output_21 = (float *)malloc(height * channels * width * sizeof(float));
        instance_normalization(output_tensors_npu_Subgraphs_28[0]->data, scale, B, channels, height, width, instance_normalization_output_21);

// printf("-----------npu_Subgraphs_29------------\r\n");
        // npu_Subgraphs_29
        npu_Subgraphs_29_inputf[0] = instance_normalization_output_21;
        npu_Subgraphs_29_input[0] = shl_ref_f32_to_input_dtype(0, npu_Subgraphs_29_inputf[0], sess_npu_Subgraphs_29);
        memcpy(input_aligned_npu_Subgraphs_29[0], npu_Subgraphs_29_input[0], input_size_npu_Subgraphs_29[0]);
        input_tensors_npu_Subgraphs_29[0]->data = input_aligned_npu_Subgraphs_29[0];

        csinn_update_input_and_run_npu_Subgraphs_29(input_tensors_npu_Subgraphs_29, sess_npu_Subgraphs_29);
        get_output(output_tensors_npu_Subgraphs_29, sess_npu_Subgraphs_29);

// printf("-----------cpu_Subgraphs_7------------\r\n");
        // cpu_Subgraphs_7
        cpu_Subgraphs_7_inputf[0] = output_tensors_npu_Subgraphs_29[0]->data;
        input_tensors_cpu_Subgraphs_7[0]->data = shl_c920_f32_to_input_dtype(0, cpu_Subgraphs_7_inputf[0], sess_cpu_Subgraphs_7);
        csinn_update_input_and_run_cpu_Subgraphs_7(input_tensors_cpu_Subgraphs_7, sess_cpu_Subgraphs_7);
        get_output(output_tensors_cpu_Subgraphs_7, sess_cpu_Subgraphs_7);

// printf("-----------npu_Subgraphs_30------------\r\n");
        // npu_Subgraphs_30
        npu_Subgraphs_30_inputf[0] = output_tensors_cpu_Subgraphs_7[0]->data;
        npu_Subgraphs_30_input[0] = shl_ref_f32_to_input_dtype(0, npu_Subgraphs_30_inputf[0], sess_npu_Subgraphs_30);
        memcpy(input_aligned_npu_Subgraphs_30[0], npu_Subgraphs_30_input[0], input_size_npu_Subgraphs_30[0]);
        input_tensors_npu_Subgraphs_30[0]->data = input_aligned_npu_Subgraphs_30[0];

        npu_Subgraphs_30_inputf[1] = output_tensors_npu_Subgraphs_28[1]->data;
        npu_Subgraphs_30_input[1] = shl_ref_f32_to_input_dtype(1, npu_Subgraphs_30_inputf[1], sess_npu_Subgraphs_30);
        memcpy(input_aligned_npu_Subgraphs_30[1], npu_Subgraphs_30_input[1], input_size_npu_Subgraphs_30[1]);
        input_tensors_npu_Subgraphs_30[1]->data = input_aligned_npu_Subgraphs_30[1];

        csinn_update_input_and_run_npu_Subgraphs_30(input_tensors_npu_Subgraphs_30, sess_npu_Subgraphs_30);

        get_output(output_tensors_npu_Subgraphs_30, sess_npu_Subgraphs_30);

        // 参数定义
        float* concat_result_4 = (float*)malloc(1 * 960 * 32 * 32 * sizeof(float));
        concat(output_tensors_npu_Subgraphs_30[0]->data, 1, 640, 32, 32, output_tensors_npu_Subgraphs_4[0]->data, 320, concat_result_4, 960);
        float *reshape_output_6 = (float *)malloc(1 * 32 * 30720 * sizeof(float));
        reshape(concat_result_4, reshape_output_6, 1, 960, 32, 32, 32, 30720);

        width = 30720;
        float *instance_normalization_output_22 = (float *)malloc(height * channels * width * sizeof(float));
        instance_normalization(reshape_output_6, scale, B, channels, height, width, instance_normalization_output_22);

// printf("-----------npu_Subgraphs_31------------\r\n");
        // npu_Subgraphs_31
        npu_Subgraphs_31_inputf[0] = output_tensors_cpu_Subgraphs_0[1]->data;
        npu_Subgraphs_31_input[0] = shl_ref_f32_to_input_dtype(0, npu_Subgraphs_31_inputf[0], sess_npu_Subgraphs_31);
        memcpy(input_aligned_npu_Subgraphs_31[0], npu_Subgraphs_31_input[0], input_size_npu_Subgraphs_31[0]);
        input_tensors_npu_Subgraphs_31[0]->data = input_aligned_npu_Subgraphs_31[0];

        csinn_update_input_and_run_npu_Subgraphs_31(input_tensors_npu_Subgraphs_31, sess_npu_Subgraphs_31);
        get_output(output_tensors_npu_Subgraphs_31, sess_npu_Subgraphs_31);

// printf("-----------npu_Subgraphs_32------------\r\n");
        // npu_Subgraphs_32
        npu_Subgraphs_32_inputf[0] = instance_normalization_output_22;
        npu_Subgraphs_32_input[0] = shl_ref_f32_to_input_dtype(0, npu_Subgraphs_32_inputf[0], sess_npu_Subgraphs_32);
        memcpy(input_aligned_npu_Subgraphs_32[0], npu_Subgraphs_32_input[0], input_size_npu_Subgraphs_32[0]);
        input_tensors_npu_Subgraphs_32[0]->data = input_aligned_npu_Subgraphs_32[0];

        csinn_update_input_and_run_npu_Subgraphs_32(input_tensors_npu_Subgraphs_32, sess_npu_Subgraphs_32);
        get_output(output_tensors_npu_Subgraphs_32, sess_npu_Subgraphs_32);

        float *add_result_3 = (float *)malloc(1 * 320 * 32 * 32 * sizeof(float));
        add_arrays(output_tensors_npu_Subgraphs_32[0]->data, output_tensors_npu_Subgraphs_31[0]->data, add_result_3, 1, 320, 32, 32);
        float *reshape_output_7 = (float *)malloc(1 * 32 * 10240 * sizeof(float));
        reshape(add_result_3, reshape_output_7, 1, 320, 32, 32, 32, 10240);

        width = 10240;
        float *instance_normalization_output_23 = (float *)malloc(height * channels * width * sizeof(float));
        instance_normalization(reshape_output_7, scale, B, channels, height, width, instance_normalization_output_23);

// printf("-----------npu_Subgraphs_33------------\r\n");
        // npu_Subgraphs_33
        npu_Subgraphs_33_inputf[0] = concat_result_4;
        npu_Subgraphs_33_input[0] = shl_ref_f32_to_input_dtype(0, npu_Subgraphs_33_inputf[0], sess_npu_Subgraphs_33);
        memcpy(input_aligned_npu_Subgraphs_33[0], npu_Subgraphs_33_input[0], input_size_npu_Subgraphs_33[0]);
        input_tensors_npu_Subgraphs_33[0]->data = input_aligned_npu_Subgraphs_33[0];

        npu_Subgraphs_33_inputf[1] = instance_normalization_output_23;
        npu_Subgraphs_33_input[1] = shl_ref_f32_to_input_dtype(1, npu_Subgraphs_33_inputf[1], sess_npu_Subgraphs_33);
        memcpy(input_aligned_npu_Subgraphs_33[1], npu_Subgraphs_33_input[1], input_size_npu_Subgraphs_33[1]);
        input_tensors_npu_Subgraphs_33[1]->data = input_aligned_npu_Subgraphs_33[1];

        csinn_update_input_and_run_npu_Subgraphs_33(input_tensors_npu_Subgraphs_33, sess_npu_Subgraphs_33);

        width = 10240;
        get_output(output_tensors_npu_Subgraphs_33, sess_npu_Subgraphs_33);
        float *instance_normalization_output_24 = (float *)malloc(height * channels * width * sizeof(float));
        instance_normalization(output_tensors_npu_Subgraphs_33[0]->data, scale, B, channels, height, width, instance_normalization_output_24);

// printf("-----------npu_Subgraphs_34------------\r\n");
        // npu_Subgraphs_34
        npu_Subgraphs_34_inputf[0] = instance_normalization_output_24;
        npu_Subgraphs_34_input[0] = shl_ref_f32_to_input_dtype(0, npu_Subgraphs_34_inputf[0], sess_npu_Subgraphs_34);
        memcpy(input_aligned_npu_Subgraphs_34[0], npu_Subgraphs_34_input[0], input_size_npu_Subgraphs_34[0]);
        input_tensors_npu_Subgraphs_34[0]->data = input_aligned_npu_Subgraphs_34[0];

        csinn_update_input_and_run_npu_Subgraphs_34(input_tensors_npu_Subgraphs_34, sess_npu_Subgraphs_34);
        get_output(output_tensors_npu_Subgraphs_34, sess_npu_Subgraphs_34);

// printf("-----------cpu_Subgraphs_8------------\r\n");
        // cpu_Subgraphs_8
        cpu_Subgraphs_8_inputf[0] = output_tensors_npu_Subgraphs_34[0]->data;
        input_tensors_cpu_Subgraphs_8[0]->data = shl_c920_f32_to_input_dtype(0, cpu_Subgraphs_8_inputf[0], sess_cpu_Subgraphs_8);
        csinn_update_input_and_run_cpu_Subgraphs_8(input_tensors_cpu_Subgraphs_8, sess_cpu_Subgraphs_8);
        get_output(output_tensors_cpu_Subgraphs_8, sess_cpu_Subgraphs_8);

// printf("-----------npu_Subgraphs_35------------\r\n");
        // npu_Subgraphs_35
        npu_Subgraphs_35_inputf[0] = output_tensors_cpu_Subgraphs_8[0]->data;
        npu_Subgraphs_35_input[0] = shl_ref_f32_to_input_dtype(0, npu_Subgraphs_35_inputf[0], sess_npu_Subgraphs_35);
        memcpy(input_aligned_npu_Subgraphs_35[0], npu_Subgraphs_35_input[0], input_size_npu_Subgraphs_35[0]);
        input_tensors_npu_Subgraphs_35[0]->data = input_aligned_npu_Subgraphs_35[0];

        npu_Subgraphs_35_inputf[1] = output_tensors_npu_Subgraphs_33[1]->data;
        npu_Subgraphs_35_input[1] = shl_ref_f32_to_input_dtype(1, npu_Subgraphs_35_inputf[1], sess_npu_Subgraphs_35);
        memcpy(input_aligned_npu_Subgraphs_35[1], npu_Subgraphs_35_input[1], input_size_npu_Subgraphs_35[1]);
        input_tensors_npu_Subgraphs_35[1]->data = input_aligned_npu_Subgraphs_35[1];

        csinn_update_input_and_run_npu_Subgraphs_35(input_tensors_npu_Subgraphs_35, sess_npu_Subgraphs_35);

        get_output(output_tensors_npu_Subgraphs_35, sess_npu_Subgraphs_35);

        // 参数定义
        float* concat_result_5 = (float*)malloc(1 * 640 * 32 * 32 * sizeof(float));
        concat(output_tensors_npu_Subgraphs_35[0]->data, 1, 320, 32, 32, output_tensors_npu_Subgraphs_0[1]->data, 320, concat_result_5, 640);
        float *reshape_output_8 = (float *)malloc(1 * 32 * 20480 * sizeof(float));
        reshape(concat_result_5, reshape_output_8, 1, 640, 23, 32, 32, 20480);

        width = 20480;
        float *instance_normalization_output_25 = (float *)malloc(height * channels * width * sizeof(float));
        instance_normalization(reshape_output_8, scale, B, channels, height, width, instance_normalization_output_25);

// printf("-----------npu_Subgraphs_36------------\r\n");
        // npu_Subgraphs_36
        npu_Subgraphs_36_inputf[0] = output_tensors_cpu_Subgraphs_0[1]->data;
        npu_Subgraphs_36_input[0] = shl_ref_f32_to_input_dtype(0, npu_Subgraphs_36_inputf[0], sess_npu_Subgraphs_36);
        memcpy(input_aligned_npu_Subgraphs_36[0], npu_Subgraphs_36_input[0], input_size_npu_Subgraphs_36[0]);
        input_tensors_npu_Subgraphs_36[0]->data = input_aligned_npu_Subgraphs_36[0];

        csinn_update_input_and_run_npu_Subgraphs_36(input_tensors_npu_Subgraphs_36, sess_npu_Subgraphs_36);
        get_output(output_tensors_npu_Subgraphs_36, sess_npu_Subgraphs_36);

// printf("-----------npu_Subgraphs_37------------\r\n");
        // npu_Subgraphs_37
        npu_Subgraphs_37_inputf[0] = instance_normalization_output_25;
        npu_Subgraphs_37_input[0] = shl_ref_f32_to_input_dtype(0, npu_Subgraphs_37_inputf[0], sess_npu_Subgraphs_37);
        memcpy(input_aligned_npu_Subgraphs_37[0], npu_Subgraphs_37_input[0], input_size_npu_Subgraphs_37[0]);
        input_tensors_npu_Subgraphs_37[0]->data = input_aligned_npu_Subgraphs_37[0];

        csinn_update_input_and_run_npu_Subgraphs_37(input_tensors_npu_Subgraphs_37, sess_npu_Subgraphs_37);
        get_output(output_tensors_npu_Subgraphs_37, sess_npu_Subgraphs_37);

        float *add_result_4 = (float *)malloc(1 * 320 * 32 * 32 * sizeof(float));
        add_arrays(output_tensors_npu_Subgraphs_37[0]->data, output_tensors_npu_Subgraphs_36[0]->data, add_result_4, 1, 320, 32, 32);
        float *reshape_output_9 = (float *)malloc(1 * 32 * 10240 * sizeof(float));
        reshape(add_result_4, reshape_output_9, 1, 320, 32, 32, 32, 10240);

        width = 10240;
        float *instance_normalization_output_26 = (float *)malloc(height * channels * width * sizeof(float));
        instance_normalization(reshape_output_9, scale, B, channels, height, width, instance_normalization_output_26);

// printf("-----------npu_Subgraphs_38------------\r\n");
        // npu_Subgraphs_38
        npu_Subgraphs_38_inputf[0] = concat_result_5;
        npu_Subgraphs_38_input[0] = shl_ref_f32_to_input_dtype(0, npu_Subgraphs_38_inputf[0], sess_npu_Subgraphs_38);
        memcpy(input_aligned_npu_Subgraphs_38[0], npu_Subgraphs_38_input[0], input_size_npu_Subgraphs_38[0]);
        input_tensors_npu_Subgraphs_38[0]->data = input_aligned_npu_Subgraphs_38[0];

        npu_Subgraphs_38_inputf[1] = instance_normalization_output_26;
        npu_Subgraphs_38_input[1] = shl_ref_f32_to_input_dtype(1, npu_Subgraphs_38_inputf[1], sess_npu_Subgraphs_38);
        memcpy(input_aligned_npu_Subgraphs_38[1], npu_Subgraphs_38_input[1], input_size_npu_Subgraphs_38[1]);
        input_tensors_npu_Subgraphs_38[1]->data = input_aligned_npu_Subgraphs_38[1];

        csinn_update_input_and_run_npu_Subgraphs_38(input_tensors_npu_Subgraphs_38, sess_npu_Subgraphs_38);

        width = 10240;
        get_output(output_tensors_npu_Subgraphs_38, sess_npu_Subgraphs_38);
        float *instance_normalization_output_27 = (float *)malloc(height * channels * width * sizeof(float));
        instance_normalization(output_tensors_npu_Subgraphs_38[0]->data, scale, B, channels, height, width, instance_normalization_output_27);

// printf("-----------npu_Subgraphs_39------------\r\n");
        // npu_Subgraphs_39
        npu_Subgraphs_39_inputf[0] = instance_normalization_output_27;
        npu_Subgraphs_39_input[0] = shl_ref_f32_to_input_dtype(0, npu_Subgraphs_39_inputf[0], sess_npu_Subgraphs_39);
        memcpy(input_aligned_npu_Subgraphs_39[0], npu_Subgraphs_39_input[0], input_size_npu_Subgraphs_39[0]);
        input_tensors_npu_Subgraphs_39[0]->data = input_aligned_npu_Subgraphs_39[0];

        csinn_update_input_and_run_npu_Subgraphs_39(input_tensors_npu_Subgraphs_39, sess_npu_Subgraphs_39);
        get_output(output_tensors_npu_Subgraphs_39, sess_npu_Subgraphs_39);

// printf("-----------cpu_Subgraphs_9------------\r\n");
        // cpu_Subgraphs_9
        cpu_Subgraphs_9_inputf[0] = output_tensors_npu_Subgraphs_39[0]->data;
        input_tensors_cpu_Subgraphs_9[0]->data = shl_c920_f32_to_input_dtype(0, cpu_Subgraphs_9_inputf[0], sess_cpu_Subgraphs_9);
        csinn_update_input_and_run_cpu_Subgraphs_9(input_tensors_cpu_Subgraphs_9, sess_cpu_Subgraphs_9);
        get_output(output_tensors_cpu_Subgraphs_9, sess_cpu_Subgraphs_9);

// printf("-----------npu_Subgraphs_40------------\r\n");
        // npu_Subgraphs_40
        npu_Subgraphs_40_inputf[0] = output_tensors_cpu_Subgraphs_9[0]->data;
        npu_Subgraphs_40_input[0] = shl_ref_f32_to_input_dtype(0, npu_Subgraphs_40_inputf[0], sess_npu_Subgraphs_40);
        memcpy(input_aligned_npu_Subgraphs_40[0], npu_Subgraphs_40_input[0], input_size_npu_Subgraphs_40[0]);
        input_tensors_npu_Subgraphs_40[0]->data = input_aligned_npu_Subgraphs_40[0];

        npu_Subgraphs_40_inputf[1] = output_tensors_npu_Subgraphs_38[1]->data;
        npu_Subgraphs_40_input[1] = shl_ref_f32_to_input_dtype(1, npu_Subgraphs_40_inputf[1], sess_npu_Subgraphs_40);
        memcpy(input_aligned_npu_Subgraphs_40[1], npu_Subgraphs_40_input[1], input_size_npu_Subgraphs_40[1]);
        input_tensors_npu_Subgraphs_40[1]->data = input_aligned_npu_Subgraphs_40[1];

        csinn_update_input_and_run_npu_Subgraphs_40(input_tensors_npu_Subgraphs_40, sess_npu_Subgraphs_40);

        width = 10240;
        get_output(output_tensors_npu_Subgraphs_40, sess_npu_Subgraphs_40);
        float *instance_normalization_output_28 = (float *)malloc(height * channels * width * sizeof(float));
        instance_normalization(output_tensors_npu_Subgraphs_40[0]->data, scale, B, channels, height, width, instance_normalization_output_28);

// printf("-----------npu_Subgraphs_41------------\r\n");
        // npu_Subgraphs_41
        npu_Subgraphs_41_inputf[0] = instance_normalization_output_28;
        npu_Subgraphs_41_input[0] = shl_ref_f32_to_input_dtype(0, npu_Subgraphs_41_inputf[0], sess_npu_Subgraphs_41);
        memcpy(input_aligned_npu_Subgraphs_41[0], npu_Subgraphs_41_input[0], input_size_npu_Subgraphs_41[0]);
        input_tensors_npu_Subgraphs_41[0]->data = input_aligned_npu_Subgraphs_41[0];

        csinn_update_input_and_run_npu_Subgraphs_41(input_tensors_npu_Subgraphs_41, sess_npu_Subgraphs_41);
        get_output(output_tensors_npu_Subgraphs_41, sess_npu_Subgraphs_41);

        int timestep = scheduler.timesteps[ii];
        int model_output_size = 1*3*32*32;
        step(&scheduler, output_tensors_npu_Subgraphs_41[0]->data, ii, timestep, latent, model_output_size);


        end_time = shl_get_timespec();
        printf("Run graph execution time: %.5fms, FPS=%.2f\n", ((float)(end_time-start_time))/1000000,
                    1000000000.0/((float)(end_time-start_time)));

        snprintf(filename_prefix, FILE_PREFIX_LENGTH, "%s", basename(data_path[0]));
        postprocess(sess_npu_Subgraphs_41, filename_prefix);
    }

    result = system("./decoder");
    
    if (result == -1) {
        perror("system");
        return 1;
    }

    for (int j = 0; j < input_num; j++) {
        shl_mem_free(inputf[j]);
    }
    for (int j = 0; j < npu_Subgraphs_0_input_num; j++) {
        shl_mem_free(npu_Subgraphs_0_input[j]);
    }
    for (int j = 0; j < npu_Subgraphs_1_input_num; j++) {
        shl_mem_free(npu_Subgraphs_1_inputf[j]);
        shl_mem_free(npu_Subgraphs_1_input[j]);
    }
    for (int j = 0; j < cpu_Subgraphs_0_input_num; j++) {
        shl_mem_free(cpu_Subgraphs_0_inputf[j]);
    }
    for (int j = 0; j < npu_Subgraphs_2_input_num; j++) {
        shl_mem_free(npu_Subgraphs_2_inputf[j]);
        shl_mem_free(npu_Subgraphs_2_input[j]);
    }
    for (int j = 0; j < npu_Subgraphs_3_input_num; j++) {
        shl_mem_free(npu_Subgraphs_3_inputf[j]);
        shl_mem_free(npu_Subgraphs_3_input[j]);
    }
    for (int j = 0; j < cpu_Subgraphs_1_input_num; j++) {
        shl_mem_free(cpu_Subgraphs_1_inputf[j]);
    }
    for (int j = 0; j < npu_Subgraphs_4_input_num; j++) {
        shl_mem_free(npu_Subgraphs_4_inputf[j]);
        shl_mem_free(npu_Subgraphs_4_input[j]);
    }
    for (int j = 0; j < npu_Subgraphs_5_input_num; j++) {
        shl_mem_free(npu_Subgraphs_5_inputf[j]);
        shl_mem_free(npu_Subgraphs_5_input[j]);
    }
    for (int j = 0; j < npu_Subgraphs_6_input_num; j++) {
        shl_mem_free(npu_Subgraphs_6_inputf[j]);
        shl_mem_free(npu_Subgraphs_6_input[j]);
    }
    for (int j = 0; j < npu_Subgraphs_7_input_num; j++) {
        shl_mem_free(npu_Subgraphs_7_inputf[j]);
        shl_mem_free(npu_Subgraphs_7_input[j]);
    }
    for (int j = 0; j < cpu_Subgraphs_2_input_num; j++) {
        shl_mem_free(cpu_Subgraphs_2_inputf[j]);
    }
    for (int j = 0; j < npu_Subgraphs_8_input_num; j++) {
        shl_mem_free(npu_Subgraphs_8_inputf[j]);
        shl_mem_free(npu_Subgraphs_8_input[j]);
    }
    for (int j = 0; j < npu_Subgraphs_9_input_num; j++) {
        shl_mem_free(npu_Subgraphs_9_input[j]);
    }
    for (int j = 0; j < npu_Subgraphs_10_input_num; j++) {
        shl_mem_free(npu_Subgraphs_10_inputf[j]);
        shl_mem_free(npu_Subgraphs_10_input[j]);
    }
    for (int j = 0; j < npu_Subgraphs_11_input_num; j++) {
        shl_mem_free(npu_Subgraphs_11_inputf[j]);
        shl_mem_free(npu_Subgraphs_11_input[j]);
    }
    for (int j = 0; j < cpu_Subgraphs_3_input_num; j++) {
        shl_mem_free(cpu_Subgraphs_3_inputf[j]);
    }
    for (int j = 0; j < npu_Subgraphs_12_input_num; j++) {
        shl_mem_free(npu_Subgraphs_12_inputf[j]);
        shl_mem_free(npu_Subgraphs_12_input[j]);
    }
    for (int j = 0; j < npu_Subgraphs_13_input_num; j++) {
        shl_mem_free(npu_Subgraphs_13_input[j]);
    }
    for (int j = 0; j < npu_Subgraphs_14_input_num; j++) {
        shl_mem_free(npu_Subgraphs_14_inputf[j]);
        shl_mem_free(npu_Subgraphs_14_input[j]);
    }
    for (int j = 0; j < npu_Subgraphs_15_input_num; j++) {
        shl_mem_free(npu_Subgraphs_15_inputf[j]);
        shl_mem_free(npu_Subgraphs_15_input[j]);
    }
    for (int j = 0; j < cpu_Subgraphs_4_input_num; j++) {
        shl_mem_free(cpu_Subgraphs_4_inputf[j]);
    }
    for (int j = 0; j < npu_Subgraphs_16_input_num; j++) {
        shl_mem_free(npu_Subgraphs_16_inputf[j]);
        shl_mem_free(npu_Subgraphs_16_input[j]);
    }
    for (int j = 0; j < npu_Subgraphs_17_input_num; j++) {
        shl_mem_free(npu_Subgraphs_17_input[j]);
    }
    for (int j = 0; j < npu_Subgraphs_18_input_num; j++) {
        shl_mem_free(npu_Subgraphs_18_inputf[j]);
        shl_mem_free(npu_Subgraphs_18_input[j]);
    }
    for (int j = 0; j < npu_Subgraphs_19_input_num; j++) {
        shl_mem_free(npu_Subgraphs_19_inputf[j]);
        shl_mem_free(npu_Subgraphs_19_input[j]);
    }
    for (int j = 0; j < cpu_Subgraphs_5_input_num; j++) {
        shl_mem_free(cpu_Subgraphs_5_inputf[j]);
    }
    for (int j = 0; j < npu_Subgraphs_20_input_num; j++) {
        shl_mem_free(npu_Subgraphs_20_inputf[j]);
        shl_mem_free(npu_Subgraphs_20_input[j]);
    }
    for (int j = 0; j < npu_Subgraphs_21_input_num; j++) {
        shl_mem_free(npu_Subgraphs_21_input[j]);
    }
    for (int j = 0; j < npu_Subgraphs_22_input_num; j++) {
        shl_mem_free(npu_Subgraphs_22_inputf[j]);
        shl_mem_free(npu_Subgraphs_22_input[j]);
    }
    for (int j = 0; j < npu_Subgraphs_23_input_num; j++) {
        shl_mem_free(npu_Subgraphs_23_inputf[j]);
        shl_mem_free(npu_Subgraphs_23_input[j]);
    }
    for (int j = 0; j < npu_Subgraphs_24_input_num; j++) {
        shl_mem_free(npu_Subgraphs_24_inputf[j]);
        shl_mem_free(npu_Subgraphs_24_input[j]);
    }
    for (int j = 0; j < cpu_Subgraphs_6_input_num; j++) {
        shl_mem_free(cpu_Subgraphs_6_inputf[j]);
    }
    for (int j = 0; j < npu_Subgraphs_25_input_num; j++) {
        shl_mem_free(npu_Subgraphs_25_inputf[j]);
        shl_mem_free(npu_Subgraphs_25_input[j]);
    }
    for (int j = 0; j < npu_Subgraphs_26_input_num; j++) {
        shl_mem_free(npu_Subgraphs_26_input[j]);
    }
    for (int j = 0; j < npu_Subgraphs_27_input_num; j++) {
        shl_mem_free(npu_Subgraphs_27_inputf[j]);
        shl_mem_free(npu_Subgraphs_27_input[j]);
    }
    for (int j = 0; j < npu_Subgraphs_28_input_num; j++) {
        shl_mem_free(npu_Subgraphs_28_inputf[j]);
        shl_mem_free(npu_Subgraphs_28_input[j]);
    }
    for (int j = 0; j < npu_Subgraphs_29_input_num; j++) {
        shl_mem_free(npu_Subgraphs_29_inputf[j]);
        shl_mem_free(npu_Subgraphs_29_input[j]);
    }
    for (int j = 0; j < cpu_Subgraphs_7_input_num; j++) {
        shl_mem_free(cpu_Subgraphs_7_inputf[j]);
    }
    for (int j = 0; j < npu_Subgraphs_30_input_num; j++) {
        shl_mem_free(npu_Subgraphs_30_inputf[j]);
        shl_mem_free(npu_Subgraphs_30_input[j]);
    }
    for (int j = 0; j < npu_Subgraphs_31_input_num; j++) {
        shl_mem_free(npu_Subgraphs_31_input[j]);
    }
    for (int j = 0; j < npu_Subgraphs_32_input_num; j++) {
        shl_mem_free(npu_Subgraphs_32_inputf[j]);
        shl_mem_free(npu_Subgraphs_32_input[j]);
    }
    for (int j = 0; j < npu_Subgraphs_33_input_num; j++) {
        shl_mem_free(npu_Subgraphs_33_inputf[j]);
        shl_mem_free(npu_Subgraphs_33_input[j]);
    }
    for (int j = 0; j < npu_Subgraphs_34_input_num; j++) {
        shl_mem_free(npu_Subgraphs_34_inputf[j]);
        shl_mem_free(npu_Subgraphs_34_input[j]);
    }
    for (int j = 0; j < cpu_Subgraphs_8_input_num; j++) {
        shl_mem_free(cpu_Subgraphs_8_inputf[j]);
    }
    for (int j = 0; j < npu_Subgraphs_35_input_num; j++) {
        shl_mem_free(npu_Subgraphs_35_inputf[j]);
        shl_mem_free(npu_Subgraphs_35_input[j]);
    }
    for (int j = 0; j < npu_Subgraphs_36_input_num; j++) {
        shl_mem_free(npu_Subgraphs_36_input[j]);
    }
    for (int j = 0; j < npu_Subgraphs_37_input_num; j++) {
        shl_mem_free(npu_Subgraphs_37_inputf[j]);
        shl_mem_free(npu_Subgraphs_37_input[j]);
    }
    for (int j = 0; j < npu_Subgraphs_38_input_num; j++) {
        shl_mem_free(npu_Subgraphs_38_inputf[j]);
        shl_mem_free(npu_Subgraphs_38_input[j]);
    }
    for (int j = 0; j < npu_Subgraphs_39_input_num; j++) {
        shl_mem_free(npu_Subgraphs_39_inputf[j]);
        shl_mem_free(npu_Subgraphs_39_input[j]);
    }
    for (int j = 0; j < cpu_Subgraphs_9_input_num; j++) {
        shl_mem_free(cpu_Subgraphs_9_inputf[j]);
    }
    for (int j = 0; j < npu_Subgraphs_40_input_num; j++) {
        shl_mem_free(npu_Subgraphs_40_inputf[j]);
        shl_mem_free(npu_Subgraphs_40_input[j]);
    }
    for (int j = 0; j < npu_Subgraphs_41_input_num; j++) {
        shl_mem_free(npu_Subgraphs_41_inputf[j]);
        shl_mem_free(npu_Subgraphs_41_input[j]);
    }
    for (int j = 0; j < npu_Subgraphs_0_input_num; j++) {
        csinn_free_tensor(input_tensors_npu_Subgraphs_0[j]);
    }
    for (int j = 0; j < npu_Subgraphs_1_input_num; j++) {
        csinn_free_tensor(input_tensors_npu_Subgraphs_1[j]);
    }
    for (int j = 0; j < cpu_Subgraphs_0_input_num; j++) {
        csinn_free_tensor(input_tensors_cpu_Subgraphs_0[j]);
    }
    for (int j = 0; j < npu_Subgraphs_2_input_num; j++) {
        csinn_free_tensor(input_tensors_npu_Subgraphs_2[j]);
    }
    for (int j = 0; j < npu_Subgraphs_3_input_num; j++) {
        csinn_free_tensor(input_tensors_npu_Subgraphs_3[j]);
    }
    for (int j = 0; j < cpu_Subgraphs_1_input_num; j++) {
        csinn_free_tensor(input_tensors_cpu_Subgraphs_1[j]);
    }
    for (int j = 0; j < npu_Subgraphs_4_input_num; j++) {
        csinn_free_tensor(input_tensors_npu_Subgraphs_4[j]);
    }
    for (int j = 0; j < npu_Subgraphs_5_input_num; j++) {
        csinn_free_tensor(input_tensors_npu_Subgraphs_5[j]);
    }
    for (int j = 0; j < npu_Subgraphs_6_input_num; j++) {
        csinn_free_tensor(input_tensors_npu_Subgraphs_6[j]);
    }
    for (int j = 0; j < npu_Subgraphs_7_input_num; j++) {
        csinn_free_tensor(input_tensors_npu_Subgraphs_7[j]);
    }
    for (int j = 0; j < cpu_Subgraphs_2_input_num; j++) {
        csinn_free_tensor(input_tensors_cpu_Subgraphs_2[j]);
    }
    for (int j = 0; j < npu_Subgraphs_8_input_num; j++) {
        csinn_free_tensor(input_tensors_npu_Subgraphs_8[j]);
    }
    for (int j = 0; j < npu_Subgraphs_9_input_num; j++) {
        csinn_free_tensor(input_tensors_npu_Subgraphs_9[j]);
    }
    for (int j = 0; j < npu_Subgraphs_10_input_num; j++) {
        csinn_free_tensor(input_tensors_npu_Subgraphs_10[j]);
    }
    for (int j = 0; j < npu_Subgraphs_11_input_num; j++) {
        csinn_free_tensor(input_tensors_npu_Subgraphs_11[j]);
    }
    for (int j = 0; j < cpu_Subgraphs_3_input_num; j++) {
        csinn_free_tensor(input_tensors_cpu_Subgraphs_3[j]);
    }
    for (int j = 0; j < npu_Subgraphs_12_input_num; j++) {
        csinn_free_tensor(input_tensors_npu_Subgraphs_12[j]);
    }
    for (int j = 0; j < npu_Subgraphs_13_input_num; j++) {
        csinn_free_tensor(input_tensors_npu_Subgraphs_13[j]);
    }
    for (int j = 0; j < npu_Subgraphs_14_input_num; j++) {
        csinn_free_tensor(input_tensors_npu_Subgraphs_14[j]);
    }
    for (int j = 0; j < npu_Subgraphs_15_input_num; j++) {
        csinn_free_tensor(input_tensors_npu_Subgraphs_15[j]);
    }
    for (int j = 0; j < cpu_Subgraphs_4_input_num; j++) {
        csinn_free_tensor(input_tensors_cpu_Subgraphs_4[j]);
    }
    for (int j = 0; j < npu_Subgraphs_16_input_num; j++) {
        csinn_free_tensor(input_tensors_npu_Subgraphs_16[j]);
    }
    for (int j = 0; j < npu_Subgraphs_17_input_num; j++) {
        csinn_free_tensor(input_tensors_npu_Subgraphs_17[j]);
    }
    for (int j = 0; j < npu_Subgraphs_18_input_num; j++) {
        csinn_free_tensor(input_tensors_npu_Subgraphs_18[j]);
    }
    for (int j = 0; j < npu_Subgraphs_19_input_num; j++) {
        csinn_free_tensor(input_tensors_npu_Subgraphs_19[j]);
    }
    for (int j = 0; j < cpu_Subgraphs_5_input_num; j++) {
        csinn_free_tensor(input_tensors_cpu_Subgraphs_5[j]);
    }
    for (int j = 0; j < npu_Subgraphs_20_input_num; j++) {
        csinn_free_tensor(input_tensors_npu_Subgraphs_20[j]);
    }
    for (int j = 0; j < npu_Subgraphs_21_input_num; j++) {
        csinn_free_tensor(input_tensors_npu_Subgraphs_21[j]);
    }
    for (int j = 0; j < npu_Subgraphs_22_input_num; j++) {
        csinn_free_tensor(input_tensors_npu_Subgraphs_22[j]);
    }
    for (int j = 0; j < npu_Subgraphs_23_input_num; j++) {
        csinn_free_tensor(input_tensors_npu_Subgraphs_23[j]);
    }
    for (int j = 0; j < npu_Subgraphs_24_input_num; j++) {
        csinn_free_tensor(input_tensors_npu_Subgraphs_24[j]);
    }
    for (int j = 0; j < cpu_Subgraphs_6_input_num; j++) {
        csinn_free_tensor(input_tensors_cpu_Subgraphs_6[j]);
    }
    for (int j = 0; j < npu_Subgraphs_25_input_num; j++) {
        csinn_free_tensor(input_tensors_npu_Subgraphs_25[j]);
    }
    for (int j = 0; j < npu_Subgraphs_26_input_num; j++) {
        csinn_free_tensor(input_tensors_npu_Subgraphs_26[j]);
    }
    for (int j = 0; j < npu_Subgraphs_27_input_num; j++) {
        csinn_free_tensor(input_tensors_npu_Subgraphs_27[j]);
    }
    for (int j = 0; j < npu_Subgraphs_28_input_num; j++) {
        csinn_free_tensor(input_tensors_npu_Subgraphs_28[j]);
    }
    for (int j = 0; j < npu_Subgraphs_29_input_num; j++) {
        csinn_free_tensor(input_tensors_npu_Subgraphs_29[j]);
    }
    for (int j = 0; j < cpu_Subgraphs_7_input_num; j++) {
        csinn_free_tensor(input_tensors_cpu_Subgraphs_7[j]);
    }
    for (int j = 0; j < npu_Subgraphs_30_input_num; j++) {
        csinn_free_tensor(input_tensors_npu_Subgraphs_30[j]);
    }
    for (int j = 0; j < npu_Subgraphs_31_input_num; j++) {
        csinn_free_tensor(input_tensors_npu_Subgraphs_31[j]);
    }
    for (int j = 0; j < npu_Subgraphs_32_input_num; j++) {
        csinn_free_tensor(input_tensors_npu_Subgraphs_32[j]);
    }
    for (int j = 0; j < npu_Subgraphs_33_input_num; j++) {
        csinn_free_tensor(input_tensors_npu_Subgraphs_33[j]);
    }
    for (int j = 0; j < npu_Subgraphs_34_input_num; j++) {
        csinn_free_tensor(input_tensors_npu_Subgraphs_34[j]);
    }
    for (int j = 0; j < cpu_Subgraphs_8_input_num; j++) {
        csinn_free_tensor(input_tensors_cpu_Subgraphs_8[j]);
    }
    for (int j = 0; j < npu_Subgraphs_35_input_num; j++) {
        csinn_free_tensor(input_tensors_npu_Subgraphs_35[j]);
    }
    for (int j = 0; j < npu_Subgraphs_36_input_num; j++) {
        csinn_free_tensor(input_tensors_npu_Subgraphs_36[j]);
    }
    for (int j = 0; j < npu_Subgraphs_37_input_num; j++) {
        csinn_free_tensor(input_tensors_npu_Subgraphs_37[j]);
    }
    for (int j = 0; j < npu_Subgraphs_38_input_num; j++) {
        csinn_free_tensor(input_tensors_npu_Subgraphs_38[j]);
    }
    for (int j = 0; j < npu_Subgraphs_39_input_num; j++) {
        csinn_free_tensor(input_tensors_npu_Subgraphs_39[j]);
    }
    for (int j = 0; j < cpu_Subgraphs_9_input_num; j++) {
        csinn_free_tensor(input_tensors_cpu_Subgraphs_8[j]);
    }
    for (int j = 0; j < npu_Subgraphs_40_input_num; j++) {
        csinn_free_tensor(input_tensors_npu_Subgraphs_40[j]);
    }
    for (int j = 0; j < npu_Subgraphs_41_input_num; j++) {
        csinn_free_tensor(input_tensors_npu_Subgraphs_41[j]);
    }
    csinn_free_session(sess_npu_Subgraphs_0);
    csinn_free_session(sess_npu_Subgraphs_1);
    csinn_free_session(sess_cpu_Subgraphs_0);
    csinn_free_session(sess_npu_Subgraphs_2);
    csinn_free_session(sess_npu_Subgraphs_3);
    csinn_free_session(sess_cpu_Subgraphs_1);
    csinn_free_session(sess_npu_Subgraphs_4);
    csinn_free_session(sess_npu_Subgraphs_5);
    csinn_free_session(sess_npu_Subgraphs_6);
    csinn_free_session(sess_npu_Subgraphs_7);
    csinn_free_session(sess_cpu_Subgraphs_2);
    csinn_free_session(sess_npu_Subgraphs_8);
    csinn_free_session(sess_npu_Subgraphs_9);
    csinn_free_session(sess_npu_Subgraphs_10);
    csinn_free_session(sess_npu_Subgraphs_11);
    csinn_free_session(sess_cpu_Subgraphs_3);
    csinn_free_session(sess_npu_Subgraphs_12);
    csinn_free_session(sess_npu_Subgraphs_13);
    csinn_free_session(sess_npu_Subgraphs_14);
    csinn_free_session(sess_npu_Subgraphs_15);
    csinn_free_session(sess_cpu_Subgraphs_4);
    csinn_free_session(sess_npu_Subgraphs_16);
    csinn_free_session(sess_npu_Subgraphs_17);
    csinn_free_session(sess_npu_Subgraphs_18);
    csinn_free_session(sess_npu_Subgraphs_19);
    csinn_free_session(sess_cpu_Subgraphs_5);
    csinn_free_session(sess_npu_Subgraphs_20);
    csinn_free_session(sess_npu_Subgraphs_21);
    csinn_free_session(sess_npu_Subgraphs_22);
    csinn_free_session(sess_npu_Subgraphs_23);
    csinn_free_session(sess_npu_Subgraphs_24);
    csinn_free_session(sess_cpu_Subgraphs_6);
    csinn_free_session(sess_npu_Subgraphs_25);
    csinn_free_session(sess_npu_Subgraphs_26);
    csinn_free_session(sess_npu_Subgraphs_27);
    csinn_free_session(sess_npu_Subgraphs_28);
    csinn_free_session(sess_npu_Subgraphs_29);
    csinn_free_session(sess_cpu_Subgraphs_7);
    csinn_free_session(sess_npu_Subgraphs_25);
    csinn_free_session(sess_npu_Subgraphs_26);
    csinn_free_session(sess_npu_Subgraphs_27);
    csinn_free_session(sess_npu_Subgraphs_28);
    csinn_free_session(sess_npu_Subgraphs_29);
    csinn_free_session(sess_cpu_Subgraphs_7);
    csinn_free_session(sess_npu_Subgraphs_30);
    csinn_free_session(sess_npu_Subgraphs_31);
    csinn_free_session(sess_npu_Subgraphs_32);
    csinn_free_session(sess_npu_Subgraphs_33);
    csinn_free_session(sess_npu_Subgraphs_34);
    csinn_free_session(sess_cpu_Subgraphs_8);
    csinn_free_session(sess_npu_Subgraphs_35);
    csinn_free_session(sess_npu_Subgraphs_36);
    csinn_free_session(sess_npu_Subgraphs_37);
    csinn_free_session(sess_npu_Subgraphs_38);
    csinn_free_session(sess_npu_Subgraphs_39);
    csinn_free_session(sess_cpu_Subgraphs_9);
    csinn_free_session(sess_npu_Subgraphs_40);
    csinn_free_session(sess_npu_Subgraphs_41);

    return 0;
}

