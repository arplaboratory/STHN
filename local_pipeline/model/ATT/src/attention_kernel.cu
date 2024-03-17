#include "attention_kernel.h"

#define CUDA_NUM_THREADS 1024
#define WARPS_PER_BLOCK 1
#define THREADS_PER_WARP 32

__global__ void attention_forward_kernel(const float* input1, const float *input2, float* output, int kernel_size, int pad_size, int stride,
    int input_cols, int input_rows, int output_cols, int output_rows, int input_ch, int output_ch, int batch_size,
    int kernel_radius){
        extern __shared__ char input_data_char[];
        float* input_data = (float*) input_data_char;

        int x1 = blockIdx.x * stride + kernel_radius;
        int y1 = blockIdx.y * stride + kernel_radius;
        int x = blockIdx.x;
        int y = blockIdx.y;
        int z = blockIdx.z;
        
        int ch_off = threadIdx.x;
        for(int ch = ch_off; ch < input_ch; ch += THREADS_PER_WARP * WARPS_PER_BLOCK){
            input_data[ch] = input1[ ((z * input_rows + y1) * input_cols + x1) * input_ch + ch];
        }

        __syncthreads();
        
        __shared__ float sum[THREADS_PER_WARP * WARPS_PER_BLOCK];

        for(int out_ch = 0; out_ch < output_ch; out_ch++){
            sum[ch_off] = 0;
            int x2 = (out_ch % kernel_size) - kernel_radius;
            int y2 = (out_ch / kernel_size) - kernel_radius;
            
            for(int ch = ch_off; ch < input_ch; ch += THREADS_PER_WARP * WARPS_PER_BLOCK){
                sum[ch_off] += input2[((z * input_rows + y1 + y2) * input_cols + x1 + x2) * input_ch + ch] * input_data[ch];
            }
            
            __syncthreads();

            if(ch_off == 0){
                for(int ch = 0; ch < THREADS_PER_WARP * WARPS_PER_BLOCK; ch++){
                    output[((blockIdx.z * output_rows + y) * output_cols + x) * output_ch + out_ch] += sum[ch];        
                }
            }

            __syncthreads();
        }
}


void attention_forward_ongpu(const float* input1, const float* input2, float* output, int kernel_size, int pad_size, int stride,
    int input_cols, int input_rows, int output_cols, int output_rows, int input_ch, int output_ch, int batch_size){

    int threads_per_block(THREADS_PER_WARP * WARPS_PER_BLOCK);
    dim3 blocks_dim(output_cols, output_rows, batch_size);

    attention_forward_kernel<<<blocks_dim, threads_per_block,  input_ch * sizeof(float)>>>(
        input1, input2, output, kernel_size, pad_size, stride, input_cols, input_rows, output_cols, output_rows, input_ch, output_ch, batch_size, 
        kernel_size / 2
    );
}

// input1 : padding, input2 : padding, grad_input : no padding, grad_output : padding
__global__ void attention_backward0_kernel(const float* input1, const float* input2, const float* grad_input, float * grad_output, int kernel_size, int pad_size, int stride,
    int input_cols, int input_rows, int output_cols, int output_rows, int input_chs, int output_chs, int batch_size, int kernel_radius){
    
    extern __shared__ char grad_input_char[];
    float* grad_data = (float*) grad_input_char;

    int x = blockIdx.x;
    int y = blockIdx.y;
    int z = blockIdx.z;
    int x1 = blockIdx.x * stride + kernel_radius;
    int y1 = blockIdx.y * stride + kernel_radius;
    int ch_off = threadIdx.x;
    
    for(int out_ch = ch_off; out_ch < output_chs; out_ch += THREADS_PER_WARP * WARPS_PER_BLOCK){
        grad_data[out_ch] = grad_input[ ((z * output_rows + blockIdx.y) * output_cols + blockIdx.x) * output_chs + out_ch];
    }
    

    __syncthreads();

    for(int out_ch = 0; out_ch < output_chs; out_ch++){

        int x2 = out_ch % kernel_size - kernel_radius;
        int y2 = out_ch / kernel_size - kernel_radius;

        for(int ch = ch_off; ch < input_chs; ch += THREADS_PER_WARP * WARPS_PER_BLOCK){
            grad_output[ ((z * input_rows + y1) * input_cols + x1) * input_chs + ch] +=  
                grad_data[out_ch] * input2[ ((z * input_rows + y1 + y2) * input_cols + x1 + x2) * input_chs + ch];
        }

        __syncthreads();
    }
}


// input1 : padding, input2 : padding, grad_input : padding, grad_output : no_padding
__global__ void attention_backward1_kernel(const float* input1, const float* input2, const float* grad_input, float * grad_output, int kernel_size, int pad_size, int stride,
    int input_cols, int input_rows, int output_cols, int output_rows, int input_chs, int output_chs, int batch_size, int kernel_radius){
    int x = blockIdx.x;
    int y = blockIdx.y;
    int z = blockIdx.z;
    int x1 = blockIdx.x * stride + kernel_radius;
    int y1 = blockIdx.y * stride + kernel_radius;
    int ch_off = threadIdx.x;

    for(int out_ch = 0; out_ch < output_chs; out_ch++){
       
        int x2 = out_ch % kernel_size - kernel_radius;
        int y2 = out_ch / kernel_size - kernel_radius;
    
        int inv_out_ch = (-y2 + kernel_radius) * kernel_size - x2 + kernel_radius;

        for(int ch = ch_off; ch < input_chs; ch += THREADS_PER_WARP * WARPS_PER_BLOCK){
            grad_output[ ((z * output_rows + y) * output_cols + x) * input_chs + ch] +=  grad_input[ ((z * input_rows + y1 + y2) * input_cols + x1 + x2) * output_chs + inv_out_ch] 
                * input1[ ((z * input_rows + y1 + y2) * input_cols + x1 + x2) * input_chs + ch];
        }

        __syncthreads();
    }
}

void attention_backward_ongpu(const float* input1, const float* input2, const float* grad_input, const float* grad_input_padding, 
    float* grad_output0, float* grad_output1, int kernel_size, int pad_size, int stride,
    int input_cols, int input_rows, int output_cols, int output_rows, int input_ch, int output_ch, int batch_size){
        int threads_per_block(THREADS_PER_WARP * WARPS_PER_BLOCK);
        dim3 blocks_dim(output_cols, output_rows, batch_size);

        attention_backward0_kernel<<<blocks_dim, threads_per_block,  output_ch * sizeof(float)>>>(
            input1, input2, grad_input, grad_output0, kernel_size, pad_size, stride, input_cols, input_rows, output_cols, 
            output_rows, input_ch, output_ch, batch_size, kernel_size / 2
        );

        attention_backward1_kernel<<<blocks_dim, threads_per_block,  0>>>(
            input1, input2, grad_input_padding, grad_output1, kernel_size, pad_size, stride, input_cols, input_rows, output_cols, 
            output_rows, input_ch, output_ch, batch_size, kernel_size / 2
        );
    }


__global__ void channel_attention_forward_kernel(const float* input1, const float* input2, float* output, int kernel_size, int pad_size, int stride, 
    int input1_cols, int input1_rows, int input2_cols, int input2_rows, int input1_chs, int input2_chs, int batch_size, int kernel_radius){
    int x = blockIdx.x;
    int y = blockIdx.y;
    int z = blockIdx.z;
    int x1 = blockIdx.x * stride + kernel_radius;
    int y1 = blockIdx.y * stride + kernel_radius;
    int ch_off = threadIdx.x;

    for(int ch1 = 0; ch1 < input1_chs; ch1++){
       
        int x2 = ch1 % kernel_size - kernel_radius;
        int y2 = ch1 / kernel_size - kernel_radius;

        for(int ch2 = ch_off; ch2 < input2_chs; ch2 += THREADS_PER_WARP * WARPS_PER_BLOCK){
            output[ (((z * input1_rows + y) * input1_cols) + x) * input2_chs + ch2] += input1[(((z * input1_rows + y) * input1_cols) + x) * input1_chs + ch1]
                * input2[ ((z * input2_rows + y1 + y2) * input2_cols + x1 + x2) * input2_chs + ch2];
        }

        __syncthreads();
    }  
}

// input1 no padding input2 padding grad_input no padding grad_output no padding
__global__ void channel_attention_backward0_kernel(const float *input1, const float* input2, const float* grad_input, float * grad_output, int kernel_size, int pad_size, 
    int stride, int input1_cols, int input1_rows, int input2_rows, int input2_cols, int input1_chs, int input2_chs, int batch_size, int kernel_radius){
        int x = blockIdx.x;
        int y = blockIdx.y;
        int z = blockIdx.z;
        int x1 = blockIdx.x * stride + kernel_radius;
        int y1 = blockIdx.y * stride + kernel_radius;
        int ch_off = threadIdx.x;

        for(int ch1 = ch_off; ch1 < input1_chs; ch1 += THREADS_PER_WARP * WARPS_PER_BLOCK){
        
            int x2 = ch1 % kernel_size - kernel_radius;
            int y2 = ch1 / kernel_size - kernel_radius;

            for(int ch2 = 0; ch2 < input2_chs; ch2++){
                grad_output[ ((z * input1_rows + y)*input1_cols + x) * input1_chs + ch1] += grad_input[ ((z * input1_rows + y) * input1_cols + x) * input2_chs + ch2]
                    * input2[ ((z * input2_rows + y1 + y2) * input2_cols + x1 + x2) * input2_chs + ch2];
            }

            __syncthreads();
        }  
    }

// input1 no padding input2 padding grad_input no padding grad_output no padding
__global__ void channel_attention_backward1_kernel(const float *input1, const float* input2, const float* grad_input, float * grad_output, int kernel_size, int pad_size, 
    int stride, int input1_cols, int input1_rows, int input2_rows, int input2_cols, int input1_chs, int input2_chs, int batch_size, int kernel_radius){
        int x = blockIdx.x;
        int y = blockIdx.y;
        int z = blockIdx.z;
        int ch_off = threadIdx.x;

        for(int ch1 = 0; ch1 < input1_chs; ch1++){
        
            int x2 = ch1 % kernel_size - kernel_radius;
            int y2 = ch1 / kernel_size - kernel_radius;

            int inv_ch1 = (-y2 + kernel_radius) * kernel_size - x2 + kernel_radius;
            
            if(y + y2 >= 0 && y + y2 < input1_rows && x + x2 >= 0 && x + x2 < input1_cols){
                for(int ch2 = ch_off; ch2 < input2_chs; ch2 += THREADS_PER_WARP * WARPS_PER_BLOCK){
                    grad_output[ ((z * input1_rows + y) * input1_cols + x) * input2_chs + ch2] += grad_input[ ((z * input1_rows + y + y2) * input1_cols + x + x2) * input2_chs + ch2]
                        * input1[ ((z * input1_rows + y + y2)*input1_cols + x + x2) * input1_chs + inv_ch1];
                }
            }

            __syncthreads();
        }  
    }

// input1 no padding, input2 padding, output no padding
void channel_attention_forward_ongpu(const float* input1, const float* input2, float* output, int kernel_size, int pad_size, int stride, 
    int input1_cols, int input1_rows, int input2_cols, int input2_rows, int input1_chs, int input2_chs, int batch_size){
        int threads_per_block(THREADS_PER_WARP * WARPS_PER_BLOCK);
        dim3 blocks_dim(input1_cols, input1_rows, batch_size);

        channel_attention_forward_kernel<<<blocks_dim, threads_per_block, 0>>>(
            input1, input2, output, kernel_size, pad_size, stride, input1_cols, input1_rows, input2_cols, input2_rows, input1_chs, input2_chs, batch_size, kernel_size / 2
        );
    }


// input1 no padding, input2 padding, grad_input no padding, grad_output1 no padding, grad_output2 padding
void channel_attention_backward_ongpu(const float* input1, const float* input2, const float* grad_input, float* grad_output1, float* grad_output2, 
    int kernel_size, int pad_size, int stride, int input1_cols, int input1_rows, int input2_cols, int input2_rows, int input1_chs, int input2_chs, int batch_size){
        int threads_per_block(THREADS_PER_WARP * WARPS_PER_BLOCK);
        dim3 blocks_dim(input1_cols, input1_rows, batch_size);

        channel_attention_backward0_kernel<<<blocks_dim, threads_per_block, 0>>>(
            input1, input2, grad_input, grad_output1, kernel_size, pad_size, stride, input1_cols, input1_rows, input2_cols, input2_rows, input1_chs, input2_chs, batch_size, kernel_size / 2
        );

        channel_attention_backward1_kernel<<<blocks_dim, threads_per_block, 0>>>(
            input1, input2, grad_input, grad_output2, kernel_size, pad_size, stride, input1_cols, input1_rows, input2_cols, input2_rows, input1_chs, input2_chs, batch_size, kernel_size / 2
        );
    }