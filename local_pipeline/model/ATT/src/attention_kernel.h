#ifndef _ATTENTION_CUDA_KERNEL
#define _ATTENTION_CUDA_KERNEL

void attention_forward_ongpu(const float* input1, const float* input2, float* output, int kernel_size, int pad_size, int stride,
    int input_cols, int input_rows, int output_cols, int output_rows, int input_ch, int output_ch, int batch_size);

void attention_backward_ongpu(const float* input1, const float* input2, const float* grad_input, const float* grad_input_padding, 
    float* grad_output0, float* grad_output1, int kernel_size, int pad_size, int stride,
    int input_cols, int input_rows, int output_cols, int output_rows, int input_ch, int output_ch, int batch_size);

void channel_attention_forward_ongpu(const float* input1, const float* input2, float* output, int kernel_size, int pad_size, int stride, 
    int input1_cols, int input1_rows, int input2_cols, int input2_rows, int input1_chs, int input2_chs, int batch_size);

void channel_attention_backward_ongpu(const float* input1, const float* input2, const float* grad_input, float* grad_output1, float* grad_output2, 
    int kernel_size, int pad_size, int stride, int input1_cols, int input1_rows, int input2_cols, int input2_rows, int input1_chs, int input2_chs, int batch_size);

#endif