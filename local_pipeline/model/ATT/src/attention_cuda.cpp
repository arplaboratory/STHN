// #include <THC/THC.h>
#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include "attention_kernel.h"

int hello_world(torch::Tensor test){
    return 1;
}

int attention_cuda_forward(torch::Tensor input1, torch::Tensor input2, torch::Tensor output, int kernel_size, int pad_size, int stride){
    int batch_size = input1.size(0);
    int input_rows = input1.size(1);
    int input_cols = input1.size(2);
    int input_ch = input1.size(3);

    int kernel_radius = kernel_size / 2;

    int output_cols = (input_cols - kernel_radius * 2 - 1) / stride + 1;
    int output_rows = (input_rows - kernel_radius * 2 - 1) / stride + 1;
    int output_ch = kernel_size * kernel_size;

    attention_forward_ongpu(input1.data<float>(), input2.data<float>(), output.data<float>(), 
    kernel_size, pad_size, stride, input_cols, input_rows, output_cols, output_rows, 
        input_ch, output_ch, batch_size);
    
    return 1;
}

int attention_cuda_backward(torch::Tensor input1, torch::Tensor input2, torch::Tensor grad_input, torch::Tensor grad_input_padding, 
    torch::Tensor grad_output1, torch::Tensor grad_output2, int kernel_size, int pad_size, int stride){
    int batch_size = input1.size(0);
    int input_rows = input1.size(1);
    int input_cols = input1.size(2);
    int input_ch = input1.size(3);

    int kernel_radius = kernel_size / 2;

    int output_cols = (input_cols - kernel_radius * 2 - 1) / stride + 1;
    int output_rows = (input_rows - kernel_radius * 2 - 1) / stride + 1;
    int output_ch = kernel_size * kernel_size;

    attention_backward_ongpu(input1.data<float>(), input2.data<float>(), grad_input.data<float>(), 
        grad_input_padding.data<float>(), grad_output1.data<float>(), grad_output2.data<float>(), kernel_size, 
    pad_size, stride, input_cols, input_rows, output_cols, output_rows, input_ch, output_ch, batch_size);

    return 1;
}

int channel_attention_cuda_forward(torch::Tensor input1, torch::Tensor input2, torch::Tensor output, int kernel_size, int pad_size, int stride){
    int batch_size = input1.size(0);
    int input1_rows = input1.size(1);
    int input1_cols = input1.size(2);
    int input1_chs = input1.size(3);

    int input2_rows = input2.size(1);
    int input2_cols = input2.size(2);
    int input2_chs = input2.size(3);

    int kernel_radius = kernel_size / 2;

    channel_attention_forward_ongpu(input1.data<float>(), input2.data<float>(), output.data<float>(), 
        kernel_size, pad_size, stride, input1_cols, input1_rows, input2_cols, input2_rows, input1_chs, input2_chs, batch_size);

    return 1;
}

int channel_attention_cuda_backward(torch::Tensor input1, torch::Tensor input2, torch::Tensor grad_input, torch::Tensor grad_output1, torch::Tensor grad_output2, 
    int kernel_size, int pad_size, int stride){
        int batch_size = input1.size(0);
        int input1_rows = input1.size(1);
        int input1_cols = input1.size(2);
        int input1_chs = input1.size(3);

        int input2_rows = input2.size(1);
        int input2_cols = input2.size(2);
        int input2_chs = input2.size(3);

        int kernel_radius = kernel_size / 2;

        channel_attention_backward_ongpu(input1.data<float>(), input2.data<float>(), grad_input.data<float>(), grad_output1.data<float>(), grad_output2.data<float>(), 
            kernel_size, pad_size, stride, input1_cols, input1_rows, input2_cols, input2_rows, input1_chs, input2_chs, batch_size);
        return 1;
    }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &attention_cuda_forward, "ATT forward");
  m.def("backward", &attention_cuda_backward, "ATT backward");
  m.def("channel_forward", &channel_attention_cuda_forward, "ATT channel forward");
  m.def("channel_backward", &channel_attention_cuda_backward, "ATT channel backward");
  m.def("test", &hello_world, "testing");
}