#include <cudnn.h>
#include <cuda_runtime.h>
#include <iostream>
using namespace std;

int main(){
    cudnnHandle_t handle; cudnnCreate(&handle);

    cudnnTensorDescriptor_t xDesc,yDesc;
    cudnnFilterDescriptor_t wDesc;
    cudnnConvolutionDescriptor_t convDesc;
    cudnnCreateTensorDescriptor(&xDesc);
    cudnnCreateTensorDescriptor(&yDesc);
    cudnnCreateFilterDescriptor(&wDesc);
    cudnnCreateConvolutionDescriptor(&convDesc);

    // TODO: 設定輸入張量 (NCHW 格式)
    // TODO: 設定卷積參數 (padding, stride)
    // TODO: 計算輸出維度 (cudnnGetConvolution2dForwardOutputDim)
    // TODO: 呼叫 cudnnConvolutionForward 完成卷積

    cudnnDestroyConvolutionDescriptor(convDesc);
    cudnnDestroyFilterDescriptor(wDesc);
    cudnnDestroyTensorDescriptor(xDesc);
    cudnnDestroyTensorDescriptor(yDesc);
    cudnnDestroy(handle);
}
