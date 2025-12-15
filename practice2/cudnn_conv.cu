#include <iostream>
#include <cudnn.h>
#include <cuda_runtime.h>
using namespace std;

int main(){
    // 已經寫好：建立 cudnn handle
    cudnnHandle_t handle; 
    cudnnCreate(&handle);

    // 已經寫好：建立 descriptor
    cudnnTensorDescriptor_t xDesc,yDesc;
    cudnnFilterDescriptor_t wDesc;
    cudnnConvolutionDescriptor_t convDesc;
    cudnnCreateTensorDescriptor(&xDesc);
    cudnnCreateTensorDescriptor(&yDesc);
    cudnnCreateFilterDescriptor(&wDesc);
    cudnnCreateConvolutionDescriptor(&convDesc);

    // TODO: 設定輸入張量 (例如 NCHW 格式: 1x1x32x32)
    // TODO: 設定 filter (例如 1x1x3x3)
    // TODO: 設定卷積參數 (padding, stride)
    // TODO: 使用 cudnnGetConvolution2dForwardOutputDim 計算輸出維度
    // TODO: 輸出結果 (例如 cout << "Output shape: ..." )

    // 已經寫好：釋放資源
    cudnnDestroyConvolutionDescriptor(convDesc);
    cudnnDestroyFilterDescriptor(wDesc);
    cudnnDestroyTensorDescriptor(xDesc);
    cudnnDestroyTensorDescriptor(yDesc);
    cudnnDestroy(handle);
}
