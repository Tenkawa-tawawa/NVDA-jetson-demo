#include <iostream>
#include <cuda_runtime.h>
using namespace std;

__global__ void matMul(const float* A,const float* B,float* C,int N){
    int row=blockIdx.y*blockDim.y+threadIdx.y;
    int col=blockIdx.x*blockDim.x+threadIdx.x;
    if(row<N && col<N){
        float sum=0;
        for(int k=0;k<N;k++) sum+=A[row*N+k]*B[k*N+col];
        C[row*N+col]=sum;
    }
}

int main(){
    int N=256; size_t size=N*N*sizeof(float);
    float *h_A=new float[N*N],*h_B=new float[N*N],*h_C=new float[N*N];
    for(int i=0;i<N*N;i++){ h_A[i]=1.0f; h_B[i]=2.0f; }

    float *d_A,*d_B,*d_C;
    cudaMalloc(&d_A,size); cudaMalloc(&d_B,size); cudaMalloc(&d_C,size);
    cudaMemcpy(d_A,h_A,size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,h_B,size,cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16,16);
    dim3 blocksPerGrid((N+15)/16,(N+15)/16);
    matMul<<<blocksPerGrid,threadsPerBlock>>>(d_A,d_B,d_C,N);

    cudaMemcpy(h_C,d_C,size,cudaMemcpyDeviceToHost);
    cout<<"C[0]="<<h_C[0]<<endl;

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    delete[] h_A; delete[] h_B; delete[] h_C;
}
