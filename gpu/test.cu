#include "device_launch_parameters.h"
#include <iostream>
#include "cuda_runtime.h"
#include <stdlib.h>
#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>

//#define DEVICE_TEST

using namespace std;

#ifdef DEVICE_TEST
int main()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    for(int i=0;i<deviceCount;i++)
    {
        cudaDeviceProp devProp;
        cudaGetDeviceProperties(&devProp, i);
        std::cout << "使用GPU device " << i << ": " << devProp.name << std::endl;
        std::cout << "设备全局内存总量： " << devProp.totalGlobalMem / 1024 / 1024 << "MB" << std::endl;
        std::cout << "SM的数量：" << devProp.multiProcessorCount << std::endl;
        std::cout << "每个线程块的共享内存大小：" << devProp.sharedMemPerBlock / 1024.0 << " KB" << std::endl;
        std::cout << "每个线程块的最大线程数：" << devProp.maxThreadsPerBlock << std::endl;
        std::cout << "设备上一个线程块（Block）种可用的32位寄存器数量： " << devProp.regsPerBlock << std::endl;
        std::cout << "每个EM的最大线程数：" << devProp.maxThreadsPerMultiProcessor << std::endl;
        std::cout << "每个EM的最大线程束数：" << devProp.maxThreadsPerMultiProcessor / 32 << std::endl;
        std::cout << "设备上多处理器的数量： " << devProp.multiProcessorCount << std::endl;
        std::cout << "======================================================" << std::endl;     
        
    }
    return 0;
}
#else

__global__ void CUDA_matrix_multiplication1(float Input[], float Weight[], float Output[]) {
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int colum = blockIdx.x*blockDim.x + threadIdx.x;
    
    float sum = 0;
    if(row<128 && colum<1) {
        for (int i=0; i<40; i++) {
            sum += Weight[row*40 + i] * Input[i*1 + colum];
        }
        Output[row*1 + colum] = sum;
    }
}

__global__ void CUDA_matrix_multiplication2(float Input[], float Weight[], float Output[]) {
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int colum = blockIdx.x*blockDim.x + threadIdx.x;
    
    float sum = 0;
    if(row<1 && colum<1) {
        for (int i=0; i<128; i++) {
            sum += Input[row*128 + i] * Weight[i*1 + colum];
        }
        Output[row*1 + colum] = sum;
    }
}

int main() {

    float *input, *weight, *output;
    float *Input, *Weight, *Output;

    // Alloc memory for CPU
    //cudaHostAlloc((void **) &input, sizeof(float)*1024*1024, cudaHostAllocDefault);
    //cudaHostAlloc((void **) &weight, sizeof(float)*1024*1024, cudaHostAllocDefault);
    //cudaHostAlloc((void **) &output, sizeof(float)*1024*1024, cudaHostAllocDefault);
    input = (float*)malloc(1024*1024*sizeof(float));
    weight = (float*)malloc(1024*1024*sizeof(float));
    output = (float*)malloc(1024*1024*sizeof(float));

    // Array initialization
    /*for(int i=0; i<3; i++) {
        for(int j=0; j<3; j++) {
            input[i*3+j] = i;
            weight[i*3+j] = 1;
        }
    }*/


    // Alloc memory for GPU
    cudaMalloc((void**)&Input, 1024*1024*sizeof(float));
    cudaMalloc((void**)&Weight, 1024*1024*sizeof(float));
    cudaMalloc((void**)&Output, 1024*1024*sizeof(float));

    // Move data from CPU to GPU
    cudaMemcpy(Input, input, 40*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(Weight, weight, 128*40*sizeof(float), cudaMemcpyHostToDevice);
    //cudaMemcpy(Output, output, shape[0]*shape[2]*sizeof(float), cudaMemcpyHostToDevice);

    // Configuration of kernels, basic block has 64 threads
    dim3 dimBlock(16,16,1);
    dim3 dimGrid(1/16+1,1/16+1,1);

    // Kernel execution
    CUDA_matrix_multiplication1<<<dimGrid, dimBlock>>>(Input, Weight, Output);
    CUDA_matrix_multiplication2<<<dimGrid, dimBlock>>>(Output, Weight, Input);

    // Move data from GPU to CPU
    //cudaMemcpy(output, Output, shape[0]*shape[2]*sizeof(float), cudaMemcpyDeviceToHost);
    
    //cudaMemcpy(Input, output, shape[0]*shape[1]*sizeof(float), cudaMemcpyHostToDevice);
    //cudaMemcpy(Shape, shape, 4*sizeof(int), cudaMemcpyHostToDevice);

    //CUDA_matrix_multiplication<<<dimGrid, dimBlock>>>(Input, Weight, Output, Shape);

    cudaMemcpy(input, Input, 1*sizeof(float), cudaMemcpyDeviceToHost);

    // Free memory
    //cudaFreeHost(input);
    //cudaFreeHost(weight);
    //cudaFreeHost(output);
    free(input);
    free(weight);
    free(output);
    cudaFree(Input);
    cudaFree(Weight);
    cudaFree(Output);
    
    return 0;
}
#endif