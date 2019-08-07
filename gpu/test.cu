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

inline double seconds()
{
    struct timeval tp;
    struct timezone tzp;
    int i = gettimeofday(&tp, &tzp);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

__global__ void CUDA_matrix_multiplication(float Input[], float Weight[], float Output[], int shape[]) {
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int colum = blockIdx.x*blockDim.x + threadIdx.x;
    
    float sum = 0;
    if(row<shape[0] && colum<shape[2]) {
        for (int i=0; i<shape[1]; i++) {
            sum += Input[row*shape[1] + i] * Weight[i*shape[2] + colum];
        }
        Output[row*shape[2] + colum] = sum;
    }
}

int main() {
    float *input, *weight, *output;
    float *Input, *Weight, *Output;
    int *Shape;
    int shape[4] = {40,1,128,0};

    // Alloc memory for CPU
    input = (float*)malloc(1024*1024*sizeof(float));
    weight = (float*)malloc(1024*1024*sizeof(float));
    output = (float*)malloc(1024*1024*sizeof(float));

    // Array initialization
    /*for(int i=0; i<16; i++) {
        input[i] = (float)rand();
        weight[i] = (float)rand();
    }*/

    double start, exetime;
    start = seconds();

    // Alloc memory for GPU
    cudaMalloc((void**)&Input, 1024*1024*sizeof(float));
    cudaMalloc((void**)&Weight, 1024*1024*sizeof(float));
    cudaMalloc((void**)&Output, 1024*1024*sizeof(float));
    cudaMalloc((void**)&Shape, 4*sizeof(int));

    // Move data from CPU to GPU
    cudaMemcpy(Input, input, shape[0]*shape[1]*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(Weight, weight, shape[1]*shape[2]*sizeof(float), cudaMemcpyHostToDevice);
    //cudaMemcpy(Output, output, shape[0]*shape[2]*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(Shape, shape, 4*sizeof(int), cudaMemcpyHostToDevice);

    // Configuration of kernels, basic block has 64 threads
    dim3 dimBlock(16,16,1);
    dim3 dimGrid(shape[0]/8+1,shape[2]/8+1,1);

    // Kernel execution
    CUDA_matrix_multiplication<<<dimGrid, dimBlock>>>(Input, Weight, Output, Shape);

    // Move data from GPU to CPU
    cudaMemcpy(output, Output, shape[0]*shape[2]*sizeof(float), cudaMemcpyDeviceToHost);
    
    shape[0] = 128; shape[1] = 1; shape[2] =1; shape[3] = 0;
    cudaMemcpy(Input, output, shape[0]*shape[1]*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(Shape, shape, 4*sizeof(int), cudaMemcpyHostToDevice);

    CUDA_matrix_multiplication<<<dimGrid, dimBlock>>>(Input, Weight, Output, Shape);
    cudaMemcpy(output, Output, shape[0]*shape[2]*sizeof(float), cudaMemcpyDeviceToHost);

    exetime = seconds() - start;
    printf("Time used:%f us\n", exetime*1000000);


    for(int i=0;i<3; i++) {
        for(int j=0; j<3; j++) {
            cout<<output[i*3+j]<<" ";
        }
        cout<<endl;
    }

    // Free memory
    free(input);
    free(weight);
    free(output);
    cudaFree(Input);
    cudaFree(Weight);
    cudaFree(Output);
    cudaFree(Shape);
    
    return 0;
}
#endif