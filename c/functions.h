#include <stdio.h>
#include <pthread.h>
#include <semaphore.h>
#include <stdbool.h>
# include <unistd.h>

#define MAX_THREAD_NUM 4
#define TASK_NUM 4
#define NORMAL 1
//#define MULTI_THREAD

float *input[MAX_THREAD_NUM];
float *output[MAX_THREAD_NUM];
float temp[MAX_THREAD_NUM][128];
float *weight;
float *bias;
int *shape;
int *shape2;
float *output2;
bool if_free;
sem_t task_num;
double t[8];
double tt;
double Max, Min;

static __inline__ unsigned long long GetCycleCount()
{
        unsigned hi,lo;
        __asm__ volatile("rdtsc":"=a"(lo),"=d"(hi));
        return ((unsigned long long)lo)|(((unsigned long long)hi)<<32);
}

void linear_int(int* input, int* weight, int* bias, int* output, int* shape);
void linear_int2(int* input, int* weight, int* bias, int* output, int* shape);

void linear_uint(unsigned int* input, unsigned int* weight, unsigned int* bias, unsigned int* output, int* shape);

void linear_int8(int8_t* input, int8_t* weight, int8_t* bias, int8_t* output, int* shape);

void linear_uint8(u_int8_t* input, u_int8_t* weight, u_int8_t* bias, u_int8_t* output, int* shape);

void linear_float(float* input, float* weight, float* bias, float* output, int* shape);
void linear_float2(float* input, float* weight, float* bias, float* output, int* shape);

void* DNN(void *arg);
