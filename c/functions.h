#include <stdio.h>
#include <pthread.h>
#include <semaphore.h>
#include <stdbool.h>

#define MAX_THREAD_NUM 4

float* input;
float* output;
float* weight;
int* shape;
int* shape2;
float* output2;
int  exist_job;
bool if_free;
sem_t task_num;

void linear_int(int* input, int* weight, int* bias, int* output, int* shape);
void linear_int2(int* input, int* weight, int* bias, int* output, int* shape);

void linear_uint(unsigned int* input, unsigned int* weight, unsigned int* bias, unsigned int* output, int* shape);

void linear_int8(int8_t* input, int8_t* weight, int8_t* bias, int8_t* output, int* shape);

void linear_uint8(u_int8_t* input, u_int8_t* weight, u_int8_t* bias, u_int8_t* output, int* shape);

void linear_float(float* input, float* weight, float* bias, float* output, int* shape);
void linear_float2(float* input, float* weight, float* bias, float* output, int* shape);

void* inner_product(void* arg);
