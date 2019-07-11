#include <stdio.h>
#include <pthread.h>
#include <semaphore.h>

#define MAX_THREAD_NUM 2

float* input;
float* output;
float* weight;
int* shape;
int  exist_job;

void linear_int(int* input, int* weight, int* bias, int* output, int* shape);

void* inner_product(void* arg);

void linear_float();