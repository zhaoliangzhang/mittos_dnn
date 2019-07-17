#include <stdlib.h>
#include "functions.h"

static __inline__ unsigned long long GetCycleCount()
{
        unsigned hi,lo;
        __asm__ volatile("rdtsc":"=a"(lo),"=d"(hi));
        return ((unsigned long long)lo)|(((unsigned long long)hi)<<32);
}

int main() {

    FILE *fp;
    char str[81];
    fp=popen("cat /proc/cpuinfo|grep cpu\\ MHz|sed -e 's/.*:[^0-9]//'","r");
    if(fp<0){
            printf("Can not read the frequency info of CPU\n");
            exit(1);
    }
    char* result = fgets(str,80,fp);
    fclose(fp);
    double fre=atof(str);
    printf("Frequency of CPU:%lf MHz\n", fre);

    int* input_int = (int*)malloc(sizeof(int)*128*40);
    int* weight_int = (int*)malloc(sizeof(int)*128*1);
    int* output_int = (int*)malloc(sizeof(int)*40*1);
    int* output2_int = (int*)malloc(sizeof(int)*1);
    int* bias_int;

    unsigned int* input_uint = (unsigned int*)malloc(sizeof(unsigned int)*128*40);
    unsigned int* weight_uint = (unsigned int*)malloc(sizeof(unsigned int)*128*1);
    unsigned int* output_uint = (unsigned int*)malloc(sizeof(unsigned int)*40*1);
    unsigned int* output2_uint = (unsigned int*)malloc(sizeof(unsigned int)*1);
    unsigned int* bias_uint;

    int8_t* input_int8 = (int8_t*)malloc(sizeof(int8_t)*128*40);
    int8_t* weight_int8 = (int8_t*)malloc(sizeof(int8_t)*128*1);
    int8_t* output_int8 = (int8_t*)malloc(sizeof(int8_t)*40*1);
    int8_t* output2_int8 = (int8_t*)malloc(sizeof(int8_t)*1);
    int8_t* bias_int8;

    u_int8_t* input_uint8 = (u_int8_t*)malloc(sizeof(u_int8_t)*128*40);
    u_int8_t* weight_uint8 = (u_int8_t*)malloc(sizeof(u_int8_t)*128*1);
    u_int8_t* output_uint8 = (u_int8_t*)malloc(sizeof(u_int8_t)*40*1);
    u_int8_t* output2_uint8 = (u_int8_t*)malloc(sizeof(u_int8_t)*1);
    u_int8_t* bias_uint8;

    float* input_float = (float*)malloc(sizeof(float)*128*128);
    float* weight_float = (float*)malloc(sizeof(float)*128*128);
    float* output_float = (float*)malloc(sizeof(float)*128*1);
    float* output2_float = (float*)malloc(sizeof(float)*1);
    float* bias_float;

    input = (float*)malloc(sizeof(float)*128*40);
    output = (float*)malloc(sizeof(float)*128*1);
    weight = (float*)malloc(sizeof(float)*40*1);
    output2 = (float*)malloc(sizeof(float)*1);
    float* bias;
    for(int i=0; i<40; i++) {
        weight[i] = (float)rand();
        //weight[i] = 1;
        weight_int[i] = (int)rand();
        for(int j=0; j<128; j++) {
            input_int[i*128+j] = (int)rand();
            input[i*128+j] = (float)rand();
            //input[i*1000+j] = i;
        }
    }
    shape = (int*)malloc(sizeof(int)*4);
    shape2 = (int*)malloc(sizeof(int)*4);
    shape[0]=128;shape[1]=128;shape[2]=1;shape[3]=0;
    shape2[0]=1;shape2[1]=128;shape2[2]=1;shape2[3]=0;
    exist_job = shape[0] - 1;

    double t1,t2;

    /*t1 = (double)GetCycleCount();
    for(int i=0; i<100; i++){
    linear_int(input_int, weight_int, bias_int, output_int, shape);
    linear_int(output_int, weight_int, bias_int, output2_int, shape);
    }
    t2 = (double)GetCycleCount();

    printf("Execution time of int:         %*lf ns\n", 15, (t2-t1)*1000/(fre*100));

    t1 = (double)GetCycleCount();
    for(int i=0; i<100; i++){
    linear_int2(input_int, weight_int, bias_int, output_int, shape);
    linear_int2(output_int, weight_int, bias_int, output2_int, shape);
    }
    t2 = (double)GetCycleCount();

    printf("Execution time of int2:        %*lf ns\n", 15, (t2-t1)*1000/(fre*100));

    t1 = (double)GetCycleCount();
    for(int i=0; i<100; i++){
    linear_uint(input_uint, weight_uint, bias_uint, output_uint, shape);
    linear_uint(output_uint, weight_uint, bias_uint, output2_uint, shape);
    }
    t2 = (double)GetCycleCount();

    printf("Execution time of unsigned int:%*lf ns\n", 15, (t2-t1)*1000/(fre*100));

    t1 = (double)GetCycleCount();
    for(int i=0; i<100; i++){
    linear_int8(input_int8, weight_int8, bias_int8, output_int8, shape);
    linear_int8(output_int8, weight_int8, bias_int8, output2_int8, shape);
    }
    t2 = (double)GetCycleCount();

    printf("Execution time of int8:        %*lf ns\n", 15, (t2-t1)*1000/(fre*100));

    t1 = (double)GetCycleCount();
    for(int i=0; i<100; i++){
    linear_uint8(input_uint8, weight_uint8, bias_uint8, output_uint8, shape);
    linear_uint8(output_uint8, weight_uint8, bias_uint8, output2_uint8, shape);
    }
    t2 = (double)GetCycleCount();

    printf("Execution time of uint8:       %*lf ns\n", 15, (t2-t1)*1000/(fre*100));*/

    t1 = (double)GetCycleCount();
    for(int i=0; i<100; i++){
    linear_float(input_float, weight_float, bias_float, output_float, shape);
    //linear_float(output_float, weight_float, bias_float, output2_float, shape2);
    }
    t2 = (double)GetCycleCount();

    printf("Execution time of float:      %*lf ns\n", 15, (t2-t1)*1000/(fre*100));

    t1 = (double)GetCycleCount();
    for(int i=0; i<100; i++){
    linear_float2(input_float, weight_float, bias_float, output_float, shape);
    //linear_float2(output_float, weight_float, bias_float, output2_float, shape2);
    }
    t2 = (double)GetCycleCount();

    printf("Execution time of float2:     %*lf ns\n", 15, (t2-t1)*1000/(fre*100));

    /*t1 = (double)GetCycleCount();
    sem_init(&task_num, 0, (unsigned int)shape[0]);
    if_free = 0;
    void* retval;
    pthread_t vector_inner[MAX_THREAD_NUM];
    int create_thread_success;
    int end_tread_success;
    sem_init(&task_num, 0, (unsigned int)shape[0]);
    for(int i=0; i<MAX_THREAD_NUM; i++) {
        create_thread_success = pthread_create(&vector_inner[i], NULL, inner_product, NULL);
    }
    for(int j=0; j<MAX_THREAD_NUM; j++) {
        end_tread_success = pthread_join(vector_inner[j], &retval);
    }
    float* temp = input;
    input = output;
    output = input;
    shape[0]=1;shape[1]=128;shape[2]=1;shape[3]=0;
    sem_init(&task_num, 0, (unsigned int)shape[0]);
    for(int j=0; j<MAX_THREAD_NUM; j++) {
        end_tread_success = pthread_join(vector_inner[j], &retval);
    }
    sem_destroy(&task_num);
    t2 = (double)GetCycleCount();

    printf("Execution time of multithreads:%*lf ns\n", 15, (t2-t1)*1000/fre);*/
 
    free(input_int);
    free(output_int);
    free(weight_int);
    free(input);
    free(output);
    free(weight);
    free(shape);

    return 0;
    
}
