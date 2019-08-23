#include <stdlib.h>
#include "functions.h"

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

    float* input_float = (float*)malloc(sizeof(float)*128*128);
    float* weight_float = (float*)malloc(sizeof(float)*128*128);
    float* output_float = (float*)malloc(sizeof(float)*128*1);
    float* output2_float = (float*)malloc(sizeof(float)*1);
    float* bias_float;

    FILE *weight_ptr;
    int unused __attribute__((unused));
    if((weight_ptr = fopen("ncnn.bin", "r")) == NULL) {
        printf("Opne weight file erro\n");
        return -1;
    }
    fseek(weight_ptr, sizeof(float), SEEK_CUR);
    unused = fread(weight_float, sizeof(float), 40*128, weight_ptr);
    fseek(weight_ptr, sizeof(float), SEEK_CUR);
    unused = fread(weight_float+40*128, sizeof(float), 40*128, weight_ptr);
    fclose(weight_ptr);

    shape = (int*)malloc(sizeof(int)*4);
    shape2 = (int*)malloc(sizeof(int)*4);
    shape[0]=128;shape[1]=40;shape[2]=1;shape[3]=0;
    shape2[0]=1;shape2[1]=128;shape2[2]=1;shape2[3]=0;

    double t1,t2;

#ifdef MULTI_THREAD
    for(int i=0; i<4; i++) {
        input[i] = (float*)malloc(sizeof(float)*128*128);
        output[i] = (float*)malloc(sizeof(float)*128*128);
    }
    output2 = (float*)malloc(sizeof(float)*1);
    weight = (float*)malloc(sizeof(float)*128*128);
    for(int i=0; i<128; i++) {
        for(int j=0; j<128; j++) {
            weight[i*128+j] = (float)rand();
            input[0][i*128+j] = (float)rand();
            input[1][i*128+j] = (float)rand();
            input[2][i*128+j] = (float)rand();
            input[3][i*128+j] = (float)rand();
        }
    }

    Max = 0;
    Min = 1e300;

    t1 = (double)GetCycleCount();
    void* retval;
    pthread_t vector_inner[MAX_THREAD_NUM];
    int create_thread_success;
    int end_tread_success;
    int id[4] = {0,1,2,3};
    //sem_init(&task_num, 0, 0);
    for(int i=0; i<MAX_THREAD_NUM; i++) {
        create_thread_success = pthread_create(&vector_inner[i], NULL, DNN, (void*)(id+i));
    }
    //sem_init(&task_num, 0, TASK_NUM);
    sem_post(&task_num);
    sem_post(&task_num);
    sem_post(&task_num);
    sem_post(&task_num);
    tt=0;
    sleep(2);
    for(int j=0; j<MAX_THREAD_NUM; j++) {
        end_tread_success = pthread_join(vector_inner[j], &retval);
    }
    sem_destroy(&task_num);
    t2 = (double)GetCycleCount();

    printf("Execution time of multithreads:                                  %*lf ns\n", 15, (t2-t1)*1000/fre - 2000000000);
    printf("Execution time of multithreads:                                  %*lf ns\n", 15, tt*1000/fre);
    printf("Execution time of multithreads(without thread creation and exit):%*lf ns\n", 15, (Max-Min)*1000/fre);
    //for(int i=0; i<8; i++){
        //printf("%lf\n", t[i]);
    //}
    printf("%lf %lf\n", Max, Min);
 
    for(int i=0; i<MAX_THREAD_NUM; i++) {
        free(input[i]);
        free(output[i]);
    }
    free(weight);
    free(shape);
    free(shape2);
#else
    t1 = (double)GetCycleCount();
    for(int i=0; i<1000; i++){
    linear_float(input_float, weight_float, bias_float, output_float, shape);
    //linear_float(output_float, weight_float, bias_float, output2_float, shape2);
    }
    t2 = (double)GetCycleCount();

    printf("Execution time of float:      %*lf ns\n", 15, (t2-t1)*1000/(fre*1000));

    t1 = (double)GetCycleCount();
    for(int i=0; i<1000; i++){
    linear_float2(input_float, weight_float, bias_float, output_float, shape);
    linear_float2(output_float, weight_float, bias_float, output2_float, shape2);
    }
    t2 = (double)GetCycleCount();

    printf("Execution time of float2:     %*lf ns\n", 15, (t2-t1)*1000/(fre*1000));
  
#endif

    return 0;
    
}
