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

    int* input_int = (int*)malloc(sizeof(int)*1000*1000);
    int* weight_int = (int*)malloc(sizeof(int)*1000*1000);
    int* output_int = (int*)malloc(sizeof(int)*1000*1000);
    float* input_float = (float*)malloc(sizeof(float)*1000*1000);
    float* output_float = (float*)malloc(sizeof(float)*1000*1000);
    float* weight_float = (float*)malloc(sizeof(float)*1000*1000);
    for(int i=0; i<1000; i++) {
        for(int j=0; j<1000; j++) {
            input_int[i*1000+j] = (int)rand();
            weight_int[i*1000+j] = (int)rand();
            input_float[i*1000+j] = (float)rand();
            weight_float[i*1000+j] = (float)rand();
        }
    }
    int* bias_int;
    float* bias_float;
    int shape[4]={3,3,3,0};

    double t1,t2;

    t1 = (double)GetCycleCount();
    linear_int(input_int, weight_int, bias_int, output_int, shape);
    t2 = (double)GetCycleCount();

    printf("Execution time of int:%lf ns\n", (t2-t1)*1000/fre);

    t1 = (double)GetCycleCount();
    linear_float(input_float, weight_float, bias_float, output_float, shape);
    t2 = (double)GetCycleCount();

    printf("Execution time of float:%lf ns\n", (t2-t1)*1000/fre);

    free(input_int);
    free(output_int);
    free(weight_int);
    free(input_float);
    free(output_float);
    free(weight_float);

    return 0;
    
}