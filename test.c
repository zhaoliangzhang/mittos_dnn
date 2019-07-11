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
    int* weight_int = (int*)malloc(sizeof(int)*1*1000);
    int* output_int = (int*)malloc(sizeof(int)*1000*1);
    input = (float*)malloc(sizeof(float)*1000*1000);
    output = (float*)malloc(sizeof(float)*1*1000);
    weight = (float*)malloc(sizeof(float)*1000*1);
    for(int i=0; i<1000; i++) {
        weight[i] = (float)rand();
        //weight[i] = 1;
        weight_int[i] = (int)rand();
        for(int j=0; j<1000; j++) {
            input_int[i*1000+j] = (int)rand();
            input[i*1000+j] = (float)rand();
            //input[i*1000+j] = i;
        }
    }
    int* bias_int;
    float* bias_float;
    shape = (int*)malloc(sizeof(int)*4);
    shape[0]=128;shape[1]=128;shape[2]=1;shape[3]=0;
    exist_job = shape[0] - 1;

    double t1,t2;

    t1 = (double)GetCycleCount();
    //for(int i=0;i<100;i++) 
    linear_int(input_int, weight_int, bias_int, output_int, shape);
    t2 = (double)GetCycleCount();

    printf("Execution time of int:%lf ns\n", (t2-t1)*1000/fre);

    t1 = (double)GetCycleCount();
    //for(int i=0;i<100;i++) 
    linear_float();
    t2 = (double)GetCycleCount();

    printf("Execution time of float:%lf ns\n", (t2-t1)*1000/fre);
    /*for(int i=0; i<10; i++) {
        for(int j=0; j<10; j++) {
            printf("%f ", input[i*1000+j]);
        }
        printf("\n");
    }
    for(int i=0;i<10;i++){
        printf("%f\n", weight[i]);
    }
    for(int i=0;i<10;i++){
        printf("%f\n", output[i]);
    }*/

    free(input_int);
    free(output_int);
    free(weight_int);
    free(input);
    free(output);
    free(weight);
    free(shape);

    return 0;
    
}