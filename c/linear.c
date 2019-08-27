#include "functions.h"

/*******************************
 * This file contains functions of matrices
 *
 * The format of functions is linear_type, as type is the data type of martix
 *
 * Parameters:
 * input:  the matrix on the left in the multiplication
 * weight: the matrix on the right in the multiplication
 * bias:   the bias to added after multiplication
 * output: the result of the multiplication
 * shape:  the shape of the matrices, indicates that function
 *         does a matrix multiplications of two matrice with
 *         shape of shape[0]*shape[1] and  shape[1]*shape[2],
 *         if shape[3] is 1, means it need to add bias after
 *         multiplication, if shape[3], then no bias
 * 
 * Detailed notes are included in funtion linear_int and linear_int2
 * ****************************/

void linear_int(int* input, int* weight, int* bias, int* output, int* shape) {
    for(int i=0; i<shape[0]; i++) {
        for(int j=0; j<shape[2]; j++) {
            output[i*shape[2]+j] = 0;
            for(int k=0; k<shape[1]; k++) {
                output[i*shape[2]+j] += input[i*shape[1]+k] * weight[k*shape[2]+j];
            }
	    //To decide whether need to add bias
            if(shape[3]){
                output[i*shape[2]+j] += bias[i*shape[2]+j];
            }
	    //Relu
            if(output[i*shape[2]+j]<0) {
                output[i*shape[2]+j] = 0;
            }
        }
    }
}

void linear_int2(int* input, int* weight, int* bias, int* output, int* shape) {
    for(int i=0; i<shape[0]; i++) {
        for(int j=0; j<shape[2]; j++) {
            output[i*shape[2]+j] = 0;
	    //loop unroll
            for(int k=0; k<shape[1]; k+=4) {
                output[i*shape[2]+j] += input[i*shape[1]+k] * weight[k*shape[2]+j];
                output[i*shape[2]+j] += input[i*shape[1]+k+1] * weight[(k+1)*shape[2]+j];
                output[i*shape[2]+j] += input[i*shape[1]+k+2] * weight[(k+2)*shape[2]+j];
                output[i*shape[2]+j] += input[i*shape[1]+k+3] * weight[(k+3)*shape[2]+j];
            }
            if(shape[3]){
                output[i*shape[2]+j] += bias[i*shape[2]+j];
            }
            if(output[i*shape[2]+j]<0) {
                output[i*shape[2]+j] = 0;
	    }
}

void linear_uint(unsigned int* input, unsigned int* weight, unsigned int* bias, unsigned int* output, int* shape) {
    unsigned int temp[shape[2]];
    for(int i=0; i<shape[0]; i++) {
        for(int j=0; j<shape[2]; j++) temp[j] = 0;
        for(int j=0; j<shape[1]; j++) {
            for(int k=0; k<shape[2]; k++) {
                temp[k] += input[i*shape[1]+j] * weight[j*shape[2]+k];
            }
        }
        if(shape[3]) {
            for(int j=0; j<shape[2]; j++) output[i*shape[2]+j] = temp[j] + bias[i*shape[2]+j];
        } else {
            for(int j=0; j<shape[2]; j++) output[i*shape[2]+j] = temp[j];
        }
    }
}

void linear_int8(int8_t* input, int8_t* weight, int8_t* bias, int8_t* output, int* shape) {
    int8_t temp[shape[2]];
    for(int i=0; i<shape[0]; i++) {
        for(int j=0; j<shape[2]; j++) temp[j] = 0;
        for(int j=0; j<shape[1]; j++) {
            for(int k=0; k<shape[2]; k++) {
                temp[k] += input[i*shape[1]+j] * weight[j*shape[2]+k];
            }
        }
        if(shape[3]) {
            for(int j=0; j<shape[2]; j++) output[i*shape[2]+j] = temp[j] + bias[i*shape[2]+j];
        } else {
            for(int j=0; j<shape[2]; j++) output[i*shape[2]+j] = temp[j];
        }
    }
}

void linear_uint8(u_int8_t* input, u_int8_t* weight, u_int8_t* bias, u_int8_t* output, int* shape) {
    u_int8_t temp[shape[2]];
    for(int i=0; i<shape[0]; i++) {
        for(int j=0; j<shape[2]; j++) temp[j] = 0;
        for(int j=0; j<shape[1]; j++) {
            for(int k=0; k<shape[2]; k++) {
                temp[k] += input[i*shape[1]+j] * weight[j*shape[2]+k];
            }
        }
        if(shape[3]) {
            for(int j=0; j<shape[2]; j++) output[i*shape[2]+j] = temp[j] + bias[i*shape[2]+j];
        } else {
            for(int j=0; j<shape[2]; j++) output[i*shape[2]+j] = temp[j];
        }
    }
}

void linear_float(float* input, float* weight, float* bias, float* output, int* shape) {
   for(int i=0; i<shape[0]; i++) {
        for(int j=0; j<shape[2]; j++) {
            output[i*shape[2]+j] = 0;
            for(int k=0; k<shape[1]; k++) {
                output[i*shape[2]+j] += input[i*shape[1]+k] * weight[k*shape[2]+j];
            }
            if(shape[3]){
                output[i*shape[2]+j] += bias[i*shape[2]+j];
            }
            if(output[i*shape[2]+j]<0) {
                output[i*shape[2]+j] = 0;
            }
        }
    }
}

void linear_float2(float* input, float* weight, float* bias, float* output, int* shape) {
    for(int i=0; i<shape[0]; i++) {
        for(int j=0; j<shape[2]; j++) {
            output[i*shape[2]+j] = 0;
            for(int k=0; k<shape[1]; k+=4) {
                output[i*shape[2]+j] += input[i*shape[1]+k] * weight[k*shape[2]+j];
                output[i*shape[2]+j] += input[i*shape[1]+k+1] * weight[(k+1)*shape[2]+j];
                output[i*shape[2]+j] += input[i*shape[1]+k+2] * weight[(k+2)*shape[2]+j];
                output[i*shape[2]+j] += input[i*shape[1]+k+3] * weight[(k+3)*shape[2]+j];
            }
            if(shape[3]){
                output[i*shape[2]+j] += bias[i*shape[2]+j];
            }
            if(output[i*shape[2]+j]<0) {
                output[i*shape[2]+j] = 0;
            }
        }
    }
}

#ifdef MULTI_THREAD
void* DNN(void *arg) {
    int ID = *(int*)arg;
    int value;
    sem_getvalue(&task_num, &value);
    printf("%d %d\n", ID, value);
    int if_wait = sem_wait(&task_num);
    sem_getvalue(&task_num, &value);
    printf("%d 55 %d\n", ID, value);
    if(!if_wait) {
        t[ID*2] = (double)GetCycleCount();
        if(Min>t[ID*2]) Min=t[ID*2];
        linear_float2(input[ID], weight, bias, temp[ID], shape);
        linear_float2(temp[ID], weight, bias, output[ID], shape2);
        t[ID*2+1] = (double)GetCycleCount();
        printf("%d %lf %lf\n", ID, t[ID*2], t[ID*2+1]);
        if(Max<t[ID*2+1]) Max=t[ID*2+1];
        tt += (t[ID*2+1]-t[ID*2]);
    }
}
#endif
