#include "functions.h"

void linear_int(int* input, int* weight, int* bias, int* output, int* shape) {
    int temp[shape[2]];
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
    float temp[shape[2]];
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