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

void linear_int2(int* input, int* weight, int* bias, int* output, int* shape) {
    for(int i=0; i<shape[0]; i++) {
        for(int j=0; j<shape[2]; j++) {
            output[i*shape[2]+j] = 0;
            for(int k=0; k<shape[1]; k++) {
                output[i*shape[2]+j] += input[i*shape[1]+k] * weight[k*shape[2]+j];
            }
        }
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
   /*for(int i=0; i<shape[0]; i++) {
        for(int j=0; j<shape[2]; j++) {
            output[i*shape[2]+j] = 0;
            for(int k=0; k<shape[1]; k++) {
                output[i*shape[2]+j] += input[i*shape[1]+k] * weight[k*shape[2]+j];
            }
        }
    }*/
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
        }
    }
}

void* inner_product(void* arg) {
    while(!sem_trywait(&task_num)) {
        int row = exist_job;
        exist_job -= 1;
        output[row] = 0;
        for(int i=0; i<shape[1]; i++) {
            output[row] += input[row*shape[1]+i]*weight[i];
        }
    }
}
