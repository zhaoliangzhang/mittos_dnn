#include "functions.h"

sem_t task_num;


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

void linear_float() {
    sem_init(&task_num, 0, (unsigned int)shape[0]);
    void* retval;

    pthread_t vector_inner[MAX_THREAD_NUM];
    int create_thread_success;
    int end_tread_success;
    for(int i=0; i<MAX_THREAD_NUM; i++) {
        create_thread_success = pthread_create(&vector_inner[i], NULL, inner_product, NULL);
    }
    for(int j=0; j<MAX_THREAD_NUM; j++) {
        end_tread_success = pthread_join(vector_inner[j], &retval);
    }

    sem_destroy(&task_num);
}
