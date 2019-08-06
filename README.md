# mittos_dnn

This is a C-based implement of DNN, the purpose is apply it to mittos.

---
## c
The C implement.  
After changing the work dictionary to this folder, use command "make" to compile.
## pytorch
The pytroch example.
Use the command below to run:
>python main.py <parameter\>

parameters can be as follows:
* train: train a new model from start
* test: test data using model trained
* print: print weights of trained model

## gpu
The CUDA example  
Use command "make" to compile after entering this folder.

---

## Running enviroment

gcc v7.4.0  
pytorch-cpu v1.1.0  
CUDA compilation tools, release 9.1