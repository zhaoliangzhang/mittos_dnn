# TOOL INSTRUMENT

## SCALE-Sim  
1. Download SCALE-Sim  
* The git link as follows:  
https://github.com/ARM-software/SCALE-Sim  
You can also find more details in the paper of SCALE-Sim:  
https://arxiv.org/abs/1811.02883
2. Define hardware architecture  
* We can use the architecture already defined by developer, which can be found in the git of SCALE-Sim. In the directory "configs/"   
* If you want to use the architecture of your own, create a .cfg file first, then define your architecture. An example of config file is shown as below:  
![Image text](https://raw.githubusercontent.com/AnandS09/SCALE-Sim/master/images/config_example.png)  
The defination of parameters as follows:  
![Image text](https://raw.githubusercontent.com/zhaoliangzhang/mittos_dnn/master/image/config_parameter.PNG)  
This is the Table 1 in SCALE-Sim, you also find it in the paper.  
The "Topology" is not necessary, I tried to add it but seems it will not use the topology file you mentioned in config file.  

3. Create topology file  
* SCALE-Sim accept a csv file contains the network we want to simulate. The format of topology file is shown below.  
![Image text](https://raw.githubusercontent.com/AnandS09/SCALE-Sim/master/images/yolo_tiny_csv.png)  
The defination of parameters as follows:
![Image text](https://raw.githubusercontent.com/zhaoliangzhang/mittos_dnn/master/image/topology_parameter.PNG)  
This is the Table 2 in the paper.
* If your network contains branch topology, the SCALE-Sim will simulate it serially.  

4. Run the simulation  
* type the command below in terminal  
 `python scale.py -arch_config=your_architecture.cfg -network=your_topology.csv`  
The .cfg and .csv file mean your hardware configuration and network topology  
The files I use are "google.cfg" and "mynet.csv", which are in the same directory as this instrument. "google.cfg" is the hardware architecture developed by SCALE-Sim developers and "mynet.csv" is a topology of a simple DNN, it has 40 input features, 1 output feature and one hidden layer with 128 features. The simulation result as follows:  
![Image text](https://raw.githubusercontent.com/zhaoliangzhang/mittos_dnn/master/image/SCALE_Sim_Result.PNG)

## GPGPUSim  
1. Download Source codes  
* The git link as follow:  
https://github.com/gpgpu-sim/gpgpu-sim_distribution.git  
* After download, change the git repository to "dev" branch

2. Enviroment required  
* GPGPU-Sim dependencies:  
gcc  
g++  
make  
makedepend  
xutils  
bision  
flex  
zlib  
CUDA Toolkit (less or equal 9.1)

* GPGPU-Sim documentation dependencies:  
doxygen  
graphvi  

* AerialVision dependencies:  
python-pmw  
python-ply  
python-numpy  
libpng12-dev  
python-matplotlib  

3. Build  
* The codes provide a script to setup build enviroment named "setup_enviroment", it can change your CUDA path to GPGPU-Sim, if you want to change it back to CUDA you installed, just run "source .bashrc.sh" at home directory of current user.
* Make sure your are working at the top directoy of GPGPU-SIm, type codes below:  
`source setup_environment <build_type>`  
"build_type" could be "debug" or "release", default is release if no parameter found.  
* Just run:  
`make`  
then wait it done, it should take sevarl minutes  

4. Run  
* First compile your application written in CUDA using nvcc  
An example of CUDA application is in the directory named "gpu" of this repository, a Makefile is included, may help you understand how to use nvcc.  
* Copy the config files to your application working directory, for example:  
`cp configs/deprecated-cfgs/SM6_GTX1080/* your_working_directory/`  
* Then you may need to modify the file you copied named "gpgpusim.config", here're some useful hints:  
* * -gpgpu_ptx_sim_mode  
if the option is setted as 1, then the simulator will just excute your application just like running it on a GPU, without telling anything about the hardware performance such as cycles used and cache history. If you just want to confirm the correctness of your codes, it will be a nice try because such mode takes less time.
* * -visualizer_enabled  
if this option is setted as 1, then visulization files will be generated, the option "-visualizer_outputfile" means the name of the visualization file
* * -trace_enabled  
if this option is setted as 1, then the activity of every cycle will be printed and it could be annoying and you can set it to 0
* * there are also other options to change the architecture of GPU, such as -gpgpu_n_clusters, -gpgpu_num_sp_units, -gpgpu_num_sfu_units, -gpgpu_shmem_num_bank, etc. You can explore more of the file.

* After modifying the config file, run `ldd name_of_your_application` to confirm your application is using libcudart.so in GPGPU-Sim. Here is an example, you may get a result like this  
![Image](https://raw.githubusercontent.com/zhaoliangzhang/mittos_dnn/master/image/linked_library_check.PNG)
Then just run your application in terminal. The link below shows the meaning of the output  
http://gpgpu-sim.org/manual/index.php/Main_Page#Understanding_Simulation_Output  

* The link below is an instruction of visulization tool  
http://gpgpu-sim.org/manual/index.php/Main_Page#Visualizing_High-Level_GPGPU-Sim_Microarchitecture_Behavior  
Be aware that the tool requires python2, not python3  

## onnx2ncnnn  
* We need to convert the model generated by machine learning toolkits (pytorch, tensorflow) to a simple binary file which we can read it easily using C codes. 
* Open Neural Network Exchange (ONXX) is a standard to present machine learning model, the main toolkit except tensorflow officially support the convertion of current model to onnx or onnx to current platform, but onnx also provides tools of convertion between tensorflow. So we can convert the models to onnx first and convert onnx to a binary file.
* Here's an example to save model as onnx in pytorch:  
https://pytorch.org/docs/master/onnx.html?highlight=onnx#module-torch.onnx  
* Here's the link to convert the tensorflow model to onnx:  
https://github.com/onnx/tensorflow-onnx  
* Convert onnx to binary file  
We use the tool in ncnn:  
https://github.com/Tencent/ncnn/tree/master/tools/onnx  
Just compile the codes and you can gain a executable file named "onnx2ncnn", then execute it:  
`./onnx2ncnn <onnx_file> <topology_file> <weight_file>`  
onnx_file is the onnx model file as the input, the topology_file and weight_file are output files and these two parameters can be missed and using the default value. toplogy_files is a chart present the topology of network, the instruction is as follow:  
https://github.com/Tencent/ncnn/wiki/operation-param-weight-table  
weigth_file is the binary file contains weight of network. Be aware, before every group, there's a float key to point the data type of the parameters, the explaintion as follows:  
https://github.com/Tencent/ncnn/wiki/param-and-model-file-structure  
* If you meet problem while compiling, there is an executable file already compiled in this directory. 