<h1 align="center">  
    <p> FedETuning </p>  
</h1>  
 
 <p align="center"> 
	 <img alt="SMILELAB" src="https://img.shields.io/badge/owner-SMILELAB-orange">
	 <img alt="Licence" src="https://img.shields.io/badge/License-Apache%202.0-yellow">
	 <img alt="Build" src="https://img.shields.io/badge/build-processing-green">
 </p>
 
<h3 align="center">  
    <p> 致力于联邦自然语言处理发展 </p>  
</h3>  
  
1️⃣ 本项目来自 [SMILELab-FL](https://github.com/SMILELab-FL/FedLab-NLP) , 我们主要致力于联邦自然语言处理的发展；  
2️⃣ `FedETuning`是一款轻量级、可自由定制联邦学习过程并面向于自然语言处理的代码框架；  
3️⃣ `FedETuning`主要实现基于 [FedLab](https://github.com/SMILELab-FL/FedLab) 和 [HuggingFace](https://github.com/huggingface/transformers)，能完成自然语言预训练大模型在联邦学习场景下工作；  
4️⃣ 任何使用问题可以联系 :e-mail: iezhuo17@gmail.com
  
## Installation

### 目录组织  
我们建议按照如下的文件结构来使用`FedETuning`:    

    data:  存放数据集，注意该数据集需要已经经过 iid/niid 划分的联邦数据集;
    pretrained:  在nlp文件夹里存储从huggingface中下载的预训练模型，如`bert-base-uncased`;
    code:  在该文件夹下面使用`FedETuning`;
    output: 存储模型输出的记录等。

目录组织如下：
```grapha  
├── workspace  
│   └── data  
|   |   ├── fedglue  
|   |   └── fedner  
│   ├── pretrained  
│   │   ├── nlp  
│   │   └── cv  
│   ├── output  
│   └── code  
│       └── FedETuning  
```  
  
运行路径生成：  
```bash  
mkdir workspace  
cd workspace  
mkdir data  
mkdir code  
mkdir pretrained  
cd pretrained  
mkdir nlp  
cd ..  
cd code  
```  
 
### 环境安装  
建议运行环境的`python`版本为`3.7+`，我们建议使用`pytorch`版本为`1.10+` 
```bash  
git clone git@git.openi.org.cn:Trustworthy-DL/FedLab-NLP.git  
cd FedETuning  
pip install -r resquirements.txt  
```

## Usage
支持 `fedavg` 联邦学习算法
```bash
bash fed_run.sh {your_file_path}/workspace {task_name} fedavg 10001 {server_gpu_id} {client1_gpu_id} {client2_gpu_id}
```

支持 `centralized` 集中训练学习算法
```bash
bash cen_run.sh {your_file_path}/workspace {task_name} centralized 10001 {server_gpu_id}
```


## License
[Apache License 2.0](https://git.openi.org.cn/Trustworthy-DL/fednlp-dev/src/branch/master/LICENSE)

