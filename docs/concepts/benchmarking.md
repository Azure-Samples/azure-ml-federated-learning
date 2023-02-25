# Real-world Example Benchmarks
For all three real-word examples (PNEUMONIA, NER, CCFRAUD), we perform benchmark testing to assess three main aspects: the training overhead, model performance, and scalability of training under FL setting.

This investigation was based on the real-world examples provided in this repository. It shows that the FL on AzureML implementation is:

**Efficient**: overhead is kept minimal (see [section 2](#2-training-overhead) ),
**Valid**: reaching parity in quality (see [section 3](#3-model-performance)),
**Scalable**: it can adapt to larger datasets by leveraging distributed training (see [section 4](#4-scalability-with-training)).

The main purpose of this benchmark is to show that FL has been implemented correctly and works as expected, rather than to focus on the exact metric value shown below, which will vary based on distinct models and datasets. In the following experiments, the detailed computes, hyperparameters, models, and tasks etc are detailed in Methodology section below.


## Table of contents
- [Methodology](#1-methodology)
    -[Training Overhead & Model Performance](#training-overhead--model-performance)
- [Training Overhead](#2-real-world-example-benchmarks)
- [Model Performance](#3-model-performance)
- [Scalability with Training](#4-scalability-with-training)

## 1. Methodology

Before we jump to the benchmarking, here is table that gives s summary of each examples:

|  Example  |           Problem         | Type of data | Number of samples | Size on disk |
|-----------|---------------------------|--------------|-----------------------------------
| PNEUMONIA |   Binary Classification   |     Image    |
|    NER    | Multi-class Classification|     Text     |
|  CCFRAUD  |   Binary Classification   |    Tabular   |


### Training Overhead & Model Performance
For experiments about **training overhead** and **model performance**, we compared three models:
<ol>
    <li> __FL__ for model trained with FL in 3 silos</li>
    <li>__Centralized-1/3__ for 1 model with 1/3 data</li>
    <li> __Centralized-1__ for 1 model with all data.</li>
</ol>
After each model is trained, it is evaluated with all test data. for each example, we privde a table illustrating all the detials:

#### PNEUMOIA



|  Example        | Price     | # In stock |
|-----------------|-----------|------------|
|       FL        | 1.99      | *7*        |
| Centralized-1/3 | **1.89**  | 5234       |
|  Centralized-1  | **1.89**  | 5234       |



## 2. Training Overhead

For training overhead, there are two main questions of interest: 

<ol>
    <li> What is the extra **wall-clock time** spent on training with FL, compared to train a regular centralized model only with 1/#silo of the data. Wall-clock time refers to the real-world time elapsed for a job from starting to finishing.</li>
    <li> What is the extra **computing time** spent on training with FL, compared to train a regular centralized model with data from all silos combined. Computing time refers to the time spent on all computing resources that the job deployed during running. Therefore in case of FL, the computing time should be calculated as the sum of time spent from all silos.</li>
</ol>

The first point is important as it indicates how quickly customers can get their model results, from job submitted to ended. The second point is essential as it is an indication of the money that customers will spend on all computing resources.  

**Key findngs**: Our benchmark indicates that the overhead on wall-clock time (1% up to 10%) and computing time (<5%) remains small, meaning that the FL implementation is efficient.

### PNEUMONIA
<p align="center">
    <img src="./pics/pneumonia_time.jpg" alt="pneumonia training time" width="550"/>
</p>
For pneumonia, FL takes only 4% longer wall-clock time than centralized-1/3, and about 5% longer computing time than centralized-1.

### NER
<p align="center">
    <img src="./pics/ner_time.jpg" alt="ner training time" width="600"/>
</p>
For ner, FL takes only 1.3% longer wall time than centralized-1/3, and only 0.1% longer computing time than centralized-1.

### CCFRAUD
<p align="center">
    <img src="./pics/cc_time.jpg" alt="ccfraud training time" width="600"/>
</p>
For ccfraud, FL takes 10% longer wall time than centralized model, while about 3% longer computing time than centralized-1.

## 3. Model Performance
Another important assessing factor for FL is the model performance. Here we also aim at two questions: 

<ol>
    <li>  How does the FL model performance compare to the centralized model trained with only partial data, which is the scenario when FL is not supported and data are confidential and restricted to each region. </li> 
    <li> How does the FL model performance compare to the centralized model trained with data from all silos, which is an ideal situation when all data are eyes-on and could be combined. </li> 
</ol>

The first point is to demonstrate the extent of improvements on model performance, when users can use FL to train with much more external data, compared to train with data from one party. The second point is to understand if the distribute-aggregate design of FL has impact on the model performance. 

**Key findngs**: Our benchmark indicates that model performace is boosted with FL comparing to a single model with partial data. It also shows that the model performance with FL is comparable to a single model trained on all data, demonstrating the validity of our implementation.

### PNEUMONIA
<p align="center">
    <img src="./pics/pneumonia_acc.jpg" alt="pneumonia model performance" width="550"/>
</p>
For pneumonia, FL achieves higher accuracy than centralized-1/3, while slightly lower than centralized-1.

### NER
<p align="center">
    <img src="./pics/ner_acc.jpg" alt="ner training time" width="600"/>
</p>
For ner, FL achieves a highest score for all four metrics. Although it is not expected that FL will outperform centeralized-1, it might be becasue the distribute-aggregate fashion improves the generalizability of the final model.


## 4. Scalability with Training

Scalability is critical for industry applications of FL on large datasets. One benefit of using FL on Azure ML is that it supports distributed training (multi GPUs and multi nodes). For this reason, we support distributed training for each real-world example, empowered by Pytorch Distributed Data Parallel (DDP) module. To test the scalability of our implementation, we artifically replicated each datasets by 10 times, and record the training time per epoch for each silo when such data is trained on different number of GPUs.

**Key findngs**: Our benchmark results shows that in all 3 scenarios, we can achieve scalability by adding more nodes and gpus to reduce wall time accordingly.

### PNEUMONIA
<p align="center">
    <img src="./pics/pneumonia_ddp.jpg" alt="pneumonia distributed training time" width="550"/>
</p>
For pneumonia, the training time scales linearly with different number of GPUs for all three silos.

### NER
<p align="center">
    <img src="./pics/ner_ddp.jpg" alt="ner distributed training time" width="600"/>
</p>
For ner, the training time scales linearly with different number of GPUs for all three silos.

### CCFRAUD
<p align="center">
    <img src="./pics/cc_ddp.jpg" alt="ccfraud distributed training time" width="600"/>
</p>
For ccfraud, the training time scales linearly with different number of GPUs for all three silos.
