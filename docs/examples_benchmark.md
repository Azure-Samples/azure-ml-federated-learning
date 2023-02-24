# Real-world Example Benchmarks
For all three real-word examples (PNEUMONIA, NER, CCFRAUD), we perform benchmark testing to assess three main aspects --- The training overhead, model performance, and scalability of training under FL setting. The main purpose of this benchmark is to show that FL has been implemented correctly and work as expected, rather than to focus on the exact metric value shown below, which will vary based on distinct models and datasets. In the following experiments, each silo is trained with the same hyperparameters to ensure the consistency of results, and the model performance was assessed with all test data combined from silos. Below are the results from each aspect.

## Table of contents

- [Training Overhead](#real-world-example-benchmarks)
- [Model Performance](#model-performance)
- [Scalability with Training](#scalability-with-training)

## Training Overhead

For training overhead, the main question of interest is mainly two-fold: 1. What is the extra wall-clock time spent on training with FL, compared to train a regular centralized model, but only with 1/#silo of the data. 2. What is the extra computing time (summed from all silos) spent on training with FL, compared to train a regular centralzied model with data from all silos combined. Thie first point is important as they indicate how quickly customers can get their model results, and the second point is essential as an indication of the money that customers will spend on computing resource. For this reason, we tested 3 models -- "FL" for model trained with FL in 3 silos, "Centralized-1/3" for 1 model with 1/3 data, and "Centralized-1" for 1 model with all data combined. 

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

## Model Performance
Another important factor that can assess the performance of FL is the model performance. Here we also aims to answer two questions 1. How does the FL model performance compare to the centralized model trained with only partial data, which is the scenario when FL is not supported and data are confidential and restricted to each region. 2. How does the FL model performance compare to the centralized model trained with data from all silos, which is an ideal situation when all data are eye-on and could be combined. This is to understand if the distributed-aggregate design of FL has impact on the model performance. For this reason, we tested 3 models -- "FL" for model trained with FL in 3 silos, "Centralized-1/3" for 1 model with 1/3 data, and "Centralized-1" for 1 model with all data combined. 

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


## Scalability with Training

For each real-world example, we also support distributed training within each silo, empowered by Pytorch Distributed Data Parallel (DDP) module. To test the scalability of our implementation, we artifically replicated each datasets by 10 times, and record the training time per epoch for each silo when such data is trained on different number of GPUs.

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
