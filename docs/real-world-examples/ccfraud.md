# Finance example for Federated Learning in Azure ML

This is a short example where we showcase possibilities of using Azure Machine Learning(AML) for training a model for credit card fraud detection in federating learning fashion. The example utilizes multitude of model architectures to demonstrate versatility of the proposed solution. We have chosen this scenario because we believe it is very common for credit card companies, banks and other similar financial institutions.

:warning: This is an example for demonstration purposes only and the authors of this repository do not take any responsibility for loss or harm made by its usage.


> To make setup of this example easier we will only provision an _open_ setup. Do not upload sensitive data to it! 

# Data

As for the source of the data for this example we have used Kaggle dataset, [**Credit Card Transactions Fraud Detection Dataset**](https://www.kaggle.com/datasets/kartik2112/fraud-detection?datasetId=817870&sortBy=voteCount&types=competitions) generated using simulation that contains both genuine and fraudulent transactions. Compared to other dataset this one does not only contain an abstraction of underlying features but rather all features in their raw format. This allows us to split data into distinct geo-location and also let us show how to do a simple **preprocessing** on the **tabular data**.

# Getting started

### Provisioning
Please follow the instructions in the [quickstart](../quickstart.md) to provision an open sandbox. Make note of the name of the resource group you provisioned, as well as the name of the workspace.

### Kaggle credentials
In the resource group, provisioned in the previous step, lookup the Key Vault and open "Secrets" tab. Here we will need to create two plian text secrets with following names and contents:
- **kaggleusername** - specifies your kaggle user name
- **kagglekey** - this is API key that can be obtained from your profile on the kaggle

### Data provisioning
Before we dive deep into training the model we need to upload the data from Kaggle dataset into the silo datastores in corresponding geolocations. This can all be performed with ease using **data provisioning pipeline**. To run it follow these steps:

- Navigate to: `examples/pipelines/utils/upload_data/` folder
- Run `submit.py --submit` command

:warning: Proceed to next step only one the pipeline finishes.

### Model choice
Please update [**config**](../../examples/pipelines/ccfraud/config.yaml), field `model_name` in the `training_parameters` section, to reflect desired model to be trained, options include: SimpleLinear, SimpleLSTM, SimpleVAE

- **SimpleLinear** - model fully composed of `torch.Linear` layers with `ReLU` activations, takes data as-is sample-by-sample
- **SimpleLSTM** - model composed by 4 LSTM layers connected to linear head with architecture similar to **SimpleLinear**, takes data ordered by time in sequences that overlap each other
- **SimpleVAE** - model composed of 2 encoder LSTM layers and 2 decoder LSTM layers that tries to recreate consumed sequence of transactions, the latent space created by encoder is consumed by a linear layer to perform prediction, takes data ordered by time in sequences that overlap each other

## Running example
- Update [**config**](../../examples/pipelines/ccfraud/config.yaml) to reflect your AML FL orchestration setup
- Update configuration names in the `preprocessing/config` folder to reflect names of the computer in your own AML FL orchestration setup
- Run the example by running `submit.py` file using python environment from the `pipelines` folder

