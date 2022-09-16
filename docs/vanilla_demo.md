# Provision a Vanilla Federated Learning Demo (5 mins)

(work in progress)

## Deploy in Azure

```bash
az login
az account set --name <subscription name>
az deployment sub create --template-file .\mlops\bicep\vanilla_demo_setup.bicep --location eastus --parameters demoBaseName="fldemo1"
```

## Create toy datasets

TODO: is there a better process here?

```bash
# a pair for eastus
az ml data create --file .\examples\data\mnist_test.yaml -w fldemo1-aml -g fldemo1-rg --set name=mnist-test-eastus
az ml data create --file .\examples\data\mnist_train.yaml -w fldemo1-aml -g fldemo1-rg --set name=mnist-train-eastus

# a pair for westus
az ml data create --file .\examples\data\mnist_test.yaml -w fldemo1-aml -g fldemo1-rg --set name=mnist-test-westus
az ml data create --file .\examples\data\mnist_train.yaml -w fldemo1-aml -g fldemo1-rg --set name=mnist-train-westus

# a pair for westus2
az ml data create --file .\examples\data\mnist_test.yaml -w fldemo1-aml -g fldemo1-rg --set name=mnist-test-westus2
az ml data create --file .\examples\data\mnist_train.yaml -w fldemo1-aml -g fldemo1-rg --set name=mnist-train-westus2
```

## Launch the experiment

TODO: figure out how to retrieve config locally

```bash
python .\examples\pipelines\fl_cross_silo_basic\submit.py --example MNIST
```
