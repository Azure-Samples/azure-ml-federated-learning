# Provision a Vanilla Federated Learning Demo (5 mins)

(work in progress)

## Deploy in Azure

```bash
az login
az account set --name <subscription name>
az deployment sub create --template-file .\mlops\bicep\vanilla_demo_setup.bicep --location eastus --parameters demoBaseName="fldemo"
```

## Launch the demo experiment

TODO: figure out how to retrieve config locally

```bash
python .\examples\pipelines\fl_cross_silo_basic\submit.py --example MNIST
```
