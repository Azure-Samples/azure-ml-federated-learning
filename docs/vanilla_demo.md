# Provision a Vanilla Federated Learning Demo (5 mins)

(work in progress)

## Deploy in Azure

```bash
az login
az account set --name <subscription name>
az deployment sub create --template-file .\mlops\bicep\vanilla_demo_setup.bicep --location eastus --parameters demoBaseName="fldemo1"
```

## Create toy datasets

```bash
# a pair for eastus
az ml data create --file .\examples\data\mnist_test.yaml -w fldemo8-aml -g fldemo8-rg --set name=mnist-test-eastus
az ml data create --file .\examples\data\mnist_train.yaml -w fldemo8-aml -g fldemo8-rg --set name=mnist-train-eastus

# a pair for westus
az ml data create --file .\examples\data\mnist_test.yaml -w fldemo8-aml -g fldemo8-rg --set name=mnist-test-westus
az ml data create --file .\examples\data\mnist_train.yaml -w fldemo8-aml -g fldemo8-rg --set name=mnist-train-westus

# a pair for westus2
az ml data create --file .\examples\data\mnist_test.yaml -w fldemo8-aml -g fldemo8-rg --set name=mnist-test-westus2
az ml data create --file .\examples\data\mnist_train.yaml -w fldemo8-aml -g fldemo8-rg --set name=mnist-train-westus2
```

