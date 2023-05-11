# Vertically federated cross-geo bank marketing campaign prediction

**Scenario** - This is a short example where we showcase possibilities of using Azure Machine Learning(AML) for training a model for bank marketing campaign in vertical federating learning fashion. The example showcases how to setup training in the case, where part of the data, including the feature space, is owned by the host. We have simulated a FL scenario by splitting the data into **distinct geo-location**. The sample utilizes **tabular data**. In case of default settings (host and 3 contributors) the features are split as follows:
- **host**:`["age", "job", "marital", "education", "default", "housing", "loan"]`
- **contributor 1**: `["contact", "month", "day_of_week", "duration"]`
- **contributor 2**: `["campaign", "pdays", "previous", "poutcome"]`
- **contributor 3**: `["emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"]`

This is also the reason why **host** needs to own both the **lower model**, responsible for creating latent space representation of the input data, as well as **upper model**, responsible for predicting label for the data. Of course the **lower model** will only learn latent representation for features directly owned by the **host**, which in default case means: `["age", "job", "marital", "education", "default", "housing", "loan"]`.

**Dataset** - This example is trained using the Kaggle dataset [**Bank marketing campaigns dataset | Opening Deposit**](https://www.kaggle.com/datasets/volodymyrgavrysh/bank-marketing-campaigns-dataset). This dataset is describing Portugal bank marketing campaign results, where the campaigns were conducted by calling clients and offering them to place a term deposit.

## Install the required dependencies

You'll need python to submit experiments to AzureML. You can install the required dependencies by running:

```bash
conda env create --file ./examples/pipelines/environment.yml
conda activate fl_experiment_conda_env
```

Alternatively, you can just install the required dependencies:

```bash
python -m pip install -r ./examples/pipelines/requirements.txt
```

## Provision an FL sandbox (with your kaggle credentials!)

To run this example, you will need to provision an AzureML workspace ready for Federated Learning. We strongly recommend you use the setup provided in the repository [quickstart](../quickstart.md), make sure you use vnet option as this is recommended for all vertical FL examples. We will use the same names for the computes and datastores created by default during this quickstart.

If you have already provisioned a sandbox during the [quickstart](../quickstart.md) you can reuse it, but you need to **make sure you have added your [Kaggle](https://www.kaggle.com/) credentials** so we can upload the required dataset in each silo. If not, please refer to our [tutorial on how to add your kaggle credentials](../tutorials/add-kaggle-credentials.md) to the workspace secret store before running the following sections.

You can also [re-provision another sandbox](../provisioning/sandboxes.md) with a different base name using the [deploy buttons on the sandbox page](../provisioning/sandboxes.md), and provide your **kaggleUsername** and **kaggleKey** as parameters, so they will be injected securely in your workspace secret store.

:notebook: take note of your workspace name, resource group and subscription id. You will need them to submit the experiment.

## Enable communication
The nodes in the vertical federated learning need to communicate during the training to exchange intermediate outputs and gradients. Current implementation enables communication only between the host and contributors and no contributor to contributor access. The communication can happen on one of the two currently available channels: **sockets** or **Redis streams**.

### Sockets
This is easier to start with as the only **requirement here is that the nodes are interconnected using vnet**. This is also the **default option**. To create this type of communication channel just create instance of a **AMLCommSocket** class with world size, rank of the node and root run id.

### Redis streams
In case it is not feasible to use vnets in your case you can fall back on using Redis. This, however, involves provisioning [Azure Cache for Redis](https://azure.microsoft.com/en-us/products/cache/). Please, make sure that you provision at least **P1 Premium Instance with 6GB cache** for the demos. However, if you would like to use it for your own data and architectures, feel free to scale it according to the needs for messages being sent between the nodes. The reason why we recommend **Premium tier** is due to the network latency and throughput, more information can be found [here](https://learn.microsoft.com/en-us/azure/azure-cache-for-redis/cache-best-practices-performance).

Once you provision *Azure Cache for Redis* (this can be provisioned in whichever subscription you like):
1. Go to *Access keys* and copy *Primary connection string*. 
2. Go to your *Azure Machine Learning workspace* in Azure Portal and click on the *Key Vault* item.
3. Open "Access Policies" tab and click "Create".
4. Select *List, Set & Delete* right under "Secret Management Operations" and press "Next".
5. Lookup currently logged in user (using user id or an email), select it and press "Next".
6. Press "Next" and "Create" in the next screens. We are now able to create a secret in the key vault.
7. Go to *Secrets* and *Generate* new one with previously copied connection string and the following name "amlcomm-redis-connection-string".
8. In your pipeline configuration file change `communication_backend` value from `socket` to `redis`
9. Continue to training section

## Communication encryption

The communication between nodes, which includes intermediate outputs and gradients, can be optionally encoded. This may be very important in case of using *Redis* as communication backend. To enable the encryption follow these steps:

1. Open the config file  `config.yaml` in `examples/pipelines/bank_marketing_vertical/`
2. Set `encrypted` field, under `communication` to `true`

## Run a job to download and store the dataset in each silo

This can all be performed with ease using a data provisioning pipeline. To run it follow these steps:

1. If you are not using the quickstart setup, adjust the config file  `config.yaml` in `examples/pipelines/utils/upload_data/` to match your setup.

2. Submit the experiment by running:

   ```bash
   python ./examples/pipelines/utils/upload_data/submit.py --example BANK_MARKETING_VERTICAL --workspace_name "<workspace-name>" --resource_group "<resource-group-name>" --subscription_id "<subscription-id>"
   ```

   > Note: You can use --offline flag when running the job to just build and validate pipeline without submitting it.

    :star: you can simplify this command by entering your workspace details in the file `config.yaml` in this same directory.

:warning: Proceed to the next step only once the pipeline completes. This pipeline will create data in 3 distinct locations.

## Run the demo experiment

1. If you are **not** using the sandbox default setup, adjust the config file  `config.yaml` in `examples/pipelines/bank_marketing_vertical/` to match your setup.
2. Submit the FL experiment by running:

   ```bash
   python ./examples/pipelines/bank_marketing_vertical/submit.py --workspace_name "<workspace-name>" --resource_group "<resource-group-name>" --subscription_id "<subscription-id>"
   ```

   > Note: You can use --offline flag when running the job to just build and validate pipeline without submitting it.

    :star: you can simplify this command by entering your workspace details in the file `config.yaml` in this same directory.

## Run the demo experiment

Follow the steps in the [Vertical Federated Learning tutorial - Steps to launch](../concepts/vertical-fl.md#steps-to-launch).