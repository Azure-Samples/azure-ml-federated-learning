# Add Kaggle credentials to your FL sandbox

Your Azure ML workspace has an attached key vault that can be used to store workspace-level secrets (users of the workspace will have access to it). We can use this to store Kaggle API key so that jobs can download data from Kaggle.

This tutorial shows you how to add your Kaggle credentials to your FL sandbox.

### Locate your workspace attached key vault

You first need to locate your workspace key vault. It is provisioned by default in our [sandboxes](../provisioning/sandboxes.md) and is named `ws-shkv-<demoBaseName>`. You can find the name of your workspace in the Azure portal.

### Option 1: using Azure CLI

1. Let's first obtain your AAD identifier (object id) by running the following command. We'll use it in the next step.

    ```bash
    az ad signed-in-user show --query id
    ```

2. Create a new key vault policy for yourself, and grant permissions to list, set & delete secrets.

    ```bash
    az keyvault set-policy -n <key-vault-name> --secret-permissions list set delete --object-id <object-id>
    ```

    > Note: The AML workspace you created with the aforementioned script contains the name of the key vault. Default is `ws-shkv-fldemo`.

3. With your newly created permissions, you can now create a secret to store the `kaggleusername`.

    ```bash
    az keyvault secret set --name kaggleusername --vault-name <key-vault-name> --value <kaggle-username>
    ```

    > Make sure to provide your *Kaggle Username*.

4. Create a secret to store the `kagglekey`.

    ```bash
    az keyvault secret set --name kagglekey --vault-name <key-vault-name> --value <kaggle-api-token>
    ```

    > Make sure to provide the *[Kaggle API Token](https://www.kaggle.com/docs/api#authentication)*.

### Option 2: using Azure UI

1. In your resource group (provisioned in the previous step), open "Access Policies" tab in the newly created key vault and click "Create".

2. Select *List, Set & Delete* right under "Secret Management Operations" and press "Next".

3. Lookup currently logged in user (using user id or an email), select it and press "Next".

4. Press "Next" and "Create" in the next screens.

    We are now able to create a secret in the key vault.

5. Open the "Secrets" tab. Create two plain text secrets:

    - **kaggleusername** - specifies your Kaggle user name
    - **kagglekey** - this is the API key that can be obtained from your account page on the Kaggle website.
