# A prototype of Task Execution API relying on Azure ML

This folder contains a prototype of a Task Execution API that relies on Azure ML. It provides functionality similar to the [task execution service](https://ga4gh.github.io/task-execution-schemas/docs/) of GA4GH (Global Alliance for Genomics and Health). It exposes the following functions:
- `list_tasks()`;
- `get_tasks()`;
- `create_task()`;
- `cancel_task()`.

## Current limitations

Since this prototype is just that, a prototype, there are still some limitations. They are listed below. Note that the limitations are not specific to the Azure ML service, and could be lifted by refining the current API implementation.
- Currently only a _single_ executor is supported, and we are not directing the outputs anywhere in particular.
- [Environment](https://docs.microsoft.com/en-us/azure/machine-learning/concept-environments), [Compute](https://docs.microsoft.com/en-us/azure/machine-learning/concept-compute-target), and [Datasets](https://docs.microsoft.com/en-us/azure/machine-learning/concept-data#datasets) need to exist in Azure ML already. We do not support creating them on-the-fly yet.
- We do not support environment variables yet.
- For listing task details, we only support 2 'view' modes: 'mini' and 'full'.

Even with these limitations, we believe the current prototype is sufficicent to demonstrate what a Task Execution API on Azure ML could look like.

## Setup

Here below are the steps you'll need to follow to try the prototype.

1. Set the current directory ('tes-api-on-azure-ml') as your working directory.
2. Create and then activate a new conda environment.
```ps1
conda env create -f environment.yml
conda activate tes-env
```
3. Prepare a config file pointing to your Azure ML workspace and put it in the 'config' directory (see [here](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-configure-environment#workspace) for more information on config files).
4. Make sure you have some datasets, a compute, and an environment already available in your Azure ML workspace. (If you don't, the links in the above section are a good starting point for learning how to create them.)
5. Open the 'demo.py' script, use the config file you just created, point to your data, compute target and environment, then you should be good to run the demo script!