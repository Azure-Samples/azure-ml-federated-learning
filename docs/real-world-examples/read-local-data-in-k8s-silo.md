# Tutorial: Read local data in an on-premises Kubernetes silo

:construction: This tutorial is a work in progress. :construction:

To-dos:
- Introduce an argument for the data path
- Clean up the component `run.py` and the `submit.py`
- ...

## Background
In many Federated Learning (FL) applications, the silos (compute + user data) need to be located on-premises. Azure ML can accommodate such external silos, provided they are running in a Kubernetes (k8s) cluster. When the user data are physically located on the same machine that holds the k8s cluster, the Azure ML job running in the k8s cluster needs access to the local file system. 

## Objective and contents of the tutorial
This tutorial will show you how to access, within an Azure ML job running on an on-premises k8s cluster, some data in the local file system. First, we will explain how to create the k8s cluster using [kind](https://kind.sigs.k8s.io/), and how to configure it properly with [Persistent Volumes](https://kubernetes.io/docs/concepts/storage/persistent-volumes/). Then, we will create an external silo based on this k8s cluster. Finally, we will run a simple test job to prove that an Azure ML job running on the k8s cluster can indeed access the local file system.

## Prerequisites
For creating the k8s cluster:
- A machine with mock data located at `/path/to/data/data_file.txt`
- kind
- Anything else???

For attaching it to Azure ML:
- See the prerequisites listed [over there](../provisioning/external-silos.md)

## Procedure

### Provision and configure your local k8s cluster
- Local machine with mock data located at `/path/to/data/data_file.txt`.
- Create k8s cluster using the kind template, check from Docker that the extra mount has been created.
- Create PV, PVC, and deploy. Check by logging in to the cluster that you can actually see the file. Make sure the `label` and `annotation` were added properly.

> Warning: Probably not a good idea to expose **all** of the local file system.

> Note: if you're using Windows, your path should be ...

### Create an external silo based on your k8s cluster
- Point to [these instructions](../provisioning/external-silos.md). Give a super quick overview, but the expectation is that the user will actually click on the link.

### Configure and run a test job
- Clone this repo if you haven't already
- Explain how to adjust config file (workspace config, + compute name)
- Explain how to point to the data (either through the command line, or through the config file)


## Next steps
Now that you have proven that Azure ML jobs running on the k8s cluster can indeed access local data, you can use your newly attached silo in a FL job. You will need to adjust the code in the _readlocaldata_ component to the type of data you are reading, then use that component in your own FL pipeline.

For a description of the example FL pipelines available in the current repository, see [here](../README.md/#real-world-examples).

## Additional resources
- The [Quickstart](../quickstart.md) document in the current repository. It might be useful to go through that Quickstart tutorial first to get acquainted with the various Azure resources and terminology.
- The [instructions on how to add an external silo](../provisioning/external-silos.md) to your FL setup (from the current repository).
- The doc [real-world examples](../README.md/#real-world-examples) in the current repository. 
- That [other repository](https://github.com/Azure/AML-Kubernetes) on Kubernetes + Azure Machine Learning, which contains a lot of information for those who want to dig deeper.
