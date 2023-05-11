# Read local data in an on-premises Kubernetes silo

## Background

In many Federated Learning (FL) applications, the silos (compute + user data) need to be located on-premises. Azure ML can accommodate such external silos, provided the computes are running a Kubernetes (k8s) cluster. When the user data are physically located on the same machine that holds the k8s cluster, the Azure ML job running in the k8s cluster needs access to the local file system.

## Objective and contents

This tutorial will show you how to configure an on-premises k8s cluster, with access to some data in the local file system. We will explain how to create the k8s cluster using [kind](https://kind.sigs.k8s.io/), and how to configure it properly with [Persistent Volumes](https://kubernetes.io/docs/concepts/storage/persistent-volumes/).

> **:grey_exclamation: Note**:
> To mimic _in the cloud_ an on-premises silo, one can create the k8s cluster in an Azure VM instead of a local machine. The steps described in this tutorial still apply.

## Prerequisites

For creating the k8s cluster:

- a machine with mock data located at `/path/to/data/data_file.txt`;
- [kind](https://kind.sigs.k8s.io/) with version >= 0.16.0 (the current tutorial was made using that version)

## Procedure - Provision and configure your local k8s cluster

This section will teach you how to properly set up the k8s cluster to make a local directory accessible from within the Azure ML job. **It is assumed that you have some data in `/path/to/data/data_file.txt`.**

### Create the k8s cluster with an extra mount

First, you will want to create a k8s cluster. Start by cloning this repository onto the machine where you want to create the cluster. Then, adjust the `hostpath` value of the `extraMounts` in [k8s_config.yaml](../../mlops/k8s_templates/k8s_config.yaml) to point to the local directory where your data are located. After that, you can create the cluster using the following command (run it from the root of the repository).

```bash
kind create cluster --config="./mlops/k8s_templates/k8s_config.yaml"
```

> **:grey_exclamation: Notes**:
>
> - If you're using linux, the `hostpath` value should be the verbatim value of the path. If you're using Windows, the path should be modified in the following manner: `C:\path\to\data` &rarr; `/run/desktop/mnt/host/c/path/to/data`.
> - It is probably not a good idea to expose **all** of the local file system.

> **:checkered_flag: Checkpoint**:\
> You can check in Docker that the extra mount has been created. Locate your cluster in the Docker Desktop app, click on the ellipses then on "View Details", then go to the "Inspect" tab.

### Create a Persistent Volume, Claim, and deploy

Now we are going to create a [Persistent Volume](https://kubernetes.io/docs/concepts/storage/persistent-volumes/) (PV) and the associated Claim (PVC), and deploy it.

- To create the PV, first make sure that the `path` value of the `hostPath` entry in [pv.yaml](../../mlops/k8s_templates/pv.yaml) matches the `containerPath` value of the `extraMounts` in [k8s_config.yaml](../../mlops/k8s_templates/k8s_config.yaml). Then run: `kubectl apply -f ./mlops/k8s_templates/pv.yaml`.
- To create the PVC, first make sure that the `name` in the `metadata` of [pvc.yaml](../../mlops/k8s_templates/pvc.yaml) matches the `name` of the `claimRef` in [pv.yaml](../../mlops/k8s_templates/pv.yaml). Also make sure that the metadata has the following `labels` and `annotations`, as **these are the key parts that are [required](https://github.com/Azure/AML-Kubernetes/blob/master/docs/pvc.md) for the Azure ML job to be able to access the local data**:

    ```yaml
    labels:
        app: demolocaldata
        ml.azure.com/pvc: "true"
    annotations:
        ml.azure.com/mountpath: "/mnt/localdata" # The path from which the local data will be accessed during the Azure ML job. You can change that to a different path if you want.
    ```

    Then run: `kubectl apply -f ./mlops/k8s_templates/pvc.yaml`.
- Finally, to deploy, first make sure that the `claimName` in [deploy_pvc.yaml](../../mlops/k8s_templates/deploy_pvc.yaml) matches the PVC name in [pvc.yaml](../../mlops/k8s_templates/pvc.yaml), and that the `mountPath` matches the `path` in [pv.yaml](../../mlops/k8s_templates/pv.yaml). Then run: `kubectl apply -f ./mlops/k8s_templates/deploy_pvc.yaml`.

> **:checkered_flag: Checkpoint**:\
> Before moving forward, we recommend you check that the local data can indeed by accessed from the k8s cluster. To do so, start by getting the name of the pod that was created by the deployment by running `kubectl get pods`. Then, open a bash session on the k8s by running `kubectl exec -it <your-pod-name> bash`. Finally, run `ls <path-in-docker>` to check that the data in that folder are indeed visible (if you didn't change the default values in the yaml files mentioned above, then your `<path-on-docker>` should be `/localdata` - it is simply the `path` in [pv.yaml](../../mlops/k8s_templates/pv.yaml)).

## Next steps

At this point, you should have a properly configured k8s cluster. You now need to create an **external silo** (on-premises) based on your k8s. Do so by following [these instructions](../provisioning/external-silos.md). Make note of the Azure ML compute and datastore names for your newly created silo, as you will need them when time comes to run a job.

**Once you're ready to consume your data in an Azure ML job, the path of the data file will be: `/mnt/localdata/data_file.txt`** (the value of the `ml.azure.com/mountpath` annotation in the PVC with the file name appended to it).

## Additional resources

- The [Quickstart](../quickstart.md) document in the current repository. It might be useful to go through that Quickstart tutorial first to get acquainted with the various Azure resources and terminology.
- The [instructions on how to add an external silo](../provisioning/external-silos.md) to your FL setup (from the current repository).
- The document on [targeted tutorials](../README.md/#targeted-tutorials) in the current repository.
- That [other repository](https://github.com/Azure/AML-Kubernetes) on Kubernetes + Azure Machine Learning, which contains a lot of information for those who want to dig deeper.
