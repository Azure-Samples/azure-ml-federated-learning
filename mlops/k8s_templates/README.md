# Kubernetes Templates

## Contents
This folder contains example yaml templates you can use for creating kubernetes (k8s) clusters meeting different types of requirements.

> Note: Please keep in mind that for both use cases you need to create an AzureML instance type, process is documented at the end of the [following document](../../docs/provisioning/silo_open_aks_with_cc.md)

## Templates for creating on-premises k8s clusters with access to local data
The use of templates is documented in [this tutorial](../../docs/targeted-tutorials/read-local-data-in-k8s-silo.md).
- [k8s_config.yaml](./k8s_config.yaml): for creating a k8s cluster using [kind](https://kind.sigs.k8s.io/). There is an extra mount added to the cluster, which is used to access the local data.
- [pv.yaml](./pv.yaml), [pvc.yaml](./pvc.yaml), [deploy_pvc](./deploy_pvc.yaml): for creating a [Persistent Volume](https://kubernetes.io/docs/concepts/storage/persistent-volumes/), claiming it, and deploying.

## Templates for creating k8s clusters using Confidential Compute
The use of templates is documented in [this tutorial](../../docs/provisioning/silo_open_aks_with_cc.md).

- ...