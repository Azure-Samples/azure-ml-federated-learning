# Kubernetes Templates

## Contents
This folder contains example yaml templates you can use for creating an on-premises kubernetes (k8s) cluster with access to local data. The use of templates is documented in [this tutorial](../../docs/targeted-tutorials/read-local-data-in-k8s-silo.md).

## List of templates
- [k8s_config.yaml](./k8s_config.yaml): for creating a k8s cluster using [kind](https://kind.sigs.k8s.io/). There is an extra mount added to the cluster, which is used to access the local data.
- [pv.yaml](./pv.yaml), [pvc.yaml](./pvc.yaml), [deploy_pvc](./deploy_pvc.yaml): for creating a [Persistent Volume](https://kubernetes.io/docs/concepts/storage/persistent-volumes/), claiming it, and deploying.