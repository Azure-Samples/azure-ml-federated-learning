# Glossary


__Data__ 
<br> 

Any file or collection of files. Data will be described in terms of classification. 
Only three classifications are required for the context of this document. "Sensitive" (cannot be moved or even looked at), "intermediate" (can be moved around, but looser restrictions on visibility), and "eyes-on" (can be moved freely and seen by everyone participating in the federated training). 

__Storage__ 
<br>

Wherever data is stored. In this file, storage is assumed to live in Azure. It may exist in locked-down virtual networks. 

__Compute__ 
<br> 

Anything that can run "code" (deliberately vague). In this file, compute is assumed to live in Azure. 

__Job__ 
<br> 

Execute code (a collection of files) in an environment (a Docker image) against data (from storage). A job can consume data from multiple storage instances and write back to multiple instances. 

__Approval__ 
<br>

REST endpoint to which the platform "asks permission" before running any job. The platform sends the approval endpoint information including: 

1. Input and output storage 
2. Which compute the job wishes to run in 
3. The author of the code the job is running 
4. Whether or not the job has been code-signed by the configured policies 

The approval endpoint can either approve / reject the job based on checked-in configuration (e.g., of which storage accounts are associated with which silo) or pass this information on for manual approval. 

:exclamation: Note that the approval endpoints do not support 3P-facing AML yet. 

__Silo__ 
<br> 

Isolated collection of storage and compute. Here, "isolated" means that the platform guarantees: 

- Only compute within the silo can "touch" storage within the silo. 
- Only data of intermediate or eyes-on classification can be moved outside the silo. 
- Only "approved" jobs can change the classification of data or move it outside the silo. 

Silos are expected to be reliable (i.e., no concerns around network connectivity or uptime). 

:exclamation:  Note that we assume a hard cap of ≤ 100 silos at current stage. 

__Orchestrator__ 
<br> 

Collection of storage and compute. The storage is for model parameters, rather than the actual data. A task orchestrator broadcasts the FL task, sends the current model to each silo, and aggregates the gradients from the silos. In this file, orchestrator is assumed to live in an AML workspace. 

__Internal Silos__ 
<br>

Collection of silos belong to the same Azure tenant.

__External Silos__ 
<br>

Collection of silos that resides in either different Azure tenant or different cloud provider.