# Tools for Provisioning

:construction: Work In Progress :construction:

This directory contains tools to help provisioning Federated Learning setups. It is organized as follows.
- `external_silos`: tools to help provisions _external_ silos, _e.g._, silos that are NOT in the same Azure tenant as the orchestrator.
- `internal_silos`: tools to help provisions _internal_ silos, _e.g._, silos that are in the same Azure tenant as the orchestrator.
- `shared`: resources (ARM templates, PS scripts, etc.) shared by the above two directories.