# Provisioning a Federated Learning setup with internal silos
:construction: This is a Work in Progress. :construction:

0. docker
  - go to `mlops` folder
  - run `docker build --file ./internal_silos/Dockerfile -t fl-rp-vanilla .`
  - `docker run -i fl-rp-vanilla`
1. run PS script
  - the working directory should now be `mlops/internal_silos`
  - run `./ps/ProvisionVanillaSetup.ps1` 