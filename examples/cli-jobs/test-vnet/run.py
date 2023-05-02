import os
import sys
import argparse
import logging
import glob
import traceback
from distutils.util import strtobool
import socket


def get_arg_parser(parser=None):
    """Parse the command line arguments for merge using argparse.

    Args:
        parser (argparse.ArgumentParser or CompliantArgumentParser):
        an argument parser instance

    Returns:
        ArgumentParser: the argument parser instance

    Notes:
        if parser is None, creates a new parser instance
    """
    # add arguments that are specific to the component
    if parser is None:
        parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--storage", type=str, nargs="+", required=False, default=None)
    parser.add_argument("--container", type=str, required=False, default=None)
    parser.add_argument("--identity", type=str, required=False, default=None)
    parser.add_argument("--fail_on_error", type=strtobool, required=False, default=True)

    return parser


def test_env() -> bool:
    for k, v in os.environ.items():
        print(f"ENV: {k}={v}")

    return True


def test_ws_secrets() -> bool:
    try:
        from azureml.core import Run, Workspace
        from azureml.core.keyvault import Keyvault
    except ImportError:
        print("ERROR: You need to install the azureml-sdk package to run this script")
        return False

    try:
        run = Run.get_context()
        ws = run.experiment.workspace
        kv = ws.get_default_keyvault()

        username = kv.get_secret("kaggleusername")
    except:
        print("ERROR: Failed to get secrets")
        print(traceback.format_exc())
        return False

    print("Sucessfully got username fom ws shared secret store")
    return True


def test_network() -> bool:
    hostname = socket.gethostname()
    ip_address = socket.gethostbyname(hostname)
    print(f"Hostname: {hostname}")
    print(f"IP Address: {ip_address}")

    return True


def test_managed_identity(identity: str = None) -> bool:
    if "DEFAULT_IDENTITY_CLIENT_ID" not in os.environ:
        print("WARNING: DEFAULT_IDENTITY_CLIENT_ID not set in environment")

    try:
        from azure.identity import DefaultAzureCredential
        from azure.identity import ManagedIdentityCredential
    except ImportError:
        print(
            "ERROR: You need to install the azure-storage-blob package to run this script"
        )
        return False

    if identity is not None:
        print(f"Using provided identity client id: {identity}")
        credential = ManagedIdentityCredential(client_id=identity)
    elif "DEFAULT_IDENTITY_CLIENT_ID" in os.environ:
        identity = os.environ["DEFAULT_IDENTITY_CLIENT_ID"]
        print(f"Using environment identity client id: {identity}")
        credential = ManagedIdentityCredential(client_id=identity)
    else:
        print("Using default identity (no env var DEFAULT_IDENTITY_CLIENT_ID set)")
        identity = "cd771eee-a5ff-4b83-8c84-87342f4d44e0"
        credential = ManagedIdentityCredential(client_id=identity)
        # credential = DefaultAzureCredential()

    try:
        print("Testing credential...")
        credential.get_token("https://management.azure.com/.default")
    except Exception as e:
        print("ERROR: Failed to get token with credential")
        print(traceback.format_exc())
        return False

    return True


def test_storage(storage_account_name: str, container: str = None) -> bool:
    fqdn = f"{storage_account_name}.blob.core.windows.net"
    account_url = f"https://{fqdn}"
    print(f"Testing access to account url: {account_url}")
    success = True

    try:
        import socket

        ip_address = socket.gethostbyname(fqdn)
        print(f"Resolved IP address for {fqdn}: {ip_address}")

    except Exception as e:
        print(f"ERROR: Failed to resolve IP address for {fqdn}")
        print(traceback.format_exc())
        return False

    try:
        import ipaddress

        if not ipaddress.ip_address(ip_address).is_private:
            print(f"ERROR: IP address for {fqdn}, {ip_address} is not private")
            success = False
    except Exception as e:
        print(f"ERROR: Failed to parse IP address for {fqdn}, {ip_address}")
        print(traceback.format_exc())
        return False

    try:
        from azure.identity import DefaultAzureCredential, ManagedIdentityCredential
        from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
    except ImportError:
        print(
            "ERROR: You need to install the azure-storage-blob package to run this script"
        )
        return False

    try:
        print("Testing credential and blob service...")
        if "DEFAULT_IDENTITY_CLIENT_ID" in os.environ:
            identity = os.environ["DEFAULT_IDENTITY_CLIENT_ID"]
            print(f"Using environment identity client id: {identity}")
            credential = ManagedIdentityCredential(client_id=identity)
        else:
            print("Using default identity (no env var DEFAULT_IDENTITY_CLIENT_ID set)")
            credential = DefaultAzureCredential()

        # Create the BlobServiceClient object
        blob_service_client = BlobServiceClient(account_url, credential=credential)
    except Exception as e:
        print("ERROR: Connection to blob storage failed")
        print(traceback.format_exc())
        return False

    if container is None:
        try:
            containers = blob_service_client.list_containers()
            for container in containers:
                print(f"Found container: {container.name}")
        except Exception as e:
            print("ERROR: Listing containers failed")
            print(traceback.format_exc())
            return False
    else:
        try:
            container_client = blob_service_client.get_container_client(container)
            print(f"Found container: {container_client.container_name}")

            print("\nListing blobs...")
            # List the blobs in the container
            blob_list = container_client.list_blobs()
            for blob in blob_list:
                print("\t" + blob.name)

        except Exception as e:
            print("ERROR: Getting container failed")
            print(traceback.format_exc())
            return False

    return success


def main(cli_args=None):
    """Component main function.

    It parses arguments and executes run() with the right arguments.

    Args:
        cli_args (List[str], optional): list of args to feed script, useful for debugging. Defaults to None.
    """
    # build an arg parser
    parser = get_arg_parser()

    # run the parser on cli args
    args = parser.parse_args(cli_args)

    print(f"Running script with arguments: {args}")

    success = True
    success &= test_env()
    success &= test_ws_secrets()
    success &= test_network()
    success &= test_managed_identity(args.identity)

    if args.storage:
        for storage in args.storage:
            success &= test_storage(storage, container=args.container)

    if not success:
        raise Exception("Test failed, check out logs above for details")
    else:
        print("All tests succeeded")


if __name__ == "__main__":
    main()
