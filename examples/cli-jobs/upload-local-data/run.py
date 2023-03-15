import os
import sys
from distutils.util import strtobool
import shutil
import pathlib
import base64
from azure.keyvault.keys import KeyClient
from azure.keyvault.keys.crypto import CryptographyClient, EncryptionAlgorithm
from azure.identity import DefaultAzureCredential, ManagedIdentityCredential

import os
import argparse
import logging
import sys
import glob


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

    parser.add_argument("--local_data_folder", type=str, required=True, help="")
    parser.add_argument("--destination_folder", type=str, required=True, help="")
    parser.add_argument(
        "--enable_output_encryption", type=strtobool, required=False, default=False
    )
    parser.add_argument(
        "--keyvault",
        type=str,
        required=False,
        default=False,
        help="url to the keyvault (if --enable_output_encryption is True))",
    )
    parser.add_argument(
        "--key_name",
        type=str,
        required=False,
        default=None,
        help="name of the key to draw for encryption (if --enable_output_encryption is True))",
    )

    return parser


def run(args):
    """Run the job using cli arguments provided.

    Args:
        args (argparse.Namespace): parsed arguments
    """
    assert args.enable_output_encryption == False or (
        args.keyvault is not None and args.key_name is not None
    ), "If --enable_output_encryption is True, --keyvault and --key_name must be provided"

    if not args.enable_output_encryption:
        # unencrypted output, just use shutil copytree
        shutil.copytree(args.local_data_folder, args.destination_folder)
    else:
        if os.environ.get("DEFAULT_IDENTITY_CLIENT_ID"):
            # running in AzureML with a compute identity assigned
            credential = ManagedIdentityCredential(
                client_id=os.environ["DEFAULT_IDENTITY_CLIENT_ID"]
            )
        else:
            credential = DefaultAzureCredential()

        # get the key from the keyvault
        key_client = KeyClient(vault_url=args.keyvault, credential=credential)
        key = key_client.get_key(args.key_name)
        print(key.properties)
        # create crypto client to help with encryption
        # crypto_client = CryptographyClient(key=key, credential=credential)
        crypto_client = key_client.get_cryptography_client(args.key_name)
        encryption_algorithm = EncryptionAlgorithm.rsa1_5 # rsa_oaep

        # use glob to loop through all files recursively
        for entry in glob.glob(args.local_data_folder + "/**", recursive=True):
            if os.path.isfile(entry):
                logging.getLogger(__name__).info(f"Encrypting input file {entry}")
                # get name of file
                file_name = pathlib.Path(os.path.basename(entry))

                # get path to the file
                full_input_dir = os.path.dirname(entry)

                # create path to the output
                rel_dir = os.path.relpath(full_input_dir, args.local_data_folder)
                full_output_dir = os.path.join(args.destination_folder, rel_dir)

                # create a name for the output file
                output_file_path = os.path.join(full_output_dir, file_name)

                # create output dir
                if not os.path.isdir(full_output_dir):
                    logging.getLogger(__name__).info(
                        f"Creating output subfolder {full_output_dir}"
                    )
                    os.makedirs(full_output_dir, exist_ok=True)

                if os.path.isfile(output_file_path):
                    logging.getLogger(__name__).warning("Overwriting existing file")
            
                logging.getLogger(__name__).info(
                    f"Encrypting input {entry} to output {output_file_path}"
                )

                with open(entry, "rb") as f:
                    plain_content = f.read()
                with open(
                    output_file_path,
                    "wb",
                ) as f:
                    f.write(
                        crypto_client.encrypt(
                            encryption_algorithm, plain_content
                        ).ciphertext
                    )


def main(cli_args=None):
    """Component main function.

    It parses arguments and executes run() with the right arguments.

    Args:
        cli_args (List[str], optional): list of args to feed script, useful for debugging. Defaults to None.
    """
    # Set logging to sys.out
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    log_format = logging.Formatter("[%(asctime)s] [%(levelname)s] - %(message)s")
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(log_format)
    logger.addHandler(handler)

    # build an arg parser
    parser = get_arg_parser()

    # run the parser on cli args
    args = parser.parse_args(cli_args)

    print(f"Running script with arguments: {args}")
    run(args)


if __name__ == "__main__":
    main()

