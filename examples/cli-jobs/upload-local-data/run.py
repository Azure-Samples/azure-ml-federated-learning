import os
import sys
import argparse
import logging
import glob
import shutil
import pathlib
from distutils.util import strtobool

from azure.keyvault.keys import KeyClient
from azure.keyvault.keys.crypto import CryptographyClient, EncryptionAlgorithm
from azure.identity import DefaultAzureCredential, ManagedIdentityCredential
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import padding


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

    parser.add_argument("--input_folder", type=str, required=True, help="")
    parser.add_argument("--output_folder", type=str, required=True, help="")
    parser.add_argument(
        "--method",
        type=str,
        choices=["encrypt", "decrypt", "copy"],
        required=False,
        default="copy",
    )

    parser.add_argument(
        "--keyvault",
        type=str,
        required=False,
        default=None,
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


def encrypt_file_aes(input_file_path, output_file_path, rsa_client):
    """Encrypt a file using AES and RSA.

    Args:
        input_file_path (str): path to the input file
        output_file_path (str): path to the output file
        rsa_client (CryptographyClient): client to use for RSA encryption

    Returns:
        None
    """
    # create a random AES key, encrypted using RSA public key
    aes_key = os.urandom(32)
    encrypted_aes_key = rsa_client.encrypt(
        EncryptionAlgorithm.rsa1_5, aes_key
    ).ciphertext
    print(len(encrypted_aes_key))

    # plus generate random IV
    iv = os.urandom(16)

    # create classes
    cipher = Cipher(algorithms.AES(aes_key), modes.CBC(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    padder = padding.PKCS7(algorithms.AES.block_size).padder()

    # read input data as a whole (for now)
    with open(input_file_path, "rb") as f:
        data = f.read()

    with open(output_file_path, "wb") as f:
        f.write(encrypted_aes_key)
        f.write(iv)
        f.write(encryptor.update(padder.update(data) + padder.finalize()))


def decrypt_file_aes(input_file_path, output_file_path, rsa_client):
    """Decrypt a file using AES and RSA.

    Args:
        input_file_path (str): path to the input file
        output_file_path (str): path to the output file
        rsa_client (CryptographyClient): client to use for RSA decryption

    Returns:
        None
    """
    # read the data from the input file
    with open(input_file_path, "rb") as f:
        encrypted_aes_key = f.read(256)
        iv = f.read(16)
        data = f.read()

    # decrypt the key using rsa_client
    aes_key = rsa_client.decrypt(
        EncryptionAlgorithm.rsa1_5, encrypted_aes_key
    ).plaintext

    decryptor = Cipher(algorithms.AES(aes_key), modes.CBC(iv)).decryptor()
    unpadder = padding.PKCS7(algorithms.AES.block_size).unpadder()
    with open(output_file_path, "wb") as f:
        f.write(unpadder.update(decryptor.update(data)) + unpadder.finalize())


def run(args):
    """Run the job using cli arguments provided.

    Args:
        args (argparse.Namespace): parsed arguments
    """
    assert args.method == "copy" or (
        args.keyvault is not None and args.key_name is not None
    ), "If --method is encrypt or decrypt, --keyvault and --key_name must be provided"

    if args.method == "copy":
        # unencrypted output, just use shutil copytree
        shutil.copytree(args.input_folder, args.output_folder)
    else:
        if os.environ.get("DEFAULT_IDENTITY_CLIENT_ID"):
            # running in AzureML with a compute identity assigned
            credential = ManagedIdentityCredential(
                client_id=os.environ["DEFAULT_IDENTITY_CLIENT_ID"]
            )
        else:
            credential = DefaultAzureCredential()

        # get a client to the keyvault
        key_client = KeyClient(vault_url=args.keyvault, credential=credential)

        # create crypto client to help with encryption
        crypto_client = key_client.get_cryptography_client(key_name=args.key_name)

        # use glob to loop through all files recursively
        for entry in glob.glob(args.input_folder + "/**", recursive=True):
            if os.path.isfile(entry):
                logging.getLogger(__name__).info(f"Encrypting input file {entry}")
                # get name of file
                file_name = pathlib.Path(os.path.basename(entry))

                # get path to the file
                full_input_dir = os.path.dirname(entry)

                # create path to the output
                rel_dir = os.path.relpath(full_input_dir, args.input_folder)
                full_output_dir = os.path.join(args.output_folder, rel_dir)

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

                if args.method == "encrypt":
                    # encrypt the file using a random AES key encrypted using RSA key from keyvault
                    encrypt_file_aes(entry, output_file_path, crypto_client)
                elif args.method == "decrypt":
                    decrypt_file_aes(entry, output_file_path, crypto_client)


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
