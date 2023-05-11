#################################################################################################
#                                           WARNING                                             #
#################################################################################################
# Should this file change please update all copies of confidential_io.py file in the repository #
#################################################################################################

import os
import io
from azure.keyvault.keys import KeyClient
from azure.keyvault.keys.crypto import CryptographyClient, EncryptionAlgorithm
from azure.identity import DefaultAzureCredential, ManagedIdentityCredential
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import padding
import logging
import warnings
from distutils.util import strtobool

_MANAGED_IDENTITY = None
_KEYVAULT_URL = None
_RSA_KEY_NAME = None
_RSA_CRYPTO_CLIENT = None

# WARNING !!!
_CONFIDENTIALITY_DISABLED = bool(
    strtobool(os.environ.get("CONFIDENTIALITY_DISABLE", "False"))
)

if _CONFIDENTIALITY_DISABLED:
    warnings.warn(
        "confidentiality has been intentionally disabled using CONFIDENTIALITY_DISABLE=True, all outputs will be left in clear."
    )


def config_global_rsa_key(keyvault_url=None, rsa_key_name=None, managed_identity=None):
    """Call this to set the keyvault url and key name to use for encryption
    at a global level. This will be used by all calls to get_rsa_client() and
    EncryptedFile().

    Args:
        keyvault_url (str): The url of the keyvault to use.
        rsa_key_name (str): The name of the RSA key to use.
        managed_identity (str): The client id of the managed identity to use.
    """
    global _KEYVAULT_URL, _RSA_KEY_NAME, _MANAGED_IDENTITY
    _KEYVAULT_URL = (
        _KEYVAULT_URL
        or keyvault_url
        or os.environ.get("CONFIDENTIALITY_KEYVAULT", None)
    )
    _RSA_KEY_NAME = (
        _RSA_KEY_NAME
        or rsa_key_name
        or os.environ.get("CONFIDENTIALITY_KEY_NAME", None)
    )
    _MANAGED_IDENTITY = (
        _MANAGED_IDENTITY
        or managed_identity
        or os.environ.get("DEFAULT_IDENTITY_CLIENT_ID", None)
    )


def get_rsa_client(keyvault_url=None, rsa_key_name=None, managed_identity=None):
    """Get crypto client from Keyvault for RSA key.

    Args:
        keyvault_url (str): The url of the keyvault to use.
        rsa_key_name (str): The name of the RSA key to use.
        managed_identity (str): The client id of the managed identity to use.

    Returns:
        CryptographyClient: The crypto client to use for encryption.
    """
    global _RSA_CRYPTO_CLIENT, _KEYVAULT_URL, _RSA_KEY_NAME, _MANAGED_IDENTITY

    # let's not recreate a new client
    if _RSA_CRYPTO_CLIENT is not None:
        logging.getLogger(__name__).info(
            "CryptographyClient already initialized, returning existing instance."
        )
        return _RSA_CRYPTO_CLIENT

    # if not done already, capture references from env vars
    config_global_rsa_key(keyvault_url=keyvault_url, rsa_key_name=rsa_key_name)

    if _KEYVAULT_URL is None:
        raise ValueError(
            "To connect to the keyvault you need to provide its url, call config_global_rsa_key(keyvault_url, rsa_key_name) first or call config_global_rsa_key() with keyvault_url not None"
        )
    if _RSA_KEY_NAME is None:
        raise ValueError(
            "To connect to the keyvault you need to provide a key name, call config_global_rsa_key(keyvault_url, rsa_key_name) first or call config_global_rsa_key() with rsa_key_name not None"
        )

    # get credentials
    if _MANAGED_IDENTITY:
        # running in AzureML with a compute identity assigned
        logging.getLogger(__name__).info(
            f"Using ManagedIdentityCredential with client_id={_MANAGED_IDENTITY} to connect to keyvault"
        )
        credential = ManagedIdentityCredential(client_id=_MANAGED_IDENTITY)
    else:
        logging.getLogger(__name__).info(
            f"Using DefaultAzureCredential to connect to keyvault"
        )
        credential = DefaultAzureCredential()

    # get a client to the keyvault
    logging.getLogger(__name__).info(f"Connecting to keyvault {_KEYVAULT_URL}...")
    key_client = KeyClient(vault_url=_KEYVAULT_URL, credential=credential)

    # create crypto client to help with encryption
    logging.getLogger(__name__).info(
        f"Obtaining CryptographyClient for key {_RSA_KEY_NAME}..."
    )
    _RSA_CRYPTO_CLIENT = key_client.get_cryptography_client(key_name=_RSA_KEY_NAME)

    return _RSA_CRYPTO_CLIENT


def read_encrypted_file(file_path, rsa_client):
    """Read an encrypted file and return the decrypted bytes.

    Args:
        file_path (str): The path to the file to read.
        rsa_client (CryptographyClient): The crypto client to use for decryption.

    Returns:
        bytes: The decrypted bytes.
    """
    # read the data from the input file
    with open(file_path, "rb") as f:
        encrypted_aes_key = f.read(
            256
        )  # TODO: figure out the right size based on rsa_client
        iv = f.read(16)
        data = f.read()

    # decrypt the key using rsa_client
    aes_key = rsa_client.decrypt(
        EncryptionAlgorithm.rsa1_5, encrypted_aes_key
    ).plaintext

    decryptor = Cipher(algorithms.AES(aes_key), modes.CBC(iv)).decryptor()
    unpadder = padding.PKCS7(algorithms.AES.block_size).unpadder()
    plain_bytes = unpadder.update(decryptor.update(data)) + unpadder.finalize()

    return plain_bytes


def write_encrypted_file(
    output_file_path, input_file_handler, rsa_client, mode="t", encoding="UTF-8"
):
    """Write an encrypted file from the input file handler.

    Args:
        output_file_path (str): The path to the file to write.
        input_file_handler (file): The file handler to read the data from.
        rsa_client (CryptographyClient): The crypto client to use for encryption.
        mode (str): The mode to use for encryption, can be "t" for text or "b" for binary.
        encoding (str): The encoding to use for text mode.
    """
    # create a random AES key, encrypted using RSA public key
    aes_key = os.urandom(32)
    encrypted_aes_key = rsa_client.encrypt(
        EncryptionAlgorithm.rsa1_5, aes_key
    ).ciphertext

    # plus generate random IV
    iv = os.urandom(16)

    # create classes
    cipher = Cipher(algorithms.AES(aes_key), modes.CBC(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    padder = padding.PKCS7(algorithms.AES.block_size).padder()

    # read input data as a whole (for now)
    data = input_file_handler.read()
    if "t" in mode:
        data = data.encode(encoding)

    with open(output_file_path, "wb") as f:
        f.write(encrypted_aes_key)
        f.write(iv)
        f.write(encryptor.update(padder.update(data) + padder.finalize()))


class EncryptedFile:
    def __init__(self, file_path, mode="rt", rsa_client=None, encoding="UTF-8"):
        """Create an encrypted file.

        Args:
            file_path (str): The path of the file to create.
            mode (str, optional): The mode to open the file in. Defaults to "t".
            rsa_client (CryptographyClient, optional): The crypto client to use for encryption. Defaults to None.
            encoding (str, optional): The encoding to use. Defaults to "UTF-8".
        """
        assert (
            "r" in mode or "w" in mode
        ), f"mode={mode} is not supported, use either 'r', 'w' only"
        assert (
            "t" in mode or "b" in mode
        ), f"mode={mode} is not supported, use either 't', 'b' only"

        if not _CONFIDENTIALITY_DISABLED:
            self._rsa_client = rsa_client or get_rsa_client()
        self._file_path = file_path
        self._file = None
        self._mode = mode
        self._encoding = encoding

    def _new_buffer(self, existing_bytes=None, mode="t", encoding="UTF-8"):
        """Create a new buffer to read or write to.

        Args:
            existing_bytes (bytes, optional): The existing bytes to read from. Defaults to None.
            mode (str, optional): The mode to open the file in. Defaults to "t".
            encoding (str, optional): The encoding to use. Defaults to "UTF-8".

        Returns:
            io.StringIO or io.BytesIO: The new buffer.
        """
        if "t" in mode:
            if existing_bytes is None:
                return io.StringIO()
            else:
                return io.StringIO(existing_bytes.decode(encoding))
        elif "b" in mode:
            if existing_bytes is None:
                return io.BytesIO()
            else:
                return io.BytesIO(existing_bytes)
        else:
            raise ValueError(
                f"mode={self._mode} is not supported, use either 't' or 'b' only"
            )

    def __enter__(self):
        """Open the file and return the buffer."""
        # disabling confidentiality means just opening as a regular file
        if _CONFIDENTIALITY_DISABLED:
            self._file = open(self._file_path, mode=self._mode)
            return self._file

        if "r" in self._mode:
            existing_bytes = read_encrypted_file(self._file_path, self._rsa_client)
        else:
            existing_bytes = None

        self._file = self._new_buffer(
            existing_bytes, mode=self._mode, encoding=self._encoding
        )

        return self._file

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close the file and write the buffer to the file."""
        # disabling confidentiality means just closing as a regular file
        if _CONFIDENTIALITY_DISABLED:
            self._file.close()
            return

        if "w" in self._mode:
            self._file.seek(0)
            write_encrypted_file(
                self._file_path,
                self._file,
                self._rsa_client,
                mode=self._mode,
                encoding=self._encoding,
            )

        self._file.close()
