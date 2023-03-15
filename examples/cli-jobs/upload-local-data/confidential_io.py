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

_MANAGED_IDENTITY = None
_KEYVAULT_URL = None
_RSA_KEY_NAME = None
_RSA_CRYPTO_CLIENT = None


def config_global_rsa_key(
    keyvault_url, rsa_key_name, managed_identity=None, lazy_init=True
):
    global _KEYVAULT_URL, _RSA_KEY_NAME, _MANAGED_IDENTITY
    _KEYVAULT_URL = keyvault_url
    _RSA_KEY_NAME = rsa_key_name
    _MANAGED_IDENTITY = managed_identity or os.environ.get(
        "DEFAULT_IDENTITY_CLIENT_ID", None
    )

    if not lazy_init:
        # initialize right away
        get_rsa_client()


def get_rsa_client(keyvault_url=None, rsa_key_name=None, managed_identity=None):
    """Get crypto client from Keyvault for RSA key."""
    global _RSA_CRYPTO_CLIENT, _KEYVAULT_URL, _RSA_KEY_NAME, _MANAGED_IDENTITY
    # let's not recreate a new client
    if _RSA_CRYPTO_CLIENT is not None:
        logging.getLogger(__name__).info(
            "CryptographyClient already initialized, returning existing instance."
        )
        return _RSA_CRYPTO_CLIENT

    # verify input values against internal global variables
    if keyvault_url:
        _KEYVAULT_URL = keyvault_url
    if rsa_key_name:
        _RSA_KEY_NAME = rsa_key_name

    if _KEYVAULT_URL is None:
        raise ValueError(
            "To connect to the keyvault you need to provide its url, call config_global_rsa_key(keyvault_url, rsa_key_name) first or call get_rsa_client() with keyvault_url not None"
        )
    if _RSA_KEY_NAME is None:
        raise ValueError(
            "To connect to the keyvault you need to provide a key name, call config_global_rsa_key(keyvault_url, rsa_key_name) first or call get_rsa_client() with rsa_key_name not None"
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


class read_encrypted:
    def __init__(self, file_path, mode="t", rsa_client=None, encoding="UTF-8"):
        assert mode in [
            "t",
            "b",
        ], f"mode={mode} is not supported, use either 't' or 'b' only"

        self._rsa_client = rsa_client or get_rsa_client()
        self._file_path = file_path
        self._file = None
        self._mode = mode
        self._encoding = encoding

    def __enter__(self):
        # read the data from the input file
        with open(self._file_path, "rb") as f:
            encrypted_aes_key = f.read(
                256
            )  # TODO: figure out the right size based on rsa_client
            iv = f.read(16)
            data = f.read()

        # decrypt the key using rsa_client
        aes_key = self._rsa_client.decrypt(
            EncryptionAlgorithm.rsa1_5, encrypted_aes_key
        ).plaintext

        decryptor = Cipher(algorithms.AES(aes_key), modes.CBC(iv)).decryptor()
        unpadder = padding.PKCS7(algorithms.AES.block_size).unpadder()
        plain_bytes = unpadder.update(decryptor.update(data)) + unpadder.finalize()

        if self._mode == "t":
            self._file = io.StringIO(plain_bytes.decode(self._encoding))
        elif self._mode == "b":
            self._file = io.BytesIO(plain_bytes)
        else:
            raise ValueError(
                f"mode={self._mode} is not supported, use either 't' or 'b' only"
            )

        return self._file

    def __exit__(self, exc_type, exc_value, traceback):
        self._file.close()


class write_encrypted:
    def __init__(self, file_path, mode="t", rsa_client=None, encoding="UTF-8"):
        assert mode in [
            "t",
            "b",
        ], f"mode={mode} is not supported, use either 't' or 'b' only"

        self._rsa_client = rsa_client or get_rsa_client()
        self._file_path = file_path
        self._file = None
        self._mode = mode
        self._encoding = encoding

    def __enter__(self):
        if self._mode == "t":
            self._file = io.StringIO()
        elif self._mode == "b":
            self._file = io.BytesIO()
        else:
            raise ValueError(
                f"mode={self._mode} is not supported, use either 't' or 'b' only"
            )

        return self._file

    def __exit__(self, exc_type, exc_value, traceback):
        # create a random AES key, encrypted using RSA public key
        aes_key = os.urandom(32)
        encrypted_aes_key = self._rsa_client.encrypt(
            EncryptionAlgorithm.rsa1_5, aes_key
        ).ciphertext

        # plus generate random IV
        iv = os.urandom(16)

        # create classes
        cipher = Cipher(
            algorithms.AES(aes_key), modes.CBC(iv), backend=default_backend()
        )
        encryptor = cipher.encryptor()
        padder = padding.PKCS7(algorithms.AES.block_size).padder()

        # read input data as a whole (for now)
        self._file.seek(0)
        data = self._file.read()
        if self._mode == "t":
            data = data.encode(self._encoding)

        with open(self._file_path, "wb") as f:
            f.write(encrypted_aes_key)
            f.write(iv)
            f.write(encryptor.update(padder.update(data) + padder.finalize()))

        self._file.close()
