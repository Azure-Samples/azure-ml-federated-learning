##########################################################################################
#                                         WARNING                                        #
##########################################################################################
# Should this file change please update all copies of aml_smpc.py file in the repository #
##########################################################################################

import sys
import pickle
import logging

from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes as crypto_hashes
from cryptography.hazmat.primitives import serialization as crypto_serialization
from cryptography.hazmat.backends import default_backend as crypto_default_backend

# Set logging to sys.out
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
log_format = logging.Formatter("[%(asctime)s] [%(levelname)s] - %(message)s")
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
handler.setFormatter(log_format)
logger.addHandler(handler)


class AMLSMPC:
    """Class for encryption and decryption using RSA and Fernet."""

    def __init__(self) -> None:
        """Initialize the class by generating a private key for current node and a dictionary for remote public keys."""
        self._key = rsa.generate_private_key(
            backend=crypto_default_backend, public_exponent=65537, key_size=2048
        )
        self._remote_keys: dict[int, rsa.RSAPublicKey] = {}

    def add_remote_public_key(
        self,
        destination,
        key,
        encoding: crypto_serialization.Encoding = crypto_serialization.Encoding.OpenSSH,
        overwrite=False,
    ):
        """Add a public key for a remote destination.

        Args:
            destination (int): Destination node id.
            key (bytes): Public key.
            encoding (crypto_serialization.Encoding, optional): Encoding of the public key. Defaults to crypto_serialization.Encoding.OpenSSH.
            overwrite (bool, optional): If True, overwrite the public key if already present. Defaults to False.
        """
        logger.debug(f"Adding public key for destination: {destination}, key: {key}")

        if destination in self._remote_keys:
            if overwrite:
                logger.warning(
                    f"Destination already present, overwriting public key for destination {destination}."
                )
            else:
                raise Exception(
                    f"Destination {destination} already present, if you want forcefully overwrite key set 'overwrite=True'."
                )

        if encoding == crypto_serialization.Encoding.OpenSSH:
            self._remote_keys[destination] = crypto_serialization.load_ssh_public_key(
                key
            )
        elif encoding == crypto_serialization.Encoding.DER:
            self._remote_keys[destination] = crypto_serialization.load_der_public_key(
                key
            )
        elif encoding == crypto_serialization.Encoding.PEM:
            self._remote_keys[destination] = crypto_serialization.load_pem_public_key(
                key
            )
        else:
            raise ValueError(
                f"Encoding {encoding} not supported, use one of OpenSSH, DER, PEM"
            )

    def get_public_key(
        self,
        encoding: crypto_serialization.Encoding = crypto_serialization.Encoding.OpenSSH,
        format: crypto_serialization.PublicFormat = crypto_serialization.PublicFormat.OpenSSH,
    ) -> bytes:
        """Get the public key of the current node.

        Args:
            encoding (crypto_serialization.Encoding, optional): Encoding of the public key. Defaults to crypto_serialization.Encoding.OpenSSH.
            format (crypto_serialization.PublicFormat, optional): Format of the public key. Defaults to crypto_serialization.PublicFormat.OpenSSH.
        """
        return self._key.public_key().public_bytes(encoding, format)

    def encrypt(
        self,
        data: bytes,
        destination: int,
        padding_type: padding.AsymmetricPadding = padding.OAEP(
            mgf=padding.MGF1(algorithm=crypto_hashes.SHA256()),
            algorithm=crypto_hashes.SHA256(),
            label=None,
        ),
    ) -> bytes:
        """Encrypt data using Fernet and the public key of the destination node.

        Args:
            data (bytes): Data to be encrypted.
            destination (int): Destination node id.
            padding_type (padding.AsymmetricPadding, optional): Encryption padding type. Defaults to padding.OAEP(mgf=padding.MGF1(algorithm=crypto_hashes.SHA256()),algorithm=crypto_hashes.SHA256(),label=None).
        """
        assert destination in self._remote_keys

        sym_key = Fernet.generate_key()
        encrypted_data = Fernet(sym_key).encrypt(data)

        encrypted_key = self._remote_keys[destination].encrypt(sym_key, padding_type)
        data = pickle.dumps(
            {"encrypted_key": encrypted_key, "encrypted_data": encrypted_data}
        )

        return data

    def decrypt(
        self,
        data: bytes,
        padding_type: padding.AsymmetricPadding = padding.OAEP(
            mgf=padding.MGF1(algorithm=crypto_hashes.SHA256()),
            algorithm=crypto_hashes.SHA256(),
            label=None,
        ),
    ) -> bytes:
        """Decrypt data using Fernet and the private key of the current node.

        Args:
            data (bytes): Data to be decrypted.
            padding_type (padding.AsymmetricPadding, optional): Encryption padding type. Defaults to padding.OAEP(mgf=padding.MGF1(algorithm=crypto_hashes.SHA256()),algorithm=crypto_hashes.SHA256(),label=None).
        """
        data = pickle.loads(data)
        assert "encrypted_key" in data and "encrypted_data" in data
        sym_key = self._key.decrypt(data["encrypted_key"], padding_type)
        data = Fernet(sym_key).decrypt(data["encrypted_data"])
        return data
