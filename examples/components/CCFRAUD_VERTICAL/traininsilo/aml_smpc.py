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
    def __init__(self) -> None:
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
            raise Exception(
                f"Encoding {encoding} not supported, use one of OpenSSH, DER, PEM"
            )

    def get_public_key(
        self,
        encoding: crypto_serialization.Encoding = crypto_serialization.Encoding.OpenSSH,
        format: crypto_serialization.PublicFormat = crypto_serialization.PublicFormat.OpenSSH,
    ) -> bytes:
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
        data = pickle.loads(data)
        assert "encrypted_key" in data and "encrypted_data" in data
        sym_key = self._key.decrypt(data["encrypted_key"], padding_type)
        data = Fernet(sym_key).decrypt(data["encrypted_data"])
        return data
