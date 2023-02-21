import tenseal as ts
import tenseal.sealapi as sealapi
import base64

# generate a new key
# context = ts.context(
#     ts.SCHEME_TYPE.CKKS,
#     poly_modulus_degree=4096,
#     plain_modulus=1032193
# )
context = ts.context(
    ts.SCHEME_TYPE.CKKS,
    8192,
    coeff_mod_bit_sizes=[60, 40, 40, 60]
)
# galois keys are required to do ciphertext rotations
context.generate_galois_keys()
context.generate_relin_keys()
# bits_scale: controls precision of the fractional part
bits_scale = 40
# set the scale
context.global_scale = pow(2, bits_scale)

bytes_context_private = context.serialize(
    save_public_key=True,
    save_secret_key=True,
    save_galois_keys=True,
    save_relin_keys=True,
)

# with open("context.key", "w", encoding='iso-8859-1') as f:
#     f.write(bytes_context.decode(encoding='iso-8859-1'))
with open("context_private.key", "w", encoding="utf-8") as f:
    f.write(base64.b64encode(bytes_context_private).decode('utf-8'))

bytes_context_public = context.serialize(
    save_public_key=True,
    save_secret_key=False,
    save_galois_keys=False,
    save_relin_keys=True,
)

# with open("context.key", "w", encoding='iso-8859-1') as f:
#     f.write(bytes_context.decode(encoding='iso-8859-1'))
with open("context_public.key", "w", encoding="utf-8") as f:
    f.write(base64.b64encode(bytes_context_public).decode('utf-8'))

v1 = [0, 1, 2, 3, 4]
v2 = [4, 3, 2, 1, 0]

plain_v1 = ts.plain_tensor(v1)
enc_v1 = ts.ckks_tensor(context, plain_v1)
plain_v2 = ts.plain_tensor(v2)
enc_v2 = ts.ckks_tensor(context, plain_v2)

result = enc_v1 + enc_v2
print(result.decrypt().tolist()) # ~ [4, 4, 4, 4, 4]

result = (enc_v1 + enc_v2) * 0.5
print(result.decrypt().tolist()) # ~ [4, 4, 4, 4, 4]

with open("v1.bin", "wb") as f:
    f.write(enc_v1.serialize())

with open("v2.bin", "wb") as f:
    f.write(enc_v2.serialize())
