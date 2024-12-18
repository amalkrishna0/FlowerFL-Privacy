import tenseal as ts
import pickle

context = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=8192,
    coeff_mod_bit_sizes=[60, 40, 40, 60]
)
context.global_scale = 2**40
context.generate_galois_keys()

public_context = context.serialize(save_public_key=True)
secret_context = context.serialize(save_secret_key=True)

with open("public_context.pkl", "wb") as f:
    pickle.dump(public_context, f)

with open("secret_context.pkl", "wb") as f:
    pickle.dump(secret_context, f)
