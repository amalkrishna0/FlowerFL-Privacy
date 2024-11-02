import tenseal as ts
import utils

# Create a CKKS context
context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192, coeff_mod_bit_sizes=[60, 40, 40, 60])

# Generate the secret key
context.generate_galois_keys()

# Save the secret key to a file
utils.write_data("keys/secret.txt", context.serialize(save_secret_key=True))
