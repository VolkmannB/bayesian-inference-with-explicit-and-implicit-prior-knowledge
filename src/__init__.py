import jax

print("Setting JAX precision to 64-bit...")
jax.config.update("jax_enable_x64", True)
