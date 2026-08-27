# Derivatives

Selective coordinate-derivative utilities avoid materializing full Jacobians and Hessians that a PDE does not use.

The second argument, `argnums`, contains positional indices of scalar arguments at the single-point function boundary:

```python
def scalar_prediction(model_state, t, x):
    return model_apply(model_state, jnp.stack((t, x)))[0]


value_and_first = value_and_jacobian(scalar_prediction, (1, 2))
value, (du_dt, du_dx) = value_and_first(model_state, t, x)

second_spatial = hessian_diagonal(scalar_prediction, 2)
(d2u_dx2,) = second_spatial(model_state, t, x)
```

The derivative tuple preserves requested index order. Batch the transformed single-point function with `jax.vmap`.

::: phijax.value_and_jacobian

::: phijax.hessian_diagonal
