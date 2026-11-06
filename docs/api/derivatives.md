# Derivatives

Selective coordinate-derivative utilities avoid materializing full Jacobians and Hessians that a PDE does not use. Both
use forward-mode differentiation and compute only the derivatives you request.

Write the model as a function of scalar coordinates, in model-input order. The second argument, `argnums`, holds the
positions of the coordinates to differentiate:

```python
def solutions(t, x):
    return model_apply(model_state, jnp.stack((t, x)))[0]  # one field


value, (du_dt, du_dx) = value_and_jacobian(solutions, (0, 1))(t, x)
(d2u_dx2,) = hessian_diagonal(solutions, 1)(t, x)
```

The derivative tuple follows the order of `argnums`. Each derivative has the same structure as the function output, so
a function returning several fields gives every field's derivative in one call:

```python
def solutions(x, y):
    return model_apply(model_state, jnp.stack((x, y)))  # [u, v, p]


(u, v, p), (d_x, d_y) = value_and_jacobian(solutions, (0, 1))(x, y)
u_x, v_x, p_x = d_x
(u_xx, v_xx, _), (u_yy, v_yy, _) = hessian_diagonal(solutions, (0, 1))(x, y)
```

The function closes over `model_state`, so gradients with respect to the parameters still flow through it under
`jax.grad`. Evaluate one point at a time and batch the point residual with `jax.vmap`.

Both helpers raise `ValueError` when `argnums` is empty or repeats an index, or when a selected argument is not a
scalar. They raise `TypeError` when an entry of `argnums` is not an integer, including a Boolean.

::: phijax.value_and_jacobian

::: phijax.hessian_diagonal
