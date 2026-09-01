# Precision

[`PrecisionPolicy`](#phijax.training.PrecisionPolicy) separates parameter, compute, output, and derivative dtypes.
`PrecisionMode`, `PrecisionAlias`, and `PrecisionName` describe accepted precision strings. `MatmulPrecision` describes
the independent JAX dot and convolution precision setting.

| Mode         | Parameter dtype | Compute dtype | Output dtype | Dynamic loss scaling |
| ------------ | --------------- | ------------- | ------------ | -------------------- |
| `64-true`    | float64         | float64       | float64      | No                   |
| `32-true`    | float32         | float32       | float32      | No                   |
| `16-true`    | float16         | float16       | float16      | No                   |
| `bf16-true`  | bfloat16        | bfloat16      | bfloat16     | No                   |
| `16-mixed`   | float32         | float16       | float32      | Yes                  |
| `bf16-mixed` | float32         | bfloat16      | float32      | No                   |

`initial_loss_scale` applies only to `16-mixed`. Non-finite gradients skip the optimizer update and reduce the scale.
BFloat16 normally does not need dynamic scaling because it retains a float32-like exponent range.

`matmul_precision` accepts `default`, `high`, or `highest`. `None` preserves the current JAX policy. An explicit
Trainer value temporarily overrides `JAX_DEFAULT_MATMUL_PRECISION` during fit and prediction. It does not change
parameter or activation dtypes. See JAX's
[matrix-multiplication precision guide](https://docs.jax.dev/en/latest/201/precision.html).

::: phijax.training.PrecisionPolicy

::: phijax.training.configure_precision
