# Models

PhiJAX supplies Flax NNX models but equations and objectives consume only a pure explicit-state callable. Split models
before regular `jax.jit` use and pass their graph definition through a closure or partial callable.

## MLP transformation order

The built-in MLP applies optional operations in this order:

```text
input normalization
-> periodic coordinate features
-> random Fourier features
-> hidden dense layers, activations, and dropout
-> output layer and optional output activation
```

`one_mlp_per_output=True` creates an independent scalar network per output. `weight_norm` and
`weight_factorization` are mutually exclusive. Stochastic dropout requires `deterministic=False` and an explicit
`dropout_key`.

```yaml
_target_: phijax.models.initialize_mlp
_partial_: true
input_dim: 2
output_dim: 1
hidden: [128, 128, 128, 128]
activation: tanh
input_norm: true
```

::: phijax.models.MLP

::: phijax.models.initialize_mlp

::: phijax.models.build_mlp

::: phijax.models.apply_mlp

## Layers

`PeriodicFeatures` replaces each selected scalar coordinate by cosine and sine features, increasing the width by one
per selected axis. Configured frequencies are angular frequencies, so the physical period is `2*pi/abs(frequency)`.

`RandomFourierFeatures` returns `2 * embed_dim` features. `FactorizedDense` provides learned magnitude and direction
parameters for random weight factorization.

::: phijax.models.PeriodicFeatures

::: phijax.models.RandomFourierFeatures

::: phijax.models.FactorizedDense

## Summary

::: phijax.models.tabulate_nnx_model
