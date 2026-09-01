# Model layers

PhiJAX provides coordinate feature mappings and dense-layer parameterizations used by its built-in architectures.
They can also be composed into custom Flax NNX models.

## Periodic features

`PeriodicFeatures` replaces each selected scalar coordinate with cosine and sine features, increasing the width by one
per selected axis. Configured frequencies are angular frequencies, so the physical period is
`2*pi/abs(frequency)`.

::: phijax.models.PeriodicFeatures

## Random Fourier features

`RandomFourierFeatures` returns `2 * embed_dim` features using the paired mapping
`x -> [cos(xB), sin(xB)]`. The construction follows the random-feature foundation of
[Rahimi and Recht (2007)](https://proceedings.neurips.cc/paper/2007/hash/013a006f03dbc5392effeb8f18fda755-Abstract.html)
and its use for coordinate networks by
[Tancik et al. (2020)](https://proceedings.neurips.cc/paper/2020/hash/55053683268957697aa39fba6f231c68-Abstract.html).
PhiJAX stores the initially random projection `B` as a trainable parameter. Keeping `B` fixed would recover the
classical random-feature interpretation.

::: phijax.models.RandomFourierFeatures

## Factorized dense layers

`FactorizedDense` uses learned scale and direction parameters following
[Wang et al. (2022)](https://arxiv.org/abs/2210.01274). PhiJAX stores dense weights in
`[input_dim, output_dim]` layout, so its factorization is `W = V diag(g)`: each entry of `g` scales one output column.
With the common `[output_dim, input_dim]` layout, the same factorization is written `W = diag(g) V`.

::: phijax.models.FactorizedDense
