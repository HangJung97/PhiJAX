# Model layers

PhiJAX provides coordinate feature mappings and dense-layer parameterizations used by its built-in architectures.
They can also be composed into custom Flax NNX models.

## Periodic features

`PeriodicFeatures` replaces each periodic coordinate with `cos(2*pi*x/T)` and `sin(2*pi*x/T)` of the physical
coordinate `x`, where `T` is the period in physical units. The network then sees that coordinate only through
functions that repeat every `T`, so its outputs and derivatives repeat exactly. Input normalization does not affect
periodic axes; `input_norm` normalizes only the other axes.

Declare periods in the DataModule, whether they are fixed or measured from the data:

```python
class CardiacDataModule(ApplicationDataModule):
    def input_periods(self):
        return {3: self.cardiac_period}  # time is input axis 3
```

Then enable periodic features in the model:

```python
model = partial(build_mlp, input_dim=4, output_dim=4, input_norm=True, periodic_features=True)
```

`periodic_features_kwargs={"periods": {axis: T}}` overrides a declared period or adds a periodic axis, for example
`{"periods": {1: 2.0}}` for a spatial axis `x` in `[-1, 1]`. Enabling periodic features without any declared or
configured period raises `ValueError`.

The Trainer stores the periods the model applies with the checkpoint, following the
[input normalization statistics](../checkpointing.md#input-normalization-statistics) rules:

| Task                                               | Periods used                        |
| -------------------------------------------------- | ----------------------------------- |
| Fresh fit                                          | DataModule `input_periods()`        |
| Full resume, prediction from a checkpoint          | Checkpoint                          |
| Transfer learning with `weights_only=True`         | Target DataModule `input_periods()` |
| Any run with `periodic_features_kwargs["periods"]` | Configured value, for those axes    |

Transfer learning therefore aligns recordings by phase. For example, cardiac cycles with different heart rates map to
the same features. Changing which axes are periodic changes the model width, so such a checkpoint cannot be restored.

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
