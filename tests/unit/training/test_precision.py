import jax
import jax.numpy as jnp
import pytest

from phijax.models import apply_mlp, initialize_mlp
from phijax.training import PrecisionPolicy


@pytest.mark.parametrize(
    ("name", "parameter_dtype", "compute_dtype", "output_dtype", "dynamic"),
    [
        ("32-true", jnp.float32, jnp.float32, jnp.float32, False),
        ("16-true", jnp.float16, jnp.float16, jnp.float16, False),
        ("bf16-true", jnp.bfloat16, jnp.bfloat16, jnp.bfloat16, False),
        ("16-mixed", jnp.float32, jnp.float16, jnp.float32, True),
        ("bf16-mixed", jnp.float32, jnp.bfloat16, jnp.float32, False),
    ],
)
def test_precision_policy_resolves_lightning_modes(
    name: str,
    parameter_dtype: jnp.dtype,
    compute_dtype: jnp.dtype,
    output_dtype: jnp.dtype,
    dynamic: bool,
) -> None:
    """Verify canonical true and mixed modes resolve the expected numerical policy.

    Args:
        name: Precision mode under test.
        parameter_dtype: Expected parameter dtype.
        compute_dtype: Expected arithmetic dtype.
        output_dtype: Expected public-output dtype.
        dynamic: Expected dynamic loss-scaling behavior.
    """
    policy = PrecisionPolicy.from_name(name)
    assert policy.parameter_dtype == parameter_dtype
    assert policy.compute_dtype == compute_dtype
    assert policy.output_dtype == output_dtype
    assert policy.dynamic_loss_scaling is dynamic


def test_mixed_precision_mlp_keeps_float32_parameters_and_outputs() -> None:
    """Verify BF16 mixed precision only lowers supported internal arithmetic."""
    graphdef, state = initialize_mlp(
        jax.random.key(1),
        2,
        1,
        hidden=(4,),
        precision="bf16-mixed",
    )
    output = apply_mlp(graphdef, state, jnp.ones((2,), dtype=jnp.float32))
    array_leaves = [leaf for leaf in jax.tree.leaves(state) if hasattr(leaf, "dtype")]
    assert array_leaves
    assert all(leaf.dtype == jnp.float32 for leaf in array_leaves)
    assert output.dtype == jnp.float32


def test_true_precision_mlp_casts_parameters_and_outputs() -> None:
    """Verify BF16 true precision uses BF16 storage and public outputs."""
    graphdef, state = initialize_mlp(
        jax.random.key(2),
        2,
        1,
        hidden=(4,),
        precision="bf16-true",
    )
    output = apply_mlp(graphdef, state, jnp.ones((2,), dtype=jnp.bfloat16))
    array_leaves = [leaf for leaf in jax.tree.leaves(state) if hasattr(leaf, "dtype")]
    assert array_leaves
    assert all(leaf.dtype == jnp.bfloat16 for leaf in array_leaves)
    assert output.dtype == jnp.bfloat16


def test_precision_policy_casts_only_floating_batch_leaves() -> None:
    """Verify batch casting preserves integer labels and nested structure."""
    batch = {"coordinates": jnp.ones((2, 3), jnp.float32), "indices": jnp.arange(2, dtype=jnp.int32)}
    converted = PrecisionPolicy.from_name("bf16-true").cast_batch(batch)
    assert converted["coordinates"].dtype == jnp.bfloat16
    assert converted["indices"].dtype == jnp.int32


def test_precision_policy_rejects_unknown_modes() -> None:
    """Verify an unsupported precision string fails before training begins."""
    with pytest.raises(ValueError, match="Unknown precision mode"):
        PrecisionPolicy.from_name("fp8-magic")
