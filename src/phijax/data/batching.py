import jax

DevicePool = dict[str, jax.Array]
BatchSize = int | str


def _chunk_layout(row_count: int, batch_size: int) -> tuple[int, int]:
    """Resolve the fixed prediction-batch count and padded row count.

    Empty inputs retain one completely masked batch so compiled prediction and reconstruction code receives stable
    non-empty batch structure.

    Args:
        row_count: Non-negative number of valid rows.
        batch_size: Positive fixed number of rows per batch.

    Returns:
        Pair containing the number of batches and total padded rows.

    Raises:
        ValueError: If `row_count` is negative or `batch_size` is not a positive integer.
    """
    if isinstance(row_count, bool) or not isinstance(row_count, int) or row_count < 0:
        raise ValueError("Prediction row count must be a non-negative integer.")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("Prediction batch size must be a positive integer.")
    batch_count = max(1, (row_count + batch_size - 1) // batch_size)
    return batch_count, batch_count * batch_size


def _validate_batch_size(policy: BatchSize, *, allow_all: bool) -> None:
    """Validate one generated or finite-source batch-size policy.

    Args:
        policy: Positive integer or the finite-source literal `all`.
        allow_all: Whether the corresponding source can return all rows.

    Raises:
        ValueError: If the policy is invalid or `all` is unsupported.
    """
    if policy == "all":
        if not allow_all:
            raise ValueError("Generated samplers do not support the `all` batch-size policy.")
        return
    if isinstance(policy, bool) or not isinstance(policy, int) or policy < 1:
        raise ValueError("A sampler batch size must be a positive integer or `all`.")


__all__ = ["BatchSize", "DevicePool"]
