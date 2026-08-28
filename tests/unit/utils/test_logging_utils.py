from phijax.utils.logging_utils import pad_keys


def test_pad_keys_applies_affixes_except_to_excluded_keys() -> None:
    """Verify metric namespaces can preserve explicitly excluded keys."""
    padded = pad_keys({"loss": 1.0, "step": 2}, prefix="train/", postfix="_raw", exclude="step")
    assert padded == {"train/loss_raw": 1.0, "step": 2}


def test_pad_keys_accepts_no_affixes_or_multiple_exclusions() -> None:
    """Verify optional affixes and sequence exclusions preserve mapping content."""
    mapping = {"a": 1, "b": 2}
    assert pad_keys(mapping, exclude=("a", "b")) == mapping
