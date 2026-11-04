def missing_extra_error(extra: str, error: ModuleNotFoundError) -> ModuleNotFoundError:
    """Explain which PhiJAX extra provides a missing optional package.

    Args:
        extra: Name of the PhiJAX extra, such as `hydra` or `io`.
        error: Error raised while importing the optional package.

    Returns:
        Error naming the missing package and the install command. Raise it `from error` to keep the original cause.
    """
    return ModuleNotFoundError(
        f"`{error.name}` is not installed. This feature requires the optional `{extra}` extra: install "
        f'`phijax[{extra}]`, for example with `uv add "phijax[{extra}]"`.',
        name=error.name,
    )
