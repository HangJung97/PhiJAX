## Summary

<!-- Explain the motivation, approach, and scope of this change. -->

## User-visible changes

<!-- Describe changes to behavior, public imports, configuration, numerical results, or generated artifacts. -->

## Breaking changes and migration

<!-- List required migration steps. Write "None" when the change is backward compatible. -->

## Validation

<!-- List the exact commands run and summarize their results. Identify checks left to CI. -->

## Before submitting

- [ ] The title is self-explanatory and the description summarizes one coherent change.
- [ ] Tests were added or updated for changed behavior.
- [ ] The relevant CPU tests pass, or checks not run locally are identified above.
- [ ] Pre-commit passes with `uv run --no-sync pre-commit run --all-files`.
- [ ] Public API and user documentation were updated, or no documentation change is needed.
- [ ] Breaking, numerical, configuration, checkpoint, and artifact changes are documented above.
- [ ] GPU, distributed, external-service, or large-data checks not run locally are identified above.

<!-- Optional: Closes #123 -->

## Did you have fun?

Make sure you had fun coding 🙃
