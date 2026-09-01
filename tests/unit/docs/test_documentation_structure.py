import ast
import re
from collections.abc import Iterable
from pathlib import Path
from urllib.parse import unquote

import pytest
import yaml

_ROOT = Path(__file__).parents[3]
_DOCS_DIRECTORY = _ROOT / "docs"
_MARKDOWN_LINK = re.compile(r"(?<!!)\[[^\]]+\]\((?P<target>[^)]+)\)")
_GOOGLE_DOCSTRING_ENTRY = re.compile(r"^    (?P<name>[A-Za-z_]\w*):")


def _navigation_pages(value: object) -> Iterable[Path]:
    """Yield Markdown paths from a nested MkDocs navigation value.

    Args:
        value: String, list, or mapping read from `mkdocs.yml`.

    Yields:
        Documentation-relative Markdown paths in navigation order.
    """
    if isinstance(value, str):
        if value.endswith(".md"):
            yield Path(value)
        return
    if isinstance(value, list):
        for item in value:
            yield from _navigation_pages(item)
        return
    if isinstance(value, dict):
        for item in value.values():
            yield from _navigation_pages(item)


def _local_link_targets(markdown_path: Path) -> Iterable[Path]:
    """Yield filesystem targets for relative Markdown links in one page.

    Args:
        markdown_path: Source Markdown page.

    Yields:
        Resolved local paths after removing URL fragments.
    """
    contents = markdown_path.read_text(encoding="utf-8")
    for match in _MARKDOWN_LINK.finditer(contents):
        target = match.group("target").strip().strip("<>")
        if target.startswith(("#", "http://", "https://", "mailto:")):
            continue
        relative_target = unquote(target.split("#", maxsplit=1)[0])
        if relative_target:
            yield (markdown_path.parent / relative_target).resolve()


def _docstring_section_names(docstring: str, section: str) -> set[str]:
    """Collect field names from one Google-style docstring section.

    Args:
        docstring: Cleaned docstring text.
        section: Section heading without the trailing colon.

    Returns:
        Names declared directly inside the requested section.
    """
    lines = docstring.splitlines()
    try:
        start = lines.index(f"{section}:") + 1
    except ValueError:
        return set()
    names: set[str] = set()
    for line in lines[start:]:
        if line and not line.startswith(" "):
            break
        if match := _GOOGLE_DOCSTRING_ENTRY.match(line):
            names.add(match.group("name"))
    return names


def _heading_positions(contents: str, headings: tuple[str, ...]) -> tuple[int, ...]:
    """Return the position of each required Markdown heading.

    Args:
        contents: Complete Markdown source.
        headings: Exact headings expected in the source.

    Returns:
        Character positions in requested order.

    Raises:
        AssertionError: If a heading is absent or repeated.
    """
    positions: list[int] = []
    for heading in headings:
        matches = tuple(re.finditer(rf"^{re.escape(heading)}$", contents, flags=re.MULTILINE))
        assert len(matches) == 1, f"Expected one `{heading}` heading, found {len(matches)}."
        positions.append(matches[0].start())
    return tuple(positions)


def test_every_documentation_page_is_in_navigation() -> None:
    """Verify readers can reach every Markdown page through the site navigation."""
    config = yaml.safe_load((_ROOT / "mkdocs.yml").read_text(encoding="utf-8"))
    navigated = set(_navigation_pages(config["nav"]))
    available = {path.relative_to(_DOCS_DIRECTORY) for path in _DOCS_DIRECTORY.rglob("*.md")}

    assert navigated == available


@pytest.mark.parametrize("markdown_path", sorted(_DOCS_DIRECTORY.rglob("*.md")))
def test_relative_documentation_links_resolve(markdown_path: Path) -> None:
    """Verify local Markdown links point to existing files inside the docs tree.

    Args:
        markdown_path: Documentation page checked for local links.
    """
    docs_root = _DOCS_DIRECTORY.resolve()
    broken = [
        target
        for target in _local_link_targets(markdown_path)
        if not target.is_relative_to(docs_root) or not target.exists()
    ]

    assert not broken, f"Broken local links in {markdown_path.relative_to(_ROOT)}: {broken}"


def test_class_attributes_do_not_repeat_constructor_parameters() -> None:
    """Keep constructor inputs out of class-level runtime attribute tables."""
    duplicates: dict[str, list[str]] = {}
    for source_path in (_ROOT / "src" / "phijax").rglob("*.py"):
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            initializer = next(
                (
                    member
                    for member in node.body
                    if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)) and member.name == "__init__"
                ),
                None,
            )
            if initializer is None:
                continue
            class_docstring = ast.get_docstring(node) or ""
            initializer_docstring = ast.get_docstring(initializer) or ""
            repeated = _docstring_section_names(class_docstring, "Attributes") & _docstring_section_names(
                initializer_docstring, "Args"
            )
            if repeated:
                path = source_path.relative_to(_ROOT)
                duplicates[f"{path}:{node.lineno}:{node.name}"] = sorted(repeated)

    assert not duplicates, f"Class attributes repeat constructor parameters: {duplicates}"


def test_api_reference_uses_configured_source_links() -> None:
    """Render stable source links without embedding complete implementation blocks."""
    config = yaml.safe_load((_ROOT / "mkdocs.yml").read_text(encoding="utf-8"))
    mkdocstrings = next(plugin["mkdocstrings"] for plugin in config["plugins"] if "mkdocstrings" in plugin)
    python_options = mkdocstrings["handlers"]["python"]["options"]

    assert mkdocstrings["custom_templates"] == "docs_templates"
    assert python_options["show_source"] is False
    assert python_options["extra"]["source_repository"] == "https://github.com/HangJung97/PhiJAX"
    assert python_options["extra"]["source_ref"] == "main"
    for template_name, object_name in (("class", "class"), ("function", "function")):
        template = _ROOT / "docs_templates" / "python" / "material" / f"{template_name}.html.jinja"
        contents = template.read_text(encoding="utf-8")
        assert f"{object_name}.relative_filepath" in contents
        assert "config.extra.source_ref" in contents
        assert "[source]" in contents


@pytest.mark.parametrize(
    ("relative_path", "headings", "generated_reference"),
    [
        (
            "api/module.md",
            (
                "# PhiModule",
                "## Basic use",
                "## Responsibilities",
                "## Module blueprints",
                "## Logging metrics",
                "## Custom modules",
                "## Lifecycle and hooks",
                "## API reference",
            ),
            "phijax.core.PhiModule",
        ),
        (
            "api/trainer.md",
            (
                "# Trainer",
                "## Basic use",
                "## What the Trainer manages",
                "## Configuration",
                "## Common and advanced workflows",
                "## Metrics and status",
                "## Cleanup and interruption",
                "## API reference",
            ),
            "phijax.training.Trainer",
        ),
    ],
)
def test_core_api_pages_are_task_first(
    relative_path: str,
    headings: tuple[str, ...],
    generated_reference: str,
) -> None:
    """Keep common workflows before advanced details and one generated class reference.

    Args:
        relative_path: Documentation path relative to `docs`.
        headings: Required page headings in reader order.
        generated_reference: Fully qualified class rendered by mkdocstrings once.
    """
    contents = (_DOCS_DIRECTORY / relative_path).read_text(encoding="utf-8")

    positions = _heading_positions(contents, headings)
    assert positions == tuple(sorted(positions))
    directive = re.compile(rf"^::: {re.escape(generated_reference)}$", flags=re.MULTILINE)
    assert len(directive.findall(contents)) == 1
