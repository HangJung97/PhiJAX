document.addEventListener("keydown", (event) => {
  if ((!event.ctrlKey && !event.metaKey) || event.altKey || event.key.toLowerCase() !== "k") {
    return;
  }

  const searchToggle = document.querySelector('[data-md-toggle="search"]');
  const searchButton = document.querySelector('.md-header__button[for="__search"]');
  const searchInput = document.querySelector('[data-md-component="search-query"]');
  if (
    !(searchToggle instanceof HTMLInputElement) ||
    !(searchButton instanceof HTMLElement) ||
    !(searchInput instanceof HTMLInputElement)
  ) {
    return;
  }

  const target = event.target;
  const isEditable =
    target instanceof HTMLElement &&
    (target.matches("input, textarea, select") || target.isContentEditable);
  if (isEditable && target !== searchInput) {
    return;
  }

  event.preventDefault();
  if (!searchToggle.checked) {
    searchButton.click();
  }
  requestAnimationFrame(() => searchInput.focus());
});
