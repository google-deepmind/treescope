"""Helper functions for tests."""

from treescope import rendering_parts, lowering


def ensure_text(text_or_part: str | rendering_parts.RenderableTreePart) -> str:
  """Ensure that a part is a string, lowering it if necessary."""
  if isinstance(text_or_part, str):
    return text_or_part
  return lowering.render_to_text_as_root(text_or_part)
