..
  This file is not actually referenced in the docs, but it is the entry point
  that generates automatic summaries for Treescope's modules. `index.rst` points
  directly at the autosummary files generated while processing this one.

:orphan:

.. autosummary::
  :toctree: api
  :template: pzmodule_full.rst
  :recursive:

  treescope.canonical_aliases
  treescope.context
  treescope.dataclass_util
  treescope.figures
  treescope.formatting_util
  treescope.handlers
  treescope.lowering
  treescope.ndarray_adapters
  treescope.renderers
  treescope.rendering_parts
  treescope.repr_lib
  treescope.type_registries
