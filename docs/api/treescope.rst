treescope
=========

.. module:: treescope

Displaying values
-----------------

To display individual values using Treescope without configuring it as the
default pretty-printer, you can use these functions:

.. autosummary::
  :toctree:
  :template: pzbase.rst

  show
  display
  render_array


Using Treescope as the default renderer
---------------------------------------

To enable Treescope as the default renderer in IPython, you can use:

.. autosummary::
  :toctree:
  :template: pzbase.rst

  basic_interactive_setup

This will configure Treescope as the default renderer of all IPython cell
outputs. It will also turn on automatic array visualization, enable interactive
customization of configuration options, and install the ``%%autovisualize``
and ``%%with`` IPython magics.

For more control, you can configure these individually:

.. autosummary::
  :toctree:
  :template: pzdata.rst

  register_as_default
  register_autovisualize_magic
  register_context_manager_magic


Automatic visualization
-----------------------

Treescope supports automatic visualization of particular leaves of a tree using
an "autovisualizer". The most common autovisualizer is the array autovizualizer,
but you can also define your own autovisualizers using any IPython rich display
object. (See the
:doc:`custom visualization guide </notebooks/building_custom_visualizations>`
for details.)

To enable an autovisualizer for all Treescope outputs, you can use ::

  treescope.active_autovisualizer.set_globally(
      treescope.ArrayAutovisualizer()  # or your own autovisualizer
  )

To enable it for a single display call, you can pass the `autovisualize`
argument to `treescope.display` or `treescope.show`, e.g. ::

  treescope.display(..., autovisualize=True)

Alternatively you can use the ``%%autovisualize`` magic to turn on automatic
visualization in a single cell, e.g. ::

  %%autovisualize treescope.ArrayAutovisualizer()
  treescope.display(...)

or just ::

  %%autovisualize
  # ^ with no arguments, uses the default array autovisualizer
  treescope.display(...)

Types for building autovisualizers:

.. autosummary::
  :toctree:
  :template: pzclass.rst

  ArrayAutovisualizer
  Autovisualizer
  IPythonVisualization
  ChildAutovisualizer
  VisualizationFromTreescopePart


Configuring rendering options
-----------------------------

Most of Treescope's rendering options are of type `context.ContextualValue`.
These can be set temporarily using `context.ContextualValue.set_scoped`, or
configured globally using `context.ContextualValue.set_globally`.

.. autosummary::
  :toctree:
  :template: pzdata.rst

  default_diverging_colormap
  default_sequential_colormap
  active_autovisualizer
  active_renderer
  default_magic_autovisualizer
  active_expansion_strategy


Rendering to strings
--------------------

Instead of displaying objects directly, you can render them to strings to
save or display later.

.. autosummary::
  :toctree:
  :template: pzbase.rst

  render_to_html
  render_to_text


Other utilities
---------------

.. autosummary::
  :toctree:
  :template: pzbase.rst

  integer_digitbox
  render_array_sharding
  using_expansion_strategy

