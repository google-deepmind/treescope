# Copyright 2024 The Treescope Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Treescope: An interactive HTML pretty-printer and array visualizer.

You can configure treescope as the default IPython pretty-printer using ::

  treescope.basic_interactive_setup()

or, for more control: ::

  treescope.register_as_default()
  treescope.register_autovisualize_magic()
  treescope.active_autovisualizer.set_globally(
      treescope.ArrayAutovisualizer()
  )

You can also pretty-print individual values using `treescope.show` or
`treescope.display`.
"""

__version__ = '0.1.0.dev0'

# pylint: disable=g-importing-member,g-multiple-import,unused-import

from . import canonical_aliases
from . import context
from . import dataclass_util
from . import figures
from . import formatting_util
from . import handlers
from . import lowering
from . import ndarray_adapters
from . import renderers
from . import rendering_parts
from . import repr_lib
from . import type_registries

from ._internal.api.array_autovisualizer import (
    ArrayAutovisualizer,
)
from ._internal.api.arrayviz import (
    default_diverging_colormap,
    default_sequential_colormap,
    integer_digitbox,
    render_array,
    render_array_sharding,
)
from ._internal.api.autovisualize import (
    IPythonVisualization,
    VisualizationFromTreescopePart,
    ChildAutovisualizer,
    Autovisualizer,
    active_autovisualizer,
)
from ._internal.api.default_renderer import (
    active_renderer,
    active_expansion_strategy,
    render_to_html,
    render_to_text,
    using_expansion_strategy,
)
from ._internal.api.ipython_integration import (
    default_magic_autovisualizer,
    basic_interactive_setup,
    display,
    register_as_default,
    register_autovisualize_magic,
    register_context_manager_magic,
    show,
)


# Set up canonical aliases for the treescope API itself.
def _setup_canonical_aliases_for_api():
  import types  # pylint: disable=g-import-not-at-top

  for key, value in globals().items():
    if isinstance(value, types.FunctionType) or isinstance(value, type):
      canonical_aliases.add_alias(
          value, canonical_aliases.ModuleAttributePath(__name__, (key,))
      )


_setup_canonical_aliases_for_api()
del _setup_canonical_aliases_for_api
