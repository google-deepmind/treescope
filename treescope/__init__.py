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
  treescope.enable_interactive_context()
  treescope.active_autovisualizer.set_interactive(treescope.ArrayAutovisualizer())

You can also pretty-print individual values using `treescope.show` or
`treescope.display`.
"""
# pylint: disable=g-importing-member,g-multiple-import,unused-import

from . import array_autovisualizer
from . import arrayviz
from . import autovisualize
from . import canonical_aliases
from . import context
from . import dataclass_util
from . import default_renderer
from . import figures
from . import formatting_util
from . import handlers
from . import lowering
from . import ndarray_adapters
from . import renderer
from . import rendering_parts
from . import repr_lib
from . import treescope_ipython
from . import type_registries

from .array_autovisualizer import (
    ArrayAutovisualizer,
)
from .arrayviz import (
    integer_digitbox,
    render_array,
    render_array_sharding,
    text_on_color,
)
from .context import (
    disable_interactive_context,
    enable_interactive_context,
)
from .default_renderer import (
    render_to_html,
    render_to_text,
    using_expansion_strategy,
)
from .treescope_ipython import (
    basic_interactive_setup,
    display,
    register_as_default,
    register_autovisualize_magic,
    register_context_manager_magic,
    show,
)
