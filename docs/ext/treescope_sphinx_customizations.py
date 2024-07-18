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
"""Sphinx customizations for Treescope."""

from typing import Any, Mapping

from sphinx import application
from treescope._internal import docs_util

# pylint: disable=protected-access


def skip_member(app, what, name, obj, options, lines):  # pylint: disable=unused-argument
  """Configures autodoc to skip private implementation details."""
  if id(obj) in docs_util._SKIPPED_FOR_AUTODOC:
    return True


def setup(app: application.Sphinx) -> Mapping[str, Any]:
  app.connect("autodoc-skip-member", skip_member)
  return dict(version="0.1", parallel_read_safe=True)
