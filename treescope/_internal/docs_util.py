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

"""Utilities for documentation generation."""

from typing import Any

_SKIPPED_FOR_AUTODOC = {}


def skip_automatic_documentation(obj: Any):
  """Marks an object as skipped for automatic documentation generation."""
  _SKIPPED_FOR_AUTODOC[id(obj)] = obj
  return obj