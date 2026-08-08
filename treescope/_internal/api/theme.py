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

"""Configurable settings for the theme of renderings."""

from typing import Literal

from treescope import context


Theme = Literal["light", "dark"]

theme: context.ContextualValue[Theme] = context.ContextualValue(
    module=__name__,
    qualname="theme",
    initial_value="light",
)
