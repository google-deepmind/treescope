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

"""Tests for rendering named enum members and unnamed flag values."""

import enum
import sys

from absl.testing import absltest
from absl.testing import parameterized
import treescope
from treescope import lowering
from treescope import rendering_parts


class ExampleFlags(enum.Flag):
  READ = 1
  WRITE = 2
  EXECUTE = 4
  READ_ALIAS = READ


class NamedFlags(enum.Flag):
  NONE = 0
  READ = 1
  WRITE = 2
  BOTH = READ | WRITE
  BOTH_ALIAS = BOTH


class ExampleEnum(enum.Enum):
  FIRST = 1
  ALIAS = FIRST


_CASES = [
    (ExampleFlags(0), "ExampleFlags(0)"),
    (ExampleFlags.READ | ExampleFlags.WRITE, "ExampleFlags(3)"),
    (~ExampleFlags.READ, "ExampleFlags(6)"),
    (ExampleFlags.READ, "ExampleFlags.READ  # value: 1"),
    (ExampleFlags.READ_ALIAS, "ExampleFlags.READ  # value: 1"),
    (NamedFlags.NONE, "NamedFlags.NONE  # value: 0"),
    (NamedFlags.BOTH, "NamedFlags.BOTH  # value: 3"),
    (NamedFlags.BOTH_ALIAS, "NamedFlags.BOTH  # value: 3"),
    (ExampleEnum.FIRST, "ExampleEnum.FIRST  # value: 1"),
    (ExampleEnum.ALIAS, "ExampleEnum.FIRST  # value: 1"),
]


class EnumFlagsTest(parameterized.TestCase):

  @parameterized.product(case=_CASES, roundtrip_mode=[False, True])
  def test_text(self, case, roundtrip_mode):
    target, expected = case
    actual = treescope.render_to_text(
        target, roundtrip_mode=roundtrip_mode, ignore_exceptions=False
    )
    if roundtrip_mode:
      self.assertEqual(actual, f"{__name__}.{expected}")
      restored = eval(  # pylint: disable=eval-used
          actual, {__name__.split(".")[0]: sys.modules[__name__.split(".")[0]]}
      )
      self.assertIs(restored, target)
    else:
      self.assertEqual(actual, expected)

  @parameterized.parameters(*_CASES)
  def test_html_and_keypath(self, target, expected):
    actual = treescope.render_to_html(
        target, ignore_exceptions=False, compressed=False
    )
    self.assertIn(type(target).__name__, actual)
    rendering = treescope.active_renderer.get().to_foldable_representation(
        target, root_keypath="settings.permissions"
    )
    html = lowering.render_to_html_as_root(
        rendering_parts.build_full_line_with_annotations(rendering),
        compressed=False,
    )
    self.assertIn("settings.permissions", html)
    self.assertIn(expected.split(".")[0].split("(")[0], html)


if __name__ == "__main__":
  absltest.main()
