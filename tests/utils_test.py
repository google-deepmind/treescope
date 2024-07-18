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

"""Tests for some self-contained utility functions."""

import dataclasses
import re
from absl.testing import absltest
from treescope import context
from treescope import dataclass_util


class ContextualValueTest(absltest.TestCase):
  """Tests for core/context.py."""

  def test_set_scoped(self):
    ctx_value = context.ContextualValue(
        initial_value=1, module=__name__, qualname=None
    )
    self.assertEqual(ctx_value.get(), 1)

    with ctx_value.set_scoped(2):
      self.assertEqual(ctx_value.get(), 2)
      with ctx_value.set_scoped(3):
        self.assertEqual(ctx_value.get(), 3)
      self.assertEqual(ctx_value.get(), 2)

    self.assertEqual(ctx_value.get(), 1)

  def test_set_globally(self):
    ctx_value = context.ContextualValue(
        initial_value=1, module=__name__, qualname=None
    )
    self.assertEqual(ctx_value.get(), 1)

    ctx_value.set_globally(2)
    self.assertEqual(ctx_value.get(), 2)

    with ctx_value.set_scoped(3):
      self.assertEqual(ctx_value.get(), 3)

    self.assertEqual(ctx_value.get(), 2)


class DataclassUtilTest(absltest.TestCase):
  """Tests for core/dataclass_util.py."""

  def test_dataclass_from_attributes(self):
    @dataclasses.dataclass(frozen=True)
    class MyWeirdInitClass:
      foo: int
      bar: int

      def __init__(self, weird_arg):
        raise NotImplementedError("shouldn't be called")

    value = dataclass_util.dataclass_from_attributes(
        MyWeirdInitClass, foo=3, bar=4
    )
    self.assertEqual(value.foo, 3)
    self.assertEqual(value.bar, 4)

    with self.assertRaisesRegex(
        ValueError,
        re.escape("Incorrect fields provided to `dataclass_from_attributes`"),
    ):
      _ = dataclass_util.dataclass_from_attributes(MyWeirdInitClass, qux=5)

  def test_init_takes_fields_normal_init(self):
    @dataclasses.dataclass(frozen=True)
    class MyDataclass:
      foo: int
      bar: int = 7

    self.assertTrue(dataclass_util.init_takes_fields(MyDataclass))

  def test_init_takes_fields_custom_compatible_init(self):
    @dataclasses.dataclass
    class MyDataclass:
      foo: int
      bar: int

      def __init__(self, foo: int = 3, bar: int = 4):
        self.foo = foo
        self.bar = bar

    self.assertTrue(dataclass_util.init_takes_fields(MyDataclass))

  def test_init_takes_fields_for_init_with_custom_args(self):
    @dataclasses.dataclass
    class MyDataclass:
      foo: int
      bar: int

      def __init__(self, weird_arg):
        self.foo = weird_arg
        self.bar = weird_arg

    self.assertFalse(dataclass_util.init_takes_fields(MyDataclass))


if __name__ == "__main__":
  absltest.main()
