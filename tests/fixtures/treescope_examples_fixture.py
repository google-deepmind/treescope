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

"""Defines some classes for use in rendering tests.

This is in a separate module so we can test treescope's handling of module
names during qualified name lookup.
"""

from __future__ import annotations

import dataclasses
import enum
import typing
from typing import Any

import jax
import pydantic
import torch
import treescope


class MyTestEnum(enum.Enum):
  FOO = 1
  BAR = 2


@dataclasses.dataclass
class DataclassWithOneChild:
  foo: Any


@dataclasses.dataclass
class DataclassWithTwoChildren:
  foo: Any
  bar: Any


@dataclasses.dataclass(frozen=True)
class EmptyDataclass:
  pass


class SomeNamedtupleClass(typing.NamedTuple):
  foo: Any
  bar: Any


class SomeOuterClass:

  @dataclasses.dataclass
  class NestedClass:
    foo: Any


def make_class_with_weird_qualname():
  @dataclasses.dataclass
  class ClassDefinedInAFunction:  # pylint: disable=redefined-outer-name
    foo: Any

  return ClassDefinedInAFunction


ClassDefinedInAFunction = make_class_with_weird_qualname()


class RedefinedClass:
  """A class that will later be redefined."""

  pass


OriginalRedefinedClass = RedefinedClass


class RedefinedClass:  # pylint: disable=function-redefined
  """The redefined class; no longer the same as `_OriginalRedefinedClass`."""

  pass


class _PrivateClass:

  def some_function(self):
    pass


def _private_function():
  pass


class SomeFunctionLikeWrapper:
  func: Any

  def __init__(self, func):
    self.func = func

  def __call__(self, *args, **kwargs):
    return self.func(*args, **kwargs)

  @property
  def __wrapped__(self):
    return self.func


@SomeFunctionLikeWrapper
def wrapped_function():
  pass


immutable_constant = (1, 2, 3)
mutable_constant = [1, 2, 3]


class SomethingCallable:

  def __call__(self, value: int) -> int:
    return value + 1


# pytype is confused about the dataclass transform here
some_callable_block = SomethingCallable()


@dataclasses.dataclass(frozen=True)
class FrozenDataclassKey:
  name: str


@jax.tree_util.register_pytree_with_keys_class
class UnknownPytreeNode:
  """A Pytree node treescope doesn't know."""

  def __init__(self, x, y):
    self.x = x
    self.y = y

  def __repr__(self):
    return f"<custom repr for UnknownPytreeNode: x={self.x}, y={self.y}>"

  def tree_flatten_with_keys(self):
    return (
        ((FrozenDataclassKey("x"), self.x), ("string_key", self.y)),
        "example_pytree_aux_data",
    )

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    del aux_data
    return cls(*children)


class UnknownObjectWithBuiltinRepr:
  pass


class UnknownObjectWithOneLineRepr:

  def __repr__(self):
    return "<custom repr for UnknownObjectWithOneLineRepr>"


class UnknownObjectWithMultiLineRepr:

  def __repr__(self):
    return "<custom repr\n  for\n  UnknownObjectWithMultiLineRepr\n>"


class UnknownObjectWithBadMultiLineRepr:

  def __repr__(self):
    return "Non-idiomatic\nmultiline\nobject"


class ObjectWithCustomHandler:

  def __treescope_repr__(self, path, subtree_renderer):
    del subtree_renderer
    return treescope.rendering_parts.text(
        f"<ObjectWithCustomHandler custom rendering! Path: {repr(path)}>"
    )


class ObjectWithCustomHandlerThatThrows:

  def __treescope_repr__(self, path, subtree_renderer):
    del path, subtree_renderer
    raise RuntimeError("Simulated treescope_repr failure!")

  def __repr__(self):
    return "<Fallback repr for ObjectWithCustomHandlerThatThrows>"


class ObjectWithReprThatThrows:

  def __repr__(self):
    raise RuntimeError("Simulated repr failure!")


class ObjectWithCustomHandlerThatThrowsDeferred:

  def __treescope_repr__(self, path, subtree_renderer):
    del path, subtree_renderer

    def _internal_main_thunk(layout_decision):
      del layout_decision
      raise RuntimeError("Simulated deferred treescope_repr failure!")

    return treescope.lowering.maybe_defer_rendering(
        main_thunk=_internal_main_thunk,
        placeholder_thunk=lambda: treescope.rendering_parts.text(
            "<deferred placeholder>"
        ),
    )


class SomePyTorchModule(torch.nn.Module):
  """A basic PyTorch module to test rendering."""

  def __init__(self):
    super().__init__()
    # Attributes
    self.attr_one = 123
    self.attr_two = "abc"
    # Child modules
    self.linear = torch.nn.Linear(10, 10)
    self.mod_list = torch.nn.ModuleList(
        [torch.nn.LayerNorm(10), torch.nn.SiLU()]
    )
    # Parameters
    self.foo = torch.nn.Parameter(torch.ones(5))
    # Buffers
    self.register_buffer("bar", torch.zeros(5))

  @classmethod
  def build(cls):
    torch.random.manual_seed(1234)
    return cls()


class SomePydanticModel(pydantic.BaseModel):
  a: int
  b: str
  c: float
