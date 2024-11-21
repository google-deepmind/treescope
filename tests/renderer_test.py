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

"""Tests for the treescope renderer and handlers."""

import ast
import collections
import dataclasses
import functools
import re
import textwrap
import types
from typing import Any, Callable
import warnings

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
import torch
import treescope
from treescope import handlers
from treescope import layout_algorithms
from treescope import lowering
from treescope import rendering_parts
from treescope.external import jax_support
from tests.fixtures import treescope_examples_fixture as fixture_lib


@dataclasses.dataclass
class CustomReprHTMLObject:
  repr_html: str

  def _repr_html_(self):
    return self.repr_html


class TreescopeRendererTest(parameterized.TestCase):

  def test_renderer_interface(self):
    renderer = treescope.active_renderer.get()

    rendering = renderer.to_text({"key": "value"})
    self.assertEqual(rendering, "{'key': 'value'}")

    rendering = renderer.to_html({"key": "value"})
    self.assertIsInstance(rendering, str)

    rendering = renderer.to_foldable_representation({"key": "value"})
    self.assertIsInstance(
        rendering, rendering_parts.RenderableAndLineAnnotations
    )

  def test_high_level_interface(self):
    rendering = treescope.render_to_text({"key": "value"})
    self.assertEqual(rendering, "{'key': 'value'}")

    rendering = treescope.render_to_html({"key": "value"})
    self.assertIsInstance(rendering, str)

  def test_error_recovery(self):
    def handler_that_crashes(node, path, subtree_renderer):
      del path, subtree_renderer
      if node == "trigger handler error":
        raise RuntimeError("handler error!")
      return NotImplemented

    def hook_that_crashes(node, path, node_renderer):
      del path, node_renderer
      if node == "trigger hook error":
        raise RuntimeError("hook error!")
      return NotImplemented

    renderer = treescope.active_renderer.get().extended_with(
        handlers=[handler_that_crashes], wrapper_hooks=[hook_that_crashes]
    )

    rendering = renderer.to_foldable_representation([1, 2, 3, "foo", 4])
    layout_algorithms.expand_to_depth(rendering.renderable, 1)
    self.assertEqual(
        lowering.render_to_text_as_root(
            rendering_parts.build_full_line_with_annotations(rendering)
        ),
        "[\n  1,\n  2,\n  3,\n  'foo',\n  4,\n]",
    )

    with self.assertRaisesWithLiteralMatch(RuntimeError, "handler error!"):
      _ = renderer.to_foldable_representation(
          [1, 2, 3, "trigger handler error", 4]
      )

    with self.assertRaisesWithLiteralMatch(RuntimeError, "hook error!"):
      _ = renderer.to_foldable_representation(
          [1, 2, 3, "trigger hook error", 4]
      )

    with warnings.catch_warnings(record=True) as recorded:
      rendering = renderer.to_foldable_representation(
          [1, 2, 3, "trigger handler error", "trigger hook error", 4],
          ignore_exceptions=True,
      )
    layout_algorithms.expand_to_depth(rendering.renderable, 1)
    self.assertEqual(
        lowering.render_to_text_as_root(
            rendering_parts.build_full_line_with_annotations(rendering)
        ),
        "[\n  1,\n  2,\n  3,\n  'trigger handler error',\n  'trigger hook"
        " error',\n  4,\n]",
    )
    self.assertLen(recorded, 2)
    self.assertEqual(type(recorded[0]), warnings.WarningMessage)
    self.assertContainsInOrder(
        [
            "Ignoring error while formatting value",
            "with",
            "<locals>.handler_that_crashes",
        ],
        str(recorded[0].message),
    )
    self.assertEqual(type(recorded[1]), warnings.WarningMessage)
    self.assertContainsInOrder(
        [
            "Ignoring error inside wrapper hook",
            "<locals>.hook_that_crashes",
        ],
        str(recorded[1].message),
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="literals",
          target=[False, True, None, Ellipsis, NotImplemented],
          expected_collapsed="[False, True, None, Ellipsis, NotImplemented]",
      ),
      dict(
          testcase_name="numbers",
          target=[1234, 1234.0],
          expected_collapsed="[1234, 1234.0]",
      ),
      dict(
          testcase_name="strings",
          target=["some string", b"some bytes"],
          expected_collapsed="['some string', b'some bytes']",
      ),
      dict(
          testcase_name="multiline_string",
          target="some string\n    with \n newlines in it",
          expected_collapsed="'some string\\n    with \\n newlines in it'",
          expected_expanded=(
              "  'some string\\n'\n  '    with \\n'\n  ' newlines in it'"
          ),
      ),
      dict(
          testcase_name="enums",
          target=[
              fixture_lib.MyTestEnum.FOO,
              fixture_lib.MyTestEnum.BAR,
          ],
          expected_collapsed="[MyTestEnum.FOO, MyTestEnum.BAR]",
          expected_expanded=textwrap.dedent("""\
              [
                MyTestEnum.FOO,  # value: 1
                MyTestEnum.BAR,  # value: 2
              ]"""),
          expected_roundtrip=textwrap.dedent("""\
              [
                tests.fixtures.treescope_examples_fixture.MyTestEnum.FOO,  # value: 1
                tests.fixtures.treescope_examples_fixture.MyTestEnum.BAR,  # value: 2
              ]"""),
      ),
      dict(
          testcase_name="empty_dict",
          target={},
          expected_collapsed="{}",
      ),
      dict(
          testcase_name="dict_with_contents",
          target={"a": "b", (1, 2, 3, 4): (5, 6, 7, 8), (): (9, 10)},
          expected_collapsed=(
              "{'a': 'b', (1, 2, 3, 4): (5, 6, 7, 8), (): (9, 10)}"
          ),
          expected_expanded=textwrap.dedent("""\
              {
                'a': 'b',
                (1, 2, 3, 4):
                  (5, 6, 7, 8),
                (): (9, 10),
              }"""),
      ),
      dict(
          testcase_name="dict_subclass",
          target=collections.OrderedDict({"a": "b", (): (9, 10)}),
          expected_collapsed="OrderedDict({'a': 'b', (): (9, 10)})",
          expected_expanded=textwrap.dedent("""\
              OrderedDict({
                'a': 'b',
                (): (9, 10),
              })"""),
          expected_roundtrip=textwrap.dedent("""\
              collections.OrderedDict({
                'a': 'b',
                (): (9, 10),
              })"""),
      ),
      dict(
          testcase_name="tuple_empty",
          target=(),
          expected_collapsed="()",
      ),
      dict(
          testcase_name="tuple_singleton",
          target=(1,),
          expected_collapsed="(1,)",
          expected_expanded=textwrap.dedent("""\
              (
                1,
              )"""),
      ),
      dict(
          testcase_name="tuple_multiple",
          target=(1, 2, 3),
          expected_collapsed="(1, 2, 3)",
          expected_expanded=textwrap.dedent("""\
              (
                1,
                2,
                3,
              )"""),
      ),
      dict(
          testcase_name="list_empty",
          target=[],
          expected_collapsed="[]",
      ),
      dict(
          testcase_name="list_singleton",
          target=[1],
          expected_collapsed="[1]",
          expected_expanded=textwrap.dedent("""\
              [
                1,
              ]"""),
      ),
      dict(
          testcase_name="list_multiple",
          target=[1, 2, 3],
          expected_collapsed="[1, 2, 3]",
          expected_expanded=textwrap.dedent("""\
              [
                1,
                2,
                3,
              ]"""),
      ),
      dict(
          testcase_name="sets_empty",
          target=[set(), frozenset()],
          expected_collapsed="[set(), frozenset({})]",
          expected_expanded=textwrap.dedent("""\
              [
                set(),
                frozenset({}),
              ]"""),
      ),
      dict(
          testcase_name="set_with_items",
          target={1, 2, 3},
          expected_collapsed="{1, 2, 3}",
          expected_expanded=textwrap.dedent("""\
              {
                1,
                2,
                3,
              }"""),
      ),
      dict(
          testcase_name="frozenset_with_items",
          target=frozenset({1, 2, 3}),
          expected_collapsed="frozenset({1, 2, 3})",
          expected_expanded=textwrap.dedent("""\
              frozenset({
                1,
                2,
                3,
              })"""),
      ),
      dict(
          testcase_name="simplenamespace",
          target=types.SimpleNamespace(foo="bar", baz="qux"),
          expected_collapsed="SimpleNamespace(foo='bar', baz='qux')",
          expected_expanded=textwrap.dedent("""\
              SimpleNamespace(
                foo='bar',
                baz='qux',
              )"""),
      ),
      dict(
          testcase_name="nametuple",
          target=fixture_lib.SomeNamedtupleClass(foo="baz", bar="qux"),
          expected_collapsed="SomeNamedtupleClass(foo='baz', bar='qux')",
          expected_expanded=textwrap.dedent("""\
              SomeNamedtupleClass(
                foo='baz',
                bar='qux',
              )"""),
          expected_roundtrip=textwrap.dedent("""\
              tests.fixtures.treescope_examples_fixture.SomeNamedtupleClass(
                foo='baz',
                bar='qux',
              )"""),
      ),
      dict(
          testcase_name="dataclass",
          target=fixture_lib.DataclassWithTwoChildren(foo="baz", bar="qux"),
          expected_collapsed="DataclassWithTwoChildren(foo='baz', bar='qux')",
          expected_expanded=textwrap.dedent("""\
              DataclassWithTwoChildren(
                foo='baz',
                bar='qux',
              )"""),
          expected_roundtrip=textwrap.dedent("""\
              tests.fixtures.treescope_examples_fixture.DataclassWithTwoChildren(
                foo='baz',
                bar='qux',
              )"""),
      ),
      dict(
          testcase_name="dataclass_empty",
          target=fixture_lib.EmptyDataclass(),
          expected_collapsed="EmptyDataclass()",
          expected_expanded="EmptyDataclass()",
          expected_roundtrip=(
              "tests.fixtures.treescope_examples_fixture.EmptyDataclass()"
          ),
      ),
      dict(
          testcase_name="ndarray_small",
          target=np.array([1, 2, 4, 8, 16]),
          expected_collapsed="np.array([ 1,  2,  4,  8, 16])",
          expected_expanded="np.array([ 1,  2,  4,  8, 16])",
      ),
      dict(
          testcase_name="ndarray_large",
          target=np.arange(3 * 7).reshape((3, 7)),
          expected_collapsed=(
              "<np.ndarray int64(3, 7) [≥0, ≤20] zero:1 nonzero:20>"
          ),
          expected_expanded=textwrap.dedent("""\
              # np.ndarray int64(3, 7) [≥0, ≤20] zero:1 nonzero:20
                array([[ 0,  1,  2,  3,  4,  5,  6],
                       [ 7,  8,  9, 10, 11, 12, 13],
                       [14, 15, 16, 17, 18, 19, 20]])"""),
      ),
      dict(
          testcase_name="jax_array_large",
          target_builder=lambda: jnp.arange(3 * 7).reshape((3, 7)),
          expected_collapsed=(
              "<jax.Array int32(3, 7) [≥0, ≤20] zero:1 nonzero:20>"
          ),
          expected_expanded=textwrap.dedent("""\
              # jax.Array int32(3, 7) [≥0, ≤20] zero:1 nonzero:20
                Array([[ 0,  1,  2,  3,  4,  5,  6],
                       [ 7,  8,  9, 10, 11, 12, 13],
                       [14, 15, 16, 17, 18, 19, 20]], dtype=int32)"""),
      ),
      dict(
          testcase_name="pytorch_tensor_small",
          target_builder=lambda: torch.tensor(np.array([1, 2, 4, 8, 16])),
          expected_collapsed="torch.tensor([ 1,  2,  4,  8, 16])",
          expected_expanded="torch.tensor([ 1,  2,  4,  8, 16])",
      ),
      dict(
          testcase_name="pytorch_tensor_large",
          target_builder=lambda: torch.tensor(np.arange(3 * 7).reshape((3, 7))),
          expected_collapsed=(
              "<torch.Tensor int64(3, 7) [≥0, ≤20] zero:1 nonzero:20>"
          ),
          expected_expanded=textwrap.dedent("""\
              # torch.Tensor int64(3, 7) [≥0, ≤20] zero:1 nonzero:20
                tensor([[ 0,  1,  2,  3,  4,  5,  6],
                        [ 7,  8,  9, 10, 11, 12, 13],
                        [14, 15, 16, 17, 18, 19, 20]])"""),
      ),
      dict(
          testcase_name="well_known_function",
          target=treescope.render_to_text,
          expected_collapsed="render_to_text",
          expected_roundtrip_collapsed="treescope.render_to_text",
      ),
      dict(
          testcase_name="well_known_type",
          target=treescope.IPythonVisualization,
          expected_collapsed="IPythonVisualization",
          expected_roundtrip_collapsed="treescope.IPythonVisualization",
      ),
      dict(
          testcase_name="ast_nodes",
          target=ast.parse("print(1, 2)").body[0],
          expand_depth=3,
          expected_expanded=textwrap.dedent("""\
              Expr(
                value=Call(
                  func=Name(
                    id='print',
                    ctx=Load(),
                  ),
                  args=[
                    Constant(value=1, kind=None),
                    Constant(value=2, kind=None),
                  ],
                  keywords=[],
                ),
              )"""),
      ),
      dict(
          testcase_name="custom_handler",
          target=[fixture_lib.ObjectWithCustomHandler()],
          expected_collapsed=(
              "[<ObjectWithCustomHandler custom rendering! Path: '[0]'>]"
          ),
      ),
      dict(
          testcase_name="custom_handler_that_throws",
          target=[fixture_lib.ObjectWithCustomHandlerThatThrows()],
          ignore_exceptions=True,
          expected_collapsed=(
              "[<Fallback repr for ObjectWithCustomHandlerThatThrows>]"
          ),
      ),
      dict(
          testcase_name="dtype_standard",
          target=np.dtype(np.float32),
          expected_collapsed="dtype('float32')",
          expected_roundtrip_collapsed="np.dtype('float32')",
      ),
      dict(
          testcase_name="dtype_extended",
          target=np.dtype(jnp.bfloat16),
          expected_collapsed="dtype('bfloat16')",
          expected_roundtrip_collapsed="np.dtype('bfloat16')",
      ),
      dict(
          testcase_name="jax_precision",
          target=[jax.lax.Precision.HIGHEST],
          expected_collapsed="[Precision.HIGHEST]",
          expected_expanded=textwrap.dedent("""\
              [
                Precision.HIGHEST,  # value: 2
              ]"""),
          expected_roundtrip_collapsed="[jax.lax.Precision.HIGHEST]",
      ),
      dict(
          testcase_name="pytorch_module",
          target_builder=fixture_lib.SomePyTorchModule.build,
          expected_collapsed=(
              "SomePyTorchModule(attr_one=123, attr_two='abc', training=True,"
              " foo=<torch.nn.Parameter float32(5,) ≈1.0 ±0.0 [≥1.0, ≤1.0]"
              " nonzero:5>, bar=torch.tensor([0., 0., 0., 0., 0.]),"
              " linear=Linear(in_features=10, out_features=10, training=True,"
              " weight=<torch.nn.Parameter float32(10, 10) ≈-0.008 ±0.028"
              " [≥-0.32, ≤0.3] nonzero:100>, bias=<torch.nn.Parameter"
              " float32(10,) ≈0.008 ±0.04 [≥-0.31, ≤0.31] nonzero:10>, ),"
              " mod_list=ModuleList(training=True, (0):"
              " LayerNorm(normalized_shape=(10,), eps=1e-05,"
              " elementwise_affine=True, training=True,"
              " weight=<torch.nn.Parameter float32(10,) ≈1.0 ±0.0 [≥1.0, ≤1.0]"
              " nonzero:10>, bias=<torch.nn.Parameter float32(10,) ≈0.0 ±0.0"
              " [≥0.0, ≤0.0] zero:10>, ), (1): SiLU(inplace=False,"
              " training=True, ), ), )"
          ),
          expected_expanded=textwrap.dedent("""\
              SomePyTorchModule(
                attr_one=123, attr_two='abc', training=True,
                # Parameters:
                foo=<torch.nn.Parameter float32(5,) ≈1.0 ±0.0 [≥1.0, ≤1.0] nonzero:5>,
                # Buffers:
                bar=torch.tensor([0., 0., 0., 0., 0.]),
                # Child modules:
                linear=Linear(in_features=10, out_features=10, training=True, weight=<torch.nn.Parameter float32(10, 10) ≈-0.008 ±0.028 [≥-0.32, ≤0.3] nonzero:100>, bias=<torch.nn.Parameter float32(10,) ≈0.008 ±0.04 [≥-0.31, ≤0.31] nonzero:10>, ),
                mod_list=ModuleList(training=True, (0): LayerNorm(normalized_shape=(10,), eps=1e-05, elementwise_affine=True, training=True, weight=<torch.nn.Parameter float32(10,) ≈1.0 ±0.0 [≥1.0, ≤1.0] nonzero:10>, bias=<torch.nn.Parameter float32(10,) ≈0.0 ±0.0 [≥0.0, ≤0.0] zero:10>, ), (1): SiLU(inplace=False, training=True, ), ),
              )"""),
          expected_roundtrip=textwrap.dedent("""\
              <tests.fixtures.treescope_examples_fixture.SomePyTorchModule(
                attr_one=123, attr_two='abc', training=True,
                # Parameters:
                foo=<torch.nn.Parameter float32(5,) ≈1.0 ±0.0 [≥1.0, ≤1.0] nonzero:5>,
                # Buffers:
                bar=torch.tensor([0., 0., 0., 0., 0.]),
                # Child modules:
                linear=<torch.nn.modules.linear.Linear(in_features=10, out_features=10, training=True, weight=<torch.nn.Parameter float32(10, 10) ≈-0.008 ±0.028 [≥-0.32, ≤0.3] nonzero:100>, bias=<torch.nn.Parameter float32(10,) ≈0.008 ±0.04 [≥-0.31, ≤0.31] nonzero:10>, )>,
                mod_list=<torch.nn.modules.container.ModuleList(training=True, (0): <torch.nn.modules.normalization.LayerNorm(normalized_shape=(10,), eps=1e-05, elementwise_affine=True, training=True, weight=<torch.nn.Parameter float32(10,) ≈1.0 ±0.0 [≥1.0, ≤1.0] nonzero:10>, bias=<torch.nn.Parameter float32(10,) ≈0.0 ±0.0 [≥0.0, ≤0.0] zero:10>, )>, (1): <torch.nn.modules.activation.SiLU(inplace=False, training=True, )>, )>,
              )>"""),
      ),
      dict(
          testcase_name="pytorch_module_expanded",
          target_builder=fixture_lib.SomePyTorchModule.build,
          expand_depth=2,
          expected_expanded=textwrap.dedent("""\
              SomePyTorchModule(
                # Attributes:
                attr_one=123,
                attr_two='abc',
                training=True,
                # Parameters:
                foo=# torch.nn.Parameter float32(5,) ≈1.0 ±0.0 [≥1.0, ≤1.0] nonzero:5
                  Parameter containing:
                  tensor([1., 1., 1., 1., 1.], requires_grad=True)
                ,
                # Buffers:
                bar=torch.tensor([0., 0., 0., 0., 0.]),
                # Child modules:
                linear=Linear(
                  in_features=10, out_features=10, training=True,
                  # Parameters:
                  weight=<torch.nn.Parameter float32(10, 10) ≈-0.008 ±0.028 [≥-0.32, ≤0.3] nonzero:100>,
                  bias=<torch.nn.Parameter float32(10,) ≈0.008 ±0.04 [≥-0.31, ≤0.31] nonzero:10>,
                ),
                mod_list=ModuleList(
                  training=True,
                  # Child modules:
                  (0): LayerNorm(normalized_shape=(10,), eps=1e-05, elementwise_affine=True, training=True, weight=<torch.nn.Parameter float32(10,) ≈1.0 ±0.0 [≥1.0, ≤1.0] nonzero:10>, bias=<torch.nn.Parameter float32(10,) ≈0.0 ±0.0 [≥0.0, ≤0.0] zero:10>, ),
                  (1): SiLU(inplace=False, training=True, ),
                ),
              )"""),
      ),
  )
  def test_object_rendering(
      self,
      *,
      target: Any = None,
      target_builder: Callable[[], Any] | None = None,
      expected_collapsed: str | None = None,
      expected_expanded: str | None = None,
      expected_roundtrip: str | None = None,
      expected_roundtrip_collapsed: str | None = None,
      expand_depth: int = 1,
      ignore_exceptions: bool = False,
  ):
    if target_builder is not None:
      assert target is None
      target = target_builder()

    renderer = treescope.active_renderer.get()
    # Render it to IR.
    with warnings.catch_warnings():
      if ignore_exceptions:
        # Also ignore warnings due to ignoring exceptions to avoid cluttering
        # logs.
        warnings.simplefilter("ignore")
      rendering = rendering_parts.build_full_line_with_annotations(
          renderer.to_foldable_representation(
              target, ignore_exceptions=ignore_exceptions
          )
      )

    # Collapse all foldables.
    layout_algorithms.expand_to_depth(rendering, 0)

    if expected_collapsed is not None:
      with self.subTest("collapsed"):
        self.assertEqual(
            lowering.render_to_text_as_root(rendering),
            expected_collapsed,
        )

    if expected_roundtrip_collapsed is not None:
      with self.subTest("roundtrip_collapsed"):
        self.assertEqual(
            lowering.render_to_text_as_root(rendering, roundtrip=True),
            expected_roundtrip_collapsed,
        )

    layout_algorithms.expand_to_depth(rendering, expand_depth)

    if expected_expanded is not None:
      with self.subTest("expanded"):
        self.assertEqual(
            lowering.render_to_text_as_root(rendering),
            expected_expanded,
        )

    if expected_roundtrip is not None:
      with self.subTest("roundtrip"):
        self.assertEqual(
            lowering.render_to_text_as_root(rendering, roundtrip=True),
            expected_roundtrip,
        )

    # Render to HTML; make sure it doesn't raise any errors.
    with self.subTest("html_no_errors"):
      _ = lowering.render_to_html_as_root(rendering)

  def test_closure_rendering(self):
    def outer_fn(x):
      def inner_fn(y):
        return x + y

      return inner_fn

    closure = outer_fn(100)

    renderer = treescope.active_renderer.get()
    # Enable closure rendering (currently disabled by default)
    renderer = renderer.extended_with(
        handlers=[
            functools.partial(
                handlers.handle_code_objects_with_reflection,
                show_closure_vars=True,
            )
        ]
    )
    # Render it to IR.
    rendering = rendering_parts.build_full_line_with_annotations(
        renderer.to_foldable_representation(closure)
    )

    layout_algorithms.expand_to_depth(rendering, 1)

    self.assertContainsInOrder(
        [
            "<function",
            "test_closure_rendering.<locals>.outer_fn.<locals>.inner_fn at 0x",
            ">",
            "# Closure variables:",
            "{'x': 100}",
            "# Defined at line ",
            " of ",
            "tests/renderer_test.py",
        ],
        lowering.render_to_text_as_root(rendering),
    )

  def test_render_jax_array_within_jitted_function(self):
    old = jax_support.SUMMARIZE_USING_NUMPY_THRESHOLD
    jax_support.SUMMARIZE_USING_NUMPY_THRESHOLD = 0
    renderer = treescope.active_renderer.get()
    x = jnp.arange(10)

    # Verify that we don't fail when called inside a jitted function.
    @jax.jit
    def go(s):
      nonlocal renderer, x
      adapter = jax_support.JAXArrayAdapter()
      self.assertNotEmpty(adapter.get_array_summary(x, False))
      return jax.numpy.sum(s)

    go(jnp.arange(3))
    jax_support.SUMMARIZE_USING_NUMPY_THRESHOLD = old

  def test_fallback_repr_pytree_node(self):
    target = [fixture_lib.UnknownPytreeNode(1234, 5678)]
    renderer = treescope.active_renderer.get()
    rendering = rendering_parts.build_full_line_with_annotations(
        renderer.to_foldable_representation(target)
    )
    layout_algorithms.expand_to_depth(rendering, 0)
    self.assertEqual(
        lowering.render_to_text_as_root(rendering),
        "[<custom repr for UnknownPytreeNode: x=1234, y=5678>]",
    )

    layout_algorithms.expand_to_depth(rendering, 2)
    rendered_text = lowering.render_to_text_as_root(rendering)
    self.assertEqual(
        "\n".join(
            line.rstrip() for line in rendered_text.splitlines(keepends=True)
        ),
        textwrap.dedent(f"""\
            [
              <custom repr for UnknownPytreeNode: x=1234, y=5678>
                #╭┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄╮
                # PyTree children:
                  FrozenDataclassKey(name='x'): 1234,
                  'string_key': 5678,
                #╰┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄╯
              ,  # {object.__repr__(target[0])}
            ]"""),
    )

  def test_fallback_repr_one_line(self):
    target = [fixture_lib.UnknownObjectWithOneLineRepr()]
    renderer = treescope.active_renderer.get()
    rendering = rendering_parts.build_full_line_with_annotations(
        renderer.to_foldable_representation(target)
    )
    layout_algorithms.expand_to_depth(rendering, 0)
    self.assertEqual(
        lowering.render_to_text_as_root(rendering),
        "[<custom repr for UnknownObjectWithOneLineRepr>]",
    )
    layout_algorithms.expand_to_depth(rendering, 2)
    self.assertEqual(
        lowering.render_to_text_as_root(rendering),
        textwrap.dedent(f"""\
            [
              <custom repr for UnknownObjectWithOneLineRepr>,  # {object.__repr__(target[0])}
            ]"""),
    )

  def test_fallback_repr_after_error(self):
    target = [fixture_lib.ObjectWithCustomHandlerThatThrows()]
    renderer = treescope.active_renderer.get()

    with self.assertRaisesWithLiteralMatch(
        RuntimeError, "Simulated treescope_repr failure!"
    ):
      renderer.to_foldable_representation(target)

    with self.assertWarnsRegex(
        UserWarning,
        "(.|\n)*".join([
            re.escape("Ignoring error while formatting value of type"),
            re.escape("ObjectWithCustomHandlerThatThrows"),
            re.escape(
                'raise RuntimeError("Simulated treescope_repr failure!")'
            ),
        ]),
    ):
      rendering = rendering_parts.build_full_line_with_annotations(
          renderer.to_foldable_representation(target, ignore_exceptions=True)
      )

    layout_algorithms.expand_to_depth(rendering, 0)
    self.assertEqual(
        lowering.render_to_text_as_root(rendering),
        "[<Fallback repr for ObjectWithCustomHandlerThatThrows>]",
    )
    layout_algorithms.expand_to_depth(rendering, 2)
    self.assertEqual(
        lowering.render_to_text_as_root(rendering),
        textwrap.dedent(f"""\
            [
              <Fallback repr for ObjectWithCustomHandlerThatThrows>,  # {object.__repr__(target[0])}
            ]"""),
    )

  def test_ignore_exceptions_in_deferred(self):
    target = [fixture_lib.ObjectWithCustomHandlerThatThrowsDeferred()]
    renderer = treescope.active_renderer.get()

    with self.assertRaisesWithLiteralMatch(
        RuntimeError, "Simulated deferred treescope_repr failure!"
    ):
      renderer.to_foldable_representation(target)

    with lowering.collecting_deferred_renderings() as deferreds:
      foldable_ir = rendering_parts.build_full_line_with_annotations(
          renderer.to_foldable_representation(target)
      )

    # It's difficult to test the IPython wrapper so we instead test the internal
    # helper function that produces the streaming HTML output.
    html_parts = lowering._render_to_html_as_root_streaming(
        root_node=foldable_ir,
        roundtrip=False,
        deferreds=deferreds,
        ignore_exceptions=True,
    )
    self.assertContainsInOrder(
        [
            "[",
            "&lt;RuntimeError during deferred rendering",
            "Traceback",
            "in _internal_main_thunk",
            "raise RuntimeError",
            "RuntimeError: Simulated deferred treescope_repr failure!",
            "&gt;",
            "]",
        ],
        "".join(html_parts),
    )

  def test_fallback_repr_multiline_idiomatic(self):
    target = [fixture_lib.UnknownObjectWithMultiLineRepr()]
    renderer = treescope.active_renderer.get()
    rendering = rendering_parts.build_full_line_with_annotations(
        renderer.to_foldable_representation(target)
    )
    layout_algorithms.expand_to_depth(rendering, 0)
    self.assertEqual(
        lowering.render_to_text_as_root(rendering),
        "[<custom repr↩  for↩  UnknownObjectWithMultiLineRepr↩>]",
    )
    layout_algorithms.expand_to_depth(rendering, 2)
    self.assertEqual(
        lowering.render_to_text_as_root(rendering),
        textwrap.dedent(f"""\
            [
              <custom repr
                for
                UnknownObjectWithMultiLineRepr
              >,  # {object.__repr__(target[0])}
            ]"""),
    )

  def test_fallback_repr_multiline_unidiomatic(self):
    target = [fixture_lib.UnknownObjectWithBadMultiLineRepr()]
    renderer = treescope.active_renderer.get()
    rendering = rendering_parts.build_full_line_with_annotations(
        renderer.to_foldable_representation(target)
    )
    layout_algorithms.expand_to_depth(rendering, 0)
    self.assertEqual(
        lowering.render_to_text_as_root(rendering),
        f"[{object.__repr__(target[0])}]",
    )
    layout_algorithms.expand_to_depth(rendering, 2)
    self.assertEqual(
        lowering.render_to_text_as_root(rendering),
        textwrap.dedent(f"""\
            [
              # {object.__repr__(target[0])}
                Non-idiomatic
                multiline
                object
              ,
            ]"""),
    )

  def test_fallback_repr_basic(self):
    target = [fixture_lib.UnknownObjectWithBuiltinRepr()]
    renderer = treescope.active_renderer.get()
    rendering = rendering_parts.build_full_line_with_annotations(
        renderer.to_foldable_representation(target)
    )
    layout_algorithms.expand_to_depth(rendering, 0)
    self.assertEqual(
        lowering.render_to_text_as_root(rendering),
        f"[{repr(target[0])}]",
    )
    layout_algorithms.expand_to_depth(rendering, 2)
    self.assertEqual(
        lowering.render_to_text_as_root(rendering),
        textwrap.dedent(f"""\
            [
              {repr(target[0])},
            ]"""),
    )

  def test_failsafe_for_throw_in_repr(self):
    target = [fixture_lib.ObjectWithReprThatThrows()]
    renderer = treescope.active_renderer.get()

    with self.assertRaisesWithLiteralMatch(
        RuntimeError, "Simulated repr failure!"
    ):
      renderer.to_foldable_representation(target)

    with self.assertWarnsRegex(
        UserWarning,
        "(.|\n)*".join([
            re.escape("Ignoring error while formatting value of type"),
            re.escape("ObjectWithReprThatThrows"),
            re.escape('raise RuntimeError("Simulated repr failure!")'),
        ]),
    ):
      rendering = rendering_parts.build_full_line_with_annotations(
          renderer.to_foldable_representation(target, ignore_exceptions=True)
      )

    layout_algorithms.expand_to_depth(rendering, 0)
    self.assertEqual(
        lowering.render_to_text_as_root(rendering),
        f"[{object.__repr__(target[0])}]",
    )
    layout_algorithms.expand_to_depth(rendering, 2)
    self.assertEqual(
        lowering.render_to_text_as_root(rendering),
        textwrap.dedent(f"""\
            [
              {object.__repr__(target[0])},  # Error occurred while formatting this object.
            ]"""),
    )

  def test_shared_values(self):
    shared = ["bar"]
    target = [shared, shared, {"foo": shared}]
    renderer = treescope.active_renderer.get()
    rendering = rendering_parts.build_full_line_with_annotations(
        renderer.to_foldable_representation(target)
    )
    layout_algorithms.expand_to_depth(rendering, 3)
    rendered_text = lowering.render_to_text_as_root(rendering)
    # Rendering may contain trailing whitespace; remove it before checking the
    # value, since it's not important.
    self.assertEqual(
        "\n".join(
            line.rstrip() for line in rendered_text.splitlines(keepends=True)
        ),
        textwrap.dedent(f"""\
            [
              [
                'bar',
              ], # Repeated python obj at 0x{id(shared):x}
              [
                'bar',
              ], # Repeated python obj at 0x{id(shared):x}
              {{
                'foo': [
                  'bar',
                ], # Repeated python obj at 0x{id(shared):x}
              }},
            ]"""),
    )

  def test_autovisualizer(self):
    target = [1, 2, "foo", 3, 4, 5, [6, 7]]

    def autovisualizer_for_test(node, path):
      if isinstance(node, str):
        return treescope.VisualizationFromTreescopePart(
            rendering_parts.RenderableAndLineAnnotations(
                rendering_parts.text("(visualization for foo goes here)"),
                rendering_parts.text(" # annotation for vis for foo"),
            ),
        )
      elif path == "[4]":
        return treescope.IPythonVisualization(
            CustomReprHTMLObject("(html rendering)"),
            replace=True,
        )
      elif path == "[5]":
        return treescope.IPythonVisualization(
            CustomReprHTMLObject("(html rendering)"),
            replace=False,
        )
      elif path == "[6]":
        return treescope.ChildAutovisualizer(inner_autovisualizer)

    def inner_autovisualizer(node, path):
      del path
      if node == 6:
        return treescope.VisualizationFromTreescopePart(
            rendering_parts.RenderableAndLineAnnotations(
                rendering_parts.text("(child visualization of 6 goes here)"),
                rendering_parts.text(" # annotation for vis for 6"),
            ),
        )

    with treescope.active_autovisualizer.set_scoped(autovisualizer_for_test):
      renderer = treescope.active_renderer.get()
      rendering = rendering_parts.build_full_line_with_annotations(
          renderer.to_foldable_representation(target)
      )
      layout_algorithms.expand_to_depth(rendering, 3)
      rendered_text = lowering.render_to_text_as_root(rendering)
      rendered_text_as_roundtrip = lowering.render_to_text_as_root(
          rendering, roundtrip=True
      )

    self.assertEqual(
        rendered_text,
        textwrap.dedent("""\
            [
              1,
              2,
              (visualization for foo goes here), # annotation for vis for foo
              3,
              <Visualization of int:
                <rich HTML visualization>
              >,
              5
              #╭┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄╮
              <rich HTML visualization>
              #╰┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄╯
              ,
              [
                (child visualization of 6 goes here), # annotation for vis for 6
                7,
              ],
            ]"""),
    )

    self.assertEqual(
        rendered_text_as_roundtrip,
        textwrap.dedent("""\
            [
              1,
              2,
              'foo',  # Visualization hidden in roundtrip mode
              3,
              4,  # Visualization hidden in roundtrip mode
              5
              #╭┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄╮
              # <rich HTML visualization>
              #╰┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄╯
              ,
              [
                6,  # Visualization hidden in roundtrip mode
                7,
              ],
            ]"""),
    )

  def test_balanced_layout(self):
    renderer = treescope.active_renderer.get()
    some_nested_object = fixture_lib.DataclassWithOneChild([
        ["foo"] * 4,
        ["12345678901234567890"] * 5,
        {"a": 1, "b": 2, "c": [{"bar": "baz"} for _ in range(5)]},
        [list(range(10)) for _ in range(6)],
    ])

    def render_and_expand(**kwargs):
      rendering = renderer.to_foldable_representation(
          some_nested_object
      ).renderable
      layout_algorithms.expand_for_balanced_layout(rendering, **kwargs)
      return lowering.render_to_text_as_root(rendering)

    with self.subTest("no_max_height"):
      self.assertEqual(
          render_and_expand(max_height=None, target_width=60),
          textwrap.dedent("""\
              DataclassWithOneChild(
                foo=[
                  ['foo', 'foo', 'foo', 'foo'],
                  [
                    '12345678901234567890',
                    '12345678901234567890',
                    '12345678901234567890',
                    '12345678901234567890',
                    '12345678901234567890',
                  ],
                  {
                    'a': 1,
                    'b': 2,
                    'c': [
                      {'bar': 'baz'},
                      {'bar': 'baz'},
                      {'bar': 'baz'},
                      {'bar': 'baz'},
                      {'bar': 'baz'},
                    ],
                  },
                  [
                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                  ],
                ],
              )"""),
      )

    with self.subTest("medium_max_height"):
      self.assertEqual(
          render_and_expand(max_height=20, target_width=60),
          textwrap.dedent("""\
              DataclassWithOneChild(
                foo=[
                  ['foo', 'foo', 'foo', 'foo'],
                  [
                    '12345678901234567890',
                    '12345678901234567890',
                    '12345678901234567890',
                    '12345678901234567890',
                    '12345678901234567890',
                  ],
                  {
                    'a': 1,
                    'b': 2,
                    'c': [{'bar': 'baz'}, {'bar': 'baz'}, {'bar': 'baz'}, {'bar': 'baz'}, {'bar': 'baz'}],
                  },
                  [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]],
                ],
              )"""),
      )

    with self.subTest("small_max_height"):
      self.assertEqual(
          render_and_expand(max_height=10, target_width=60),
          textwrap.dedent("""\
              DataclassWithOneChild(
                foo=[
                  ['foo', 'foo', 'foo', 'foo'],
                  ['12345678901234567890', '12345678901234567890', '12345678901234567890', '12345678901234567890', '12345678901234567890'],
                  {'a': 1, 'b': 2, 'c': [{'bar': 'baz'}, {'bar': 'baz'}, {'bar': 'baz'}, {'bar': 'baz'}, {'bar': 'baz'}]},
                  [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]],
                ],
              )"""),
      )

    with self.subTest("long_target_width"):
      self.assertEqual(
          render_and_expand(max_height=None, target_width=150),
          textwrap.dedent("""\
              DataclassWithOneChild(
                foo=[
                  ['foo', 'foo', 'foo', 'foo'],
                  ['12345678901234567890', '12345678901234567890', '12345678901234567890', '12345678901234567890', '12345678901234567890'],
                  {'a': 1, 'b': 2, 'c': [{'bar': 'baz'}, {'bar': 'baz'}, {'bar': 'baz'}, {'bar': 'baz'}, {'bar': 'baz'}]},
                  [
                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                  ],
                ],
              )"""),
      )

  def test_balanced_layout_after_manual_expansion(self):
    renderer = treescope.active_renderer.get()
    some_nested_object = [
        fixture_lib.DataclassWithOneChild(
            [["foo"] * 4, (["baz"] * 5, ["qux"] * 5)]
        )
    ]

    rendering = renderer.to_foldable_representation(
        some_nested_object
    ).renderable
    layout_algorithms.expand_for_balanced_layout(
        rendering,
        max_height=3,
        target_width=40,
        recursive_expand_height_for_collapsed_nodes=10,
    )

    # Initially collapsed due to height constraint.
    self.assertEqual(
        lowering.render_to_text_as_root(rendering),
        textwrap.dedent("""\
        [
          DataclassWithOneChild(foo=[['foo', 'foo', 'foo', 'foo'], (['baz', 'baz', 'baz', 'baz', 'baz'], ['qux', 'qux', 'qux', 'qux', 'qux'])]),
        ]"""),
    )

    # But expands multiple levels once we expand the collapsed node manually.
    # (In the browser, this would be done by clicking the expand marker.)
    target_foldable = (
        rendering.foldables_in_this_part()[0]
        .as_expanded_part()
        .foldables_in_this_part()[0]
    )
    self.assertEqual(
        target_foldable.get_expand_state(),
        rendering_parts.ExpandState.COLLAPSED,
    )
    target_foldable.set_expand_state(rendering_parts.ExpandState.EXPANDED)
    self.assertEqual(
        lowering.render_to_text_as_root(rendering),
        textwrap.dedent("""\
            [
              DataclassWithOneChild(
                foo=[
                  ['foo', 'foo', 'foo', 'foo'],
                  (
                    ['baz', 'baz', 'baz', 'baz', 'baz'],
                    ['qux', 'qux', 'qux', 'qux', 'qux'],
                  ),
                ],
              ),
            ]"""),
    )

  def test_balanced_layout_relaxes_height_constraint_once(self):
    renderer = treescope.active_renderer.get()
    some_nested_object = [
        fixture_lib.DataclassWithOneChild(
            [fixture_lib.DataclassWithOneChild(["abcdefghik"] * 20)]
        )
    ]

    # With a relaxed only-child expansion constraint (the default), we still
    # expand the large list because it's the only nontrivial object.
    rendering = renderer.to_foldable_representation(
        some_nested_object
    ).renderable
    layout_algorithms.expand_for_balanced_layout(
        rendering, max_height=10, target_width=40
    )
    self.assertEqual(
        lowering.render_to_text_as_root(rendering),
        textwrap.dedent("""\
            [
              DataclassWithOneChild(
                foo=[
                  DataclassWithOneChild(
                    foo=[
                      'abcdefghik',
                      'abcdefghik',
                      'abcdefghik',
                      'abcdefghik',
                      'abcdefghik',
                      'abcdefghik',
                      'abcdefghik',
                      'abcdefghik',
                      'abcdefghik',
                      'abcdefghik',
                      'abcdefghik',
                      'abcdefghik',
                      'abcdefghik',
                      'abcdefghik',
                      'abcdefghik',
                      'abcdefghik',
                      'abcdefghik',
                      'abcdefghik',
                      'abcdefghik',
                      'abcdefghik',
                    ],
                  ),
                ],
              ),
            ]"""),
    )

    # Without a relaxed only-child expansion constraint, it stays collapsed.
    rendering = renderer.to_foldable_representation(
        some_nested_object
    ).renderable
    layout_algorithms.expand_for_balanced_layout(
        rendering,
        max_height=10,
        target_width=40,
        relax_height_constraint_for_only_child=False,
    )
    self.assertEqual(
        lowering.render_to_text_as_root(rendering),
        textwrap.dedent("""\
            [
              DataclassWithOneChild(
                foo=[
                  DataclassWithOneChild(
                    foo=['abcdefghik', 'abcdefghik', 'abcdefghik', 'abcdefghik', 'abcdefghik', 'abcdefghik', 'abcdefghik', 'abcdefghik', 'abcdefghik', 'abcdefghik', 'abcdefghik', 'abcdefghik', 'abcdefghik', 'abcdefghik', 'abcdefghik', 'abcdefghik', 'abcdefghik', 'abcdefghik', 'abcdefghik', 'abcdefghik'],
                  ),
                ],
              ),
            ]"""),
    )


if __name__ == "__main__":
  absltest.main()
