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

"""Tests for NDArray adapters and array visualization."""

from logging import warn, warning
import re
import warnings

from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
import numpy as np
import torch
import treescope
from treescope import ndarray_adapters
from treescope import type_registries
from . import helpers


class NdarrayAdaptersTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    type_registries.update_registries_for_imports()

  @parameterized.product(
      array_type=["jax", "torch"],
      dtype=[np.int32, np.float32, np.bool_],
  )
  def test_adapter_positional_numpy_consistency(self, array_type, dtype):

    reshaped_arange = np.arange(19 * 23).reshape((19, 23))

    if dtype == np.bool_:
      array_np = (reshaped_arange % 2) == 0
    else:
      array_np = reshaped_arange.astype(dtype)

    mask_np = (reshaped_arange % 3) != 0

    if array_type == "jax":
      array = jnp.array(array_np)
      mask = jnp.array(mask_np)
    elif array_type == "torch":
      array = torch.tensor(array_np)
      mask = torch.tensor(mask_np)
    else:
      raise ValueError(f"Unsupported array_type: {array_type}")

    np_adapter = type_registries.lookup_ndarray_adapter(array_np)
    self.assertIsNotNone(np_adapter)

    cur_adapter = type_registries.lookup_ndarray_adapter(array)
    self.assertIsNotNone(cur_adapter)

    with self.subTest("axis_info"):
      self.assertEqual(
          np_adapter.get_axis_info_for_array_data(array_np),
          cur_adapter.get_axis_info_for_array_data(array),
      )

    with self.subTest("data_untruncated_unmasked"):
      data_out_np, mask_out_np = np_adapter.get_array_data_with_truncation(
          array=array_np, mask=None, edge_items_per_axis=(None, None)
      )
      data_out, mask_out = cur_adapter.get_array_data_with_truncation(
          array=array, mask=None, edge_items_per_axis=(None, None)
      )
      np.testing.assert_array_equal(data_out_np, data_out)
      np.testing.assert_array_equal(mask_out_np, mask_out)

    with self.subTest("data_untruncated_masked"):
      data_out_np, mask_out_np = np_adapter.get_array_data_with_truncation(
          array=array_np, mask=mask_np, edge_items_per_axis=(None, None)
      )
      data_out, mask_out = cur_adapter.get_array_data_with_truncation(
          array=array, mask=mask, edge_items_per_axis=(None, None)
      )
      np.testing.assert_array_equal(data_out_np, data_out)
      np.testing.assert_array_equal(mask_out_np, mask_out)

    with self.subTest("data_semitruncated_unmasked"):
      data_out_np, mask_out_np = np_adapter.get_array_data_with_truncation(
          array=array_np, mask=None, edge_items_per_axis=(2, None)
      )
      data_out, mask_out = cur_adapter.get_array_data_with_truncation(
          array=array, mask=None, edge_items_per_axis=(2, None)
      )
      np.testing.assert_array_equal(data_out_np, data_out)
      np.testing.assert_array_equal(mask_out_np, mask_out)

    with self.subTest("data_semitruncated_unmasked"):
      data_out_np, mask_out_np = np_adapter.get_array_data_with_truncation(
          array=array_np, mask=mask_np, edge_items_per_axis=(2, None)
      )
      data_out, mask_out = cur_adapter.get_array_data_with_truncation(
          array=array, mask=mask, edge_items_per_axis=(2, None)
      )
      np.testing.assert_array_equal(data_out_np, data_out)
      np.testing.assert_array_equal(mask_out_np, mask_out)

    with self.subTest("data_truncated_unmasked"):
      data_out_np, mask_out_np = np_adapter.get_array_data_with_truncation(
          array=array_np, mask=None, edge_items_per_axis=(2, 4)
      )
      data_out, mask_out = cur_adapter.get_array_data_with_truncation(
          array=array, mask=None, edge_items_per_axis=(2, 4)
      )
      np.testing.assert_array_equal(data_out_np, data_out)
      np.testing.assert_array_equal(mask_out_np, mask_out)

    with self.subTest("data_truncated_unmasked"):
      data_out_np, mask_out_np = np_adapter.get_array_data_with_truncation(
          array=array_np, mask=mask_np, edge_items_per_axis=(2, 4)
      )
      data_out, mask_out = cur_adapter.get_array_data_with_truncation(
          array=array, mask=mask, edge_items_per_axis=(2, 4)
      )
      np.testing.assert_array_equal(data_out_np, data_out)
      np.testing.assert_array_equal(mask_out_np, mask_out)

    with self.subTest("data_truncated_broadcast_mask"):
      data_out_np, mask_out_np = np_adapter.get_array_data_with_truncation(
          array=array_np, mask=mask_np[0, :], edge_items_per_axis=(3, 7)
      )
      data_out, mask_out = cur_adapter.get_array_data_with_truncation(
          array=array, mask=mask[0, :], edge_items_per_axis=(3, 7)
      )
      np.testing.assert_array_equal(data_out_np, data_out)
      np.testing.assert_array_equal(mask_out_np, mask_out)

    for fast in (True, False):
      with self.subTest("summary_fast" if fast else "summary_slow"):
        summary_info_np = helpers.ensure_text(
            np_adapter.get_array_summary(array_np, fast=True)
        )
        summary_info = helpers.ensure_text(
            cur_adapter.get_array_summary(array, fast=True)
        )
        self.assertEqual(
            summary_info_np.split(" ", 1)[1], summary_info.split(" ", 1)[1]
        )

  def test_pytorch_named_axes_info(self):
    data = np.arange(19 * 23).reshape((19, 23))
    with warnings.catch_warnings():
      warnings.filterwarnings(
          "ignore",
          category=UserWarning,
          module="torch",
          message=re.escape(
              "Named tensors and all their associated APIs are an experimental"
              " feature and subject to change."
          ),
      )
      array = torch.tensor(data).rename("foo", None)
      adapter = type_registries.lookup_ndarray_adapter(array)
      self.assertIsNotNone(adapter)
      self.assertEqual(
          adapter.get_axis_info_for_array_data(array),
          (
              ndarray_adapters.NamedPositionalAxisInfo(0, "foo", 19),
              ndarray_adapters.PositionalAxisInfo(1, 23),
          ),
      )

  @parameterized.product(
      array_type=["numpy", "jax", "torch"],
      dtype=[np.int32, np.float32, np.bool_],
  )
  def test_array_rendering_without_error(self, array_type, dtype):
    reshaped_arange = np.arange(19 * 23).reshape((19, 23))

    if dtype == np.bool_:
      array_np = (reshaped_arange % 2) == 0
    else:
      array_np = reshaped_arange.astype(dtype)

    if array_type == "numpy":
      array = array_np
    elif array_type == "jax":
      array = jnp.array(array_np)
    elif array_type == "torch":
      array = torch.tensor(array_np)
    else:
      raise ValueError(f"Unsupported array_type: {array_type}")

    with self.subTest("explicit_unmasked"):
      res = treescope.render_array(array)
      self.assertTrue(hasattr(res, "_repr_html_"))

    with self.subTest("explicit_masked"):
      res = treescope.render_array(array, valid_mask=array > 100)
      self.assertTrue(hasattr(res, "_repr_html_"))

    with self.subTest("explicit_masked_truncated"):
      res = treescope.render_array(
          array, valid_mask=array > 100, truncate=True, maximum_size=100
      )
      self.assertTrue(hasattr(res, "_repr_html_"))

    with self.subTest("automatic"):
      with treescope.active_autovisualizer.set_scoped(
          treescope.ArrayAutovisualizer()
      ):
        res = treescope.render_to_html(
            array, ignore_exceptions=False, compressed=False
        )
        self.assertIsInstance(res, str)
        self.assertIn("arrayviz", res)


if __name__ == "__main__":
  absltest.main()
