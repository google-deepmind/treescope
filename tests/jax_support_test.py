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

import contextlib
import json
import os
import socket
import subprocess
import sys
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from jax import numpy as jnp
import jax
import numpy as np
import treescope
import treescope.external.jax_support
from treescope.external import jax_support
from treescope.external import numpy_support
from . import conftest
from . import helpers

P = jax.sharding.PartitionSpec


class JaxSupportTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("bfloat16", jnp.bfloat16), ("float32", jnp.float32)
  )
  def test_compute_summary(self, dtype):
    expected = " ≈5e-05 ±2.9e-05 [≥5e-06, ≤9.5e-05] nonzero:10"
    inp = jnp.arange(10, dtype=dtype) * 1e-5 + 5e-6
    self.assertEqual(
        treescope.external.jax_support.summarize_array_data(inp), expected
    )

  def test_summarize_prng_key(self):
    keys = jax.random.split(jax.random.key(0, impl="threefry2x32"), 10)
    summarized = treescope.external.jax_support.summarize_array_data(keys)
    self.assertEqual(summarized, "")

    full_summary = helpers.ensure_text(
        treescope.external.jax_support.JAXArrayAdapter().get_array_summary(
            keys, fast=False
        )
    )
    self.assertEqual(full_summary, "jax.Array key<fry>(10,)")

  def test_faster_array_repr_large_prng_key_array(self):
    keys = jax.random.split(jax.random.key(0, impl="threefry2x32"), 1001)

    rendered = treescope.external.jax_support.faster_array_repr(keys)

    self.assertEqual(rendered, repr(keys))

  def test_set_up_treescope_rejects_old_jax(self):
    registry = treescope.type_registries.TREESCOPE_HANDLER_REGISTRY
    registered_before = dict(registry)
    with mock.patch.object(jax, "__version_info__", (0, 4, 37)):
      with self.assertWarnsRegex(UserWarning, "requires JAX >= 0.7.1"):
        treescope.external.jax_support.set_up_treescope()
    self.assertEqual(dict(registry), registered_before)


def _cpu_mesh(axis_type: jax.sharding.AxisType) -> jax.sharding.Mesh:
  return jax.make_mesh(
      (2, 4),
      ("a", "b"),
      axis_types=(axis_type, axis_type),
      devices=jax.devices("cpu")[:8],
  )


def _force_device_truncation():
  """Makes small arrays use the same truncation code as very large ones."""
  return mock.patch.object(jax_support, "SUMMARIZE_USING_NUMPY_THRESHOLD", 0)


def _force_device_summary():
  """Makes small arrays use the same summary code as very large ones."""
  stack = contextlib.ExitStack()
  stack.enter_context(_force_device_truncation())
  stack.enter_context(
      jax_support.summarization_threshold.set_scoped({"default": None})
  )
  return stack


class ShardedJaxArrayTest(parameterized.TestCase):
  """Tests for sharded arrays, using the CPU devices set up in conftest.py."""

  def setUp(self):
    super().setUp()
    # Make sure the multi-device setup in conftest.py took effect; otherwise
    # these tests would silently test less than they claim to.
    self.assertLen(
        jax.devices("cpu"),
        conftest.NUM_CPU_DEVICES,
        msg="jax_num_cpu_devices from tests/conftest.py did not take effect.",
    )
    # Tests must not leak a mesh context into each other.
    self.assertTrue(jax.sharding.get_abstract_mesh().empty)

  @parameterized.product(
      axis_type=["Explicit", "Auto"],
      spec=[P("a", "b"), P(None, ("a", "b")), P("b", None), P()],
      mesh_context=["none", "same", "other"],
  )
  def test_truncation_matches_numpy(self, axis_type, spec, mesh_context):
    axis_type = getattr(jax.sharding.AxisType, axis_type)
    mesh = _cpu_mesh(axis_type)
    other_mesh = jax.make_mesh(
        (8,), ("z",), axis_types=(axis_type,), devices=jax.devices("cpu")[:8]
    )
    data = np.arange(64 * 48, dtype=np.float32).reshape(64, 48) - 100
    data[0, 0] = np.nan
    data[-1, -1] = -0.0
    array = jax.device_put(data, jax.sharding.NamedSharding(mesh, spec))
    edge_items = (5, 3)
    (
        expected_values,
        expected_valid,
    ) = numpy_support.NumpyArrayAdapter().get_array_data_with_truncation(
        data, None, edge_items
    )

    # Note: `jax.set_mesh` takes effect as soon as it is called, so only call
    # it for the context we actually enter.
    make_context = {
        "none": contextlib.nullcontext,
        "same": lambda: jax.set_mesh(mesh),
        "other": lambda: jax.set_mesh(other_mesh),
    }[mesh_context]
    with _force_device_truncation(), make_context():
      (
          values,
          valid,
      ) = jax_support.JAXArrayAdapter().get_array_data_with_truncation(
          array, None, edge_items
      )

    # Compare bit patterns, so that NaN and -0.0 must match exactly.
    self.assertEqual(values.dtype, expected_values.dtype)
    self.assertEqual(values.tobytes(), expected_values.tobytes())
    np.testing.assert_array_equal(valid, expected_valid)

  @parameterized.named_parameters(
      ("bfloat16", jnp.bfloat16),
      ("int8", jnp.int8),
      ("uint32", jnp.uint32),
      ("bool", jnp.bool_),
      ("complex64", jnp.complex64),
  )
  def test_truncation_dtypes(self, dtype):
    mesh = _cpu_mesh(jax.sharding.AxisType.Explicit)
    data = (np.arange(16 * 24 * 40).reshape(16, 24, 40) % 7).astype(dtype)
    array = jax.device_put(
        data, jax.sharding.NamedSharding(mesh, P(None, "a", "b"))
    )
    edge_items = (2, None, 4)
    (
        expected_values,
        expected_valid,
    ) = numpy_support.NumpyArrayAdapter().get_array_data_with_truncation(
        data, None, edge_items
    )
    with _force_device_truncation():
      values, valid = jax_support.truncate_array_and_mask(
          array, np.array(True).reshape(1, 1, 1), edge_items
      )
    self.assertEqual(values.dtype, expected_values.dtype)
    self.assertEqual(values.tobytes(), expected_values.tobytes())
    np.testing.assert_array_equal(valid, expected_valid)

  @parameterized.named_parameters(
      ("no_mesh_context", False), ("under_global_mesh", True)
  )
  def test_truncation_single_device_array(self, under_global_mesh):
    # Users of explicit sharding often set a global mesh with `jax.set_mesh`,
    # and single-device arrays must still render under it.
    mesh = _cpu_mesh(jax.sharding.AxisType.Explicit)
    data = np.arange(30 * 20, dtype=np.int32).reshape(30, 20)
    array = jax.device_put(data, jax.devices("cpu")[3])
    (
        expected_values,
        expected_valid,
    ) = numpy_support.NumpyArrayAdapter().get_array_data_with_truncation(
        data, None, (4, 4)
    )
    context = (
        jax.set_mesh(mesh) if under_global_mesh else contextlib.nullcontext()
    )
    with _force_device_truncation(), context:
      (
          values,
          valid,
      ) = jax_support.JAXArrayAdapter().get_array_data_with_truncation(
          array, None, (4, 4)
      )
    np.testing.assert_array_equal(values, expected_values)
    np.testing.assert_array_equal(valid, expected_valid)

  @parameterized.named_parameters(
      ("full_sharded_mask", (64, 48), P("a", "b")),
      ("broadcast_sharded_mask", (64, 1), P("a", None)),
      ("full_single_device_mask", (64, 48), None),
  )
  def test_truncation_with_mask(self, mask_shape, mask_spec):
    mesh = _cpu_mesh(jax.sharding.AxisType.Explicit)
    data = np.arange(64 * 48, dtype=np.float32).reshape(64, 48)
    mask_data = np.random.default_rng(0).random(mask_shape) > 0.5
    array = jax.device_put(data, jax.sharding.NamedSharding(mesh, P("a", "b")))
    if mask_spec is None:
      mask = jax.device_put(mask_data, jax.devices("cpu")[0])
    else:
      mask = jax.device_put(
          mask_data, jax.sharding.NamedSharding(mesh, mask_spec)
      )
    edge_items = (5, 3)
    (
        expected_values,
        expected_valid,
    ) = numpy_support.NumpyArrayAdapter().get_array_data_with_truncation(
        data, mask_data, edge_items
    )
    with _force_device_truncation():
      (
          values,
          valid,
      ) = jax_support.JAXArrayAdapter().get_array_data_with_truncation(
          array, mask, edge_items
      )
    np.testing.assert_array_equal(values, expected_values)
    np.testing.assert_array_equal(valid, expected_valid)

  @parameterized.product(
      axis_type=["Explicit", "Auto"],
      mesh_context=["none", "same", "other"],
  )
  def test_summary_of_sharded_array(self, axis_type, mesh_context):
    axis_type = getattr(jax.sharding.AxisType, axis_type)
    mesh = _cpu_mesh(axis_type)
    other_mesh = jax.make_mesh(
        (8,), ("z",), axis_types=(axis_type,), devices=jax.devices("cpu")[:8]
    )
    data = np.arange(64 * 48, dtype=np.float32).reshape(64, 48) - 100
    data[0, 0] = np.nan
    array = jax.device_put(data, jax.sharding.NamedSharding(mesh, P("a", "b")))
    # Small arrays are normally summarized on the host, using NumPy.
    expected = jax_support.summarize_array_data(jnp.asarray(data))

    make_context = {
        "none": contextlib.nullcontext,
        "same": lambda: jax.set_mesh(mesh),
        "other": lambda: jax.set_mesh(other_mesh),
    }[mesh_context]
    with _force_device_summary(), make_context():
      summary = jax_support.summarize_array_data(array)
    self.assertEqual(summary, expected)

  def test_summary_of_single_device_array_under_global_mesh(self):
    mesh = _cpu_mesh(jax.sharding.AxisType.Explicit)
    data = np.arange(30 * 20, dtype=np.float32).reshape(30, 20)
    array = jax.device_put(data, jax.devices("cpu")[3])
    expected = jax_support.summarize_array_data(array)  # On the host.
    with _force_device_summary(), jax.set_mesh(mesh):
      summary = jax_support.summarize_array_data(array)
    self.assertEqual(summary, expected)

  def test_render_explicitly_sharded_array_regression(self):
    # Regression test for https://github.com/google-deepmind/treescope/issues/70
    # Uses the default thresholds: on CPU, arrays this large are truncated
    # on device and are too large to summarize.
    mesh = _cpu_mesh(jax.sharding.AxisType.Explicit)
    data = np.arange(512 * 256, dtype=np.float32).reshape(512, 256)
    array = jax.device_put(data, jax.sharding.NamedSharding(mesh, P("b", "a")))

    # The text rendering should match JAX's own repr of the same data on a
    # single device, which copies the whole array to the host.
    expected_repr = repr(jax.device_put(data, jax.devices("cpu")[0]))
    self.assertEqual(jax_support.faster_array_repr(array), expected_repr)
    self.assertIn(
        expected_repr.splitlines()[0], treescope.render_to_text(array)
    )

    with treescope.active_autovisualizer.set_scoped(
        treescope.ArrayAutovisualizer()
    ):
      html = treescope.render_to_html(array, compressed=False)
    self.assertIn("arrayviz", html)


class MultiProcessJaxArrayTest(absltest.TestCase):
  """Tests arrays that span several processes, as on multi-host systems.

  Launches several local processes (see `multiprocess_worker.py`) that use
  `jax.distributed` to act as the hosts of one JAX cluster.
  """

  NUM_PROCESSES = 2
  TIMEOUT_SECONDS = 120

  def test_truncation_of_array_spanning_processes(self):
    # Ask the OS for a free port for the coordinator.
    with socket.socket() as sock:
      sock.bind(("127.0.0.1", 0))
      port = sock.getsockname()[1]
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [os.path.dirname(tests_dir), env.get("PYTHONPATH", "")]
    )
    processes = [
        subprocess.Popen(
            [
                sys.executable,
                os.path.join(tests_dir, "multiprocess_worker.py"),
                str(process_id),
                str(self.NUM_PROCESSES),
                str(port),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        for process_id in range(self.NUM_PROCESSES)
    ]
    outputs = []
    try:
      for process in processes:
        outputs.append(process.communicate(timeout=self.TIMEOUT_SECONDS)[0])
    finally:
      # If one process failed, the others may be stuck waiting for it.
      for process in processes:
        if process.poll() is None:
          process.kill()
          process.communicate()

    for process_id, (process, output) in enumerate(zip(processes, outputs)):
      self.assertEqual(process.returncode, 0, msg=output)
      result_lines = [
          line for line in output.splitlines() if line.startswith("RESULT ")
      ]
      self.assertLen(result_lines, 1, msg=output)
      self.assertEqual(
          json.loads(result_lines[0].removeprefix("RESULT ")),
          {
              "process_id": process_id,
              # Checks that the multi-process setup actually took effect.
              "global_devices": 2 * self.NUM_PROCESSES,
              "local_devices": 2,
              "Auto": "match",
              "Explicit": "match",
          },
          msg=output,
      )


if __name__ == "__main__":
  absltest.main()
