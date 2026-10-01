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

import os
import subprocess
import sys
import textwrap

from absl.testing import absltest
from absl.testing import parameterized
from jax import numpy as jnp
import jax
import treescope.external.jax_support
from . import helpers


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

  def test_truncate_explicitly_sharded_array(self):
    if not hasattr(jax.sharding, "AxisType"):
      self.skipTest("JAX does not support explicit sharding.")

    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
    script = textwrap.dedent(
        """
        import jax
        import numpy as np
        import treescope.external.jax_support

        devices = np.asarray(jax.devices())
        mesh = jax.sharding.Mesh(
            devices.reshape((1, 4)),
            axis_names=("dp", "tp"),
            axis_types=(
                jax.sharding.AxisType.Explicit,
                jax.sharding.AxisType.Explicit,
            ),
        )
        sharding = jax.sharding.NamedSharding(
            mesh, jax.sharding.PartitionSpec("tp", None)
        )
        original = np.arange(1600 * 64, dtype=np.int32).reshape((1600, 64))
        array = jax.device_put(original, sharding)

        truncated, valid = (
            treescope.external.jax_support.JAXArrayAdapter()
            .get_array_data_with_truncation(array, None, (3, 3))
        )

        assert truncated.shape == (7, 7)
        np.testing.assert_array_equal(truncated[:3, :3], original[:3, :3])
        np.testing.assert_array_equal(truncated[-3:, -3:], original[-3:, -3:])
        expected_valid = np.ones((7, 7), dtype=bool)
        expected_valid[3, :] = False
        expected_valid[:, 3] = False
        np.testing.assert_array_equal(valid, expected_valid)
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    self.assertEqual(completed.returncode, 0, completed.stderr)

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


if __name__ == "__main__":
  absltest.main()
