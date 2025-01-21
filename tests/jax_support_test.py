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
