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

"""Pytest configuration shared by all tests."""

# Number of CPU devices that JAX should expose, so that sharded arrays can be
# tested without accelerators. Tests that rely on this should check that it
# took effect, since it only applies if JAX has not initialized its backends
# yet.
NUM_CPU_DEVICES = 8

try:
  import jax
except ImportError:
  pass
else:
  jax.config.update("jax_num_cpu_devices", NUM_CPU_DEVICES)
