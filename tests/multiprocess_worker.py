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

"""One process of a multi-process JAX test, launched by jax_support_test.py.

Usage: python multiprocess_worker.py <process_id> <num_processes> <port>

Each process exposes 2 CPU devices, joins the other processes through
`jax.distributed`, truncates arrays sharded across all processes' devices,
and prints its results as a JSON line prefixed with "RESULT ".
"""

import json
import sys
from unittest import mock

import jax
import numpy as np

LOCAL_DEVICES = 2


def main():
  process_id, num_processes, port = (int(arg) for arg in sys.argv[1:4])
  jax.config.update("jax_num_cpu_devices", LOCAL_DEVICES)
  jax.config.update("jax_cpu_collectives_implementation", "gloo")
  jax.distributed.initialize(
      coordinator_address=f"127.0.0.1:{port}",
      num_processes=num_processes,
      process_id=process_id,
  )
  # Import treescope only after JAX is configured.
  from treescope.external import jax_support  # pylint: disable=import-outside-toplevel
  from treescope.external import numpy_support  # pylint: disable=import-outside-toplevel

  results: dict[str, int | str] = {
      "process_id": process_id,
      "global_devices": len(jax.devices()),
      "local_devices": len(jax.local_devices()),
  }
  data = np.arange(64 * 48, dtype=np.float32).reshape(64, 48)
  edge_items = (5, 3)
  (
      expected,
      _,
  ) = numpy_support.NumpyArrayAdapter().get_array_data_with_truncation(
      data, None, edge_items
  )
  for axis_type in ("Auto", "Explicit"):
    mesh = jax.make_mesh(
        (len(jax.devices()),),
        ("a",),
        axis_types=(getattr(jax.sharding.AxisType, axis_type),),
    )
    array = jax.make_array_from_callback(
        data.shape,
        jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("a")),
        lambda index: data[index],
    )
    try:
      with mock.patch.object(jax_support, "SUMMARIZE_USING_NUMPY_THRESHOLD", 0):
        values, _ = jax_support.truncate_array_and_mask(
            array, np.ones((1, 1), dtype=bool), edge_items
        )
      if np.array_equal(values, expected):
        results[axis_type] = "match"
      else:
        results[axis_type] = "mismatch"
    except Exception as e:  # pylint: disable=broad-exception-caught
      results[axis_type] = f"{type(e).__name__}: {e}"
  print("RESULT " + json.dumps(results), flush=True)


if __name__ == "__main__":
  main()
