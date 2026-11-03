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

"""Lazy setup logic for adding JAX support to treescope."""

from __future__ import annotations

import functools
import typing
from typing import Any, Mapping, Sequence
import warnings

import numpy as np
from treescope import canonical_aliases
from treescope import context
from treescope import dtype_util
from treescope import lowering
from treescope import ndarray_adapters
from treescope import renderers
from treescope import rendering_parts
from treescope import repr_lib
from treescope import type_registries
from treescope.external import numpy_support

# pylint: disable=import-outside-toplevel
try:
  import jax
except ImportError:
  assert not typing.TYPE_CHECKING
  jax = None
# pylint: enable=import-outside-toplevel


MINIMUM_JAX_VERSION = (0, 7, 1)
"""The oldest JAX version supported by treescope's JAX integration.

With older versions of JAX, `set_up_treescope` warns and leaves JAX support
disabled, so that JAX values are rendered like other Python objects.
"""


def _is_subdtype(dtype, base) -> bool:
  """Safely checks for dtype subtyping."""
  assert jax is not None
  jnp = jax.numpy
  try:
    return jnp.issubdtype(dtype, base)
  except TypeError:
    return False


summarization_threshold: context.ContextualValue[Mapping[str, int | None]] = (
    context.ContextualValue(
        module=__name__,
        qualname="summarization_threshold",
        initial_value={
            "tpu": 1_000_000_000,
            "gpu": 10_000_000,
            "default": 100_000,
        },
    )
)
"""Threshold for summarization of NDArrays for each backend.

This threshold determines the largest number of elements we will
summarize with summary statistics (e.g. mean, standard deviation)
when rendering in treescope. Larger values may make it slower to
display large NDArrays.

Each key should be the name of a JAX array platform, e.g. "cpu" or
"tpu". It can also be "numpy" to refer to Numpy arrays, or "default"
to refer to any other accelerator. The value is the size of the
array at which point we avoid showing summary statistics. `None`
means no limit.

This configuration argument is intended to be set at the top level
by the user, e.g. in IPython.
"""


SUMMARIZE_USING_NUMPY_THRESHOLD = 10_000_000
"""Threshold for using NumPy to summarize and render JAX Arrays.

Moving arrays to main memory and using numpy is significantly faster because
it avoid jitting the summary and rendering functions.
"""


def _is_locally_available(array: jax.Array) -> bool:
  """Checks if the array is available locally."""
  return getattr(array, "is_fully_addressable", False) or getattr(
      array, "is_fully_replicated", False
  )


def safe_to_summarize(array: jax.Array) -> bool:
  """Checks if the array is safe to summarize (not a tracer, not replicated)."""
  assert jax is not None, "JAX is not available."
  if isinstance(array, jax.core.Tracer):
    return False
  if array.is_deleted():
    return False
  if not _is_locally_available(array):
    return False
  thresh_dict = summarization_threshold.get()
  [platform] = set(device.platform for device in array.devices())
  thresh = thresh_dict.get(platform)
  if thresh is None:
    thresh = thresh_dict["default"]
  return thresh is None or array.size < thresh


def _as_mesh_array(array: jax.Array) -> tuple[jax.Array, jax.sharding.Mesh]:
  """Returns a view of `array` with a `NamedSharding`, along with its mesh.

  JAX arrays either have a `NamedSharding` or live on a single device. A
  single-device array is re-wrapped (without copying) as a replicated array on
  a one-device mesh, so that it can be handled the same way as a sharded one.

  Args:
    array: The array to view.

  Returns:
    A tuple ``(array_view, mesh)``, where ``array_view.sharding`` is a
    `NamedSharding` over ``mesh``.
  """
  assert jax is not None, "JAX is not available."
  sharding = array.sharding
  if isinstance(sharding, jax.sharding.NamedSharding):
    return array, sharding.mesh
  devices = list(sharding.device_set)
  if len(devices) != 1:
    raise ValueError(f"Unsupported sharding for a JAX array: {sharding}")
  mesh = jax.sharding.Mesh(np.array(devices), ("single_device",))
  array_view = jax.make_array_from_single_device_arrays(
      array.shape,
      jax.sharding.NamedSharding(
          mesh,
          jax.sharding.PartitionSpec(),
          memory_kind=sharding.memory_kind,
      ),
      [shard.data for shard in array.addressable_shards],
  )
  return array_view, mesh


def _edge_positions(size: int, edge_items: int | None) -> np.ndarray:
  """Returns the source index for each position along a truncated axis."""
  if edge_items is None:
    return np.arange(size)
  # The gap element in the middle is marked with -1, which no shard contains.
  return np.concatenate([
      np.arange(edge_items),
      [-1],
      np.arange(size - edge_items, size),
  ])


def _gather_edges_from_each_shard(
    array: jax.Array,
    mesh: jax.sharding.Mesh,
    spec: jax.sharding.PartitionSpec,
    edge_items_per_axis: tuple[int | None, ...],
) -> tuple[jax.Array, jax.Array]:
  """Gathers the edge elements of every shard of an array, in one program.

  Each device computes which edge elements (in global coordinates) fall
  inside its local block of the array, and gathers them locally. The small
  per-shard results are then all-gathered, so that the output is replicated
  across devices (and readable from every process, for arrays that span
  multiple hosts). This works for any sharding, including
  explicit-mode (`jax.sharding.AxisType.Explicit`) shardings, which do not
  support slicing a sharded axis to a size that isn't divisible by the number
  of shards.

  Args:
    array: The array to truncate.
    mesh: The mesh of the array's `NamedSharding`.
    spec: The partition spec of the array's `NamedSharding`.
    edge_items_per_axis: Number of edge items to keep for each axis, or None
      to keep the whole axis.

  Returns:
    A tuple ``(candidates, owned)`` of replicated arrays whose shape is the
    truncated shape with a new leading axis, which has one entry for each
    distinct shard. ``owned[k]`` is True for the elements that shard ``k`` contains,
    and ``candidates[k]`` holds their values (other entries are arbitrary).
    Each element of the truncated array, except for the gap elements, is
    owned by exactly one shard.
  """
  assert jax is not None, "JAX is not available."
  jnp = jax.numpy
  spec = tuple(spec) + (None,) * (array.ndim - len(spec))
  mesh_axes_per_dim = [
      () if entry is None else (entry,) if isinstance(entry, str) else entry
      for entry in spec
  ]
  sharded_mesh_axes = tuple(
      name for names in mesh_axes_per_dim for name in names
  )
  positions_per_dim = [
      _edge_positions(size, edge_items)
      for size, edge_items in zip(array.shape, edge_items_per_axis)
  ]

  def gather_local_edges(block):
    values = block
    owned = jnp.ones((1,) * block.ndim, dtype=bool)
    for dim, positions in enumerate(positions_per_dim):
      block_size = block.shape[dim]
      if mesh_axes_per_dim[dim]:
        block_start = jax.lax.axis_index(mesh_axes_per_dim[dim]) * block_size
      else:
        block_start = 0
      local_positions = jnp.asarray(positions) - block_start
      in_block = (local_positions >= 0) & (local_positions < block_size)
      values = jnp.take(
          values, jnp.clip(local_positions, 0, block_size - 1), axis=dim
      )
      broadcast_shape = [1] * block.ndim
      broadcast_shape[dim] = len(positions)
      owned = owned & in_block.reshape(broadcast_shape)
    owned = jnp.broadcast_to(owned, values.shape)
    if not sharded_mesh_axes:
      return values[None], owned[None]
    # Collect every shard's (small) result on every device, so that the
    # output is replicated. This keeps it readable from any process when the
    # array spans multiple hosts.
    return (
        jax.lax.all_gather(values, sharded_mesh_axes),
        jax.lax.all_gather(owned, sharded_mesh_axes),
    )

  replicated = jax.sharding.PartitionSpec()
  return jax.shard_map(
      gather_local_edges,
      mesh=mesh,
      in_specs=jax.sharding.PartitionSpec(*spec),
      out_specs=(replicated, replicated),
      # The outputs are all-gathered, so they are the same on every device,
      # but JAX's static replication check doesn't treat `all_gather` results
      # as replicated.
      check_vma=False,
  )(array)


_gather_edges_from_each_shard_jit = (
    jax.jit(_gather_edges_from_each_shard, static_argnums=(1, 2, 3))
    if jax is not None
    else None
)


def _truncate_on_device(
    array: jax.Array,
    edge_items_per_axis: tuple[int | None, ...],
) -> tuple[np.ndarray, np.ndarray]:
  """Truncates a JAX array on its devices and copies the result to the host.

  Args:
    array: The array to truncate.
    edge_items_per_axis: Number of edge items to keep for each axis, or None
      to keep the whole axis.

  Returns:
    A tuple ``(values, valid)`` of host arrays with the truncated shape.
    ``valid`` is False only for the gap elements, whose values are zero.
  """
  assert jax is not None, "JAX is not available."
  array, mesh = _as_mesh_array(array)
  # Always run under the array's own mesh, so that this doesn't depend on any
  # mesh context set by the caller: JAX rejects computations on an array under
  # a mesh context for a different mesh (e.g. a single-device array while a
  # global mesh is set).
  with jax.set_mesh(mesh):
    candidates, owned = _gather_edges_from_each_shard_jit(
        array, mesh, array.sharding.spec, edge_items_per_axis
    )
  candidates, owned = jax.device_get((candidates, owned))
  owner = np.argmax(owned, axis=0)
  values = np.take_along_axis(candidates, owner[None], axis=0)[0]
  valid = np.any(owned, axis=0)
  values[~valid] = 0
  return values, valid


def truncate_array_and_mask(
    array: jax.Array,
    mask: jax.Array | np.ndarray,
    edge_items_per_axis: tuple[int | None, ...],
) -> tuple[np.ndarray, np.ndarray]:
  """Truncates an array along the given axes, and copies it to the host.

  Args:
    array: Array to truncate.
    mask: Mask array, which must have the same number of dimensions as `array`,
      and whose axis sizes must be either 1 or the same as that axis of `array`
      (e.g. they are broadcast compatible).
    edge_items_per_axis: Number of edge items to keep for each axis, or None
      to keep the whole axis.

  Returns:
    A tuple containing a truncated version of the array along with a valid
    mask, both as NumPy arrays. Values taken from the original array have the
    valid mask as True (if the original mask was True), and there is one
    extra element in the middle of each truncated axis with valid as False
    (standing in for the omitted elements).
  """
  assert jax is not None, "JAX is not available."
  numpy_adapter = numpy_support.NumpyArrayAdapter()
  if array.size < SUMMARIZE_USING_NUMPY_THRESHOLD and safe_to_summarize(array):
    # Copying a small array to the host is cheaper than compiling a program.
    return numpy_adapter.get_array_data_with_truncation(
        np.asarray(array), np.asarray(mask), edge_items_per_axis
    )

  values, valid = _truncate_on_device(array, edge_items_per_axis)
  # Truncate the mask too, keeping any length-1 (broadcasted) axes as-is.
  mask_edge_items_per_axis = tuple(
      None if mask_size == 1 else edge_items
      for mask_size, edge_items in zip(np.shape(mask), edge_items_per_axis)
  )
  if isinstance(mask, jax.Array) and any(
      edge_items is not None for edge_items in mask_edge_items_per_axis
  ):
    mask_values, _ = _truncate_on_device(mask, mask_edge_items_per_axis)
  else:
    mask_values, _ = numpy_adapter.get_array_data_with_truncation(
        np.asarray(mask), None, mask_edge_items_per_axis
    )
  return values, valid & mask_values.astype(bool)


def faster_array_repr(array: jax.Array) -> str:
  """Computes ``repr(array)``, only copying the rendered array elements.

  ``repr(array)`` on a very large jax Array can be slow, because it copies the
  entire array to host memory even when only a few elements are actually needed.
  We can avoid this by truncating the array on device before fetching it.

  Args:
    array: The array to summarize.

  Returns:
    A string representation of the array. May differ slightly from the ordinary
    ``repr``, but should contain the same elements.
  """
  assert jax is not None, "JAX is not available."
  jnp = jax.numpy
  if array.size < np.get_printoptions()["threshold"]:
    return repr(array)

  is_prng_key = jax.dtypes.issubdtype(array.dtype, jax.dtypes.prng_key)
  if is_prng_key:
    # Typed PRNG keys deliberately cannot be converted to NumPy arrays. Render
    # their underlying key data instead, matching JAX's native representation.
    display_array = jax.random.key_data(array)
  else:
    display_array = array

  if array.aval is not None and array.aval.weak_type:
    dtype_str = f"dtype={array.dtype.name}, weak_type=True)"
  else:
    dtype_str = f"dtype={array.dtype.name})"

  edgeitems = np.get_printoptions()["edgeitems"]
  edge_items_per_axis = []
  for size in display_array.shape:
    if size > 2 * edgeitems + 1:
      edge_items_per_axis.append(edgeitems)
    else:
      edge_items_per_axis.append(None)
  array_edges, _ = truncate_array_and_mask(
      display_array,
      np.ones((1,) * display_array.ndim, dtype=jnp.bool_),
      edge_items_per_axis=tuple(edge_items_per_axis),
  )
  if is_prng_key:
    return (
        f"Array({array.shape}, dtype={array.dtype}) overlaying:\n"
        + np.array2string(
            np.array(array_edges),
            separator=" ",
            threshold=0,
            edgeitems=edgeitems,
        )
    )
  prefix = "Array("
  datastring = np.array2string(
      np.array(array_edges),
      prefix=prefix,
      suffix=",",
      separator=", ",
      threshold=0,
      edgeitems=edgeitems,
  )
  return f"{prefix}{datastring}, {dtype_str}"


def make_checked_dataclasslike_renderer(
    cls: type[Any],
    fields: Sequence[str],
    fields_with_none_default: Sequence[str] = (),
) -> renderers.TreescopeNodeHandler:
  """Builds a roundtrippable renderer for a dataclass-like class.

  This function can be used to safely render classes that behave like Python
  dataclasses (i.e. they can be roundtripped by calling the constructor with
  attributes as keyword arguments). It is robust to potential new attributes
  being added by checking that it is possible to rebuild the instance correctly.
  This can be ued to render JAX builtin classes.

  Args:
    cls: The class to render.
    fields: A sequence of attribute names to render as keyword args.
    fields_with_none_default: A sequence of attribute names to render as keyword
      args only if they exist and their value is not None.

  Returns:
    A node handler for nodes of this type, which returns a simple rendering
    whenever the object is correctly described by these attributes.
  """

  def render_it(
      node: Any,
      path: str | None,
      subtree_renderer: renderers.TreescopeSubtreeRenderer,
  ) -> (
      rendering_parts.RenderableTreePart
      | rendering_parts.RenderableAndLineAnnotations
      | type(NotImplemented)
  ):
    if type(node) is not cls:  # pylint: disable=unidiomatic-typecheck
      return NotImplemented
    try:
      attributes = {k: getattr(node, k) for k in fields}
    except AttributeError:
      return NotImplemented
    for k in fields_with_none_default:
      if hasattr(node, k) and getattr(node, k) is not None:
        attributes[k] = getattr(node, k)

    # Make sure we can correctly round-trip it.
    rebuilt = cls(**attributes)
    if rebuilt != node:
      return NotImplemented
    else:
      return repr_lib.render_object_constructor(
          object_type=cls,
          attributes=attributes,
          path=path,
          subtree_renderer=subtree_renderer,
          roundtrippable=True,
      )

  return render_it


def render_precision(
    node: jax.lax.Precision,
    path: str | None,
    subtree_renderer: renderers.TreescopeSubtreeRenderer,
) -> (
    rendering_parts.RenderableTreePart
    | rendering_parts.RenderableAndLineAnnotations
    | type(NotImplemented)
):
  """Renders jax.lax.Precision."""
  assert jax is not None, "JAX is not available."
  if type(node) is not jax.lax.Precision:  # pylint: disable=unidiomatic-typecheck
    return NotImplemented
  return repr_lib.render_enumlike_item(
      object_type=jax.lax.Precision,
      item_name=node.name,
      item_value=node.value,
      path=path,
      subtree_renderer=subtree_renderer,
  )


def _compute_summary(
    x: jax.Array, is_floating: bool, is_integer: bool, is_bool: bool, xnp=None
) -> dict[str, jax.Array]:
  """Computes a summary of the given array."""
  if xnp is None:
    assert jax is not None, "JAX is not available."
    xnp = jax.numpy
  x = xnp.array(x)
  result = {}
  if is_floating:
    isfinite = xnp.isfinite(x)
    inf_to_nan = xnp.where(isfinite, x, xnp.array(xnp.nan, dtype=x.dtype))
    nanmean = functools.partial(xnp.nanmean, dtype=xnp.float32)
    nanstd = functools.partial(xnp.nanstd, dtype=xnp.float32)
    nanmin = lambda x: xnp.nanmin(x).astype(xnp.float32)
    nanmax = lambda x: xnp.nanmax(x).astype(xnp.float32)
    result.update(mean=nanmean(inf_to_nan), std=nanstd(inf_to_nan))
    result.update(nanmin=nanmin(x), nanmax=nanmax(x))
    result.update(
        nan=xnp.count_nonzero(xnp.isnan(x)),
        inf=xnp.count_nonzero(xnp.isposinf(x)),
    )
    result["any_finite"] = xnp.any(isfinite)
    result["-inf"] = xnp.count_nonzero(xnp.isneginf(x))
  if is_integer:
    result.update(min=xnp.min(x), max=xnp.max(x))
  if is_floating or is_integer:
    result.update(zero=xnp.count_nonzero(x == 0), nonzero=xnp.count_nonzero(x))
  if is_bool:
    result.update(
        true=xnp.count_nonzero(x), false=xnp.count_nonzero(xnp.logical_not(x))
    )
  return result


def _summarize_array_data_unconditionally(array: jax.Array) -> list[str]:
  """Summarized the data of a JAX array."""
  assert jax is not None, "JAX is not available."
  jnp = jax.numpy
  output_parts = []
  # If treescope is called from inside a jitted function, this makes our own
  # computations run eagerly instead of being staged into the traced function.
  with jax.core.ensure_compile_time_eval():
    is_floating = _is_subdtype(array.dtype, jnp.floating)
    is_integer = _is_subdtype(array.dtype, jnp.integer)
    is_bool = _is_subdtype(array.dtype, jnp.bool_)
    if not (is_floating or is_integer or is_bool):
      # Non-numeric non-bool data type (perhaps JAX PRNG key dtype). Can't
      # summarize values.
      return []

    if array.size < SUMMARIZE_USING_NUMPY_THRESHOLD:
      stat = _compute_summary(array, is_floating, is_integer, is_bool, xnp=np)
    else:
      compute_summary = jax.jit(_compute_summary, static_argnums=(1, 2, 3))
      # Run under the array's own mesh. Explicitly-sharded arrays need a
      # matching mesh context here, and JAX rejects computations on an array
      # under a mesh context for a different mesh (e.g. a single-device array
      # while a global mesh is set).
      array, mesh = _as_mesh_array(array)
      with jax.set_mesh(mesh):
        stat = compute_summary(array, is_floating, is_integer, is_bool)
      # Get values in parallel.
      stat = jax.device_get(stat)

    # pylint: disable=inconsistent-quotes
    if is_floating and stat["any_finite"]:
      output_parts.append(f" ≈{stat['mean']:.2} ±{stat['std']:.2}")
      output_parts.append(f" [≥{stat['nanmin']:.2}, ≤{stat['nanmax']:.2}]")

    if is_integer:
      output_parts.append(f" [≥{stat['min']:_d}, ≤{stat['max']:_d}]")
    # pylint: enable=inconsistent-quotes

    def append_if_present(output_parts, *names):
      for name in names:
        if stat[name]:
          output_parts.append(f" {name}:{stat[name]:_d}")

    if is_floating or is_integer:
      append_if_present(output_parts, "zero", "nonzero")
    if is_floating:
      append_if_present(output_parts, "nan", "inf", "-inf")

    if is_bool:
      append_if_present(output_parts, "true", "false")
    return output_parts


def summarize_array_data(array: jax.Array) -> str:
  """Summarized the data of a JAX array.

  Args:
    array: The array to summarize.

  Returns:
    A string summarizing the data of the array.
  """

  output_parts = []

  if isinstance(array, jax.core.Tracer):
    output_parts.append(" - tracer.")
  elif array.is_deleted():
    output_parts.append(" - deleted!")
  elif not _is_locally_available(array):
    output_parts.append(" - multi-host array!")
  elif safe_to_summarize(array):
    output_parts.extend(_summarize_array_data_unconditionally(array))
  else:
    output_parts.append("- too large to summarize.")

  return "".join(output_parts)


class JAXArrayAdapter(ndarray_adapters.NDArrayAdapter[jax.Array]):
  """Array adapter for JAX arrays."""

  def get_axis_info_for_array_data(
      self, array: jax.Array
  ) -> tuple[ndarray_adapters.AxisInfo, ...]:
    assert jax is not None, "JAX is not available."
    return tuple(
        ndarray_adapters.PositionalAxisInfo(i, size)
        for i, size in enumerate(array.shape)
    )

  def get_array_data_with_truncation(
      self,
      array: jax.Array,
      mask: jax.Array | None,
      edge_items_per_axis: tuple[int | None, ...],
  ) -> tuple[np.ndarray, np.ndarray]:
    assert jax is not None, "JAX is not available."
    assert not isinstance(array, jax.core.Tracer)
    assert not array.is_deleted()
    if mask is None:
      mask = np.array(True)

    if edge_items_per_axis == (None,) * array.ndim:
      # No truncation.
      return np.array(array), np.broadcast_to(mask, array.shape)

    return truncate_array_and_mask(array, mask, edge_items_per_axis)

  def get_array_summary(
      self, array: jax.Array, fast: bool
  ) -> rendering_parts.RenderableTreePart:
    output_parts = [
        rendering_parts.abbreviatable(
            rendering_parts.text("jax.Array "),
            rendering_parts.text("jax "),
        )
    ]

    output_parts.append(dtype_util.get_dtype_name(array.dtype))
    output_parts.append(repr(array.shape))
    if array.is_deleted():
      output_parts.append(" - deleted!")
    elif not fast:
      output_parts.append(
          rendering_parts.abbreviatable(
              rendering_parts.text(summarize_array_data(array)),
              rendering_parts.empty_part(),
          )
      )

    return rendering_parts.siblings(*output_parts)

  def get_numpy_dtype(self, array: jax.Array) -> np.dtype | None:
    if isinstance(array.dtype, np.dtype):
      return array.dtype
    else:
      return None

  def get_sharding_info_for_array_data(
      self, array: jax.Array
  ) -> ndarray_adapters.ShardingInfo | None:
    assert jax is not None, "JAX is not available."
    if isinstance(array, jax.core.Tracer) or array.is_deleted():
      return None

    [platform] = set(device.platform for device in array.sharding.device_set)
    device_map = array.sharding.devices_indices_map(array.shape)
    return ndarray_adapters.ShardingInfo(
        shard_shape=array.sharding.shard_shape(array.shape),
        device_index_to_shard_slices={
            device.id: slices for device, slices in device_map.items()
        },
        device_type=platform.upper(),
        fully_replicated=array.is_fully_replicated,
    )

  def should_autovisualize(self, array: jax.Array) -> bool:
    assert jax is not None, "JAX is not available."
    return not isinstance(array, jax.core.Tracer) and not array.is_deleted()


def render_jax_arrays(
    node: jax.Array,
    path: str | None,
    subtree_renderer: renderers.TreescopeSubtreeRenderer,
) -> (
    rendering_parts.RenderableTreePart
    | rendering_parts.RenderableAndLineAnnotations
    | type(NotImplemented)
):
  """Renders a JAX array."""
  assert jax is not None, "JAX is not available."
  del subtree_renderer
  assert isinstance(node, jax.Array)
  if isinstance(node, jax.core.Tracer):
    return NotImplemented

  adapter = JAXArrayAdapter()

  if node.is_deleted():
    return rendering_parts.error_color(
        rendering_parts.siblings(
            rendering_parts.text("<"),
            adapter.get_array_summary(node, fast=True),
            rendering_parts.text(">"),
        )
    )

  def _placeholder() -> rendering_parts.RenderableTreePart:
    return rendering_parts.deferred_placeholder_style(
        rendering_parts.siblings(
            rendering_parts.text("<"),
            adapter.get_array_summary(node, fast=True),
            rendering_parts.text(">"),
        )
    )

  def _thunk(placeholder_expand_state: rendering_parts.ExpandState | None):
    # Is this array simple enough to render without a summary?
    node_repr = faster_array_repr(node)
    if "\n" not in node_repr and "..." not in node_repr:
      rendering = rendering_parts.abbreviation_color(
          rendering_parts.text(f"<jax.{node_repr}>")
      )
    else:
      if node_repr.count("\n") <= 15:
        if placeholder_expand_state is None:
          default_expand_state = rendering_parts.ExpandState.WEAKLY_EXPANDED
        else:
          default_expand_state = placeholder_expand_state
      else:
        # Always start big NDArrays in collapsed mode to hide irrelevant detail.
        default_expand_state = rendering_parts.ExpandState.COLLAPSED

      # Render it with a summary.
      summarized = adapter.get_array_summary(node, fast=False)
      rendering = rendering_parts.build_custom_foldable_tree_node(
          label=rendering_parts.abbreviation_color(
              rendering_parts.comment_color_when_expanded(
                  rendering_parts.siblings(
                      rendering_parts.fold_condition(
                          expanded=rendering_parts.text("# "),
                          collapsed=rendering_parts.text("<"),
                      ),
                      summarized,
                      rendering_parts.fold_condition(
                          collapsed=rendering_parts.text(">")
                      ),
                  )
              )
          ),
          contents=rendering_parts.fold_condition(
              expanded=rendering_parts.indented_children(
                  [rendering_parts.text(node_repr)]
              )
          ),
          path=path,
          expand_state=default_expand_state,
      ).renderable

    return rendering

  return rendering_parts.RenderableAndLineAnnotations(
      renderable=lowering.maybe_defer_rendering(
          main_thunk=_thunk, placeholder_thunk=_placeholder
      ),
      annotations=rendering_parts.build_copy_button(path),
  )


def set_up_treescope():
  """Sets up treescope to render JAX objects."""
  if jax is None:
    raise RuntimeError(
        "Cannot set up JAX support in treescope: JAX cannot be imported."
    )
  if jax.__version_info__ < MINIMUM_JAX_VERSION:
    minimum = ".".join(str(part) for part in MINIMUM_JAX_VERSION)
    warnings.warn(
        f"Treescope's JAX support requires JAX >= {minimum}, but JAX"
        f" {jax.__version__} is installed. JAX support will be disabled, so"
        " JAX values will be rendered like other Python objects."
    )
    return
  type_registries.TREESCOPE_HANDLER_REGISTRY[jax.ShapeDtypeStruct] = (
      make_checked_dataclasslike_renderer(
          jax.ShapeDtypeStruct,
          fields=("shape", "dtype"),
          fields_with_none_default=("sharding",),
      )
  )
  type_registries.TREESCOPE_HANDLER_REGISTRY[jax.tree_util.SequenceKey] = (
      make_checked_dataclasslike_renderer(
          jax.tree_util.SequenceKey, fields=("idx",)
      )
  )
  type_registries.TREESCOPE_HANDLER_REGISTRY[jax.tree_util.DictKey] = (
      make_checked_dataclasslike_renderer(
          jax.tree_util.DictKey, fields=("key",)
      )
  )
  type_registries.TREESCOPE_HANDLER_REGISTRY[jax.tree_util.GetAttrKey] = (
      make_checked_dataclasslike_renderer(
          jax.tree_util.GetAttrKey, fields=("name",)
      )
  )
  type_registries.TREESCOPE_HANDLER_REGISTRY[
      jax.tree_util.FlattenedIndexKey
  ] = make_checked_dataclasslike_renderer(
      jax.tree_util.FlattenedIndexKey, fields=("key",)
  )
  type_registries.TREESCOPE_HANDLER_REGISTRY[jax.lax.Precision] = (
      render_precision
  )

  # The concrete type of a JAX array is a private type that is dynamically
  # registered as a jax.Array subclass, so we need to add it to the list of
  # dynamically-checked virtual base classes.
  type_registries.VIRTUAL_BASE_CLASSES.append(jax.Array)
  type_registries.IMMUTABLE_TYPES_REGISTRY[jax.Array] = True
  type_registries.NDARRAY_ADAPTER_REGISTRY[jax.Array] = JAXArrayAdapter()
  type_registries.TREESCOPE_HANDLER_REGISTRY[jax.Array] = render_jax_arrays

  for jax_api_module in [
      jax.lax,
      jax.numpy,
      jax.scipy,
      jax.random,
      jax.nn,
      jax.custom_derivatives,
      jax,
  ]:
    canonical_aliases.populate_from_public_api(
        jax_api_module, canonical_aliases.prefix_filter("jax")
    )

  for key_cls_name in [
      "SequenceKey",
      "DictKey",
      "GetAttrKey",
      "FlattenedIndexKey",
  ]:
    canonical_aliases.add_alias(
        getattr(jax.tree_util, key_cls_name),
        canonical_aliases.ModuleAttributePath("jax.tree_util", (key_cls_name,)),
        on_conflict="ignore",
    )
