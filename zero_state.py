import numpy as np
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util

def _hasattr(obj, attr_name):
  try:
    getattr(obj, attr_name)
  except AttributeError:
    return False
  else:
    return True

def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
    if inputs is not None:
      # Validate the given batch_size and dtype against inputs if provided.
      inputs = ops.convert_to_tensor(inputs, name="inputs")
      if batch_size is not None:
        if tensor_util.is_tensor(batch_size):
          static_batch_size = tensor_util.constant_value(batch_size, partial=True)
        else:
          static_batch_size = batch_size
        if inputs.shape.dims[0].value != static_batch_size:
          raise ValueError(
              "batch size from input tensor is different from the "
              "input param. Input tensor batch: {}, batch_size: {}".format(inputs.shape.dims[0].value, batch_size))

      if dtype is not None and inputs.dtype != dtype:
        raise ValueError(
            "dtype from input tensor is different from the "
            "input param. Input tensor dtype: {}, dtype: {}".format(inputs.dtype, dtype))

      batch_size = inputs.shape.dims[0].value or array_ops.shape(inputs)[0]
      dtype = inputs.dtype
    if batch_size is None or dtype is None:
      raise ValueError(
          "batch_size and dtype cannot be None while constructing initial "
          "state: batch_size={}, dtype={}".format(batch_size, dtype))
    return self.zero_state(batch_size, dtype)

def zero_state(self, batch_size, dtype):
    """Return zero-filled state tensor(s).
    Args:
      batch_size: int, float, or unit Tensor representing the batch size.
      dtype: the data type to use for the state.
    Returns:
      If `state_size` is an int or TensorShape, then the return value is a
      `N-D` tensor of shape `[batch_size, state_size]` filled with zeros.
      If `state_size` is a nested list or tuple, then the return value is
      a nested list or tuple (of the same structure) of `2-D` tensors with
      the shapes `[batch_size, s]` for each s in `state_size`.
    """
    # Try to use the last cached zero_state. This is done to avoid recreating
    # zeros, especially when eager execution is enabled.
    state_size = self.state_size
    is_eager = context.executing_eagerly()
    if is_eager and _hasattr(self, "_last_zero_state"):
      (last_state_size, last_batch_size, last_dtype,
       last_output) = getattr(self, "_last_zero_state")
      if (last_batch_size == batch_size and last_dtype == dtype and
          last_state_size == state_size):
        return last_output
    with ops.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
      output = _zero_state_tensors(state_size, batch_size, dtype)
    if is_eager:
      self._last_zero_state = (state_size, batch_size, dtype, output)
    return output
