from typing import Any, Callable, Dict, Sequence, Union

import numpy as np
import jax.numpy as jnp
import flax
try:
    import tensorflow as tf
    _TfTensor = tf.Tensor
except ImportError:
    _TfTensor = Any


PRNGKey = Any
Params = flax.core.FrozenDict[str, Any]
Shape = Sequence[int]
Dtype = Any  # this could be a real type?
InfoDict = Dict[str, float]
Array = Union[np.ndarray, jnp.ndarray, _TfTensor]
Data = Union[Array, Dict[str, "Data"]]
Batch = Dict[str, Data]
# A method to be passed into TrainState.__call__
ModuleMethod = Union[str, Callable, None]
