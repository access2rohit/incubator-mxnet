import numpy as np
import mxnet as mx
from mxnet import ndarray as nd

ctx=mx.cpu()

def create_2d_tensor(rows, columns):
  a = np.arange(0, columns).reshape(1, columns)
#   a = np.arange(0, columns)
  #b = np.broadcast_to(a, shape=(rows, columns))
  return nd.array(a, dtype=np.int64)

b = create_2d_tensor(rows=1, columns=5000000000)
print(b.shape)
