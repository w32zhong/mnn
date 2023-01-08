import cupy as cp


class Tensor():
    def __init__(self, data):
        if type(data) in (tuple, list):
            self._data = cp.array(data)
        else:
            self._data = data

    def __repr__(self):
        return self._data.__repr__()

    def __getattr__(self, name):
        if name == 'T':
            t = cp.moveaxis(self._data, -1, -2)
            return Tensor(t)
        attr = getattr(self._data, name)
        def wrapped_method(*args, **kwargs):
            result = attr(*args, **kwargs)
            if isinstance(result, cp.core.core.ndarray):
                return Tensor(result)
            else:
                return res
            return result
        if callable(attr):
            return wrapped_method
        else:
            return attr

    def __getitem__(self, item):
        return Tensor(self._data.__getitem__(item))

    def __setitem__(self, key, val):
        self._data.__setitem__(key, val)

    def __neg__(self, x):
        return Tensor(-x._data)

    def __add__(self, x):
        return Tensor(self._data + x._data)

    def __sub__(self, x):
        return Tensor(self._data - x._data)

    def __mul__(self, x):
        return Tensor(self._data * x._data)

    def __matmul__(self, x):
        return Tensor(self._data @ x._data)

    @staticmethod
    def zeros(shape):
        return Tensor(cp.zeros(shape))

    @staticmethod
    def randn(*shape):
        return Tensor(cp.random.randn(*shape))

    def diag_embed(self):
        last_dim = self.shape[-1]
        if last_dim == 1:
            tensor = self.squeeze(-1)
            last_dim = tensor.shape[-1]
        else:
            tensor = self
        diag = cp.broadcast_to(
            cp.expand_dims(tensor._data, axis=-1),
            (*tensor.shape, last_dim)
        ) * cp.eye(last_dim, last_dim)
        return Tensor(diag)


if __name__ == '__main__':
    d = Tensor.randn(1, 3, 1)
    s = d.squeeze()
    print(s.shape)
    d = d.diag_embed()
    d[0][2][0] = 0.123
    print(d.T)
    s = d.sum(axis=1)
    print(s)
    print(s + Tensor([[1.0, 2.0, 3.0]]))
