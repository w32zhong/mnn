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
                return result
            return result
        if callable(attr):
            return wrapped_method
        else:
            return attr

    def __getitem__(self, item):
        if isinstance(item, tuple):
            item = tuple(map(lambda x: x._data, item))
        return Tensor(self._data.__getitem__(item))

    def __setitem__(self, key, val):
        if isinstance(key, Tensor):
            self._data.__setitem__(key._data, val)
        elif isinstance(key, tuple):
            key = tuple(map(lambda x: x._data, key))
            self._data.__setitem__(key, val)
        else:
            self._data.__setitem__(key, val)

    def __neg__(self):
        return Tensor(- self._data)

    def __add__(self, x):
        if isinstance(x, Tensor):
            return Tensor(self._data + x._data)
        else:
            return Tensor(self._data + x)

    def __sub__(self, x):
        if isinstance(x, Tensor):
            return Tensor(self._data - x._data)
        else:
            return Tensor(self._data - x)

    def __mul__(self, x):
        if isinstance(x, Tensor):
            return Tensor(self._data * x._data)
        else:
            return Tensor(self._data * x)

    def __eq__(self, x):
        if isinstance(x, Tensor):
            return Tensor(self._data == x._data)
        else:
            raise NotImplemented

    def __rmul__(self, x):
        if isinstance(x, Tensor):
            return Tensor(self._data * x._data)
        else:
            return Tensor(self._data * x)

    def __matmul__(self, x):
        if isinstance(x, Tensor):
            return Tensor(self._data @ x._data)
        else:
            return Tensor(self._data @ x)

    def __truediv__(self, x):
        return Tensor(self._data / x)

    def __rtruediv__(self, x):
        return Tensor(x / self._data)

    def __pow__(self, x):
        return Tensor(self._data ** x)

    def __lt__(self, x):
        if isinstance(x, Tensor):
            return Tensor(self._data < x._data)
        else:
            return Tensor(self._data < x)

    def __le__(self, x):
        if isinstance(x, Tensor):
            return Tensor(self._data <= x._data)
        else:
            return Tensor(self._data <= x)

    def __gt__(self, x):
        if isinstance(x, Tensor):
            return Tensor(self._data > x._data)
        else:
            return Tensor(self._data > x)

    def __ge__(self, x):
        if isinstance(x, Tensor):
            return Tensor(self._data >= x._data)
        else:
            return Tensor(self._data >= x)

    @staticmethod
    def zeros(shape):
        return Tensor(cp.zeros(shape))

    @staticmethod
    def randint(shape, high, low=0):
        I = cp.random.randint(low, high=high, size=shape)
        return Tensor(I)

    @staticmethod
    def randn(*shape):
        return Tensor(cp.random.randn(*shape))

    @staticmethod
    def ones_like(tensor):
        return Tensor(cp.ones_like(tensor._data))

    @staticmethod
    def eye(*args, **kwargs):
        return Tensor(cp.eye(*args, **kwargs))

    @staticmethod
    def maximum(*args, **kwargs):
        return Tensor(cp.maximum(*args, **kwargs))

    @staticmethod
    def exp(*args, **kwargs):
        return Tensor(cp.exp(*args, **kwargs))

    @staticmethod
    def log(*args, **kwargs):
        return Tensor(cp.log(*args, **kwargs))

    @staticmethod
    def arange(*args, **kwargs):
        return Tensor(cp.arange(*args, **kwargs))

    def stacked(self, height=None):
        width = self.shape[-1]
        if height is None:
            height = width
        stacked = cp.broadcast_to(
            self._data,
            (self.shape[0], height, width)
        )
        return Tensor(stacked)

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

    def unsqueeze(self, *args, **kwargs):
        return Tensor(cp.expand_dims(self._data, *args, **kwargs))

    def pr(self, quit_=False):
        import inspect
        l = inspect.stack()[1].frame.f_locals
        g = globals()
        for k, v in [*l.items(), *g.items(), ('self', self)]:
            if id(v) == id(self):
                print(f'\033[93m pr() \033[0m: {k} shape:', v.shape)
                break
        else:
            raise KeyError
        if quit_: quit()
        return self


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
    s.pr()
