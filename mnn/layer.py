from tensor import Tensor


class BaseLayer():
    def __init__(self):
        self.name = self.__class__.__name__
        self.params = {}
        self.zero_grads()

    def zero_grads(self):
        self.grads = {}

    def accumulate_grads(self, key, val):
        reduced_val = self.batch_reduced_val(val)
        if key in self.grads:
            self.grads[key] += reduced_val
        else:
            self.grads[key] = reduced_val

    def step(self, lr=0.001):
        for key, grads in self.grads.items():
            assert grads.shape[1:] == self.params[key].shape[1:]
            self.params[key] -= lr * grads

    def batch_reduced_val(self, val):
        batch_size = val.shape[0]
        return val.sum(axis=0, keepdims=True) / batch_size


class LinearLayer(BaseLayer):
    def __init__(self, *shape, bias=True):
        super().__init__()
        n, m = shape
        self.params['w'] = Tensor.randn(1, m, n)
        self.bias = bias
        if self.bias:
            self.params['b'] = Tensor.randn(1, m, 1)

    def forward(self, inputs, feedbacks=None):
        self.last_inputs = inputs
        if self.bias:
            return self.params['w'] @ inputs + self.params['b']
        else:
            return self.params['w'] @ inputs

    def backward(self, gradients):
        grads_w = gradients @ self.last_inputs.T
        self.accumulate_grads('w', grads_w)

        grads_b = gradients
        self.accumulate_grads('b', grads_b)

        jacob_x = self.params['w']
        grads_x = jacob_x.T @ gradients
        return grads_x


if __name__ == '__main__':
    B = 12
    inputs = Tensor.randn(B, 3, 1)
    linear_layer = LinearLayer(3, 2)
    outputs = linear_layer.forward(inputs)
    print(outputs.shape)
