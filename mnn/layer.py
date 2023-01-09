from .tensor import Tensor


class BaseLayer():
    def __init__(self):
        self.name = self.__class__.__name__
        self.params = {}
        self._zero_grads()

    def _zero_grads(self):
        self.grads = {}

    def _accumulate_grads(self, key, val):
        reduced_val = self.batch_reduced(val)
        if key in self.grads:
            self.grads[key] += reduced_val
        else:
            self.grads[key] = reduced_val

    def step(self, lr=0.001):
        for key, grads in self.grads.items():
            assert grads.shape[1:] == self.params[key].shape[1:]
            self.params[key] -= lr * grads

    def batch_reduced(self, val):
        batch_size = val.shape[0]
        return val.sum(axis=0, keepdims=True) / batch_size


class LinearLayer(BaseLayer):
    def __init__(self, *shape, bias=True):
        r'''
        An linear layer to compute $y_{m \times 1} = W_{m \times n} x_{n \times 1} + b_{m \times 1}$,
        where

        $$
        W = \begin{bmatrix}
        w_{1,1} & w_{1, 2} & ... & w_{1, n} \\\\
        w_{2,1} & w_{2, 2} & ... & w_{2, n} \\\\
        \vdots \\\\
        w_{m,1} & w_{m, 2} & ... & w_{m, n} \\\\
        \end{bmatrix}
        $$
        '''
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
        self._accumulate_grads('w', grads_w)

        if self.bias:
            grads_b = gradients
            self._accumulate_grads('b', grads_b)

        jacob_x = self.params['w']
        grads_x = jacob_x.T @ gradients
        return grads_x


class ReluLayer(BaseLayer):
    def forward(self, inputs, feedbacks=None):
        self.last_inputs = inputs
        return Tensor.maximum(inputs, 0.0)

    def backward(self, gradients):
        flat_jacob = Tensor.ones_like(self.last_inputs)
        flat_jacob[self.last_inputs < 0] = 0.0
        return gradients * flat_jacob


class MSELossLayer(BaseLayer):
    def forward(self, inputs, feedbacks=None):
        inputs = inputs.squeeze(-1)
        self.last_error = inputs - feedbacks
        batch_size = inputs.shape[0]
        batch_loss = ((inputs - feedbacks) ** 2).sum(axis=1)
        return self.batch_reduced(batch_loss)

    def backward(self):
        gradients = 2 * self.last_error
        return gradients.unsqueeze(axis=-1)


if __name__ == '__main__':
    B = 12
    inputs = Tensor.randn(B, 3, 1)

    linear_layer = LinearLayer(3, 2)
    outputs = linear_layer.forward(inputs)
    print(outputs.shape)
    gradients = linear_layer.backward(Tensor.randn(B, 2, 1))
    print(gradients.shape)

    relu_layer = ReluLayer()
    outputs = relu_layer.forward(inputs)
    print(outputs.shape)
    gradients = relu_layer.backward(Tensor.randn(B, 3, 1))
    print(gradients.shape)

    loss_layer = MSELossLayer()
    outputs = loss_layer.forward(inputs, feedbacks=Tensor.randn(B, 3))
    print(outputs)
    gradients = loss_layer.backward()
    print(gradients.shape)
