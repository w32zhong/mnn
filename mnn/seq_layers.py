from .tensor import Tensor
from .layer import *


class SequentialLayers():
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, inputs, targets=None, debug=False):
        v = inputs
        for layer in self.layers:
            if debug: print('forward', layer.name, v.shape, end=' => ')
            v = layer.forward(v, feedbacks=targets)
            if debug: print(v.shape)
        return v

    def backward(self, debug=False):
        gradients = []
        for layer in reversed(self.layers):
            if len(gradients) == 0:
                tmp = layer.backward()
            else:
                tmp = layer.backward(gradients[-1])
            gradients.append(tmp)
            if debug: print('backward', layer.name, ' => ', tmp.shape)
        return gradients

    def step(self):
        for layer in self.layers:
            layer.step()


if __name__ == '__main__':
    B = 4
    inputs = Tensor.randn(B, 32, 1)
    targets = Tensor.randn(B, 10)

    net = SequentialLayers([
        LinearLayer(32, 40),
        ReluLayer(),
        LinearLayer(40, 10, bias=False),
        ReluLayer(),
        MSELossLayer()
    ])

    debug = False
    for ep in range(1 if debug else 10):
        loss = net(inputs, targets, debug=debug)
        print(f'epoch#{ep + 1}', loss)
        net.backward(debug=debug)
        net.step()
