from .tensor import Tensor
from .layer import *


class SequentialLayers():
    r'''
    ## Background knowledge

    Each layer $f$ in neural network is just a function mapping from
    $f: \mathbb{R}^m \rightarrow \mathbb{R}^n $.

    Without loss of generality, suppose $z(t) = f(x(t), y(t))$, we can show:

    $$
    \begin{aligned}
    z'(t) &= \lim_{dt \to 0} \frac{f(x(t+dt), y(t+dt)) - f(x(t), y(t))}{dt} \\\\
          &= \lim_{dt \to 0} \frac{  f(x(t+dt), y(t+dt)) - f(x(t+dt), y(t))  + f(x(t+dt), y(t)) - f(x(t), y(t))  }{dt} \\\\
          &= \lim_{dt \to 0} \frac{f(x(t+dt), y(t+dt)) - f(x(t+dt), y(t))}{dt}  + \lim_{dt \to 0} \frac{f(x(t+dt), y(t)) - f(x(t), y(t))}{dt} \\\\
          &= \lim_{dt \to 0} \frac{f(x(t+dt), y(t+dt)) - f(x(t+dt), y(t))}  {y(t+dt) - y(t)} \times  \frac{y(t+dt) - y(t)}{dt} \\\\  &+ \lim_{dt \to 0} \frac{f(x(t+dt), y(t)) - f(x(t), y(t))}  {x(x+dt) - x(t)} \times  \frac{x(x+dt) - x(t)}{dt} \\\\
          &\doteq \lim_{dt \to 0} \frac{f(x(t+dt), y(t) + \Delta y) - f(x(t+dt), y(t))}  {\Delta y} \times  \frac{y(t+dt) - y(t)}{dt} \\\\  &+ \lim_{dt \to 0} \frac{f(x(t) + \Delta x, y(t)) - f(x(t), y(t))}  {\Delta x} \times  \frac{x(x+dt) - x(t)}{dt} \\\\
          &= \frac{\partial z}{\partial y} \times \frac{\partial y}{\partial t}  + \frac{\partial z}{\partial x} \times \frac{\partial x}{\partial t} \end{aligned}
    $$

    '''
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
        r'''
        ## Gradients w.r.t. $W$
        '''
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
