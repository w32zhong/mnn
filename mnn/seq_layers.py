from mnn.tensor import Tensor
from mnn.layer import *


class SequentialLayers():
    r'''
    ## Backpropagation background

    Each layer $f$ in neural network is just a function mapping from
    $\mathbb{R}^m \rightarrow \mathbb{R}^n $.

    Without loss of generality, consider the scaler version $z(t) = f(x(t), y(t))$, we can show:

    $$
    \begin{aligned}
        z'(t) =& \lim_{dt \to 0} \frac{f(x(t+dt), y(t+dt)) - f(x(t), y(t))}{dt} \\\\
              =& \lim_{dt \to 0} \frac{  f(x(t+dt), y(t+dt)) - f(x(t+dt), y(t))  + f(x(t+dt), y(t)) - f(x(t), y(t))  }{dt} \\\\
              =& \lim_{dt \to 0} \frac{f(x(t+dt), y(t+dt)) - f(x(t+dt), y(t))}{dt}  + \lim_{dt \to 0} \frac{f(x(t+dt), y(t)) - f(x(t), y(t))}{dt} \\\\
              =& \lim_{dt \to 0} \frac{f(x(t+dt), y(t+dt)) - f(x(t+dt), y(t))}  {y(t+dt) - y(t)} \times  \frac{y(t+dt) - y(t)}{dt} + \\\\
              & \lim_{dt \to 0} \frac{f(x(t+dt), y(t)) - f(x(t), y(t))}  {x(x+dt) - x(t)} \times  \frac{x(x+dt) - x(t)}{dt} \\\\
              \doteq& \lim_{dt \to 0} \frac{f(x(t+dt), y(t) + \Delta y) - f(x(t+dt), y(t))}  {\Delta y} \times  \frac{y(t+dt) - y(t)}{dt} + \\\\
              & \lim_{dt \to 0} \frac{f(x(t) + \Delta x, y(t)) - f(x(t), y(t))}  {\Delta x} \times  \frac{x(x+dt) - x(t)}{dt} \\\\
              =& \left.\frac{\partial f}{\partial y}\right|_{y=y(t)} \cdot \frac{\partial y}{\partial t}
              + \left.\frac{\partial f}{\partial x}\right|_{x=x(t)} \cdot \frac{\partial x}{\partial t}
    \end{aligned}
    $$

    iff $dt \rightarrow 0$ implies $\Delta x \rightarrow 0$ and $\Delta y \rightarrow 0$ (Lipschitz continuity).

    In more general case when $z(t) = f(x(t))$ where $x \in \mathbb{R}^n, t \in \mathbb{R}^m, f: \mathbb{R}^n \rightarrow \mathbb{R}$ and $x: \mathbb{R}^m \rightarrow \mathbb{R}^n$,

    $$
    \begin{aligned}
    \frac{\partial z}{\partial t_i}
     =&
     \begin{bmatrix} \frac{\partial f}{\partial x_1} & ... & \frac{\partial f}{\partial x_n} \end{bmatrix}_{x = x(t)}
     \cdot
     \begin{bmatrix} \frac{\partial x_1}{\partial t_i} \\\\ \vdots \\\\ \frac{\partial x_n}{\partial t_i} \end{bmatrix} \\\\
     \doteq&
     \nabla_x^T f (x = x(t))
     \cdot
     \begin{bmatrix} \frac{\partial x_1}{\partial t_i} \\\\ \vdots \\\\ \frac{\partial x_n}{\partial t_i} \end{bmatrix} \\\\
    \end{aligned}
    $$

    therefore

    $$
    \tag{1}
    \nabla_t^T z(t) \doteq \begin{bmatrix} \frac{\partial f}{\partial t_1}, ..., \frac{\partial f}{\partial t_m} \end{bmatrix}
     =
     \nabla_x^T f (x = x(t))
     \cdot
     \begin{bmatrix}
        \partial x_1 / \partial t_1 & \partial x_1 / \partial t_2 & ... & \partial x_1 / \partial t_m \\\\
        \partial x_2 / \partial t_1 & \partial x_2 / \partial t_2 & ... & \partial x_2 / \partial t_m \\\\
        \vdots & \ddots \\\\
        \partial x_n / \partial t_1 & \partial x_n / \partial t_2 & ... & \partial x_n / \partial t_m \\\\
     \end{bmatrix}
    $$

    where the RHS matrix is called the Jacobian matrix $J_t x$.
    '''
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, inputs, targets=None, debug=False):
        v = inputs
        for layer in self.layers:
            if debug: print('forward', layer.name, v.shape, end=' => ')
            v = layer.forward(v, feedbacks=targets)
            if debug: print(v.shape, 'sum:', v.sum())
        return v

    def backward(self, debug=False):
        r'''
        ## Backpropagation

        As seen in Eq. (1), we can propagate gradient w.r.t. $t$ back
        from down-stream gradients using

        $$
            \nabla_t^T z(t) = \nabla_x^T f (x = x(t)) \cdot J_t x
        $$
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
        r'''
        ## Gradient descent

        At time $k$, to update the parameter $t$ to achieve lower $z$ value (loss):

        $$
        t^{(k + 1)} = t^{(k)} - \eta \cdot \nabla_t z
        $$
        '''
        for layer in self.layers:
            layer.step()

    def zero_grads(self):
        for layer in self.layers:
            layer._zero_grads()


if __name__ == '__main__':
    import sys
    B = 4 # batch size
    C = 5 # classes
    debug = False
    testcase = sys.argv[1] if len(sys.argv) > 1 else 1

    def testcase1():
        inputs = Tensor.randn(B, 32, 1)
        targets = Tensor.randn(B, 10)
        net = SequentialLayers([
            LinearLayer(32, 40),
            ReluLayer(),
            LinearLayer(40, 10, bias=False),
            ReluLayer(),
            MSELossLayer()
        ])
        return inputs, targets, net

    def testcase2():
        inputs = Tensor.randn(B, 32, 1)
        targets = Tensor.randint(shape=(B, 1), high=C)
        net = SequentialLayers([
            LinearLayer(32, 40),
            ReluLayer(),
            LinearLayer(40, C, bias=False),
            CrossEntropyLossLayer()
        ])
        return inputs, targets, net

    def testcase3():
        inputs = Tensor.randn(B, 32, 1)
        targets = Tensor.randint(shape=(B, 1), high=C)
        net = SequentialLayers([
            LinearLayer(32, 40),
            ReluLayer(),
            LinearLayer(40, C, bias=False),
            LogSoftmaxLayer(),
            NllLossLayer()
        ])
        return inputs, targets, net

    def testcase4():
        inputs = Tensor.randn(B, 32, 1)
        targets = Tensor.randint(shape=(B, 1), high=C)
        net = SequentialLayers([
            LinearLayer(32, 40),
            ReluLayer(),
            LinearLayer(40, C, bias=False),
            SoftmaxLayer(),
            LogLayer(),
            NllLossLayer()
        ])
        return inputs, targets, net

    inputs, targets, net = globals()['testcase'+ str(testcase)]()

    for ep in range(1 if debug else 20):
        loss = net(inputs, targets, debug=debug)
        print(f'epoch#{ep + 1}', loss)
        net.zero_grads()
        gradients = net.backward(debug=debug)
        if debug:
            for layer, grads in zip(net.layers, gradients):
                print(layer.name, 'gradients sum:', grads.sum())
        net.step()
