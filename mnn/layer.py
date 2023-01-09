from .tensor import Tensor


class BaseLayer():
    def __init__(self):
        self.name = self.__class__.__name__
        self.params = {}
        self._zero_grads()

    def _zero_grads(self):
        self.grads = {}

    def _accumulate_grads(self, key, val):
        reduced_val = self._batch_reduced(val)
        if key in self.grads:
            self.grads[key] += reduced_val
        else:
            self.grads[key] = reduced_val

    def step(self, lr=0.001):
        for key, grads in self.grads.items():
            assert grads.shape[1:] == self.params[key].shape[1:]
            self.params[key] -= lr * grads

    def _batch_reduced(self, val):
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
        self.last_inputs = inputs
        if self.bias:
            return self.params['w'] @ inputs + self.params['b']
        else:
            return self.params['w'] @ inputs

    def backward(self, gradients):
        r'''
        ## Gradients w.r.t. $W$

        Because $y_i = w_{i,1} x_1 + w_{i,2} x_2 + ... + w_{i,n} x_n + b_i$,
        we have
        $\partial y_i / \partial w_{i,j} = x_j$ and this derivative is $0$
        for $w_{k,j}$ when $k \not= i$.

        When we "flatten" the $W$ into a "long vector" $w$,
        the Jacobian w.r.t. $w$ then becomes:

        $$
        J_w = \begin{bmatrix}
        x_1 & x_2 & ... & x_n &   0 &   0 & ... &   0 &  0  & 0   & 0   \\\\
        0   & 0   & ... & 0   & x_1 & x_2 & ... &   0 &  0  & 0   & 0   \\\\
        \vdots \\\\
        0   & 0   & ... & 0   & 0   & 0   & ... & x_1 & x_2 & ... & x_n \\\\
        \end{bmatrix}_{m \times (mn)}
        $$

        If we chain the gradient product
        (assuming the final loss is scaler $\ell$):

        $$
        \nabla^T_w \ell = \nabla^T_y \ell \times J_w =
        \begin{bmatrix}
            x_1 \frac{\partial \ell}{\partial y_1} & x_2 \frac{\partial \ell}{\partial y_1} & ... & x_n \frac{\partial \ell}{\partial y_1} &
            x_1 \frac{\partial \ell}{\partial y_2} & x_2 \frac{\partial \ell}{\partial y_2} & ... & x_n \frac{\partial \ell}{\partial y_2} &
            ...
        \end{bmatrix}
        $$

        As apparently it is a recycling patten, we can "unroll"
        the Jacobian to a matrix so that it matches the dimension of $W$:

        $$
        \nabla_W \ell =
        \begin{bmatrix}
            x_1 \frac{\partial \ell}{\partial y_1} & x_2 \frac{\partial \ell}{\partial y_1} & ... & x_n \frac{\partial \ell}{\partial y_1} \\
            x_1 \frac{\partial \ell}{\partial y_2} & x_2 \frac{\partial \ell}{\partial y_2} & ... & x_n \frac{\partial \ell}{\partial y_2} \\
            \vdots
        \end{bmatrix}
        $$

        thus it can be written in

        $$
        \tag{1}
        \nabla_W \ell = (\nabla_y \ell)_{m \times 1} \times (x^T)_{1 \times n}
        $$

        ## Gradients w.r.t. $b$

        Because $y_i = w_{i,1} x_1 + w_{i,2} x_2 + ... + w_{i,n} x_n + b_i$,
        the Jacobian w.r.t. $b$ is an identity matrix, and the gradients of
        it is just the down-stream gradients:

        $$
        \tag{2}
        \nabla^T_b \ell = \nabla^T_y \ell \times J_b =
        \nabla^T_y \ell \times E = \nabla^T_y \ell
        $$

        ## Gradients w.r.t. inputs $x$

        This is the gradients to be back propagated to upstream,
        instead of being used to update the parameters of this layer.

        The Jacobian w.r.t. $w$ is, according to $y_i = w_{i,1} x_1 + w_{i,2} x_2 + ... + w_{i,n} x_n + b_i$,

        $$
        J_x = \begin{bmatrix}
        \frac{\partial y_1}{\partial x_1} & \frac{\partial y_1}{\partial x_2} & ... & \frac{\partial y_1}{\partial x_n} \\\\
        \frac{\partial y_2}{\partial x_1} & \frac{\partial y_2}{\partial x_2} & ... & \frac{\partial y_2}{\partial x_n} \\\\
        \vdots & \ddots \\\\
        \frac{\partial y_m}{\partial x_1} & \frac{\partial y_m}{\partial x_2} & ... & \frac{\partial y_m}{\partial x_n} \\\\
        \end{bmatrix}
        =
        \begin{bmatrix}
        w_{1,1} & w_{1, 2} & ... & w_{1, n} \\\\
        w_{2,1} & w_{2, 2} & ... & w_{2, n} \\\\
        \vdots \\\\
        w_{m,1} & w_{m, 2} & ... & w_{m, n} \\\\
        \end{bmatrix}
        =
        W
        $$

        as a result,

        $$
        \tag{3}
        \nabla_x \ell = J_x^T \times \nabla_y \ell
                      =  W^T \times \nabla_y \ell
        $$
        '''
        grads_w = gradients @ self.last_inputs.T # Eq. (1)
        self._accumulate_grads('w', grads_w)

        if self.bias:
            grads_b = gradients # Eq. (2)
            self._accumulate_grads('b', grads_b)

        jacob_x = self.params['w']
        grads_x = jacob_x.T @ gradients # Eq. (3)
        return grads_x


class ReluLayer(BaseLayer):
    def forward(self, inputs, feedbacks=None):
        r'''
        Relu activation function:

        $$
        f_i(x) = \left \\{
        \begin{aligned}
        x && (x \ge 0) \\\\
        0 && (\text{otherwise})
        \end{aligned}
        \right.
        $$
        '''
        self.last_inputs = inputs
        return Tensor.maximum(inputs, 0.0)

    def backward(self, gradients):
        r'''
        $$
        \begin{aligned}
        \nabla_x \ell =& \nabla_f \ell \cdot
        \begin{bmatrix}
        f_1'(x_1) & 0 & ... & 0 \\\\
        0 & f_2'(x_2) & ... & 0 \\\\
        \ddots \\\\
        0 & 0 & ... & f_n'(x_n)
        \end{bmatrix} \\\\
        =& \nabla_f \ell \odot
        \begin{bmatrix}
        f_1'(x_1) & f_2'(x_2) & ... & f_n'(x_n)
        \end{bmatrix}^T \\\\
        =& \nabla_f \ell \odot \nabla_x f
        \end{aligned}
        $$
        '''
        flat_jacob = Tensor.ones_like(self.last_inputs)
        flat_jacob[self.last_inputs < 0] = 0.0
        return gradients * flat_jacob


class MSELossLayer(BaseLayer):
    def forward(self, inputs, feedbacks=None):
        r'''
        Loss $\ell_i = \sum_{i} (x_i - y_i)^2$
        '''
        inputs = inputs.squeeze(-1)
        self.last_error = inputs - feedbacks
        batch_size = inputs.shape[0]
        batch_loss = ((inputs - feedbacks) ** 2).sum(axis=1)
        return self._batch_reduced(batch_loss)

    def backward(self):
        r'''
        Gradient $\nabla_x \ell = 2(x - y)$
        '''
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
