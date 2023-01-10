from mnn.tensor import Tensor


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

    def step(self, lr=0.01):
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
            x_1 \frac{\partial \ell}{\partial y_1} & x_2 \frac{\partial \ell}{\partial y_1} & ... & x_n \frac{\partial \ell}{\partial y_1} \\\\
            x_1 \frac{\partial \ell}{\partial y_2} & x_2 \frac{\partial \ell}{\partial y_2} & ... & x_n \frac{\partial \ell}{\partial y_2} \\\\
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


class SoftmaxLayer(BaseLayer):
    def __init__(self, *shape, axis=1):
        super().__init__()
        self.axis = axis

    @staticmethod
    def stable_softmax(inputs, axis):
        r'''
        ## Numerical-stable softmax

        Since some exponentials are quite large numbers, we can use

        $$
        y_i(x) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}
        = \frac{\exp(x_i - m)}{\sum_j \exp(x_j - m)}
        $$

        where $m = \max(x_1, x_2, ...)$.
        '''
        inputs_max = inputs.max(axis=axis, keepdims=True)
        stable_exps = Tensor.exp(inputs - inputs_max)
        sum_exps = stable_exps.sum(axis=axis, keepdims=True)
        return stable_exps / sum_exps

    def forward(self, inputs, feedbacks=None):
        self.saved_forward = SoftmaxLayer.stable_softmax(inputs, self.axis)
        return self.saved_forward

    def backward(self, gradients):
        r'''
        When $i = k$,

        $$
        \begin{aligned}
        \partial y_i / \partial x_k
        =& \frac{\exp(x_i)[\sum_j \exp(x_j)] - \exp(x_i)\exp(x_i)}{[\sum_j \exp(x_j)]^2} \\\\
        =& \frac{\exp(x_i) \left([\sum_j \exp(x_j)] -\exp(x_i)\right)}{[\sum_j \exp(x_j)]^2} \\\\
        =& \frac{\exp(x_i)}{\sum_j \exp(x_j)} \cdot
           \frac{[\sum_j \exp(x_j)] - \exp(x_i)}{\sum_j \exp(x_j)} \\\\
        =& y_i \cdot (1 - y_i) \\\\
        =& y_i - y_i y_k
        \end{aligned}
        $$

        When $i \not= k$,

        $$
        \begin{aligned}
        \partial y_i / \partial x_k
        =& \frac{0 - \exp(x_i)\exp(x_k)}{[\sum_j \exp(x_j)]^2} \\\\
        =& - y_i y_k
        \end{aligned}
        $$

        As a result, the Jacobian matrix

        $$
        J_x y =
        \begin{bmatrix}
            y_1 - y_1 y_1 & - y_1 y_2 & ... &  - y_1 y_n \\\\
            y_2 y_1 & y_2 - y_1 y_2 & ... &  - y_1 y_n \\\\
            \vdots & \ddots \\\\
            y_n y_1 & - y_n y_2 & ... & y_n - y_n y_n
        \end{bmatrix}
        = \operatorname{diag}(y) - y \times y^T
        $$
        '''
        softmax = self.saved_forward
        # diagonalize the last dimension
        diag = softmax.diag_embed()
        # generate softmax_{i,j} symmetric matrix
        symm = softmax @ softmax.T
        # compute Jacobian matrix wrt. inputs x
        jacob_x = diag - symm # B x n x n
        return jacob_x @ gradients


class LogLayer(BaseLayer):
    def forward(self, inputs, feedbacks=None):
        r"""
        $$
        y = \log(x)
        $$
        """
        self.last_inputs = inputs
        return Tensor.log(inputs)

    def backward(self, gradients):
        r"""
        Because this layer is element-wise operation, i.e.,
        $J_x y = \operatorname{diag}(x^{-1})$,
        the final gradient can be also simplified into Hadamard product:

        $$
        \nabla_x \ell = J^T_x y \cdot \nabla_y \ell = x^{-1} \odot \nabla_y \ell
        $$
        """
        reciprocal = 1 / self.last_inputs
        return reciprocal * gradients


class LogSoftmaxLayer(BaseLayer):
    def __init__(self, *shape, axis=1):
        super().__init__()
        self.axis = axis

    @staticmethod
    def stable_log_softmax(inputs, axis):
        r"""
        Simplify:

        $$
        \begin{aligned}
        z_i(x) =& \log \frac{\exp(x_i - m)}{\sum_j \exp(x_j - m)} \\\\
               =& x_i - m - \log(\sum_j \exp(x_j - m)) \\\\
        \end{aligned}
        $$
        """
        inputs_max = inputs.max(axis=axis, keepdims=True)
        inputs_off = inputs - inputs_max

        stable_exps = Tensor.exp(inputs_off)
        sum_exps = stable_exps.sum(axis=axis, keepdims=True)

        softmax = stable_exps / sum_exps
        log_softmax = inputs_off - Tensor.log(sum_exps)

        return log_softmax, softmax

    def forward(self, inputs, feedbacks=None):
        r"""
        $$
        z(x) = \log y(x)
        $$

        where $y_i(x) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}$
        """
        log_softmax, softmax = LogSoftmaxLayer.stable_log_softmax(
            inputs, self.axis
        )
        self.saved_forward = softmax
        return log_softmax

    def backward(self, gradients):
        r'''
        For softmax function, when $i = k$, we have
        $\partial y_i / \partial x_k = y_i \cdot (1 - y_i)$
        and when $i \not= k$, we have
        $\partial y_i / \partial x_k = - y_i y_k$

        Therefore, for the log-softmax function, when $i = k$, we have

        $$
        \partial z_i / \partial x_k = y_i^{-1} \cdot y_i \cdot (1 - y_i)
        = 1 - y_k
        $$

        and when $i \not= k$,

        $$
        \partial z_i / \partial x_k = y_i^{-1} \cdot (-y_i y_k) = - y_k
        $$

        As a result, the Jacobian matrix

        $$
        J_x z =
        \begin{bmatrix}
            1 - y_1 & - y_2 & ... &  - y_n \\\\
            - y_1 & 1 - y_2 & ... &  - y_n \\\\
            \vdots & \ddots \\\\
            - y_1 & - y_2 & ... & 1 - y_n
        \end{bmatrix}
        $$

        which can be seen as an identity matrix minus a "stacked" softmax vectors.

        The gradient is, by definition,

        $$
        \nabla_x \ell = J^T_x z \cdot \nabla_z \ell
        $$
        '''
        softmax = self.saved_forward
        softmax_T = softmax.T
        softmax_dim = softmax_T.shape[-1]
        stacked = softmax_T.stacked()
        # compute Jacobian matrix wrt. inputs x
        identity = Tensor.eye(softmax_dim, softmax_dim).unsqueeze(0)
        jacob_x = identity - stacked
        return jacob_x.T @ gradients


class NllLossLayer(BaseLayer):
    r'''
    This is to simulate PyTorch NLL layer which computes a negative expectation loss.
    Labels are passed in as integer indices.
    '''
    def forward(self, inputs, feedbacks=None):
        r'''
        $$
        \ell(q) = -\sum^n_{i=0} p_i q_i = -q_l
        $$

        where $p \in \{0, 1\}$ indicates true probability,
        but in actual implementation,
        the label is given as an index $l \in \mathbb{N}$.
        '''
        batch_size = inputs.shape[0]
        inputs = inputs.squeeze(-1)
        indices = feedbacks.squeeze(-1)
        self.saved_context = (inputs, indices, batch_size)

        neg_likelihood = - inputs[Tensor.arange(batch_size), indices]
        return self._batch_reduced(neg_likelihood)

    def backward(self):
        r'''
        In this case, the gradient vector w.r.t. $q$ is simply
        an almost-zero vector $v$ where $v_l = -1$.
        '''
        inputs, indices, batch_size = self.saved_context
        jacob_q = Tensor.zeros(inputs.shape)
        jacob_q[Tensor.arange(batch_size), indices] = -1.0
        return jacob_q.unsqueeze(-1)


class CrossEntropyLossLayer(BaseLayer):
    r'''
    This is to simulate PyTorch soft entropy layer which
    includes a softmax layer at the first layer.
    Labels are passed in as integer indices.
    '''
    def __init__(self, *shape, axis=1):
        super().__init__()
        self.axis = axis

    def forward(self, inputs, feedbacks=None):
        r"""
        $$
        \ell = -\sum_i p_i \log y_i(x) = -\log y_l(x)
        $$

        where $y_i(x) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}$
        and $p_i$ are real probabilities.
        In real implementation, label is specified as index $l$.
        """
        batch_size = inputs.shape[0]
        indices = feedbacks.squeeze(-1)

        log_softmax, softmax = LogSoftmaxLayer.stable_log_softmax(
            inputs, self.axis
        )
        use_log_softmax = log_softmax[Tensor.arange(batch_size), indices]
        cross_entropy = - use_log_softmax.squeeze(-1)

        self.saved_context = (batch_size, indices, softmax)
        return self._batch_reduced(cross_entropy)

    def backward(self):
        r'''
        For the log-softmax function $z(x)$,
        when $i = k$, we have
        $\partial z_i / \partial x_k = 1 - y_k$.
        And when $i \not= k$, we have
        $\partial z_i / \partial x_k = - y_k$.

        (recall that $y(x)$ is the softmax function)

        Therefore, in the cross entropy case,

        $$
        \begin{aligned}
         \partial \ell / \partial x_k
        =& - \frac{\partial}{\partial x_k} \left[ \sum_i p_i z_i(x) \right] \\\\
        =& - \left[ p_k \cdot (1 - y_k) - \sum_{j \not= k} p_j y_k \right] \\\\
        =& - p_k + p_k y_k + \sum_{j \not= k} p_j y_k \\\\
        =& - p_k + \sum_j p_j y_k \\\\
        =& - p_k + y_k \\\\
        \end{aligned}
        $$

        Assume the specified label is $l$, we will have the gradient vector
        (also assuming $p_l = 1$)

        $$
        \nabla_x \ell = \begin{bmatrix}
            y_1 & y_2 & ... & y_{k-1} & y_k - 1 & y_{k+1} & ... & y_n
        \end{bmatrix}^T
        $$
        '''
        batch_size, indices, softmax = self.saved_context
        gradients = softmax
        gradients[Tensor.arange(batch_size), indices] -= 1.0
        return gradients


if __name__ == '__main__':
    B = 12
    D = 3
    inputs = Tensor.randn(B, D, 1)

    linear_layer = LinearLayer(D, 2)
    outputs = linear_layer.forward(inputs)
    print(outputs.shape)
    gradients = linear_layer.backward(Tensor.randn(B, 2, 1))
    print(gradients.shape)

    relu_layer = ReluLayer()
    outputs = relu_layer.forward(inputs)
    print(outputs.shape)
    gradients = relu_layer.backward(Tensor.randn(B, D, 1))
    print(gradients.shape)

    loss_layer = MSELossLayer()
    outputs = loss_layer.forward(inputs, feedbacks=Tensor.randn(B, D))
    print(outputs)
    gradients = loss_layer.backward()
    print(gradients.shape)

    softmax_layer = SoftmaxLayer()
    outputs = softmax_layer.forward(inputs)
    print(outputs.shape)
    gradients = softmax_layer.backward(Tensor.randn(B, D, 1))
    print(gradients.shape)

    log_layer = LogLayer()
    outputs = log_layer.forward(inputs)
    print(outputs.shape)
    gradients = log_layer.backward(Tensor.randn(B, D, 1))
    print(gradients.shape)

    log_softmax_layer = LogSoftmaxLayer()
    outputs = log_softmax_layer.forward(inputs)
    print(outputs.shape)
    gradients = log_softmax_layer.backward(Tensor.randn(B, D, 1))
    print(gradients.shape)

    nll_loss_layer = NllLossLayer()
    outputs = nll_loss_layer.forward(inputs,
        feedbacks=Tensor.randint(shape=(B, 1), high=D))
    print(outputs.shape)
    gradients = nll_loss_layer.backward()
    print(gradients.shape)

    cross_entropy_layer = CrossEntropyLossLayer()
    outputs = cross_entropy_layer.forward(inputs,
        feedbacks=Tensor.randint(shape=(B, 1), high=D))
    print(outputs)
    gradients = cross_entropy_layer.backward()
    print(gradients.shape)
