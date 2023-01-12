<!-- markdownlint-disable -->

<a href="../mnn/layer.py#L45"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>class</kbd> `LinearLayer`




<a href="../mnn/layer.py#L46"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(*shape, bias=True)
```








---

<a href="../mnn/layer.py#L74"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>method</kbd> `backward`

```python
backward(gradients)
```

## Gradients w.r.t. $W$ 

Because $y_i = w_{i,1} x_1 + w_{i,2} x_2 + ... + w_{i,n} x_n + b_i$, we have $\partial y_i / \partial w_{i,j} = x_j$ and this derivative is $0$ for $w_{k,j}$ when $k \not= i$. 

When we "flatten" the $W$ into a "long vector" $w$, the Jacobian w.r.t. $w$ then becomes: 

$$ J_w = \begin{bmatrix} x_1 & x_2 & ... & x_n &   0 &   0 & ... &   0 &  0  & 0   & 0   \\\\ 0   & 0   & ... & 0   & x_1 & x_2 & ... &   0 &  0  & 0   & 0   \\\\ \vdots \\\\ 0   & 0   & ... & 0   & 0   & 0   & ... & x_1 & x_2 & ... & x_n \\\\ \end{bmatrix}_{m \times (mn)} $$ 

If we chain the gradient product (assuming the final loss is scaler $\ell$): 

$$ \nabla^T_w \ell = \nabla^T_y \ell \times J_w = \begin{bmatrix}  x_1 \frac{\partial \ell}{\partial y_1} & x_2 \frac{\partial \ell}{\partial y_1} & ... & x_n \frac{\partial \ell}{\partial y_1} &  x_1 \frac{\partial \ell}{\partial y_2} & x_2 \frac{\partial \ell}{\partial y_2} & ... & x_n \frac{\partial \ell}{\partial y_2} &  ... \end{bmatrix} $$ 

As apparently it is a recycling patten, we can "unroll" the Jacobian to a matrix so that it matches the dimension of $W$: 

$$ \nabla_W \ell = \begin{bmatrix}  x_1 \frac{\partial \ell}{\partial y_1} & x_2 \frac{\partial \ell}{\partial y_1} & ... & x_n \frac{\partial \ell}{\partial y_1} \\\\  x_1 \frac{\partial \ell}{\partial y_2} & x_2 \frac{\partial \ell}{\partial y_2} & ... & x_n \frac{\partial \ell}{\partial y_2} \\\\  \vdots \end{bmatrix} $$ 

thus it can be written in 

$$ \tag{1} \nabla_W \ell = (\nabla_y \ell)_{m \times 1} \times (x^T)_{1 \times n} $$ 

## Gradients w.r.t. $b$ 

Because $y_i = w_{i,1} x_1 + w_{i,2} x_2 + ... + w_{i,n} x_n + b_i$, the Jacobian w.r.t. $b$ is an identity matrix, and the gradients of it is just the down-stream gradients: 

$$ \tag{2} \nabla^T_b \ell = \nabla^T_y \ell \times J_b = \nabla^T_y \ell \times E = \nabla^T_y \ell $$ 

## Gradients w.r.t. inputs $x$ 

This is the gradients to be back propagated to upstream, instead of being used to update the parameters of this layer. 

The Jacobian w.r.t. $w$ is, according to $y_i = w_{i,1} x_1 + w_{i,2} x_2 + ... + w_{i,n} x_n + b_i$, 

$$ J_x = \begin{bmatrix} \frac{\partial y_1}{\partial x_1} & \frac{\partial y_1}{\partial x_2} & ... & \frac{\partial y_1}{\partial x_n} \\\\ \frac{\partial y_2}{\partial x_1} & \frac{\partial y_2}{\partial x_2} & ... & \frac{\partial y_2}{\partial x_n} \\\\ \vdots & \ddots \\\\ \frac{\partial y_m}{\partial x_1} & \frac{\partial y_m}{\partial x_2} & ... & \frac{\partial y_m}{\partial x_n} \\\\ \end{bmatrix} = \begin{bmatrix} w_{1,1} & w_{1, 2} & ... & w_{1, n} \\\\ w_{2,1} & w_{2, 2} & ... & w_{2, n} \\\\ \vdots \\\\ w_{m,1} & w_{m, 2} & ... & w_{m, n} \\\\ \end{bmatrix} = W $$ 

as a result, 

$$ \tag{3} \nabla_x \ell = J_x^T \times \nabla_y \ell  =  W^T \times \nabla_y \ell $$ 

---

<a href="../mnn/layer.py#L54"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>method</kbd> `forward`

```python
forward(inputs, feedbacks=None)
```

An linear layer to compute $y_{m \times 1} = W_{m \times n} x_{n \times 1} + b_{m \times 1}$, where 

$$ W = \begin{bmatrix} w_{1,1} & w_{1, 2} & ... & w_{1, n} \\\\ w_{2,1} & w_{2, 2} & ... & w_{2, n} \\\\ \vdots \\\\ w_{m,1} & w_{m, 2} & ... & w_{m, n} \\\\ \end{bmatrix} $$ 

---

<a href="../mnn/layer.py#L35"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>method</kbd> `step`

```python
step(lr=0.01)
```





