<!-- markdownlint-disable -->

<a href="../mnn/layer.py#L344"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>class</kbd> `LogSoftmaxLayer`




<a href="../mnn/layer.py#L345"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(*shape, axis=1)
```








---

<a href="../mnn/layer.py#L386"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>method</kbd> `backward`

```python
backward(gradients)
```

For softmax function, when $i = k$, we have $\partial y_i / \partial x_k = y_i \cdot (1 - y_i)$ and when $i \not= k$, we have $\partial y_i / \partial x_k = - y_i y_k$ 

Therefore, for the log-softmax function, when $i = k$, we have 

$$ \partial z_i / \partial x_k = y_i^{-1} \cdot y_i \cdot (1 - y_i) = 1 - y_k $$ 

and when $i \not= k$, 

$$ \partial z_i / \partial x_k = y_i^{-1} \cdot (-y_i y_k) = - y_k $$ 

As a result, the Jacobian matrix 

$$ J_x z = \begin{bmatrix}  1 - y_1 & - y_2 & ... &  - y_n \\\\ 
    - y_1 & 1 - y_2 & ... &  - y_n \\\\  \vdots & \ddots \\\\ 
    - y_1 & - y_2 & ... & 1 - y_n \end{bmatrix} $$ 

which can be seen as an identity matrix minus a "stacked" softmax vectors. 

The gradient is, by definition, 

$$ \nabla_x \ell = J^T_x z \cdot \nabla_z \ell $$ 

---

<a href="../mnn/layer.py#L372"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>method</kbd> `forward`

```python
forward(inputs, feedbacks=None)
```

$$ z(x) = \log y(x) $$ 

where $y_i(x) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}$ 

---

<a href="../mnn/layer.py#L349"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>method</kbd> `stable_log_softmax`

```python
stable_log_softmax(inputs, axis)
```

Simplify: 

$$ \begin{aligned} z_i(x) =& \log \frac{\exp(x_i - m)}{\sum_j \exp(x_j - m)} \\\\  =& x_i - m - \log(\sum_j \exp(x_j - m)) \\\\ \end{aligned} $$ 

---

<a href="../mnn/layer.py#L35"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>method</kbd> `step`

```python
step(lr=0.01)
```





