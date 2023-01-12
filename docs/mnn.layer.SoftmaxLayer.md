<!-- markdownlint-disable -->

<a href="../mnn/layer.py#L243"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>class</kbd> `SoftmaxLayer`




<a href="../mnn/layer.py#L244"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(*shape, axis=1)
```








---

<a href="../mnn/layer.py#L271"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>method</kbd> `backward`

```python
backward(gradients)
```

When $i = k$, 

$$ \begin{aligned} \partial y_i / \partial x_k =& \frac{\exp(x_i)[\sum_j \exp(x_j)] - \exp(x_i)\exp(x_i)}{[\sum_j \exp(x_j)]^2} \\\\ =& \frac{\exp(x_i) \left([\sum_j \exp(x_j)] -\exp(x_i)\right)}{[\sum_j \exp(x_j)]^2} \\\\ =& \frac{\exp(x_i)}{\sum_j \exp(x_j)} \cdot  \frac{[\sum_j \exp(x_j)] - \exp(x_i)}{\sum_j \exp(x_j)} \\\\ =& y_i \cdot (1 - y_i) \\\\ =& y_i - y_i y_k \end{aligned} $$ 

When $i \not= k$, 

$$ \begin{aligned} \partial y_i / \partial x_k =& \frac{0 - \exp(x_i)\exp(x_k)}{[\sum_j \exp(x_j)]^2} \\\\ =& - y_i y_k \end{aligned} $$ 

As a result, the Jacobian matrix 

$$ J_x y = \begin{bmatrix}  y_1 - y_1 y_1 & - y_1 y_2 & ... &  - y_1 y_n \\\\  y_2 y_1 & y_2 - y_1 y_2 & ... &  - y_1 y_n \\\\  \vdots & \ddots \\\\  y_n y_1 & - y_n y_2 & ... & y_n - y_n y_n \end{bmatrix} = \operatorname{diag}(y) - y \times y^T $$ 

---

<a href="../mnn/layer.py#L267"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>method</kbd> `forward`

```python
forward(inputs, feedbacks=None)
```





---

<a href="../mnn/layer.py#L248"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>method</kbd> `stable_softmax`

```python
stable_softmax(inputs, axis)
```

## Numerical-stable softmax 

Since some exponentials are quite large numbers, we can use 

$$ y_i(x) = \frac{\exp(x_i)}{\sum_j \exp(x_j)} = \frac{\exp(x_i - m)}{\sum_j \exp(x_j - m)} $$ 

where $m = \max(x_1, x_2, ...)$. 

---

<a href="../mnn/layer.py#L35"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>method</kbd> `step`

```python
step(lr=0.01)
```





