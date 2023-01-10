<!-- markdownlint-disable -->

<a href="../mnn/layer.py#L430"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>class</kbd> `CrossEntropyLayer`
This is to simulate PyTorch soft entropy layer which includes a softmax layer at the first layer. Labels are passed in as integer indices. 

<a href="../mnn/layer.py#L436"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(*shape, axis=1)
```








---

<a href="../mnn/layer.py#L460"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>method</kbd> `backward`

```python
backward()
```

For the log-softmax function $z(x)$, when $i = k$, we have $\partial z_i / \partial x_k = 1 - y_k$. And when $i \not= k$, we have $\partial z_i / \partial x_k = - y_k$. 

(recall that $y(x)$ is the softmax function) 

Therefore, in the cross entropy case, 

$$ \begin{aligned} \partial \ell / \partial x_k =& - \frac{\partial}{\partial x_k} \left[ \sum_i p_i z_i(x) \right] \\\\ =& - \left[ p_k \cdot (1 - y_k) - \sum_{j \not= k} p_j y_k \right] \\\\ =& - p_k + p_k y_k + \sum_{j \not= k} p_j y_k \\\\ =& - p_k + \sum_j p_j y_k \\\\ =& - p_k + y_k \\\\ \end{aligned} $$ 

Assume the specified label is $l$, we will have the gradient vector (also assuming $p_l = 1$) 

$$ \nabla_x \ell = \begin{bmatrix}  y_1 & y_2 & ... & y_{k-1} & y_k - 1 & y_{k+1} & ... & y_n \end{bmatrix}^T $$ 

---

<a href="../mnn/layer.py#L440"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>method</kbd> `forward`

```python
forward(inputs, feedbacks=None)
```

$$ \ell = -\sum_i p_i \log y_i(x) = -\log y_l(x) $$ 

where $y_i(x) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}$ and $p_i$ are real probabilities. In real implementation, label is specified as index $l$. 

---

<a href="../mnn/layer.py#L20"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>method</kbd> `step`

```python
step(lr=0.001)
```





