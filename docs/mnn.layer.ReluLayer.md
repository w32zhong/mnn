<!-- markdownlint-disable -->

<a href="../mnn/layer.py#L168"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>class</kbd> `ReluLayer`




<a href="../mnn/layer.py#L5"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__()
```








---

<a href="../mnn/layer.py#L185"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>method</kbd> `backward`

```python
backward(gradients)
```

$$ \begin{aligned} \nabla_x \ell =& \nabla_f \ell \cdot \begin{bmatrix} f_1'(x_1) & 0 & ... & 0 \\\\ 0 & f_2'(x_2) & ... & 0 \\\\ \ddots \\\\ 0 & 0 & ... & f_n'(x_n) \end{bmatrix} \\\\ =& \nabla_f \ell \odot \nabla_x f \end{aligned} $$ 

---

<a href="../mnn/layer.py#L169"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>method</kbd> `forward`

```python
forward(inputs, feedbacks=None)
```

Relu activation function: 

$$ f_i(x) = \left\{ \begin{aligned} x && (x \ge 0) \\\\ 0 && (\text{otherwise}) \end{aligned} \right. $$ 

---

<a href="../mnn/layer.py#L20"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>method</kbd> `step`

```python
step(lr=0.001)
```





