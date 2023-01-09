<!-- markdownlint-disable -->

<a href="../mnn/layer.py#L305"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>class</kbd> `LogLayer`




<a href="../mnn/layer.py#L5"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__()
```








---

<a href="../mnn/layer.py#L315"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>method</kbd> `backward`

```python
backward(gradients)
```

Because this layer is element-wise operation, i.e., $J_x y = \operatorname{diag}(x^{-1})$, the final gradient can be also simplified into Hadamard product: 

$$ \nabla_x \ell = J^T_x y \cdot \nabla_y \ell = x^{-1} \odot \nabla_y \ell $$ 

---

<a href="../mnn/layer.py#L306"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>method</kbd> `forward`

```python
forward(inputs, feedbacks=None)
```

$$ y = \log(x) $$ 

---

<a href="../mnn/layer.py#L20"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>method</kbd> `step`

```python
step(lr=0.001)
```





