<!-- markdownlint-disable -->

<a href="../mnn/layer.py#L224"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>class</kbd> `MSELossLayer`




<a href="../mnn/layer.py#L5"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__()
```








---

<a href="../mnn/layer.py#L235"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>method</kbd> `backward`

```python
backward()
```

Gradient $\nabla_x \ell = 2(x - y)$ 

---

<a href="../mnn/layer.py#L225"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>method</kbd> `forward`

```python
forward(inputs, feedbacks=None)
```

Loss $\ell_i = \sum_{i} (x_i - y_i)^2$ 

---

<a href="../mnn/layer.py#L35"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>method</kbd> `step`

```python
step(lr=0.01)
```





