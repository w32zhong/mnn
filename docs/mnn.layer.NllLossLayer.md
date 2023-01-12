<!-- markdownlint-disable -->

<a href="../mnn/layer.py#L436"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>class</kbd> `NllLossLayer`
This is to simulate PyTorch NLL layer which computes a negative expectation loss. Labels are passed in as integer indices. 

<a href="../mnn/layer.py#L5"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__()
```








---

<a href="../mnn/layer.py#L459"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>method</kbd> `backward`

```python
backward()
```

In this case, the gradient vector w.r.t. $q$ is simply an almost-zero vector $v$ where $v_l = -1$. 

---

<a href="../mnn/layer.py#L441"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>method</kbd> `forward`

```python
forward(inputs, feedbacks=None)
```

$$ \ell(q) = -\sum^n_{i=0} p_i q_i = -q_l $$ 

where $p \in \{0, 1\}$ indicates true probability, but in actual implementation, the label is given as an index $l \in \mathbb{N}$. 

---

<a href="../mnn/layer.py#L35"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>method</kbd> `step`

```python
step(lr=0.01)
```





