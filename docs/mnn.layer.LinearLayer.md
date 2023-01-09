<!-- markdownlint-disable -->

<a href="../mnn/layer.py#L30"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>class</kbd> `LinearLayer`




<a href="../mnn/layer.py#L31"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(*shape, bias=True)
```

An linear layer to compute $y_{m \times 1} = W_{m \times n} x_{n \times 1} + b_{m \times 1}$, where 

$$ W = \begin{bmatrix} w_{1,1} & w_{1, 2} & ... & w_{1, n} \\\\ w_{2,1} & w_{2, 2} & ... & w_{2, n} \\\\ \vdots \\\\ w_{m,1} & w_{m, 2} & ... & w_{m, n} \\\\ \end{bmatrix} $$ 




---

<a href="../mnn/layer.py#L59"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>method</kbd> `backward`

```python
backward(gradients)
```





---

<a href="../mnn/layer.py#L25"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>method</kbd> `batch_reduced`

```python
batch_reduced(val)
```





---

<a href="../mnn/layer.py#L52"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>method</kbd> `forward`

```python
forward(inputs, feedbacks=None)
```





---

<a href="../mnn/layer.py#L20"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>method</kbd> `step`

```python
step(lr=0.001)
```





