<!-- markdownlint-disable -->

<a href="../mnn/layer.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `mnn.layer`






---

<a href="../mnn/layer.py#L4"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `BaseLayer`




<a href="../mnn/layer.py#L5"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `BaseLayer.__init__`

```python
__init__()
```








---

<a href="../mnn/layer.py#L13"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `BaseLayer.accumulate_grads`

```python
accumulate_grads(key, val)
```





---

<a href="../mnn/layer.py#L25"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `BaseLayer.batch_reduced_val`

```python
batch_reduced_val(val)
```





---

<a href="../mnn/layer.py#L20"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `BaseLayer.step`

```python
step(lr=0.001)
```





---

<a href="../mnn/layer.py#L10"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `BaseLayer.zero_grads`

```python
zero_grads()
```






---

<a href="../mnn/layer.py#L30"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `LinearLayer`




<a href="../mnn/layer.py#L31"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `LinearLayer.__init__`

```python
__init__(*shape, bias=True)
```








---

<a href="../mnn/layer.py#L13"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `LinearLayer.accumulate_grads`

```python
accumulate_grads(key, val)
```





---

<a href="../mnn/layer.py#L46"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `LinearLayer.backward`

```python
backward(gradients)
```





---

<a href="../mnn/layer.py#L25"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `LinearLayer.batch_reduced_val`

```python
batch_reduced_val(val)
```





---

<a href="../mnn/layer.py#L39"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `LinearLayer.forward`

```python
forward(inputs, feedbacks=None)
```





---

<a href="../mnn/layer.py#L20"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `LinearLayer.step`

```python
step(lr=0.001)
```





---

<a href="../mnn/layer.py#L10"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `LinearLayer.zero_grads`

```python
zero_grads()
```






---

<a href="../mnn/layer.py#L59"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ReluLayer`




<a href="../mnn/layer.py#L5"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `ReluLayer.__init__`

```python
__init__()
```








---

<a href="../mnn/layer.py#L13"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `ReluLayer.accumulate_grads`

```python
accumulate_grads(key, val)
```





---

<a href="../mnn/layer.py#L64"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `ReluLayer.backward`

```python
backward(gradients)
```





---

<a href="../mnn/layer.py#L25"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `ReluLayer.batch_reduced_val`

```python
batch_reduced_val(val)
```





---

<a href="../mnn/layer.py#L60"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `ReluLayer.forward`

```python
forward(inputs, feedbacks=None)
```





---

<a href="../mnn/layer.py#L20"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `ReluLayer.step`

```python
step(lr=0.001)
```





---

<a href="../mnn/layer.py#L10"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `ReluLayer.zero_grads`

```python
zero_grads()
```






---

<a href="../mnn/layer.py#L70"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `MSELossLayer`




<a href="../mnn/layer.py#L5"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `MSELossLayer.__init__`

```python
__init__()
```








---

<a href="../mnn/layer.py#L13"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `MSELossLayer.accumulate_grads`

```python
accumulate_grads(key, val)
```





---

<a href="../mnn/layer.py#L78"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `MSELossLayer.backward`

```python
backward()
```





---

<a href="../mnn/layer.py#L25"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `MSELossLayer.batch_reduced_val`

```python
batch_reduced_val(val)
```





---

<a href="../mnn/layer.py#L71"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `MSELossLayer.forward`

```python
forward(inputs, feedbacks=None)
```





---

<a href="../mnn/layer.py#L20"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `MSELossLayer.step`

```python
step(lr=0.001)
```





---

<a href="../mnn/layer.py#L10"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `MSELossLayer.zero_grads`

```python
zero_grads()
```








---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
