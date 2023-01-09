<!-- markdownlint-disable -->

<a href="../mnn/seq_layers.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `mnn.seq_layers`






---

<a href="../mnn/seq_layers.py#L5"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `SequentialLayers`
## Background knowledge 

Each layer $f$ in neural network is just a function mapping from $f: \mathbb{R}^m \rightarrow \mathbb{R}^n $. 

Without loss of generality, suppose $z(t) = f(x(t), y(t))$, we can show: 

$$ \begin{aligned} z'(t) &= \lim_{dt \to 0} \frac{f(x(t+dt), y(t+dt)) - f(x(t), y(t))}{dt} \\\\  &= \lim_{dt \to 0} \frac{  f(x(t+dt), y(t+dt)) - f(x(t+dt), y(t))  + f(x(t+dt), y(t)) - f(x(t), y(t))  }{dt} \\\\  &= \lim_{dt \to 0} \frac{f(x(t+dt), y(t+dt)) - f(x(t+dt), y(t))}{dt}  + \lim_{dt \to 0} \frac{f(x(t+dt), y(t)) - f(x(t), y(t))}{dt} \\\\  &= \lim_{dt \to 0} \frac{f(x(t+dt), y(t+dt)) - f(x(t+dt), y(t))}  {y(t+dt) - y(t)} \times  \frac{y(t+dt) - y(t)}{dt} \\\\  &+ \lim_{dt \to 0} \frac{f(x(t+dt), y(t)) - f(x(t), y(t))}  {x(x+dt) - x(t)} \times  \frac{x(x+dt) - x(t)}{dt} \\\\  &\doteq \lim_{dt \to 0} \frac{f(x(t+dt), y(t) + \Delta y) - f(x(t+dt), y(t))}  {\Delta y} \times  \frac{y(t+dt) - y(t)}{dt} \\\\  &+ \lim_{dt \to 0} \frac{f(x(t) + \Delta x, y(t)) - f(x(t), y(t))}  {\Delta x} \times  \frac{x(x+dt) - x(t)}{dt} \\\\  &= \frac{\partial z}{\partial y} \times \frac{\partial y}{\partial t}  + \frac{\partial z}{\partial x} \times \frac{\partial x}{\partial t} \end{aligned} $$ 

<a href="../mnn/seq_layers.py#L25"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(layers)
```








---

<a href="../mnn/seq_layers.py#L36"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `backward`

```python
backward(debug=False)
```

## Gradients w.r.t. $W$ 

---

<a href="../mnn/seq_layers.py#L50"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `step`

```python
step()
```






