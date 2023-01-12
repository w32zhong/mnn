<!-- markdownlint-disable -->

<a href="../mnn/seq_layers.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `mnn.seq_layers`






---

<a href="../mnn/seq_layers.py#L5"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `SequentialLayers`
## Backpropagation background 

Each layer $f$ in neural network is just a function mapping from $\mathbb{R}^m \rightarrow \mathbb{R}^n $. 

Without loss of generality, consider the scaler version $z(t) = f(x(t), y(t))$, we can show: 

$$ \begin{aligned}  z'(t) =& \lim_{dt \to 0} \frac{f(x(t+dt), y(t+dt)) - f(x(t), y(t))}{dt} \\\\  =& \lim_{dt \to 0} \frac{  f(x(t+dt), y(t+dt)) - f(x(t+dt), y(t))  + f(x(t+dt), y(t)) - f(x(t), y(t))  }{dt} \\\\  =& \lim_{dt \to 0} \frac{f(x(t+dt), y(t+dt)) - f(x(t+dt), y(t))}{dt}  + \lim_{dt \to 0} \frac{f(x(t+dt), y(t)) - f(x(t), y(t))}{dt} \\\\  =& \lim_{dt \to 0} \frac{f(x(t+dt), y(t+dt)) - f(x(t+dt), y(t))}  {y(t+dt) - y(t)} \times  \frac{y(t+dt) - y(t)}{dt} + \\\\  & \lim_{dt \to 0} \frac{f(x(t+dt), y(t)) - f(x(t), y(t))}  {x(x+dt) - x(t)} \times  \frac{x(x+dt) - x(t)}{dt} \\\\  \doteq& \lim_{dt \to 0} \frac{f(x(t+dt), y(t) + \Delta y) - f(x(t+dt), y(t))}  {\Delta y} \times  \frac{y(t+dt) - y(t)}{dt} + \\\\  & \lim_{dt \to 0} \frac{f(x(t) + \Delta x, y(t)) - f(x(t), y(t))}  {\Delta x} \times  \frac{x(x+dt) - x(t)}{dt} \\\\  =& \left.\frac{\partial f}{\partial y}\right|_{y=y(t)} \cdot \frac{\partial y}{\partial t}  + \left.\frac{\partial f}{\partial x}\right|_{x=x(t)} \cdot \frac{\partial x}{\partial t} \end{aligned} $$ 

iff $dt \rightarrow 0$ implies $\Delta x \rightarrow 0$ and $\Delta y \rightarrow 0$ (Lipschitz continuity). 

In more general case when $z(t) = f(x(t))$ where $x \in \mathbb{R}^n, t \in \mathbb{R}^m, f: \mathbb{R}^n \rightarrow \mathbb{R}$ and $x: \mathbb{R}^m \rightarrow \mathbb{R}^n$, 

$$ \begin{aligned} \frac{\partial z}{\partial t_i} =& \begin{bmatrix} \frac{\partial f}{\partial x_1} & ... & \frac{\partial f}{\partial x_n} \end{bmatrix}_{x = x(t)} \cdot \begin{bmatrix} \frac{\partial x_1}{\partial t_i} \\\\ \vdots \\\\ \frac{\partial x_n}{\partial t_i} \end{bmatrix} \\\\ \doteq& \nabla_x^T f (x = x(t)) \cdot \begin{bmatrix} \frac{\partial x_1}{\partial t_i} \\\\ \vdots \\\\ \frac{\partial x_n}{\partial t_i} \end{bmatrix} \\\\ \end{aligned} $$ 

therefore 

$$ \tag{1} \nabla_t^T z(t) \doteq \begin{bmatrix} \frac{\partial f}{\partial t_1}, ..., \frac{\partial f}{\partial t_m} \end{bmatrix} = \nabla_x^T f (x = x(t)) \cdot \begin{bmatrix}  \partial x_1 / \partial t_1 & \partial x_1 / \partial t_2 & ... & \partial x_1 / \partial t_m \\\\  \partial x_2 / \partial t_1 & \partial x_2 / \partial t_2 & ... & \partial x_2 / \partial t_m \\\\  \vdots & \ddots \\\\  \partial x_n / \partial t_1 & \partial x_n / \partial t_2 & ... & \partial x_n / \partial t_m \\\\ \end{bmatrix} $$ 

where the RHS matrix is called the Jacobian matrix $J_t x$. 

<a href="../mnn/seq_layers.py#L64"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(layers)
```








---

<a href="../mnn/seq_layers.py#L75"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `backward`

```python
backward(debug=False)
```

## Backpropagation 

As seen in Eq. (1), we can propagate gradient w.r.t. $t$ back from down-stream gradients using 

$$  \nabla_t^T z(t) = \nabla_x^T f (x = x(t)) \cdot J_t x $$ 

---

<a href="../mnn/seq_layers.py#L122"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_config`

```python
get_config()
```





---

<a href="../mnn/seq_layers.py#L125"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `load_weights`

```python
load_weights(state_dict, config=None, verbose=False)
```





---

<a href="../mnn/seq_layers.py#L113"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `state_dict`

```python
state_dict()
```





---

<a href="../mnn/seq_layers.py#L96"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `step`

```python
step()
```

## Gradient descent 

At time $k$, to update the parameter $t$ to achieve lower $z$ value (loss): 

$$ t^{(k + 1)} = t^{(k)} - \eta \cdot \nabla_t z $$ 

---

<a href="../mnn/seq_layers.py#L109"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `zero_grads`

```python
zero_grads()
```






