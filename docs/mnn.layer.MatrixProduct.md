<!-- markdownlint-disable -->

<a href="../mnn/layer.py#L540"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>class</kbd> `MatrixProduct`




<a href="../mnn/layer.py#L5"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__()
```








---

<a href="../mnn/layer.py#L554"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>method</kbd> `backward`

```python
backward(gradients)
```

Jacobian matrix of a flatten $\bar{D}$ w.r.t. a flatten $\bar{Q}$: 

$$ \begin{aligned} \partial \bar{D} / \partial \bar{Q} =& \begin{bmatrix} \frac{\partial D_{1,1}}{\partial Q_{1,1}} & \frac{\partial D_{1,1}}{\partial Q_{1,2}} & ... & \frac{\partial D_{1,1}}{\partial Q_{1,d}} & 0 & 0 & 0 & 0 & 0 & ... & 0 \\\\ \frac{\partial D_{1,2}}{\partial Q_{1,1}} & \frac{\partial D_{1,2}}{\partial Q_{1,2}} & ... & \frac{\partial D_{1,2}}{\partial Q_{1,d}} & 0 & 0 & 0 & 0 & 0 & ... & 0 \\\\ \vdots \\\\ \frac{\partial D_{1,m}}{\partial Q_{1,1}} & \frac{\partial D_{1,m}}{\partial Q_{1,2}} & ... & \frac{\partial D_{1,m}}{\partial Q_{1,d}} & 0 & 0 & 0 & 0 & 0 & ... & 0 \\\\ 0 & 0 & ... & 0 & \frac{\partial D_{2,1}}{\partial Q_{2,1}} & \frac{\partial D_{2,1}}{\partial Q_{2,2}} & ... & \frac{\partial D_{2,1}}{\partial Q_{2,d}} & 0 & ... & 0\\\\ 0 & 0 & ... & 0 & \frac{\partial D_{2,2}}{\partial Q_{2,1}} & \frac{\partial D_{2,2}}{\partial Q_{2,2}} & ... & \frac{\partial D_{2,2}}{\partial Q_{2,d}} & 0 & ... & 0\\\\ \ddots & \ddots & \ddots & \ddots & \ddots & \ddots & \ddots & \ddots & \ddots & \ddots  & \ddots \end{bmatrix} \\\\ =& \begin{bmatrix} \ddots & & \\\\ & \left( \frac{\partial D_{j,\star}}{\partial Q_{j,\star}} \right)_{m \times d}, \quad j = 1, ..., n & \\\\ & & \ddots \end{bmatrix} \\\\ =& \begin{bmatrix} \ddots & & \\\\ & \left( P^T \right)_{m \times d}& \\\\ & & \ddots \end{bmatrix} \end{aligned} $$ 

Similarly, Jacobian matrix of another order flatten $\widetilde{D}$ w.r.t. a flatten $\widetilde{P}$: 

$$ \begin{aligned} \partial \widetilde{D} / \partial \widetilde{P} =& \begin{bmatrix} \frac{\partial D_{1,1}}{\partial P_{1,1}} & \frac{\partial D_{1,1}}{\partial P_{2,1}} & ... & \frac{\partial D_{1,1}}{\partial P_{d,1}} & 0 & 0 & 0 & 0 & 0 & ... & 0 \\\\ \frac{\partial D_{2,1}}{\partial P_{1,1}} & \frac{\partial D_{2,1}}{\partial P_{2,1}} & ... & \frac{\partial D_{2,1}}{\partial P_{d,1}} & 0 & 0 & 0 & 0 & 0 & ... & 0 \\\\ \vdots \\\\ \frac{\partial D_{n,1}}{\partial P_{1,1}} & \frac{\partial D_{n,1}}{\partial P_{2,1}} & ... & \frac{\partial D_{n,1}}{\partial P_{d,1}} & 0 & 0 & 0 & 0 & 0 & ... & 0 \\\\ 0 & 0 & ... & 0 & \frac{\partial D_{1,2}}{\partial P_{1,2}} & \frac{\partial D_{1,2}}{\partial P_{2,2}} & ... & \frac{\partial D_{1,2}}{\partial P_{d,2}} & 0 & ... & 0\\\\ 0 & 0 & ... & 0 & \frac{\partial D_{2,2}}{\partial P_{1,2}} & \frac{\partial D_{2,2}}{\partial P_{2,2}} & ... & \frac{\partial D_{2,2}}{\partial P_{d,2}} & 0 & ... & 0\\\\ \ddots & \ddots & \ddots & \ddots & \ddots & \ddots & \ddots & \ddots & \ddots & \ddots  & \ddots \end{bmatrix}\\\\ =& \begin{bmatrix} \ddots & & \\\\ & \left( \frac{\partial D_{\star,j}}{\partial P_{\star,j}} \right)_{n \times d}, \quad j = 1, ..., m & \\\\ & & \ddots \end{bmatrix} \\\\ =& \begin{bmatrix} \ddots & & \\\\ & Q_{n \times d} & \\\\ & & \ddots \end{bmatrix} \end{aligned} $$ 

Recall that for $y=f(x)$, 

$$ \nabla_x^T \ell =  \nabla_y^T \ell \times J_x y $$ 

therefore, 

$$ \tag{4} \nabla_{\bar{Q}}^T \ell =  \left( \nabla_{\bar{D}}^T \ell \right)_{1 \times nm}  \times  \left( \frac{\partial \bar{D} }{ \partial \bar{Q} } \right)_{nm \times nd}  \stackrel{\text{unroll}}{\Rightarrow}\quad \nabla_{Q} \ell = \left( D \cdot P^T \right)_{n \times d} $$ 

$$ \tag{5} \nabla_{\widetilde{P}}^T \ell =  \left( \nabla_{\widetilde{D}}^T \ell \right)_{1 \times nm}  \times  \left( \frac{\partial \widetilde{D} }{ \partial \widetilde{P} } \right)_{nm \times md}  \stackrel{\text{unroll}}{\Rightarrow}\quad \nabla_{P^T} \ell = \left( D^T \cdot Q \right)_{m \times d} $$ 

since Eq. (5) has \widetilde{P} rolled as a transposed form, its gradients is w.r.t. $P^T$, hence the final form of Eq. (5) is 

$$ \tag{6} \nabla_{P} \ell = \left( Q^T \cdot D \right)_{d \times m} $$ 

---

<a href="../mnn/layer.py#L541"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>method</kbd> `forward`

```python
forward(inputs, feedbacks=None)
```

Assume 

$$ D_{n \times m} = Q_{n \times d} \cdot P_{d \times m} \\ $$ 

where $p_i \in \mathbb{R}^{n \times 1}$ and $q_j \in \mathbb{R}^{d \times 1}$. 

---

<a href="../mnn/layer.py#L35"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>method</kbd> `step`

```python
step(lr=0.01)
```





