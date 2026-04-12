# 4. Gradients of Matrix Multiply, Element-wise, and Reduction Operations

## Motivation

Document 3 established that every backward function computes a VJP: $\bar{\mathbf{x}} = \mathbf{J}^T \bar{\mathbf{z}}$. But we never form $\mathbf{J}$ explicitly — we derive efficient formulas specific to each operation. This document derives those formulas **from scratch** for the three fundamental operation families: **matrix multiplication**, **element-wise functions**, and **reductions**.

These three cover the vast majority of neural network operations: linear layers are matmul, activations are element-wise, and losses are reductions.

---

## 4.1 Matrix Multiplication: $\mathbf{Y} = \mathbf{A}\mathbf{B}$

This is the most important gradient derivation in deep learning. Let's do it with full rigor.

### Setup

$$\mathbf{A} \in \mathbb{R}^{m \times n}, \quad \mathbf{B} \in \mathbb{R}^{n \times p}, \quad \mathbf{Y} = \mathbf{AB} \in \mathbb{R}^{m \times p}$$

We want: given $\bar{\mathbf{Y}} = \frac{\partial L}{\partial \mathbf{Y}} \in \mathbb{R}^{m \times p}$ (the upstream gradient), compute $\bar{\mathbf{A}} = \frac{\partial L}{\partial \mathbf{A}} \in \mathbb{R}^{m \times n}$ and $\bar{\mathbf{B}} = \frac{\partial L}{\partial \mathbf{B}} \in \mathbb{R}^{n \times p}$.

### Step 1: Write the element-wise formula

$$Y_{ij} = \sum_{k=1}^{n} A_{ik} B_{kj}$$

### Step 2: Compute partial derivatives element-wise

**Derivative of $Y_{ij}$ with respect to $A_{i'k'}$:**

$$\frac{\partial Y_{ij}}{\partial A_{i'k'}} = \begin{cases} B_{k'j} & \text{if } i = i' \\ 0 & \text{if } i \neq i' \end{cases}$$

Because only the term $A_{ik'}B_{k'j}$ in the sum depends on $A_{i'k'}$, and only when $i = i'$.

**Derivative of $Y_{ij}$ with respect to $B_{k'j'}$:**

$$\frac{\partial Y_{ij}}{\partial B_{k'j'}} = \begin{cases} A_{ik'} & \text{if } j = j' \\ 0 & \text{if } j \neq j' \end{cases}$$

### Step 3: Apply the chain rule

$$\frac{\partial L}{\partial A_{i'k'}} = \sum_{i,j} \frac{\partial L}{\partial Y_{ij}} \cdot \frac{\partial Y_{ij}}{\partial A_{i'k'}} = \sum_j \bar{Y}_{i'j} \cdot B_{k'j}$$

Notice: the sum over $i$ collapsed because $\frac{\partial Y_{ij}}{\partial A_{i'k'}} = 0$ when $i \neq i'$.

$$\frac{\partial L}{\partial A_{i'k'}} = \sum_j \bar{Y}_{i'j} B_{k'j} = \sum_j \bar{Y}_{i'j} B^T_{jk'}$$

**This is the $(i', k')$ element of $\bar{\mathbf{Y}} \mathbf{B}^T$!**

$$\boxed{\frac{\partial L}{\partial \mathbf{A}} = \bar{\mathbf{Y}} \mathbf{B}^T}$$

Similarly for $\mathbf{B}$:

$$\frac{\partial L}{\partial B_{k'j'}} = \sum_{i,j} \frac{\partial L}{\partial Y_{ij}} \cdot \frac{\partial Y_{ij}}{\partial B_{k'j'}} = \sum_i \bar{Y}_{ij'} A_{ik'} = \sum_i A^T_{k'i} \bar{Y}_{ij'}$$

$$\boxed{\frac{\partial L}{\partial \mathbf{B}} = \mathbf{A}^T \bar{\mathbf{Y}}}$$

### Shape verification

| Quantity | Shape | Check |
|----------|-------|-------|
| $\bar{\mathbf{Y}}$ | $m \times p$ | Given |
| $\mathbf{B}^T$ | $p \times n$ | |
| $\bar{\mathbf{Y}} \mathbf{B}^T$ | $m \times n$ | ✓ matches $\mathbf{A}$ |
| $\mathbf{A}^T$ | $n \times m$ | |
| $\mathbf{A}^T \bar{\mathbf{Y}}$ | $n \times p$ | ✓ matches $\mathbf{B}$ |

### Mnemonic

For $\mathbf{Y} = \mathbf{A}\mathbf{B}$:
- **Gradient of left factor**: upstream × (right factor)$^T$ → $\bar{\mathbf{Y}} \mathbf{B}^T$
- **Gradient of right factor**: (left factor)$^T$ × upstream → $\mathbf{A}^T \bar{\mathbf{Y}}$

This pattern recurs everywhere. When you see $\mathbf{Y} = \mathbf{W}\mathbf{x}$:
- $\frac{\partial L}{\partial \mathbf{W}} = \bar{\mathbf{y}} \mathbf{x}^T$ (outer product)
- $\frac{\partial L}{\partial \mathbf{x}} = \mathbf{W}^T \bar{\mathbf{y}}$

---

## 4.2 Numerical Verification of Matmul Gradient

```python
import torch

A = torch.randn(3, 4, requires_grad=True)
B = torch.randn(4, 2, requires_grad=True)

Y = A @ B                      # (3, 2)
L = Y.sum()                    # scalar loss (simplest reduction)

L.backward()

# For L = sum(Y), dL/dY is all ones
grad_Y = torch.ones(3, 2)

# Verify formulas
print("A.grad matches grad_Y @ B.T:", torch.allclose(A.grad, grad_Y @ B.T))    # True
print("B.grad matches A.T @ grad_Y:", torch.allclose(B.grad, A.T @ grad_Y))    # True
```

### With a more complex loss:

```python
A = torch.randn(3, 4, requires_grad=True)
B = torch.randn(4, 2, requires_grad=True)

Y = A @ B
L = (Y ** 2).sum()     # L = sum(Y_ij^2)

L.backward()

# dL/dY = 2*Y (derivative of Y_ij^2 is 2*Y_ij)
grad_Y = 2 * (A @ B).detach()

# Verify
print("A.grad matches grad_Y @ B.T:", torch.allclose(A.grad, grad_Y @ B.T))    # True
print("B.grad matches A.T @ grad_Y:", torch.allclose(B.grad, A.T @ grad_Y))    # True
```

---

## 4.3 Matmul with Batch Dimension

For batched operations: $\mathbf{Y} = \mathbf{X}\mathbf{W}$ where $\mathbf{X} \in \mathbb{R}^{B \times D_{in}}$, $\mathbf{W} \in \mathbb{R}^{D_{in} \times D_{out}}$.

The formulas still hold:

$$\frac{\partial L}{\partial \mathbf{X}} = \bar{\mathbf{Y}} \mathbf{W}^T \in \mathbb{R}^{B \times D_{in}}$$

$$\frac{\partial L}{\partial \mathbf{W}} = \mathbf{X}^T \bar{\mathbf{Y}} \in \mathbb{R}^{D_{in} \times D_{out}}$$

Notice that $\mathbf{X}^T \bar{\mathbf{Y}}$ has shape $(D_{in} \times B)(B \times D_{out}) = D_{in} \times D_{out}$ — the batch dimension disappears through the matmul! This is how the batch is aggregated: the matrix multiplication over the batch dimension inherently sums the per-sample gradients.

**Expanding**: $\frac{\partial L}{\partial W_{ij}} = \sum_{b=1}^B X_{bi}^T \bar{Y}_{bj} = \sum_{b=1}^B x_i^{(b)} \bar{y}_j^{(b)}$

This IS the sum of per-sample outer products — computed in one matmul.

---

## 4.4 Element-wise Operations

### General case

If $\mathbf{y} = f(\mathbf{x})$ where $f$ acts element-wise ($y_i = f(x_i)$), the Jacobian is **diagonal**:

$$\mathbf{J} = \text{diag}(f'(x_1), f'(x_2), \ldots, f'(x_n))$$

The VJP simplifies to the **Hadamard (element-wise) product**:

$$\bar{\mathbf{x}} = \mathbf{J}^T \bar{\mathbf{y}} = \bar{\mathbf{y}} \odot f'(\mathbf{x})$$

**No matrix multiplication needed** — just element-wise multiply. This is $O(n)$ instead of $O(n^2)$.

### ReLU: $f(x) = \max(0, x)$

$$f'(x) = \begin{cases} 1 & \text{if } x > 0 \\ 0 & \text{if } x < 0 \\ \text{undefined} & \text{if } x = 0 \end{cases}$$

Convention: $f'(0) = 0$ (or any value; measure-zero event).

$$\bar{\mathbf{x}} = \bar{\mathbf{y}} \odot \mathbb{1}[\mathbf{x} > 0]$$

where $\mathbb{1}[\mathbf{x} > 0]$ is a binary mask.

```python
x = torch.tensor([-1.0, 0.5, 2.0, -0.3], requires_grad=True)
y = torch.relu(x)
L = y.sum()
L.backward()
print(f"x =      {x.data}")      # [-1.0, 0.5, 2.0, -0.3]
print(f"x.grad = {x.grad}")      # [ 0.0, 1.0, 1.0,  0.0]
# Gradient is 1 where x > 0, 0 where x ≤ 0
```

### Sigmoid: $\sigma(x) = \frac{1}{1 + e^{-x}}$

$$\sigma'(x) = \sigma(x)(1 - \sigma(x))$$

**Derivation:**

$$\sigma(x) = (1 + e^{-x})^{-1}$$

$$\sigma'(x) = -1 \cdot (1 + e^{-x})^{-2} \cdot (-e^{-x}) = \frac{e^{-x}}{(1 + e^{-x})^2}$$

Note that $1 - \sigma(x) = \frac{e^{-x}}{1 + e^{-x}}$, so:

$$\sigma'(x) = \sigma(x) \cdot (1 - \sigma(x))$$

VJP: $\bar{\mathbf{x}} = \bar{\mathbf{y}} \odot \sigma(\mathbf{x}) \odot (1 - \sigma(\mathbf{x}))$

```python
x = torch.tensor([0.0, 1.0, -1.0], requires_grad=True)
y = torch.sigmoid(x)
L = y.sum()
L.backward()
print(f"x.grad = {x.grad}")
# sigmoid(0)*(1-sigmoid(0)) = 0.25
# sigmoid(1)*(1-sigmoid(1)) ≈ 0.1966
# sigmoid(-1)*(1-sigmoid(-1)) ≈ 0.1966
```

### Tanh: $\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$

$$\tanh'(x) = 1 - \tanh^2(x) = \text{sech}^2(x)$$

VJP: $\bar{\mathbf{x}} = \bar{\mathbf{y}} \odot (1 - \tanh^2(\mathbf{x}))$

### Power: $y = x^n$

$$\frac{dy}{dx} = nx^{n-1}$$

VJP: $\bar{\mathbf{x}} = \bar{\mathbf{y}} \odot n \mathbf{x}^{n-1}$

### Exponential: $y = e^x$

$$\frac{dy}{dx} = e^x$$

VJP: $\bar{\mathbf{x}} = \bar{\mathbf{y}} \odot e^{\mathbf{x}} = \bar{\mathbf{y}} \odot \mathbf{y}$

(Note: the gradient uses the output $\mathbf{y}$, not the input $\mathbf{x}$ — this is what's saved in the computation graph)

---

## 4.5 Reduction Operations

Reductions compress dimensions: they take a tensor and reduce one or more axes.

### Sum: $L = \sum_{i} x_i$

$$\frac{\partial L}{\partial x_i} = 1 \quad \forall i$$

VJP: $\bar{\mathbf{x}} = \bar{L} \cdot \mathbf{1}$ (broadcast scalar to input shape)

### Sum over axis: $y_j = \sum_{i} X_{ij}$ (sum rows)

For a matrix $\mathbf{X} \in \mathbb{R}^{m \times n}$, $\mathbf{y} \in \mathbb{R}^n$:

$$\frac{\partial y_j}{\partial X_{i'j'}} = \begin{cases} 1 & \text{if } j = j' \\ 0 & \text{otherwise} \end{cases}$$

VJP: $\bar{X}_{ij} = \bar{y}_j$ — the gradient is **broadcast back** along the summed dimension.

```python
X = torch.randn(3, 4, requires_grad=True)
y = X.sum(dim=0)   # (4,) — sum over rows
L = (y ** 2).sum()
L.backward()
print(f"X.grad shape: {X.grad.shape}")  # (3, 4)
# Each column of X.grad is identical (broadcast)
print(f"X.grad:\n{X.grad}")
```

### Mean: $L = \frac{1}{n}\sum_{i} x_i$

$$\frac{\partial L}{\partial x_i} = \frac{1}{n}$$

VJP: $\bar{\mathbf{x}} = \frac{\bar{L}}{n} \cdot \mathbf{1}$

### Max (argmax): $y = \max_i(x_i) = x_{k}$ where $k = \arg\max_i x_i$

$$\frac{\partial y}{\partial x_i} = \begin{cases} 1 & \text{if } i = k \\ 0 & \text{otherwise} \end{cases}$$

Only the maximum element gets gradient. All others get zero.

```python
x = torch.tensor([1.0, 3.0, 2.0], requires_grad=True)
y = x.max()
y.backward()
print(f"x.grad = {x.grad}")  # [0, 1, 0] — only max element
```

---

## 4.6 Combined Example: Linear Layer + ReLU + MSE Loss

Let's trace the full backward pass for a common pattern:

$$\mathbf{Y} = \text{ReLU}(\mathbf{X}\mathbf{W} + \mathbf{b}), \quad L = \frac{1}{2B}\|\mathbf{Y} - \mathbf{T}\|_F^2$$

where Frobenius norm $\|\cdot\|_F^2 = \sum_{ij}(\cdot)_{ij}^2$.

**Shapes**: $\mathbf{X} \in \mathbb{R}^{B \times D_{in}}$, $\mathbf{W} \in \mathbb{R}^{D_{in} \times D_{out}}$, $\mathbf{b} \in \mathbb{R}^{D_{out}}$, $\mathbf{Y} \in \mathbb{R}^{B \times D_{out}}$.

**Forward pass decomposition**:

| Step | Op | Output | Shape |
|------|----|--------|-------|
| 1 | $\mathbf{Z}_1 = \mathbf{X}\mathbf{W}$ | matmul | $B \times D_{out}$ |
| 2 | $\mathbf{Z}_2 = \mathbf{Z}_1 + \mathbf{b}$ | add (broadcast) | $B \times D_{out}$ |
| 3 | $\mathbf{Z}_3 = \text{ReLU}(\mathbf{Z}_2)$ | element-wise | $B \times D_{out}$ |
| 4 | $\mathbf{D} = \mathbf{Z}_3 - \mathbf{T}$ | subtract | $B \times D_{out}$ |
| 5 | $\mathbf{S} = \mathbf{D}^2$ | element-wise | $B \times D_{out}$ |
| 6 | $L = \frac{1}{2B}\text{sum}(\mathbf{S})$ | reduction | scalar |

**Backward pass** (reverse order):

**Step 6 backward**: $\bar{\mathbf{S}} = \frac{\partial L}{\partial \mathbf{S}} = \frac{1}{2B} \cdot \mathbf{1}_{B \times D_{out}}$

**Step 5 backward**: $\bar{\mathbf{D}} = \bar{\mathbf{S}} \odot 2\mathbf{D} = \frac{1}{B}\mathbf{D}$

**Step 4 backward**: $\bar{\mathbf{Z}}_3 = \bar{\mathbf{D}}$ (and $\bar{\mathbf{T}} = -\bar{\mathbf{D}}$, usually not needed)

**Step 3 backward**: $\bar{\mathbf{Z}}_2 = \bar{\mathbf{Z}}_3 \odot \mathbb{1}[\mathbf{Z}_2 > 0]$

**Step 2 backward**: $\bar{\mathbf{Z}}_1 = \bar{\mathbf{Z}}_2$ (addition passes through)
$\bar{\mathbf{b}} = \sum_{i=1}^{B} (\bar{\mathbf{Z}}_2)_i$ (sum over batch, undo broadcast)

**Step 1 backward**: $\bar{\mathbf{W}} = \mathbf{X}^T \bar{\mathbf{Z}}_1$ (matmul rule: left factor transposed × upstream)
$\bar{\mathbf{X}} = \bar{\mathbf{Z}}_1 \mathbf{W}^T$ (matmul rule: upstream × right factor transposed)

### Complete numerical example

```python
import torch

torch.manual_seed(42)

B, D_in, D_out = 2, 3, 2
X = torch.randn(B, D_in)
W = torch.randn(D_in, D_out, requires_grad=True)
b = torch.randn(D_out, requires_grad=True)
T = torch.randn(B, D_out)

# Forward
Z1 = X @ W                    # (2, 2)
Z2 = Z1 + b                   # (2, 2) — b broadcast
Z3 = torch.relu(Z2)           # (2, 2)
D = Z3 - T                    # (2, 2)
S = D ** 2                    # (2, 2)
L = S.sum() / (2 * B)         # scalar

# Backward (automatic)
L.backward()

# Manual backward
with torch.no_grad():
    grad_S = torch.ones(B, D_out) / (2 * B)
    grad_D = grad_S * 2 * D
    grad_Z3 = grad_D
    grad_Z2 = grad_Z3 * (Z2 > 0).float()
    grad_Z1 = grad_Z2
    grad_b_manual = grad_Z2.sum(dim=0)
    grad_W_manual = X.T @ grad_Z1
    grad_X_manual = grad_Z1 @ W.T

print(f"W.grad matches: {torch.allclose(W.grad, grad_W_manual)}")       # True
print(f"b.grad matches: {torch.allclose(b.grad, grad_b_manual)}")       # True
```

---

## 4.7 The General Recipe: How to Derive Any Gradient

Given an operation $\mathbf{y} = f(\mathbf{x})$ and upstream $\bar{\mathbf{y}} = \frac{\partial L}{\partial \mathbf{y}}$:

### Method 1: Element-wise derivation (always works)

1. Write $y_i = f_i(x_1, \ldots, x_n)$ explicitly
2. Compute $\frac{\partial y_i}{\partial x_j}$ (Jacobian elements)
3. Apply: $\bar{x}_j = \sum_i \bar{y}_i \frac{\partial y_i}{\partial x_j}$
4. Recognize the result as a matrix operation (matmul, Hadamard, etc.)

### Method 2: Differential approach (more elegant, used in Document 7)

1. Write the differential: $d\mathbf{Y} = ?$ in terms of $d\mathbf{A}$, $d\mathbf{B}$, etc.
2. For the loss: note that $dL = \text{tr}(\bar{\mathbf{Y}}^T d\mathbf{Y})$ (trace inner product)
3. Manipulate using trace properties to isolate $d\mathbf{A}$
4. Read off the gradient from the coefficient

This second method is much more powerful and is covered in detail in Documents 7.

### Method 3: Shape matching + pattern recognition (quick sanity check)

For $\mathbf{Y} = f(\mathbf{A}, \mathbf{B})$:
- $\bar{\mathbf{A}}$ must have shape of $\mathbf{A}$
- The formula must involve $\bar{\mathbf{Y}}$ and $\mathbf{B}$ (the "other" input)
- Find the arrangement that makes shapes match

Example: $\mathbf{Y}_{m \times p} = \mathbf{A}_{m \times n} \mathbf{B}_{n \times p}$
- $\bar{\mathbf{A}}$ is $m \times n$, need to combine $\bar{\mathbf{Y}}_{m \times p}$ and $\mathbf{B}_{n \times p}$
- Only $\bar{\mathbf{Y}} \mathbf{B}^T = (m \times p)(p \times n) = m \times n$ ✓

---

## 4.8 Operations Summary Table

| Operation | Forward | Backward ($\bar{\mathbf{x}}$ given $\bar{\mathbf{y}}$) |
|-----------|---------|--------|
| $\mathbf{Y} = \mathbf{A}\mathbf{B}$ | Matmul | $\bar{\mathbf{A}} = \bar{\mathbf{Y}}\mathbf{B}^T$, $\bar{\mathbf{B}} = \mathbf{A}^T\bar{\mathbf{Y}}$ |
| $\mathbf{y} = f(\mathbf{x})$ | Element-wise | $\bar{\mathbf{x}} = \bar{\mathbf{y}} \odot f'(\mathbf{x})$ |
| $\mathbf{y} = \mathbf{x} + \mathbf{z}$ | Addition | $\bar{\mathbf{x}} = \bar{\mathbf{y}}$, $\bar{\mathbf{z}} = \bar{\mathbf{y}}$ |
| $\mathbf{y} = \mathbf{x} \odot \mathbf{z}$ | Hadamard | $\bar{\mathbf{x}} = \bar{\mathbf{y}} \odot \mathbf{z}$, $\bar{\mathbf{z}} = \bar{\mathbf{y}} \odot \mathbf{x}$ |
| $y = \sum_i x_i$ | Sum | $\bar{\mathbf{x}} = \bar{y} \cdot \mathbf{1}$ |
| $y = \frac{1}{n}\sum_i x_i$ | Mean | $\bar{\mathbf{x}} = \frac{\bar{y}}{n} \cdot \mathbf{1}$ |
| $\mathbf{y} = \mathbf{x} + \mathbf{b}$ (broadcast) | Add bias | $\bar{\mathbf{x}} = \bar{\mathbf{y}}$, $\bar{\mathbf{b}} = \text{sum}(\bar{\mathbf{y}}, \text{broadcast dims})$ |
| $y = \mathbf{x}^T\mathbf{x}$ | Dot product | $\bar{\mathbf{x}} = 2\bar{y}\mathbf{x}$ |
| $y = x^n$ | Power | $\bar{x} = \bar{y} \cdot nx^{n-1}$ |

---

**Next**: Document 5 tackles convolution — a structured operation whose Jacobian is sparse and can be reduced to matmul via the im2col trick.
