# 12. Backpropagation Through the Transformer: Deriving Every Gradient

## Motivation

Documents 9-11 built the complete transformer architecture forward. Now we go backward. This document derives the **gradient of every operation** in the attention mechanism and transformer block, using the tools from Documents 3-7. This is what framework authors compute when implementing `.backward()` for attention, and what you need to understand to implement custom attention variants, optimize memory usage (FlashAttention), or debug gradient issues.

We proceed node by node through the computation graph, deriving each gradient using both element-wise analysis (Doc 4) and the trace trick (Doc 7).

---

## 12.1 The Plan: Backward Through the Computation Graph

Recall the single-head attention (no batch, for clarity — 3D/4D extensions are mechanical):

$$\mathbf{Q} = \mathbf{X}\mathbf{W}_Q \quad (1)$$
$$\mathbf{K} = \mathbf{X}\mathbf{W}_K \quad (2)$$
$$\mathbf{V} = \mathbf{X}\mathbf{W}_V \quad (3)$$
$$\mathbf{S} = \mathbf{Q}\mathbf{K}^T \quad (4)$$
$$\mathbf{S}' = \mathbf{S}/\sqrt{d_k} \quad (5)$$
$$\mathbf{A} = \text{softmax}_\text{row}(\mathbf{S}') \quad (6)$$
$$\mathbf{Y} = \mathbf{A}\mathbf{V} \quad (7)$$

We are given $\bar{\mathbf{Y}} = \frac{\partial L}{\partial \mathbf{Y}} \in \mathbb{R}^{n \times d_v}$ (the upstream gradient from whatever comes after attention). We need to compute gradients with respect to everything: $\bar{\mathbf{A}}, \bar{\mathbf{V}}, \bar{\mathbf{S}'}, \bar{\mathbf{S}}, \bar{\mathbf{Q}}, \bar{\mathbf{K}}, \bar{\mathbf{W}}_Q, \bar{\mathbf{W}}_K, \bar{\mathbf{W}}_V, \bar{\mathbf{X}}$.

**Strategy**: Walk backward through nodes (7) → (6) → (5) → (4) → (1,2,3), applying the gradient rule at each step.

---

## 12.2 Node 7: $\mathbf{Y} = \mathbf{A}\mathbf{V}$ (Matmul)

This is a standard matmul. From Doc 4:

$$\bar{\mathbf{A}} = \bar{\mathbf{Y}} \mathbf{V}^T \in \mathbb{R}^{n \times n}$$

$$\bar{\mathbf{V}} = \mathbf{A}^T \bar{\mathbf{Y}} \in \mathbb{R}^{n \times d_v}$$

### Shape verification

| Quantity | Shape | Check |
|----------|-------|-------|
| $\bar{\mathbf{Y}}$ | $n \times d_v$ | Given |
| $\mathbf{V}^T$ | $d_v \times n$ | |
| $\bar{\mathbf{Y}} \mathbf{V}^T$ | $n \times n$ | ✓ matches $\mathbf{A}$ |
| $\mathbf{A}^T$ | $n \times n$ | |
| $\mathbf{A}^T \bar{\mathbf{Y}}$ | $n \times d_v$ | ✓ matches $\mathbf{V}$ |

### Trace trick derivation (alternative)

$$dL = \text{tr}(\bar{\mathbf{Y}}^T d\mathbf{Y}) = \text{tr}(\bar{\mathbf{Y}}^T (d\mathbf{A} \cdot \mathbf{V} + \mathbf{A} \cdot d\mathbf{V}))$$

For $\bar{\mathbf{A}}$ (treating $\mathbf{V}$ as constant):
$$dL = \text{tr}(\bar{\mathbf{Y}}^T d\mathbf{A} \cdot \mathbf{V}) = \text{tr}(\mathbf{V}\bar{\mathbf{Y}}^T d\mathbf{A}) = \text{tr}((\bar{\mathbf{Y}}\mathbf{V}^T)^T d\mathbf{A})$$

So $\bar{\mathbf{A}} = \bar{\mathbf{Y}}\mathbf{V}^T$. ✓

---

## 12.3 Node 6: $\mathbf{A} = \text{softmax}_\text{row}(\mathbf{S}')$ (Row-wise Softmax)

This is the most complex gradient in the attention mechanism. From Doc 5, softmax applied to a vector $\mathbf{z}$ giving $\mathbf{a} = \text{softmax}(\mathbf{z})$ has Jacobian:

$$\frac{\partial a_i}{\partial z_j} = a_i(\delta_{ij} - a_j)$$

Or in matrix form: $\mathbf{J}_\text{softmax} = \text{diag}(\mathbf{a}) - \mathbf{a}\mathbf{a}^T$

### VJP for one row

For a single row $i$ of $\mathbf{S}'$, the softmax VJP is:

$$\bar{S}'_{i,:} = \bar{A}_{i,:} \cdot \mathbf{J}_i = \bar{A}_{i,:} \left(\text{diag}(\mathbf{a}_i) - \mathbf{a}_i \mathbf{a}_i^T\right)$$

where $\mathbf{a}_i = \mathbf{A}_{i,:}$ is the $i$-th row of $\mathbf{A}$ (a probability vector).

Expanding:
$$\bar{S}'_{ij} = \sum_k \bar{A}_{ik} (A_{ik} \delta_{kj} - A_{ik} A_{ij})$$

$$= \bar{A}_{ij} A_{ij} - A_{ij} \sum_k \bar{A}_{ik} A_{ik}$$

$$= A_{ij}(\bar{A}_{ij} - \underbrace{\sum_k \bar{A}_{ik} A_{ik}}_{\text{scalar per row } i})$$

### Defining the row-wise dot product

Let $c_i = \sum_k \bar{A}_{ik} A_{ik}$ — this is the dot product of row $i$ of $\bar{\mathbf{A}}$ with row $i$ of $\mathbf{A}$.

In matrix notation:
$$\mathbf{c} = \text{row\_sum}(\bar{\mathbf{A}} \odot \mathbf{A}) \in \mathbb{R}^n$$

where $\text{row\_sum}$ sums each row to a scalar.

### Final formula

$$\bar{\mathbf{S}'} = \mathbf{A} \odot (\bar{\mathbf{A}} - \mathbf{c}\mathbf{1}^T)$$

or equivalently (expanding $\mathbf{c}\mathbf{1}^T$, which broadcasts $c_i$ across all columns of row $i$):

$$\boxed{\bar{\mathbf{S}'} = \mathbf{A} \odot \left(\bar{\mathbf{A}} - (\bar{\mathbf{A}} \odot \mathbf{A})\mathbf{1}_n\mathbf{1}_n^T \right)}$$

A cleaner way to write this (where broadcasting handles the outer product):

$$\bar{S}'_{ij} = A_{ij}\left(\bar{A}_{ij} - \sum_k \bar{A}_{ik} A_{ik}\right)$$

### Implementation

```python
def softmax_backward(grad_A, A):
    """
    grad_A: (n, n) — upstream gradient dL/dA
    A: (n, n) — attention weights (output of softmax)
    Returns: grad_S_prime (n, n) — dL/dS'
    """
    # Row-wise dot product of grad_A and A
    c = (grad_A * A).sum(dim=-1, keepdim=True)  # (n, 1)
    
    # Softmax VJP
    grad_S_prime = A * (grad_A - c)  # (n, n)
    
    return grad_S_prime
```

The formula is `A * (grad_A - c)` — element-wise multiply of $\mathbf{A}$ with the "centered" upstream gradient (subtract the mean weighted by attention weights).

### Why this is efficient

The naive approach would form the $n \times n$ Jacobian for each row and multiply — $O(n^3)$. Our formula is $O(n^2)$: one element-wise multiply, one row reduction, and one more element-wise multiply.

---

## 12.4 Node 5: $\mathbf{S}' = \mathbf{S}/\sqrt{d_k}$ (Scalar Division)

This is element-wise division by a constant. The gradient simply passes through:

$$\bar{\mathbf{S}} = \bar{\mathbf{S}'} / \sqrt{d_k}$$

**Why?** The function $f(S_{ij}) = S_{ij}/\sqrt{d_k}$ has derivative $f'(S_{ij}) = 1/\sqrt{d_k}$. By the element-wise gradient rule (Doc 4):

$$\bar{S}_{ij} = \bar{S}'_{ij} \cdot \frac{1}{\sqrt{d_k}}$$

---

## 12.5 Node 4: $\mathbf{S} = \mathbf{Q}\mathbf{K}^T$ (Matmul with Transpose)

This requires careful handling. The standard matmul gradient from Doc 4 gives us:

For $\mathbf{S} = \mathbf{Q}\mathbf{K}^T$, treat it as $\mathbf{S} = \mathbf{Q}\mathbf{M}$ where $\mathbf{M} = \mathbf{K}^T$:

$$\bar{\mathbf{Q}} = \bar{\mathbf{S}} \mathbf{M}^T = \bar{\mathbf{S}} (\mathbf{K}^T)^T = \bar{\mathbf{S}} \mathbf{K}$$

$$\bar{\mathbf{M}} = \mathbf{Q}^T \bar{\mathbf{S}}$$

But $\bar{\mathbf{M}}$ is the gradient w.r.t. $\mathbf{K}^T$, and we need the gradient w.r.t. $\mathbf{K}$. Since the gradient of the transpose of a matrix w.r.t. the original is just the transpose of the upstream gradient:

$$\bar{\mathbf{K}} = \bar{\mathbf{M}}^T = (\mathbf{Q}^T \bar{\mathbf{S}})^T = \bar{\mathbf{S}}^T \mathbf{Q}$$

### Summary for $\mathbf{S} = \mathbf{Q}\mathbf{K}^T$

$$\boxed{\bar{\mathbf{Q}} = \bar{\mathbf{S}} \mathbf{K}} \in \mathbb{R}^{n \times d_k}$$

$$\boxed{\bar{\mathbf{K}} = \bar{\mathbf{S}}^T \mathbf{Q}} \in \mathbb{R}^{n \times d_k}$$

### Shape verification

| Quantity | Shape | Check |
|----------|-------|-------|
| $\bar{\mathbf{S}}$ | $n \times n$ | |
| $\mathbf{K}$ | $n \times d_k$ | |
| $\bar{\mathbf{S}} \mathbf{K}$ | $n \times d_k$ | ✓ matches $\mathbf{Q}$ |
| $\bar{\mathbf{S}}^T$ | $n \times n$ | |
| $\bar{\mathbf{S}}^T \mathbf{Q}$ | $n \times d_k$ | ✓ matches $\mathbf{K}$ |

### Trace trick derivation (verification)

$$dL = \text{tr}(\bar{\mathbf{S}}^T d\mathbf{S}) = \text{tr}(\bar{\mathbf{S}}^T d(\mathbf{Q}\mathbf{K}^T))$$

For $\bar{\mathbf{Q}}$ (treating $\mathbf{K}$ as constant):
$$dL = \text{tr}(\bar{\mathbf{S}}^T d\mathbf{Q} \cdot \mathbf{K}^T) = \text{tr}(\mathbf{K}^T \bar{\mathbf{S}}^T d\mathbf{Q}) = \text{tr}((\bar{\mathbf{S}}\mathbf{K})^T d\mathbf{Q})$$

So $\bar{\mathbf{Q}} = \bar{\mathbf{S}}\mathbf{K}$. ✓

For $\bar{\mathbf{K}}$ (treating $\mathbf{Q}$ as constant):
$$dL = \text{tr}(\bar{\mathbf{S}}^T \mathbf{Q} \cdot (d\mathbf{K})^T) = \text{tr}((d\mathbf{K})^T (\bar{\mathbf{S}}^T \mathbf{Q}))$$

Hmm, that has the transpose on $d\mathbf{K}$. Using $\text{tr}(\mathbf{A}^T \mathbf{B}) = \text{tr}(\mathbf{B}^T \mathbf{A})$:

$$dL = \text{tr}((\bar{\mathbf{S}}^T \mathbf{Q})^T d\mathbf{K}) = \text{tr}(\mathbf{Q}^T \bar{\mathbf{S}} \, d\mathbf{K})$$

Wait — let's be more careful. We have $(d\mathbf{K})^T$, so $d\mathbf{S} = \mathbf{Q}(d\mathbf{K})^T$ and:

$$dL = \text{tr}(\bar{\mathbf{S}}^T \mathbf{Q}(d\mathbf{K})^T)$$

Using cyclic property: $= \text{tr}((d\mathbf{K})^T \bar{\mathbf{S}}^T \mathbf{Q})$

Using $\text{tr}(\mathbf{C}^T \mathbf{D}) = \text{tr}(\mathbf{D}^T \mathbf{C})$: $= \text{tr}(\mathbf{Q}^T \bar{\mathbf{S}} \, d\mathbf{K})$

This doesn't have the right form $\text{tr}(\bar{\mathbf{K}}^T d\mathbf{K})$. We need:

$$\text{tr}(\mathbf{Q}^T \bar{\mathbf{S}} d\mathbf{K}) = \text{tr}((\bar{\mathbf{S}}^T \mathbf{Q})^T d\mathbf{K})$$

So $\bar{\mathbf{K}} = \bar{\mathbf{S}}^T \mathbf{Q}$. ✓

---

## 12.6 Nodes 1-3: $\mathbf{Q} = \mathbf{X}\mathbf{W}_Q$ (Linear Projection Gradients)

From Doc 4, for $\mathbf{Q} = \mathbf{X}\mathbf{W}_Q$:

$$\bar{\mathbf{X}}_\text{from Q} = \bar{\mathbf{Q}} \mathbf{W}_Q^T$$

$$\bar{\mathbf{W}}_Q = \mathbf{X}^T \bar{\mathbf{Q}}$$

Similarly for $\mathbf{K}$ and $\mathbf{V}$:

$$\bar{\mathbf{X}}_\text{from K} = \bar{\mathbf{K}} \mathbf{W}_K^T, \qquad \bar{\mathbf{W}}_K = \mathbf{X}^T \bar{\mathbf{K}}$$

$$\bar{\mathbf{X}}_\text{from V} = \bar{\mathbf{V}} \mathbf{W}_V^T, \qquad \bar{\mathbf{W}}_V = \mathbf{X}^T \bar{\mathbf{V}}$$

### The critical step: accumulating gradients from multiple paths

$\mathbf{X}$ is used in THREE places (Q, K, V projections). By the multivariate chain rule (Doc 3), the total gradient is the **sum** of contributions from all paths:

$$\boxed{\bar{\mathbf{X}}_\text{attention} = \bar{\mathbf{Q}} \mathbf{W}_Q^T + \bar{\mathbf{K}} \mathbf{W}_K^T + \bar{\mathbf{V}} \mathbf{W}_V^T}$$

This is the **key point** about computation graphs with fan-out. When a node feeds into multiple downstream nodes, its gradient is the sum of all incoming gradients.

---

## 12.7 Complete Backward Pass: All Formulas Together

Given: $\bar{\mathbf{Y}} \in \mathbb{R}^{n \times d_v}$

**Step 1** — Through $\mathbf{Y} = \mathbf{A}\mathbf{V}$:
$$\bar{\mathbf{A}} = \bar{\mathbf{Y}}\mathbf{V}^T, \qquad \bar{\mathbf{V}} = \mathbf{A}^T \bar{\mathbf{Y}}$$

**Step 2** — Through softmax:
$$c_i = \sum_j \bar{A}_{ij} A_{ij}, \qquad \bar{\mathbf{S}'} = \mathbf{A} \odot (\bar{\mathbf{A}} - \mathbf{c})$$

where $\mathbf{c}$ is broadcast along columns.

**Step 3** — Through scaling:
$$\bar{\mathbf{S}} = \bar{\mathbf{S}'} / \sqrt{d_k}$$

**Step 4** — Through $\mathbf{S} = \mathbf{Q}\mathbf{K}^T$:
$$\bar{\mathbf{Q}} = \bar{\mathbf{S}}\mathbf{K}, \qquad \bar{\mathbf{K}} = \bar{\mathbf{S}}^T \mathbf{Q}$$

**Step 5** — Through linear projections:
$$\bar{\mathbf{W}}_Q = \mathbf{X}^T \bar{\mathbf{Q}}, \quad \bar{\mathbf{W}}_K = \mathbf{X}^T \bar{\mathbf{K}}, \quad \bar{\mathbf{W}}_V = \mathbf{X}^T \bar{\mathbf{V}}$$
$$\bar{\mathbf{X}} = \bar{\mathbf{Q}}\mathbf{W}_Q^T + \bar{\mathbf{K}}\mathbf{W}_K^T + \bar{\mathbf{V}}\mathbf{W}_V^T$$

### Implementation

```python
import torch
import torch.nn.functional as F

def attention_forward(X, W_Q, W_K, W_V):
    """Forward pass, saving intermediates for backward."""
    Q = X @ W_Q
    K = X @ W_K
    V = X @ W_V
    
    d_k = Q.shape[-1]
    S = Q @ K.T
    S_prime = S / (d_k ** 0.5)
    A = F.softmax(S_prime, dim=-1)
    Y = A @ V
    
    # Return output and cache for backward
    cache = (X, W_Q, W_K, W_V, Q, K, V, S, S_prime, A)
    return Y, cache

def attention_backward(grad_Y, cache):
    """Hand-derived backward pass for attention."""
    X, W_Q, W_K, W_V, Q, K, V, S, S_prime, A = cache
    d_k = Q.shape[-1]
    
    # Step 1: Through Y = A @ V
    grad_A = grad_Y @ V.T          # (n, n)
    grad_V = A.T @ grad_Y          # (n, d_v)
    
    # Step 2: Through softmax
    # c_i = sum_j grad_A[i,j] * A[i,j]
    c = (grad_A * A).sum(dim=-1, keepdim=True)  # (n, 1)
    grad_S_prime = A * (grad_A - c)              # (n, n)
    
    # Step 3: Through scaling
    grad_S = grad_S_prime / (d_k ** 0.5)        # (n, n)
    
    # Step 4: Through S = Q @ K.T
    grad_Q = grad_S @ K            # (n, d_k)
    grad_K = grad_S.T @ Q          # (n, d_k)
    
    # Step 5: Through linear projections
    grad_W_Q = X.T @ grad_Q        # (d, d_k)
    grad_W_K = X.T @ grad_K        # (d, d_k)
    grad_W_V = X.T @ grad_V        # (d, d_v)
    
    # Accumulate gradients for X from all three paths
    grad_X = grad_Q @ W_Q.T + grad_K @ W_K.T + grad_V @ W_V.T  # (n, d)
    
    return grad_X, grad_W_Q, grad_W_K, grad_W_V

# === Numerical verification ===
torch.manual_seed(42)
n, d, d_k = 4, 8, 8

X = torch.randn(n, d, dtype=torch.float64, requires_grad=True)
W_Q = torch.randn(d, d_k, dtype=torch.float64, requires_grad=True)
W_K = torch.randn(d, d_k, dtype=torch.float64, requires_grad=True)
W_V = torch.randn(d, d_k, dtype=torch.float64, requires_grad=True)

# Forward
Y, cache = attention_forward(X, W_Q, W_K, W_V)
L = (Y ** 2).sum()  # Simple scalar loss

# Autograd backward
L.backward()

# Our manual backward
grad_Y = 2 * Y.detach()  # dL/dY = 2Y for L = sum(Y^2)
grad_X_manual, grad_WQ_manual, grad_WK_manual, grad_WV_manual = attention_backward(
    grad_Y, cache
)

# Compare
print("grad_X  matches:", torch.allclose(X.grad, grad_X_manual, atol=1e-10))
print("grad_WQ matches:", torch.allclose(W_Q.grad, grad_WQ_manual, atol=1e-10))
print("grad_WK matches:", torch.allclose(W_K.grad, grad_WK_manual, atol=1e-10))
print("grad_WV matches:", torch.allclose(W_V.grad, grad_WV_manual, atol=1e-10))
```

---

## 12.8 Backprop Through LayerNorm

### The computation graph

$$\mu = \frac{1}{d}\sum_j x_j, \quad \sigma^2 = \frac{1}{d}\sum_j(x_j - \mu)^2, \quad \hat{x}_j = \frac{x_j - \mu}{\sqrt{\sigma^2 + \epsilon}}, \quad y_j = \gamma_j \hat{x}_j + \beta_j$$

For a single token $\mathbf{x} \in \mathbb{R}^d$ (the batch and sequence dimensions are spectators).

### Step-by-step backward

Given $\bar{\mathbf{y}} \in \mathbb{R}^d$:

**Through affine**: $\bar{\hat{x}}_j = \bar{y}_j \gamma_j$, and $\bar{\gamma}_j = \bar{y}_j \hat{x}_j$, $\bar{\beta}_j = \bar{y}_j$.

**Through normalization**: Let $s = (\sigma^2 + \epsilon)^{-1/2}$ (a scalar).

$$\hat{x}_j = (x_j - \mu) \cdot s$$

This has three paths: through $x_j$ directly, through $\mu$, and through $s$ (which depends on $\sigma^2$, which depends on $\mu$).

The full derivation (using the trace trick on the vectorized form) gives:

$$\bar{\mathbf{x}} = \frac{1}{\sqrt{\sigma^2 + \epsilon}} \left(\bar{\hat{\mathbf{x}}} - \frac{1}{d}\sum_j \bar{\hat{x}}_j - \frac{\hat{\mathbf{x}}}{d}\sum_j \bar{\hat{x}}_j \hat{x}_j\right)$$

### Detailed derivation

Let $\mathbf{c} = \mathbf{x} - \mu\mathbf{1}$ (centered input), $s = 1/\sqrt{\sigma^2 + \epsilon}$.

Then $\hat{\mathbf{x}} = s \cdot \mathbf{c}$ and $\sigma^2 = \frac{1}{d}\mathbf{c}^T\mathbf{c}$.

**Gradient w.r.t. $\mathbf{c}$** (from $\hat{\mathbf{x}} = s \cdot \mathbf{c}$ where $s$ also depends on $\mathbf{c}$):

Using the product rule: $d\hat{\mathbf{x}} = ds \cdot \mathbf{c} + s \cdot d\mathbf{c}$

From $s = (\frac{1}{d}\mathbf{c}^T\mathbf{c} + \epsilon)^{-1/2}$:

$$ds = -\frac{1}{2}(\sigma^2 + \epsilon)^{-3/2} \cdot d\sigma^2 = -\frac{s^3}{2} \cdot \frac{2}{d}\mathbf{c}^T d\mathbf{c} = -\frac{s^3}{d}\mathbf{c}^T d\mathbf{c}$$

So:
$$d\hat{\mathbf{x}} = -\frac{s^3}{d}(\mathbf{c}^T d\mathbf{c})\mathbf{c} + s \, d\mathbf{c}$$

The VJP:
$$dL = \bar{\hat{\mathbf{x}}}^T d\hat{\mathbf{x}} = -\frac{s^3}{d}(\bar{\hat{\mathbf{x}}}^T \mathbf{c})(\mathbf{c}^T d\mathbf{c}) + s \, \bar{\hat{\mathbf{x}}}^T d\mathbf{c}$$

$$= \left(s \bar{\hat{\mathbf{x}}} - \frac{s^3}{d}(\bar{\hat{\mathbf{x}}}^T \mathbf{c})\mathbf{c}\right)^T d\mathbf{c}$$

So $\bar{\mathbf{c}} = s\bar{\hat{\mathbf{x}}} - \frac{s^3}{d}(\bar{\hat{\mathbf{x}}}^T \mathbf{c})\mathbf{c}$.

Using $\hat{\mathbf{x}} = s\mathbf{c}$:

$$\bar{\mathbf{c}} = s\bar{\hat{\mathbf{x}}} - \frac{s}{d}(\bar{\hat{\mathbf{x}}}^T \hat{\mathbf{x}})\hat{\mathbf{x}}$$

**Gradient w.r.t. $\mathbf{x}$** (from $\mathbf{c} = \mathbf{x} - \mu\mathbf{1}$ where $\mu = \frac{1}{d}\mathbf{1}^T\mathbf{x}$):

$$d\mathbf{c} = d\mathbf{x} - d\mu \cdot \mathbf{1} = d\mathbf{x} - \frac{1}{d}\mathbf{1}\mathbf{1}^T d\mathbf{x} = (\mathbf{I} - \frac{1}{d}\mathbf{1}\mathbf{1}^T)d\mathbf{x}$$

So:
$$\bar{\mathbf{x}} = (\mathbf{I} - \frac{1}{d}\mathbf{1}\mathbf{1}^T)^T \bar{\mathbf{c}} = \bar{\mathbf{c}} - \frac{1}{d}\mathbf{1}(\mathbf{1}^T\bar{\mathbf{c}})$$

$$= \bar{\mathbf{c}} - \frac{1}{d}\sum_j \bar{c}_j$$

Combining everything:

$$\boxed{\bar{\mathbf{x}} = \frac{s}{d}\left(d \cdot \bar{\hat{\mathbf{x}}} - (\bar{\hat{\mathbf{x}}}^T \hat{\mathbf{x}})\hat{\mathbf{x}} - (\mathbf{1}^T \bar{\hat{\mathbf{x}}})\mathbf{1}\right)}$$

where $s = 1/\sqrt{\sigma^2 + \epsilon}$ and $\bar{\hat{\mathbf{x}}} = \bar{\mathbf{y}} \odot \boldsymbol{\gamma}$.

### Implementation

```python
def layernorm_backward(grad_y, x, gamma, eps=1e-5):
    """
    Backward pass for LayerNorm, derived from scratch.
    
    grad_y: (d,) — upstream gradient
    x: (d,) — input to LayerNorm
    gamma: (d,) — scale parameter
    """
    d = x.shape[-1]
    
    # Recompute forward quantities
    mu = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    s = 1.0 / torch.sqrt(var + eps)
    x_hat = (x - mu) * s
    
    # Gradient through affine
    grad_x_hat = grad_y * gamma          # (d,)
    grad_gamma = (grad_y * x_hat).sum(dim=0)  # reduce over batch/seq if present
    grad_beta = grad_y.sum(dim=0)
    
    # Gradient through normalization
    # Using our derived formula: (s/d) * (d * grad_x_hat - (grad_x_hat · x_hat) * x_hat - sum(grad_x_hat))
    dot = (grad_x_hat * x_hat).sum(dim=-1, keepdim=True)  # scalar per token
    grad_x = s / d * (d * grad_x_hat - dot * x_hat - grad_x_hat.sum(dim=-1, keepdim=True))
    
    return grad_x, grad_gamma, grad_beta
```

---

## 12.9 Backprop Through the Residual Connection

### For $\mathbf{Z} = \mathbf{X} + f(\mathbf{X})$

The gradient splits into two paths:

$$\bar{\mathbf{X}} = \bar{\mathbf{Z}} + \bar{\mathbf{Z}} \cdot \frac{\partial f(\mathbf{X})}{\partial \mathbf{X}}$$

The first term ($\bar{\mathbf{Z}}$) is the **skip connection gradient** — it passes through unchanged, providing the gradient highway that prevents vanishing gradients.

In code, this is trivially:

```python
# Forward: Z = X + sublayer(LayerNorm(X))
# Backward:
grad_X = grad_Z  # from the identity path
grad_sublayer_out = grad_Z  # same gradient goes to sublayer
# Then backprop through sublayer and LayerNorm, add result to grad_X
grad_X = grad_X + grad_through_sublayer_and_norm
```

---

## 12.10 Backprop Through the FFN

### Forward: $\mathbf{H} = \text{GELU}(\mathbf{Y}_\text{norm}\mathbf{W}_1^T + \mathbf{b}_1)$, then $\mathbf{Z}_\text{ffn} = \mathbf{H}\mathbf{W}_2^T + \mathbf{b}_2$

### Backward (from output gradient $\bar{\mathbf{Z}}_\text{ffn}$):

**Through second linear layer** ($\mathbf{Z}_\text{ffn} = \mathbf{H}\mathbf{W}_2^T + \mathbf{b}_2$):

$$\bar{\mathbf{H}} = \bar{\mathbf{Z}}_\text{ffn} \mathbf{W}_2, \quad \bar{\mathbf{W}}_2 = \bar{\mathbf{Z}}_\text{ffn}^T \mathbf{H}, \quad \bar{\mathbf{b}}_2 = \bar{\mathbf{Z}}_\text{ffn}\text{.sum(dim=0..seq)}$$

**Through GELU** ($\mathbf{H} = \text{GELU}(\mathbf{P})$):

GELU is approximately $x \cdot \Phi(x)$ where $\Phi$ is the Gaussian CDF. Its gradient:

$$\text{GELU}'(x) = \Phi(x) + x \phi(x)$$

where $\phi(x)$ is the Gaussian PDF. This is element-wise:

$$\bar{\mathbf{P}} = \bar{\mathbf{H}} \odot \text{GELU}'(\mathbf{P})$$

**Through first linear layer** ($\mathbf{P} = \mathbf{Y}_\text{norm}\mathbf{W}_1^T + \mathbf{b}_1$):

$$\bar{\mathbf{Y}}_\text{norm} = \bar{\mathbf{P}} \mathbf{W}_1, \quad \bar{\mathbf{W}}_1 = \bar{\mathbf{P}}^T \mathbf{Y}_\text{norm}, \quad \bar{\mathbf{b}}_1 = \bar{\mathbf{P}}\text{.sum(dim=0..seq)}$$

---

## 12.11 Extending to 3D/4D: Batched Backward

All the 2D gradients above extend to 3D/4D mechanically:

### Rule: matmul gradients in higher dimensions

For $\mathbf{Y} = \mathbf{A} \mathbin{@} \mathbf{B}$ with $\mathbf{A} \in \mathbb{R}^{B \times h \times m \times k}$:

$$\bar{\mathbf{A}} = \bar{\mathbf{Y}} \mathbin{@} \mathbf{B}.\text{transpose}(-2,-1)$$

$$\bar{\mathbf{B}} = \mathbf{A}.\text{transpose}(-2,-1) \mathbin{@} \bar{\mathbf{Y}}$$

The batch dimensions $(B, h)$ are spectators — the gradient formula operates on the last two dimensions identically to the 2D case.

### Rule: softmax backward in higher dimensions

The softmax backward formula applies independently to each "row" (last dimension):

```python
# For S_prime of shape (B, h, n, n):
c = (grad_A * A).sum(dim=-1, keepdim=True)  # (B, h, n, 1)
grad_S_prime = A * (grad_A - c)              # (B, h, n, n)
```

### Rule: gradient w.r.t. a broadcast (shared) weight

When $\mathbf{W}_Q \in \mathbb{R}^{d \times d_k}$ is shared across the batch:

$$\mathbf{Q}[b] = \mathbf{X}[b] \mathbf{W}_Q \quad \text{for each } b$$

The gradient w.r.t. $\mathbf{W}_Q$ **sums over the batch**:

$$\bar{\mathbf{W}}_Q = \sum_{b=1}^B \mathbf{X}[b]^T \bar{\mathbf{Q}}[b]$$

In einsum: `grad_W_Q = torch.einsum('bni,bnj->ij', X, grad_Q)`

Or equivalently: `grad_W_Q = X.reshape(-1, d).T @ grad_Q.reshape(-1, d_k)`

---

## 12.12 Memory Analysis: What Must Be Saved for Backward?

The attention backward requires these intermediates from the forward pass:

| Tensor | Shape | Purpose |
|--------|-------|---------|
| $\mathbf{X}$ | $B \times n \times d$ | Linear projection gradients |
| $\mathbf{Q}$ | $B \times h \times n \times d_k$ | Score matmul backward |
| $\mathbf{K}$ | $B \times h \times n \times d_k$ | Score matmul backward |
| $\mathbf{V}$ | $B \times h \times n \times d_v$ | Output matmul backward |
| $\mathbf{A}$ | $B \times h \times n \times n$ | Softmax backward, output matmul backward |

The attention matrix $\mathbf{A}$ dominates when $n > d$: it's $O(Bhn^2)$ vs $O(Bhnd_k)$ for Q, K, V.

### FlashAttention insight

FlashAttention (Dao et al., 2022) avoids storing $\mathbf{A}$ by **recomputing** it during the backward pass from $\mathbf{Q}$, $\mathbf{K}$, and the softmax normalization constants. This trades compute for memory — the recomputation is fast because it's done in SRAM (fast on-chip memory) rather than reading $\mathbf{A}$ from HBM (slow off-chip memory).

---

## 12.13 Complete 2D Backward: Numerical Validation

```python
import torch
import torch.nn.functional as F
from torch.autograd import gradcheck

class ManualAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, W_Q, W_K, W_V):
        Q = X @ W_Q
        K = X @ W_K
        V = X @ W_V
        d_k = Q.shape[-1]
        S = Q @ K.T
        S_prime = S / (d_k ** 0.5)
        A = F.softmax(S_prime, dim=-1)
        Y = A @ V
        ctx.save_for_backward(X, W_Q, W_K, W_V, Q, K, V, A)
        return Y
    
    @staticmethod
    def backward(ctx, grad_Y):
        X, W_Q, W_K, W_V, Q, K, V, A = ctx.saved_tensors
        d_k = Q.shape[-1]
        
        # Node 7: Y = A @ V
        grad_A = grad_Y @ V.T
        grad_V = A.T @ grad_Y
        
        # Node 6: softmax
        c = (grad_A * A).sum(dim=-1, keepdim=True)
        grad_S_prime = A * (grad_A - c)
        
        # Node 5: scaling
        grad_S = grad_S_prime / (d_k ** 0.5)
        
        # Node 4: S = Q @ K.T
        grad_Q = grad_S @ K
        grad_K = grad_S.T @ Q
        
        # Nodes 1-3: linear projections
        grad_W_Q = X.T @ grad_Q
        grad_W_K = X.T @ grad_K
        grad_W_V = X.T @ grad_V
        grad_X = grad_Q @ W_Q.T + grad_K @ W_K.T + grad_V @ W_V.T
        
        return grad_X, grad_W_Q, grad_W_K, grad_W_V

# Gradient check with double precision
torch.manual_seed(42)
n, d, d_k = 4, 6, 6

X = torch.randn(n, d, dtype=torch.float64, requires_grad=True)
W_Q = torch.randn(d, d_k, dtype=torch.float64, requires_grad=True)
W_K = torch.randn(d, d_k, dtype=torch.float64, requires_grad=True)
W_V = torch.randn(d, d_k, dtype=torch.float64, requires_grad=True)

print("Gradient check (finite differences vs our backward):")
result = gradcheck(ManualAttention.apply, (X, W_Q, W_K, W_V), eps=1e-6, atol=1e-4)
print(f"  Passed: {result}")
```

---

## 12.14 The Backward Through a Full Transformer Block

### Forward (pre-norm)

$$\mathbf{Z} = \mathbf{X} + \text{MHA}(\text{LN}_1(\mathbf{X}))$$
$$\text{Output} = \mathbf{Z} + \text{FFN}(\text{LN}_2(\mathbf{Z}))$$

### Backward (given $\bar{\text{Output}}$)

**Phase 1: FFN residual block**

1. $\bar{\mathbf{Z}} = \bar{\text{Output}}$ (skip connection path)
2. $\bar{\mathbf{Z}}_\text{ffn\_out} = \bar{\text{Output}}$ (through FFN path)
3. Backprop through FFN → $\bar{\text{LN}_2\text{\_out}}$, $\bar{\mathbf{W}}_1$, $\bar{\mathbf{W}}_2$, etc.
4. Backprop through LayerNorm₂ → $\bar{\mathbf{Z}}_\text{from\_ffn}$, $\bar{\boldsymbol{\gamma}}_2$, $\bar{\boldsymbol{\beta}}_2$
5. Accumulate: $\bar{\mathbf{Z}} = \bar{\mathbf{Z}} + \bar{\mathbf{Z}}_\text{from\_ffn}$

**Phase 2: Attention residual block**

6. $\bar{\mathbf{X}} = \bar{\mathbf{Z}}$ (skip connection path)
7. $\bar{\text{attn\_out}} = \bar{\mathbf{Z}}$ (through attention path, same gradient to both paths)
8. Backprop through MHA → $\bar{\text{LN}_1\text{\_out}}$, $\bar{\mathbf{W}}_Q$, $\bar{\mathbf{W}}_K$, $\bar{\mathbf{W}}_V$, $\bar{\mathbf{W}}_O$
9. Backprop through LayerNorm₁ → $\bar{\mathbf{X}}_\text{from\_attn}$, $\bar{\boldsymbol{\gamma}}_1$, $\bar{\boldsymbol{\beta}}_1$
10. Accumulate: $\bar{\mathbf{X}} = \bar{\mathbf{X}} + \bar{\mathbf{X}}_\text{from\_attn}$

**The gradient highway**: Notice that $\bar{\mathbf{X}}$ always includes the direct $\bar{\mathbf{Z}}$ term from the residual, which itself includes the direct $\bar{\text{Output}}$ term. The gradient flows directly from the output to any layer's input through the residual chain.

---

## 12.15 Summary: The Gradient Cookbook

| Operation | Forward | Backward |
|-----------|---------|----------|
| Linear: $\mathbf{Y} = \mathbf{X}\mathbf{W}$ | matmul | $\bar{\mathbf{X}} = \bar{\mathbf{Y}}\mathbf{W}^T$, $\bar{\mathbf{W}} = \mathbf{X}^T\bar{\mathbf{Y}}$ |
| Transpose: $\mathbf{K}^T$ | swap axes | $\bar{\mathbf{K}} = (\bar{\mathbf{K}^T})^T$ (swap same axes) |
| Scale: $\mathbf{S}/\alpha$ | element-wise | $\bar{\mathbf{S}} = \bar{\mathbf{S}'}/\alpha$ |
| Softmax: $\mathbf{A} = \sigma(\mathbf{S})$ | exp + normalize | $\bar{\mathbf{S}} = \mathbf{A} \odot (\bar{\mathbf{A}} - \mathbf{c})$ |
| Residual: $\mathbf{Z} = \mathbf{X} + f(\mathbf{X})$ | add | $\bar{\mathbf{X}} = \bar{\mathbf{Z}} + \bar{f}$ |
| LayerNorm | normalize + affine | See §12.8 |
| GELU | element-wise | $\bar{\mathbf{x}} = \bar{\mathbf{y}} \odot \text{GELU}'(\mathbf{x})$ |
| Fan-out ($\mathbf{X}$ used in Q,K,V) | — | **Sum** gradients from all paths |

Every gradient in the transformer is a combination of these primitives. Once you internalize this table, you can derive the backward pass for **any** transformer variant.

---

## 12.16 Exercises

1. **Full numerical verification**: Implement the complete backward for a single transformer block (attention + residual + LayerNorm + FFN + residual + LayerNorm) and verify against `torch.autograd.gradcheck`.

2. **Causal mask backward**: The causal mask sets some entries of $\mathbf{S}$ to $-\infty$. Show that the gradient $\bar{S}_{ij} = 0$ for masked positions (because $A_{ij} = 0$ there, and $\bar{S}_{ij} = A_{ij}(\ldots)$).

3. **Memory budget**: For a model with $B=32, n=2048, d=1024, h=16, L=24$, compute the total memory needed to store all intermediates for the backward pass.

4. **Gradient flow visualization**: For a 3-layer transformer, compute $\|\bar{\mathbf{X}}_\text{layer 0}\|$ (the gradient at the first layer) and show that it remains well-conditioned (neither vanishing nor exploding) due to residual connections.

5. **Custom backward for FlashAttention**: Implement attention backward that does NOT store $\mathbf{A}$ — instead, recompute it from $\mathbf{Q}, \mathbf{K}$ and the logsumexp values. Compare memory usage and verify correctness.
