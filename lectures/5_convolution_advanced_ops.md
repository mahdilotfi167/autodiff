# 5. Convolution, Softmax, and Advanced Operation Gradients

## Motivation

Convolution is the backbone of computer vision models. Softmax appears in every classifier and every attention mechanism. Both have rich gradient structures that don't reduce to simple element-wise formulas. Understanding how their gradients are derived — and how framework authors implement them efficiently — is essential to understanding modern deep learning internals.

---

## Part I: Convolution

## 5.1 1D Convolution: The Simplest Case

### Forward

A 1D convolution of input $\mathbf{x} \in \mathbb{R}^N$ with kernel $\mathbf{w} \in \mathbb{R}^K$ (no padding, stride 1) produces output $\mathbf{y} \in \mathbb{R}^{N-K+1}$:

$$y_i = \sum_{k=0}^{K-1} w_k \cdot x_{i+k}, \quad i = 0, 1, \ldots, N-K$$

### As a matrix operation (Toeplitz structure)

This convolution can be written as $\mathbf{y} = \mathbf{T}_x \mathbf{w}$, where $\mathbf{T}_x$ is a **Toeplitz matrix**:

For $\mathbf{x} = (x_0, x_1, x_2, x_3, x_4)$ and kernel size $K=3$:

$$\mathbf{T}_x = \begin{pmatrix} x_0 & x_1 & x_2 \\ x_1 & x_2 & x_3 \\ x_2 & x_3 & x_4 \end{pmatrix}$$

So:
$$\mathbf{y} = \begin{pmatrix} x_0 & x_1 & x_2 \\ x_1 & x_2 & x_3 \\ x_2 & x_3 & x_4 \end{pmatrix} \begin{pmatrix} w_0 \\ w_1 \\ w_2 \end{pmatrix} = \begin{pmatrix} x_0 w_0 + x_1 w_1 + x_2 w_2 \\ x_1 w_0 + x_2 w_1 + x_3 w_2 \\ x_2 w_0 + x_3 w_1 + x_4 w_2 \end{pmatrix}$$

### Gradient of the kernel

$$\frac{\partial y_i}{\partial w_k} = x_{i+k}$$

$$\frac{\partial L}{\partial w_k} = \sum_i \bar{y}_i \cdot x_{i+k}$$

**This is itself a convolution!** The gradient of the kernel is the convolution of the input with the upstream gradient (cross-correlation of $\mathbf{x}$ and $\bar{\mathbf{y}}$).

Using our Toeplitz form: $\frac{\partial L}{\partial \mathbf{w}} = \mathbf{T}_x^T \bar{\mathbf{y}}$ — exactly the matmul gradient formula!

### Gradient of the input

$$\frac{\partial y_i}{\partial x_j}$$

For a given output $y_i = \sum_k w_k x_{i+k}$, the derivative w.r.t. $x_j$ is nonzero only when $j = i + k$ for some $k \in \{0, \ldots, K-1\}$, i.e., when $i = j - k$:

$$\frac{\partial L}{\partial x_j} = \sum_i \bar{y}_i \frac{\partial y_i}{\partial x_j} = \sum_{k=0}^{K-1} \bar{y}_{j-k} \cdot w_k$$

(where $\bar{y}_{j-k}$ is zero if $j-k$ is out of bounds).

**This is a convolution of $\bar{\mathbf{y}}$ (zero-padded) with the flipped kernel $\tilde{\mathbf{w}} = (w_{K-1}, \ldots, w_1, w_0)$.**

### Summary for 1D conv

| Gradient | Formula | Interpretation |
|----------|---------|----------------|
| $\frac{\partial L}{\partial \mathbf{w}}$ | cross-correlation of input and upstream | Conv(input, grad_output) |
| $\frac{\partial L}{\partial \mathbf{x}}$ | "full" convolution of upstream with flipped kernel | Conv(grad_output_padded, flip(kernel)) |

---

## 5.2 2D Convolution

### Forward

Input: $\mathbf{X} \in \mathbb{R}^{H \times W}$, Kernel: $\mathbf{K} \in \mathbb{R}^{K_H \times K_W}$

$$Y_{ij} = \sum_{a=0}^{K_H-1} \sum_{b=0}^{K_W-1} K_{ab} \cdot X_{i+a, j+b}$$

Output: $\mathbf{Y} \in \mathbb{R}^{(H-K_H+1) \times (W-K_W+1)}$

### Multi-channel convolution (realistic)

Input: $\mathbf{X} \in \mathbb{R}^{C_{in} \times H \times W}$ (channels × height × width)
Kernel: $\mathbf{K} \in \mathbb{R}^{C_{out} \times C_{in} \times K_H \times K_W}$
Output: $\mathbf{Y} \in \mathbb{R}^{C_{out} \times H' \times W'}$

$$Y_{c_{out}, i, j} = \sum_{c_{in}=0}^{C_{in}-1} \sum_{a=0}^{K_H-1} \sum_{b=0}^{K_W-1} K_{c_{out}, c_{in}, a, b} \cdot X_{c_{in}, i+a, j+b}$$

Each output channel $c_{out}$ is the sum of $C_{in}$ cross-correlations.

---

## 5.3 The im2col Trick: Reducing Conv to Matmul

Computing convolution directly (nested loops over spatial dimensions and channels) is slow. The **im2col** trick restructures the data so convolution becomes a single matrix multiplication.

### How it works

For each output position $(i, j)$, extract the input patch that the kernel would multiply — a vector of length $C_{in} \times K_H \times K_W$. Stack all these patches as **rows** of a matrix $\mathbf{X}_{col}$:

$$\mathbf{X}_{col} \in \mathbb{R}^{(H' \cdot W') \times (C_{in} \cdot K_H \cdot K_W)}$$

Reshape the kernel as:
$$\mathbf{K}_{col} \in \mathbb{R}^{(C_{in} \cdot K_H \cdot K_W) \times C_{out}}$$

Then: $\mathbf{Y}_{col} = \mathbf{X}_{col} \mathbf{K}_{col}$

This is a **standard matmul**! All gradient formulas from Document 4 apply directly.

### Gradient via im2col

Since $\mathbf{Y}_{col} = \mathbf{X}_{col} \mathbf{K}_{col}$:

$$\frac{\partial L}{\partial \mathbf{K}_{col}} = \mathbf{X}_{col}^T \bar{\mathbf{Y}}_{col}$$
$$\frac{\partial L}{\partial \mathbf{X}_{col}} = \bar{\mathbf{Y}}_{col} \mathbf{K}_{col}^T$$

Then $\frac{\partial L}{\partial \mathbf{X}_{col}}$ is "un-im2col'd" (col2im) back to the original input shape, accumulating overlapping patches via addition.

### Why this matters

- Framework authors don't write special gradient code for conv — they express it as matmul after im2col
- cuDNN (NVIDIA's library) uses im2col + optimized GEMM (General Matrix Multiply) as one of its conv algorithms
- Other algorithms exist (Winograd, FFT-based) but im2col is the conceptual foundation

### Toy example

```python
import torch
import torch.nn.functional as F

# Input: 1 sample, 1 channel, 4×4
X = torch.randn(1, 1, 4, 4, requires_grad=True)

# Kernel: 1 output channel, 1 input channel, 2×2
K = torch.randn(1, 1, 2, 2, requires_grad=True)

# Forward
Y = F.conv2d(X, K)  # shape: (1, 1, 3, 3)
L = Y.sum()
L.backward()

print(f"X shape:      {X.shape}")       # (1, 1, 4, 4)
print(f"K shape:      {K.shape}")       # (1, 1, 2, 2)
print(f"Y shape:      {Y.shape}")       # (1, 1, 3, 3)
print(f"X.grad shape: {X.grad.shape}")  # (1, 1, 4, 4) — same as X
print(f"K.grad shape: {K.grad.shape}")  # (1, 1, 2, 2) — same as K

# Manual im2col equivalent
X_unfold = F.unfold(X, kernel_size=2)  # im2col
print(f"X_unfold shape: {X_unfold.shape}")  # (1, 4, 9) — 4=C*kH*kW, 9=H'*W'
# Y_manual = K_reshaped @ X_unfold
K_flat = K.view(1, -1)  # (1, 4)
Y_manual = K_flat @ X_unfold  # (1, 1, 9) → reshape to (1, 1, 3, 3)
print(f"Matmul matches conv: {torch.allclose(Y_manual.view(1,1,3,3), Y)}")  # True
```

---

## Part II: Softmax

## 5.4 Softmax: Forward

$$\text{softmax}(\mathbf{x})_i = \frac{e^{x_i}}{\sum_j e^{x_j}} = s_i$$

Properties:
- Output is a probability distribution: $s_i > 0$ and $\sum_i s_i = 1$
- Numerically computed as $s_i = \frac{e^{x_i - \max(\mathbf{x})}}{\sum_j e^{x_j - \max(\mathbf{x})}}$ for stability

---

## 5.5 Softmax: Jacobian Derivation

The Jacobian $\frac{\partial s_i}{\partial x_j}$ has two cases depending on whether $i = j$:

**Case $i = j$:** (derivative of $s_i$ w.r.t. its own logit $x_i$)

$$\frac{\partial s_i}{\partial x_i} = \frac{e^{x_i} \cdot \sum_k e^{x_k} - e^{x_i} \cdot e^{x_i}}{(\sum_k e^{x_k})^2}$$

(quotient rule: $\frac{d}{dx}\frac{f}{g} = \frac{f'g - fg'}{g^2}$)

$$= \frac{e^{x_i}}{\sum_k e^{x_k}} - \left(\frac{e^{x_i}}{\sum_k e^{x_k}}\right)^2 = s_i - s_i^2 = s_i(1 - s_i)$$

**Case $i \neq j$:** (derivative of $s_i$ w.r.t. a different logit $x_j$)

$$\frac{\partial s_i}{\partial x_j} = \frac{0 - e^{x_i} \cdot e^{x_j}}{(\sum_k e^{x_k})^2} = -s_i s_j$$

(numerator $e^{x_i}$ is constant w.r.t. $x_j$; denominator's derivative contributes $e^{x_j}$)

### Combined formula

$$\frac{\partial s_i}{\partial x_j} = s_i(\delta_{ij} - s_j)$$

where $\delta_{ij}$ is the Kronecker delta.

### As a matrix

$$\mathbf{J}_{\text{softmax}} = \text{diag}(\mathbf{s}) - \mathbf{s}\mathbf{s}^T$$

where $\text{diag}(\mathbf{s})$ is the diagonal matrix with $s_i$ on the diagonal.

---

## 5.6 Softmax VJP (Efficiently)

The naive approach: form $\mathbf{J} \in \mathbb{R}^{n \times n}$ and multiply → $O(n^2)$.

But we can do better:

$$\bar{\mathbf{x}} = \mathbf{J}^T \bar{\mathbf{s}} = (\text{diag}(\mathbf{s}) - \mathbf{s}\mathbf{s}^T) \bar{\mathbf{s}}$$
$$= \text{diag}(\mathbf{s}) \bar{\mathbf{s}} - \mathbf{s}\mathbf{s}^T \bar{\mathbf{s}}$$
$$= \mathbf{s} \odot \bar{\mathbf{s}} - \mathbf{s} (\mathbf{s}^T \bar{\mathbf{s}})$$
$$= \mathbf{s} \odot \bar{\mathbf{s}} - \mathbf{s} \cdot \underbrace{(\mathbf{s} \cdot \bar{\mathbf{s}})}_{\text{scalar } c}$$

Let $c = \sum_i s_i \bar{s}_i$ (dot product of softmax output and upstream gradient):

$$\boxed{\bar{\mathbf{x}} = \mathbf{s} \odot (\bar{\mathbf{s}} - c)}$$

**This is $O(n)$, not $O(n^2)$!** No need to form the Jacobian.

### Derivation step-by-step of the simplification:

1. $\text{diag}(\mathbf{s}) \bar{\mathbf{s}}$: element $i$ is $s_i \bar{s}_i$ → this is $\mathbf{s} \odot \bar{\mathbf{s}}$
2. $\mathbf{s}\mathbf{s}^T \bar{\mathbf{s}}$: first compute $\mathbf{s}^T \bar{\mathbf{s}} = \sum_j s_j \bar{s}_j = c$ (scalar), then $\mathbf{s} \cdot c$
3. Subtract: $\mathbf{s} \odot \bar{\mathbf{s}} - c \cdot \mathbf{s} = \mathbf{s} \odot (\bar{\mathbf{s}} - c)$

```python
import torch
import torch.nn.functional as F

x = torch.randn(5, requires_grad=True)
s = F.softmax(x, dim=0)
L = (s * torch.tensor([1.0, 0, 0, 0, 0])).sum()  # pick first element
L.backward()

# Manual VJP
with torch.no_grad():
    grad_s = torch.tensor([1.0, 0, 0, 0, 0])  # dL/ds
    c = (s * grad_s).sum()
    grad_x_manual = s * (grad_s - c)

print(f"Matches: {torch.allclose(x.grad, grad_x_manual)}")  # True
```

---

## 5.7 Cross-Entropy Loss (Softmax + Log + NLL combined)

In practice, you rarely backprop through softmax alone. Cross-entropy loss combines softmax with the log and negative log-likelihood:

$$L = -\log(s_{t}) = -\log\left(\frac{e^{x_t}}{\sum_j e^{x_j}}\right) = -x_t + \log\left(\sum_j e^{x_j}\right)$$

where $t$ is the target class index.

### Gradient (direct derivation — simpler than going through softmax)

$$\frac{\partial L}{\partial x_i} = \begin{cases} s_i - 1 & \text{if } i = t \\ s_i & \text{if } i \neq t \end{cases}$$

Or compactly: $\frac{\partial L}{\partial \mathbf{x}} = \mathbf{s} - \mathbf{e}_t$ where $\mathbf{e}_t$ is the one-hot vector for class $t$.

**Derivation**:
$$\frac{\partial L}{\partial x_i} = -\delta_{it} + \frac{e^{x_i}}{\sum_j e^{x_j}} = -\delta_{it} + s_i$$

This is beautifully simple and numerically stable — which is why `F.cross_entropy` in PyTorch takes raw logits, not softmax outputs.

```python
import torch
import torch.nn.functional as F

logits = torch.randn(1, 5, requires_grad=True)  # 5 classes
target = torch.tensor([2])                        # true class = 2

loss = F.cross_entropy(logits, target)
loss.backward()

# Manual gradient
with torch.no_grad():
    s = F.softmax(logits, dim=1)
    grad_manual = s.clone()
    grad_manual[0, target[0]] -= 1

print(f"Matches: {torch.allclose(logits.grad, grad_manual)}")  # True
```

---

## Part III: Layer Normalization

## 5.8 LayerNorm Forward

$$\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}, \quad y_i = \gamma \hat{x}_i + \beta$$

where $\mu = \frac{1}{n}\sum_i x_i$, $\sigma^2 = \frac{1}{n}\sum_i (x_i - \mu)^2$.

### Why the gradient is complex

$\mu$ and $\sigma^2$ both depend on ALL elements of $\mathbf{x}$. So when computing $\frac{\partial \hat{x}_i}{\partial x_j}$, we must account for how $x_j$ affects $\hat{x}_i$ through:
1. Direct effect ($i = j$): the numerator changes
2. Indirect effect via $\mu$: changing $x_j$ shifts the mean
3. Indirect effect via $\sigma^2$: changing $x_j$ changes the variance

### VJP for LayerNorm (result)

Given upstream $\bar{\mathbf{y}}$, the gradient w.r.t. the pre-norm $\mathbf{x}$:

$$\bar{\gamma} = \sum_i \bar{y}_i \hat{x}_i, \quad \bar{\beta} = \sum_i \bar{y}_i$$

$$\bar{\hat{\mathbf{x}}} = \bar{\mathbf{y}} \odot \gamma$$

$$\bar{\sigma^2} = -\frac{1}{2}(\sigma^2 + \epsilon)^{-3/2} \sum_i \bar{\hat{x}}_i (x_i - \mu)$$

$$\bar{\mu} = -\frac{1}{\sqrt{\sigma^2 + \epsilon}} \sum_i \bar{\hat{x}}_i$$

$$\bar{x}_i = \frac{\bar{\hat{x}}_i}{\sqrt{\sigma^2 + \epsilon}} + \frac{2\bar{\sigma^2}(x_i - \mu)}{n} + \frac{\bar{\mu}}{n}$$

This can be simplified (and often is in practice) to:

$$\bar{\mathbf{x}} = \frac{1}{\sqrt{\sigma^2 + \epsilon}} \left(\bar{\hat{\mathbf{x}}} - \frac{1}{n}\sum_i \bar{\hat{x}}_i - \frac{\hat{\mathbf{x}}}{n}\sum_i \bar{\hat{x}}_i \hat{x}_i\right)$$

The key insight: the gradient depends on the **statistics of the entire normalization group**, not just local information.

---

## 5.9 How Framework Authors Derive These

The process PyTorch/JAX authors follow when implementing a new operation:

1. **Write the mathematical definition** of the forward pass
2. **Derive the Jacobian or VJP** using one of:
   - Element-wise derivation (Section 4.7 Method 1)
   - Matrix differential calculus with trace trick (Document 7)
   - Reference from academic literature
3. **Simplify** the VJP expression to minimize FLOPs
4. **Implement** in C++/CUDA (or Python for prototyping)
5. **Validate** using numerical gradient checking (Document 6)
6. **Register** the forward/backward pair with the autograd engine

For complex operations (LayerNorm, multi-head attention), authors often:
- Decompose into simpler operations and let autograd compose the gradients
- Or write a fused kernel with hand-derived backward for performance

---

## 5.10 PyTorch Verification: LayerNorm

```python
import torch
import torch.nn as nn

x = torch.randn(2, 4, requires_grad=True)  # 2 samples, 4 features
ln = nn.LayerNorm(4)

y = ln(x)
L = y.sum()
L.backward()

print(f"x.grad shape: {x.grad.shape}")   # (2, 4) — same as x
print(f"x.grad:\n{x.grad}")

# Verify with numerical gradient
eps = 1e-5
grad_numerical = torch.zeros_like(x)
for i in range(x.shape[0]):
    for j in range(x.shape[1]):
        x_plus = x.data.clone()
        x_plus[i, j] += eps
        x_minus = x.data.clone()
        x_minus[i, j] -= eps
        L_plus = ln(x_plus).sum()
        L_minus = ln(x_minus).sum()
        grad_numerical[i, j] = (L_plus - L_minus) / (2 * eps)

print(f"\nNumerical gradient matches: {torch.allclose(x.grad, grad_numerical, atol=1e-4)}")
```

---

## 5.11 Summary: Gradient Structures

| Operation | Jacobian Structure | Key Property |
|-----------|-------------------|-------------|
| Element-wise | Diagonal | $O(n)$ VJP |
| Matmul $\mathbf{AB}$ | Dense (but structured) | VJP is matmul with transpose |
| 1D Conv | Sparse, Toeplitz | VJP is conv with flipped kernel |
| 2D Conv | Sparse, block-Toeplitz | VJP via im2col → matmul |
| Softmax | Dense ($\text{diag}(\mathbf{s}) - \mathbf{ss}^T$) | $O(n)$ VJP via dot-product trick |
| Cross-entropy (fused) | — | $O(n)$: just $\mathbf{s} - \mathbf{e}_t$ |
| LayerNorm | Dense (all-to-all via statistics) | VJP involves normalization group stats |

### The unifying principle

Every gradient derivation follows the same recipe:
1. Establish how each output element depends on each input element
2. Apply chain rule: upstream gradient × local Jacobian
3. Find an efficient computation that avoids forming the full Jacobian

The **trace trick** (Document 7) provides an elegant shortcut for steps 1-3 that avoids element-wise reasoning entirely.

---

**Next**: Document 6 covers numerical gradient checking — the essential validation tool that framework authors use to verify their hand-derived gradients are correct.
