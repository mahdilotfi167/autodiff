# 2. Batching & Gradient Accumulation

## Motivation

In practice, you never train on a single sample. You train on **batches** of, say, 32 or 256 samples simultaneously. This raises the core question:

> **Does the framework compute gradients for each sample individually and then sum them? Or does it process the entire batch at once through a single computation graph?**

**Answer**: It builds **one graph** for the entire batch. The batch dimension is just another tensor dimension — operations like matmul, addition, and activation functions all operate on the full batch tensor **vectorized**, in parallel. There is **one** forward pass, **one** backward pass, and the resulting gradients are naturally aggregated (typically via a reduction like `mean` or `sum` in the loss function).

---

## 2.1 The Batch Dimension

In deep learning, tensors typically have a **batch dimension** as the first axis:

| Tensor | Shape | Meaning |
|--------|-------|---------|
| Input $\mathbf{X}$ | $(B, D_{in})$ | $B$ samples, each of dimension $D_{in}$ |
| Weight $\mathbf{W}$ | $(D_{in}, D_{out})$ | Shared across all samples |
| Output $\mathbf{Y}$ | $(B, D_{out})$ | $B$ outputs, one per sample |
| Loss $L$ | scalar | Single number (reduced across batch) |

The weight matrix $\mathbf{W}$ has **no batch dimension** — it is shared. This is fundamental: **parameters are shared across all samples in the batch**.

---

## 2.2 A Single Graph for the Whole Batch

Consider a linear layer with $B=2$ samples, $D_{in}=3$, $D_{out}=2$:

$$\mathbf{Y} = \mathbf{X}\mathbf{W} + \mathbf{b}$$

Where:
- $\mathbf{X} \in \mathbb{R}^{2 \times 3}$ (2 samples, 3 features)
- $\mathbf{W} \in \mathbb{R}^{3 \times 2}$ (shared weights)
- $\mathbf{b} \in \mathbb{R}^{2}$ (shared bias, broadcast to both samples)
- $\mathbf{Y} \in \mathbb{R}^{2 \times 2}$ (2 samples, 2 outputs)

The computation graph looks like:

```
  X (2×3)     W (3×2)       b (2,)
      \       /              |
      [matmul]          [broadcast]
         \                /
         Y_raw (2×2)     b (2×2)
              \          /
              [add]-----
                |
             Y (2×2)
                |
            [loss_fn]
                |
             L (scalar)
```

This is **one single graph**. The batch dimension (size 2) is simply carried through every operation. PyTorch does not create separate graphs or separate forward passes per sample.

---

## 2.3 How the Loss Reduces the Batch

The loss function almost always **reduces** across the batch dimension to produce a scalar. This reduction is where the "aggregation" happens:

### Mean reduction (default in most PyTorch losses)
$$L = \frac{1}{B} \sum_{i=1}^{B} \ell_i$$

where $\ell_i$ is the per-sample loss.

### Sum reduction
$$L = \sum_{i=1}^{B} \ell_i$$

**This choice matters for gradient magnitudes.** With mean reduction, gradients are scaled by $\frac{1}{B}$. With sum reduction, they are $B$ times larger.

---

## 2.4 Toy Hand Calculation: Batch of 2 Through Linear Layer

Let's trace everything explicitly.

**Setup:**
$$\mathbf{X} = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}, \quad
\mathbf{W} = \begin{pmatrix} 0.5 \\ -0.5 \end{pmatrix}, \quad
b = 0.1$$

Shapes: $\mathbf{X} \in \mathbb{R}^{2 \times 2}$, $\mathbf{W} \in \mathbb{R}^{2 \times 1}$, $b \in \mathbb{R}$.

**Forward pass:**

$$\mathbf{Y} = \mathbf{X}\mathbf{W} + b = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}\begin{pmatrix} 0.5 \\ -0.5 \end{pmatrix} + 0.1 = \begin{pmatrix} 0.5 - 1.0 \\ 1.5 - 2.0 \end{pmatrix} + 0.1 = \begin{pmatrix} -0.4 \\ -0.4 \end{pmatrix}$$

So $y_1 = -0.4$ (sample 1), $y_2 = -0.4$ (sample 2).

**Loss (MSE with target $\mathbf{t} = (1, 1)^T$, mean reduction):**

$$L = \frac{1}{2}\left[(y_1 - t_1)^2 + (y_2 - t_2)^2\right] = \frac{1}{2}\left[(-1.4)^2 + (-1.4)^2\right] = \frac{1}{2}(1.96 + 1.96) = 1.96$$

**Backward pass:**

Step 1: $\frac{\partial L}{\partial y_i} = \frac{1}{B}(y_i - t_i)$

$$\frac{\partial L}{\partial \mathbf{y}} = \frac{1}{2}\begin{pmatrix} 2(-1.4) \\ 2(-1.4) \end{pmatrix} = \begin{pmatrix} -1.4 \\ -1.4 \end{pmatrix}$$

Step 2: $\mathbf{y} = \mathbf{X}\mathbf{W} + b$, so $\frac{\partial \mathbf{y}}{\partial \mathbf{W}} = \mathbf{X}^T$

$$\frac{\partial L}{\partial \mathbf{W}} = \mathbf{X}^T \frac{\partial L}{\partial \mathbf{y}} = \begin{pmatrix} 1 & 3 \\ 2 & 4 \end{pmatrix}\begin{pmatrix} -1.4 \\ -1.4 \end{pmatrix} = \begin{pmatrix} -1.4 - 4.2 \\ -2.8 - 5.6 \end{pmatrix} = \begin{pmatrix} -5.6 \\ -8.4 \end{pmatrix}$$

Step 3: $\frac{\partial L}{\partial b} = \sum_i \frac{\partial L}{\partial y_i} = -1.4 + (-1.4) = -2.8$

**Observe**: The gradient $\frac{\partial L}{\partial \mathbf{W}}$ is a **single** $(2 \times 1)$ vector — same shape as $\mathbf{W}$. It is NOT two separate gradients. The matrix multiplication $\mathbf{X}^T \frac{\partial L}{\partial \mathbf{y}}$ naturally combines information from **all samples** in one operation.

---

## 2.5 The Key Insight: Vectorized Computation, Not Per-Sample

Let's be very precise about what happens:

### What DOES happen (vectorized):
1. Forward: One matmul $\mathbf{Y} = \mathbf{X}\mathbf{W}$ computes all $B$ outputs simultaneously
2. Loss: One reduction `mean()` or `sum()` aggregates all $B$ per-sample losses
3. Backward: One matmul $\frac{\partial L}{\partial \mathbf{W}} = \mathbf{X}^T \frac{\partial L}{\partial \mathbf{Y}}$ computes the gradient using all $B$ samples at once

### What does NOT happen:
1. ~~Loop over $B$ samples, compute gradient for each, sum them~~ **No.**
2. ~~Build $B$ separate computation graphs~~ **No.**
3. ~~Store $B$ separate gradient tensors, then reduce~~ **No.**

### Why they're mathematically equivalent

Although the framework computes everything at once, the result IS the sum of per-sample gradients. Here's the proof:

The per-sample gradient for sample $i$ would be:

$$\frac{\partial \ell_i}{\partial \mathbf{W}} = \mathbf{x}_i^T \frac{\partial \ell_i}{\partial y_i}$$

where $\mathbf{x}_i$ is the $i$-th row of $\mathbf{X}$.

The total gradient with mean reduction:

$$\frac{\partial L}{\partial \mathbf{W}} = \frac{1}{B}\sum_{i=1}^{B} \mathbf{x}_i^T \frac{\partial \ell_i}{\partial y_i}$$

Now, $\mathbf{X}^T \frac{\partial L}{\partial \mathbf{y}}$ in matrix form IS this sum (the matrix product performs the sum over $i$ via the shared dimension). So the vectorized version and the per-sample-then-sum version give **identical** results — but the vectorized version does it in one BLAS call, exploiting GPU parallelism.

---

## 2.6 Gradient Accumulation (Multiple Backward Passes)

Sometimes your batch is too large to fit in GPU memory. **Gradient accumulation** simulates a large batch by running multiple small batches and **summing** their gradients before updating.

### The mechanism

PyTorch **adds** to `.grad` by default (it does NOT overwrite). This is the `AccumulateGrad` node:

```python
# Gradient accumulation: effective batch size = 4 * 8 = 32
optimizer.zero_grad()            # Reset .grad to zero

for i in range(4):               # 4 mini-batches of size 8
    X_mini = get_batch(size=8)
    loss = model(X_mini)
    loss = loss / 4              # Scale to get mean over effective batch
    loss.backward()              # ADDS to .grad (does not overwrite!)

optimizer.step()                 # Update using accumulated gradient
```

After 4 calls to `.backward()`, each parameter's `.grad` contains the sum of gradients from all 4 mini-batches. Dividing the loss by 4 ensures the final accumulated gradient equals what you'd get from a single batch of 32.

### Why `.grad` accumulates by default

This is a design choice by PyTorch. The `AccumulateGrad` node does:
```python
param.grad = param.grad + new_gradient  # if grad already exists
param.grad = new_gradient                # if grad is None
```

This is why you **must call `optimizer.zero_grad()`** before each logical step — otherwise gradients from previous iterations leak in.

### Common bug
```python
# BUG: forgot zero_grad()
for epoch in range(100):
    loss = compute_loss()
    loss.backward()          # Gradients keep accumulating across epochs!
    optimizer.step()
```

---

## 2.7 Shape Rules: How Gradients Handle Broadcasting

When the batch dimension is involved, we encounter **broadcasting**. The gradient must **undo** the broadcast via summation.

### Rule: If a tensor was broadcast (expanded) during forward, its gradient is **summed** along the broadcast dimensions during backward.

**Example**: Adding bias $\mathbf{b} \in \mathbb{R}^{D}$ to a batch $\mathbf{Y} \in \mathbb{R}^{B \times D}$:

Forward:
$$\mathbf{Z}_{ij} = \mathbf{Y}_{ij} + b_j \quad \text{(b is broadcast from (D,) to (B, D))}$$

Backward:
$$\frac{\partial L}{\partial b_j} = \sum_{i=1}^{B} \frac{\partial L}{\partial Z_{ij}}$$

The gradient of $\mathbf{b}$ has shape $(D,)$ — same as $\mathbf{b}$ — obtained by summing the output gradient across the batch dimension.

### General broadcasting gradient rule

For any operation where input shape $(s_1)$ was broadcast to match output shape $(s_2)$:

1. Compute gradient w.r.t. the broadcast shape $(s_2)$
2. **Sum** over all dimensions that were broadcast (i.e., dimensions where $s_1$ had size 1 or was missing)
3. Result has shape $(s_1)$

```python
import torch

# b has shape (3,), Y has shape (4, 3)
b = torch.randn(3, requires_grad=True)
Y = torch.randn(4, 3, requires_grad=True)

Z = Y + b       # b broadcast from (3,) to (4,3)
L = Z.sum()
L.backward()

print(f"Y.grad shape: {Y.grad.shape}")  # (4, 3) — no broadcast, no sum
print(f"b.grad shape: {b.grad.shape}")   # (3,)   — summed over batch dim
print(f"b.grad:       {b.grad}")         # each entry is sum over 4 samples
# Since dL/dZ = all ones, b.grad = [4, 4, 4]
```

---

## 2.8 Multiple Outputs per Sample

What if the network outputs a vector per sample (e.g., logits for classification)?

$$\mathbf{Y} \in \mathbb{R}^{B \times C} \quad \text{(B samples, C classes)}$$

The loss still reduces to a scalar:
$$L = \frac{1}{B}\sum_{i=1}^{B} \text{CrossEntropy}(\mathbf{y}_i, t_i)$$

The gradient $\frac{\partial L}{\partial \mathbf{Y}} \in \mathbb{R}^{B \times C}$ — one gradient per sample per class. This entire $(B \times C)$ tensor flows backward through the graph.

---

## 2.9 Complete PyTorch Example: Tracing Batch Gradients

```python
import torch
import torch.nn as nn

# Setup
B, D_in, D_out = 4, 3, 2
X = torch.randn(B, D_in)                       # 4 samples, 3 features
targets = torch.randn(B, D_out)                # 4 targets, 2 outputs

# Model: single linear layer
linear = nn.Linear(D_in, D_out, bias=True)     # W: (3,2), b: (2,)

# Forward
Y = linear(X)                                   # (4, 2)
loss = nn.MSELoss(reduction='mean')(Y, targets)  # scalar

print(f"X shape:      {X.shape}")               # (4, 3)
print(f"Y shape:      {Y.shape}")               # (4, 2)
print(f"loss:         {loss.item():.4f}")

# Backward
loss.backward()

# Inspect gradients
W = linear.weight  # PyTorch stores weight as (D_out, D_in) = (2, 3)
b = linear.bias

print(f"\nW shape:      {W.shape}")              # (2, 3)
print(f"W.grad shape: {W.grad.shape}")           # (2, 3) — same as W
print(f"b shape:      {b.shape}")                # (2,)
print(f"b.grad shape: {b.grad.shape}")           # (2,) — same as b

# Verify: gradient is averaged over batch
# Manual computation for comparison:
# dL/dY = (2/B)(Y - targets) for MSE with mean reduction, then /2 from MSE definition
# Actually MSELoss = mean((Y-t)^2), so dL/dY = 2*(Y-targets)/B/D_out ... 
# Let PyTorch handle it; the point is shapes match.

print(f"\nW.grad:\n{W.grad}")
print(f"b.grad: {b.grad}")
```

---

## 2.10 Per-Sample Gradients (Advanced)

Standard PyTorch does NOT give you per-sample gradients $\frac{\partial \ell_i}{\partial \mathbf{W}}$ — only the aggregated $\frac{\partial L}{\partial \mathbf{W}} = \frac{1}{B}\sum_i \frac{\partial \ell_i}{\partial \mathbf{W}}$.

If you need per-sample gradients (for differential privacy, influence functions, etc.), you use:

1. **`torch.vmap`** (functorch): Vectorized map that efficiently computes per-sample gradients
2. **Loop** (slow): Loop over batch, compute gradient for each sample
3. **BackPACK** library: Extensions for per-sample gradient statistics

```python
# Using torch.func (functorch) for per-sample gradients
from torch.func import grad, vmap

def compute_loss_single(params, x, target):
    """Loss for a single sample."""
    W, b = params
    y = x @ W + b
    return ((y - target) ** 2).sum()

# Per-sample gradient via vmap
per_sample_grad = vmap(grad(compute_loss_single), in_dims=(None, 0, 0))
# None = don't vectorize over params, 0 = vectorize over batch dim

W = torch.randn(3, 2, requires_grad=True)
b = torch.randn(2, requires_grad=True)
X = torch.randn(4, 3)
targets = torch.randn(4, 2)

grads = per_sample_grad((W, b), X, targets)
print(f"Per-sample W gradient shape: {grads[0].shape}")  # (4, 3, 2)
# ^ 4 samples, each has a (3, 2) gradient for W
```

---

## 2.11 Summary: The Batch Mental Model

```
              ONE computation graph
              for the ENTIRE batch
                     |
    ┌────────────────┼────────────────┐
    │                |                │
 sample 1       sample 2         sample B
    │                |                │
    └───── batched in dim 0 ─────────┘
                     |
              ONE forward pass
             (vectorized matmul)
                     |
              loss reduction
        (mean or sum across batch)
                     |
              ONE scalar loss
                     |
              ONE backward pass
                     |
         ONE gradient per parameter
       (shape = parameter shape, always)
```

| Question | Answer |
|----------|--------|
| One graph per sample? | **No.** One graph for the entire batch. |
| Gradients computed per-sample? | **No.** Computed all-at-once via vectorized ops (matmul). |
| Are per-sample results summed? | **Yes, but implicitly** — the matrix multiplication inherently sums across samples. |
| Why call `zero_grad()`? | Because `.grad` **accumulates** (adds) by default. |
| Gradient shape? | Always equals parameter shape. |
| How does bias gradient work? | Broadcasting during forward → summing during backward. |

---

**Next**: Document 3 covers the mathematical framework that makes all of this work — matrix calculus, Jacobians, and the backpropagation algorithm as reverse-mode automatic differentiation.
