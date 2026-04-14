# 10. Multi-Head Attention and 3D Tensor Calculus

## Motivation

Document 9 derived single-head attention for a 2D input $\mathbf{X} \in \mathbb{R}^{n \times d}$. But real transformers have **three** dimensions: batch ($B$), sequence length ($n$), and model dimension ($d$). Furthermore, they use **multi-head attention** — running $h$ attention heads in parallel, each on a different subspace, then concatenating.

This document addresses the hard part: **how do researchers reason about operations on 3D (and 4D) tensors?** How do they figure out which axes to transpose, which matmuls to use, and how reshaping/splitting works? We develop the mathematical framework for batched tensor operations and derive multi-head attention from scratch.

---

## 10.1 The 3D Tensor Setup

### Notation

In practice, inputs arrive in batches:

$$\mathbf{X} \in \mathbb{R}^{B \times n \times d}$$

- **Axis 0** ($B$): batch — independent sequences processed in parallel
- **Axis 1** ($n$): sequence position — tokens within a sequence
- **Axis 2** ($d$): feature — the embedding dimension of each token

Think of it as: $B$ copies of the 2D problem from Doc 9, stacked along a new axis.

### The fundamental principle

> **The batch dimension is a "spectator."** Operations that act on the last two dimensions are applied independently to each batch element.

This means: if you understand the 2D case, the 3D case is "just" applying it $B$ times. But the devil is in the details of how to express this with efficient tensor operations.

---

## 10.2 Batched Matrix Multiplication (BMM)

### The 2D case (review)

$$\mathbf{C} = \mathbf{A}\mathbf{B}, \quad A \in \mathbb{R}^{m \times k}, \; B \in \mathbb{R}^{k \times p}, \; C \in \mathbb{R}^{m \times p}$$

$$C_{ij} = \sum_l A_{il} B_{lj}$$

### The 3D case: batched matmul

Given 3D tensors:

$$\mathbf{A} \in \mathbb{R}^{B \times m \times k}, \quad \mathbf{B} \in \mathbb{R}^{B \times k \times p}$$

The batched matmul is:

$$\mathbf{C} = \mathbf{A} \mathbin{@} \mathbf{B} \in \mathbb{R}^{B \times m \times p}$$

$$C_{bij} = \sum_l A_{bil} B_{blj}$$

**Key**: the batch index $b$ is carried along unchanged. For each $b$, this is a standard 2D matmul:

$$\mathbf{C}[b] = \mathbf{A}[b] \cdot \mathbf{B}[b]$$

### Dimensional analysis rule for batched matmul

```
(B × m × k) @ (B × k × p) → (B × m × p)
  ↑              ↑              ↑
  batch          batch          batch
  matches        matches        preserved
```

**The batch dimensions must match.** The last two dimensions follow the standard matmul rule: inner dimensions ($k$) must match, outer dimensions ($m, p$) determine the output.

### PyTorch notation

```python
A = torch.randn(B, m, k)
B_mat = torch.randn(B, k, p)
C = A @ B_mat           # (B, m, p) — uses torch.bmm internally
C = torch.bmm(A, B_mat) # explicit batched matmul — identical
```

### Broadcasting: when batch dims don't match

PyTorch's `@` operator supports **broadcasting**: if one tensor has batch dimension 1 (or no batch dimension), it is broadcast:

```python
A = torch.randn(B, m, k)
W = torch.randn(k, p)     # no batch dimension
C = A @ W                  # (B, m, p) — W is reused for every batch
```

$$C_{bij} = \sum_l A_{bil} W_{lj}$$

This is exactly what happens in the linear projections: $\mathbf{Q} = \mathbf{X}\mathbf{W}_Q$ where $\mathbf{X}$ is batched but $\mathbf{W}_Q$ is shared across batches.

---

## 10.3 Transpose in 3D: Which Axes to Swap

### The critical question

In 2D, "transpose" is unambiguous: swap rows and columns. In 3D, there are **three** possible axis swaps:
- Swap axes 0 and 1: $(B, n, d) \to (n, B, d)$ — rarely useful
- Swap axes 0 and 2: $(B, n, d) \to (d, n, B)$ — rarely useful
- **Swap axes 1 and 2**: $(B, n, d) \to (B, d, n)$ — **THIS is the "transpose" for batched sequences**

### Why swap axes 1 and 2?

When we need $\mathbf{K}^T$ in the attention formula, we mean: transpose each $n \times d_k$ matrix independently. In 3D:

$$\mathbf{K} \in \mathbb{R}^{B \times n \times d_k} \xrightarrow{\text{transpose}(-2, -1)} \mathbf{K}^T \in \mathbb{R}^{B \times d_k \times n}$$

We swap the **last two axes** only. The batch dimension is untouched.

### PyTorch syntax

```python
K = torch.randn(B, n, d_k)
K_T = K.transpose(-2, -1)   # (B, d_k, n)
K_T = K.transpose(1, 2)     # equivalent
K_T = K.permute(0, 2, 1)    # equivalent using permute
```

### Dimensional verification: attention scores in 3D

$$\mathbf{S} = \mathbf{Q} \mathbin{@} \mathbf{K}^T$$

$$\underbrace{\mathbf{Q}}_{B \times n \times d_k} \mathbin{@} \underbrace{\mathbf{K}^T}_{B \times d_k \times n} = \underbrace{\mathbf{S}}_{B \times n \times n}$$

Each batch element gets its own $n \times n$ score matrix. ✓

---

## 10.4 3D Attention (Batched Single-Head)

Before multi-head, let's write the complete batched single-head attention:

```python
def batched_attention(X, W_Q, W_K, W_V):
    """
    X:   (B, n, d)
    W_Q: (d, d_k)   — shared across batch (broadcasting handles it)
    W_K: (d, d_k)
    W_V: (d, d_v)
    """
    # Projections: (B, n, d) @ (d, d_k) → (B, n, d_k) via broadcasting
    Q = X @ W_Q          # (B, n, d_k)
    K = X @ W_K          # (B, n, d_k)
    V = X @ W_V          # (B, n, d_v)
    
    d_k = Q.shape[-1]
    
    # Scores: (B, n, d_k) @ (B, d_k, n) → (B, n, n)
    S = Q @ K.transpose(-2, -1) / (d_k ** 0.5)
    
    # Softmax along last dim (over keys, for each query)
    A = F.softmax(S, dim=-1)    # (B, n, n)
    
    # Output: (B, n, n) @ (B, n, d_v) → (B, n, d_v)
    Y = A @ V                    # (B, n, d_v)
    
    return Y
```

**Every operation naturally extends to 3D** because:
1. Linear projections use broadcasting (weight matrix has no batch dim)
2. Score computation uses batched matmul
3. Softmax applies along `dim=-1` (independently per row, per batch)
4. Output matmul uses batched matmul

---

## 10.5 Multi-Head Attention: The Motivation

### Why multiple heads?

Single-head attention computes ONE set of attention weights per token pair. But a token might be relevant to another token in multiple ways:

- **Syntactic**: "The cat sat on the mat" — "sat" attends to "cat" (subject-verb)
- **Semantic**: "sat" attends to "mat" (location)
- **Positional**: "sat" attends to neighboring words

One attention head can only produce one weighting. **Multi-head attention** runs $h$ attention heads in parallel, each learning a different attention pattern.

### The naive approach (what you'd write first on paper)

Run $h$ independent attention functions and concatenate:

$$\text{head}_i = \text{Attention}(\mathbf{X}\mathbf{W}_Q^{(i)}, \mathbf{X}\mathbf{W}_K^{(i)}, \mathbf{X}\mathbf{W}_V^{(i)})$$

$$\text{MultiHead}(\mathbf{X}) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) \mathbf{W}_O$$

where:
- $\mathbf{W}_Q^{(i)} \in \mathbb{R}^{d \times d_k}$, typically $d_k = d/h$
- $\mathbf{W}_K^{(i)} \in \mathbb{R}^{d \times d_k}$
- $\mathbf{W}_V^{(i)} \in \mathbb{R}^{d \times d_v}$, typically $d_v = d/h$
- $\mathbf{W}_O \in \mathbb{R}^{hd_v \times d}$ — output projection

Each head produces $\mathbf{Y}_i \in \mathbb{R}^{n \times d_v}$. Concatenating $h$ heads: $[\mathbf{Y}_1; \ldots; \mathbf{Y}_h] \in \mathbb{R}^{n \times hd_v}$. With $d_v = d/h$, the concatenation is $n \times d$.

### The problem with the naive approach

Running $h$ separate attention computations means $h$ separate matmuls for Q, K, V projections. This is **slow** — GPUs prefer one large operation over many small ones.

### The efficient approach (the real trick)

**Key insight**: Instead of $h$ separate small projections, do ONE large projection and then **reshape** (split the head dimension out of the feature dimension).

---

## 10.6 The Reshape Trick: How Heads Are Actually Implemented

### Step 1: One big projection

Instead of $h$ separate $\mathbf{W}_Q^{(i)} \in \mathbb{R}^{d \times d_k}$, use one:

$$\mathbf{W}_Q \in \mathbb{R}^{d \times d} \quad (\text{where } d = h \cdot d_k)$$

$$\mathbf{Q}_\text{big} = \mathbf{X}\mathbf{W}_Q \in \mathbb{R}^{B \times n \times d}$$

This single matmul computes ALL heads' queries simultaneously. The last dimension contains the queries for all $h$ heads, concatenated.

### Step 2: Reshape to expose the head dimension

Split the last dimension $d$ into $(h, d_k)$:

$$\text{reshape}: \mathbb{R}^{B \times n \times d} \to \mathbb{R}^{B \times n \times h \times d_k}$$

```python
Q = Q_big.view(B, n, h, d_k)    # (B, n, h, d_k)
```

**What does this reshape mean mathematically?** It is just reindexing. If $Q_\text{big}[b, i, j]$ is the element at batch $b$, position $i$, feature $j$, and $j = g \cdot d_k + l$ where $g$ is the head index and $l$ is the within-head index, then:

$$Q[b, i, g, l] = Q_\text{big}[b, i, g \cdot d_k + l]$$

**No data is moved.** The reshape is a zero-cost reinterpretation of the memory layout.

### Step 3: Transpose to make heads a batch dimension

We need to compute attention **independently per head**. The batched matmul mechanism operates on the leading dimensions. So we need heads in a "batch-like" position:

$$\text{transpose}: \mathbb{R}^{B \times n \times h \times d_k} \to \mathbb{R}^{B \times h \times n \times d_k}$$

```python
Q = Q.transpose(1, 2)    # (B, h, n, d_k)
```

**Why this specific transpose?** We are choosing to:
- Keep $B$ in position 0 (batch stays first)
- Move $h$ to position 1 (heads become a "second batch dimension")
- Keep $n$ and $d_k$ as the last two dims (so standard BMM works on them)

Now the tensor is $B \times h \times n \times d_k$. The `@` operator will batch over dimensions 0 and 1 (both $B$ and $h$), and matmul over dimensions 2 and 3.

### The complete coordinate transformation (what happens to each element)

$$Q[b, i, g, l] \xrightarrow{\text{transpose}(1,2)} Q'[b, g, i, l]$$

In the new layout, $Q'[b, g]$ is a 2D matrix of shape $n \times d_k$ — the queries for head $g$ in batch element $b$. This is exactly what we need for the 2D attention computation.

---

## 10.7 4D Batched Matmul: How It All Fits Together

### Score computation with 4D tensors

After reshaping and transposing:

$$\mathbf{Q} \in \mathbb{R}^{B \times h \times n \times d_k}, \quad \mathbf{K} \in \mathbb{R}^{B \times h \times n \times d_k}$$

To compute $\mathbf{Q}\mathbf{K}^T$ per head per batch:

$$\mathbf{K}^T: \text{transpose}(-2, -1) \Rightarrow \mathbb{R}^{B \times h \times d_k \times n}$$

$$\mathbf{S} = \mathbf{Q} \mathbin{@} \mathbf{K}^T \in \mathbb{R}^{B \times h \times n \times n}$$

**How does 4D matmul work?** PyTorch's `@` batches over ALL leading dimensions:

$$S[b, g, i, j] = \sum_l Q[b, g, i, l] \cdot K[b, g, j, l]$$

The batch dimensions $(B, h)$ are spectators. The matmul acts on the last two dimensions. This gives us an $n \times n$ attention matrix for each head and each batch element.

### Full dimension trace through multi-head attention

| Step | Operation | Shape | Notes |
|------|-----------|-------|-------|
| Input | $\mathbf{X}$ | $B \times n \times d$ | |
| Q projection | $\mathbf{X}\mathbf{W}_Q$ | $B \times n \times d$ | Broadcasting: $(d, d)$ weights |
| Reshape | `.view(B,n,h,d_k)` | $B \times n \times h \times d_k$ | Split heads from features |
| Transpose | `.transpose(1,2)` | $B \times h \times n \times d_k$ | Heads become batch-like |
| K transpose | `K.transpose(-2,-1)` | $B \times h \times d_k \times n$ | For score matmul |
| Scores | $\mathbf{Q} @ \mathbf{K}^T / \sqrt{d_k}$ | $B \times h \times n \times n$ | Per-head attention scores |
| Softmax | `softmax(dim=-1)` | $B \times h \times n \times n$ | Over keys dimension |
| Attend | $\mathbf{A} @ \mathbf{V}$ | $B \times h \times n \times d_v$ | Per-head output |
| Transpose back | `.transpose(1,2)` | $B \times n \times h \times d_v$ | Undo head transpose |
| Reshape | `.contiguous().view(B,n,d)` | $B \times n \times d$ | Merge heads back |
| Output proj | $\cdot \mathbf{W}_O$ | $B \times n \times d$ | Final linear |

---

## 10.8 The Output Projection: Why and How

### Why concatenate then project?

After multi-head attention, we have $h$ sets of outputs, each of dimension $d_v = d/h$. Concatenation gives us a $d$-dimensional vector. The output projection $\mathbf{W}_O \in \mathbb{R}^{d \times d}$ serves two purposes:

1. **Mix information across heads**: Without $\mathbf{W}_O$, the heads are independent. The output projection allows the model to combine insights from different heads.

2. **Parameter budget**: The total parameters are $(3 \times d \times d + d \times d) = 4d^2$ regardless of $h$. The number of heads is a free parameter that doesn't change the total compute/parameter count (approximately).

### The projection

$$\mathbf{Y}_\text{final} = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) \mathbf{W}_O$$

In code, this is just another matmul after reshaping:

```python
# After transpose back and reshape: (B, n, d)
Y_final = Y_concat @ W_O    # (B, n, d) @ (d, d) → (B, n, d)
```

---

## 10.9 Why `.contiguous()` Appears (Memory Layout)

### The problem

After `.transpose(1, 2)`, the tensor's memory layout is **non-contiguous**. The `.view()` operation requires contiguous memory. Hence the pattern:

```python
Y.transpose(1, 2).contiguous().view(B, n, d)
```

### What contiguous means

A tensor is contiguous if its elements are laid out in memory in the order you'd traverse them by iterating the last dimension fastest. After a transpose, the logical order changes but the physical memory doesn't move — the tensor just stores different strides.

`.contiguous()` copies the data into a new memory layout matching the logical order. This is a real memory copy — $O(BnHd_k)$ — but it's necessary for the subsequent reshape.

**Alternative**: Use `.reshape()` instead of `.view()`, which handles non-contiguous tensors automatically (it calls `.contiguous()` internally if needed).

---

## 10.10 Complete Multi-Head Attention Implementation

```python
import torch
import torch.nn.functional as F

def multi_head_attention(X, W_Q, W_K, W_V, W_O, h):
    """
    Multi-head attention from scratch, with full dimension tracking.
    
    X:   (B, n, d) — input
    W_Q: (d, d)    — query weights (all heads packed)
    W_K: (d, d)    — key weights
    W_V: (d, d)    — value weights
    W_O: (d, d)    — output projection
    h:   int       — number of heads
    """
    B, n, d = X.shape
    d_k = d // h    # per-head dimension
    
    # === STEP 1: Project (one big matmul per Q, K, V) ===
    Q = X @ W_Q      # (B, n, d)
    K = X @ W_K      # (B, n, d)
    V = X @ W_V      # (B, n, d)
    
    # === STEP 2: Split heads ===
    # Reshape: (B, n, d) → (B, n, h, d_k)
    Q = Q.view(B, n, h, d_k)
    K = K.view(B, n, h, d_k)
    V = V.view(B, n, h, d_k)
    
    # Transpose: (B, n, h, d_k) → (B, h, n, d_k)
    Q = Q.transpose(1, 2)
    K = K.transpose(1, 2)
    V = V.transpose(1, 2)
    
    # === STEP 3: Compute attention (now it's just batched 2D attention) ===
    # Scores: (B, h, n, d_k) @ (B, h, d_k, n) → (B, h, n, n)
    S = Q @ K.transpose(-2, -1) / (d_k ** 0.5)
    
    # Attention weights: softmax over last dim
    A = F.softmax(S, dim=-1)     # (B, h, n, n)
    
    # Attend: (B, h, n, n) @ (B, h, n, d_k) → (B, h, n, d_k)
    Y = A @ V                     # (B, h, n, d_k)
    
    # === STEP 4: Merge heads ===
    # Transpose back: (B, h, n, d_k) → (B, n, h, d_k)
    Y = Y.transpose(1, 2)
    
    # Reshape: (B, n, h, d_k) → (B, n, d)
    Y = Y.contiguous().view(B, n, d)
    
    # === STEP 5: Output projection ===
    Y = Y @ W_O                   # (B, n, d)
    
    return Y

# Test
torch.manual_seed(42)
B, n, d, h = 2, 6, 16, 4
d_k = d // h  # 4

X = torch.randn(B, n, d)
W_Q = torch.randn(d, d)
W_K = torch.randn(d, d)
W_V = torch.randn(d, d)
W_O = torch.randn(d, d)

Y = multi_head_attention(X, W_Q, W_K, W_V, W_O, h)
print(f"Input:  {X.shape}")   # (2, 6, 16)
print(f"Output: {Y.shape}")   # (2, 6, 16)
```

### Verifying against `nn.MultiheadAttention`

```python
import torch.nn as nn

# Create PyTorch's MHA and copy our weights into it
mha = nn.MultiheadAttention(embed_dim=d, num_heads=h, batch_first=True, bias=False)

# PyTorch packs W_Q, W_K, W_V into one matrix: in_proj_weight (3d × d)
with torch.no_grad():
    mha.in_proj_weight.copy_(torch.cat([W_Q.T, W_K.T, W_V.T], dim=0))
    mha.out_proj.weight.copy_(W_O.T)

Y_pytorch, _ = mha(X, X, X)  # self-attention: Q=K=V source is X
Y_ours = multi_head_attention(X, W_Q, W_K, W_V, W_O, h)

print(f"Match: {torch.allclose(Y_ours, Y_pytorch, atol=1e-5)}")
```

---

## 10.11 Dimension Reasoning Framework: A General Method

When you encounter a new tensor operation and need to figure out which dimensions to transpose, reshape, or permute, use this systematic approach:

### Rule 1: Start from the element-wise definition

Write what the output element $Y[b, i, j, ...]$ should equal in terms of input elements. This removes all ambiguity.

**Example**: Attention scores
$$S[b, g, i, j] = \frac{1}{\sqrt{d_k}} \sum_l Q[b, g, i, l] \cdot K[b, g, j, l]$$

This IS a matmul over the $l$ index, with $b, g$ as batch dimensions, $i$ as "row", and $j$ as "column" (but $j$ indexes into $K$'s first non-batch dim, which holds $K$ transposed).

### Rule 2: Identify which indices are summed over (contracted)

The summed index determines which dimensions must be "inner" in the matmul:
- $\sum_l (\ldots)_{il} \cdot (\ldots)_{lj}$: standard matmul — inner dims match
- $\sum_l (\ldots)_{il} \cdot (\ldots)_{jl}$: need to transpose the second operand so $l$ becomes the inner dim

### Rule 3: Identify spectator (batch) dimensions

Any index that appears in both inputs AND the output without being summed is a batch dimension. It must be in a leading position for `@` to work.

### Rule 4: Use shape equations to verify

Write the shapes and confirm the matmul is valid:

```
(B, h, n, d_k) @ (B, h, d_k, n) → (B, h, n, n)
         ^^              ^^
         inner dims match
```

---

## 10.12 Einsum: The Universal Language for Tensor Operations

When dimension manipulation gets confusing, `torch.einsum` lets you specify operations using the element-wise definition directly:

```python
# Standard attention scores: S[b,g,i,j] = sum_l Q[b,g,i,l] * K[b,g,j,l]
S = torch.einsum('bgil,bgjl->bgij', Q, K) / (d_k ** 0.5)

# This is equivalent to:
S = Q @ K.transpose(-2, -1) / (d_k ** 0.5)
```

```python
# Attention output: Y[b,g,i,l] = sum_j A[b,g,i,j] * V[b,g,j,l]
Y = torch.einsum('bgij,bgjl->bgil', A, V)

# Equivalent to:
Y = A @ V
```

### When einsum is essential

For operations that DON'T map cleanly to matmul + transpose:

```python
# Bilinear form: Y[b,i,j] = sum_k sum_l Q[b,i,k] * W[k,l] * K[b,j,l]
Y = torch.einsum('bik,kl,bjl->bij', Q, W, K)
```

This would require multiple matmuls and transposes to express without einsum.

### Einsum notation rules

- Each tensor's indices are listed (e.g., `bgil` for a 4D tensor)
- Repeated indices in the inputs are summed over (unless they appear in the output)
- Output indices specify the shape of the result
- `->` separates inputs from output

```python
# Examples translating between notations:
# Matrix multiply: C_ij = sum_k A_ik B_kj
C = torch.einsum('ik,kj->ij', A, B)

# Batched matmul: C_bij = sum_k A_bik B_bkj  
C = torch.einsum('bik,bkj->bij', A, B)

# Outer product: C_ij = a_i * b_j
C = torch.einsum('i,j->ij', a, b)

# Trace: t = sum_i A_ii
t = torch.einsum('ii->', A)

# Batch trace: t_b = sum_i A_bii
t = torch.einsum('bii->b', A)
```

---

## 10.13 The Attention Mask: Handling Variable-Length and Causal Constraints

### Causal (autoregressive) mask

In language modeling, token $i$ should only attend to tokens $j \leq i$ (can't see the future). We enforce this by setting future scores to $-\infty$ before softmax:

$$S'_{ij} = \begin{cases} S_{ij} & \text{if } j \leq i \\ -\infty & \text{if } j > i \end{cases}$$

Since $\text{softmax}(-\infty) = 0$, future tokens get zero attention weight.

```python
# Create causal mask: upper triangular = True (positions to mask)
mask = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)

# Apply mask before softmax
S_masked = S.masked_fill(mask, float('-inf'))  # broadcasting handles (B, h, n, n)
A = F.softmax(S_masked, dim=-1)
```

### Dimensional analysis of the mask

The mask has shape $(n, n)$ or $(1, 1, n, n)$ (broadcast over $B, h$). This is a boolean tensor — it doesn't need batch or head dimensions because the same causal structure applies to all.

### Padding mask

For variable-length sequences in a batch, some positions are padding. We mask them in the key dimension:

```python
# padding_mask: (B, n) — True where token is padding
# Expand for broadcast: (B, 1, 1, n) to match (B, h, n, n) scores
key_padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, n)
S_masked = S.masked_fill(key_padding_mask, float('-inf'))
```

---

## 10.14 Computation Graph: Complete Multi-Head Attention

```
     X (B, n, d)
     ├──────────────────┬──────────────────┐
     ↓                  ↓                  ↓
   [@ W_Q]            [@ W_K]            [@ W_V]
     ↓                  ↓                  ↓
   Q_big              K_big              V_big
  (B,n,d)            (B,n,d)            (B,n,d)
     ↓                  ↓                  ↓
  [view]              [view]              [view]
  (B,n,h,d_k)        (B,n,h,d_k)        (B,n,h,d_k)
     ↓                  ↓                  ↓
  [transpose(1,2)]    [transpose(1,2)]    [transpose(1,2)]
     ↓                  ↓                  ↓
     Q                  K                  V
  (B,h,n,d_k)        (B,h,n,d_k)        (B,h,n,d_k)
     │                  │                  │
     │          [transpose(-2,-1)]         │
     │                  ↓                  │
     │                 K^T                 │
     │            (B,h,d_k,n)              │
     │                  │                  │
     └──── [bmm] ──────┘                   │
              ↓                            │
           S = QK^T                        │
         (B,h,n,n)                         │
              ↓                            │
          [/ √d_k]                         │
              ↓                            │
         [+ mask]  (optional)              │
              ↓                            │
         [softmax]                         │
              ↓                            │
             A                             │
          (B,h,n,n)                        │
              │                            │
              └────── [bmm] ──────────────┘
                        ↓
                    Y_heads
                  (B,h,n,d_k)
                        ↓
                 [transpose(1,2)]
                        ↓
                  (B,n,h,d_k)
                        ↓
                   [contiguous]
                        ↓
                [view → (B,n,d)]
                        ↓
                    [@ W_O]
                        ↓
                     Y_final
                    (B, n, d)
```

---

## 10.15 Summary: How to Reason About Multi-Head Attention Dimensions

The researcher's paper trail for multi-head attention:

1. **Start 2D**: Derive attention for $\mathbf{X} \in \mathbb{R}^{n \times d}$ (Doc 9).

2. **Add batch**: Prepend $B$ dimension. All 2D matmuls become 3D batched matmuls. Batch is a spectator.

3. **Add heads via reshape**: Instead of $h$ separate Q projections, do one big one and split: `view(B, n, h, d_k)`.

4. **Move heads to batch position**: `transpose(1, 2)` → $(B, h, n, d_k)$. Now heads and batch are both spectators.

5. **Standard attention on last-2 dims**: Everything from the 2D case applies to the last two dimensions.

6. **Reverse the reshape**: `transpose(1, 2)` → `contiguous()` → `view(B, n, d)`.

7. **Output projection**: One final matmul to mix across heads.

**The general principle**: Whenever you have multiple independent copies of an operation (heads), reshape the tensor so the copy index becomes a batch dimension, run the operation as a batched matmul, then reshape back.

---

## 10.16 Exercises

1. **Dimension trace**: Write the shape at every step of multi-head attention for $B=4, n=32, d=64, h=8$. Verify every matmul is valid.

2. **Einsum version**: Rewrite the entire multi-head attention using only `torch.einsum` (no `.transpose()`, no `@`). Compare outputs.

3. **Contiguous investigation**: After `Q.view(B,n,h,d_k).transpose(1,2)`, check `Q.is_contiguous()`. What are the strides? Why does this matter for `.view()`?

4. **Custom head dimensions**: Modify the implementation so that $d_k \neq d_v$ (e.g., $d_k = d/(2h)$, $d_v = d/h$). Track how all shapes change.

5. **Causal mask correctness**: Implement causal masking and verify that for a 4-token sequence, token 0 attends only to itself, token 1 to tokens 0-1, etc. Print the attention weights matrix.
