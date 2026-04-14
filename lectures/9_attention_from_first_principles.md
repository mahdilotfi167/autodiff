# 9. Deriving Attention from First Principles

## Motivation

You now have the complete toolkit: computation graphs (Doc 1), Jacobians and the chain rule (Doc 3), matmul gradient formulas (Doc 4), softmax gradients (Doc 5), the trace trick (Doc 7), and custom gradient implementation (Doc 8). With these tools, we can **derive** the attention mechanism from scratch — not just state the formula, but understand the **design process**: what problem the authors were solving, what they wrote on paper, which design choices they explored, and how dimensional analysis forced certain transpositions and reshapes.

This document reconstructs the thinking process that leads to the "Scaled Dot-Product Attention" formula. By the end, you will be able to derive it yourself from a blank sheet of paper.

---

## 9.1 The Problem Statement (What Goes on the Paper First)

### What do we want?

We have a **sequence** of vectors. Think of a sentence: each word (token) is represented as a vector. We want a mechanism where **each token can look at every other token** and selectively gather information.

**Input**: A sequence of $n$ vectors, each of dimension $d$:

$$\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_n \in \mathbb{R}^d$$

Stack them as rows of a matrix:

$$\mathbf{X} \in \mathbb{R}^{n \times d}$$

where row $i$ is the representation of token $i$.

**Desired output**: A new sequence of $n$ vectors, where each output vector is a **context-dependent mixture** of the input vectors:

$$\mathbf{Y} \in \mathbb{R}^{n \times d}$$

where each row $\mathbf{y}_i$ "knows about" other tokens in the sequence — weighted by how relevant they are to token $i$.

### The design question on paper

The researcher writes:

> "I want $\mathbf{y}_i = \sum_{j=1}^{n} \alpha_{ij} \cdot (\text{some function of } \mathbf{x}_j)$"

where $\alpha_{ij} \geq 0$ and $\sum_j \alpha_{ij} = 1$ (a probability distribution over tokens for each output position).

**This is the starting point.** Everything that follows is answering two questions:
1. How do we compute the weights $\alpha_{ij}$?
2. What "function of $\mathbf{x}_j$" do we use?

---

## 9.2 Step 1: The Similarity Function (What Should $\alpha_{ij}$ Measure?)

### First attempt: raw dot product

The simplest way to measure "how relevant is token $j$ to token $i$" is the dot product of their representations:

$$e_{ij} = \mathbf{x}_i^T \mathbf{x}_j$$

This is a scalar measuring similarity. If $\mathbf{x}_i$ and $\mathbf{x}_j$ point in similar directions, $e_{ij}$ is large.

### Writing this for ALL pairs simultaneously (the matrix view)

On paper, the researcher realizes: I need $e_{ij}$ for **all** $i, j$ pairs. That is an $n \times n$ matrix:

$$\mathbf{E} = \mathbf{X} \mathbf{X}^T \in \mathbb{R}^{n \times n}$$

**Dimensional analysis on paper:**

$$\underbrace{\mathbf{X}}_{n \times d} \cdot \underbrace{\mathbf{X}^T}_{d \times n} = \underbrace{\mathbf{E}}_{n \times n}$$

The $(i, j)$ entry of $\mathbf{E}$ is:

$$E_{ij} = \sum_{k=1}^d X_{ik} X_{jk} = \mathbf{x}_i^T \mathbf{x}_j \quad \checkmark$$

**This is the first "aha" moment**: computing ALL pairwise similarities is just a single matmul with the transpose.

### Why $\mathbf{X}\mathbf{X}^T$ and not $\mathbf{X}^T\mathbf{X}$?

This is a question the researcher must answer on paper:

- $\mathbf{X}\mathbf{X}^T$ has shape $n \times n$ — an entry for each pair of **tokens**. This is what we want.
- $\mathbf{X}^T\mathbf{X}$ has shape $d \times d$ — an entry for each pair of **features**. This is a Gram matrix of features, not what we need.

**Rule written on paper**: "When tokens are rows, token-to-token similarity = $\mathbf{X}\mathbf{X}^T$."

---

## 9.3 Step 2: From Similarity Scores to Weights (Why Softmax?)

### The raw scores $e_{ij}$ can be anything

The dot products $e_{ij}$ range over $(-\infty, +\infty)$. We need valid **weights**: non-negative and summing to 1 over $j$ for each $i$.

**The researcher writes three options on paper:**

1. **Hardmax**: Pick the single most relevant token. $\alpha_{ij} = 1$ if $j = \arg\max_k e_{ik}$, else 0.
   - Problem: not differentiable, loses information from other tokens.

2. **Normalize**: $\alpha_{ij} = e_{ij} / \sum_k e_{ik}$
   - Problem: $e_{ij}$ can be negative, so $\alpha_{ij}$ can be negative. Not a valid probability.

3. **Softmax**: $\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_k \exp(e_{ik})}$
   - Always positive ✓
   - Sums to 1 ✓
   - Differentiable ✓
   - Approaches hardmax as scores become extreme ✓
   - We already know its gradient (Doc 5) ✓

**Decision**: softmax wins.

### Matrix form (applied row-wise)

$$\mathbf{A} = \text{softmax}_{\text{row}}(\mathbf{E}) \in \mathbb{R}^{n \times n}$$

Where $\text{softmax}_{\text{row}}$ means: apply softmax independently to each row. Row $i$ of $\mathbf{A}$ is a probability distribution over all $n$ tokens.

### Current formula on paper

$$\mathbf{Y} = \mathbf{A} \mathbf{X} = \text{softmax}(\mathbf{X}\mathbf{X}^T) \cdot \mathbf{X}$$

**Dimensional check:**

$$\underbrace{\mathbf{A}}_{n \times n} \cdot \underbrace{\mathbf{X}}_{n \times d} = \underbrace{\mathbf{Y}}_{n \times d} \quad \checkmark$$

Each output row: $\mathbf{y}_i = \sum_j \alpha_{ij} \mathbf{x}_j$ — a weighted average of all input vectors. ✓

---

## 9.4 Step 3: The Q/K/V Decomposition (The Key Design Insight)

### The problem with $\text{softmax}(\mathbf{X}\mathbf{X}^T)\mathbf{X}$

The researcher stares at $\mathbf{Y} = \text{softmax}(\mathbf{X}\mathbf{X}^T)\mathbf{X}$ and notices problems:

1. **Same representation for three different roles.** Token $i$ plays three roles simultaneously:
   - As the **query** (the one looking for relevant tokens): used as $\mathbf{x}_i$ in $e_{ij} = \mathbf{x}_i^T \mathbf{x}_j$
   - As the **key** (the one being compared against): used as $\mathbf{x}_j$ in $e_{ij} = \mathbf{x}_i^T \mathbf{x}_j$
   - As the **value** (the information to gather): used in $\mathbf{y}_i = \sum_j \alpha_{ij} \mathbf{x}_j$

   A single vector $\mathbf{x}_i$ must simultaneously encode "what I'm looking for", "what I can be found by", and "what information I carry." This is too constraining.

2. **No learnable parameters.** The formula $\text{softmax}(\mathbf{X}\mathbf{X}^T)\mathbf{X}$ has **zero** trainable weights. It is entirely determined by the input. We want the network to **learn** what to attend to.

3. **Symmetry problem.** Since $e_{ij} = \mathbf{x}_i^T \mathbf{x}_j = \mathbf{x}_j^T \mathbf{x}_i = e_{ji}$, the attention weights before softmax are symmetric. But relevance is not always symmetric ("the cat sat on **the mat**" — "mat" might attend to "cat" differently than "cat" attends to "mat").

### The solution: three separate linear projections

The researcher writes on paper:

> "Let me project $\mathbf{X}$ into three different spaces — one for each role."

$$\mathbf{Q} = \mathbf{X}\mathbf{W}_Q, \quad \mathbf{K} = \mathbf{X}\mathbf{W}_K, \quad \mathbf{V} = \mathbf{X}\mathbf{W}_V$$

where:
- $\mathbf{W}_Q \in \mathbb{R}^{d \times d_k}$ — projects into **query** space
- $\mathbf{W}_K \in \mathbb{R}^{d \times d_k}$ — projects into **key** space
- $\mathbf{W}_V \in \mathbb{R}^{d \times d_v}$ — projects into **value** space

**Dimensional analysis:**

$$\underbrace{\mathbf{X}}_{n \times d} \cdot \underbrace{\mathbf{W}_Q}_{d \times d_k} = \underbrace{\mathbf{Q}}_{n \times d_k}$$

Same for $\mathbf{K}$ (shape $n \times d_k$) and $\mathbf{V}$ (shape $n \times d_v$).

### Why $d_k$ can differ from $d$

The query/key dimension $d_k$ only needs to be large enough to compute meaningful similarities. It does not need to match the model dimension $d$. The value dimension $d_v$ determines the output dimension (which may differ, though in practice often $d_v = d_k = d/h$ for multi-head).

### Updated formula

$$\mathbf{Y} = \text{softmax}\left(\mathbf{Q}\mathbf{K}^T\right) \mathbf{V}$$

**Dimensional analysis (critical — this is what forces the transpose):**

Score matrix: $\underbrace{\mathbf{Q}}_{n \times d_k} \cdot \underbrace{\mathbf{K}^T}_{d_k \times n} = \underbrace{\mathbf{S}}_{n \times n}$

**Why must we transpose $\mathbf{K}$?** Let's check what happens without it:
- $\mathbf{Q}\mathbf{K}$ would be $(n \times d_k)(n \times d_k)$ — **dimensions don't match for matmul!**
- $\mathbf{Q}^T\mathbf{K}$ would be $(d_k \times n)(n \times d_k) = d_k \times d_k$ — wrong shape (feature-to-feature, not token-to-token).
- $\mathbf{Q}\mathbf{K}^T = (n \times d_k)(d_k \times n) = n \times n$ — **correct!** Token-to-token similarity matrix.

**This is how the transpose is determined**: it is the ONLY arrangement that gives an $n \times n$ score matrix from two $n \times d_k$ matrices. The researcher doesn't arbitrarily choose to transpose; dimensional analysis **forces** it.

Output: $\underbrace{\text{softmax}(\mathbf{S})}_{n \times n} \cdot \underbrace{\mathbf{V}}_{n \times d_v} = \underbrace{\mathbf{Y}}_{n \times d_v}$

### What $S_{ij} = \mathbf{q}_i^T \mathbf{k}_j$ means element-wise

$$S_{ij} = \sum_{l=1}^{d_k} Q_{il} K_{jl} = \mathbf{q}_i^T \mathbf{k}_j$$

This is the dot product between the query of token $i$ and the key of token $j$. High score = token $j$ is relevant to token $i$.

---

## 9.5 Step 4: The Scaling Factor (Why $\frac{1}{\sqrt{d_k}}$?)

### The problem the researcher encounters

The researcher implements $\text{softmax}(\mathbf{Q}\mathbf{K}^T)\mathbf{V}$ and notices: **as $d_k$ grows, the dot products get larger, softmax saturates, and gradients vanish.**

### Analysis on paper

If the entries of $\mathbf{Q}$ and $\mathbf{K}$ are i.i.d. with mean 0 and variance 1:

$$S_{ij} = \sum_{l=1}^{d_k} Q_{il} K_{jl}$$

Each term $Q_{il} K_{jl}$ has:
- Mean: $\mathbb{E}[Q_{il} K_{jl}] = \mathbb{E}[Q_{il}] \cdot \mathbb{E}[K_{jl}] = 0$ (independent, zero-mean)
- Variance: $\text{Var}(Q_{il} K_{jl}) = \text{Var}(Q_{il}) \cdot \text{Var}(K_{jl}) = 1 \cdot 1 = 1$ (for zero-mean independent variables)

The sum of $d_k$ such independent terms has:
- Mean: $\mathbb{E}[S_{ij}] = 0$
- Variance: $\text{Var}(S_{ij}) = d_k$
- Standard deviation: $\text{std}(S_{ij}) = \sqrt{d_k}$

**So the scores grow as $\sqrt{d_k}$.** For $d_k = 64$, scores have std ≈ 8. For $d_k = 512$, scores have std ≈ 22.6.

When softmax receives inputs with large magnitude, it pushes almost all mass onto the maximum element (approaches one-hot). The gradient of softmax at near-one-hot outputs is nearly zero (from Doc 5: $\frac{\partial \text{softmax}_i}{\partial z_j} = \text{softmax}_i(\delta_{ij} - \text{softmax}_j)$, which is tiny when one softmax output is near 1).

### The fix: normalize by $\sqrt{d_k}$

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}$$

After scaling, $\text{Var}(S_{ij}/\sqrt{d_k}) = d_k / d_k = 1$. The scores have unit variance regardless of $d_k$.

### Why not learnable scaling?

The researcher considers $\text{softmax}(\beta \cdot \mathbf{Q}\mathbf{K}^T)$ with learnable $\beta$, but:
- The fixed $1/\sqrt{d_k}$ is a near-optimal initialization
- A learnable temperature adds complexity with minimal benefit
- The Q/K projections can implicitly learn any scaling through their weight norms

---

## 9.6 The Complete Attention Computation Graph

### Final formula (single head, no batch)

$$\mathbf{Y} = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}$$

### The computation graph (what the framework builds)

```
   X (n × d)
   ├──────────────┬──────────────┐
   ↓              ↓              ↓
 [× W_Q]       [× W_K]       [× W_V]
   ↓              ↓              ↓
   Q              K              V
 (n×d_k)       (n×d_k)       (n×d_v)
   │              │              │
   │        [transpose]          │
   │              ↓              │
   │            K^T              │
   │          (d_k×n)            │
   │              │              │
   └─── [matmul] ─┘              │
          ↓                      │
         S = Q K^T               │
        (n × n)                  │
          ↓                      │
      [÷ √d_k]                  │
          ↓                      │
      S / √d_k                  │
        (n × n)                  │
          ↓                      │
      [softmax_row]              │
          ↓                      │
         A                       │
       (n × n)                   │
          │                      │
          └────── [matmul] ──────┘
                     ↓
                    Y = A V
                  (n × d_v)
```

### Step-by-step with concrete dimensions (example: $n=4, d=8, d_k=d_v=8$)

| Step | Operation | Input Shapes | Output Shape | # Multiplies |
|------|-----------|-------------|--------------|-------------|
| 1 | $\mathbf{Q} = \mathbf{X}\mathbf{W}_Q$ | $(4 \times 8)(8 \times 8)$ | $4 \times 8$ | 256 |
| 2 | $\mathbf{K} = \mathbf{X}\mathbf{W}_K$ | $(4 \times 8)(8 \times 8)$ | $4 \times 8$ | 256 |
| 3 | $\mathbf{V} = \mathbf{X}\mathbf{W}_V$ | $(4 \times 8)(8 \times 8)$ | $4 \times 8$ | 256 |
| 4 | $\mathbf{S} = \mathbf{Q}\mathbf{K}^T$ | $(4 \times 8)(8 \times 4)$ | $4 \times 4$ | 128 |
| 5 | Scale: $\mathbf{S}/\sqrt{8}$ | $4 \times 4$ | $4 \times 4$ | 16 (scalar div) |
| 6 | $\mathbf{A} = \text{softmax}(\mathbf{S}/\sqrt{d_k})$ | $4 \times 4$ | $4 \times 4$ | ~48 (exp + sum) |
| 7 | $\mathbf{Y} = \mathbf{A}\mathbf{V}$ | $(4 \times 4)(4 \times 8)$ | $4 \times 8$ | 128 |

### Naming the intermediate nodes for backprop

Label every edge in the graph:

$$\mathbf{Q} = \mathbf{X}\mathbf{W}_Q$$
$$\mathbf{K} = \mathbf{X}\mathbf{W}_K$$
$$\mathbf{V} = \mathbf{X}\mathbf{W}_V$$
$$\mathbf{S} = \mathbf{Q}\mathbf{K}^T$$
$$\mathbf{S}' = \mathbf{S} / \sqrt{d_k}$$
$$\mathbf{A} = \text{softmax}_\text{row}(\mathbf{S}')$$
$$\mathbf{Y} = \mathbf{A}\mathbf{V}$$

Each line is ONE node in the computation graph. We know the gradient rule for each:
- Matrix multiply: Doc 4
- Scalar division: element-wise (Doc 4)
- Softmax: Doc 5
- Transpose: trivial (gradient of transpose is transpose of gradient)

---

## 9.7 Implementation: Verifying the Derivation

```python
import torch
import torch.nn.functional as F

def attention_from_scratch(X, W_Q, W_K, W_V):
    """
    Attention derived step-by-step, matching our computation graph.
    
    X: (n, d) — input sequence
    W_Q: (d, d_k) — query projection
    W_K: (d, d_k) — key projection  
    W_V: (d, d_v) — value projection
    """
    # Step 1-3: Linear projections
    Q = X @ W_Q          # (n, d_k)
    K = X @ W_K          # (n, d_k)
    V = X @ W_V          # (n, d_v)
    
    d_k = Q.shape[-1]
    
    # Step 4: Score matrix (here is where K^T appears)
    S = Q @ K.T           # (n, n) — token-to-token similarity
    
    # Step 5: Scale
    S_scaled = S / (d_k ** 0.5)
    
    # Step 6: Softmax (row-wise)
    A = F.softmax(S_scaled, dim=-1)   # (n, n)
    
    # Step 7: Weighted sum of values
    Y = A @ V             # (n, d_v)
    
    return Y, A  # Return attention weights for inspection

# Test
torch.manual_seed(42)
n, d, d_k, d_v = 4, 8, 8, 8

X = torch.randn(n, d)
W_Q = torch.randn(d, d_k)
W_K = torch.randn(d, d_k)
W_V = torch.randn(d, d_v)

Y, A = attention_from_scratch(X, W_Q, W_K, W_V)
print(f"Input shape:  {X.shape}")       # (4, 8)
print(f"Output shape: {Y.shape}")       # (4, 8)
print(f"Attention matrix shape: {A.shape}")  # (4, 4)
print(f"\nAttention weights (each row sums to 1):")
print(A)
print(f"Row sums: {A.sum(dim=-1)}")     # [1, 1, 1, 1]
```

### Verifying against PyTorch's built-in

```python
# PyTorch's scaled_dot_product_attention does the same thing
# (it expects Q, K, V directly)
Q = X @ W_Q
K = X @ W_K
V = X @ W_V

Y_pytorch = F.scaled_dot_product_attention(Q, K, V)
Y_ours, _ = attention_from_scratch(X, W_Q, W_K, W_V)

print(f"\nOur implementation matches PyTorch: {torch.allclose(Y_ours, Y_pytorch, atol=1e-6)}")
```

---

## 9.8 The Design Space: Alternatives the Authors Considered

Understanding **why** the authors chose this specific design requires seeing what they rejected.

### Alternative similarity functions

| Method | Formula | Pros | Cons |
|--------|---------|------|------|
| Dot product | $\mathbf{q}_i^T \mathbf{k}_j$ | Fast, simple matmul | Scales with $d_k$ |
| Scaled dot product | $\mathbf{q}_i^T \mathbf{k}_j / \sqrt{d_k}$ | **Selected** | — |
| Additive (Bahdanau) | $\mathbf{v}^T \tanh(\mathbf{W}_1 \mathbf{q}_i + \mathbf{W}_2 \mathbf{k}_j)$ | More expressive | Slower (can't use batched matmul) |
| Multiplicative (general) | $\mathbf{q}_i^T \mathbf{W} \mathbf{k}_j$ | Learnable metric | Extra parameter matrix |
| Cosine | $\frac{\mathbf{q}_i^T \mathbf{k}_j}{\|\mathbf{q}_i\| \|\mathbf{k}_j\|}$ | Bounded | Norm computation overhead |

The scaled dot product wins because **it can be computed as a single batched matrix multiply**, which GPUs optimize extremely well.

### Why not absolute position in the similarity?

Early attention (Bahdanau, 2014) was between an encoder and decoder — cross-attention. The "Attention Is All You Need" paper (Vaswani et al., 2017) introduced **self-attention** where the same sequence serves as queries, keys, and values. Position information is injected separately via positional encodings (covered in Doc 11).

---

## 9.9 Complexity Analysis

### Time complexity

The dominant cost is the score matrix computation and the attention-value product:

- $\mathbf{Q}\mathbf{K}^T$: $O(n^2 d_k)$
- $\mathbf{A}\mathbf{V}$: $O(n^2 d_v)$
- Linear projections: $O(n d \cdot d_k)$ each

**Total**: $O(n^2 d)$ — **quadratic in sequence length**. This is THE bottleneck that drives research into efficient attention (linear attention, sparse attention, FlashAttention, etc.).

### Memory complexity

- Score matrix $\mathbf{S}$: $O(n^2)$ — must store for softmax backward
- Attention weights $\mathbf{A}$: $O(n^2)$ — must store for value matmul backward

For $n = 4096$, $\mathbf{A}$ alone takes $4096^2 \times 4$ bytes ≈ 67MB per head per batch element. This motivates FlashAttention (which avoids materializing the full $n \times n$ matrix).

---

## 9.10 Summary: The Paper Trail

Here is the sequence of steps a researcher writes on paper to arrive at the attention formula:

1. **Goal**: Output = weighted sum of input representations. Write $\mathbf{y}_i = \sum_j \alpha_{ij} f(\mathbf{x}_j)$.

2. **Weights**: Dot product similarity → softmax normalization. Write $\alpha_{ij} = \text{softmax}_j(\mathbf{x}_i^T \mathbf{x}_j)$.

3. **Matrix form**: Bundle all tokens → $\mathbf{Y} = \text{softmax}(\mathbf{X}\mathbf{X}^T)\mathbf{X}$. Verify dimensions.

4. **Expressiveness**: Separate roles via learned projections → Q, K, V. Write $\mathbf{Q} = \mathbf{X}\mathbf{W}_Q$, etc.

5. **Dimension check**: $\mathbf{Q}\mathbf{K}^T$ is the ONLY arrangement giving $n \times n$. The transpose is forced.

6. **Scale**: Variance analysis → divide by $\sqrt{d_k}$.

7. **Final formula**: $\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}$

8. **Verify**: Implement, check dimensions, validate gradients flow.

Each step is motivated by a clear problem with the previous step. **Nothing is arbitrary** — the formula is the unique solution to a sequence of design constraints.

---

## 9.11 Exercises

1. **Derive from scratch**: Starting from "output = weighted average of inputs", derive the full attention formula on paper. At each step, check dimensions.

2. **Additive attention**: Implement Bahdanau attention $e_{ij} = \mathbf{v}^T \tanh(\mathbf{W}_1 \mathbf{q}_i + \mathbf{W}_2 \mathbf{k}_j)$. Why can't this be written as a single matmul?

3. **Variance verification**: Generate random $\mathbf{Q}, \mathbf{K}$ with entries $\sim \mathcal{N}(0, 1)$. Compute $\mathbf{Q}\mathbf{K}^T$ for various $d_k$ and verify $\text{Var}(S_{ij}) \approx d_k$. Then verify that after scaling by $1/\sqrt{d_k}$, the variance is ~1.

4. **Symmetry breaking**: Show that with $\mathbf{W}_Q \neq \mathbf{W}_K$, the score matrix $\mathbf{Q}\mathbf{K}^T$ is NOT symmetric (unlike $\mathbf{X}\mathbf{X}^T$).

5. **Attention patterns**: For a 4-token sequence, manually compute the attention matrix $\mathbf{A}$ and verify that each row sums to 1 and all entries are non-negative.
