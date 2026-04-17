# 30. State Space Models & Alternatives to Attention

## Motivation

Attention is powerful but has a fundamental problem: $O(N^2)$ complexity in sequence length. For a 1M-token context window, the attention matrix has $10^{12}$ entries — impractical. This has driven a search for sub-quadratic alternatives. This lecture covers the landscape of attention alternatives, why they exist, what they trade off, and whether they can actually replace transformers.

---

## 30.1 Why Attention is Expensive (And Why We Can't Just "Fix" It)

### The quadratic wall

Standard self-attention computes:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

$QK^T$ is $(N \times d) \times (d \times N) = N \times N$. For each head, for each layer, you compute, store, and backpropagate through this $N \times N$ matrix.

| Sequence length $N$ | Attention FLOPs | Memory (bf16) |
|---|---|---|
| 1K | 1M | 2 MB |
| 8K | 64M | 128 MB |
| 32K | 1B | 2 GB |
| 128K | 16B | 32 GB |
| 1M | 1T | 2 TB |

FlashAttention (Lecture 23) reduces **memory** via tiling but doesn't reduce **FLOPs**. It makes the quadratic cheaper to compute but doesn't change the fundamental scaling.

### Why not just use sparse attention?

Sparse attention patterns (local windows, strided, etc.) reduce complexity to $O(N \sqrt{N})$ or $O(N \log N)$. But:

1. **You lose global information flow**: If token 1 can't attend to token $N$, information requires multiple hops (layers) to propagate from one end to the other.
2. **Fixed patterns miss important connections**: Which tokens are "important" depends on content, not position. A fixed sparse pattern will miss some critical dependencies.
3. **Still not truly linear**: $O(N \sqrt{N})$ is better than $O(N^2)$ but worse than $O(N)$ — and for very long sequences, even this is too much.

The goal: $O(N)$ complexity with $O(1)$ per-step cost during generation, while retaining the quality of full attention.

---

## 30.2 The RNN Revival: Why the Old Idea Returns

### RNNs: constant memory, linear time

Classical RNNs process sequences with a fixed-size hidden state:

$$h_t = f(h_{t-1}, x_t)$$

- **Inference**: $O(1)$ per step (just update the state)
- **Training**: $O(N)$ total (process each token once)
- **Memory**: $O(d)$ state size, independent of sequence length

This is exactly what we want! So why did RNNs lose to transformers?

### Why traditional RNNs failed

1. **Vanishing/exploding gradients**: For sequence length $N$, the gradient involves $\prod_{t=1}^{N} \frac{\partial h_t}{\partial h_{t-1}}$. If this product shrinks (or grows) exponentially, long-range learning fails.

2. **Sequential training**: Each $h_t$ depends on $h_{t-1}$ → can't parallelize across the sequence during training. Transformers compute all positions simultaneously.

3. **Fixed-size bottleneck**: The entire sequence history must be compressed into a fixed-size vector. A 4096-dim hidden state can't faithfully represent 100K tokens of context.

LSTMs and GRUs partially addressed gradient flow but not parallelization or the bottleneck. Transformers won because they parallelize during training and give every token direct access to every other token.

### The new question

Can we design a sequence model with:
- $O(N)$ training (like RNNs or better than $O(N^2)$ attention)?
- Parallelizable training (unlike classical RNNs)?
- $O(1)$ inference per step (like RNNs)?
- Quality competitive with transformers?

State space models (SSMs) and their descendants attempt exactly this.

---

## 30.3 Linear Attention: The Kernel Trick

### Reformulating attention

Standard attention:

$$\text{Attn}(Q, K, V)_i = \frac{\sum_j \exp(q_i^T k_j / \sqrt{d}) \cdot v_j}{\sum_j \exp(q_i^T k_j / \sqrt{d})}$$

The $\exp(q_i^T k_j / \sqrt{d})$ acts as a kernel: $\kappa(q_i, k_j) = \exp(q_i^T k_j / \sqrt{d})$.

**Linear attention** replaces this kernel with a decomposable one:

$$\kappa(q, k) = \phi(q)^T \phi(k)$$

where $\phi$ is a feature map. Then:

$$\text{LinAttn}(Q, K, V)_i = \frac{\phi(q_i)^T \sum_j \phi(k_j) v_j^T}{\phi(q_i)^T \sum_j \phi(k_j)}$$

The key: $S = \sum_j \phi(k_j) v_j^T$ is a $d' \times d$ matrix that can be computed **incrementally**:

$$S_t = S_{t-1} + \phi(k_t) v_t^T$$

This is an RNN! The "state" $S_t$ summarizes all past key-value information. Complexity: $O(N \cdot d'^2)$ — linear in $N$.

### Why linear attention underperforms

The softmax in standard attention creates a **sharp, content-dependent** weighting: only the most relevant keys get high weight. Linear attention produces **blurred** weightings — it can't focus as sharply.

The result: linear attention consistently underperforms standard attention, especially on tasks requiring precise retrieval from context (e.g., "What did the author say in paragraph 3?").

### Feature map choices

| Feature map $\phi$ | Description | Quality |
|---|---|---|
| $\phi(x) = \text{elu}(x) + 1$ | Simple, fast | Low |
| $\phi(x) = \text{RandomFourierFeatures}(x)$ | Approximates softmax kernel | Medium |
| $\phi(x) = \text{Learned}(x)$ | Data-dependent | Medium-High |
| $\phi(x) = \text{Taylor expansion of } e^x$ | Polynomial approximation | Medium |

None fully close the gap with softmax attention.

---

## 30.4 State Space Models: S4 and the Structured Approach

### The continuous-time formulation

State space models start from a continuous-time linear system:

$$\dot{h}(t) = Ah(t) + Bx(t)$$
$$y(t) = Ch(t) + Dx(t)$$

where:
- $h(t) \in \mathbb{R}^{N_h}$ is the hidden state
- $A \in \mathbb{R}^{N_h \times N_h}$ is the state transition matrix (the "memory")
- $B \in \mathbb{R}^{N_h \times 1}$ is the input projection
- $C \in \mathbb{R}^{1 \times N_h}$ is the output projection

### Discretization

To process discrete sequences, discretize with step size $\Delta$:

$$\bar{A} = \exp(\Delta A) \approx (I + \Delta A / 2)(I - \Delta A / 2)^{-1}$$
$$\bar{B} = (\Delta A)^{-1}(\bar{A} - I) \cdot \Delta B$$

The discrete recurrence:

$$h_t = \bar{A} h_{t-1} + \bar{B} x_t$$
$$y_t = C h_t$$

This looks like an RNN! But the key innovation is in how $A$ is parameterized.

### The HiPPO matrix: long-range memory by construction

The breakthrough of S4 (Gu et al., 2021): Choose $A$ to be the **HiPPO** (High-order Polynomial Projection Operator) matrix:

$$A_{nk} = -\begin{cases} (2n+1)^{1/2}(2k+1)^{1/2} & \text{if } n > k \\ n+1 & \text{if } n = k \\ 0 & \text{if } n < k \end{cases}$$

This specific matrix has the property that $h(t)$ **optimally compresses the history** $x(0), x(1), \ldots, x(t)$ into a polynomial basis. The hidden state literally stores the coefficients of a polynomial approximation of the entire input history.

**Why this matters**: Traditional RNNs with random $A$ matrices forget exponentially. HiPPO-initialized matrices remember polynomially — they can retain information from thousands or millions of steps ago.

### The convolution view (parallelizable training)

Unrolling the recurrence:

$$y_t = C\bar{A}^t \bar{B} x_0 + C\bar{A}^{t-1}\bar{B} x_1 + \cdots + C\bar{B} x_t$$

This is a **convolution**: $y = K * x$ where $K_t = C\bar{A}^t \bar{B}$ is the convolution kernel.

Computing $K$ requires the matrix powers $\bar{A}^0, \bar{A}^1, \ldots, \bar{A}^{N-1}$.

For the structured HiPPO matrix, this can be computed in $O(N \log N)$ via FFT, giving:
- **Training**: $O(N \log N)$ — almost linear, parallelizable via FFT
- **Inference**: $O(1)$ per step — just the RNN recurrence $h_t = \bar{A}h_{t-1} + \bar{B}x_t$

This is the key duality: **convolutional for training, recurrent for inference**.

---

## 30.5 Mamba: Making SSMs Content-Dependent

### The limitation of S4: fixed dynamics

In S4, the matrices $A$, $B$, $C$, $\Delta$ are **constant** — they don't depend on the input. The same transformation is applied regardless of what the input is.

This means S4 can't do **content-based selection**: it can't decide to "pay attention" to some tokens and "ignore" others based on their content. In contrast, standard attention has content-dependent weights ($QK^T$ depends on the actual token values).

### Mamba's selective state spaces (Gu & Dao, 2023)

Make $B$, $C$, and $\Delta$ **functions of the input**:

$$B_t = \text{Linear}(x_t), \quad C_t = \text{Linear}(x_t), \quad \Delta_t = \text{softplus}(\text{Linear}(x_t))$$

Now the model can:
- **Select** which information to store ($B_t$ controls what enters the state)
- **Select** which information to output ($C_t$ controls what's read from the state)
- **Select** how much to "remember vs forget" ($\Delta_t$ controls the discretization rate — large $\Delta_t$ means more forgetting)

### The input-dependent $\Delta$ as a "gate"

When $\Delta_t$ is large:
$$\bar{A}_t = \exp(\Delta_t A) \approx 0 \quad \text{(forgets previous state)}$$
$$\bar{B}_t \approx I \quad \text{(overwrites with new input)}$$

When $\Delta_t$ is small:
$$\bar{A}_t \approx I \quad \text{(preserves previous state)}$$
$$\bar{B}_t \approx 0 \quad \text{(ignores new input)}$$

This is functionally similar to an LSTM's forget gate, but emerges from the continuous-time SSM framework.

### The parallelization trick

With input-dependent parameters, you can't use the convolution trick anymore (kernel depends on position). Mamba uses a **parallel scan** algorithm:

The recurrence $h_t = \bar{A}_t h_{t-1} + \bar{B}_t x_t$ is associative:

$$(A_2, B_2 x_2) \circ (A_1, B_1 x_1) = (A_2 A_1, A_2 B_1 x_1 + B_2 x_2)$$

Because this operation is associative, you can compute the full sequence using a **parallel prefix scan** in $O(N \log N / P)$ time on $P$ processors — essentially $O(N)$ on a GPU with sufficient parallelism.

### Mamba architecture block

```
Input x
  │
  ├─→ Linear expand (D → 2E)
  │     │
  │     ├─→ Conv1d → SiLU → SSM → output branch
  │     │
  │     └─→ SiLU → gate branch
  │
  └─→ Multiply(output_branch, gate_branch)
       │
       └─→ Linear project (E → D)
```

Key design choices:
- **No attention mechanism at all** — the SSM replaces it entirely
- **Causal Conv1d** (kernel size 3-4) provides local context
- **Gated architecture** similar to GLU (see Lecture 13)
- **Expand ratio** $E/D \approx 2$ (expand the hidden dimension inside the block)

### Mamba's performance

On language modeling (perplexity):
- Mamba-3B ≈ Transformer-3B (roughly matching on standard benchmarks)
- Much faster at inference for long sequences (linear vs quadratic)
- Competitive but not clearly better on reasoning-heavy tasks

On long-range tasks (sequence length 16K+):
- Mamba handles very long sequences efficiently
- But transformers with FlashAttention and context extension also handle long sequences

---

## 30.6 Linear RNNs: RWKV, RetNet, Griffin

### RWKV: "An RNN with Transformer-level Performance"

RWKV (Peng et al., 2023) combines ideas from transformers and RNNs:

**Time-mixing** (replaces self-attention):
$$r_t = W_r \cdot (\mu_r \odot x_t + (1-\mu_r) \odot x_{t-1})$$
$$k_t = W_k \cdot (\mu_k \odot x_t + (1-\mu_k) \odot x_{t-1})$$
$$v_t = W_v \cdot (\mu_v \odot x_t + (1-\mu_v) \odot x_{t-1})$$
$$wkv_t = \frac{\sum_{i=1}^{t-1} e^{-(t-1-i)w+k_i} v_i + e^{u+k_t} v_t}{\sum_{i=1}^{t-1} e^{-(t-1-i)w+k_i} + e^{u+k_t}}$$
$$o_t = W_o \cdot (\sigma(r_t) \odot \text{wkv}_t)$$

The WKV mechanism is essentially attention with **exponential decay** — recent tokens get more weight. The decay rate $w$ is learned per-channel.

**Key properties**:
- $O(N)$ training via recurrent formulation
- $O(1)$ inference per step
- Trained at 14B+ parameter scale (comparable to mid-range transformers)

### RetNet: Retentive Network

RetNet (Sun et al., 2023) decomposes attention:

$$\text{Retention}(Q, K, V) = (QK^T \odot D) V$$

where $D_{ij} = \gamma^{i-j}$ for $i \geq j$ (causal, exponential decay).

Three computation modes:
1. **Parallel** (for training): compute $QK^T \odot D$ directly — $O(N^2)$ but parallelizable
2. **Recurrent** (for inference): maintain state $S_t = \gamma S_{t-1} + k_t v_t^T$ — $O(1)$ per step
3. **Chunkwise** (hybrid): process chunks in parallel, propagate state between chunks

### Griffin (Google DeepMind, 2024)

Hybrid architecture interleaving:
- **Local attention** (sliding window, ~1024 tokens) for precise short-range dependencies
- **Recurrent layers** (gated linear recurrence) for long-range state

This acknowledges that **pure recurrence struggles with precise retrieval** but is great for summarizing long-range context.

---

## 30.7 Why Attention Still Dominates

Despite the efficiency advantages, attention-based transformers remain the dominant architecture. Here's why:

### 1. The retrieval problem

SSMs and linear models compress the entire history into a fixed-size state. This means they can't perfectly retrieve a specific piece of information from far back in the context.

**The "needle in a haystack" test**: Hide a specific fact early in a long context, then ask about it at the end.
- Transformers: Near-perfect retrieval up to context length
- SSMs: Degraded retrieval for information more than ~4K-8K tokens back (depending on state size)

The state acts as a **lossy compression** of history. Some information is inevitably lost.

### 2. In-context learning requires precise attention

In-context learning (see Lecture 24) requires the model to implement an algorithm over the provided examples. This requires:
- Identifying which examples are relevant (content-dependent selection)
- Copying patterns from examples to the query (precise retrieval)

Both are easier with attention's explicit $O(N^2)$ comparison matrix than with a compressed state.

### 3. Training infrastructure

The entire deep learning ecosystem is optimized for transformers:
- FlashAttention, FSDP, tensor parallelism — all designed for attention
- SSM-specific optimizations (parallel scan, hardware-efficient recurrence) are less mature
- Switching architectures means giving up years of engineering optimization

### 4. Scaling evidence

The scaling laws for transformers are well-established (Lecture 19). For SSMs and alternatives, the evidence is thinner:
- Mamba has been trained up to ~3B parameters
- RWKV up to ~14B
- Neither has been tested at the 70B+ scale where transformers really shine

It's unclear whether SSMs scale as favorably. The concern: as models get larger, the fixed-size state becomes a more severe bottleneck.

---

## 30.8 Hybrid Architectures: The Best of Both Worlds?

### The emerging pattern

Rather than pure attention or pure recurrence, the best architectures may combine both:

**Jamba (AI21, 2024)**: Alternates transformer and Mamba layers
- 1:7 ratio (1 attention layer per 7 Mamba layers)
- Gets most of the efficiency benefit (long contexts are cheap)
- Retains retrieval capability (attention layers handle precise lookups)

**Griffin**: Similar hybrid with local attention + recurrence

**Zamba (Zyphra, 2024)**: Shared attention layer + Mamba backbone

### Why hybrids work

Different types of information need different processing:
- **Long-range context/summary**: Recurrent layers compress efficiently
- **Precise key-value retrieval**: Attention layers excel
- **Local patterns**: Convolutions or local attention

A hybrid model can route different information through the appropriate mechanism.

### The architectural search question

What's the optimal ratio of attention to recurrence layers? Early results suggest:
- ~10-20% attention layers is sufficient for retrieval
- The remaining 80-90% can be efficient recurrent layers
- But this may change with scale

---

## 30.9 Other Approaches Worth Knowing

### Sparse Mixture of Experts (MoE)

Not an attention alternative per se, but addresses the same problem — scaling efficiently:

- Only activate a subset of parameters for each token
- $N$ expert FFN layers, router selects top-$k$ for each token
- **Mixtral-8x7B**: 8 experts, top-2 routing → 47B total params, ~13B active per token

MoE addresses the **parameter efficiency** problem rather than the **sequence length** problem, but the two combine: a Mamba-MoE model would be efficient in both dimensions.

### xLSTM (Extended LSTM)

Hochreiter's group revisiting LSTMs with modern scaling insights:
- **sLSTM**: Scalar LSTM with exponential gating (prevents vanishing gradients more aggressively)
- **mLSTM**: Matrix LSTM where the cell state is a matrix, not a vector (more capacity)
- Competitive with transformers at ~1B scale

### Hyena

Replace attention with **long convolutions**:
$$y = (h_N * (h_{N-1} * (\cdots (h_1 * (S_1 \cdot x)) \cdots) \cdot S_N))$$

where $h_i$ are learned convolution filters and $S_i$ are element-wise gates.

- $O(N \log N)$ via FFT
- Data-dependent through the gating
- Competitive on some benchmarks but hasn't scaled to frontier models

### Based: Linear attention done right

Combine a simple linear attention (for recurrence) with a sliding window attention (for local precision). Train with the "linear attention + Taylor expansion" trick:

$$\kappa(q, k) = 1 + q^T k + \frac{(q^T k)^2}{2} + \cdots \approx \exp(q^T k)$$

Truncating at second order gives a practical linear attention that more closely approximates softmax.

---

## 30.10 The Big Picture: Where Architecture is Heading

### The convergence hypothesis

All successful sequence models are converging on the same set of ingredients:
1. **Content-dependent gating** (attention weights, SSM gates, LSTM forget gates)
2. **Residual connections** (skip around every block)
3. **Normalization** (LayerNorm, RMSNorm)
4. **Non-linear expansion** (FFN, SwiGLU)
5. **Multi-head/multi-channel processing**

The differences are primarily in **how content-dependent gating is implemented**:
- Attention: global, explicit, $O(N^2)$
- SSM: compressed, implicit, $O(N)$
- Linear attention: global, approximate, $O(N)$
- Hybrid: both, at different layers

### The prediction

Short-term (2025-2026): Hybrid architectures become standard for production. Pure transformers remain the research default due to simplicity and established scaling laws.

Medium-term (2027+): If context windows need to grow beyond 1M tokens (for video, codebases, long documents), efficient alternatives become mandatory. The architecture that wins will likely be a hybrid that keeps attention for precision and uses recurrence for compression.

Long-term: Architecture may matter less than scale. As training compute grows, the quality gap between architectures shrinks. The "optimal" architecture may change depending on the scale, task, and deployment constraints.

### What to tell your interviewer

"Attention is $O(N^2)$ but gives the best quality per parameter. SSMs like Mamba offer $O(N)$ complexity and $O(1)$ inference, but struggle with precise retrieval from long contexts. The practical sweet spot appears to be hybrid architectures that use attention sparingly for retrieval and recurrence for efficient long-range processing. The right architecture depends on the deployment constraints — if you need million-token contexts with real-time inference, hybrids are compelling; if you need maximum quality at moderate context lengths, transformers with FlashAttention are hard to beat."
