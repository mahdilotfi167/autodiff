# 14. Tokenization, Embeddings, and Positional Encoding

## Motivation

Before a GPT can process text, it must convert raw text into numbers. This seemingly mundane step has profound effects on model quality, efficiency, and capabilities. The choices here — tokenization algorithm, vocabulary size, embedding dimension, and positional encoding — are among the most consequential hyperparameters in LLM design.

This document derives each component from first principles: **why** we tokenize the way we do, how embeddings work mathematically, and the evolution from absolute positional embeddings to rotary position embeddings (RoPE).

---

## 14.1 Why Tokenization Matters: The Input Representation Problem

### The fundamental question

A neural network operates on numbers. Text is a sequence of characters. We need a mapping:

$$\text{raw text} \xrightarrow{\text{tokenizer}} \text{sequence of integers} \xrightarrow{\text{embedding}} \text{sequence of vectors}$$

### Design space for tokenization

The researcher considers three granularities:

**Option 1: Character-level**
- Vocabulary: ~256 characters (ASCII/UTF-8 bytes)
- "Hello world" → [72, 101, 108, 108, 111, 32, 119, 111, 114, 108, 100]
- Pro: tiny vocabulary, no unknown tokens
- Con: sequences become very long (a 1000-word document ≈ 5000 characters). Attention is $O(n^2)$, so 5× longer sequences cost 25× more compute.

**Option 2: Word-level**
- Vocabulary: ~100,000+ English words
- "Hello world" → [4521, 892]
- Pro: short sequences, semantically meaningful units
- Con: huge vocabulary → huge embedding matrix. Can't handle misspellings, rare words, new words, or non-English text. Out-of-vocabulary (OOV) problem.

**Option 3: Subword-level** (the winner)
- "unhappiness" → ["un", "happiness"] or ["un", "happ", "iness"]
- Vocabulary: 30,000-100,000 tokens
- Pro: compact sequences, handles rare words via decomposition, no OOV
- Con: token boundaries are sometimes unintuitive

> **Design decision**: Subword tokenization — the best trade-off between sequence length and vocabulary coverage.

---

## 14.2 Byte Pair Encoding (BPE): The Algorithm

### The core idea

BPE starts with individual characters (or bytes) and iteratively **merges** the most frequent adjacent pair into a new token. This is a compression algorithm repurposed for tokenization.

### Algorithm

```
Input: training corpus, desired vocabulary size V

1. Initialize vocabulary = all individual bytes (256 tokens)
2. While |vocabulary| < V:
   a. Count all adjacent token pairs in the corpus
   b. Find the most frequent pair (a, b)
   c. Merge: replace all occurrences of (a, b) with new token "ab"
   d. Add "ab" to vocabulary
3. Return vocabulary + merge rules
```

### Worked example

Corpus: "low low low low low lower lower newest newest newest widest"

**Initial tokens** (character level): l, o, w, e, r, n, s, t, i, d, (space), ...

**Iteration 1**: Most frequent pair is `(l, o)` → merge to `lo`

**Iteration 2**: Most frequent pair is `(lo, w)` → merge to `low`

**Iteration 3**: Most frequent pair is `(e, s)` → merge to `es`

**Iteration 4**: Most frequent pair is `(es, t)` → merge to `est`

**Iteration 5**: Most frequent pair is `(n, e)` → merge to `ne`

And so on until we reach the target vocabulary size.

### Key properties of BPE

1. **Frequent words become single tokens**: "the", "and", "is" are likely single tokens
2. **Rare words are decomposed**: "pneumonoultramicroscopicsilicovolcanoconiosis" → multiple subword tokens
3. **Deterministic**: given the same corpus and vocab size, the same merges happen
4. **Byte-level BPE** (GPT-2/3): Start from 256 byte values instead of characters. This handles any Unicode text without special preprocessing.

### GPT's tokenizer specifics

| Model | Tokenizer | Vocab size | Base |
|---|---|---|---|
| GPT-2 / GPT-3 | Byte-level BPE | 50,257 | 256 byte tokens + 50,000 merges + 1 special |
| LLaMA-1 | SentencePiece BPE | 32,000 | Byte-fallback |
| LLaMA-3 | tiktoken BPE | 128,256 | Byte-level |
| GPT-4 | tiktoken (cl100k) | 100,277 | Byte-level |

---

## 14.3 The Effect of Vocabulary Size

### The trade-off

$$\text{Vocabulary size } V \text{ controls a fundamental trade-off:}$$

**Larger $V$** (e.g., 128K):
- Shorter sequences (more text per token) → faster processing, longer effective context
- Larger embedding matrix $\mathbf{W}_E \in \mathbb{R}^{V \times d}$ → more parameters, more memory
- Each token seen less frequently during training → sparser learning signal
- Better "fertility" (tokens per word): English ≈ 1.0-1.3 tokens/word

**Smaller $V$** (e.g., 32K):
- Longer sequences (less text per token) → slower processing, shorter effective context
- Smaller embedding matrix → fewer parameters
- Each token seen more frequently → denser learning signal
- Worse fertility: English ≈ 1.3-1.5 tokens/word

### The math: how $V$ affects compute

For a sequence of $W$ words with average fertility $f$ (tokens per word):

- Sequence length: $n = W \cdot f$
- Attention cost per layer: $O(n^2 d) = O(W^2 f^2 d)$
- FFN cost per layer: $O(n d^2) = O(W f d^2)$
- Embedding cost: $O(V d)$

Doubling the vocabulary roughly halves the fertility $f$, which **quarters** the attention cost but **doubles** the embedding cost. For long sequences, the attention saving dominates.

### The practical sweet spot

- $V \approx 32,000$: works well for monolingual English models (LLaMA-1)
- $V \approx 50,000-100,000$: good for English-heavy multilingual (GPT-2/3/4)
- $V \approx 128,000+$: better for truly multilingual models (LLaMA-3, Gemma)

The trend is toward larger vocabularies as compute becomes more available and multilingual performance matters more.

---

## 14.4 Special Tokens

Every tokenizer defines **special tokens** with reserved IDs:

| Token | Purpose | Used in |
|---|---|---|
| `<bos>` / `<s>` | Beginning of sequence | Many models |
| `<eos>` / `</s>` | End of sequence / end of generation | All models |
| `<pad>` | Padding for batching | Training |
| `<unk>` | Unknown token (rare in BPE) | Fallback |
| `<|im_start|>`, `<|im_end|>` | Chat formatting markers | ChatGPT/GPT-4 |
| `<|endoftext|>` | Document separator / EOS | GPT-2/3 |

Special tokens are critical for:
1. Telling the model where documents begin and end
2. Formatting multi-turn conversations
3. Enabling controlled generation (stop conditions)

---

## 14.5 Token Embeddings: From Integers to Vectors

### The embedding operation

Given token ID $t \in \{0, 1, \ldots, V-1\}$, the embedding is a lookup:

$$\mathbf{e}_t = \mathbf{W}_E[t, :] \in \mathbb{R}^d$$

where $\mathbf{W}_E \in \mathbb{R}^{V \times d}$ is the embedding matrix.

### This is equivalent to a one-hot multiplication

$$\mathbf{e}_t = \mathbf{o}_t^T \mathbf{W}_E$$

where $\mathbf{o}_t \in \mathbb{R}^V$ is a one-hot vector with 1 at position $t$ and 0 elsewhere.

But we **never** actually construct the one-hot vector — that would be wasteful ($V$ can be 100K+). Instead, it's a direct table lookup: index into row $t$ of $\mathbf{W}_E$.

### For a sequence of $n$ tokens

$$\mathbf{X}_{\text{emb}} = \mathbf{W}_E[\text{token\_ids}] \in \mathbb{R}^{B \times n \times d}$$

Each position gets its token's embedding vector.

### Gradient through embedding

The embedding is differentiable. During backprop, the gradient $\bar{\mathbf{W}}_E$ is sparse: only the rows corresponding to tokens in the current batch receive nonzero gradients.

$$\bar{\mathbf{W}}_E[t, :] = \sum_{\text{positions } i \text{ where token } = t} \bar{\mathbf{X}}_{\text{emb}}[:, i, :]$$

This sparsity is why techniques like **sparse Adam** or gradient accumulation for embeddings matter.

---

## 14.6 Positional Encoding: The Position Problem

### Why position information is needed

Attention is **permutation-equivariant**: if you shuffle the input tokens, the attention mechanism shuffles the output in the same way. The function $f(\mathbf{X}) = \text{softmax}(\mathbf{Q}\mathbf{K}^T/\sqrt{d_k})\mathbf{V}$ treats the input as a **set**, not a **sequence**.

But language is inherently sequential! "The dog bit the man" ≠ "The man bit the dog."

**The model needs position information injected explicitly.**

### The design space (what the researcher writes on paper)

> "I need a function $\text{PE}: \{1, \ldots, n\} \rightarrow \mathbb{R}^d$ that maps position index to a vector. Requirements:
> 1. Each position gets a unique encoding
> 2. The encoding should help the model understand **relative** distances
> 3. Should generalize to unseen sequence lengths
> 4. Should not add too many parameters"

---

## 14.7 Absolute Positional Embeddings (GPT-1/2/3)

### Learned absolute embeddings

The simplest approach: learn a separate embedding for each position.

$$\mathbf{P} \in \mathbb{R}^{n_{\max} \times d}$$

where $n_{\max}$ is the maximum context length (e.g., 1024 for GPT-2, 2048 for GPT-3).

The input to the first transformer block is:

$$\mathbf{X} = \mathbf{W}_E[\text{token\_ids}] + \mathbf{P}[:n, :]$$

### Properties

- **Parameters**: $n_{\max} \times d$ (for GPT-2: $1024 \times 768 = 786,432$)
- **Pro**: simple, flexible — the model learns whatever positional pattern is useful
- **Con**: **cannot extrapolate** beyond $n_{\max}$. If trained with max length 1024, the model has no embedding for position 1025.
- **Con**: the model learns **absolute** positions, but language understanding often depends on **relative** positions ("the word 3 tokens ago" matters more than "the word at position 847")

### Why this is limiting

Absolute positional embeddings bake in: "position 5 always gets vector $\mathbf{p}_5$." But for many linguistic patterns, what matters is the **distance** between tokens, not their absolute positions." "The cat sat" should work the same whether it starts at position 0 or position 500.

---

## 14.8 Sinusoidal Positional Encoding (Original Transformer)

### The idea

The original Transformer used a fixed (non-learned) positional encoding based on sinusoidal functions:

$$\text{PE}(pos, 2i) = \sin\left(\frac{pos}{10000^{2i/d}}\right)$$
$$\text{PE}(pos, 2i+1) = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$

where $pos$ is the position and $i$ is the dimension index.

### Why sinusoids?

The researcher's reasoning:

1. **Unique encoding per position**: the combination of frequencies creates a unique "fingerprint" for each position
2. **Relative position via linear transformation**: for any fixed offset $k$, $\text{PE}(pos + k)$ can be expressed as a linear transformation of $\text{PE}(pos)$:

$$\begin{pmatrix} \sin(\omega(pos+k)) \\ \cos(\omega(pos+k)) \end{pmatrix} = \begin{pmatrix} \cos(\omega k) & \sin(\omega k) \\ -\sin(\omega k) & \cos(\omega k) \end{pmatrix} \begin{pmatrix} \sin(\omega pos) \\ \cos(\omega pos) \end{pmatrix}$$

This means the model can learn to attend to relative positions via a learned linear operation. But this capability must be **learned** — it's not structurally enforced.

3. **Extrapolation**: since the functions are defined for any $pos$, the encoding extrapolates to any sequence length (in theory; in practice, performance degrades).

### Limitations

Sinusoidal encodings were superseded because:
- The model must **learn** to extract relative positions from the additive encoding — this uses model capacity
- The dot product $\mathbf{q}_i^T \mathbf{k}_j$ mixes content and position in complex ways that are hard to disentangle

---

## 14.9 Rotary Position Embedding (RoPE): The Modern Standard

### The key insight

RoPE (Su et al., 2021) encodes position **directly into the attention computation** rather than adding it to the input. The idea: **rotate** query and key vectors by an angle proportional to their position. Then the dot product between query at position $m$ and key at position $n$ naturally depends on $(m - n)$, the relative distance.

### Derivation

The researcher asks: "Can I modify the attention score so that it **automatically** depends on relative position?"

We want:
$$\langle f(\mathbf{q}, m), f(\mathbf{k}, n) \rangle = g(\mathbf{q}, \mathbf{k}, m - n)$$

where $f$ is some position-dependent transformation and $g$ depends only on the **relative** position $m - n$.

**Solution in 2D**: Consider a 2D subspace of the query/key vectors. Apply a rotation matrix:

$$f(\mathbf{q}, m) = R_m \mathbf{q} = \begin{pmatrix} \cos(m\theta) & -\sin(m\theta) \\ \sin(m\theta) & \cos(m\theta) \end{pmatrix} \begin{pmatrix} q_0 \\ q_1 \end{pmatrix}$$

$$f(\mathbf{k}, n) = R_n \mathbf{k} = \begin{pmatrix} \cos(n\theta) & -\sin(n\theta) \\ \sin(n\theta) & \cos(n\theta) \end{pmatrix} \begin{pmatrix} k_0 \\ k_1 \end{pmatrix}$$

Now compute the dot product:

$$\langle R_m \mathbf{q}, R_n \mathbf{k} \rangle = \mathbf{q}^T R_m^T R_n \mathbf{k} = \mathbf{q}^T R_{n-m} \mathbf{k}$$

Because rotation matrices have the property $R_m^T R_n = R_{n-m}$. The dot product depends only on relative position $(n - m)$! ✓

### Extension to $d$ dimensions

For $d$-dimensional vectors, pair up dimensions $(0,1), (2,3), \ldots, (d-2, d-1)$ and apply independent rotations:

$$R_m = \begin{pmatrix} \cos(m\theta_0) & -\sin(m\theta_0) & & & \\ \sin(m\theta_0) & \cos(m\theta_0) & & & \\ & & \cos(m\theta_1) & -\sin(m\theta_1) & \\ & & \sin(m\theta_1) & \cos(m\theta_1) & \\ & & & & \ddots \end{pmatrix}$$

where $\theta_i = 10000^{-2i/d}$ (same frequency schedule as sinusoidal PE).

### Implementation (efficient form)

Instead of constructing the full rotation matrix, RoPE is applied element-wise:

```python
def apply_rope(x, freqs_cos, freqs_sin):
    """Apply RoPE to queries or keys.
    x: (B, n, h, d_k)   — h heads, d_k dims per head
    freqs_cos, freqs_sin: (n, d_k//2) — precomputed
    """
    # Split into pairs
    x_even = x[..., 0::2]   # (B, n, h, d_k//2)
    x_odd  = x[..., 1::2]   # (B, n, h, d_k//2)

    # Apply rotation
    x_rotated_even = x_even * freqs_cos - x_odd * freqs_sin
    x_rotated_odd  = x_even * freqs_sin + x_odd * freqs_cos

    # Interleave back
    x_out = torch.stack([x_rotated_even, x_rotated_odd], dim=-1)
    return x_out.flatten(-2)  # (B, n, h, d_k)

def precompute_freqs(d_k, max_len, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, d_k, 2).float() / d_k))
    positions = torch.arange(max_len)
    angles = positions[:, None] * freqs[None, :]  # (max_len, d_k//2)
    return torch.cos(angles), torch.sin(angles)
```

### Where RoPE is applied

RoPE is applied to $\mathbf{Q}$ and $\mathbf{K}$ **after** the linear projections but **before** the dot product:

$$\text{attn}(\mathbf{X}) = \text{softmax}\left(\frac{R_{\text{pos}} \mathbf{Q} \cdot (R_{\text{pos}} \mathbf{K})^T}{\sqrt{d_k}}\right) \mathbf{V}$$

**Note**: RoPE is NOT applied to $\mathbf{V}$. Values carry content, not positional information. The position information is only in the attention pattern (which tokens attend to which), not in the values being aggregated.

### Context length extension with RoPE

A major advantage of RoPE: it can be extended beyond the training context length. Several techniques exist:

1. **NTK-aware scaling** (Code Llama): modify the frequency base $\theta$ to interpolate
2. **YaRN**: combine NTK scaling with attention scaling
3. **Linear interpolation**: simply divide position indices by a scaling factor

These work because RoPE's rotational structure degrades gracefully, unlike learned absolute embeddings which simply have no embedding for unseen positions.

---

## 14.10 ALiBi: Attention with Linear Biases

### The idea

ALiBi (Press et al., 2022) takes a different approach: don't encode position in the embeddings at all. Instead, add a **linear bias** directly to the attention scores:

$$\text{score}(i, j) = \mathbf{q}_i^T \mathbf{k}_j - m \cdot |i - j|$$

where $m$ is a head-specific slope (a fixed hyperparameter, not learned).

### The slopes

For $h$ heads, the slopes are set to a geometric sequence:

$$m_1 = 2^{-8/h}, \quad m_2 = 2^{-16/h}, \quad \ldots, \quad m_h = 2^{-8}$$

For $h=8$: slopes = $\frac{1}{2}, \frac{1}{4}, \frac{1}{8}, \frac{1}{16}, \frac{1}{32}, \frac{1}{64}, \frac{1}{128}, \frac{1}{256}$

### Effect

- Nearby tokens have scores that are barely penalized
- Distant tokens have scores that are heavily penalized
- Different heads penalize distance at different rates — some heads focus locally, others globally
- **No position embeddings needed at all** — saves parameters
- Excellent length extrapolation (can generalize well beyond training length)

### ALiBi vs. RoPE

| Property | RoPE | ALiBi |
|---|---|---|
| Applied to | Q, K vectors (multiplicative) | Attention scores (additive) |
| Parameters | None (fixed) | None (fixed slopes) |
| Relative position | Yes (via rotation) | Yes (via distance penalty) |
| Length extrapolation | Good (with scaling) | Excellent (native) |
| Adoption | LLaMA, Mistral, Gemma, most models | BLOOM, MPT |
| Dominance | **More popular** | Less popular |

RoPE has become the dominant choice in practice, partly because it can be extended with various scaling techniques and has been validated at very large scale (LLaMA series).

---

## 14.11 The Embedding Scale Factor

### Why scale embeddings?

In some implementations, the token embeddings are multiplied by $\sqrt{d}$:

$$\mathbf{X} = \sqrt{d} \cdot \mathbf{W}_E[\text{tokens}]$$

**Reason**: Embedding weights are typically initialized from $\mathcal{N}(0, 1)$ or $\mathcal{N}(0, 0.02)$. With $d$ dimensions, the L2 norm of an embedding vector is $\approx \sqrt{d} \cdot \sigma_{\text{init}}$. The scale factor ensures that the embedding magnitudes are compatible with the positional encoding (sinusoidal) magnitudes, which have norm $\approx \sqrt{d/2}$.

**Modern models** (GPT-2, LLaMA) typically do NOT use this scaling — they rely on better initialization schemes and layer normalization to handle scale.

---

## 14.12 Summary: The Complete Input Pipeline

```
Raw text: "The quick brown fox"
    │
    ▼
[BPE Tokenizer]
    │  "The" → 464, " quick" → 2068, " brown" → 7586, " fox" → 21831
    ▼
Token IDs: [464, 2068, 7586, 21831]  shape: (1, 4)
    │
    ▼
[Token Embedding]  W_E[token_ids]  shape: (1, 4, d)
    │
    ▼ (if using absolute PE)
[+ Positional Embedding]  + P[:4, :]  shape: (1, 4, d)
    │
    ▼ (if using RoPE — applied later inside attention)
[RoPE applied to Q, K in each attention layer]
    │
    ▼
Input to first transformer block: (1, 4, d)
```

### Design decisions summary

| Choice | Options | Modern default | Why |
|---|---|---|---|
| Tokenization | Char / Word / Subword | Byte-level BPE | Balance of efficiency and coverage |
| Vocab size | 32K / 50K / 100K / 128K | 128K (trending up) | Better multilingual fertility |
| Position encoding | Sinusoidal / Learned / RoPE / ALiBi | RoPE | Relative positions + extensibility |
| Embedding scaling | $\sqrt{d}$ / none | None | RMSNorm handles scale |
| Weight tying | Tied / Untied | Untied (at scale) | Marginal quality gain |
