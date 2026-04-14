# 11. The Complete Transformer Block: LayerNorm, Residuals, FFN, and Positional Encoding

## Motivation

Documents 9-10 derived multi-head attention — the core mechanism. But a transformer is more than attention. The **transformer block** wraps attention with several critical components: layer normalization, residual connections, a feed-forward network, and positional encoding. Each was designed to solve a specific problem, and each has a clear mathematical derivation.

This document derives each component from first principles, shows how they compose into the full transformer block, and provides the complete computation graph with every dimension traced.

---

## 11.1 The Residual Connection: Why $\mathbf{X} + f(\mathbf{X})$ Instead of $f(\mathbf{X})$

### The problem

Deep networks suffer from **degradation**: as depth increases, training accuracy **decreases** (not just test accuracy). This is surprising — a deeper network should be at least as good, since it could learn the identity for extra layers.

### The insight (what the researcher writes on paper)

> "If the optimal function is close to identity, learning $f(\mathbf{x}) = \mathbf{x}$ is hard (requires precise weight tuning). But learning $f(\mathbf{x}) = 0$ is easy (just set weights near zero). So let the network learn a **residual**: $\mathbf{y} = \mathbf{x} + g(\mathbf{x})$."

If $g$ should be zero, the weights just need to be small. The network learns **deviations** from identity, not the full mapping.

### Mathematical formulation

For a sub-layer $\text{SubLayer}(\mathbf{X})$ (e.g., multi-head attention):

$$\mathbf{Y} = \mathbf{X} + \text{SubLayer}(\mathbf{X})$$

### Shape requirement

$$\underbrace{\mathbf{X}}_{B \times n \times d} + \underbrace{\text{SubLayer}(\mathbf{X})}_{B \times n \times d} = \underbrace{\mathbf{Y}}_{B \times n \times d}$$

The SubLayer MUST output the same shape as its input. This is why:
- Multi-head attention has the output projection $\mathbf{W}_O$ mapping back to dimension $d$
- The FFN's output layer maps back to $d$

### Gradient flow through residuals

The gradient through a residual connection:

$$\frac{\partial \mathbf{Y}}{\partial \mathbf{X}} = \mathbf{I} + \frac{\partial \text{SubLayer}(\mathbf{X})}{\partial \mathbf{X}}$$

The identity term $\mathbf{I}$ means that **gradients always have a direct path** through the residual. Even if $\frac{\partial \text{SubLayer}}{\partial \mathbf{X}}$ is small (vanishing gradients), the gradient through the identity path is exactly 1. This is the fundamental reason residual connections work.

---

## 11.2 Layer Normalization: The Derivation

### The problem

During training, the distribution of inputs to each layer changes as the preceding layers update (internal covariate shift). Also, different tokens in a sequence may have very different magnitudes, making optimization difficult.

### What normalization should look like

The researcher writes: "I want to normalize each token's feature vector to have zero mean and unit variance."

For a single token $\mathbf{x} \in \mathbb{R}^d$:

$$\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}$$

where:
$$\mu = \frac{1}{d}\sum_{i=1}^d x_i, \qquad \sigma^2 = \frac{1}{d}\sum_{i=1}^d (x_i - \mu)^2$$

$\epsilon > 0$ is a small constant for numerical stability (typically $10^{-5}$).

### Why layer norm and not batch norm?

**Batch norm** normalizes across the batch dimension: for each feature, compute mean and variance over all batch elements.

**Layer norm** normalizes across the feature dimension: for each token, compute mean and variance over all features.

For sequence models:
- Batch norm statistics depend on batch size and vary at test time with different sequence lengths — problematic
- Layer norm statistics are per-token and independent of batch/sequence — stable

### The 3D matrix form

For input $\mathbf{X} \in \mathbb{R}^{B \times n \times d}$:

$$\mu_{bi} = \frac{1}{d}\sum_{j=1}^d X_{bij}$$

$$\sigma^2_{bi} = \frac{1}{d}\sum_{j=1}^d (X_{bij} - \mu_{bi})^2$$

$$\hat{X}_{bij} = \frac{X_{bij} - \mu_{bi}}{\sqrt{\sigma^2_{bi} + \epsilon}}$$

**Note**: $\mu$ and $\sigma^2$ have shape $(B, n)$ — one scalar per token per batch element. The normalization is applied independently to each token's $d$-dimensional feature vector.

### Learnable affine transform

After normalization, apply a learnable scale and shift:

$$Y_{bij} = \gamma_j \hat{X}_{bij} + \beta_j$$

where $\boldsymbol{\gamma}, \boldsymbol{\beta} \in \mathbb{R}^d$ are learnable parameters (shared across all tokens and all batch elements).

**Why?** The normalization removes mean/variance information, which might be useful. The affine transform lets the network recover it if needed. At initialization, $\gamma_j = 1, \beta_j = 0$ (identity).

### Complete computation graph for LayerNorm

```
    X (B, n, d)
        |
   [mean over d]  →  μ (B, n, 1)
        |
   [X - μ]  →  X_centered (B, n, d)
        |
   [X_centered² → mean over d]  →  σ² (B, n, 1)
        |
   [1/√(σ² + ε)]  →  inv_std (B, n, 1)
        |
   [X_centered × inv_std]  →  X_hat (B, n, d)
        |
   [X_hat × γ + β]  →  Y (B, n, d)
```

### Implementation

```python
def layer_norm(X, gamma, beta, eps=1e-5):
    """
    X:     (B, n, d)
    gamma: (d,) — learnable scale
    beta:  (d,) — learnable shift
    """
    # Compute statistics along last dimension
    mu = X.mean(dim=-1, keepdim=True)       # (B, n, 1)
    var = X.var(dim=-1, keepdim=True, unbiased=False)  # (B, n, 1)
    
    # Normalize
    X_hat = (X - mu) / torch.sqrt(var + eps)  # (B, n, d)
    
    # Affine transform (broadcasting: gamma and beta are (d,))
    Y = gamma * X_hat + beta                   # (B, n, d)
    
    return Y

# Test
B, n, d = 2, 4, 8
X = torch.randn(B, n, d)
gamma = torch.ones(d)
beta = torch.zeros(d)

Y = layer_norm(X, gamma, beta)
print(f"Input shape:  {X.shape}")   # (2, 4, 8)
print(f"Output shape: {Y.shape}")   # (2, 4, 8)

# Verify normalization: each token should have ~zero mean, ~unit variance
print(f"Output mean per token: {Y.mean(dim=-1)}")       # ≈ 0
print(f"Output var per token:  {Y.var(dim=-1)}")         # ≈ 1
```

---

## 11.3 Pre-Norm vs Post-Norm: Where to Put LayerNorm

### Post-Norm (original transformer, Vaswani 2017)

$$\mathbf{Y} = \text{LayerNorm}(\mathbf{X} + \text{SubLayer}(\mathbf{X}))$$

The normalization is applied AFTER the residual addition.

### Pre-Norm (modern default, used in GPT-2+)

$$\mathbf{Y} = \mathbf{X} + \text{SubLayer}(\text{LayerNorm}(\mathbf{X}))$$

The normalization is applied BEFORE the sub-layer, and the residual bypasses it.

### Why pre-norm is preferred

Gradient flow analysis:

**Post-norm**: The gradient must pass through LayerNorm at every layer. LayerNorm's gradient involves the inverse standard deviation, which can amplify or suppress gradients.

**Pre-norm**: The residual connection provides a **clean gradient highway**:

$$\frac{\partial \mathbf{Y}}{\partial \mathbf{X}} = \mathbf{I} + \frac{\partial}{\partial \mathbf{X}}\text{SubLayer}(\text{LayerNorm}(\mathbf{X}))$$

The identity term is unobstructed — no normalization in the direct path. This leads to more stable training, especially for deep models.

---

## 11.4 The Feed-Forward Network (FFN)

### Design

After attention aggregates information across tokens, the FFN processes each token **independently** (no cross-token interaction):

$$\text{FFN}(\mathbf{x}) = \mathbf{W}_2 \cdot \text{activation}(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2$$

where:
- $\mathbf{W}_1 \in \mathbb{R}^{d_{ff} \times d}$ — expand to a larger internal dimension
- $\mathbf{W}_2 \in \mathbb{R}^{d \times d_{ff}}$ — project back to model dimension
- Typically $d_{ff} = 4d$ (4× expansion)

### Why the expansion?

The 2-layer MLP with expansion acts as a **memory lookup**: the first layer computes $d_{ff}$ "features" (each detecting a pattern), and the second layer combines them. With $d_{ff} = 4d$, the network has massive capacity per token.

### 3D matrix form

For $\mathbf{X} \in \mathbb{R}^{B \times n \times d}$:

$$\mathbf{H} = \text{activation}(\mathbf{X}\mathbf{W}_1^T + \mathbf{b}_1) \in \mathbb{R}^{B \times n \times d_{ff}}$$

$$\text{FFN}(\mathbf{X}) = \mathbf{H}\mathbf{W}_2^T + \mathbf{b}_2 \in \mathbb{R}^{B \times n \times d}$$

**Note**: The same $\mathbf{W}_1, \mathbf{W}_2$ are applied to every token independently. This is equivalent to a 1D convolution with kernel size 1, or a shared MLP applied point-wise.

### Activation function choices

| Activation | Formula | Used in |
|-----------|---------|---------|
| ReLU | $\max(0, x)$ | Original transformer |
| GELU | $x \cdot \Phi(x)$ where $\Phi$ is Gaussian CDF | GPT, BERT |
| SiLU/Swish | $x \cdot \sigma(x)$ | LLaMA, PaLM |
| GLU variants | $(\mathbf{X}\mathbf{W}_1) \odot \sigma(\mathbf{X}\mathbf{V})$ | LLaMA (SwiGLU) |

### Implementation

```python
def feed_forward(X, W1, b1, W2, b2):
    """
    X:  (B, n, d)
    W1: (d_ff, d)
    b1: (d_ff,)
    W2: (d, d_ff)
    b2: (d,)
    """
    # Expand: (B, n, d) @ (d, d_ff) → (B, n, d_ff)
    H = F.gelu(X @ W1.T + b1)
    
    # Contract: (B, n, d_ff) @ (d_ff, d) → (B, n, d)
    Y = H @ W2.T + b2
    
    return Y
```

---

## 11.5 Positional Encoding: The Missing Piece

### The problem

Attention is **permutation-equivariant**: if you shuffle the input tokens, the output shuffles the same way. The attention weights $\alpha_{ij}$ depend only on the content of tokens $i$ and $j$, not their positions.

**Proof on paper**: If $\mathbf{P}$ is a permutation matrix:

$$\text{Attention}(\mathbf{PX}) = \text{softmax}\left(\frac{(\mathbf{PX}\mathbf{W}_Q)(\mathbf{PX}\mathbf{W}_K)^T}{\sqrt{d_k}}\right)(\mathbf{PX}\mathbf{W}_V)$$

$$= \text{softmax}\left(\frac{\mathbf{P}\mathbf{Q}(\mathbf{P}\mathbf{K})^T}{\sqrt{d_k}}\right)\mathbf{P}\mathbf{V}$$

$$= \text{softmax}\left(\frac{\mathbf{P}\mathbf{Q}\mathbf{K}^T\mathbf{P}^T}{\sqrt{d_k}}\right)\mathbf{P}\mathbf{V}$$

$$= \mathbf{P} \cdot \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V} = \mathbf{P} \cdot \text{Attention}(\mathbf{X})$$

(The softmax commutes because $\mathbf{P}(\cdot)\mathbf{P}^T$ just permutes rows and columns together, and row-wise softmax is applied to permuted rows.)

So attention treats "the cat sat" and "sat the cat" identically — clearly wrong for language.

### The fix: add position information to the input

$$\mathbf{X}_\text{input} = \mathbf{X}_\text{token} + \mathbf{E}_\text{pos}$$

where $\mathbf{E}_\text{pos} \in \mathbb{R}^{n \times d}$ encodes position.

### Sinusoidal positional encoding (Vaswani et al.)

The authors designed a fixed encoding using sine and cosine functions at different frequencies:

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right)$$

$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$

where $pos$ is the token position and $i$ is the dimension index.

### Why this specific formula? (The design reasoning)

The researcher wants positional encodings with these properties:

1. **Unique per position**: Each position gets a distinct vector.
2. **Bounded**: Values stay in $[-1, 1]$ regardless of sequence length.
3. **Relative positions are accessible**: For any fixed offset $k$, $PE_{pos+k}$ can be written as a linear function of $PE_{pos}$. This means attention can learn to attend to relative positions.

**Property 3 proof sketch**: Using the trig identity $\sin(a+b) = \sin a \cos b + \cos a \sin b$:

$$PE_{(pos+k, 2i)} = \sin\left(\frac{pos+k}{\omega_i}\right) = \sin\left(\frac{pos}{\omega_i}\right)\cos\left(\frac{k}{\omega_i}\right) + \cos\left(\frac{pos}{\omega_i}\right)\sin\left(\frac{k}{\omega_i}\right)$$

This is a **linear combination** of $PE_{(pos, 2i)}$ and $PE_{(pos, 2i+1)}$ with coefficients that depend only on $k$ (not $pos$). So a linear layer can extract relative position information.

### Learned positional encoding (modern approach)

Most modern transformers (GPT, BERT) simply learn $\mathbf{E}_\text{pos} \in \mathbb{R}^{n_{\max} \times d}$ as a trainable parameter:

```python
pos_embedding = nn.Embedding(max_seq_len, d)
positions = torch.arange(n)          # (n,)
E_pos = pos_embedding(positions)     # (n, d)
X = X_token + E_pos                  # broadcasting: (B, n, d) + (n, d)
```

### Rotary Positional Encoding (RoPE) — the modern standard

RoPE (Su et al., 2021) encodes position by **rotating** the query and key vectors. Instead of adding to the input, it modifies the attention computation:

$$\text{score}(q_i, k_j) = \text{Re}[(R_i q) \cdot \overline{(R_j k)}] = q^T R_{i-j} k$$

where $R_i$ is a rotation matrix depending on position $i$. The score depends only on the **relative position** $i - j$.

This is used in LLaMA, Mistral, and most modern LLMs.

---

## 11.6 The Complete Transformer Block

### Pre-Norm architecture (modern standard)

```python
def transformer_block(X, attn_params, ffn_params, ln1_params, ln2_params):
    """
    X: (B, n, d)
    
    Pre-norm transformer block:
      Y = X + MHA(LayerNorm(X))
      Z = Y + FFN(LayerNorm(Y))
    """
    # Block 1: Multi-head self-attention with residual
    X_norm1 = layer_norm(X, **ln1_params)           # (B, n, d)
    attn_out = multi_head_attention(X_norm1, **attn_params)  # (B, n, d)
    Y = X + attn_out                                 # residual connection
    
    # Block 2: Feed-forward with residual
    Y_norm2 = layer_norm(Y, **ln2_params)            # (B, n, d)
    ffn_out = feed_forward(Y_norm2, **ffn_params)    # (B, n, d)
    Z = Y + ffn_out                                  # residual connection
    
    return Z
```

### Post-Norm architecture (original paper)

```python
def transformer_block_postnorm(X, attn_params, ffn_params, ln1_params, ln2_params):
    """Post-norm: normalize AFTER residual addition."""
    attn_out = multi_head_attention(X, **attn_params)
    Y = layer_norm(X + attn_out, **ln1_params)       # norm after residual
    
    ffn_out = feed_forward(Y, **ffn_params)
    Z = layer_norm(Y + ffn_out, **ln2_params)         # norm after residual
    
    return Z
```

---

## 11.7 Complete Computation Graph of a Transformer Block

```
   X (B, n, d)
   │
   ├────────────────────── (residual path 1) ──────────────────┐
   ↓                                                           │
  [LayerNorm₁]                                                 │
   ↓                                                           │
  X_norm (B, n, d)                                             │
   ├──────────────┬──────────────┐                             │
   ↓              ↓              ↓                             │
 [× W_Q]       [× W_K]       [× W_V]                          │
   ↓              ↓              ↓                             │
   Q_big         K_big         V_big    (B, n, d)              │
   ↓              ↓              ↓                             │
 [view+transpose per Doc 10]                                   │
   ↓              ↓              ↓                             │
   Q              K              V      (B, h, n, d_k)         │
   │              │              │                             │
   └── [attn] ────┘──────────────┘                             │
         ↓                                                     │
    Y_heads (B, h, n, d_k)                                     │
         ↓                                                     │
   [concat heads → (B, n, d)]                                  │
         ↓                                                     │
      [× W_O]                                                  │
         ↓                                                     │
    attn_out (B, n, d)                                         │
         │                                                     │
         └──────────────────── [add] ──────────────────────────┘
                                 ↓
                              Y (B, n, d)
                                 │
   ┌──────────────── (residual path 2) ────────────────────────┤
   │                                                           │
   │                           [LayerNorm₂]                    │
   │                              ↓                            │
   │                         Y_norm (B, n, d)                  │
   │                              ↓                            │
   │                        [× W₁ᵀ + b₁]                      │
   │                              ↓                            │
   │                         H (B, n, d_ff)                    │
   │                              ↓                            │
   │                           [GELU]                          │
   │                              ↓                            │
   │                        [× W₂ᵀ + b₂]                      │
   │                              ↓                            │
   │                       ffn_out (B, n, d)                   │
   │                              │                            │
   └────────────────────── [add] ─┘                            
                              ↓
                           Z (B, n, d)
```

---

## 11.8 Stacking Transformer Blocks

### The full transformer

A transformer with $L$ layers just stacks $L$ identical blocks (with separate parameters):

```python
def transformer(X, blocks_params, embed_params):
    """
    X_tokens: (B, n) — integer token IDs
    """
    # Token + position embedding
    X = token_embed(X_tokens) + pos_embed(positions)   # (B, n, d)
    
    # Pass through L transformer blocks
    for l in range(L):
        X = transformer_block(X, **blocks_params[l])
    
    # Final layer norm (pre-norm architecture requires this)
    X = layer_norm(X, **final_ln_params)
    
    return X
```

### Parameter count (for reference)

For each transformer block with dimensions $d$, $h$ heads, $d_{ff} = 4d$:

| Component | Parameters | Count |
|-----------|-----------|-------|
| $\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V$ | $3 \times d^2$ | $3d^2$ |
| $\mathbf{W}_O$ | $d^2$ | $d^2$ |
| $\mathbf{W}_1, \mathbf{b}_1$ | $d \times 4d + 4d$ | $4d^2 + 4d$ |
| $\mathbf{W}_2, \mathbf{b}_2$ | $4d \times d + d$ | $4d^2 + d$ |
| LayerNorm (×2) | $2 \times 2d$ | $4d$ |
| **Total per block** | | $\approx 12d^2$ |

For GPT-3 with $d = 12288$, $L = 96$: approximately $96 \times 12 \times 12288^2 \approx 174$ billion parameters.

---

## 11.9 Cross-Attention: When Q, K, V Come from Different Sources

### Motivation

In encoder-decoder models (translation, etc.), the decoder needs to attend to the encoder's output. The queries come from the decoder, but keys and values come from the encoder.

### Mathematical formulation

$$\mathbf{Q} = \mathbf{X}_\text{decoder}\mathbf{W}_Q, \quad \mathbf{K} = \mathbf{X}_\text{encoder}\mathbf{W}_K, \quad \mathbf{V} = \mathbf{X}_\text{encoder}\mathbf{W}_V$$

### Dimensions

$$\mathbf{X}_\text{decoder} \in \mathbb{R}^{B \times n_\text{dec} \times d}, \quad \mathbf{X}_\text{encoder} \in \mathbb{R}^{B \times n_\text{enc} \times d}$$

$$\mathbf{Q} \in \mathbb{R}^{B \times n_\text{dec} \times d_k}, \quad \mathbf{K} \in \mathbb{R}^{B \times n_\text{enc} \times d_k}$$

Score matrix: $\mathbf{Q}\mathbf{K}^T \in \mathbb{R}^{B \times n_\text{dec} \times n_\text{enc}}$

**Note**: The score matrix is now **rectangular**, not square! Each decoder token attends to all encoder tokens.

Output: $\mathbf{A}\mathbf{V} \in \mathbb{R}^{B \times n_\text{dec} \times d_v}$ — one output per decoder token.

---

## 11.10 The GPT-Style Decoder-Only Architecture

Modern LLMs (GPT, LLaMA, etc.) use only the **decoder** part with causal masking:

```python
def gpt_forward(token_ids, model_params):
    """
    token_ids: (B, n) — integer token IDs
    
    Returns logits: (B, n, vocab_size)
    """
    B, n = token_ids.shape
    
    # 1. Embedding
    X = model_params['token_embed'][token_ids]    # (B, n, d)
    X = X + model_params['pos_embed'][:n]         # (B, n, d)
    
    # 2. Transformer blocks with causal mask
    causal_mask = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
    
    for block in model_params['blocks']:
        X = transformer_block_causal(X, block, causal_mask)
    
    # 3. Final LayerNorm
    X = layer_norm(X, model_params['ln_f'])       # (B, n, d)
    
    # 4. Language model head: project to vocabulary
    logits = X @ model_params['token_embed'].T    # (B, n, vocab_size)
    # Weight tying: reuse token embedding matrix as output projection
    
    return logits
```

### Weight tying

The output projection reuses the token embedding matrix (transposed). This reduces parameters and provides a natural "inverse" mapping from hidden states back to token probabilities.

$$\text{logits} = \mathbf{X} \mathbf{E}_\text{token}^T$$

where $\mathbf{E}_\text{token} \in \mathbb{R}^{V \times d}$ is the embedding matrix ($V$ = vocabulary size).

**Dimension**: $(B, n, d) \times (d, V) = (B, n, V)$ — a distribution over vocabulary for each position. ✓

---

## 11.11 Putting It All Together: Full Forward Pass Dimension Trace

For a GPT-style model with $B=2, n=8, d=64, h=8, d_{ff}=256, V=1000$:

| Step | Operation | Shape | Notes |
|------|-----------|-------|-------|
| Input | token IDs | $(2, 8)$ | Integer indices |
| Token embed | lookup | $(2, 8, 64)$ | $\mathbf{E}_\text{token}[ids]$ |
| Pos embed | add | $(2, 8, 64)$ | $+ \mathbf{E}_\text{pos}[:8]$ |
| LayerNorm₁ | normalize | $(2, 8, 64)$ | Per-token, over $d=64$ |
| Q/K/V proj | 3× matmul | $(2, 8, 64)$ each | Broadcasting |
| Split heads | reshape+transpose | $(2, 8, 8, 8)$ | $h=8, d_k=8$ |
| Scores | batched matmul | $(2, 8, 8, 8)$ | 4D: $B \times h \times n \times n$ |
| Scale + mask | divide + fill | $(2, 8, 8, 8)$ | $/ \sqrt{8}$, causal mask |
| Softmax | per-row | $(2, 8, 8, 8)$ | Over last dim |
| Attend | batched matmul | $(2, 8, 8, 8)$ | $\mathbf{A}\mathbf{V}$ |
| Merge heads | transpose+reshape | $(2, 8, 64)$ | Concatenate heads |
| Output proj | matmul | $(2, 8, 64)$ | $\times \mathbf{W}_O$ |
| Residual | add | $(2, 8, 64)$ | $\mathbf{X} + \text{attn}(\mathbf{X})$ |
| LayerNorm₂ | normalize | $(2, 8, 64)$ | |
| FFN W₁ | matmul | $(2, 8, 256)$ | Expand to $d_{ff}$ |
| GELU | element-wise | $(2, 8, 256)$ | |
| FFN W₂ | matmul | $(2, 8, 64)$ | Contract back to $d$ |
| Residual | add | $(2, 8, 64)$ | |
| (... repeat $L$ times ...) | | | |
| Final LN | normalize | $(2, 8, 64)$ | |
| LM head | matmul with $\mathbf{E}^T$ | $(2, 8, 1000)$ | Logits over vocab |
| Softmax | per-position | $(2, 8, 1000)$ | Probabilities |

---

## 11.12 Summary: The Design Principles

The transformer block is composed from simple, well-understood components:

1. **Attention** (Docs 9-10): Learned, content-based information routing between tokens
2. **Residual connections**: Gradient highways — identity + learned residual
3. **Layer normalization**: Stabilize activations, normalize per-token
4. **FFN**: Per-token nonlinear processing with expansion
5. **Positional encoding**: Break permutation symmetry

Each component addresses a specific failure mode:
- Without attention: tokens can't communicate
- Without residuals: gradients vanish in deep networks
- Without normalization: training is unstable
- Without FFN: model has no per-token nonlinearity beyond attention
- Without position encoding: model can't distinguish word order

The full transformer is **not magic** — it is a carefully engineered stack of operations, each derived from a clear mathematical/empirical motivation.

---

## 11.13 Exercises

1. **Gradient highway**: For a 10-layer pre-norm transformer, trace the gradient path from the loss to layer 1's input that goes ONLY through residual connections (bypassing all sub-layers). What is the gradient along this path?

2. **LayerNorm implementation**: Implement LayerNorm from scratch and verify against `nn.LayerNorm`. Check that gradients match using `torch.autograd.gradcheck`.

3. **Parameter counting**: For a transformer with $d=512, h=8, d_{ff}=2048, L=6$, compute the total number of parameters (excluding embeddings).

4. **Permutation equivariance**: Verify experimentally that without positional encoding, permuting the input tokens produces the same permutation in the output.

5. **Cross-attention dimensions**: For an encoder-decoder model with $n_\text{enc} = 20, n_\text{dec} = 10$, trace all tensor shapes through a cross-attention layer.
