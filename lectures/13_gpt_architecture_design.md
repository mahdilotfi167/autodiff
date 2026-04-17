# 13. Designing the GPT Architecture: The Decoder-Only Transformer

## Motivation

Documents 9-12 derived the transformer's core components: attention, multi-head attention, layer normalization, residual connections, FFN, and backpropagation through all of them. But a transformer is a general architecture — it can be used as an encoder (BERT), encoder-decoder (original Transformer, T5), or **decoder-only** (GPT).

The GPT family made a single, consequential design choice: **use only the decoder half of the transformer, with causal masking, trained on next-token prediction.** This document explains **why** this choice was made, what its consequences are, and how the full GPT architecture is assembled from the components we've already derived.

By the end, you'll understand every architectural decision in GPT and be able to draw the complete computation graph from memory.

---

## 13.1 Encoder vs. Decoder vs. Decoder-Only: The Design Space

### The original Transformer (Vaswani et al., 2017)

The original "Attention is All You Need" paper proposed two stacks:

1. **Encoder**: processes the input sequence with **bidirectional** self-attention (every token sees every other token)
2. **Decoder**: generates output tokens auto-regressively with **causal** self-attention (each token sees only previous tokens) plus **cross-attention** to the encoder's output

This was designed for **sequence-to-sequence** tasks (translation, summarization).

### The three architectural families

| Architecture | Attention type | Training objective | Key models |
|---|---|---|---|
| **Encoder-only** | Bidirectional | Masked language model (MLM) | BERT, RoBERTa |
| **Encoder-Decoder** | Bidirectional enc + Causal dec | Span corruption / seq2seq | T5, BART, UL2 |
| **Decoder-only** | Causal (unidirectional) | Next-token prediction | GPT-1/2/3/4, LLaMA, Chinchilla |

### Why decoder-only won for LLMs

The researcher writes on paper: "Which architecture is best for a **general-purpose** language model?"

**Argument 1: Simplicity.** A decoder-only model has **one** stack, **one** attention type, **one** objective. No cross-attention, no encoder, no separate input/output sequences. Fewer moving parts → easier to scale.

**Argument 2: Generality.** Next-token prediction subsumes other tasks. Classification? Generate "positive" or "negative." Translation? Generate the translated text. Summarization? Same. The model is a general conditional text generator: $P(\text{next token} | \text{preceding tokens})$.

**Argument 3: Scaling efficiency.** Every token in the training sequence provides a training signal (each position predicts the next). In BERT's MLM, only ~15% of tokens are masked and provide loss. Decoder-only uses 100% of positions for training.

**Argument 4: Emergent capabilities.** Empirically, decoder-only models trained at scale exhibit in-context learning (few-shot prompting), chain-of-thought reasoning, and tool use — capabilities not observed (or much weaker) in encoder-only models of similar size.

> **Design decision**: Use a decoder-only transformer for maximum generality and scaling efficiency.

---

## 13.2 The Causal Mask: What Makes a Decoder a Decoder

### The core constraint

In a decoder, when predicting token $t_i$, the model must **not** see tokens $t_{i+1}, t_{i+2}, \ldots, t_n$. Otherwise it would be "cheating" — using future information during training that won't be available during generation.

### Mathematical formulation

Recall from Doc 9 the attention score matrix:

$$\mathbf{S} = \frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}} \in \mathbb{R}^{n \times n}$$

The **causal mask** sets all entries where $j > i$ (future positions) to $-\infty$ **before** softmax:

$$S'_{ij} = \begin{cases} S_{ij} & \text{if } j \leq i \\ -\infty & \text{if } j > i \end{cases}$$

After softmax:

$$A_{ij} = \frac{\exp(S'_{ij})}{\sum_{k=1}^n \exp(S'_{ik})} = \begin{cases} \frac{\exp(S_{ij})}{\sum_{k \leq i} \exp(S_{ik})} & \text{if } j \leq i \\ 0 & \text{if } j > i \end{cases}$$

Because $\exp(-\infty) = 0$, future tokens get **zero** attention weight.

### The mask as a matrix

$$\mathbf{M} = \begin{pmatrix} 0 & -\infty & -\infty & \cdots & -\infty \\ 0 & 0 & -\infty & \cdots & -\infty \\ 0 & 0 & 0 & \cdots & -\infty \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & 0 & \cdots & 0 \end{pmatrix}$$

Applied as: $\mathbf{S}' = \mathbf{S} + \mathbf{M}$

In PyTorch:

```python
def causal_mask(n, device='cpu'):
    """Lower-triangular mask: 0 for allowed positions, -inf for masked."""
    mask = torch.triu(torch.ones(n, n, device=device), diagonal=1) * float('-inf')
    return mask  # shape: (n, n)

# Usage in attention
scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)  # (B, h, n, n)
scores = scores + causal_mask(n, device=scores.device)  # broadcast over B, h
attn_weights = torch.softmax(scores, dim=-1)
```

### Why additive mask, not multiplicative?

The researcher considers two options:

1. **Multiplicative**: $\mathbf{S}' = \mathbf{S} \odot \mathbf{M}_{\text{binary}}$ where $M_{ij} = \mathbf{1}[j \leq i]$
   - Problem: zeroed-out entries become 0 in scores, but $\text{softmax}(0) = \frac{1}{\text{partition}} \neq 0$. The model would still attend to future tokens!

2. **Additive** with $-\infty$: $\mathbf{S}' = \mathbf{S} + \mathbf{M}_{-\infty}$
   - $\exp(-\infty) = 0$, so these entries contribute **exactly zero** to the softmax output. ✓

---

## 13.3 The GPT Blueprint: Full Architecture

### High-level structure

```
Input token IDs: (B, n)
        ↓
[Token Embedding]  →  (B, n, d_model)
        +
[Positional Encoding]  →  (B, n, d_model)
        ↓
╔══════════════════════════════════╗
║  Transformer Block ×N_layers     ║
║  ┌─────────────────────────┐     ║
║  │ LayerNorm               │     ║
║  │ Causal Multi-Head Attn  │     ║
║  │ + Residual Connection   │     ║
║  ├─────────────────────────┤     ║
║  │ LayerNorm               │     ║
║  │ FFN (MLP)               │     ║
║  │ + Residual Connection   │     ║
║  └─────────────────────────┘     ║
╚══════════════════════════════════╝
        ↓
[Final LayerNorm]
        ↓
[Linear (unembedding)]  →  (B, n, V)   where V = vocab size
        ↓
[Softmax / Cross-Entropy Loss]
```

### Precise dimensions at each stage

Let:
- $B$ = batch size
- $n$ = sequence length (context window)
- $d$ = model dimension (`d_model`)
- $h$ = number of attention heads
- $d_k = d / h$ = dimension per head
- $d_{\text{ff}}$ = FFN inner dimension (typically $4d$)
- $V$ = vocabulary size
- $N$ = number of transformer layers

| Stage | Output shape | Parameters |
|---|---|---|
| Token embedding | $(B, n, d)$ | $\mathbf{W}_E \in \mathbb{R}^{V \times d}$ |
| Positional encoding | $(B, n, d)$ | Depends on method (see Doc 14) |
| Layer norm | $(B, n, d)$ | $\boldsymbol{\gamma}, \boldsymbol{\beta} \in \mathbb{R}^d$ |
| Q, K, V projections | $(B, n, d)$ each | $\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V \in \mathbb{R}^{d \times d}$ |
| Reshape to heads | $(B, h, n, d_k)$ | — |
| Attention scores | $(B, h, n, n)$ | — |
| Attention output | $(B, h, n, d_k)$ | — |
| Concat + output proj | $(B, n, d)$ | $\mathbf{W}_O \in \mathbb{R}^{d \times d}$ |
| FFN layer 1 | $(B, n, d_{\text{ff}})$ | $\mathbf{W}_1 \in \mathbb{R}^{d \times d_\text{ff}}, \mathbf{b}_1 \in \mathbb{R}^{d_\text{ff}}$ |
| FFN activation | $(B, n, d_{\text{ff}})$ | — |
| FFN layer 2 | $(B, n, d)$ | $\mathbf{W}_2 \in \mathbb{R}^{d_\text{ff} \times d}, \mathbf{b}_2 \in \mathbb{R}^d$ |
| Final layer norm | $(B, n, d)$ | $\boldsymbol{\gamma}, \boldsymbol{\beta} \in \mathbb{R}^d$ |
| Unembedding (LM head) | $(B, n, V)$ | $\mathbf{W}_U \in \mathbb{R}^{d \times V}$ (often tied to $\mathbf{W}_E^T$) |

---

## 13.4 Pre-Norm vs. Post-Norm: A Critical Design Choice

### Post-Norm (original Transformer)

$$\mathbf{Y} = \text{LayerNorm}(\mathbf{X} + \text{SubLayer}(\mathbf{X}))$$

LayerNorm is applied **after** the residual addition.

### Pre-Norm (GPT-2 and later)

$$\mathbf{Y} = \mathbf{X} + \text{SubLayer}(\text{LayerNorm}(\mathbf{X}))$$

LayerNorm is applied **before** the sublayer, and the residual bypasses normalization.

### Why Pre-Norm?

The researcher analyzes gradient flow:

**Post-Norm gradient**:
$$\frac{\partial \mathbf{Y}}{\partial \mathbf{X}} = \frac{\partial}{\partial \mathbf{X}} \text{LayerNorm}(\mathbf{X} + \text{SubLayer}(\mathbf{X}))$$

The LayerNorm wraps everything — gradients must flow **through** the normalization, which involves division by standard deviation. If the std is very small, gradients can explode.

**Pre-Norm gradient**:
$$\frac{\partial \mathbf{Y}}{\partial \mathbf{X}} = \mathbf{I} + \frac{\partial \text{SubLayer}(\text{LayerNorm}(\mathbf{X}))}{\partial \mathbf{X}}$$

The identity path $\mathbf{I}$ is **clean** — no normalization operation between residual streams. Gradients have an unobstructed highway from the loss to the first layer.

**Empirical consequence**: Pre-Norm is much more stable during training. Post-Norm often requires careful learning rate warmup and can diverge at large scale. Pre-Norm is used by GPT-2, GPT-3, LLaMA, and most modern LLMs.

> **Design decision**: Use Pre-Norm for training stability at scale.

### The extra final LayerNorm

With Pre-Norm, the output of the last transformer block has **not** been normalized (the norm happens inside each block, before the sublayer). So GPT adds a **final LayerNorm** before the unembedding projection:

$$\text{logits} = \text{LayerNorm}(\mathbf{h}_N) \cdot \mathbf{W}_U$$

---

## 13.5 The Unembedding Layer and Weight Tying

### The unembedding projection

The final layer maps from the $d$-dimensional representation space to the $V$-dimensional vocabulary space:

$$\text{logits} = \mathbf{h} \cdot \mathbf{W}_U \in \mathbb{R}^{B \times n \times V}$$

where $\mathbf{W}_U \in \mathbb{R}^{d \times V}$.

Each row of the output gives the unnormalized log-probability (logit) of every token in the vocabulary.

### Weight tying

The embedding matrix $\mathbf{W}_E \in \mathbb{R}^{V \times d}$ maps token IDs to vectors. The unembedding matrix $\mathbf{W}_U \in \mathbb{R}^{d \times V}$ maps vectors back to vocabulary logits.

**Observation**: $\mathbf{W}_U$ should be $\mathbf{W}_E^T$. The embedding teaches the model: "token $i$ corresponds to vector $\mathbf{e}_i$." The unembedding should measure: "how similar is this hidden state to each token's embedding?"

$$\text{logit}_i = \mathbf{h}^T \mathbf{e}_i = (\mathbf{W}_E^T \mathbf{h})_i$$

This is **weight tying** (Press & Wolf, 2017):

$$\mathbf{W}_U = \mathbf{W}_E^T$$

**Benefits**:
1. **Fewer parameters**: saves $d \times V$ parameters (for GPT-3 175B with $V=50257$ and $d=12288$, this saves ~617M parameters)
2. **Better generalization**: the output space is constrained to be consistent with the input space
3. **Semantic coherence**: tokens with similar embeddings produce similar logits

**Not all models use tying**: GPT-2 uses weight tying. GPT-3 does NOT (separate $\mathbf{W}_E$ and $\mathbf{W}_U$). LLaMA does NOT. At very large scale, the model has enough capacity that tying can slightly hurt performance.

---

## 13.6 The Training Objective: Next-Token Prediction

### Autoregressive language modeling

Given a sequence of tokens $(t_1, t_2, \ldots, t_n)$, the model is trained to predict each token given all preceding tokens:

$$P(t_1, t_2, \ldots, t_n) = \prod_{i=1}^n P(t_i | t_1, \ldots, t_{i-1})$$

This factorization is exact (chain rule of probability) — no approximation. The model learns the conditional distribution at each position.

### The loss function

For a single sequence, the cross-entropy loss is:

$$\mathcal{L} = -\frac{1}{n} \sum_{i=1}^n \log P_\theta(t_i | t_1, \ldots, t_{i-1})$$

Where $P_\theta(t_i | \cdot)$ is the softmax of the logit for the correct token $t_i$:

$$P_\theta(t_i | t_1, \ldots, t_{i-1}) = \frac{\exp(z_{t_i})}{\sum_{v=1}^V \exp(z_v)}$$

where $z_v$ is the logit for vocabulary token $v$ at position $i$.

### How the causal mask creates $n$ training examples from one sequence

This is elegant. Given a single sequence of length $n$:

| Position $i$ | Sees tokens | Predicts |
|---|---|---|
| 1 | (start) | $t_1$ |
| 2 | $t_1$ | $t_2$ |
| 3 | $t_1, t_2$ | $t_3$ |
| $\vdots$ | $\vdots$ | $\vdots$ |
| $n$ | $t_1, \ldots, t_{n-1}$ | $t_n$ |

**All $n$ predictions happen in parallel** in a single forward pass, thanks to the causal mask. The mask ensures position $i$ only attends to positions $\leq i$, so each position's output is a valid conditional prediction.

This is **far more efficient** than training $n$ separate models or making $n$ sequential forward passes. One forward pass, $n$ loss terms.

### Teacher forcing

During training, we always feed the **ground truth** tokens as input, not the model's own predictions. This is called "teacher forcing." The model sees the real sequence at every position — the causal mask prevents information leakage, not the input itself.

During **generation** (inference), we switch to autoregressive mode: compute one token at a time, feeding each generated token back as input.

---

## 13.7 Activation Functions: GELU, SwiGLU, and Why ReLU Is Obsolete

### The FFN sublayer

In the original Transformer:

$$\text{FFN}(\mathbf{x}) = \text{ReLU}(\mathbf{x}\mathbf{W}_1 + \mathbf{b}_1)\mathbf{W}_2 + \mathbf{b}_2$$

### Why not ReLU?

ReLU: $f(x) = \max(0, x)$

**Problem**: ReLU has a hard zero for $x < 0$. This means:
- Gradients are exactly 0 for negative inputs ("dying ReLU" problem)
- The function has a discontinuous derivative at $x=0$
- In practice, large fractions of neurons can become permanently dead during training

### GELU (Gaussian Error Linear Unit) — GPT-1/2/3

$$\text{GELU}(x) = x \cdot \Phi(x) = x \cdot \frac{1}{2}\left[1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right]$$

where $\Phi(x)$ is the CDF of the standard normal distribution.

**Intuition**: GELU multiplies $x$ by a soft gate $\Phi(x) \in [0, 1]$. For large positive $x$, $\Phi(x) \approx 1$, so $\text{GELU}(x) \approx x$. For large negative $x$, $\Phi(x) \approx 0$, so $\text{GELU}(x) \approx 0$. Near zero, the gate transitions smoothly.

**Key property**: GELU is smooth everywhere — no discontinuous gradients. This leads to better optimization dynamics.

### SwiGLU — LLaMA, PaLM, Gemma

The **Gated Linear Unit (GLU)** family introduces a learned gating mechanism:

$$\text{SwiGLU}(\mathbf{x}) = (\mathbf{x}\mathbf{W}_1) \odot \text{SiLU}(\mathbf{x}\mathbf{W}_{\text{gate}})$$

where $\text{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$ (also called "Swish").

**Note**: SwiGLU has **three** weight matrices ($\mathbf{W}_1, \mathbf{W}_{\text{gate}}, \mathbf{W}_2$) instead of two, so the inner dimension $d_{\text{ff}}$ is typically adjusted to $\frac{8}{3}d$ (rounded to a multiple of 256) to keep parameter count similar:

$$\text{FFN}_{\text{SwiGLU}}(\mathbf{x}) = \left[(\mathbf{x}\mathbf{W}_1) \odot \text{SiLU}(\mathbf{x}\mathbf{W}_{\text{gate}})\right] \mathbf{W}_2$$

**Why is SwiGLU better?** The gating mechanism allows the network to learn **which features to let through**, independently of the value. This provides more expressive control than a single non-linearity. Empirically, SwiGLU consistently outperforms GELU and ReLU at matched parameter counts.

```python
class SwiGLU_FFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.w2(self.w1(x) * F.silu(self.w_gate(x)))
```

### Summary of activation evolution

| Model | Activation | FFN formula | Inner dim |
|---|---|---|---|
| Original Transformer | ReLU | $\text{ReLU}(\mathbf{x}\mathbf{W}_1)\mathbf{W}_2$ | $4d$ |
| GPT-1/2/3 | GELU | $\text{GELU}(\mathbf{x}\mathbf{W}_1)\mathbf{W}_2$ | $4d$ |
| LLaMA, PaLM | SwiGLU | $(\mathbf{x}\mathbf{W}_1 \odot \text{SiLU}(\mathbf{x}\mathbf{W}_g))\mathbf{W}_2$ | $\frac{8}{3}d$ |

---

## 13.8 Parameter Count: Where the Parameters Live

### Counting parameters for a single transformer layer

For one transformer block with Pre-Norm, multi-head attention, and FFN:

**Multi-Head Attention:**
- $\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V$: each $d \times d$ → $3d^2$ parameters
- $\mathbf{W}_O$: $d \times d$ → $d^2$ parameters
- Attention total: $4d^2$

**FFN (standard):**
- $\mathbf{W}_1$: $d \times 4d$ → $4d^2$
- $\mathbf{W}_2$: $4d \times d$ → $4d^2$
- Biases: $4d + d = 5d$ (if used)
- FFN total: $8d^2$ (ignoring biases)

**Layer Norms** (2 per block):
- Each: $2d$ (scale $\gamma$ + shift $\beta$)
- Total: $4d$

**One transformer block total**: $\approx 12d^2$ (ignoring biases and small terms)

### Total model parameter count

$$P_{\text{total}} = \underbrace{Vd}_{\text{token embed}} + \underbrace{N \cdot 12d^2}_{\text{transformer blocks}} + \underbrace{2d}_{\text{final LN}} + \underbrace{Vd}_{\text{unembed (if untied)}}$$

For models using weight tying, the last term is zero.

### Real-world examples

| Model | $N$ | $d$ | $h$ | $d_{\text{ff}}$ | $V$ | Total params |
|---|---|---|---|---|---|---|
| GPT-2 Small | 12 | 768 | 12 | 3072 | 50257 | 124M |
| GPT-2 Medium | 24 | 1024 | 16 | 4096 | 50257 | 355M |
| GPT-2 Large | 36 | 1280 | 20 | 5120 | 50257 | 774M |
| GPT-2 XL | 48 | 1600 | 25 | 6400 | 50257 | 1.5B |
| GPT-3 | 96 | 12288 | 96 | 49152 | 50257 | 175B |
| LLaMA-7B | 32 | 4096 | 32 | 11008 | 32000 | 6.7B |
| LLaMA-65B | 80 | 8192 | 64 | 22016 | 32000 | 65.2B |
| LLaMA-3-70B | 80 | 8192 | 64 | 28672 | 128256 | 70.6B |

### Observation: The $12d^2 N$ rule

For large models, the embedding parameters ($Vd$) are a small fraction. The parameter count is dominated by:

$$P \approx 12 d^2 N$$

This gives a useful back-of-the-envelope formula. For GPT-3: $12 \times 12288^2 \times 96 \approx 173\text{B}$ ✓

---

## 13.9 The Attention Sink and Biases: GPT Design Details

### Bias terms: to include or not?

| Component | GPT-2 | GPT-3 | LLaMA |
|---|---|---|---|
| QKV projection bias | Yes | Yes | **No** |
| Output projection bias | Yes | Yes | **No** |
| FFN bias | Yes | Yes | **No** |
| LayerNorm bias ($\beta$) | Yes | Yes | **No** (uses RMSNorm) |

**Modern trend**: Remove all biases. This simplifies the model, slightly reduces parameters, and empirically doesn't hurt performance. LLaMA demonstrated that biases are unnecessary at scale.

### RMSNorm vs. LayerNorm

LLaMA and many modern models replace LayerNorm with **RMSNorm** (Root Mean Square Normalization):

$$\text{RMSNorm}(\mathbf{x}) = \frac{\mathbf{x}}{\text{RMS}(\mathbf{x})} \odot \boldsymbol{\gamma}$$

where:
$$\text{RMS}(\mathbf{x}) = \sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2 + \epsilon}$$

**Difference from LayerNorm**: RMSNorm does not subtract the mean — it only divides by the root mean square. This removes the centering step and the learnable bias $\beta$.

**Why**: Empirically, the re-centering (mean subtraction) in LayerNorm contributes little. RMSNorm is ~10-15% faster (one less reduction operation) with equivalent model quality.

---

## 13.10 Putting It All Together: The Complete GPT Forward Pass

### Pseudocode

```python
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_enc = PositionalEncoding(config)  # See Doc 14
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        self.ln_f = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying (optional)
        # self.lm_head.weight = self.tok_emb.weight

    def forward(self, token_ids):
        # token_ids: (B, n) — integer token IDs
        B, n = token_ids.shape

        # Embedding + positional encoding
        x = self.tok_emb(token_ids)       # (B, n, d)
        x = self.pos_enc(x)               # (B, n, d) — adds position info
        x = self.drop(x)

        # N transformer blocks
        for block in self.blocks:
            x = block(x)                   # (B, n, d)

        # Final norm + unembedding
        x = self.ln_f(x)                  # (B, n, d)
        logits = self.lm_head(x)          # (B, n, V)

        return logits


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = RMSNorm(config.d_model)
        self.attn = CausalMultiHeadAttention(config)
        self.ln2 = RMSNorm(config.d_model)
        self.ffn = SwiGLU_FFN(config.d_model, config.d_ff)

    def forward(self, x):
        # Pre-Norm with residual connections
        x = x + self.attn(self.ln1(x))    # (B, n, d)
        x = x + self.ffn(self.ln2(x))     # (B, n, d)
        return x
```

### The computation graph (complete)

```
token_ids: (B, n)
    │
    ▼
[Embedding lookup]  ──→  x: (B, n, d)
    │
    ▼
[+ Positional Encoding]  ──→  x: (B, n, d)
    │
    ▼
[Dropout]
    │
    ╔═══════════════════════════════════════════════════════╗
    ║  Block 1                                              ║
    ║  ┌──────────┐    ┌───────┐    ┌───────────┐          ║
    ║  │ RMSNorm  │───→│ CMHA  │───→│ + residual│──→ x     ║
    ║  └──────────┘    └───────┘    └───────────┘          ║
    ║       └───────────────────────────┘↑ skip connection  ║
    ║                                                       ║
    ║  ┌──────────┐    ┌───────┐    ┌───────────┐          ║
    ║  │ RMSNorm  │───→│ FFN   │───→│ + residual│──→ x     ║
    ║  └──────────┘    └───────┘    └───────────┘          ║
    ║       └───────────────────────────┘↑ skip connection  ║
    ╚═══════════════════════════════════════════════════════╝
    │
    ... (×N layers)
    │
    ▼
[Final RMSNorm]  ──→  (B, n, d)
    │
    ▼
[LM Head (linear)]  ──→  logits: (B, n, V)
    │
    ▼
[Cross-entropy with targets shifted by 1]  ──→  scalar loss
```

---

## 13.11 GPT-1 → GPT-2 → GPT-3: What Changed?

### GPT-1 (Radford et al., 2018)

- 12 layers, $d=768$, 12 heads → 117M parameters
- Learned absolute positional embeddings
- Post-Norm (original Transformer style)
- GELU activation
- Pre-trained on BookCorpus (~800M tokens)
- Fine-tuned on specific tasks with task-specific heads

### GPT-2 (Radford et al., 2019)

Key changes from GPT-1:
- **Pre-Norm** instead of Post-Norm
- **No task-specific fine-tuning**: zero-shot/few-shot only
- Larger: up to 1.5B parameters
- Better data: WebText (~40GB, ~8M documents)
- Demonstrated that language models can perform tasks **without explicit fine-tuning**
- Moved LayerNorm before attention/FFN (Pre-Norm)
- Added extra LayerNorm after final transformer block

### GPT-3 (Brown et al., 2020)

Key changes from GPT-2:
- **Massive scale**: 175B parameters (117× GPT-2 XL)
- **Alternating dense and sparse attention** in some layers (banded sparse patterns)
- Trained on 300B tokens from filtered Common Crawl + books + Wikipedia
- **In-context learning**: the model can perform tasks from a few examples in the prompt, no gradient updates needed
- **No weight tying** between embedding and unembedding
- Demonstrated **emergent abilities** that appear at scale

### The architectural evolution summary

| Feature | GPT-1 | GPT-2 | GPT-3 | Modern (LLaMA-style) |
|---|---|---|---|---|
| Norm | Post-Norm | Pre-Norm | Pre-Norm | Pre-Norm |
| Norm type | LayerNorm | LayerNorm | LayerNorm | RMSNorm |
| Activation | GELU | GELU | GELU | SwiGLU |
| Position encoding | Learned absolute | Learned absolute | Learned absolute | RoPE |
| Weight tying | Yes | Yes | No | No |
| Biases | Yes | Yes | Yes | No |
| Attention | MHA | MHA | MHA | GQA |
| Vocab size | 40478 | 50257 | 50257 | 32000-128000+ |

---

## 13.12 Key Takeaways

1. **Decoder-only** was chosen for simplicity, generality, and scaling efficiency. The causal mask is the single structural change from a bidirectional transformer.

2. **Pre-Norm** over Post-Norm for stable gradients — the residual stream has a clean identity path.

3. **Weight tying** saves parameters but is optional at large scale.

4. **The loss is next-token prediction** — cross-entropy on shifted logits. Every position in the sequence provides a training signal.

5. **Parameter count** scales as $\approx 12d^2 N$, dominated by the attention and FFN weight matrices.

6. **Modern refinements** (RMSNorm, SwiGLU, no biases, RoPE, GQA) each provide small but compounding improvements. We derive these in subsequent documents.

7. The architecture is remarkably simple: **embedding → N × (norm + attention + residual + norm + FFN + residual) → norm → unembed**. The power comes from scale, data, and training.
