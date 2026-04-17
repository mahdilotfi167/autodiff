# 19. Scaling Laws, Architectural Advances, and State-of-the-Art

## Motivation

Documents 13-18 covered the mechanics of building and training a GPT. But how do you decide **how big** to make it? How much data does it need? And what efficiency techniques allow frontier models to push beyond brute-force scaling?

This document covers **scaling laws** (the mathematical relationships between compute, parameters, and data), **architectural innovations** that improve efficiency (GQA, MoE, FlashAttention, KV cache), and a survey of state-of-the-art models with their key design decisions.

---

## 19.1 Scaling Laws: The Science of "How Big?"

### The Kaplan scaling laws (OpenAI, 2020)

Kaplan et al. discovered that language model performance follows **power laws** with respect to model size, dataset size, and compute:

$$L(N) = \left(\frac{N_c}{N}\right)^{\alpha_N}, \quad L(D) = \left(\frac{D_c}{D}\right)^{\alpha_D}, \quad L(C) = \left(\frac{C_c}{C}\right)^{\alpha_C}$$

where $L$ is the cross-entropy loss, $N$ is parameters, $D$ is dataset size (tokens), $C$ is compute (FLOPS), and $N_c, D_c, C_c, \alpha_N, \alpha_D, \alpha_C$ are fitted constants.

Fitted values:
- $\alpha_N \approx 0.076$ (loss decreases as $N^{-0.076}$)
- $\alpha_D \approx 0.095$ (loss decreases as $D^{-0.095}$)
- $\alpha_C \approx 0.050$ (loss decreases as $C^{-0.050}$)

### The compute-optimal allocation (Chinchilla scaling, 2022)

Hoffmann et al. (DeepMind) asked: **given a fixed compute budget $C$, how should we split it between model size $N$ and data $D$?**

The compute approximation:
$$C \approx 6ND$$

(6 FLOPS per parameter per token: 2 forward + 4 backward)

### Kaplan vs. Chinchilla: contradictory recommendations

| | Kaplan (2020) | Chinchilla (2022) |
|---|---|---|
| Optimal $N$ scaling | $N \propto C^{0.73}$ | $N \propto C^{0.50}$ |
| Optimal $D$ scaling | $D \propto C^{0.27}$ | $D \propto C^{0.50}$ |
| Recommendation | Scale model size much faster | **Scale data and model equally** |
| Tokens per parameter | ~20 | **~20** (but different $N$!) |

**Chinchilla's finding**: Kaplan massively underfitted on data. Most existing models (GPT-3, etc.) were **too large for their training data**. A smaller model trained on more data consistently outperforms a larger model trained on less data.

### The Chinchilla formula

For compute-optimal training:
$$N_{\text{opt}} = 0.6 \cdot \frac{C^{0.5}}{6}, \qquad D_{\text{opt}} = 0.6 \cdot \frac{C^{0.5}}{6}$$

Or equivalently: **train for approximately 20 tokens per parameter**.

| Compute budget | Optimal $N$ | Optimal $D$ | Tokens/param |
|---|---|---|---|
| $10^{18}$ FLOPS | ~400M | ~8B | ~20 |
| $10^{20}$ FLOPS | ~4B | ~80B | ~20 |
| $10^{22}$ FLOPS | ~40B | ~800B | ~20 |
| $10^{24}$ FLOPS | ~400B | ~8T | ~20 |

### Were existing models Chinchilla-optimal?

| Model | $N$ | $D$ | Tokens/param | Chinchilla-optimal? |
|---|---|---|---|---|
| GPT-3 | 175B | 300B | 1.7 | ❌ Severely undertrained |
| Chinchilla | 70B | 1.4T | 20 | ✅ By design |
| LLaMA-1-7B | 7B | 1T | 143 | ❌ Overtrained (but intentionally) |
| LLaMA-1-65B | 65B | 1.4T | 21.5 | ✅ Near-optimal |
| LLaMA-3-8B | 8B | 15T | 1875 | ❌ Massively overtrained |
| LLaMA-3-70B | 70B | 15T | 214 | ❌ Overtrained |

### Beyond Chinchilla: inference-optimal training

Chinchilla optimizes for **training compute**. But in production, the model serves millions of users. **Inference compute** dominates the total cost.

A smaller model that's been trained longer:
- Costs more to train (more tokens than Chinchilla-optimal)
- Costs less to serve per query (fewer parameters → faster inference)
- Can be the overall most cost-efficient

This is why LLaMA-3 is "overtrained" — it's optimized for **deployment**, not training efficiency.

The **inference-aware** optimal allocation:
$$D_{\text{opt}} \propto N^{0.5+\Delta}$$

where $\Delta$ depends on the expected inference volume. More inference → more overtraining.

---

## 19.2 Grouped-Query Attention (GQA): Reducing KV Cache

### The KV cache problem

During autoregressive generation, we cache the Key and Value matrices to avoid recomputation:

$$\text{KV cache per layer} = 2 \times n \times h \times d_k \times \text{bytes}$$

For LLaMA-70B with full MHA ($h=64, d_k=128, n=8192$, bf16):
$$\text{KV cache} = 2 \times 8192 \times 64 \times 128 \times 2 \times 80 \text{ layers} = 21.5 \text{ GB per sequence!}$$

For batch size 32: $\approx 688$ GB just for KV cache. This is the **bottleneck** for inference throughput.

### Three attention variants

**Multi-Head Attention (MHA)**: each head has its own Q, K, V.
- $h$ query heads, $h$ KV heads
- KV cache: $O(n \cdot h \cdot d_k)$ per layer

**Multi-Query Attention (MQA)**: all heads **share** one K, V pair.
- $h$ query heads, **1** KV head
- KV cache: $O(n \cdot d_k)$ per layer — $h\times$ reduction!
- Con: quality loss due to shared KV

**Grouped-Query Attention (GQA)**: a middle ground. Groups of query heads share K, V.
- $h$ query heads, $h_{\text{kv}}$ KV heads (where $h_{\text{kv}}$ divides $h$)
- Each KV head serves $h / h_{\text{kv}}$ query heads
- KV cache: $O(n \cdot h_{\text{kv}} \cdot d_k)$ per layer

### The GQA computation

```
Standard MHA (h=32 heads, each has Q, K, V):
  Q heads: Q₁, Q₂, ..., Q₃₂
  K heads: K₁, K₂, ..., K₃₂
  V heads: V₁, V₂, ..., V₃₂

GQA with h_kv=8 (4 query heads per KV group):
  Q heads: Q₁, Q₂, Q₃, Q₄,  Q₅, Q₆, Q₇, Q₈,  ..., Q₂₉, Q₃₀, Q₃₁, Q₃₂
  K heads: K₁, K₁, K₁, K₁,  K₂, K₂, K₂, K₂,  ..., K₈,  K₈,  K₈,  K₈
  V heads: V₁, V₁, V₁, V₁,  V₂, V₂, V₂, V₂,  ..., V₈,  V₈,  V₈,  V₈
```

### GQA implementation

```python
class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads  # queries per KV group
        self.head_dim = d_model // n_heads

        self.wq = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, d_model, bias=False)

    def forward(self, x):
        B, n, _ = x.shape
        q = self.wq(x).view(B, n, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, n, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, n, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Repeat KV heads for each query group
        k = k.repeat_interleave(self.n_rep, dim=1)  # (B, n_heads, n, d_k)
        v = v.repeat_interleave(self.n_rep, dim=1)

        # Standard attention from here
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores + causal_mask(n, device=x.device)
        attn = F.softmax(scores, dim=-1)
        out = attn @ v  # (B, n_heads, n, d_k)

        out = out.transpose(1, 2).contiguous().view(B, n, -1)
        return self.wo(out)
```

### GQA in real models

| Model | Type | $h$ | $h_{\text{kv}}$ | KV reduction |
|---|---|---|---|---|
| GPT-3 | MHA | 96 | 96 | 1× |
| LLaMA-1 | MHA | 32/40/52/64 | Same | 1× |
| LLaMA-2-70B | GQA | 64 | 8 | 8× |
| LLaMA-3-8B | GQA | 32 | 8 | 4× |
| LLaMA-3-70B | GQA | 64 | 8 | 8× |
| Mistral-7B | GQA | 32 | 8 | 4× |
| Gemma-7B | MHA | 16 | 16 | 1× |

---

## 19.3 FlashAttention: Making Attention IO-Aware

### The bottleneck is memory bandwidth, not compute

Standard attention:
1. Compute $\mathbf{S} = \mathbf{Q}\mathbf{K}^T$ → $(B, h, n, n)$ — **write** to GPU HBM
2. Compute $\mathbf{A} = \text{softmax}(\mathbf{S})$ → **read** $\mathbf{S}$ from HBM, **write** $\mathbf{A}$
3. Compute $\mathbf{O} = \mathbf{A}\mathbf{V}$ → **read** $\mathbf{A}$ from HBM

The $n \times n$ attention matrix is read/written multiple times. For $n = 8192$: this is 67M entries per head per layer. The computation is fast, but moving data to/from GPU memory (HBM) is slow.

### The FlashAttention approach (Dao et al., 2022)

**Key idea**: never materialize the full $n \times n$ attention matrix. Instead, compute attention in **tiles** that fit in GPU SRAM (on-chip memory, ~20 MB, 10-100× faster than HBM).

**Algorithm sketch**:
1. Divide Q, K, V into blocks
2. For each block of Q:
   a. Load the Q block into SRAM
   b. For each block of K, V:
      - Load K, V block into SRAM
      - Compute partial attention scores
      - Update running softmax (online softmax algorithm)
      - Update running output
   c. Write final output for this Q block to HBM

### Online softmax

The challenge: softmax requires knowing the **maximum** value across the entire row (for numerical stability). But we're processing K in blocks.

**Solution**: maintain a running maximum and scaling factor:

$$m_{i+1} = \max(m_i, \max(\text{new scores}))$$
$$\ell_{i+1} = e^{m_i - m_{i+1}} \cdot \ell_i + \sum e^{\text{new scores} - m_{i+1}}$$
$$\mathbf{O}_{i+1} = \frac{e^{m_i - m_{i+1}} \cdot \ell_i \cdot \mathbf{O}_i + e^{\text{new scores} - m_{i+1}} \cdot \text{new V}}{\ell_{i+1}}$$

### FlashAttention vs. standard attention

| | Standard | FlashAttention |
|---|---|---|
| HBM reads/writes | $O(n^2)$ | $O(n^2 d / M)$ where $M$ = SRAM size |
| Peak memory | $O(n^2)$ | $O(n)$ |
| Wall-clock speed | Baseline | **2-4× faster** |
| Exact computation | ✓ | ✓ (NOT an approximation!) |

**Critical**: FlashAttention computes the **exact same result** as standard attention. It's an IO optimization, not an approximation.

### FlashAttention-2 and FlashAttention-3

| Version | Key improvement |
|---|---|
| FlashAttention-1 | Tiled attention, online softmax |
| FlashAttention-2 | Better GPU work partitioning, 2× faster than FA-1 |
| FlashAttention-3 | Asynchronous computation, FP8 support (H100) |

FlashAttention is now the **default** attention implementation in all major frameworks.

---

## 19.4 KV Cache and Efficient Inference

### The KV cache mechanism

During autoregressive generation, each new token only needs to attend to all previous tokens. The Q for the new token is $(B, h, 1, d_k)$, but K and V span all positions.

**Without cache**: recompute K, V for all previous tokens at every generation step. Cost: $O(n^2)$ per token generated.

**With cache**: store K, V from all previous steps. At each step, only compute K, V for the new token and append to the cache.

```python
class CachedAttention:
    def __init__(self):
        self.k_cache = None  # (B, h, 0, d_k) initially
        self.v_cache = None

    def forward(self, q, k, v, mask=None):
        # q: (B, h, 1, d_k) — just the new token
        # k, v: (B, h, 1, d_k) — K, V for just the new token

        if self.k_cache is not None:
            k = torch.cat([self.k_cache, k], dim=2)  # (B, h, n+1, d_k)
            v = torch.cat([self.v_cache, v], dim=2)

        self.k_cache = k
        self.v_cache = v

        # Standard attention: (B, h, 1, d_k) @ (B, h, d_k, n+1) = (B, h, 1, n+1)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        if mask is not None:
            scores = scores + mask
        attn = F.softmax(scores, dim=-1)
        out = attn @ v  # (B, h, 1, d_k)
        return out
```

### KV cache memory budget

$$\text{KV cache} = 2 \times N_{\text{layers}} \times n_{\text{ctx}} \times h_{\text{kv}} \times d_k \times \text{bytes}$$

| Model | $N$ | $h_{\text{kv}}$ | $d_k$ | $n_{\text{ctx}}$ | KV cache (bf16) |
|---|---|---|---|---|---|
| LLaMA-3-8B | 32 | 8 | 128 | 8192 | 2 × 32 × 8192 × 8 × 128 × 2 = **1 GB** |
| LLaMA-3-70B | 80 | 8 | 128 | 8192 | 2 × 80 × 8192 × 8 × 128 × 2 = **2.6 GB** |
| GPT-3 | 96 | 96 | 128 | 2048 | 2 × 96 × 2048 × 96 × 128 × 2 = **9.2 GB** |

**Key insight**: GQA's $h_{\text{kv}}$ reduction directly translates to proportionally smaller KV cache.

### PagedAttention (vLLM)

For serving multiple requests, KV caches have different lengths and are allocated/freed dynamically.

**Problem**: naive allocation wastes memory with fragmentation.

**PagedAttention** (Kwon et al., 2023): manage KV cache like virtual memory pages. Cache blocks are non-contiguous in physical memory but contiguous in logical space.

**Result**: ~2-4× more requests can be served simultaneously, dramatically improving throughput.

---

## 19.5 Mixture of Experts (MoE): Conditional Computation

### The idea

Instead of one large FFN, use $E$ smaller FFN "experts" and a **router** that selects $k$ experts per token:

$$\text{MoE-FFN}(\mathbf{x}) = \sum_{i \in \text{TopK}(G(\mathbf{x}))} G(\mathbf{x})_i \cdot \text{Expert}_i(\mathbf{x})$$

where $G(\mathbf{x})$ is the router that produces a probability distribution over experts:

$$G(\mathbf{x}) = \text{softmax}(\mathbf{x} \cdot \mathbf{W}_{\text{gate}})$$

and $\text{TopK}$ selects the $k$ highest-scoring experts.

### Why MoE?

**The key property**: total parameters are much larger than **active** parameters.

| Metric | Dense 70B | MoE 8×22B |
|---|---|---|
| Total parameters | 70B | 176B (8 experts × 22B) |
| Active parameters per token | 70B | ~44B (top-2 routing) |
| Training FLOPS per token | $\propto 70\text{B}$ | $\propto 44\text{B}$ |
| Quality | Good | Better (more total knowledge) |

MoE gets the **quality** of a much larger model at the **compute cost** of a smaller model.

### MoE architecture details

Only the FFN layers are replaced with MoE. Attention layers remain dense (shared across all tokens).

```
Standard: Attention → FFN
MoE:      Attention → Router → Top-k Experts → Weighted sum
```

### Load balancing

**Problem**: without regularization, the router learns to send all tokens to 1-2 experts ("expert collapse"). The other experts go unused.

**Solution**: auxiliary load-balancing loss:

$$\mathcal{L}_{\text{balance}} = \alpha \cdot E \cdot \sum_{i=1}^{E} f_i \cdot P_i$$

where $f_i$ = fraction of tokens routed to expert $i$ and $P_i$ = average router probability for expert $i$.

This loss penalizes uneven distribution: it is minimized when all experts handle equal load.

### MoE models

| Model | Experts | Active | Total params | Active params |
|---|---|---|---|---|
| Mixtral 8×7B | 8 | 2 | 46.7B | 12.9B |
| Mixtral 8×22B | 8 | 2 | 176B | 44B |
| GPT-4 (rumored) | 16 | 2 | ~1.8T | ~220B |
| DeepSeek-V2 | 160 | 6 | 236B | 21B |
| DeepSeek-V3 | 256 | 8 | 671B | 37B |
| Grok-1 | 8 | 2 | 314B | ~86B |

---

## 19.6 Long-Context Techniques

### The attention bottleneck: $O(n^2)$

For $n = 1,000,000$ tokens, the attention matrix has $10^{12}$ entries per head per layer. Even with FlashAttention (no memory issue), the **compute** is enormous.

### Sliding window attention

Mistral's approach: each token only attends to the last $W$ tokens (window size):

$$A_{ij} = 0 \quad \text{if } |i - j| > W$$

**Why it works**: information propagates through the residual stream. After $L$ layers with window $W$, the effective receptive field is $L \times W$. For $L=32, W=4096$: effective context = 131K tokens.

Cost: $O(n \cdot W)$ instead of $O(n^2)$.

### Ring attention

For sequences that truly require full attention over extreme lengths:
- Distribute the sequence across multiple GPUs (each holds a chunk)
- Each GPU computes attention for its Q chunk against all K/V chunks
- K/V chunks are passed in a ring topology between GPUs

This is how models like Gemini achieve 1M+ context.

### RoPE scaling for context extension

Pretrained with context $L$, extend to $L' > L$:

1. **Position interpolation**: scale positions by $L/L'$ so they fit within the training range
2. **NTK-aware scaling**: modify the frequency base $\theta' = \theta \cdot (L'/L)^{d/(d-2)}$
3. **YaRN**: combined NTK + attention scaling + temperature

These allow extending context 4-8× beyond training length with minimal quality loss.

---

## 19.7 Speculative Decoding: Faster Inference

### The problem

Autoregressive generation is slow: each token requires a full forward pass through the model, and tokens are generated sequentially. The model is **memory-bandwidth bound** — the GPU is mostly idle waiting for weights to be loaded from HBM.

### The idea

Use a small "draft" model to propose $k$ tokens, then the large model **verifies** all $k$ tokens in parallel (single forward pass):

```
Draft model (fast): generates [t₁, t₂, t₃, t₄, t₅] quickly
Target model (slow): verifies in one pass
  → accepts [t₁, t₂, t₃], rejects t₄
  → generates correct t₄ from target model
Result: 3 tokens verified + 1 corrected = 4 tokens from 1 target forward pass
```

### Why it preserves quality

The verification step uses the exact target model probabilities. Accepted tokens are **mathematically equivalent** to tokens generated by the target model alone. The output distribution is unchanged — speculative decoding is a speedup technique with **zero quality loss**.

### Speedup

$$\text{Speedup} = \frac{1}{1 - \alpha} \quad \text{(approximate, where } \alpha \text{ = acceptance rate)}$$

Typical acceptance rates: 60-80%. Speedup: 2-5×.

---

## 19.8 State-of-the-Art Models: A Technical Survey

### The frontier models (as of early 2026)

| Model | Org | Params | Architecture | Key innovations |
|---|---|---|---|---|
| GPT-4o | OpenAI | ~200B (est.) | Decoder-only | Multimodal, fast inference |
| Claude 3.5 Sonnet | Anthropic | Unknown | Decoder-only | 200K context, Constitutional AI |
| Gemini 1.5 Pro | Google | Unknown (MoE) | Decoder-only | 1M context, multimodal |
| LLaMA-3.1-405B | Meta | 405B | Dense decoder | Open weights, 128K context |
| DeepSeek-V3 | DeepSeek | 671B (37B active) | MoE decoder | FP8 training, 256 experts |
| Mistral Large | Mistral | Unknown (MoE) | Decoder-only | Sliding window + full attention |
| Qwen-2.5-72B | Alibaba | 72B | Dense decoder | Strong multilingual |

### Open-weight model families

| Family | Size range | License | Key features |
|---|---|---|---|
| LLaMA-3 | 1B - 405B | Custom open | GQA, SwiGLU, RoPE, 128K vocab |
| Mistral/Mixtral | 7B - 8×22B | Apache-2 | Sliding window, GQA, MoE |
| Qwen-2.5 | 0.5B - 72B | Apache-2 | Strong coding, 128K context |
| Gemma-2 | 2B - 27B | Custom open | Knowledge distillation |
| DeepSeek-V3 | 671B | MIT | MoE, multi-head latent attention |
| OLMo-2 | 7B - 13B | Apache-2 | Fully open (data + code + weights) |

### Common architectural choices in SOTA models

| Feature | Consensus |
|---|---|
| Architecture | Decoder-only |
| Normalization | RMSNorm (Pre-Norm) |
| Activation | SwiGLU |
| Positional encoding | RoPE |
| Attention | GQA ($h_{\text{kv}} = 8$) |
| Biases | None |
| Vocab size | 100K-128K (byte-level BPE) |
| Context length | 8K-128K (training), extendable |
| Precision | bf16 training, sometimes FP8 |

There is **remarkable convergence** in architecture. The differentiators are now:
1. **Data quality and curation**
2. **Training scale (compute budget)**
3. **Post-training quality** (alignment, safety)
4. **Inference efficiency** (MoE, quantization, speculative decoding)

---

## 19.9 Emerging Directions

### Test-time compute (inference-time reasoning)

Instead of training a larger model, use the same model but give it more compute at inference time:

- **Chain-of-thought prompting**: let the model "think step by step"
- **Self-consistency**: generate multiple solutions, take majority vote
- **Tree search**: explore multiple reasoning paths, evaluate with a verifier
- **OpenAI o1/o3**: trained to use CoT internally, spending more tokens on harder problems

**Key insight**: a smaller model that "thinks harder" can match a larger model that answers immediately.

### Multi-token prediction

Standard: predict one token at a time.
New approach (Meta, 2024): predict $k$ tokens simultaneously, each with its own head:

$$\mathcal{L}_{\text{multi}} = \sum_{j=1}^{k} \mathcal{L}_{\text{CE}}(\text{head}_j(\mathbf{h}_i), t_{i+j})$$

Benefits:
- Richer training signal per position
- Natural self-speculative decoding (the drafting heads are built-in)
- Improved code generation (where planning ahead matters)

### Native multimodality

Modern frontier models process text, images, audio, and video in a unified architecture:

```
[Image patch embeddings] + [Text token embeddings]
    │                           │
    └────────────┬──────────────┘
                 ▼
        [Shared Transformer]
                 ▼
            [Output]
```

The transformer architecture is **modality-agnostic** — it processes sequences of vectors. Different modalities just need different input encoders.

### State-space models and alternatives

**Mamba** (Gu & Dao, 2023) and similar architectures replace attention with selective state-space models:
- $O(n)$ instead of $O(n^2)$ compute
- No KV cache needed
- Competitive with transformers at small-medium scale

**Current status**: transformers still dominate at frontier scale, but hybrid architectures (Jamba = Mamba + Transformer layers) show promise.

---

## 19.10 Key Takeaways

1. **Scaling laws predict performance**: loss follows power laws in parameters, data, and compute.

2. **Chinchilla showed data matters**: train for ~20 tokens per parameter for compute-optimal; more for inference-optimal.

3. **GQA reduces KV cache** by 4-8× with minimal quality loss — essential for efficient inference.

4. **FlashAttention** is a pure IO optimization that provides 2-4× speedup with exact results.

5. **MoE scales total knowledge without proportional compute cost**: the most effective way to build very large models.

6. **Speculative decoding** accelerates inference 2-5× with zero quality loss.

7. **Architecture has converged**: the differentiators are now data, compute, alignment, and inference efficiency.

8. **Context length is rapidly expanding**: from 2K (GPT-3) to 1M+ (Gemini), enabled by RoPE scaling, ring attention, and FlashAttention.

9. **Test-time compute** is an emerging frontier: letting models "think longer" on hard problems.

10. **The trend**: smaller active parameters, more total experts, longer context, better data, smarter inference.
