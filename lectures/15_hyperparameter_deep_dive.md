# 15. Hyperparameter Deep Dive: Cause and Effect of Every Design Choice

## Motivation

Documents 13-14 established the GPT architecture and its input pipeline. But knowing the architecture is not enough — you must know how to **size** it. How many layers? How wide? How many heads? What context length? These hyperparameters interact in subtle ways, and choosing poorly leads to wasted compute, training instability, or poor model quality.

This document systematically analyzes every major hyperparameter: what it controls, how it affects model quality and compute, what happens when you change it, and what the empirical evidence says about optimal values.

---

## 15.1 The Hyperparameter Taxonomy

### Architectural hyperparameters

| Symbol | Name | Typical range | Controls |
|---|---|---|---|
| $d$ | Model dimension | 512 - 16384 | Width of residual stream, embedding size |
| $N$ | Number of layers | 6 - 128 | Depth of the network |
| $h$ | Number of attention heads | 8 - 128 | Parallel attention patterns |
| $d_k = d/h$ | Head dimension | 64 - 128 | Capacity per attention head |
| $d_{\text{ff}}$ | FFN inner dimension | $4d$ or $\frac{8}{3}d$ | Capacity of each FFN layer |
| $V$ | Vocabulary size | 32K - 256K | Input/output token space |
| $n_{\text{ctx}}$ | Context length | 512 - 128K+ | Maximum input sequence length |

### Training hyperparameters

| Symbol | Name | Typical range | Controls |
|---|---|---|---|
| $B$ | Batch size (tokens) | 256K - 16M | Gradient estimate quality per step |
| $\eta$ | Learning rate | $10^{-5}$ - $10^{-3}$ | Step size for parameter updates |
| $T$ | Training tokens | 100B - 15T | Total data seen during training |
| WD | Weight decay | 0.01 - 0.1 | Regularization strength |
| $\beta_1, \beta_2$ | Adam momentum | 0.9, 0.95-0.999 | Optimizer dynamics |
| $\epsilon$ | Adam epsilon | $10^{-8}$ - $10^{-5}$ | Numerical stability |
| WU | Warmup steps | 200 - 5000 | LR ramp-up period |

---

## 15.2 Model Dimension ($d$): The Width of the Residual Stream

### What $d$ controls

$d$ is the **width** of every hidden representation in the model. Every token is represented as a vector in $\mathbb{R}^d$ throughout all layers. This is the single most important architectural hyperparameter.

### Cause and effect

**Increasing $d$ (wider model)**:

| Effect | Mechanism |
|---|---|
| More parameters: $P \propto d^2$ | Attention weights are $d \times d$, FFN weights are $d \times 4d$ |
| More FLOPS per token: $\propto d^2$ | Matmuls scale quadratically with dimension |
| More memory per token: $\propto d$ | Activations stored for backprop |
| Better representation capacity | Higher-dimensional space can represent more complex features |
| Better attention discrimination | Q, K vectors have more dimensions to form distinct attention patterns |
| Diminishing returns at extreme width | Eventually, depth becomes the bottleneck |

**The relationship between $d$ and $N$**: For a fixed parameter budget $P \approx 12d^2 N$:

$$d = \sqrt{\frac{P}{12N}}$$

There's an optimal $d/N$ ratio. Going too wide with few layers wastes capacity (the model can represent complex features but can't compose them hierarchically). Going too deep with narrow width limits per-layer capacity.

### Empirical findings

From Kaplan et al. (2020) and subsequent scaling laws work:

- The optimal depth-to-width ratio is approximately $N \propto d^{0.6}$
- For a given compute budget, performance is relatively flat across a range of $d/N$ ratios, but extreme ratios (very wide + shallow or very narrow + deep) are suboptimal

### Dimension alignment

In practice, $d$ is chosen to be a multiple of:
- $h$ (number of heads) — so $d_k = d/h$ is an integer
- 128 (for tensor core alignment on modern GPUs)
- Often a power of 2 or a product of small primes

Common values: 768, 1024, 2048, 4096, 8192, 12288, 16384.

---

## 15.3 Number of Layers ($N$): Depth and Compositional Reasoning

### What $N$ controls

$N$ determines how many times the input representations are iteratively refined. Each layer applies attention (information mixing across tokens) followed by FFN (per-token nonlinear transformation).

### Cause and effect

**Increasing $N$ (deeper model)**:

| Effect | Mechanism |
|---|---|
| More parameters: $P \propto N$ | Each layer adds $12d^2$ parameters |
| More FLOPS per token: $\propto N$ | Linear in depth |
| More sequential computation | Cannot parallelize across layers (each depends on the previous) |
| Better compositional reasoning | Information can be combined across more steps |
| Better feature abstraction | Early layers learn low-level patterns, later layers learn high-level |
| Harder to train | Deeper models are more prone to vanishing/exploding gradients |
| Diminishing returns per layer | Later layers contribute less to performance |

### The role of depth in reasoning

Consider a multi-step reasoning task: "Alice is older than Bob. Bob is older than Charlie. Who is youngest?"

- **Layer 1-4**: Parse entities and relationships
- **Layer 5-8**: Connect relationships (transitive reasoning)
- **Layer 9-12**: Produce the answer

Shallow networks cannot compose these steps. There is evidence that **depth is essential for certain reasoning tasks** that require multi-step computation, where width cannot compensate.

### Empirical: layer contributions are NOT uniform

Studies (e.g., by Geva et al., 2022; Elhage et al., 2021) show:

- **Early layers** (1-5): learn basic syntax, word identity, local patterns
- **Middle layers** (5-20): learn semantic relationships, factual recall
- **Late layers** (20+): learn task-specific behavior, prediction refinement
- The **last few layers** often contribute disproportionately to prediction quality
- Some middle layers can be **pruned** with minimal quality loss

### The "deep vs. wide" experiment

For a fixed compute budget, the researcher tests:

| Configuration | $N$ | $d$ | Params | Performance |
|---|---|---|---|---|
| Deep & narrow | 48 | 1024 | ~600M | Good at reasoning, slower per-step |
| Balanced | 24 | 1536 | ~640M | Best overall |
| Wide & shallow | 12 | 2048 | ~600M | Good at simple tasks, weaker reasoning |

The balanced configuration typically wins for general-purpose language modeling.

---

## 15.4 Number of Attention Heads ($h$): Parallel Attention Patterns

### What $h$ controls

$h$ determines how many independent attention patterns are computed in each layer. For $h$ heads with model dimension $d$:
- Each head has $d_k = d/h$ dimensions
- Each head can attend to a different set of positions
- All heads are concatenated and projected back to $d$

### Cause and effect

**Increasing $h$ (more heads)**:

| Effect | Mechanism |
|---|---|
| More diverse attention patterns | Each head can specialize |
| Smaller $d_k$ per head | Less capacity per individual attention pattern |
| No change in total parameters | QKV projections have the same total size |
| Different compute characteristics | More heads = more parallelizable but each head is smaller |

### Head dimension $d_k$: the critical sub-parameter

$$d_k = d / h$$

This is the dimension of each head's Q, K, V vectors. It directly affects:

1. **Attention resolution**: The softmax is over a dot product in $\mathbb{R}^{d_k}$. With larger $d_k$, the dot product has higher variance (roughly $\propto \sqrt{d_k}$), leading to **sharper** attention. With smaller $d_k$, attention becomes more **diffuse**.

2. **Scaling factor**: The $1/\sqrt{d_k}$ scaling in attention normalizes the dot product variance. This is exact when Q, K entries are i.i.d. with zero mean and unit variance.

3. **Capacity**: Each head can represent attention patterns over a $d_k$-dimensional space. Very small $d_k$ (e.g., 16) limits the patterns each head can distinguish.

### The modern consensus: $d_k = 128$

| Model | $d$ | $h$ | $d_k$ |
|---|---|---|---|
| GPT-2 | 768/1024/1280/1600 | 12/16/20/25 | 64 |
| GPT-3 | 12288 | 96 | 128 |
| LLaMA-7B | 4096 | 32 | 128 |
| LLaMA-70B | 8192 | 64 | 128 |
| Mistral-7B | 4096 | 32 | 128 |

**Observation**: $d_k = 128$ has become standard. When scaling the model, increase $h$ (more heads) rather than $d_k$. This adds more parallel attention patterns while keeping each head's capacity constant.

### What heads learn to attend to

Research on attention head interpretability (Voita et al., 2019; Clark et al., 2019):

- **Positional heads**: attend to fixed offsets (e.g., "always attend to the previous token")
- **Syntactic heads**: attend along dependency parse edges (subject→verb, adjective→noun)
- **Rare-word heads**: attend to infrequent tokens that need contextual disambiguation
- **Induction heads** (Olsson et al., 2022): recognize patterns "A B ... A" and predict B — the mechanism behind in-context learning

Not all heads are equally important. Some can be pruned with minimal quality loss.

---

## 15.5 FFN Inner Dimension ($d_{\text{ff}}$): Per-Token Processing Capacity

### What $d_{\text{ff}}$ controls

The FFN sublayer expands the representation to $d_{\text{ff}}$, applies a nonlinearity, and projects back to $d$:

$$\text{FFN}(\mathbf{x}) = \sigma(\mathbf{x}\mathbf{W}_1) \cdot \mathbf{W}_2$$

### The expansion ratio

Traditionally: $d_{\text{ff}} = 4d$

With SwiGLU (3 matrices instead of 2): $d_{\text{ff}} = \frac{8}{3}d \approx 2.67d$ (adjusted to preserve parameter count)

### Cause and effect

**Increasing $d_{\text{ff}}$:**

| Effect | Mechanism |
|---|---|
| More parameters per layer: $\propto d \cdot d_{\text{ff}}$ | Two (or three) large matrices |
| More per-token capacity | Wider bottleneck = more features |
| Better factual recall | FFN layers store "knowledge" as key-value memories (Geva et al., 2021) |
| More FLOPS per token | Dominant compute cost in the model |

### FFN as key-value memory

Research shows each FFN neuron $i$ activates on specific input patterns (keys) and adds specific output patterns (values):

$$\text{FFN}(\mathbf{x}) = \sum_{i=1}^{d_{\text{ff}}} \sigma(\mathbf{w}_i^T \mathbf{x}) \cdot \mathbf{v}_i$$

This is structurally identical to an attention mechanism where "keys" are the rows of $\mathbf{W}_1$ and "values" are the columns of $\mathbf{W}_2$. This interpretation explains why larger $d_{\text{ff}}$ helps factual recall: more neurons = more facts.

---

## 15.6 Context Length ($n_{\text{ctx}}$): How Far the Model Can See

### What $n_{\text{ctx}}$ controls

$n_{\text{ctx}}$ is the maximum number of tokens the model can process in a single forward pass. This determines how much context is available for each prediction.

### Cause and effect

**Increasing $n_{\text{ctx}}$:**

| Effect | Mechanism |
|---|---|
| Attention memory: $O(n_{\text{ctx}}^2)$ | Score matrix is $(n \times n)$ per head per layer |
| Attention compute: $O(n_{\text{ctx}}^2 \cdot d)$ | QK^T matmul and softmax-V matmul |
| KV cache at inference: $O(n_{\text{ctx}} \cdot N \cdot h \cdot d_k)$ | Must store all past K, V for generation |
| Better long-range coherence | Model can condition on more preceding context |
| Better in-context learning | More room for demonstrations in the prompt |
| Harder to train efficiently | Very long sequences → memory bottleneck |

### The quadratic wall

For standard attention, doubling $n_{\text{ctx}}$ quadruples memory and compute for the attention layers:

$$\text{Attention FLOPS per layer} = 2 \cdot h \cdot n^2 \cdot d_k + 2 \cdot h \cdot n^2 \cdot d_k = 4n^2 hd_k = 4n^2d$$

For GPT-3 ($n=2048, d=12288$): attention FLOPS per layer $= 4 \times 2048^2 \times 12288 \approx 2.1 \times 10^{11}$

Compare to FFN FLOPS per layer: $2 \times n \times d \times d_{\text{ff}} + 2 \times n \times d_{\text{ff}} \times d = 4nd \cdot d_{\text{ff}} = 4 \times 2048 \times 12288 \times 49152 \approx 4.9 \times 10^{12}$

At $n = 2048$, FFN dominates. But at $n = 32768$, attention cost exceeds FFN. This is why long-context models require special attention mechanisms (see Doc 19).

### Context length evolution

| Model | Year | $n_{\text{ctx}}$ |
|---|---|---|
| GPT-2 | 2019 | 1,024 |
| GPT-3 | 2020 | 2,048 |
| GPT-3.5-turbo | 2023 | 4,096 / 16,384 |
| Claude 2 | 2023 | 100,000 |
| GPT-4-turbo | 2023 | 128,000 |
| Claude 3.5 | 2024 | 200,000 |
| Gemini 1.5 | 2024 | 1,000,000+ |

---

## 15.7 Batch Size: Gradient Quality vs. Throughput

### What batch size controls

In LLM training, batch size is measured in **tokens** (not sequences):

$$B_{\text{tokens}} = B_{\text{sequences}} \times n_{\text{ctx}}$$

The batch size determines the quality of the gradient estimate:

$$\hat{g} = \frac{1}{B_{\text{tokens}}} \sum_{i=1}^{B_{\text{tokens}}} \nabla_\theta \ell_i(\theta)$$

### Cause and effect

**Larger batch size:**

| Effect | Mechanism |
|---|---|
| Lower gradient noise | More samples → better average |
| Fewer optimizer steps for same data | Each step processes more tokens |
| Better GPU utilization | More parallelism |
| Requires more memory or more GPUs | Linear in batch size |
| Can hurt generalization | Too-clean gradients may miss sharp minima (debated) |
| Allows higher learning rate | Cleaner gradients tolerate larger steps |

### The critical batch size

There exists a **critical batch size** $B_{\text{crit}}$ below which most of the gradient noise is "useful" and above which larger batches give diminishing returns:

$$B_{\text{crit}} \approx \frac{B_{\text{noise}}}{L}$$

where $B_{\text{noise}}$ is a property of the loss landscape and $L$ is the current loss. As training progresses and loss decreases, $B_{\text{crit}}$ increases — you can use larger batches later in training.

### Batch size in practice

| Model | Batch size (tokens) | Batch size (sequences) |
|---|---|---|
| GPT-2 | 512K | 512 × 1024 |
| GPT-3 | 3.2M (ramped from 32K) | 1536 × 2048 |
| LLaMA-1 | 4M | 1024 × 4096 |
| LLaMA-3-405B | 16M | ~4000 × 4096 |

**Batch size warmup**: Start with a small batch size and increase during training. This gives noisier (more exploratory) gradients early on, and cleaner (more precise) gradients later.

---

## 15.8 Learning Rate: The Most Sensitive Hyperparameter

### The learning rate schedule

Modern LLM training uses a learning rate schedule with three phases:

```
LR
 ↑
η_max │         ╭──────────────╮
      │        ╱                ╲
      │       ╱                  ╲
      │      ╱                    ╲
η_min │─────╱                      ╲───────────
      │
      └──────────────────────────────────────→ Steps
      │warmup│      cosine decay        │cooldown
```

$$\eta(t) = \begin{cases} \eta_{\max} \cdot \frac{t}{T_{\text{warmup}}} & t < T_{\text{warmup}} \\ \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})(1 + \cos(\pi \cdot \frac{t - T_{\text{warmup}}}{T_{\text{total}} - T_{\text{warmup}}})) & t \geq T_{\text{warmup}} \end{cases}$$

### Cause and effect

**Learning rate too high:**
- Loss spikes or diverges during training
- Parameter updates overshoot optimal values
- Attention entropy collapses (all heads produce near-uniform distributions)
- Gradient norm explodes

**Learning rate too low:**
- Training progresses very slowly
- May not converge within compute budget
- Underfitting

**Optimal $\eta$ depends on**:
1. Model size (larger models need smaller $\eta$)
2. Batch size (larger batches allow larger $\eta$)
3. Training phase (warmup → peak → decay)

### The $\eta \propto 1/\sqrt{d}$ rule

A common heuristic: peak learning rate scales inversely with the square root of model dimension. This comes from the observation that larger models have larger activation norms, so each gradient step has a proportionally larger effect on the output.

| Model size | Typical peak $\eta$ |
|---|---|
| 125M | $6 \times 10^{-4}$ |
| 1.3B | $2 \times 10^{-4}$ |
| 13B | $1 \times 10^{-4}$ |
| 70B | $1.5 \times 10^{-5}$ |
| 175B | $6 \times 10^{-5}$ |

### Warmup: why it's necessary

During the first few hundred steps:
- Embeddings are random → attention patterns are noise
- Gradients have very high variance
- Large learning rates would cause irreversible loss spikes

**Warmup** linearly increases $\eta$ from ~0 to $\eta_{\max}$ over $T_{\text{warmup}}$ steps. This gives the model time to establish stable representations before taking large parameter steps.

Typical warmup: 0.1-1% of total training steps.

---

## 15.9 Weight Decay: Regularization at Scale

### What weight decay does

Weight decay adds a penalty proportional to the squared magnitude of weights:

$$\theta_{t+1} = \theta_t - \eta \left(\hat{g}_t + \lambda \theta_t\right)$$

This is equivalent to multiplying weights by $(1 - \eta\lambda)$ each step, shrinking them toward zero.

### Decoupled weight decay (AdamW)

In AdamW (Loshchilov & Hutter, 2019), weight decay is applied **separately** from the gradient update:

$$\theta_{t+1} = (1 - \eta \lambda) \theta_t - \eta \cdot \text{Adam}(\hat{g}_t)$$

This is crucial because standard L2 regularization interacts poorly with Adam's adaptive learning rates. AdamW decouples them.

### Which parameters get weight decay?

| Parameter | Weight decay? | Why |
|---|---|---|
| Linear layer weights | Yes | Main regularization target |
| Embedding weights | Sometimes | Often excluded or given different WD |
| Bias terms | No | Low-dimensional, don't benefit |
| LayerNorm/RMSNorm $\gamma, \beta$ | No | Normalization parameters, shouldn't be shrunk |

### Effect of weight decay value

| $\lambda$ | Effect |
|---|---|
| 0 | No regularization. Model may memorize training data. |
| 0.01 | Light regularization. Standard for LLMs. |
| 0.1 | Strong regularization. Common in fine-tuning. |
| 1.0 | Very strong. Typically too aggressive for pretraining. |

Standard value: $\lambda = 0.1$ for most modern LLMs.

---

## 15.10 Dropout: Mostly Absent in Modern LLMs

### The surprising finding

GPT-2 used dropout ($p = 0.1$) during pretraining. GPT-3 and most modern models use **no dropout at all** for pretraining.

### Why no dropout?

1. **Data is not repeated**: with trillions of tokens, the model sees each example once or very few times. Overfitting to individual examples is not a concern.

2. **Weight decay is sufficient**: AdamW weight decay provides enough regularization.

3. **Dropout hurts throughput**: randomly zeroing activations wastes compute that could be used for learning.

4. **Inconsistency at scale**: dropout during training but not during inference creates a distributional shift. At very large scale, even small shifts matter.

**Exception**: Dropout is still useful for **fine-tuning**, where the dataset is small and overfitting is a real risk.

---

## 15.11 Adam Parameters: $\beta_1, \beta_2, \epsilon$

### The Adam update rule

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t \quad \text{(first moment / momentum)}$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \quad \text{(second moment / adaptive LR)}$$
$$\hat{m}_t = m_t / (1 - \beta_1^t), \quad \hat{v}_t = v_t / (1 - \beta_2^t) \quad \text{(bias correction)}$$
$$\theta_{t+1} = \theta_t - \eta \cdot \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)$$

### Cause and effect of each parameter

**$\beta_1$ (momentum decay):**
- Default: 0.9, sometimes 0.9-0.95
- Higher $\beta_1$: smoother updates, more momentum → good for noisy gradients, but slower to react to changes
- Lower $\beta_1$: more responsive to current gradient → can be more unstable

**$\beta_2$ (second moment decay):**
- Default: 0.95-0.999
- Higher $\beta_2$: slower adaptation of learning rate per parameter → more stable but slower to adapt
- Lower $\beta_2$: faster adaptation → can cause training instability
- GPT-3 uses $\beta_2 = 0.95$ (unusually low) — this makes the optimizer more aggressive about adjusting per-parameter learning rates
- LLaMA uses $\beta_2 = 0.95$
- Many models use 0.999 or 0.998

**$\epsilon$ (numerical stability):**
- Default: $10^{-8}$
- Rarely tuned, but can affect training in mixed precision
- Some implementations use $10^{-5}$ for bfloat16 stability

---

## 15.12 Gradient Clipping: Preventing Catastrophic Updates

### What it does

Before applying the optimizer step, clip the global gradient norm:

$$\text{if } \|\nabla \theta\|_2 > G_{\max}: \quad \nabla \theta \leftarrow \frac{G_{\max}}{\|\nabla \theta\|_2} \cdot \nabla \theta$$

### Why it's essential

LLM training occasionally produces **loss spikes** — sudden increases in loss caused by high-variance gradient batches. Without clipping, a single bad batch can destabilize the entire model.

### Typical value

$G_{\max} = 1.0$ is near-universal across GPT-2, GPT-3, LLaMA, Mistral, etc.

### Effect of the clipping threshold

| $G_{\max}$ | Effect |
|---|---|
| 0.1 | Very conservative. Slows training. |
| 1.0 | Standard. Clips ~5-20% of steps in practice. |
| 10.0 | Permissive. Almost never triggers. |
| $\infty$ | No clipping. Risk of divergence. |

---

## 15.13 Initialization: Preventing Day-One Divergence

### Why initialization matters

At step 0, before any training, the model must produce reasonable outputs and gradients. Bad initialization → immediate divergence or extremely slow convergence.

### The standard initialization

**Linear layer weights**: $\mathbf{W} \sim \mathcal{N}(0, \sigma^2)$ with $\sigma = 0.02$ (GPT-2) or $\sigma = 1/\sqrt{d}$ (Xavier-like)

**Output projection scaling**: The output projection of attention and the second FFN layer are sometimes initialized with $\sigma = \frac{0.02}{\sqrt{2N}}$ where $N$ is the number of layers.

**Why $1/\sqrt{2N}$?** Each layer adds to the residual stream. After $N$ layers with 2 residual additions each (attention + FFN), the variance of the residual stream grows. Scaling the initialization by $1/\sqrt{2N}$ keeps the total contribution bounded.

### The $\mu$P (maximal update parametrization) approach

Developed by Yang et al. (2022). The key idea: choose initialization and learning rate per-layer so that the optimal hyperparameters transfer across model sizes.

In standard parametrization (SP):
- Optimal $\eta$ changes as you scale the model
- You must re-tune hyperparameters at every scale

In $\mu$P:
- Initialization scales as $\sigma \propto 1/d$
- Learning rates scale per-layer
- Hyperparameters found at small scale **transfer** to large scale

This is extremely valuable: tune hyperparameters on a 10M-parameter model, then apply them to a 10B-parameter model with confidence.

---

## 15.14 Numerical Precision: fp32, fp16, bf16, fp8

### The precision trade-off

| Format | Bits | Exponent | Mantissa | Range | Use |
|---|---|---|---|---|---|
| fp32 | 32 | 8 | 23 | $\pm 3.4 \times 10^{38}$ | Master weights, optimizer states |
| fp16 | 16 | 5 | 10 | $\pm 65504$ | Legacy mixed precision |
| bf16 | 16 | 8 | 7 | $\pm 3.4 \times 10^{38}$ | **Modern standard** for training |
| fp8 (E4M3) | 8 | 4 | 3 | $\pm 448$ | Emerging for forward pass |

### Why bf16 dominates modern training

- **Same range as fp32** (8-bit exponent) → no overflow issues
- **Lower precision** (7-bit mantissa vs. 23) → acceptable for gradient computation
- **2× throughput** vs. fp32 on modern GPUs (A100, H100)
- **Simpler than fp16** mixed precision: fp16's narrow range requires careful loss scaling

### Mixed precision training

Standard recipe:
1. Store **master weights** in fp32 (for accumulating small gradient updates)
2. Cast to bf16 for forward and backward passes (fast matmuls)
3. Compute gradients in bf16
4. Update master weights in fp32

```python
# Conceptual mixed-precision training loop
for batch in dataloader:
    # Forward in bf16
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        logits = model(batch.input_ids)
        loss = cross_entropy(logits, batch.labels)

    # Backward in bf16
    loss.backward()

    # Update in fp32 (optimizer holds fp32 master weights)
    optimizer.step()
    optimizer.zero_grad()
```

### Memory impact of precision

For a model with $P$ parameters:

| Component | fp32 | bf16 mixed |
|---|---|---|
| Model weights | $4P$ bytes | $2P$ (bf16) + $4P$ (master) = $6P$ bytes |
| Adam $m, v$ states | $8P$ bytes | $8P$ bytes (always fp32) |
| Gradients | $4P$ bytes | $2P$ bytes |
| **Total** | **$16P$ bytes** | **$16P$ bytes** |

Mixed precision doesn't save optimizer memory — the savings come from activation memory and faster matmuls.

---

## 15.15 Hyperparameter Interactions: The Entangled Web

### The key interactions

```
Batch size ←→ Learning rate
  ↕                 ↕
Model size ←→ Training tokens
  ↕                 ↕
Width (d) ←→ Depth (N)
```

**Batch size × Learning rate**: The "linear scaling rule" — if you double batch size, you can double learning rate (approximately). More precisely, the optimal learning rate scales as $\eta \propto \sqrt{B}$.

**Model size × Training tokens**: Scaling laws (Chinchilla) dictate the optimal ratio. For compute-optimal training, double the model → double the data. See Doc 19.

**Width × Depth**: For a fixed parameter budget, there's an optimal ratio. Too wide → poor at reasoning. Too deep → training instability, diminishing returns per layer.

**Context length × Batch size**: Doubling context length halves the number of sequences per batch (for fixed batch size in tokens), changing the diversity of each gradient estimate.

---

## 15.16 A Recipe for Choosing Hyperparameters

### Step-by-step decision process

1. **Determine your compute budget** (GPU-hours or FLOPS)

2. **Use scaling laws to choose model size and training tokens**:
   $$C \approx 6PT \quad \text{(where } P = \text{params}, T = \text{tokens)}$$
   Chinchilla-optimal: $P$ and $T$ should be scaled roughly equally (see Doc 19)

3. **Choose $d$ and $N$ for the target parameter count**:
   $$P = 12d^2N + 2Vd$$
   Use $d_k = 128$, $h = d/d_k$, $d_{\text{ff}} = \frac{8}{3}d$ (with SwiGLU)

4. **Set learning rate** based on model size (see table in 15.8)

5. **Set batch size**: start from scaling law recommendations, then adjust:
   - Start with ~0.5-1M tokens per batch
   - Ramp up to 4-16M during training

6. **Set warmup** to 0.1-1% of total steps

7. **Use defaults for everything else**:
   - Weight decay: 0.1
   - $\beta_1 = 0.9, \beta_2 = 0.95$
   - Gradient clip: 1.0
   - No dropout for pretraining
   - bf16 mixed precision
   - Cosine schedule to $\eta_{\min} = 0.1 \cdot \eta_{\max}$

8. **Run small-scale experiments** ($\mu$P if possible) to validate choices before committing to full training

---

## 15.17 Key Takeaways

1. **$d$ and $N$ dominate**: model quality primarily depends on these two, and their ratio matters.

2. **$d_k = 128$ is standard**: scale by adding heads, not by increasing head dimension.

3. **Learning rate is the most sensitive**: too high → divergence, too low → wasted compute. Use warmup + cosine decay.

4. **Batch size should be warmed up**: start small, increase during training.

5. **Weight decay replaces dropout** for pretraining at scale.

6. **bf16 is the standard precision**: same dynamic range as fp32, 2× throughput.

7. **Hyperparameters interact**: changing one (e.g., batch size) requires adjusting others (e.g., learning rate). Use scaling laws and $\mu$P to navigate this space efficiently.

8. **The optimal configuration depends on your compute budget**: there is no single "best" architecture — only the best architecture for your budget.
