# 16. Pretraining from Scratch: Data, Optimization, and Infrastructure

## Motivation

Documents 13-15 defined the architecture and hyperparameters. Now we train it. Pretraining is where the model acquires its knowledge, language ability, and reasoning capacity. It is also where most of the compute budget is spent — a single GPT-3 training run costs millions of dollars.

This document covers the complete pretraining pipeline: data collection and cleaning, the training objective, optimizer mechanics, distributed training strategies, training dynamics (loss curves, instabilities, emergent behaviors), and the infrastructure required.

---

## 16.1 The Data Pipeline: Garbage In, Garbage Out

### Why data quality matters more than architecture

A well-known finding from the scaling laws literature: **data quality has a larger impact on model performance than architectural choices** at matched compute. A clean 1T-token dataset outperforms a noisy 10T-token dataset.

### The data pipeline

```
Web crawl (petabytes)
    │
    ▼
[URL filtering]  — remove adult, spam, malware sites
    │
    ▼
[HTML extraction]  — extract text from HTML, remove boilerplate
    │
    ▼
[Language filtering]  — keep target languages
    │
    ▼
[Quality filtering]  — perplexity filter, heuristic rules
    │
    ▼
[Deduplication]  — exact and near-duplicate removal
    │
    ▼
[PII removal]  — remove emails, phone numbers, SSNs
    │
    ▼
[Toxic content filtering]  — classifier-based removal
    │
    ▼
[Data mixing]  — blend web, books, code, academic papers
    │
    ▼
Clean training corpus (trillions of tokens)
```

### Data sources and their properties

| Source | Tokens (typical) | Quality | Diversity | Examples |
|---|---|---|---|---|
| Web crawl (Common Crawl) | 1-10T | Low-Medium | Very high | CC, C4, RefinedWeb |
| Books | 10-100B | High | Medium | BookCorpus, Gutenberg, Libgen |
| Wikipedia | 3-10B | Very high | Medium | All languages |
| Academic papers | 50-200B | High | Specialized | arXiv, Semantic Scholar |
| Code | 200B-1T | High (structured) | Medium | GitHub, StackOverflow |
| Curated web | 100B-1T | High | High | Filtered CC subsets |
| Conversational | 10-100B | Medium | High | Reddit, forums |

### Deduplication: why it's critical

**Exact deduplication**: Remove documents with identical content (after normalization). Use hashing.

**Near-deduplication** (MinHash/LSH): Find and remove documents that share >80% of their n-grams. This removes:
- Boilerplate text (terms of service, cookie notices)
- Syndicated content (same article on multiple sites)
- Templates (auto-generated pages)

**Effect of deduplication**:
- Without dedup: model memorizes repeated content, wasting capacity
- Lee et al. (2022): Deduplication improves perplexity **and** reduces memorization
- Typical compression: 3-10× fewer tokens after aggressive dedup

### Quality filtering

**Perplexity-based filtering** (used by CCNet, LLaMA):
1. Train a small language model on high-quality text (Wikipedia)
2. Score each document by perplexity
3. Keep low-perplexity documents (they "look like" Wikipedia in style)

**Heuristic filters** (used by C4, RefinedWeb, Dolma):
- Minimum/maximum document length
- Remove documents with too many repeated characters/words
- Remove documents with excessive special characters
- Require complete sentences (ending in punctuation)
- Remove documents with too high a fraction of uppercase text

### Data mixing ratios

The proportion of each data source matters enormously:

| Model | Web | Books | Wikipedia | Code | Academic | Other |
|---|---|---|---|---|---|---|
| GPT-3 | 60% | 16% | 3% | — | — | 21% |
| LLaMA-1 | 67% | 4.5% | 4.5% | 4.5% | — | 19.5% |
| LLaMA-3 | ~78% | ~5% | ~2% | ~10% | — | ~5% |
| Dolma (OLMo) | 77% | 6% | 4% | 10% | 3% | — |

**Key observations**:
- Web always dominates (diversity, volume)
- Code improves reasoning abilities even for non-code tasks
- Books improve long-form coherence
- Wikipedia improves factual accuracy
- Over-representing any one source causes distribution shift

---

## 16.2 The Training Objective: Autoregressive Cross-Entropy

### The loss function (review from Doc 13)

$$\mathcal{L} = -\frac{1}{n} \sum_{i=1}^n \log P_\theta(t_i | t_1, \ldots, t_{i-1})$$

This is the **average negative log-likelihood** per token. It measures how well the model predicts each token given its context.

### Perplexity: the standard evaluation metric

$$\text{PPL} = \exp(\mathcal{L}) = \exp\left(-\frac{1}{n}\sum_{i=1}^n \log P_\theta(t_i | t_{<i})\right)$$

**Interpretation**: If PPL = 50, the model is as uncertain as if it were choosing uniformly among 50 options at each position. Lower is better.

**Typical values**:
- Random (uniform over $V=50K$): PPL ≈ 50,000
- After 1 epoch on Wikipedia: PPL ≈ 30-50
- GPT-2 (1.5B) on test set: PPL ≈ 20
- GPT-3 (175B) on test set: PPL ≈ 9-15
- State-of-the-art (2024): PPL ≈ 5-8 on standard benchmarks

### The loss landscape in bits

Converting to bits-per-byte (BPB) or bits-per-character (BPC) allows comparison across tokenizers:

$$\text{BPB} = \frac{\mathcal{L}}{\ln(2)} \cdot \frac{\text{tokens}}{\text{bytes}}$$

The information-theoretic limit for English text is estimated at ~0.8-1.5 BPB.

---

## 16.3 The Optimizer: AdamW in Detail

### Why Adam?

SGD with momentum works well for vision models but is suboptimal for transformers. Reasons:

1. **Sparse gradients**: embedding layers have very sparse gradients. SGD assigns the same learning rate to all parameters. Adam adapts per-parameter.

2. **Heterogeneous curvature**: different parameters (attention weights, FFN weights, embeddings) have vastly different optimal learning rates. Adam's second moment ($v_t$) handles this automatically.

3. **Memory of gradient direction**: Adam's first moment ($m_t$) provides momentum even for rarely-updated parameters.

### The full AdamW algorithm

```
Input: parameters θ, learning rate η, (β₁, β₂), weight decay λ, ε
Initialize: m₀ = 0, v₀ = 0, t = 0

For each step:
    t ← t + 1
    g_t ← ∇L(θ_{t-1})                          # Compute gradients
    m_t ← β₁ · m_{t-1} + (1 - β₁) · g_t       # Update first moment
    v_t ← β₂ · v_{t-1} + (1 - β₂) · g_t²      # Update second moment
    m̂_t ← m_t / (1 - β₁ᵗ)                      # Bias correction
    v̂_t ← v_t / (1 - β₂ᵗ)                      # Bias correction
    θ_t ← θ_{t-1} - η · (m̂_t / (√v̂_t + ε) + λ · θ_{t-1})  # Update + WD
```

### Memory cost of AdamW

Per-parameter, AdamW stores:
- Master weights: 4 bytes (fp32)
- First moment $m_t$: 4 bytes (fp32)
- Second moment $v_t$: 4 bytes (fp32)
- Gradients: 2 bytes (bf16)

**Total**: ~14 bytes per parameter. For a 70B model: $70 \times 10^9 \times 14 \approx 980$ GB just for optimizer state + weights.

This is why distributed training is not optional — no single GPU has enough memory.

---

## 16.4 Distributed Training: Parallelism Strategies

### The memory problem

Consider training a 70B-parameter model:

| Component | Size |
|---|---|
| Model weights (bf16) | 140 GB |
| Optimizer states (fp32) | 560 GB |
| Gradients (bf16) | 140 GB |
| Activations (bf16) | 100-500 GB (depends on batch size, seq len) |
| **Total** | **940+ GB** |

A single H100 GPU has 80 GB memory. We need at least 12 GPUs just for model state, plus more for activations.

### Strategy 1: Data Parallelism (DP)

```
GPU 0: Full model copy + Batch slice 0 → gradients → All-Reduce → Update
GPU 1: Full model copy + Batch slice 1 → gradients → All-Reduce → Update
GPU 2: Full model copy + Batch slice 2 → gradients → All-Reduce → Update
...
```

Each GPU holds a **complete copy** of the model and processes a different data slice. Gradients are averaged across GPUs (All-Reduce), then each GPU updates its copy identically.

**Pro**: Simple, near-linear scaling in throughput
**Con**: Every GPU must fit the entire model — impossible for large models

### Strategy 2: ZeRO (Zero Redundancy Optimizer)

ZeRO (Rajbhandari et al., 2019) partitions the redundant state across GPUs:

| ZeRO Stage | Partitions | Memory per GPU |
|---|---|---|
| Stage 1 | Optimizer states | $\frac{O}{N_{\text{GPU}}} + M + G$ |
| Stage 2 | + Gradients | $\frac{O + G}{N_{\text{GPU}}} + M$ |
| Stage 3 | + Model weights | $\frac{O + G + M}{N_{\text{GPU}}}$ |

where $O$ = optimizer states, $G$ = gradients, $M$ = model weights.

**ZeRO-3** gives the same memory savings as model parallelism but with the simplicity of data parallelism.

### Strategy 3: Tensor Parallelism (TP)

Split individual operations across GPUs:

```
GPU 0: W_Q[:, :d/2]      GPU 1: W_Q[:, d/2:]
GPU 0: W_K[:, :d/2]      GPU 1: W_K[:, d/2:]
...
→ Each GPU computes attention on half the heads
→ All-Reduce after output projection
```

Each matmul is split across GPUs. Requires high-bandwidth interconnect (NVLink at 900 GB/s, not PCIe at 64 GB/s).

**Pro**: Reduces memory per GPU, efficient for large layers
**Con**: Requires communication at every layer (high bandwidth needed)

### Strategy 4: Pipeline Parallelism (PP)

Split the model's **layers** across GPUs:

```
GPU 0: Layers 0-15
GPU 1: Layers 16-31
GPU 2: Layers 32-47
GPU 3: Layers 48-63
```

Micro-batches flow through the pipeline:

```
Time →
GPU 0: [MB1][MB2][MB3][MB4]
GPU 1:      [MB1][MB2][MB3][MB4]
GPU 2:           [MB1][MB2][MB3][MB4]
GPU 3:                [MB1][MB2][MB3][MB4]
```

**Pro**: Minimal communication (only activations between stages)
**Con**: Pipeline "bubbles" waste GPU time; requires careful scheduling

### Strategy 5: Sequence Parallelism (SP)

For very long sequences, split the sequence across GPUs:

```
GPU 0: tokens [0:n/2]    — all layers
GPU 1: tokens [n/2:n]    — all layers
→ Communicate at attention (each GPU needs all positions for Q·K^T)
```

Used together with TP for long-context models.

### Real-world combinations

| Model | GPUs | DP | TP | PP |
|---|---|---|---|---|
| GPT-3 (175B) | ~1024 V100s | 64 | 8 | 2 |
| LLaMA-65B | 2048 A100s | 256 | 8 | 1 |
| LLaMA-3-405B | 16,384 H100s | 1024 | 8 | 2 |

These combine: $N_{\text{total}} = N_{\text{DP}} \times N_{\text{TP}} \times N_{\text{PP}}$

---

## 16.5 Training Dynamics: What Happens During Pretraining

### The loss curve

A typical pretraining loss curve:

```
Loss
  ↑
  │╲
  │ ╲
  │  ╲
  │   ╲
  │    ╲╲
  │      ╲╲╲
  │         ╲╲╲╲╲╲
  │                ╲╲╲╲╲╲╲╲╲╲
  │                           ╲╲╲╲──────
  └────────────────────────────────────→ Tokens
  0       100B     500B     1T     2T
```

**Phase 1** (0-10B tokens): Rapid loss decrease. Model learns basic statistics (token frequencies, common bigrams/trigrams).

**Phase 2** (10-100B tokens): Steady improvement. Model learns grammar, syntax, simple factual associations.

**Phase 3** (100B-1T tokens): Slower improvement. Model learns complex reasoning, rare facts, long-range dependencies.

**Phase 4** (1T+ tokens): Diminishing returns. Each token contributes less. This is where scaling laws determine when to stop.

### Loss spikes and instabilities

**Loss spikes**: sudden increases in training loss, followed by recovery.

```
Loss
  ↑
  │         ╱╲
  │        ╱  ╲
  │╲      ╱    ╲
  │ ╲╲╲╲╲╱      ╲╲╲╲╲╲╲──
  └───────────────────────→ Steps
```

**Common causes**:
1. **Data artifacts**: a batch with unusually different distribution (e.g., all code)
2. **Numerical instability**: overflow in attention logits or FFN activations
3. **Loss of representational rank**: attention heads collapse to similar patterns
4. **Hardware failures**: silent data corruption (bit flips) on GPUs

**Mitigation strategies**:
1. Gradient clipping (see Doc 15)
2. Learning rate reduction after spike
3. Skip bad batches
4. Z-loss regularization (penalize large logits)
5. QK-norm: normalize Q and K before computing attention scores
6. Checkpointing: save model frequently, rewind if needed

### Z-loss: preventing logit explosion

PaLM (Chowdhery et al., 2022) introduced the Z-loss:

$$\mathcal{L}_z = \alpha \cdot \log^2\left(\sum_{v=1}^V \exp(z_v)\right)$$

where $z_v$ are the logits and $\alpha \approx 10^{-4}$.

This penalizes large logits, preventing the softmax partition function from growing unbounded. Large logits cause numerical instability in bf16.

---

## 16.6 Activation Checkpointing (Gradient Checkpointing)

### The memory problem for activations

During backprop, you need the activations from the forward pass to compute gradients. For a model with $N$ layers, sequence length $n$, and batch size $B$:

$$\text{Activation memory} = O(N \cdot B \cdot n \cdot d)$$

For large models, this exceeds GPU memory.

### The trade-off: memory vs. compute

**Full activation caching**: Store all activations. Memory: $O(N \cdot B \cdot n \cdot d)$. Compute: $1\times$ forward.

**Full recomputation**: Store nothing. During backward, recompute each layer's activations from the input. Memory: $O(B \cdot n \cdot d)$ (just current layer). Compute: $2\times$ forward (one extra forward pass).

**Selective checkpointing** (the practical approach): Store activations at every $k$-th layer. During backward, recompute the intermediate layers from the checkpoint. Memory: $O(N/k \cdot B \cdot n \cdot d)$. Compute: $1 + 1/k$ times forward.

With $k = \sqrt{N}$: memory is $O(\sqrt{N} \cdot B \cdot n \cdot d)$ and compute overhead is $\sim 33\%$.

### What to checkpoint

**Cheap to store, expensive to compute** → store it (attention scores after softmax)

**Expensive to store, cheap to compute** → recompute it (RMSNorm, GELU/SiLU activations)

Common practice: checkpoint at layer boundaries (store the input to each transformer block).

---

## 16.7 The Complete Training Loop

### Pseudocode

```python
def pretrain(model, dataset, config):
    optimizer = AdamW(
        model.parameters(),
        lr=0,  # will be set by scheduler
        betas=(config.beta1, config.beta2),
        weight_decay=config.weight_decay,
        eps=config.eps
    )
    scheduler = CosineWithWarmup(
        optimizer,
        warmup_steps=config.warmup_steps,
        total_steps=config.total_steps,
        eta_min=config.lr_min
    )

    step = 0
    for epoch in range(config.num_epochs):
        for batch in dataset.get_batches(config.batch_size, config.seq_len):
            # batch.input_ids: (B, n) — token IDs
            # batch.labels:    (B, n) — shifted by 1 (labels[i] = input_ids[i+1])

            # Forward pass (bf16)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = model(batch.input_ids)  # (B, n, V)
                loss = F.cross_entropy(
                    logits[:, :-1, :].reshape(-1, config.vocab_size),
                    batch.input_ids[:, 1:].reshape(-1),
                    ignore_index=config.pad_token_id
                )

            # Backward pass
            loss.backward()

            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.max_grad_norm
            )

            # Optimizer step
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            # Logging
            step += 1
            if step % config.log_interval == 0:
                log(step=step, loss=loss.item(), lr=scheduler.get_lr(),
                    grad_norm=grad_norm, tokens_seen=step * config.batch_tokens)

            # Checkpointing
            if step % config.save_interval == 0:
                save_checkpoint(model, optimizer, scheduler, step)

            # Evaluation
            if step % config.eval_interval == 0:
                eval_loss = evaluate(model, eval_dataset)
                log(eval_loss=eval_loss, eval_ppl=math.exp(eval_loss))
```

### The label shift

A subtle but critical detail: the model predicts the **next** token. So:
- Input: `[t_0, t_1, t_2, ..., t_{n-1}]`
- Labels: `[t_1, t_2, t_3, ..., t_n]`

The loss at position $i$ compares the model's prediction at position $i$ to the actual token at position $i+1$:

```python
# Standard approach: slice logits and labels
loss = F.cross_entropy(
    logits[:, :-1, :].contiguous().view(-1, vocab_size),  # predictions at pos 0..n-2
    input_ids[:, 1:].contiguous().view(-1)                 # labels at pos 1..n-1
)
```

---

## 16.8 Gradient Accumulation: Simulating Large Batches

### When batches don't fit in memory

If the desired batch size is 4M tokens but GPU memory only supports 256K tokens per step:

$$\text{accumulation steps} = \frac{4M}{256K} = 16$$

Accumulate gradients over 16 micro-batches before taking an optimizer step:

```python
accumulation_steps = config.batch_tokens // (micro_batch_size * seq_len * num_gpus)

optimizer.zero_grad()
for micro_step in range(accumulation_steps):
    batch = next(data_iter)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        logits = model(batch.input_ids)
        loss = F.cross_entropy(...) / accumulation_steps  # Scale loss

    loss.backward()  # Gradients accumulate in .grad

# After all micro-batches
clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
optimizer.zero_grad()
```

**Key**: divide the loss by `accumulation_steps` so the accumulated gradient has the correct scale (average over the full batch, not sum).

---

## 16.9 Data Loading and Packing

### Naive approach (padding)

Sequences have different lengths. Naive batching pads shorter sequences to the batch's maximum length:

```
Sequence 1: [tok tok tok tok tok PAD PAD PAD]
Sequence 2: [tok tok tok tok tok tok tok tok]
Sequence 3: [tok tok tok PAD PAD PAD PAD PAD]
```

**Problem**: PAD tokens waste compute. With 50% padding, half the FLOPS are wasted.

### Packing: concatenate documents

Instead, concatenate documents into fixed-length chunks:

```
[DOC1 tok tok tok <eos> DOC2 tok tok tok tok <eos> DOC3 tok tok]
```

Each training sequence is exactly $n_{\text{ctx}}$ tokens, with no padding. An `<eos>` token separates documents.

**Critical**: The causal mask must prevent attention across document boundaries. Some implementations add a **document mask** that prevents tokens after `<eos>` from attending to tokens before `<eos>` in a different document.

### Packing in practice

```python
def pack_documents(documents, seq_len, eos_token_id):
    """Pack documents into fixed-length sequences."""
    buffer = []
    sequences = []

    for doc_tokens in documents:
        buffer.extend(doc_tokens)
        buffer.append(eos_token_id)

        while len(buffer) >= seq_len:
            sequences.append(buffer[:seq_len])
            buffer = buffer[seq_len:]

    return sequences  # Each is exactly seq_len tokens
```

---

## 16.10 How Long to Train: The Data Repetition Question

### The epoch question

Classical ML trains for multiple epochs (passes through the data). For LLMs:

- **Training on unique data is always better** (Muennighoff et al., 2023)
- Repeating data leads to diminishing returns and eventually **degradation**
- The "value of a token" decreases sharply after 4-8 repetitions
- Most frontier models train for <2 epochs over their data

### When you run out of data

If your high-quality data has $D$ tokens and optimal training requires $T > D$ tokens:

Options (in order of preference):
1. Get more data (more web crawling, multilingual data, synthetic data)
2. Accept sub-optimal training (use $T = D$ tokens)
3. Repeat data carefully (up to 4×, with quality-based weighting)
4. Use a smaller model (reduce $P$ until $T = D$ is compute-optimal)

---

## 16.11 Evaluating During Pretraining

### Training metrics

| Metric | What it tells you | How to compute |
|---|---|---|
| Training loss | How well the model fits training data | Average cross-entropy |
| Validation loss | How well the model generalizes | Cross-entropy on held-out data |
| Gradient norm | Training stability | L2 norm of gradients |
| Learning rate | Current schedule position | From scheduler |
| Tokens/second | Hardware utilization | Wall clock timing |
| MFU (Model FLOPS Utilization) | How efficiently you use the GPU | Actual / theoretical peak FLOPS |

### Benchmark evaluation during training

Run periodic evaluations on downstream benchmarks to track capability emergence:

| Category | Benchmarks | What they test |
|---|---|---|
| Language modeling | WikiText, C4 val, The Pile | Next-token prediction quality |
| Knowledge/QA | TriviaQA, NaturalQuestions | Factual recall |
| Reasoning | HellaSwag, ARC, PIQA | Common-sense reasoning |
| Math | GSM8K, MATH | Mathematical reasoning |
| Code | HumanEval, MBPP | Code generation |
| MMLU | 57 subjects | Broad knowledge |

### MFU: the efficiency metric

$$\text{MFU} = \frac{\text{Observed FLOPS}}{\text{Theoretical peak FLOPS}} \times 100\%$$

For a transformer with parameters $P$ processing $B$ tokens per second:

$$\text{Observed FLOPS} \approx 6P \times B_{\text{tokens/sec}}$$

(Factor 6: 2 for forward, 4 for backward per parameter)

Typical MFU values:
- Naive implementation: 20-30%
- Good implementation: 40-50%
- Excellent (FlashAttention, fused kernels): 50-60%
- State-of-the-art: 55-65%

---

## 16.12 Checkpointing and Fault Tolerance

### Why fault tolerance matters

Training a 70B model on 2048 GPUs for 3 months:
- Probability of GPU failure per day per GPU: ~0.1%
- Expected GPU failures during training: $2048 \times 90 \times 0.001 \approx 184$ failures

Without fault tolerance, training would never complete.

### Checkpointing strategy

1. **Save every $N$ steps** (typically every 500-2000 steps)
2. **Async checkpointing**: write checkpoints to storage without pausing training
3. **Keep multiple checkpoints**: in case the latest is corrupted
4. **Checkpoint validation**: verify the saved state loads correctly

### What to save

```python
checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'step': step,
    'rng_state': torch.random.get_rng_state(),
    'cuda_rng_state': torch.cuda.get_rng_state_all(),
    'config': config,
    'data_position': data_loader.position,  # Resume from same data point
}
```

**Critical**: save the data loader position. Otherwise, on restart, you'll re-process data you've already seen, wasting tokens and potentially overfitting.

---

## 16.13 Key Takeaways

1. **Data quality > data quantity**: aggressive filtering and deduplication matter more than raw volume.

2. **The training objective is simple**: next-token prediction with cross-entropy. Every position provides a training signal.

3. **AdamW is standard**: with cosine LR schedule, warmup, gradient clipping, and weight decay.

4. **Distributed training is mandatory**: 3D parallelism (DP + TP + PP) for large models, with ZeRO for memory efficiency.

5. **Loss spikes are expected**: handle with gradient clipping, Z-loss, QK-norm, and checkpoint/rewind.

6. **Packing eliminates padding waste**: concatenate documents with EOS separators.

7. **Train for ~1 epoch on unique data**: repetition is suboptimal.

8. **Monitor MFU**: target 50%+ utilization.

9. **Fault tolerance is essential**: checkpoint frequently, validate saves, keep multiple copies.

10. **The complete training run** for a frontier model involves: petabytes of data, thousands of GPUs, months of wall time, and millions of dollars in compute.
