# 21. Training Failures, Debugging & Diagnostics: The Practitioner's Survival Guide

## Motivation

Textbooks teach you how to build a model. They don't teach you what to do when it breaks. Real training runs fail in subtle, expensive ways — and the ability to diagnose failures from a loss curve, gradient histogram, or activation dump is what separates a senior researcher from a student. This lecture is the "clinical medicine" of deep learning: pattern recognition for pathology.

---

## 21.1 Reading Loss Curves Like an X-Ray

The loss curve is your single most informative diagnostic. Learn to read it.

### Healthy Training

A healthy loss curve for an LLM pretraining run:

1. **Phase 1 (steps 0–warmup)**: Rapid drop. The model learns basic token frequency (bias terms).
2. **Phase 2 (post-warmup to ~10% of training)**: Steep decline. The model learns n-gram statistics, common patterns.
3. **Phase 3 (10–80% of training)**: Steady power-law improvement. Each doubling of tokens yields approximately the same loss reduction.
4. **Phase 4 (>80% of training)**: Flattening. Approaching the irreducible loss of the data distribution.

$$L(t) \approx a \cdot t^{-\alpha} + L_{\text{irreducible}}$$

The irreducible loss $L_{\text{irreducible}}$ depends on data quality, vocabulary size, and the entropy of natural language (typically around 1.5-2.0 nats for English text).

### Pathological Patterns

| Pattern | What it looks like | Likely cause | Fix |
|---|---|---|---|
| **Loss spike (transient)** | Sudden jump, recovers within 100-1000 steps | Bad data batch, hardware fault (bit flip), gradient explosion | Usually self-resolving. If frequent: check data quality, add Z-loss |
| **Loss spike (non-recovering)** | Sudden jump, never returns to pre-spike level | Learning rate too high, catastrophic forgetting | Rollback to checkpoint before spike, reduce LR |
| **Periodic oscillation** | Regular sinusoidal loss pattern | Learning rate too high for this stage; data ordering artifacts | Reduce LR; shuffle data more aggressively |
| **Plateau then cliff** | Flat loss, then sudden drop | Grokking (Sec 21.7); phase transition in learned representations | Wait (if you can afford it). This is sometimes good! |
| **Slow monotonic decline** | Loss drops but very slowly | LR too low; batch size too large (low gradient noise) | Increase LR; reduce batch size; check if model is at capacity |
| **Divergence (NaN)** | Loss → NaN or ±∞ | Numerical overflow (Sec 21.3); exploding gradients | Reduce LR, add gradient clipping, check bf16 numerics |
| **Train-eval gap widens** | Train loss drops, eval plateau | Overfitting or data contamination | More data, more regularization, check for leakage |
| **Eval loss rises** | Train still improving, eval getting worse | Classic overfitting | Stop training, use checkpoint with best eval loss |

### Pro tip: The Spike-Recovery Diagnostic

When you see a loss spike:

1. **Check the gradient norm at that step.** If it spiked too → the spike came from a bad gradient (usually a bad data batch).
2. **Check if all GPUs saw the spike.** If only some did → hardware fault.
3. **Check if the spike corresponds to a data boundary.** When you hit a new data source in your mixture → common cause.
4. **Measure the "recovery gap":** loss_after_recovery − loss_before_spike. If > 0, the spike caused permanent damage.

```python
# Monitoring code: detect and log spikes
def log_training_step(step, loss, prev_loss, grad_norm, threshold=2.0):
    if loss > prev_loss * threshold:
        print(f"⚠ SPIKE at step {step}: loss {prev_loss:.4f} → {loss:.4f} "
              f"(×{loss/prev_loss:.1f}), grad_norm={grad_norm:.2f}")
        # Log: data batch ID, GPU ID, recent LR value
        return True  # Flag for potential checkpoint rollback
    return False
```

---

## 21.2 Gradient Flow Diagnostics

### The "gradient ratio" test

For every parameter tensor $\mathbf{W}$, compute:

$$r = \frac{\|\nabla_W L\|}{\|\mathbf{W}\|} \cdot \eta$$

where $\eta$ is the learning rate. This is the ratio of the update magnitude to the weight magnitude.

| $r$ value | Interpretation |
|---|---|
| $r \in [10^{-3}, 10^{-1}]$ | Healthy. Updates are meaningful but not violent. |
| $r < 10^{-4}$ | Vanishing gradient / dead parameter. This layer is barely learning. |
| $r > 1$ | Exploding. The update is larger than the weight itself. |
| $r \approx 0$ for specific layers | Gradient is blocked. Check residual connections, norm layers. |

```python
def gradient_health_check(model, lr):
    """Run after loss.backward(), before optimizer.step()."""
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            weight_norm = param.data.norm().item()
            ratio = (grad_norm / (weight_norm + 1e-8)) * lr
            if ratio < 1e-5:
                print(f"  VANISHING: {name} ratio={ratio:.2e}")
            elif ratio > 0.5:
                print(f"  EXPLODING: {name} ratio={ratio:.2e}")
```

### Layer-wise gradient norm distribution

Plot $\|\nabla_{W_\ell} L\|$ for each layer $\ell$. In a healthy transformer:

- **Pre-Norm architecture**: gradient norms should be roughly constant across layers (the identity path through residual connections guarantees this).
- **Post-Norm architecture**: gradient norms typically decay from output to input layers (one reason Pre-Norm is preferred).

**If you see gradient norms that vary by more than 10× across layers**, something is wrong with your residual connections or normalization.

### The dead head problem

Attention heads can "die" — converging to a near-uniform attention pattern that contributes almost nothing:

$$\text{dead head: } \max_j A_{ij} - \frac{1}{n} < \epsilon \quad \text{for most } i$$

Detection:

```python
def check_attention_health(attn_weights, threshold=0.01):
    """attn_weights: (B, n_heads, seq_len, seq_len)"""
    # Entropy of each head's attention distribution
    entropy = -(attn_weights * attn_weights.log().clamp(min=-100)).sum(dim=-1)
    max_entropy = math.log(attn_weights.shape[-1])
    relative_entropy = entropy / max_entropy  # 1.0 = uniform = dead
    
    dead_mask = relative_entropy.mean(dim=(0, 2)) > 0.95  # Per-head average
    n_dead = dead_mask.sum().item()
    if n_dead > 0:
        print(f"WARNING: {n_dead} dead heads detected (uniform attention)")
    return dead_mask
```

**Why heads die**: Learning rate too high early in training → head converges to uniform before it could specialize. Fix: longer warmup, lower initial LR.

---

## 21.3 Numerical Stability: The Silent Killer

### bf16 softmax overflow

In bfloat16, the maximum value is ~3.4×10³⁸, but **precision is only 7 significant bits** (vs. 23 for fp32). The danger zone:

```
bf16 range:  [-3.4e38, 3.4e38]
bf16 precision: ~1/128 relative error
Maximum safe logit for softmax: ~88 (exp(88) ≈ 1.65e38)
```

If any logit exceeds ~88 in bf16, `exp(logit)` → inf → NaN after division.

**Why does this happen?** During attention:

$$\text{logit}_{ij} = \frac{\mathbf{q}_i^T \mathbf{k}_j}{\sqrt{d_k}}$$

If $\|\mathbf{q}\| \cdot \|\mathbf{k}\|$ grows (e.g., due to lack of normalization), logits can exceed safe range.

**Fixes (in order of preference):**

1. **Subtract max before exp** (standard log-sum-exp trick — already in PyTorch's `F.softmax`):
$$\text{softmax}(x_i) = \frac{e^{x_i - \max(x)}}{{\sum_j e^{x_j - \max(x)}}}$$

2. **QK-Norm**: Normalize Q and K before computing attention (used in some models):
$$\hat{\mathbf{q}} = \frac{\mathbf{q}}{\|\mathbf{q}\|}, \quad \hat{\mathbf{k}} = \frac{\mathbf{k}}{\|\mathbf{k}\|}$$

3. **Logit capping**: Clamp logits before softmax:
$$\text{logit} \leftarrow \text{tanh}(\text{logit} / c) \cdot c, \quad c \approx 50$$
Used by PaLM and Gemini.

4. **Compute attention in fp32**: Cast Q, K to fp32 before the dot product:
```python
scores = (q.float() @ k.float().transpose(-2, -1)) / math.sqrt(head_dim)
attn = F.softmax(scores, dim=-1).to(q.dtype)
```

### The Z-loss (logit entropy regularizer)

Used by PaLM, prevents logits from growing unbounded:

$$\mathcal{L}_z = \alpha \cdot \log^2 \left( \sum_i e^{z_i} \right)$$

where $z$ are the final logits (before softmax). This penalizes large log-sum-exp values, keeping logits in a numerically safe range.

```python
def z_loss(logits, alpha=1e-4):
    """Z-loss regularizer (PaLM). Prevents logit drift."""
    logsumexp = torch.logsumexp(logits, dim=-1)  # (B, n)
    return alpha * (logsumexp ** 2).mean()

# In training loop:
loss = F.cross_entropy(logits.view(-1, V), labels.view(-1))
loss = loss + z_loss(logits)
```

### RMSNorm numerical precision

RMSNorm computes $\sqrt{\text{mean}(x^2)}$. In bf16:

- $x^2$ can overflow if $|x| > \sim 1.8 \times 10^{19}$ (rare, but possible after many layers)
- The mean reduces the range, but the square root of mean of squares loses precision for very small or very large values

**Production fix**: Compute the norm in fp32:

```python
def rmsnorm_stable(x, weight, eps=1e-6):
    rms = x.float().pow(2).mean(dim=-1, keepdim=True).add(eps).rsqrt()
    return (x * rms).to(x.dtype) * weight
```

### Matrix multiplication precision

In bf16 matmul, errors accumulate with the inner dimension $d$:

$$\text{relative error} \approx \mathcal{O}\left(\frac{\sqrt{d}}{2^7}\right)$$

For $d = 4096$: relative error ≈ 50%. This is why TF32 tensor cores (19 bits) or fp32 accumulation are used for matmul on modern GPUs:

```python
# PyTorch defaults to fp32 accumulation for bf16 matmul on A100+
# If not, force it:
torch.backends.cuda.matmul.allow_tf32 = True
```

---

## 21.4 Activation Statistics: Your Monitoring Dashboard

### What to track during training

At every $N$ steps (e.g., every 100), log for each layer:

| Metric | Healthy range | Alarm |
|---|---|---|
| Activation mean | $\|mean\| < 1$ | $\|mean\| > 10$ → pre-activation shift |
| Activation std | $\sigma \in [0.5, 2.0]$ | $\sigma > 10$ → explosion; $\sigma < 0.01$ → collapse |
| Gradient norm | Comparable across layers | >100× variation across layers |
| Weight norm | Growing slowly | Sudden jumps or NaN |
| Attention entropy | $H > 0.5 \times H_{\max}$ | $H \to H_{\max}$ (dead) or $H \to 0$ (collapsed) |

```python
class ActivationMonitor:
    """Hook-based activation monitoring."""
    def __init__(self, model, log_every=100):
        self.stats = {}
        self.step = 0
        self.log_every = log_every
        
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, RMSNorm)):
                module.register_forward_hook(
                    lambda m, inp, out, n=name: self._record(n, out)
                )
    
    def _record(self, name, output):
        if self.step % self.log_every == 0:
            with torch.no_grad():
                self.stats[name] = {
                    'mean': output.float().mean().item(),
                    'std': output.float().std().item(),
                    'max': output.float().abs().max().item(),
                    'has_nan': output.isnan().any().item(),
                    'has_inf': output.isinf().any().item(),
                }
```

### The "activation norm grows with depth" problem

In transformers, the residual stream accumulates contributions from each layer:

$$\mathbf{x}_\ell = \mathbf{x}_0 + \sum_{i=1}^{\ell} f_i(\mathbf{x}_{i-1})$$

If each $f_i$ contributes something with norm $\sigma$, then:

$$\|\mathbf{x}_\ell\| \approx \|\mathbf{x}_0\| + \ell \cdot \sigma$$

The norm grows **linearly with depth**. This is why:

1. Pre-Norm is essential (normalizes before each sub-layer)
2. Output projection initialization should scale as $1/\sqrt{2N}$ (Doc 15.13)
3. Deep models (>100 layers) need special techniques (Post-Norm+ variants, DeepNorm)

---

## 21.5 Data-Related Failures

### The "loss doesn't decrease" checklist

When you start training and loss barely moves:

1. ✅ **Is the data getting to the model?** Print the first batch. Does it look like real text? (You'd be surprised how often it's all zeros or padding.)
2. ✅ **Is the label shift correct?** For autoregressive LM: input = tokens[:-1], labels = tokens[1:]. Off-by-one = random performance.
3. ✅ **Is the loss function correct?** Cross-entropy with the right vocab size? Check: initial loss ≈ $\log(V)$ (random prediction).
4. ✅ **Are you shuffling?** If all batches from same document → misleading loss. Need cross-document shuffling.
5. ✅ **Is padding masked?** If using variable-length sequences, unmasked padding tokens create spurious gradients.
6. ✅ **Is the learning rate reaching the model?** Check optimizer param groups. Frozen parameters → LR has no effect.

### Expected initial loss

For a randomly initialized language model:

$$L_{\text{initial}} \approx \log(V)$$

For $V = 32000$: $L_{\text{initial}} \approx 10.37$.

**If initial loss is significantly different**, something is wrong:

| Initial loss | Likely cause |
|---|---|
| $\gg \log(V)$ | Weight initialization too large, or loss not averaged properly |
| $\ll \log(V)$ | Data leakage (labels in input), or bug in masking |
| $= \log(V)$ exactly | Model output layer is zero-initialized (predicting uniform) — this is actually fine |
| Highly variable | Batch size too small, or data not shuffled |

### The memorization test

Before training for real: try to memorize a **single batch** of 4-8 sequences. If the model can't drive loss to near 0 on one batch within 200-500 steps, something is fundamentally broken.

```python
# THE most important sanity check
def memorization_test(model, config, device='cuda', steps=300):
    model = model.to(device)
    batch = torch.randint(0, config.vocab_size, (4, 64), device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    for step in range(steps):
        logits, _ = model(batch[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, config.vocab_size),
                               batch[:, 1:].reshape(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    assert loss.item() < 0.01, f"FAILED: model can't memorize. Loss={loss.item():.4f}"
    print(f"✓ Memorization test passed (loss={loss.item():.6f})")
```

### Data ordering matters more than you think

Common mistakes:

- **Sorted data**: If your data pipeline sorts by length (common in NLP), the model sees all short sequences first → terrible generalization.
- **Domain clustering**: If data is grouped by source (all Wikipedia, then all code, then all web) → catastrophic forgetting of earlier domains.
- **Epoch boundaries**: Transition between epochs can cause spikes if not handled smoothly.

**Best practice**: Pre-shuffle into random chunks. Within each chunk, pack sequences efficiently (Doc 16.9). Across chunks, load randomly.

---

## 21.6 Optimizer State Debugging

### AdamW state inspection

Each parameter in AdamW has two state tensors:

- $m$ (first moment / momentum): exponential moving average of gradient
- $v$ (second moment): exponential moving average of squared gradient

When things go wrong:

```python
def inspect_optimizer_state(optimizer, param_name_map):
    """Inspect Adam state for anomalies."""
    for group in optimizer.param_groups:
        for p in group['params']:
            state = optimizer.state.get(p, {})
            if 'exp_avg' in state:
                m = state['exp_avg']
                v = state['exp_avg_sq']
                name = param_name_map.get(id(p), 'unknown')
                
                # Effective step size: m / sqrt(v)
                effective_update = m / (v.sqrt() + 1e-8)
                
                print(f"{name}:")
                print(f"  m: mean={m.mean():.4e}, std={m.std():.4e}")
                print(f"  v: mean={v.mean():.4e}, max={v.max():.4e}")
                print(f"  update: mean={effective_update.mean():.4e}")
                
                # Warning signs
                if v.max() > 1e6:
                    print(f"  ⚠ Very large v! Gradient variance is huge.")
                if m.abs().max() > 100:
                    print(f"  ⚠ Very large m! Gradient is biased in one direction.")
```

### The "Adam is too adaptive" problem

Adam's per-parameter learning rate means:

- Parameters with **consistently large gradients** get a **small** effective LR ($\propto 1/\sqrt{v}$)
- Parameters with **sparse or small gradients** get a **large** effective LR

This is usually good, but can be pathological:

- **Embedding layers**: frequent tokens get small updates, rare tokens get large updates → vocabulary imbalance
- **Post-norm layers**: norm parameters get very different effective LR from weight matrices

**Mitigation**: Weight decay (which acts on the raw weight, not the adaptive rate) helps equalize. This is a key reason weight decay is essential in transformer training.

### Learning rate warmup: deeper reason than "stability"

The standard explanation: "warmup prevents instability from large initial gradients."

The deeper reason: **Adam's second moment estimate is biased** in early training.

At step $t$, the bias-corrected second moment is:
$$\hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

For $\beta_2 = 0.95$, after 1 step: $\hat{v}_1 = v_1 / 0.05 = 20 \cdot v_1$. After 20 steps: $\hat{v}_{20} = v_{20} / 0.36 \approx 2.8 \cdot v_{20}$.

The bias correction amplifies the first few updates. But if $v$ is estimated from very few samples, it's noisy. So the corrected estimate is a **noisy large number** → unstable updates.

Warmup gives the optimizer time to get a stable estimate of $v$ before taking large steps.

**Implication for practice**: warmup steps should be proportional to $1 / (1 - \beta_2)$. For $\beta_2 = 0.95$, that's ~20 steps minimum. For $\beta_2 = 0.999$, ~1000 steps.

---

## 21.7 Grokking & Phase Transitions

### What is grokking?

Grokking: the phenomenon where a model **memorizes** the training data quickly (achieving 0 training loss) but then **much later** suddenly generalizes (test loss drops).

```
Steps 0-100:    Train loss → 0,  Test loss stays high
Steps 100-1000: Train loss = 0,  Test loss stays high  (memorizing)
Steps 1000-???  Train loss = 0,  Test loss drops suddenly (grokking!)
```

### Why it happens (current theory):

1. Memorization is a "shortcut" — the model uses the large capacity to store input-output pairs directly
2. Weight decay / regularization slowly pushes the model toward simpler (lower weight norm) solutions
3. At some point, the "circuit" that implements the general algorithm becomes lower-loss-landscape cost than the memorization circuit
4. Phase transition: the model abruptly switches from memorization to algorithm

### Implications for LLM training:

- **Training longer sometimes helps even after overfitting** — especially with weight decay
- The algorithm-learning phase may require **5-100×** more steps than memorization
- **Weight decay is the crucial ingredient** — without it, grokking doesn't happen
- This suggests LLMs might benefit from training well beyond the point of "convergence"

### Double descent

**Test loss as a function of model size** shows a U-shape, then another descent:

$$
\text{Classical regime} \rightarrow \overset{\text{peak}}{\text{interpolation threshold}} \rightarrow \text{modern regime (overparameterized)}
$$

At the interpolation threshold (model has just enough capacity to memorize training set), test loss is **worst**. Adding more parameters past this point causes test loss to drop again.

**Key insight for practitioners**: If your model is performing poorly, making it **slightly larger** might make it **worse** (you're at the peak). Making it **much larger** will make it better. Jump over the threshold.

---

## 21.8 The "NaN Autopsy"

When your training goes NaN, trace backwards:

### Step 1: Which tensor went NaN first?

```python
# Register hook on every module to catch the first NaN
def nan_hook(module, input, output):
    if isinstance(output, torch.Tensor) and output.isnan().any():
        raise RuntimeError(f"NaN detected in {module.__class__.__name__}")
    elif isinstance(output, tuple):
        for i, o in enumerate(output):
            if isinstance(o, torch.Tensor) and o.isnan().any():
                raise RuntimeError(f"NaN detected in output[{i}] of {module.__class__.__name__}")

for module in model.modules():
    module.register_forward_hook(nan_hook)
```

### Step 2: Common NaN causes (in order of frequency):

| Cause | Where | Detection |
|---|---|---|
| bf16 softmax overflow | Attention scores | Check max logit before softmax |
| log(0) | Cross-entropy with hard zeros | Add epsilon: `log(p + 1e-8)` |
| Division by zero | Normalization layers, attention | Check denominator values |
| inf × 0 | Masked attention + softmax | Use additive masking (−inf) correctly |
| exp overflow | Logits > 88 in bf16 | Check logit magnitudes |
| sqrt of negative | Rare numerical precision issue | Use `clamp(min=eps)` before sqrt |
| Gradient overflow | During backward pass | Gradient clipping + fp32 accumulation |

### Step 3: The fix order

1. First: check data (NaN in input → NaN everywhere)
2. Second: check specific operation (add the NaN hook)
3. Third: check precision (run in fp32 — if it works, it's a numerical issue)
4. Fourth: reduce learning rate (if it works, it's a stability issue)
5. Fifth: check initialization (too large → immediate explosion)

---

## 21.9 Distributed Training Failures

### Gradient synchronization bugs

In distributed training, gradients are averaged across GPUs. If one GPU has different data, batch size, or even a different random seed for dropout, the synchronized gradient is an **average of inconsistent things**.

**Detection**: Run with 1 GPU and N GPUs on the same data. If loss curves diverge → synchronization bug.

### The "all-reduce is correct but slow" problem

In hybrid parallelism (TP + DP + PP), communication can dominate compute if partitioned badly:

- **Tensor parallel**: all-reduce after every attention and FFN → N all-reduces per layer
- **Pipeline parallel**: bubble time when pipeline is under-filled
- **Overlap**: overlapping all-reduce with computation (backward) saves time, but introduces bugs if not done carefully

**Key metric**: MFU (Model FLOPs Utilization). For A100s:

| MFU | Assessment |
|---|---|
| > 50% | Excellent |
| 30-50% | Good |
| 10-30% | Something is bottlenecked (usually communication) |
| < 10% | Broken (probably waiting on communication or data loading) |

### Memory debugging

When you OOM:

```python
def memory_report():
    """Print GPU memory breakdown."""
    print(f"Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    print(f"Reserved:  {torch.cuda.memory_reserved()/1e9:.2f} GB")
    print(f"Max alloc: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

# Rule of thumb for memory per GPU:
#   Model params (bf16):   2 * P bytes
#   Optimizer state (fp32): 12 * P bytes (weights + m + v in fp32)
#   Activations:           B * seq_len * d_model * n_layers * ~10 bytes
#   Gradients (bf16):      2 * P bytes
#   Total ≈ 16P + activations
```

For a 7B model: $16 \times 7 \times 10^9 = 112$ GB just for parameters + optimizer. That's why you need ZeRO-3 or model parallelism.

---

## 21.10 The Debugging Checklist (Condensed)

When loss doesn't go down, check in order:

```
□ 1. Data sanity: print first batch, verify it's real text
□ 2. Label alignment: input[:-1] → labels[1:] for autoregressive
□ 3. Initial loss = log(V)? If not, something is very wrong
□ 4. Memorization test: can you overfit 1 batch?
□ 5. Gradient flow: are all parameters getting gradients?
□ 6. Learning rate: is it > 0? Is the scheduler working?
□ 7. Masking: are padding tokens masked in loss? Is causal mask correct?
□ 8. Data shuffling: is data coming in diverse batches?
□ 9. Numerical: any NaN/Inf in activations?
□ 10. Architecture: is the residual connection correct? (x + f(x), not just f(x))
```

This checklist has saved more training runs than any algorithmic improvement.
