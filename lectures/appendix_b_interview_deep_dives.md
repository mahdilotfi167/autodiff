# Appendix B: Interview Deep Dives — Questions, Thinking Frameworks & Model Answers

## About This Document

These are the types of technical questions asked at Anthropic, OpenAI, Google DeepMind, and top AI startups for research engineer and ML researcher positions. Each question includes: the question, why they ask it, the expected thinking framework, and a model answer. The goal is not to memorize answers but to practice the **thinking patterns** interviewers look for.

---

## Category 1: Architecture & First Principles

### Q1: "Why does the transformer use scaled dot-product attention instead of additive attention?"

**Why they ask**: Tests whether you understand the computational tradeoff, not just the formula.

**Framework**: Compare computational complexity, then discuss practical GPU considerations.

**Model answer**:

Additive attention: $\text{score}(q, k) = \mathbf{v}^T \tanh(\mathbf{W}_1 \mathbf{q} + \mathbf{W}_2 \mathbf{k})$. This requires a learned weight matrix and a nonlinearity — sequential operations that are slow on GPUs.

Dot-product attention: $\text{score}(q, k) = \mathbf{q}^T \mathbf{k}$. This is a single batched matrix multiply (GEMM) — the operation GPUs are most optimized for. For $n$ queries and $n$ keys, all scores are computed in one $\mathcal{O}(n^2 d)$ GEMM.

The scaling factor $1/\sqrt{d_k}$ compensates for the fact that dot products grow with dimension $d_k$: if entries of $\mathbf{q}$ and $\mathbf{k}$ are independent with variance 1, then $\mathbf{q}^T \mathbf{k}$ has variance $d_k$. Without scaling, large $d_k$ pushes softmax into saturation → vanishing gradients.

Bahdanau (2015) actually found additive attention slightly better in quality. But Vaswani (2017) showed that with scaling, dot-product matches quality and is much faster. Engineering won.

---

### Q2: "Walk me through what happens when you double the model dimension $d$ while keeping everything else fixed."

**Why they ask**: Tests dimensional analysis skills and awareness of cascading effects.

**Framework**: Go parameter-by-parameter through the architecture.

**Model answer**:

Let's trace through a GPT with $d \rightarrow 2d$:

| Component | Parameters before | Parameters after | Change |
|---|---|---|---|
| Embedding ($V \times d$) | $Vd$ | $2Vd$ | 2× |
| Attention QKV ($3 \times d \times d$) per layer | $3d^2$ | $12d^2$ | **4×** |
| Attention output ($d \times d$) per layer | $d^2$ | $4d^2$ | **4×** |
| FFN ($d \times 4d + 4d \times d$) per layer | $8d^2$ | $32d^2$ | **4×** |
| LM head ($d \times V$) | $Vd$ | $2Vd$ | 2× |

The per-layer parameters scale as $d^2$, so doubling $d$ **quadruples** the per-layer parameter count. The embedding/head scale linearly.

FLOPs per token: also $\propto d^2$ per layer → 4× more compute.

Activation memory per layer: $\propto n \cdot d$ → 2× more memory.

KV cache: $\propto n \cdot n_h \cdot d_k$. If $d_k = d / n_h$ and $n_h$ stays fixed, $d_k$ doubles → KV cache 2×.

**The punchline**: Doubling width is expensive — ~4× more parameters and compute per layer. This is why scaling laws often prefer depth (adding layers) over width (increasing $d$) until a certain ratio.

---

### Q3: "Why Pre-Norm instead of Post-Norm?"

**Framework**: Gradient flow analysis.

**Model answer**:

Post-Norm: $\mathbf{x}_{\ell+1} = \text{LN}(\mathbf{x}_\ell + f(\mathbf{x}_\ell))$

The LayerNorm wraps around the residual connection. During backpropagation:

$$\frac{\partial \mathbf{x}_{\ell+1}}{\partial \mathbf{x}_\ell} = \frac{\partial \text{LN}}{\partial (\mathbf{x}_\ell + f(\mathbf{x}_\ell))} \cdot \left(\mathbf{I} + \frac{\partial f}{\partial \mathbf{x}_\ell}\right)$$

The LayerNorm Jacobian modifies the gradient at every layer. Over $N$ layers, these modifications compound → gradient can vanish or become unstable.

Pre-Norm: $\mathbf{x}_{\ell+1} = \mathbf{x}_\ell + f(\text{LN}(\mathbf{x}_\ell))$

Gradient:
$$\frac{\partial \mathbf{x}_{\ell+1}}{\partial \mathbf{x}_\ell} = \mathbf{I} + \frac{\partial f}{\partial \text{LN}} \cdot \frac{\partial \text{LN}}{\partial \mathbf{x}_\ell}$$

The identity matrix $\mathbf{I}$ provides a clean gradient path. Even if the second term vanishes, the gradient at least equals the identity. Over $N$ layers:

$$\frac{\partial \mathbf{x}_N}{\partial \mathbf{x}_0} = \mathbf{I} + \sum_{\text{paths}} (\text{higher-order terms})$$

The gradient is always at least $\mathbf{I}$ — it can't vanish.

**The tradeoff**: Pre-Norm enables deeper models but the final representation is unnormalized (it's the sum of all layer outputs, not LayerNorm'd). That's why we need a final RMSNorm before the LM head.

---

## Category 2: Training & Optimization

### Q4: "You start a pretraining run and the loss barely decreases after 1000 steps. Walk me through your debugging process."

**Why they ask**: Tests practical debugging skills. This is the #1 practical question at every lab.

**Framework**: Follow the debugging checklist (Doc 21.10), prioritize cheap checks first.

**Model answer**:

"I'd follow a systematic checklist, starting with the cheapest checks:

1. **Initial loss check**: Is loss ≈ log(V)? If it's much lower, there's a data leakage bug. If much higher, initialization is wrong.

2. **Data sanity**: Decode and print the first batch. Is it actual text? Surprising number of bugs are 'model is training on zeros.'

3. **Label alignment**: For autoregressive LMs, input = tokens[:-1], labels = tokens[1:]. Print both and verify alignment. Off-by-one → random performance.

4. **Memorization test**: Take 4 sequences, try to overfit them to near-zero loss. If this fails, the model *can't learn at all* — architecture or gradient bug.

5. **Gradient check**: Are all parameters receiving gradients? `for name, p in model.named_parameters(): if p.grad is None: print(name)`. Common bug: frozen parameters, detached tensors.

6. **Learning rate**: Is it actually nonzero? Check `optimizer.param_groups[0]['lr']`. Scheduler bugs can keep LR at 0 during warmup.

7. **Numerical issues**: Check for NaN/Inf in activations. Run one batch in fp32 — if it works in fp32 but not bf16, it's a precision issue.

8. **Loss masking**: If using padding, are padding tokens masked in the loss? Unmasked padding creates phantom gradients.

Each of these takes <5 minutes to check. I'd do them all before touching hyperparameters."

---

### Q5: "Explain why AdamW uses decoupled weight decay instead of L2 regularization. When does the difference actually matter?"

**Framework**: Derive the update rules for both, show when they diverge.

**Model answer**:

L2 regularization modifies the loss: $\tilde{L} = L + \frac{\lambda}{2}\|\mathbf{w}\|^2$

The gradient becomes: $\tilde{g} = g + \lambda w$

Adam's update with L2:
$$w_{t+1} = w_t - \eta \cdot \frac{m_t}{(\sqrt{v_t} + \epsilon)}$$
where $m_t$ and $v_t$ are computed from $\tilde{g} = g + \lambda w$.

The problem: Adam's adaptive denominator $\sqrt{v_t}$ scales the regularization term differently for each parameter. Parameters with large historical gradients get *less* effective regularization.

Decoupled weight decay (AdamW):
$$w_{t+1} = (1 - \eta\lambda)w_t - \eta \cdot \frac{m_t}{\sqrt{v_t} + \epsilon}$$

The decay $\eta\lambda w_t$ is applied *directly* to the weight, independent of the adaptive learning rate. Every parameter gets the same proportional decay.

**When it matters**: It matters most when the adaptive learning rates vary widely across parameters — which they do in transformers (embedding layers vs. attention layers have very different gradient magnitudes). With L2, some layers would be over-regularized and others under-regularized.

---

### Q6: "Your training loss suddenly spikes at step 50,000. The spike is 3× the pre-spike loss. What do you do?"

**Framework**: Diagnose → decide → (potentially) intervene.

**Model answer**:

"First, **observe**: Does it recover? Wait 100-500 steps. If loss returns to near pre-spike level, it's probably a transient data artifact or stochastic event. Log it and move on.

If it **doesn't recover** within ~500 steps:

1. **Check gradient norm at the spike step**. If grad norm also spiked → the cause is in the gradient (bad batch, numerical overflow).

2. **Check the data batch**. Can I identify which batch triggered it? Sometimes a single corrupted document (e.g., base64-encoded binary) causes extreme gradients.

3. **Check hardware**. Silent data corruption (bit flips in GPU memory) can cause spikes. If running on many GPUs, check if the spike occurred on all of them or just one.

**Action plan**:
- If the spike didn't recover: **roll back to the checkpoint before the spike** and **skip the problematic data batch**.
- If spikes are recurring (>1 per 10K steps): reduce peak learning rate by 30-50%, or add a Z-loss regularizer on logits.
- If spikes happen at data source boundaries: improve data shuffling.
- Nuclear option: reduce LR, increase warmup, add gradient clipping. But this slows training."

---

## Category 3: Scaling & Efficiency

### Q7: "You have a budget of 10²⁴ FLOPs. How do you decide the model size and dataset size?"

**Framework**: Apply Chinchilla scaling laws, then adjust for practical constraints.

**Model answer**:

"The Chinchilla-optimal allocation says:

$$N^* \propto C^{0.5}, \quad D^* \propto C^{0.5}$$

with about 20 tokens per parameter.

For $C = 10^{24}$ FLOPs:
- Using approximately $C \approx 6ND$ (the FLOPs-per-token approximation):
- $10^{24} = 6 \times N \times 20N = 120N^2$
- $N^* \approx \sqrt{10^{24}/120} \approx 2.9 \times 10^{11}$ ≈ **290B parameters**
- $D^* = 20 \times 290B \approx 5.8T$ tokens

But this is training-compute-optimal, not inference-optimal. If I'm serving this model:

- 290B parameters → expensive to serve
- Alternative: Train a 70B model on 23T tokens ($6 \times 70B \times 23T \approx 10^{24}$)
- Same compute, 4× smaller model → 4× cheaper inference
- Quality will be slightly worse, but the 70B model is much more practical

The decision depends on the use case:
- Research benchmark: go big (290B)
- Production API: go smaller, train longer (70B on more data)
- Edge deployment: even smaller (7B on even more data)"

---

### Q8: "How does FlashAttention achieve exact attention without approximation while being 2-3× faster?"

**Framework**: IO complexity, not algorithmic complexity.

**Model answer**:

"Standard attention has $\mathcal{O}(n^2)$ FLOPs, and FlashAttention also has $\mathcal{O}(n^2)$ FLOPs. The speedup is not from reducing computation — it's from reducing **memory IO**.

Standard implementation:
1. Compute $\mathbf{S} = \mathbf{QK}^T$ — writes $n \times n$ matrix to HBM
2. Compute $\mathbf{P} = \text{softmax}(\mathbf{S})$ — reads and writes $n \times n$ matrix
3. Compute $\mathbf{O} = \mathbf{PV}$ — reads $n \times n$ matrix

Total HBM access: $\mathcal{O}(n^2)$ reads and writes. For $n = 8K$, $d = 128$: the $n \times n$ matrices are $8K \times 8K = 256$ MB. This is the bottleneck.

FlashAttention's insight: tile the computation. Process blocks of $B_r \times B_c$ at a time (where $B_r, B_c$ fit in SRAM):

1. Load a block of $\mathbf{Q}$ ($B_r \times d$) into SRAM
2. Iterate over blocks of $\mathbf{K}, \mathbf{V}$ ($B_c \times d$ each)
3. Compute partial attention within SRAM using the **online softmax** trick
4. Write only the final output (not the $n \times n$ intermediate)

The online softmax trick maintains running statistics:
$$m_{\text{new}} = \max(m_{\text{old}}, \max(\mathbf{s}_{\text{block}}))$$
$$\ell_{\text{new}} = e^{m_{\text{old}} - m_{\text{new}}} \cdot \ell_{\text{old}} + \sum e^{s_j - m_{\text{new}}}$$

This avoids ever materializing the $n \times n$ matrix in HBM.

Total HBM access: $\mathcal{O}(n^2 d^2 / M)$ where $M$ is SRAM size. Since $M \gg d^2$ on modern GPUs, this is much less than $\mathcal{O}(n^2)$.

**The result is mathematically exact** — same output to floating point precision. Just computed in a more memory-efficient order."

---

## Category 4: Alignment & Safety

### Q9: "Derive the DPO loss from first principles. Why does it work without a separate reward model?"

**Framework**: Start from RLHF objective, solve for optimal policy, substitute back.

**Model answer**:

"Start with the RLHF objective:

$$\max_\pi \mathbb{E}_{y \sim \pi}[r(x, y)] - \beta D_{\text{KL}}[\pi(y|x) \| \pi_{\text{ref}}(y|x)]$$

The optimal policy has a closed-form solution:

$$\pi^*(y|x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y|x) \exp\left(\frac{r(x,y)}{\beta}\right)$$

Rearranging for the reward:

$$r(x, y) = \beta \log \frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)$$

Now, the Bradley-Terry preference model says:

$$P(y_w \succ y_l | x) = \sigma(r(x, y_w) - r(x, y_l))$$

Substituting the reward expression (the $Z(x)$ terms cancel!):

$$P(y_w \succ y_l | x) = \sigma\left(\beta \log \frac{\pi^*(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi^*(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)$$

This gives us the DPO loss. We replace $\pi^*$ with our policy $\pi_\theta$ and maximize:

$$\mathcal{L}_{\text{DPO}} = -\mathbb{E}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right]$$

**Why no reward model**: The key insight is that the optimal policy already implicitly defines a reward function. DPO parameterizes the reward through the policy itself, bypassing the need to explicitly learn $r$.

**The gradient is intuitive**: it increases the probability of the preferred response and decreases the probability of the dispreferred response, with the magnitude depending on how "surprised" the model is by the preference (how much the current policy disagrees with the label)."

---

### Q10: "How would you detect if a model is being deceptive?"

**Why they ask**: Core Anthropic question. Tests alignment thinking.

**Framework**: Define deception precisely, then propose detection methods at different levels.

**Model answer**:

"First, I'd define deception precisely: a model is deceptive if its **internal state** represents one thing but its **output** claims another.

Detection at three levels:

**Level 1: Behavioral tests**
- Present scenarios where honesty and helpfulness conflict
- Check if the model adjusts its answers based on whether it thinks it's being evaluated
- Test consistency: ask the same question in different contexts and check for contradictions

**Level 2: Representation probing**
- Train linear probes on the model's internal activations to predict 'truthful' vs 'deceptive' statements
- If the model 'knows' the truth internally but outputs a falsehood, the probe will detect the mismatch
- Use the contrast-consistent search (CCS) approach: find a direction in activation space that is consistent with logical negation

**Level 3: Mechanistic analysis**
- Use sparse autoencoders to find features related to deception, honesty, and self-awareness
- Use activation patching to identify circuits that activate when the model 'decides' to be deceptive
- Look for features that track 'am I being evaluated?' vs 'am I being used normally'

**The fundamental challenge**: We need to distinguish between:
- The model being wrong (honest but mistaken)
- The model being evasive (honest but incomplete)
- The model being deceptive (internally representing truth, outputting falsehood)

Only the internal representation analysis (levels 2-3) can distinguish these."

---

## Category 5: Systems & Implementation

### Q11: "You need to serve a 70B model with <200ms time-to-first-token. How do you architect this?"

**Framework**: Work backwards from the latency requirement.

**Model answer**:

"70B model in bf16 = 140 GB. Single A100 (80GB) can't hold it. Need multi-GPU.

**Option 1: Tensor Parallelism across 2 nodes (4 GPUs each)**
- 140 GB / 8 GPUs = 17.5 GB per GPU + KV cache
- Within-node: NVLink (900 GB/s) → low communication overhead
- Cross-node: InfiniBand (400 Gb/s) → significant overhead
- TTFT depends on prompt length. For 1K tokens: ~100ms with 8-way TP on NVLink

**Option 2: Quantize to INT4 + 2-GPU TP**
- 70B INT4 = 35 GB → fits on 2 A100s with room for KV cache
- 2-way TP on NVLink: minimal communication overhead
- TTFT for 1K tokens: ~80ms

**Option 3: Single H100 (80GB) with FP8 quantization**
- 70B FP8 = 70 GB → fits on one H100
- No communication overhead
- H100 memory bandwidth: 3.35 TB/s
- TTFT for 1K tokens: ~50ms

**I'd choose Option 3** if H100s are available. Single-GPU eliminates all communication latency. FP8 has minimal quality impact and is natively supported on H100.

For the serving stack: vLLM or TensorRT-LLM with continuous batching, PagedAttention for KV cache management, and prefix caching for the system prompt."

---

### Q12: "Explain the memory requirements for training a 7B parameter model."

**Framework**: Break down into components.

**Model answer**:

"For a 7B model in mixed precision (bf16 forward/backward, fp32 optimizer state):

| Component | Per-parameter bytes | Total |
|---|---|---|
| Model weights (bf16) | 2 | 14 GB |
| Gradients (bf16) | 2 | 14 GB |
| Adam first moment (fp32) | 4 | 28 GB |
| Adam second moment (fp32) | 4 | 28 GB |
| Master weights (fp32) | 4 | 28 GB |

**Optimizer state alone: 84 GB.** Total with model + gradients: **112 GB.**

Plus activations (the tensors saved for backward pass):
$$\text{Activations} \approx B \times n \times d \times N \times (\text{bytes per activation}) \times (\text{saved tensors per layer})$$

For $B=4$, $n=2048$, $d=4096$, $N=32$, ~10 saved tensors per layer at 2 bytes:
$$4 \times 2048 \times 4096 \times 32 \times 10 \times 2 \approx 21 \text{ GB}$$

With activation checkpointing (Doc 16.6), this drops by ~5× to ~4 GB (recompute activations during backward).

**Total**: 112 + 4 = **~116 GB** (with activation checkpointing).

This barely fits on 2 A100s (160 GB total). In practice, you'd use ZeRO-2 (shard optimizer state across GPUs) or ZeRO-3 (shard everything)."

---

## Category 6: Open-Ended Research Questions

### Q13: "What's the most important unsolved problem in LLMs?"

**Why they ask**: Tests research taste and depth of understanding.

**Multiple valid answers** (pick one and defend it):

**Answer A: Hallucination**
"Models generate fluent text that is factually wrong, and they do it with the same confidence as correct text. This is arguably the biggest barrier to trustworthy deployment. Current mitigations (retrieval augmentation, citation, uncertainty estimation) help but don't solve the fundamental problem: the model doesn't have a principled mechanism for distinguishing what it knows from what it's guessing."

**Answer B: Alignment robustness**
"Current alignment methods (RLHF, DPO) are brittle. Jailbreaks consistently work across models, and the alignment surface is shallow — small perturbations in input can bypass all safety training. We need alignment that's robust to distribution shift, not just optimized for the training distribution of harmful prompts."

**Answer C: Data efficiency**
"We're running out of human-generated text. Current models need trillions of tokens — more than humanity has produced. Either we need better data efficiency (learning more from less) or reliable synthetic data generation. This will become the binding constraint sooner than compute."

---

### Q14: "If you could change one thing about the transformer architecture, what would it be?"

**No single right answer. Good answers show depth.**

**Example**: "I'd make attention complexity subquadratic without sacrificing quality. Current linear attention approximations lose too much quality. If we could compute exact attention in $\mathcal{O}(n \log n)$ or $\mathcal{O}(n \sqrt{n})$, it would enable dramatically longer contexts without the current engineering complexity of ring attention, sliding windows, and chunked prefill. The right solution might involve structured sparsity in the attention pattern that maintains the expressiveness of full attention."

---

## How to Use This Document

1. **Practice the thinking framework, not the answer.** Interviewers detect memorized answers. Practice structuring your response: state your approach, work through it, and acknowledge tradeoffs.

2. **Go deep when you can, broad when you must.** If you know the answer deeply, show that depth. If you're unsure, sketch the high-level approach and be honest about where your knowledge ends.

3. **Connect to practical experience.** "When I implemented this..." or "In my experiments, I found..." is more convincing than "The paper says..."

4. **State assumptions explicitly.** "Assuming an A100 with 80GB memory and 2 TB/s bandwidth..." shows you know the hardware landscape.

5. **Be willing to say "I don't know, but here's how I'd figure it out."** This is always better than guessing. Top labs value intellectual honesty.
