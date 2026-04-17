# Appendix A: Tribal Knowledge — 75 Things Textbooks Don't Teach You

## About This Document

These are observations, heuristics, and hard-won lessons from practitioners who train and deploy LLMs at scale. None of these are "theorems" — they're patterns that hold in practice and the kind of knowledge that separates senior researchers from paper-readers.

---

## Architecture & Design

**1. The transformer won not because it's theoretically superior, but because it's GPU-friendly.** Attention is a large batched matrix multiply — exactly what GPU tensor cores optimize for. An architecture that's 2× more theoretically efficient but 5× harder to parallelize will lose in practice.

**2. Decoder-only won empirically, not theoretically.** There's no proof that decoder-only is better than encoder-decoder. It just happened to scale better in practice, partially because the pretraining objective (next token) is simpler and partially because it works for every task format without special handling.

**3. The "4× FFN expansion" (standard: $d_{ff} = 4d$) is a historical accident.** It came from the original transformer paper and was never rigorously optimized. SwiGLU with $d_{ff} \approx 8/3 \cdot d$ gives the same parameter count with better quality. Many practitioners still use 4× because "it's what the paper says."

**4. Bias terms are removed in modern LLMs for a reason most people don't know.** It's not for parameter savings (biases are tiny). It's because biases interact badly with tensor parallelism — they need an additional all-reduce or careful handling during sharding. Removing them simplifies the distributed implementation.

**5. Weight tying (embedding = unembedding) saves parameters but hurts at large scale.** At 7B+, the embedding and LM head want to be in different subspaces. LLaMA doesn't tie weights. GPT-2 does. The field moved away from tying.

**6. The number of layers matters more than the number of heads.** A 32-layer model with 16 heads consistently outperforms a 16-layer model with 32 heads (same parameter count). Depth > width for reasoning, but width matters for knowledge storage (in MLPs).

**7. RoPE's base frequency (θ=10000) is not magic.** Different base frequencies trade off short-range vs long-range position sensitivity. θ=10000 works well for 2K-8K context. For 100K+ context, θ=500000 or higher is needed (see NTK-aware scaling).

**8. RMSNorm > LayerNorm not because of quality, but because it's faster.** RMSNorm skips the mean subtraction, saving one reduction operation per call. Across billions of tokens, this adds up. Quality is comparable.

---

## Training Dynamics

**9. The learning rate is the most important hyperparameter, and the second most important is the learning rate.** No other hyperparameter comes close in impact. If you can only tune one thing, tune the peak learning rate.

**10. Cosine schedule beats linear decay because of the "annealing" effect at the end.** The final phase of cosine decay acts as simulated annealing — slowly settling into a local minimum. The sharp transition to low LR forces the model to commit to a solution.

**11. "Warm restarts" (cyclical LR) work because each restart escapes local minima.** But they're rarely used for LLMs because each restart wastes compute re-exploring the loss landscape. Cosine with a single cycle is the sweet spot.

**12. Batch size has a "critical batch size" below which increasing it helps linearly.** Above the critical batch size, each doubling gives diminishing returns. The critical batch size increases during training as the loss landscape becomes smoother.

$$B_{\text{crit}} \approx \frac{B_{\text{noise}}}{L - L_*}$$

where $B_{\text{noise}}$ is the gradient noise scale and $L - L_*$ is the distance to optimal loss. As loss decreases, critical batch size increases. **This is why large batch sizes work better later in training.**

**13. Gradient accumulation is NOT equivalent to larger batch size.** With gradient accumulation over $K$ micro-batches, each micro-batch's gradients are computed with different model weights (slightly, because of momentum). With true large batch, all gradients use identical weights. The noise structure is different. In practice, the difference is small but measurable.

**14. AdamW's weight decay acts as a constant pull toward zero, not a gradient-dependent regularizer.** Unlike L2 regularization (which adds $\lambda w$ to the gradient), decoupled weight decay subtracts $\lambda w$ from the weight directly. The difference:
- L2: effective regularization scales with learning rate
- Decoupled WD: regularization is independent of learning rate

This matters when you change the learning rate (e.g., during cosine decay). With L2, reducing LR also reduces regularization. With decoupled WD, regularization stays constant.

**15. Setting β₂ = 0.95 (instead of the default 0.999) is critical for transformer training.** Transformers have heavy-tailed gradient distributions. With β₂ = 0.999, the second moment estimate is dominated by rare large gradients → the effective learning rate oscillates. β₂ = 0.95 adapts faster to the current gradient landscape.

**16. Loss spikes are more common with larger models.** Theory: larger models have sharper loss landscapes → small perturbations (bad batch, numerical issue) cause larger excursions. Empirical finding: loss spike frequency scales approximately as $\propto N^{0.3}$.

**17. "Stable training" often means "learning slowly."** Some instability is a sign the model is exploring. If you've eliminated all spikes, you may have over-regularized. The goal is controlled instability, not zero instability.

**18. Gradient clipping (max_norm = 1.0) is consensus but not optimal.** For very large models, adaptive clipping (clip to a percentile of historical gradient norms) works better than fixed thresholds. But fixed 1.0 is a good default.

---

## Data

**19. Data quality > data quantity, by a large margin.** A 7B model trained on 1T clean tokens outperforms a 13B model trained on 1T noisy tokens. The model is only as good as its training distribution.

**20. You cannot train on the same data twice and expect linear improvement.** Repeating data helps some (up to 4-8 repeats are OK), but each repetition gives diminishing returns. After ~16 repeats, the model starts memorizing rather than generalizing. Unique tokens matter.

**21. The "70% web / 15% code / 10% books / 5% academic" mix is a rough starting point.** Optimal ratios depend on your downstream tasks. More code → better reasoning. More books → better long-form coherence. More academic → better factual accuracy. But the interactions are complex.

**22. Code is the secret ingredient for reasoning.** Models trained with code in the mix consistently perform better on logical reasoning, even on tasks that have nothing to do with code. Theory: code is the largest source of highly structured, logically rigorous text on the internet.

**23. Deduplication matters more than most people think.** Near-duplicate documents in the training set cause the model to memorize specific phrasings instead of learning general patterns. Aggressive deduplication (up to removing documents with 80% n-gram overlap) consistently improves quality.

**24. Data filtering is where the art is.** Heuristic filters (URL blocklists, language detection, perplexity filtering) remove 80-95% of raw web data. The remaining 5-20% is what you train on. Every lab has different filters, and they guard them carefully because they're a competitive advantage.

**25. Tokenizer failures are real and underappreciated.** BPE can produce non-intuitive tokenizations that hurt specific tasks:
- Numbers: "123456" might tokenize as ["123", "456"] or ["1", "234", "56"] depending on the tokenizer
- Code: Indentation (spaces/tabs) can consume many tokens
- Non-English: Many words take 3-5× more tokens in non-Latin scripts
- Edge case: "\n\n" might be one token or two, affecting output formatting

**26. The vocabulary size tradeoff is real.** $V = 32K$: fast training, some OOV issues. $V = 100K$: slower softmax, better coverage. $V = 250K$ (GPT-4): excellent coverage but slower. Sweet spot: $V \in [32K, 128K]$.

---

## Numerical Precision

**27. bf16 is almost always better than fp16 for training.** bf16 has the same exponent range as fp32 (8 bits) → no overflow. fp16 has only 5 exponent bits → overflows at 65504. The lower precision of bf16 (7-bit mantissa vs 10-bit) rarely matters in practice.

**28. The most dangerous bf16 operation is the attention dot product.** The inner product of two 4096-dim vectors in bf16 accumulates rounding errors proportional to $\sqrt{d}$. For $d = 4096$, the relative error is ~50%. This is mitigated by fp32 accumulation in tensor cores.

**29. Loss scaling is NOT needed for bf16.** Loss scaling (multiplying loss before backward, dividing gradients after) is only needed for fp16 to prevent gradient underflow. bf16 has sufficient exponent range, so loss scaling adds complexity for no benefit.

**30. Gradient accumulation in bf16 is dangerous.** Adding many small bf16 gradients can lose significant precision. Best practice: accumulate gradients in fp32, even if forward/backward are in bf16:

```python
# Correct: accumulate in fp32
optimizer = AdamW(model.parameters())  # Adam state is already fp32 by default
for micro_batch in micro_batches:
    with torch.autocast('cuda', dtype=torch.bfloat16):
        loss = model(micro_batch).loss / n_accumulation
    loss.backward()  # Gradients stored in param.grad (same dtype as param)
# optimizer.step() reads bf16 grads but updates fp32 optimizer state → safe
```

---

## Scaling

**31. The Chinchilla ratio (20 tokens per parameter) is a guideline, not a law.** It's optimal for a fixed compute budget when you can train a new model. In practice, you often have a fixed model and want to train longer → more tokens per parameter is fine (50-200 tok/param is common).

**32. Scaling laws have a hidden constant that matters at small scale.** $L(N) = aN^{-\alpha} + c$. The constant $c$ (irreducible loss) means that at small $N$, the power law doesn't hold well. Don't calibrate scaling laws on small models and extrapolate to large ones.

**33. Scaling laws are measured on pretraining loss, not downstream performance.** The correlation between pretraining loss and downstream task accuracy is strong but not perfect. Some tasks improve smoothly with loss; others have threshold effects.

**34. "Inference-optimal" is different from "training-optimal."** Chinchilla finds the cheapest way to reach a given loss. But if you serve the model a billion times, you want a smaller model trained on more data (same quality, cheaper inference). This is why LLaMA exists.

**35. Computing scaling laws requires ablations that cost real money.** To find the right model size for your compute budget, you need to train 5-10 small models at different sizes. This "tax" is worth it — it prevents you from wasting your entire compute budget on the wrong size.

---

## Post-Training & Alignment

**36. SFT quality matters more than SFT quantity.** 10K high-quality instruction-response pairs can outperform 1M low-quality ones. The model already has the capability from pretraining; SFT just teaches it the output format.

**37. DPO has largely replaced RLHF-PPO in practice.** PPO requires 4 models in memory simultaneously (policy, reference, reward, value). DPO needs 2 (policy, reference). The quality difference is small, the engineering simplification is enormous.

**38. The reference model in DPO matters more than people realize.** If the reference model is the SFT model, DPO learns deltas from SFT. If it's the base model, DPO learns deltas from base. The choice affects stability and the extent of policy change.

**39. Constitutional AI (Anthropic's approach) uses the model to evaluate itself.** Instead of human preference labels, the model critiques its own outputs using a set of principles ("constitution"). This scales better than human labeling but risks encoding the model's own biases.

**40. Safety training and capability training are in tension.** Every safety intervention (refusal training, content filtering) reduces the model's willingness to discuss certain topics → reduces helpfulness on those topics. This is the alignment tax.

**41. Jailbreaks are fundamentally about distribution shift.** The model's safety training covers the distribution of "normal" harmful requests. Adversarial prompts push the input into an out-of-distribution region where the safety features don't activate.

---

## Fine-Tuning

**42. LoRA rank 16 is almost always sufficient.** Higher ranks give marginal improvement. The "low-rank hypothesis" holds remarkably well — task adaptation requires very few dimensions of change.

**43. LoRA works better on attention layers than FFN layers.** Attention layers control information routing (which positions attend to which). FFN layers store knowledge. Task adaptation usually changes routing more than knowledge.

**44. QLoRA's NF4 quantization loses almost nothing for fine-tuning.** The base model weights are "anchors" — small perturbations from the LoRA adapter are what matter. Quantizing the anchors slightly doesn't affect the perturbations.

**45. Fine-tuning on too-small datasets causes "mode collapse."** The model starts producing the same response format for every input. Fix: add a KL penalty against the base model, or use dropout, or just get more data.

**46. Learning rate for fine-tuning should be 10-100× lower than pretraining.** Pretraining LR ~3e-4 → fine-tuning LR ~1e-5 to 3e-5. Too high → catastrophic forgetting. Too low → no learning.

**47. Catastrophic forgetting is real but overblown.** With proper hyperparameters (low LR, short training, weight decay), the model retains most of its pretraining knowledge. The bigger risk is not fine-tuning enough, not fine-tuning too much.

---

## Inference & Deployment

**48. The KV cache, not the model weights, is the memory bottleneck for long-context tasks.** A 7B model is 14GB. Its KV cache at 128K context with GQA is 16GB — already larger than the model itself.

**49. Batching is the single biggest lever for inference cost.** Going from batch 1 to batch 64 can reduce cost-per-token by 50×. This is why high-traffic APIs are much cheaper per token than low-traffic ones.

**50. Speculative decoding helps most when you're not batching.** If you're already running batch 64, the GPU is utilized. Speculative decoding adds complexity for marginal gain. It's a single-user latency optimization, not a throughput optimization.

**51. INT4 quantization is acceptable for most applications.** The quality drop from bf16 → INT4 (with good quantization like AWQ) is typically <2% on benchmarks. Users rarely notice. The 4× memory reduction is massive.

**52. Time-to-first-token (TTFT) is usually the user-facing latency that matters.** Users notice the initial wait, not the per-token speed (streaming makes per-token delay invisible). Optimize prefill latency, not just decode throughput.

**53. Continuous batching is now standard.** vLLM, TensorRT-LLM, SGLang all implement it. If you're doing static batching, you're throwing away 50%+ of your GPU budget.

---

## Evaluation & Debugging

**54. The single most important debugging technique: overfit one batch.** If the model can't memorize 4 sequences in 300 steps, something is fundamentally broken. This catches: data loading bugs, loss function errors, masking issues, architecture bugs. Run it first, always.

**55. Initial loss ≈ log(V) is your first sanity check.** If it's off by more than 10%, investigate immediately.

**56. Benchmarks are contaminated. Always assume some contamination.** Train your own held-out evaluation set from data created after the training cutoff if you need reliable measurements.

**57. Human evaluation has higher variance than you think.** Inter-annotator agreement for "helpfulness" is typically κ ≈ 0.4-0.6. You need many samples to detect real differences between models.

**58. LLM-as-Judge prefers its own outputs.** GPT-4 rating GPT-4 outputs vs Claude outputs is biased toward GPT-4. Use cross-model evaluation or diverse judge panels.

---

## Research Taste & Thinking

**59. Read papers for the negative results section.** Positive results are published; negative results teach you where not to look. The "limitations" and "failed experiments" sections are the most educational.

**60. The "bitter lesson" has limits.** Scale + data beats hand-engineering... on average. But specific domain knowledge still matters for edge cases, efficiency, and safety. Nobody at Anthropic believes "just scale it."

**61. Compute-efficient experimentation is the meta-skill.** The best researchers get signal from small experiments. If you need to train a 70B model to test your idea, your idea is too expensive to iterate on. Test at 125M-1B first.

**62. Most published "improvements" don't replicate at scale.** A method that gains 2% at 125M parameters often gives 0.1% at 7B. Scale is the great equalizer — simple methods catch up.

**63. The most impactful research is often engineering, not algorithms.** FlashAttention is "just" a systems optimization. But it enabled everything that followed (long context, efficient training, cheap inference). Same for PagedAttention, continuous batching, and GQA.

---

## Career & Interview

**64. Top labs care more about "can you think clearly about a problem" than "do you know the answer."** In interviews, articulate your reasoning. Say "I think X because Y, and if Y is wrong then Z would change." Don't just recite facts.

**65. The three skills that matter most: (1) implement things from scratch, (2) debug when things break, (3) design experiments that give signal.** Everything else (math, paper reading, coding speed) supports these three.

**66. Know the difference between "I read about it" and "I implemented it."** Interviewers can tell. If you've implemented attention from scratch, you can answer questions about it that paper-readers can't.

**67. Have an opinion about open questions.** "Why does in-context learning work?" "Is scaling sufficient for AGI?" "Should we use RLHF or DPO?" Have a stance, support it, and be willing to update.

**68. Anthropic cares deeply about alignment and safety.** If you're interviewing there, understand Constitutional AI, interpretability, and the core alignment problem. "How do we build AI that's helpful, honest, and harmless?" is not a tagline — it's the research agenda.

**69. OpenAI is more applied-research oriented.** They care about capabilities AND deployment. Know about scaling laws, training efficiency, and how to ship models to millions of users.

**70. Google DeepMind values theoretical depth.** They want you to derive things from first principles, understand the math behind methods, and propose novel approaches grounded in theory.

---

## Meta-Observations

**71. Transformers are feature extractors that happen to be sequence models.** The self-attention mechanism is a set operation — it's permutation equivariant. The sequence structure comes entirely from positional encodings. This is why transformers generalize to images, proteins, and other non-sequence data.

**72. Next-token prediction is unreasonably effective because language encodes world knowledge.** Text describes causal relationships, physical processes, social dynamics, and logical arguments. Predicting the next word requires modeling all of these. The objective is simple; the required knowledge is not.

**73. The gap between "open-source" and frontier models is largely data + alignment.** The architecture and training code are known. What's secret is: (a) the exact data mixture and filtering pipeline, (b) the alignment recipe and preference data, (c) the evaluation suite and iteration process.

**74. The field moves fast enough that papers from 18 months ago may be obsolete.** Don't build on old methods without checking if they've been superseded. Read surveys, follow key researchers on social media, and check "related work" sections of recent papers.

**75. The hardest part of building an LLM is not the model — it's the infrastructure.** Distributed training across thousands of GPUs, fault tolerance, data pipelines at petabyte scale, evaluation suites, deployment systems — the "glue code" is 90% of the work.
