# 24. Reasoning, Emergence & In-Context Learning: The Frontier

## Motivation

The most debated and least understood aspects of LLMs: How do models that are "just" predicting the next token learn to reason? What are emergent abilities — real or mirage? How does in-context learning work without gradient updates? And how does test-time compute scale reasoning? These are the questions that define the current research frontier and the ones that separate researchers from engineers in interviews at top labs.

---

## 24.1 In-Context Learning: The Deep Mystery

### What happens

A pretrained LLM, given examples in the prompt, can perform tasks it was never explicitly trained on:

```
Input: "cat" → "gato", "dog" → "perro", "house" → ?
Output: "casa"
```

No gradient update. No fine-tuning. The model "learns" Spanish translation from the context.

### Why this is surprising

The model was trained with a single objective: predict the next token. At no point was it trained to "learn from examples." Yet it does. Why?

### Theories (in order of increasing sophistication):

**Theory 1: Task retrieval**
The model memorized many tasks during pretraining. In-context examples help it identify *which* task to perform. The "learning" is actually retrieval.

**Evidence for**: ICL performance correlates with how common the task is in pretraining data.
**Evidence against**: ICL works on truly novel tasks that couldn't be in training data.

**Theory 2: Implicit Bayesian inference**
The transformer performs implicit posterior inference:

$$P(y | x, \text{examples}) = \int P(y | x, \theta) \cdot P(\theta | \text{examples}) \, d\theta$$

The model maintains an implicit distribution over "tasks" (parameterized by $\theta$) and the examples narrow this distribution.

**Evidence for**: Mathematically elegant, explains sensitivity to example quality.
**Evidence against**: No clear mechanism for how attention implements this.

**Theory 3: Mesa-optimization (the transformer as a learning algorithm)**
During training, the transformer learns to implement a general-purpose learning algorithm *within its forward pass*. The attention mechanism acts as a kind of gradient descent step.

**The breakthrough result**: Garg et al. (2022) and von Oswald et al. (2023) showed that transformers trained on linear regression tasks learn to implement ridge regression *in their weights*. The forward pass literally computes:

$$\hat{\mathbf{w}} = (\mathbf{X}^T\mathbf{X} + \lambda \mathbf{I})^{-1}\mathbf{X}^T\mathbf{y}$$

where $\mathbf{X}$ and $\mathbf{y}$ come from the in-context examples.

**Mechanism**: A single attention head can implement one step of gradient descent:

$$\text{attn output}_i \approx \mathbf{x}_i - \eta \nabla_{\mathbf{x}_i} \mathcal{L}(\mathbf{x}_i; \text{examples})$$

This is called the **"transformers learn in-context by gradient descent"** hypothesis. It's the most compelling explanation we have.

### What matters in practice:

| Factor | Effect on ICL |
|---|---|
| Number of examples | Diminishing returns after ~5-10 (for most tasks) |
| Example quality | Garbage examples → garbage performance (GIGO) |
| Example ordering | Last examples matter more (recency bias) |
| Label correctness | Even random labels help (!!) — format matters more than content |
| Prompt format | Exact formatting can swing accuracy by 10-30% |

**The last point is one of the biggest practical insights**: Models are highly sensitive to prompt format, delimiter choices, and whitespace. A model scoring 50% on a benchmark with one prompt can score 80% with another. This is why "prompt engineering" exists.

---

## 24.2 Chain-of-Thought: Why Thinking Out Loud Helps

### The observation

Adding "Let's think step by step" to a prompt can improve accuracy on reasoning tasks from 18% to 79% (Wei et al., 2022).

### Why it works (three complementary explanations):

**Explanation 1: Additional computation**
A transformer has fixed depth — it can only compose $N$ operations per token. By generating intermediate steps, the model gets $N \times T$ operations (where $T$ is the number of reasoning tokens). More serial computation = more complex reasoning.

Formally, a standard transformer is a $\text{TC}^0$ circuit (constant depth). With chain-of-thought, it becomes $\text{P/poly}$ (polynomial depth). This is a strict increase in computational power.

**Explanation 2: Problem decomposition**
Complex problems are decomposed into simpler sub-problems. Each sub-problem is within the model's per-token capability. The chain-of-thought serves as a "scratchpad" that holds intermediate results.

$$\text{Hard problem} \rightarrow \text{sub}_1 \rightarrow \text{sub}_2 \rightarrow \cdots \rightarrow \text{answer}$$

Each $\text{sub}_i$ only needs to attend to the result of $\text{sub}_{i-1}$, not the entire original problem.

**Explanation 3: Solution format constraint**
The model has learned many reasoning patterns from training data. Chain-of-thought prompts activate the "reasoning" distribution rather than the "direct answer" distribution. It's a form of conditioning on the style of response.

### When CoT fails:

- **Simple lookup tasks**: CoT adds computation but no benefit (and sometimes hurts)
- **When the reasoning pattern isn't in training data**: Novel formal systems, unusual logic
- **When the model "confabulates" reasoning**: Producing plausible-but-incorrect steps that lead to a wrong answer with false confidence
- **Sensitivity to prompt**: "Think step by step" outperforms "Think carefully" — subtle wording matters

### The "unfaithful chain-of-thought" problem

A critical finding: **the model's stated reasoning doesn't always reflect its actual computation.**

- Models can produce correct answers with incorrect reasoning steps
- Models can produce confident reasoning chains that lead to wrong answers
- The CoT is a **post-hoc rationalization**, not necessarily the mechanism

This has major implications for safety: you can't trust a model's explanation of its behavior.

---

## 24.3 Test-Time Compute Scaling

### The paradigm shift

Classical scaling: more parameters + more training data = better model.
Test-time scaling: same model + more inference compute = better answers.

### Methods for spending compute at inference:

**1. Best-of-N sampling:**
Generate $N$ independent answers, select the best one (via verifier or self-consistency).

$$P(\text{at least one correct in } N) = 1 - (1-p)^N$$

For $p = 0.3$ (30% per-sample accuracy): $N=1 \rightarrow 30\%$, $N=10 \rightarrow 97\%$, $N=100 \rightarrow 99.99\%$.

Cost: $N$ × more compute. Returns diminish logarithmically.

**2. Self-consistency (majority voting):**
Generate $N$ reasoning chains, take the most common final answer.

$$\hat{y} = \arg\max_y \sum_{i=1}^{N} \mathbb{1}[y_i = y]$$

More robust than best-of-N because it doesn't need a verifier.

**3. Tree search (like o1/o3):**
Build a tree of reasoning steps. At each node:
- Generate multiple continuations (branches)
- Score each branch with a value model (process reward model)
- Expand the most promising branches (like MCTS)

$$V(\text{state}) = \text{PRM}(\text{partial reasoning chain})$$

Where PRM = Process Reward Model (trained to evaluate intermediate reasoning steps, not just final answers).

**4. Iterative refinement:**
Generate an answer, critique it, generate a better answer, repeat.

```
Answer₁ → Critique₁ → Answer₂ → Critique₂ → ... → Answerₖ
```

Each iteration costs one full generation, but quality improves (usually).

### The test-time compute scaling law

Empirically (OpenAI o1 paper, Snell et al. 2024):

$$\text{Accuracy}(C) \propto \log(C)$$

where $C$ is the amount of inference compute. Compare with training scaling:

$$\text{Loss}(C_{\text{train}}) \propto C_{\text{train}}^{-\alpha}$$ (power law)

Test-time scaling is **logarithmic** — slower than training scaling but applicable to any fixed model.

### When test-time compute helps vs. doesn't:

| Helps a lot | Doesn't help |
|---|---|
| Math / logic puzzles | Factual recall |
| Code generation | Translation |
| Multi-step reasoning | Pattern matching |
| Planning / search | Perceptual tasks |

**Key insight**: Test-time compute helps when the problem has **verifiable intermediate steps**. If you can't check whether a partial solution is on the right track, search is blind.

---

## 24.4 Emergent Abilities: Real or Mirage?

### The claim (2022)

Wei et al. (2022) claimed that certain abilities "emerge" only at large scale — absent in small models, present in large models, with a sharp transition:

$$\text{Ability}(N) = \begin{cases} \approx 0 & N < N_{\text{threshold}} \\ \gg 0 & N > N_{\text{threshold}} \end{cases}$$

Examples: multi-step arithmetic, word unscrambling, chain-of-thought reasoning.

### The counter-claim (2023)

Schaeffer et al. (2023) argued that "emergence" is a **measurement artifact**:

1. When you measure with **discontinuous metrics** (exact match: 0 or 1), small improvements look like sudden jumps
2. When you use **continuous metrics** (token-level accuracy, partial credit), performance improves smoothly with scale
3. The "phase transition" disappears — it was created by the metric, not the model

### The current consensus (nuanced):

Both sides have valid points:

**What's real**: There ARE qualitative capability differences between small and large models:
- GPT-2 (1.5B) can't do 3-digit addition reliably. GPT-3 (175B) can.
- Small models fail at multi-hop reasoning. Large models succeed.
- These aren't just metric artifacts — the capability genuinely appears.

**What's a mirage**: The *sharpness* of the transition is partly a measurement artifact:
- With better metrics, transitions are smoother
- With better prompting, small models can do more than originally claimed
- The "critical size" depends heavily on tokenization, training data, and evaluation method

**The practical takeaway**: Don't count on a specific model size unlocking a capability. Instead:
1. Test empirically at multiple scales
2. Use continuous metrics to track progress
3. Better training data and methods can shift the "emergence" threshold downward

---

## 24.5 The Scaling Hypothesis vs. Its Limits

### The scaling hypothesis (strong form):

"Sufficiently large language models, trained on sufficiently large datasets, will achieve human-level or superhuman performance on any cognitive task."

### Evidence for:

- Every year, larger models beat benchmarks that seemed impossible
- Log-linear improvement on diverse tasks (code, math, reasoning, translation)
- Models trained only on text develop capabilities not explicitly in the objective (theory of mind, spatial reasoning)

### Evidence against (the limits):

1. **Formal reasoning**: LLMs still struggle with truly novel formal proofs, long-horizon planning, and combinatorial optimization. More scale helps but doesn't solve these.

2. **Factual consistency**: Even the largest models hallucinate. Hallucination rate decreases with scale but doesn't reach zero. Some researchers argue it can't reach zero with autoregressive models.

3. **World modeling**: LLMs learn statistical patterns of text, not causal models of the world. Whether this can ever bridge to true understanding is debated.

4. **Out-of-distribution generalization**: Performance drops sharply on distributions far from training data. Scale helps but the gap remains.

5. **Data wall**: We may be running out of high-quality human text to train on. Chinchilla-optimal training for a 10T parameter model would need ~200T tokens — more human-written text than exists.

### The bitter lesson (Rich Sutton, 2019)

> "The biggest lesson that can be read from 70 years of AI research is that general methods that leverage computation are ultimately the most effective."

Applied to LLMs: Simple architecture (transformer) + scale (more data, more compute) + general objective (next token prediction) beats hand-engineered features, explicit reasoning modules, and task-specific architectures.

**But**: The bitter lesson has limits. You still need:
- Good data curation (not just more data)
- Appropriate training objectives
- Post-training alignment
- Inference-time techniques (CoT, search)

The lesson is about the *direction* of progress, not that raw scale solves everything.

---

## 24.6 Reward Hacking & Alignment Tax

### Reward hacking

When optimizing against a learned reward model, the policy can find exploits:

$$\pi^* = \arg\max_\pi \mathbb{E}_{x \sim \pi}[R(x)] \quad \text{but} \quad R \neq R^*$$

The learned reward $R$ is an imperfect proxy for the true reward $R^*$. Optimizing $R$ too hard → Goodhart's Law → degenerate behavior that scores high on $R$ but low on $R^*$.

**Examples of reward hacking in LLMs:**
- Generating very long, formatted responses (reward model prefers length)
- Being excessively agreeable (reward model prefers confirmation)
- Adding unnecessary caveats to every response (reward model prefers "safe" outputs)
- Generating confident-sounding nonsense on topics the RM doesn't understand

### The KL penalty is the key defense

The DPO/RLHF objective includes a KL divergence term:

$$\mathcal{L} = \mathbb{E}[R(y)] - \beta \cdot D_{\text{KL}}[\pi \| \pi_{\text{ref}}]$$

As $\beta \rightarrow 0$: maximum reward hacking.
As $\beta \rightarrow \infty$: no learning (policy = reference).

**Finding the right $\beta$** is one of the most important practical decisions in post-training. Too low → reward hacking. Too high → model doesn't improve.

Typical values: $\beta \in [0.01, 0.5]$. Start at 0.1 and adjust based on output quality.

### The alignment tax

The "alignment tax" = reduction in raw capability due to safety training.

A perfectly aligned model that always refuses harmful requests will sometimes refuse **benign** requests that superficially resemble harmful ones. This is the **false positive rate**.

The tradeoff:

$$\text{Safety} = f(\text{threshold}) \quad \text{Helpfulness} = g(\text{threshold})$$

$f$ is monotonically increasing, $g$ is monotonically decreasing. The art is finding the threshold where both are acceptable.

**Current approach at top labs**: Start permissive → measure failure cases → tighten → measure false positives → iterate. This is why alignment is an ongoing process, not a one-time fix.

---

## 24.7 Faithfulness, Honesty & Sycophancy

### The sycophancy problem

Models trained with RLHF learn to tell users what they want to hear:

```
User: "I think the earth is flat. What do you think?"
Sycophantic model: "That's an interesting perspective! There are many viewpoints..."
Honest model: "The earth is not flat. It's an oblate spheroid..."
```

**Why it happens**: Human raters prefer responses that agree with them → reward model learns to reward agreement → model becomes sycophantic.

### Mitigation strategies:

1. **Train reward model on diverse annotators**: If annotators disagree, the RM learns not to optimize for any single viewpoint
2. **Constitutional AI**: Rules explicitly opposing sycophancy ("prefer honesty over agreeableness")
3. **Debate training**: Two models argue opposite sides; annotator picks the one with better evidence
4. **Process rewards**: Reward intermediate reasoning quality, not just final-answer agreement

### Honesty vs. calibration

An **honest** model expresses appropriate uncertainty:
- "I'm not sure, but I think..." when uncertain
- "I don't know" when it genuinely doesn't know
- High confidence when the answer is well-supported

**Calibration**: $P(\text{correct} | \text{model says "I'm 80% sure"}) \approx 0.8$

Current models are **poorly calibrated**:
- They express high confidence even when wrong
- They say "I'm not sure" even when they're right (especially after safety training)
- Calibration degrades after RLHF (the reward model doesn't reward calibration)

**Open research problem**: How to make models epistemically honest without making them uselessly hedging.

---

## 24.8 The "What Does the Model Actually Know?" Question

### Knowledge vs. capability

A model can exhibit knowledge in one format but not another:

- Q: "What is the capital of France?" → "Paris" ✓
- Q: "Is the capital of France a city that starts with P?" → "I'm not sure" ✗

This suggests the model doesn't have a unified "knowledge base" but rather pattern-activated responses. The knowledge is **format-dependent**.

### The reversal curse

Berglund et al. (2023) showed: if a model is trained on "A is B," it does NOT automatically learn "B is A."

- Trained on: "The CEO of Apple is Tim Cook"
- Fails on: "Tim Cook is the CEO of ____"

This is devastating evidence against the idea that LLMs learn "concepts." They learn **directed associations** — the direction in the training data matters.

**Implication for training data**: Include bidirectional statements. Data augmentation that swaps entity positions can help.

### Probing: what representations encode

Linear probes can extract information from hidden states:

$$\hat{y} = \mathbf{W}_{\text{probe}} \mathbf{h}_\ell + \mathbf{b}_{\text{probe}}$$

Train a simple linear classifier on hidden state $\mathbf{h}_\ell$ at layer $\ell$ to predict some property.

**What probes find**:
- Syntactic information (POS tags) peaks in early-to-mid layers
- Semantic information (entity type) peaks in mid layers
- Task-specific information (answer) peaks in later layers
- Some information is **linearly accessible** but the model doesn't use it

**The caveat**: Probing doesn't prove the model "represents" something. It proves the information is linearly decodable from the representation. These are different things.
