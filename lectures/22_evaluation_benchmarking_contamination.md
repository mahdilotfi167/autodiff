# 22. Evaluation, Benchmarking & Contamination: Measuring What Matters

## Motivation

"If you can't measure it, you can't improve it." But in LLMs, **what you measure shapes what you build** — and most benchmarks measure the wrong thing. This lecture covers the art and science of evaluation: how to measure model quality, how benchmarks lie, how contamination corrupts results, and what evaluation at Anthropic/OpenAI/DeepMind actually looks like behind closed doors.

---

## 22.1 The Evaluation Stack

LLM evaluation operates at multiple levels:

| Level | What it measures | Examples | Trust level |
|---|---|---|---|
| **Perplexity** | Next-token prediction quality | Eval set PPL | High (but narrow) |
| **Academic benchmarks** | Specific capabilities | MMLU, HumanEval, GSM8K | Medium (gameable) |
| **Arena / head-to-head** | Human preference | Chatbot Arena, LMSYS | High (but expensive) |
| **Task-specific evals** | Real-world utility | Internal eval suites | Highest (but private) |
| **Red team / safety** | Failure modes | Jailbreak success rate | Critical |

### Why perplexity isn't enough

Perplexity measures the geometric mean surprise of the model on held-out text:

$$\text{PPL} = \exp\left(-\frac{1}{T}\sum_{t=1}^{T} \log P(x_t | x_{<t})\right)$$

**What it captures**: raw language modeling ability, fluency, factual recall.

**What it misses**:
- **Instruction following** — a model can have great PPL but ignore user requests
- **Reasoning** — predicting the next token in a proof ≠ being able to prove things
- **Safety** — a model that outputs harmful text fluently has great PPL
- **Calibration** — PPL doesn't measure whether the model "knows what it knows"

**The key insight**: Perplexity correlates with downstream task performance... up to a point. Beyond ~3.0 PPL on standard text, improvements in PPL give diminishing returns on tasks. Post-training quality dominates.

---

## 22.2 Academic Benchmarks: How They Work and How They Fail

### MMLU (Massive Multitask Language Understanding)

- 57 subjects, multiple choice (A/B/C/D)
- Measures: broad factual knowledge + reasoning
- **Scoring**: percentage of correct answers (chance = 25%)

**Pro tip**: There are two ways to evaluate MMLU, and they give **very different** numbers:

1. **Likelihood-based**: Compare $P(\text{"A"} | \text{question})$ vs $P(\text{"B"} | \text{question})$ etc. Pick the highest probability letter.
2. **Generation-based**: Have the model generate an answer, extract the letter.

Likelihood-based is more reliable but can be gamed. Generation-based depends on the prompt format. **Always report which method you used.**

### HumanEval / MBPP (Code Generation)

- Generate code for a function, run unit tests
- Metric: **pass@k** — probability that at least 1 of k samples passes all tests

$$\text{pass@k} = 1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}$$

where $n$ = total samples, $c$ = correct samples. This is the **unbiased estimator** (not just "top k accuracy").

**Common mistake**: Reporting pass@1 with greedy decoding. This is **not** the same as pass@1 estimated from multiple samples. The unbiased estimator samples 200 completions and computes the formula above.

### GSM8K (Grade School Math)

- 8.5K grade school math word problems
- Evaluate final numerical answer (exact match)

**The dirty secret**: Models that score well on GSM8K have often **memorized the format**, not learned to reason. Evidence:
- Performance drops dramatically on **rephrased** versions of the same problems
- Performance drops when numbers are changed (even when the same algorithm applies)
- Chain-of-thought helps not because the model reasons, but because it constrains the output format

### The "benchmark is saturated" problem

When a benchmark is widely used:

1. Models are optimized to do well on it (directly or indirectly)
2. Test set patterns leak into training data
3. The benchmark stops distinguishing between models
4. Example: MMLU scores went from ~30% (GPT-3) to ~90% in 3 years — is the task solved or the benchmark broken?

**Real evaluation at top labs**: Internal eval suites that are constantly refreshed, never published, and specifically test for failure modes discovered in the previous model generation.

---

## 22.3 Contamination: The Elephant in the Room

### What is contamination?

Benchmark contamination: when test data (or very similar data) appears in the training set. The model "memorizes" the answer instead of "reasoning" to it.

### How it happens

1. **Direct leakage**: The benchmark's test set is literally in the training data (it's on the internet!)
2. **Indirect leakage**: Training data contains discussions *about* the benchmark (e.g., blog posts solving MMLU questions)
3. **Format leakage**: Even if exact questions aren't in training data, the *format* and *style* of questions are
4. **Temporal leakage**: Benchmark created from data before cutoff, but similar problems exist in training corpus

### Detection methods

**N-gram overlap**: Check if n-grams from test questions appear in training data.

```python
def check_contamination(test_example: str, training_data: set, n: int = 13):
    """Check if any n-gram from test appears in training."""
    words = test_example.split()
    for i in range(len(words) - n + 1):
        ngram = ' '.join(words[i:i+n])
        if ngram in training_data:
            return True, ngram
    return False, None
```

**Canary strings**: Insert unique strings in test data. If the model can complete them, training set contamination is confirmed.

**Performance on rephrased variants**: If performance drops sharply when questions are rephrased (same content, different wording), the model memorized surface patterns.

**Temporal holdout**: Evaluate on problems created *after* the training data cutoff. If performance drops more than expected, earlier benchmarks were contaminated.

### The GPT-4 contamination study

OpenAI's GPT-4 technical report included a contamination analysis:
- For MMLU: ~1% of questions had 70%+ overlap with training data
- Contaminated questions showed only +1-3% accuracy boost
- But: this was self-reported — independent analysis suggested higher contamination rates

**The uncomfortable truth**: Every model trained on internet data is contaminated to some degree. The question is how much, and whether the benchmark still measures what we think.

---

## 22.4 Human Evaluation: The Gold Standard (With Caveats)

### Chatbot Arena (LMSYS)

- Users chat with two anonymous models, pick the preferred response
- Rankings computed via Elo/Bradley-Terry model
- As of now, the most trusted public evaluation

**Why it works**: Real users, real tasks, no optimization target for model developers.

**Why it's imperfect**:
- **Length bias**: Users prefer longer responses (even when shorter is better)
- **Format bias**: Markdown, bullet points, emojis boost ratings
- **Sycophancy bias**: Users prefer models that agree with them
- **Sampling bias**: Arena users are disproportionately tech-savvy
- **Style vs substance**: A confident wrong answer often beats a hedging correct one

### Internal human evaluation methodology

How top labs actually do it:

1. **Task-specific rubrics**: Each capability (coding, math, creative writing, safety) has a separate rubric with specific criteria
2. **Blind comparison**: Evaluators see two responses without knowing which model generated which
3. **Inter-annotator agreement**: For each task, compute Cohen's κ. If κ < 0.4, the rubric is ambiguous — fix it before drawing conclusions
4. **Calibration set**: Include responses from a known good model and a known bad model to verify evaluators are consistent
5. **Statistical significance**: Don't declare wins without proper confidence intervals (typically need 100+ comparisons per task)

### The inter-annotator agreement problem

For subjective tasks (creative writing, helpfulness), human annotators often disagree:

$$\kappa = \frac{p_o - p_e}{1 - p_e}$$

where $p_o$ is observed agreement, $p_e$ is chance agreement.

| κ value | Strength | Implication |
|---|---|---|
| > 0.8 | Near-perfect agreement | Task is well-defined (e.g., math correctness) |
| 0.6-0.8 | Substantial | OK for evaluation |
| 0.4-0.6 | Moderate | Results should be interpreted cautiously |
| < 0.4 | Low | The rubric is broken or the task is too subjective |

**Implication**: For tasks with low agreement, **the variance in human evaluation is larger than the difference between models**. You need many more samples to detect real differences.

---

## 22.5 LLM-as-Judge

Using a stronger model (e.g., GPT-4) to evaluate a weaker model's outputs.

### How it works

```
Prompt: "Rate the following response on a scale of 1-10..."
- Or: "Which response is better, A or B?"
- Judge model provides rating/ranking
```

### Failure modes (and they're serious):

1. **Self-preference bias**: Models prefer outputs that match their own style. GPT-4-as-judge systematically prefers GPT-4 outputs.
2. **Position bias**: In pairwise comparison, some models prefer the first response, others the second. Always randomize order and average.
3. **Verbosity bias**: Judge models prefer longer responses.
4. **Authority bias**: If the response "sounds confident," judges rate it higher regardless of correctness.
5. **Circularity**: If you use GPT-4 to generate training data AND evaluate → you're measuring self-consistency, not quality.

### Mitigation for LLM-as-Judge:

- Use multiple judge models and check agreement
- Include "ground truth" examples to calibrate the judge
- Use structured rubrics (not open-ended "rate quality")
- Report position-swapped results to quantify bias
- For factual tasks, prefer automated metrics over LLM judges

---

## 22.6 Evaluation Metrics That Actually Matter

### For language modeling:

| Metric | Formula | What it tells you |
|---|---|---|
| Perplexity | $\exp(-\frac{1}{T}\sum \log P(x_t))$ | Raw prediction quality |
| Bits-per-byte | $\frac{\text{Cross-entropy}}{\text{bytes per token}} \times \frac{1}{\log 2}$ | Vocabulary-independent comparison |
| Token accuracy | $\frac{\text{correct predictions}}{T}$ | How often is greedy decode correct |

**Key insight**: Bits-per-byte is the only fair way to compare models with different tokenizers. A token-level perplexity of 5.0 with a 50K vocabulary is not comparable to 5.0 with a 32K vocabulary.

### For code generation:

- **pass@k**: Unbiased estimator (see 22.2)
- **Functional correctness**: Do the outputs actually work?
- **Repo-level evaluation**: Can the model modify a real codebase? (Much harder than function-level)

### For reasoning:

- **Chain-of-thought correctness**: Is the intermediate reasoning valid, even if the answer is right?
- **Sensitivity analysis**: Change numbers/names — does the answer change appropriately?
- **Negation testing**: "Which of these is NOT true?" (Models fail disproportionately on negation)

### For safety:

- **Refusal rate on benign queries**: How often does the model refuse legitimate requests? (Type I error)
- **Compliance rate on harmful queries**: How often does the model comply with harmful requests? (Type II error)
- **The tradeoff**: Reducing compliance on harmful queries almost always increases refusal on benign queries. The art is finding the right threshold.

---

## 22.7 Building Your Own Evaluation Suite

### Principles:

1. **Test what you care about, not what's easy to measure**
2. **Include adversarial examples**: If you only test the happy path, you'll miss failure modes
3. **Version your eval set**: As you fix failures, add them to eval. Never delete old tests.
4. **Track metrics over time**: Plot eval scores across checkpoints and training runs
5. **Separate capability from safety**: A model can be highly capable AND unsafe

### Example evaluation config:

```python
eval_suite = {
    "knowledge": {
        "datasets": ["mmlu_val", "triviaqa", "naturalquestions"],
        "metric": "accuracy",
        "n_few_shot": 5,
    },
    "reasoning": {
        "datasets": ["gsm8k_rephrased", "math_500", "arc_challenge"],
        "metric": "exact_match",
        "use_cot": True,
    },
    "coding": {
        "datasets": ["humaneval", "mbpp", "swe_bench_lite"],
        "metric": "pass@1",
        "n_samples": 20,  # For unbiased pass@k estimates
    },
    "instruction_following": {
        "datasets": ["ifeval", "mt_bench"],
        "metric": "constraint_satisfaction_rate",
    },
    "safety": {
        "datasets": ["harmful_requests_v3", "benign_refusal_check"],
        "metric": "refusal_rate",
        "report_both_types": True,  # Type I and Type II
    },
    "contamination_check": {
        "method": "13gram_overlap",
        "report_for_each_dataset": True,
    },
}
```

---

## 22.8 The Meta-Lesson: Goodhart's Law

> "When a measure becomes a target, it ceases to be a good measure."

This is the fundamental problem of LLM evaluation:

- RLHF optimizes for human preference → model becomes sycophantic
- Optimizing for MMLU → model memorizes test patterns
- Optimizing for safety benchmarks → model refuses too aggressively
- Optimizing for Arena score → model generates verbose, formatted responses

**What top labs actually do**: Maintain a large, private evaluation suite that is **never** used as a training signal. The eval suite is updated whenever a new failure mode is discovered. The goal is to **detect** problems, not to **optimize** for a score.

The best evaluation is a red team that actively tries to break the model, combined with automated metrics that catch regressions. Neither alone is sufficient.
