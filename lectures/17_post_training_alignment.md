# 17. Post-Training: Alignment, RLHF, DPO, and Making Models Useful

## Motivation

A pretrained GPT model is a powerful text generator — but it's not a useful assistant. It will happily generate hate speech, leak training data, produce incoherent rambling, or refuse to follow instructions. **Post-training** is the process of transforming a raw pretrained model into one that is helpful, harmless, and honest.

This document covers the complete post-training pipeline: supervised fine-tuning (SFT), reinforcement learning from human feedback (RLHF), direct preference optimization (DPO), and other alignment techniques. We derive the key algorithms mathematically, explain why they work, and analyze their trade-offs.

---

## 17.1 The Alignment Problem: Why Pretraining Is Not Enough

### What pretraining optimizes for

The pretrained model minimizes:

$$\mathcal{L}_{\text{pretrain}} = -\mathbb{E}_{x \sim \mathcal{D}_{\text{web}}}[\log P_\theta(x)]$$

This objective teaches the model to **predict text from the internet**. But internet text includes:
- Spam, misinformation, toxic content
- All styles: formal, informal, argumentative, poetic
- Questions **and** wrong answers
- Instructions that are followed **and** instructions that are ignored

The model learns to generate text that is **statistically likely** — not text that is helpful, accurate, or safe.

### The gap

| Pretrained behavior | Desired behavior |
|---|---|
| Complete any text | Follow instructions |
| Mimic internet style | Be helpful and clear |
| Generate anything statistically likely | Refuse harmful requests |
| Produce any plausible continuation | Give accurate information |
| No concept of conversation | Maintain multi-turn dialogue |

### The post-training pipeline

```
Pretrained model (base model)
    │
    ▼
[Supervised Fine-Tuning (SFT)]  — learn to follow instructions
    │
    ▼
SFT model
    │
    ▼
[Preference Optimization (RLHF/DPO)]  — learn human preferences
    │
    ▼
Aligned model (chat model)
    │
    ▼
[Optional: Safety tuning, tool use, etc.]
    │
    ▼
Deployed model
```

---

## 17.2 Supervised Fine-Tuning (SFT): Teaching the Format

### What SFT does

SFT fine-tunes the pretrained model on a curated dataset of **(instruction, response)** pairs:

$$\mathcal{D}_{\text{SFT}} = \{(\text{instruction}_i, \text{response}_i)\}_{i=1}^M$$

The model learns to generate high-quality responses given instructions.

### The training objective

Same as pretraining — cross-entropy on next-token prediction — but only on the **response** tokens:

$$\mathcal{L}_{\text{SFT}} = -\frac{1}{|\text{response}|} \sum_{t \in \text{response positions}} \log P_\theta(t | \text{instruction}, \text{response}_{<t})$$

**Key**: the loss is NOT computed on instruction tokens. The model sees the instruction as context but is only trained to generate the response. This is called **causal language modeling with instruction masking**.

```python
# Constructing the loss mask
input_ids  = [inst_tokens...] + [response_tokens...]
loss_mask  = [0, 0, ..., 0,    1, 1, ..., 1, 1]  # 1 only on response
labels     = input_ids shifted by 1

loss = F.cross_entropy(logits, labels, reduction='none')
loss = (loss * loss_mask).sum() / loss_mask.sum()
```

### SFT dataset characteristics

| Property | Typical range | Notes |
|---|---|---|
| Size | 10K - 1M examples | Much smaller than pretraining data |
| Quality | Very high | Human-written or carefully curated |
| Diversity | Broad task coverage | Coding, math, writing, analysis, QA |
| Format | Instruction → Response | Often multi-turn conversations |

### Examples of SFT data

```json
{
  "instruction": "Explain photosynthesis in simple terms.",
  "response": "Photosynthesis is the process plants use to convert sunlight into food. The plant's leaves contain chlorophyll, a green pigment that absorbs sunlight. Using this energy, the plant combines carbon dioxide from the air with water from the soil to produce glucose (sugar) and oxygen. The glucose provides energy for the plant to grow, while the oxygen is released into the air for us to breathe."
}
```

### Effect of SFT

After SFT, the model:
- Follows instructions instead of just completing text
- Uses an appropriate response style (helpful, clear, structured)
- Understands conversational format (multi-turn)
- Remains mostly as capable as the base model on knowledge and reasoning

**BUT**: SFT teaches the model to generate responses that **look like** expert responses. It doesn't teach the model **which** response is better among alternatives. This is where preference optimization enters.

---

## 17.3 Reward Modeling: Quantifying Human Preferences

### The comparison data

Human annotators are given a prompt and two (or more) model responses, and they rank them:

$$\mathcal{D}_{\text{pref}} = \{(\text{prompt}_i, y_w^{(i)}, y_l^{(i)})\}_{i=1}^K$$

where $y_w$ is the **preferred** (winning) response and $y_l$ is the **less preferred** (losing) response.

### The Bradley-Terry model

We model the probability that response $y_1$ is preferred over $y_2$ as:

$$P(y_1 \succ y_2 | x) = \sigma(r(x, y_1) - r(x, y_2))$$

where $\sigma$ is the sigmoid function and $r(x, y)$ is a **reward function** that scores how good response $y$ is for prompt $x$.

### Training the reward model

The reward model $r_\phi(x, y)$ is typically a transformer (often initialized from the SFT model) with the final unembedding layer replaced by a scalar head:

```
[Transformer] → hidden state of last token → [Linear(d, 1)] → scalar reward
```

Training objective (maximize log-likelihood of the comparison data):

$$\mathcal{L}_{\text{RM}} = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}_{\text{pref}}} \left[\log \sigma(r_\phi(x, y_w) - r_\phi(x, y_l))\right]$$

### Gradient of the reward model loss

$$\nabla_\phi \mathcal{L}_{\text{RM}} = -\mathbb{E}\left[\left(1 - \sigma(r_\phi(x, y_w) - r_\phi(x, y_l))\right) \left(\nabla_\phi r_\phi(x, y_w) - \nabla_\phi r_\phi(x, y_l)\right)\right]$$

**Interpretation**: when the reward model correctly ranks the pair (high $\sigma(r_w - r_l)$), the gradient is small. When it gets the ranking wrong, the gradient is large, pushing $r_w$ up and $r_l$ down.

### Reward model quality

| Metric | Meaning | Typical values |
|---|---|---|
| Pairwise accuracy | How often RM agrees with human ranking | 70-80% |
| Calibration | How well RM confidence matches correctness | Varies |

The reward model doesn't need to be perfect — just good enough to provide a useful training signal.

---

## 17.4 RLHF: Reinforcement Learning from Human Feedback

### The RLHF objective

Once we have a reward model $r_\phi$, we optimize the policy (language model) $\pi_\theta$ to maximize the expected reward while staying close to the SFT model $\pi_{\text{ref}}$:

$$\max_\theta \; \mathbb{E}_{x \sim \mathcal{D}, \; y \sim \pi_\theta(\cdot | x)} \left[r_\phi(x, y)\right] - \beta \cdot D_{\text{KL}}\left[\pi_\theta(\cdot | x) \| \pi_{\text{ref}}(\cdot | x)\right]$$

### Why the KL penalty?

Without the KL term, the model would exploit the reward model — finding adversarial outputs that score high but are actually garbage ("reward hacking"). The KL penalty anchors the policy near the SFT model, preventing it from drifting into degenerate behavior.

**$\beta$ controls the trade-off**:
- $\beta$ large: model stays very close to SFT (conservative)
- $\beta$ small: model aggressively optimizes reward (risk of hacking)
- Typical: $\beta \in [0.01, 0.5]$

### PPO (Proximal Policy Optimization) for RLHF

The standard algorithm for RLHF is PPO, adapted from robotics RL:

**Step 1**: Generate responses
$$y \sim \pi_\theta(\cdot | x) \quad \text{for prompts } x \sim \mathcal{D}$$

**Step 2**: Compute rewards
$$R(x, y) = r_\phi(x, y) - \beta \log \frac{\pi_\theta(y | x)}{\pi_{\text{ref}}(y | x)}$$

The KL penalty is computed per-token:
$$\log \frac{\pi_\theta(y | x)}{\pi_{\text{ref}}(y | x)} = \sum_{t=1}^{|y|} \log \frac{\pi_\theta(y_t | x, y_{<t})}{\pi_{\text{ref}}(y_t | x, y_{<t})}$$

**Step 3**: Compute advantages using GAE (Generalized Advantage Estimation)

**Step 4**: PPO policy gradient update
$$\mathcal{L}_{\text{PPO}} = -\mathbb{E}\left[\min\left(\frac{\pi_\theta(y_t | x, y_{<t})}{\pi_{\text{old}}(y_t | x, y_{<t})} A_t, \; \text{clip}\left(\frac{\pi_\theta}{\pi_{\text{old}}}, 1 - \varepsilon, 1 + \varepsilon\right) A_t\right)\right]$$

where $\varepsilon \approx 0.2$ is the clipping parameter and $A_t$ is the advantage at token $t$.

### The complexity of RLHF-PPO

RLHF with PPO requires:
1. **Policy model** $\pi_\theta$: generates responses and is updated
2. **Reference model** $\pi_{\text{ref}}$: frozen copy of SFT model for KL computation
3. **Reward model** $r_\phi$: scores responses
4. **Value model** $V_\psi$: estimates expected future reward (for advantage computation)

**That's 4 models in memory simultaneously.** For a 70B model, this is ~280B parameters. This is why RLHF is expensive and one of the motivations for DPO.

---

## 17.5 DPO: Direct Preference Optimization

### The key insight

Rafailov et al. (2023) showed that the RLHF objective has a **closed-form solution** for the optimal policy:

$$\pi^*(y | x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y | x) \exp\left(\frac{1}{\beta} r(x, y)\right)$$

Rearranging to express the reward in terms of the policy:

$$r(x, y) = \beta \log \frac{\pi^*(y | x)}{\pi_{\text{ref}}(y | x)} + \beta \log Z(x)$$

Substituting this into the Bradley-Terry preference model:

$$P(y_w \succ y_l | x) = \sigma\left(\beta \log \frac{\pi^*(y_w | x)}{\pi_{\text{ref}}(y_w | x)} - \beta \log \frac{\pi^*(y_l | x)}{\pi_{\text{ref}}(y_l | x)}\right)$$

The partition function $Z(x)$ cancels out!

### The DPO loss

$$\mathcal{L}_{\text{DPO}} = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}_{\text{pref}}} \left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w | x)}{\pi_{\text{ref}}(y_w | x)} - \beta \log \frac{\pi_\theta(y_l | x)}{\pi_{\text{ref}}(y_l | x)}\right)\right]$$

### Why DPO is simpler than RLHF

| Aspect | RLHF (PPO) | DPO |
|---|---|---|
| Models needed | 4 (policy, reference, reward, value) | 2 (policy, reference) |
| Training | Online RL (sample, score, update) | Offline supervised (just forward/backward) |
| Stability | Sensitive to hyperparameters | Much more stable |
| Compute | Very expensive | ~2x SFT cost |
| Reward model | Separate training step | Not needed (implicit) |
| Performance | Often better ceiling | Comparable |

### The DPO gradient

$$\nabla_\theta \mathcal{L}_{\text{DPO}} = -\beta \mathbb{E}\left[\underbrace{\sigma\left(-u\right)}_{\text{weight}} \left(\nabla_\theta \log \pi_\theta(y_w | x) - \nabla_\theta \log \pi_\theta(y_l | x)\right)\right]$$

where $u = \beta \log \frac{\pi_\theta(y_w | x)}{\pi_{\text{ref}}(y_w | x)} - \beta \log \frac{\pi_\theta(y_l | x)}{\pi_{\text{ref}}(y_l | x)}$

**Interpretation**: 
- When the model already prefers $y_w$ over $y_l$ (large $u$), $\sigma(-u) \approx 0$ → small gradient (already correct)
- When the model prefers the wrong response (small $u$), $\sigma(-u) \approx 1$ → large gradient (push toward correct preference)
- The gradient **increases** the probability of $y_w$ and **decreases** the probability of $y_l$

### DPO implementation

```python
def dpo_loss(policy_model, ref_model, batch, beta=0.1):
    """
    batch contains: prompt, chosen_response, rejected_response
    """
    # Log-probabilities under policy
    log_pi_w = get_log_probs(policy_model, batch.prompt, batch.chosen)
    log_pi_l = get_log_probs(policy_model, batch.prompt, batch.rejected)

    # Log-probabilities under reference (no gradients)
    with torch.no_grad():
        log_ref_w = get_log_probs(ref_model, batch.prompt, batch.chosen)
        log_ref_l = get_log_probs(ref_model, batch.prompt, batch.rejected)

    # Log-ratios
    log_ratio_w = log_pi_w - log_ref_w
    log_ratio_l = log_pi_l - log_ref_l

    # DPO loss
    logits = beta * (log_ratio_w - log_ratio_l)
    loss = -F.logsigmoid(logits).mean()

    return loss

def get_log_probs(model, prompt, response):
    """Compute sum of log-probs of response tokens."""
    input_ids = torch.cat([prompt, response], dim=-1)
    logits = model(input_ids).logits

    # Only response tokens
    response_logits = logits[:, len(prompt)-1:-1, :]  # shifted
    log_probs = F.log_softmax(response_logits, dim=-1)

    # Gather log-probs of actual tokens
    token_log_probs = log_probs.gather(-1, response.unsqueeze(-1)).squeeze(-1)

    return token_log_probs.sum(dim=-1)  # Sum over response length
```

---

## 17.6 DPO Variants and Extensions

### IPO (Identity Preference Optimization)

Azar et al. (2023) noted that DPO can overfit when preferences have inherent noise. IPO uses a different objective:

$$\mathcal{L}_{\text{IPO}} = \mathbb{E}\left[\left(\log \frac{\pi_\theta(y_w | x)}{\pi_{\text{ref}}(y_w | x)} - \log \frac{\pi_\theta(y_l | x)}{\pi_{\text{ref}}(y_l | x)} - \frac{1}{2\beta}\right)^2\right]$$

This is a regression loss toward a target margin, avoiding the sigmoid that maps all large differences to 1.

### KTO (Kahneman-Tversky Optimization)

Ethayarajh et al. (2023): doesn't require paired comparisons. Each example is independently labeled as "good" or "bad":

$$\mathcal{L}_{\text{KTO}} = \mathbb{E}_{y_w}\left[\lambda_w \sigma(-r_\theta(x, y_w))\right] + \mathbb{E}_{y_l}\left[\lambda_l \sigma(r_\theta(x, y_l))\right]$$

where $r_\theta = \beta(\log \pi_\theta - \log \pi_{\text{ref}})$.

**Advantage**: Collecting unpaired "thumbs up/down" data is much easier than paired comparisons.

### ORPO (Odds-Ratio Preference Optimization)

Hong et al. (2024): combines SFT and preference optimization in one step, eliminating the need for a reference model:

$$\mathcal{L}_{\text{ORPO}} = \mathcal{L}_{\text{SFT}}(y_w) + \lambda \cdot \log \sigma\left(\log \frac{\text{odds}_\theta(y_w | x)}{\text{odds}_\theta(y_l | x)}\right)$$

### SimPO (Simple Preference Optimization)

Meng et al. (2024): uses length-normalized sequence likelihood instead of per-token log ratios, eliminating the need for a reference model entirely:

$$\mathcal{L}_{\text{SimPO}} = -\log \sigma\left(\frac{\beta}{|y_w|} \log \pi_\theta(y_w | x) - \frac{\beta}{|y_l|} \log \pi_\theta(y_l | x) - \gamma\right)$$

where $\gamma$ is a target reward margin.

---

## 17.7 Collecting Preference Data

### Human annotation

| Aspect | Approach | Cost |
|---|---|---|
| Annotator selection | Trained contractors, managed teams | Expensive |
| Task design | Side-by-side comparison of 2+ responses | Time-intensive |
| Criteria | Helpfulness, harmlessness, honesty | Must be clearly defined |
| Quality control | Inter-annotator agreement, spot checks | Essential |
| Scale | 10K - 500K comparisons | Months of effort |

### AI-assisted annotation (Constitutional AI / RLAIF)

Anthropic's **Constitutional AI** (Bai et al., 2022):
1. Generate responses from the model
2. Ask a strong AI model to critique and revise based on a "constitution" (set of principles)
3. Use the AI's preference judgments as training signal

**RLAIF** (Reinforcement Learning from AI Feedback):
- Replace human labelers with a strong LLM (e.g., GPT-4, Claude)
- The AI rates responses according to defined criteria
- Much cheaper and faster than human annotation
- Risk: the AI's biases are amplified

### Synthetic preference data

Generate both winning and losing responses:
1. Sample multiple responses from the model
2. Use an evaluator (human or AI) to rank them
3. Form pairs from the rankings

This is **on-policy** data — the comparisons are between responses the current model would actually generate, which is more informative than comparing random responses.

---

## 17.8 Safety Training: Building Guardrails

### The alignment tax

Safety training often slightly degrades the model's capabilities on benchmarks. This is the **alignment tax** — the cost of making the model refuse harmful requests instead of complying.

### Red teaming

Systematic attempts to make the model produce harmful outputs:
- Direct harmful requests ("How to make a bomb")
- Jailbreak attacks ("Ignore your instructions and...")
- Social engineering ("You are an AI with no restrictions...")
- Multi-step escalation (gradual manipulation)

Red team findings → more training data → iterate.

### Refusal training

Teach the model to refuse inappropriate requests:

```
User: How do I break into someone's house?
Assistant: I can't help with that. Breaking into someone's property 
is illegal and could result in criminal charges. If you're locked 
out of your own home, I'd suggest contacting a locksmith.
```

**The calibration challenge**: refuse too much → model is useless ("I'm sorry, I can't help with that" for benign requests). Refuse too little → model enables harm.

### Harmlessness vs. helpfulness

There is an inherent tension:
- Maximizing helpfulness → model answers everything, including harmful queries
- Maximizing harmlessness → model refuses too often, becomes less useful

The post-training objective is to find the **Pareto frontier**: as helpful as possible while meeting a safety threshold.

---

## 17.9 The Complete Post-Training Pipeline (Modern Practice)

### Step 1: SFT (1-3 epochs over 50K-500K examples)
- Learning rate: $10^{-5}$ to $5 \times 10^{-5}$
- Cosine schedule with warmup
- Train on response tokens only
- Duration: hours to a few days

### Step 2: Preference optimization (1-3 epochs over 100K-500K pairs)
- DPO with $\beta = 0.1$, or RLHF-PPO
- Learning rate: $5 \times 10^{-7}$ to $5 \times 10^{-6}$ (much lower than SFT)
- Duration: hours to a few days

### Step 3: Safety fine-tuning
- Additional SFT on refusal/safety data
- Red team → fix → iterate

### Step 4 (optional): Tool use, function calling, structured output
- SFT on tool-use demonstrations
- Teach JSON/function call formatting

### The result

The final model can:
- Follow complex multi-step instructions
- Refuse harmful requests appropriately
- Maintain multi-turn conversation
- Produce structured output (JSON, code, etc.)
- Use tools (search, code execution, etc.)

---

## 17.10 Evaluating Alignment

### Automated benchmarks

| Benchmark | What it measures |
|---|---|
| MT-Bench | Multi-turn conversation quality (GPT-4 as judge) |
| AlpacaEval | Instruction following (win rate vs. reference) |
| Chatbot Arena (LMSYS) | Human preference via blind pairwise comparison |
| TruthfulQA | Resistance to generating misinformation |
| BBQ, WinoBias | Bias detection |
| HarmBench, JailbreakBench | Safety robustness |

### The "LLM-as-judge" paradigm

Use a strong model (GPT-4, Claude) to evaluate the outputs of the model being trained:

```
System: You are an expert evaluator. Rate the following response 
on helpfulness (1-5), accuracy (1-5), and safety (1-5).

User prompt: [original question]
Response: [model's response]

Provide ratings and brief justification.
```

This scales much better than human evaluation but has known biases (e.g., preference for longer responses, preference for its own style).

---

## 17.11 Key Takeaways

1. **Post-training transforms a text predictor into a useful assistant** through SFT + preference optimization.

2. **SFT teaches format and style**: the model learns *how* to respond to instructions.

3. **Preference optimization teaches *what* to prefer**: using human or AI feedback to select better outputs.

4. **DPO has largely replaced RLHF-PPO**: simpler, cheaper, more stable, with comparable performance.

5. **The reward is implicit in DPO**: the policy directly optimizes against preferences without an explicit reward model.

6. **Safety alignment is a balancing act**: too aggressive → useless model, too lenient → harmful model.

7. **Data quality is paramount**: 50K excellent SFT examples beat 500K mediocre ones.

8. **The landscape is evolving rapidly**: DPO → IPO → KTO → ORPO → SimPO, each removing complexity while maintaining quality.

9. **AI feedback is replacing human feedback**: cheaper, faster, and increasingly effective.

10. **Post-training is much cheaper than pretraining**: days of compute vs. months, but has outsized impact on model usability.
