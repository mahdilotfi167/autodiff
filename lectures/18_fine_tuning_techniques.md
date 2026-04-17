# 18. Fine-Tuning Techniques: LoRA, QLoRA, and Parameter-Efficient Methods

## Motivation

Pretraining costs millions of dollars. Post-training (SFT + alignment) costs thousands. But adapting a model to **your specific task** should be cheap and fast. This is the domain of **fine-tuning** — updating a pretrained model on task-specific data.

Full fine-tuning (updating all parameters) is effective but expensive: it requires storing optimizer states for every parameter and risks catastrophic forgetting. **Parameter-efficient fine-tuning (PEFT)** methods update only a small fraction of parameters, achieving comparable quality at a fraction of the cost.

This document derives the key PEFT methods from first principles, analyzes their trade-offs, and provides practical guidance for choosing and implementing them.

---

## 18.1 Full Fine-Tuning: The Baseline

### What it does

Update ALL model parameters $\theta$ on the task-specific dataset $\mathcal{D}_{\text{task}}$:

$$\theta^* = \arg\min_\theta \; \mathcal{L}_{\text{task}}(\theta; \mathcal{D}_{\text{task}})$$

Starting from the pretrained weights $\theta_0$.

### Memory requirements

For a model with $P$ parameters in full fine-tuning with AdamW:

| Component | Memory |
|---|---|
| Model weights (bf16) | $2P$ bytes |
| Optimizer states (fp32 $m, v$) | $8P$ bytes |
| Master weights (fp32) | $4P$ bytes |
| Gradients (bf16) | $2P$ bytes |
| **Total** | **$16P$ bytes** |

For LLaMA-70B ($P = 70\text{B}$): $16 \times 70\text{B} = 1.12$ TB. That's 14 H100 GPUs (80GB each) just for model state.

### When full fine-tuning makes sense

- You have massive task-specific data (>100K examples)
- You have sufficient compute (many GPUs)
- Maximum quality is required
- The task is very different from the pretraining distribution

### When it doesn't

- Limited data (<10K examples) → overfitting risk
- Limited compute (1-4 GPUs) → can't fit optimizer states
- Multiple tasks → storing one full copy per task is wasteful
- Quick iteration needed → full training is slow

---

## 18.2 The Key Insight Behind PEFT: The Low-Rank Hypothesis

### Why do we think PEFT can work?

**Observation** (Aghajanyan et al., 2021): the weight changes during fine-tuning have **low intrinsic dimensionality**. That is:

$$\Delta\mathbf{W} = \mathbf{W}_{\text{fine-tuned}} - \mathbf{W}_{\text{pretrained}}$$

has a rank much lower than the full matrix rank.

**Intuition**: pretraining already learns good representations. Fine-tuning makes relatively small, structured adjustments — not random changes across all dimensions.

**Evidence**: Aghajanyan et al. showed that you can project gradient updates onto a random low-dimensional subspace (as low as $d_{\text{intrinsic}} \approx 200$ for billion-parameter models) and still achieve 90%+ of full fine-tuning performance.

This motivates: **instead of updating the full weight matrix, update a low-rank approximation**.

---

## 18.3 LoRA: Low-Rank Adaptation

### The core idea

Hu et al. (2021) proposed: freeze all pretrained weights $\mathbf{W}_0$ and add a trainable low-rank decomposition:

$$\mathbf{W} = \mathbf{W}_0 + \Delta\mathbf{W} = \mathbf{W}_0 + \mathbf{B}\mathbf{A}$$

where:
- $\mathbf{W}_0 \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ — frozen pretrained weights
- $\mathbf{B} \in \mathbb{R}^{d_{\text{out}} \times r}$ — trainable, initialized to **zero**
- $\mathbf{A} \in \mathbb{R}^{r \times d_{\text{in}}}$ — trainable, initialized from $\mathcal{N}(0, \sigma^2)$
- $r$ — the **rank**, typically $r \in \{4, 8, 16, 32, 64\}$

### Why this works mathematically

The forward pass for a linear layer becomes:

$$\mathbf{h} = \mathbf{W}_0 \mathbf{x} + \mathbf{B}\mathbf{A}\mathbf{x} = (\mathbf{W}_0 + \mathbf{B}\mathbf{A})\mathbf{x}$$

### Shape analysis

For a weight matrix of shape $(d_{\text{out}}, d_{\text{in}})$:

| Component | Shape | Parameters |
|---|---|---|
| Original $\mathbf{W}_0$ | $(d_{\text{out}}, d_{\text{in}})$ | $d_{\text{out}} \cdot d_{\text{in}}$ (frozen) |
| $\mathbf{B}$ | $(d_{\text{out}}, r)$ | $d_{\text{out}} \cdot r$ (trainable) |
| $\mathbf{A}$ | $(r, d_{\text{in}})$ | $r \cdot d_{\text{in}}$ (trainable) |
| **Trainable total** | | $r(d_{\text{out}} + d_{\text{in}})$ |

**Compression ratio**: 
$$\frac{\text{trainable params}}{\text{full params}} = \frac{r(d_{\text{out}} + d_{\text{in}})}{d_{\text{out}} \cdot d_{\text{in}}} \approx \frac{2r}{d} \quad \text{(when } d_{\text{out}} = d_{\text{in}} = d\text{)}$$

For $d = 4096, r = 16$: compression ratio = $\frac{2 \times 16}{4096} = 0.78\%$ — less than 1% of parameters are trainable!

### Initialization: why $\mathbf{B} = \mathbf{0}$?

At the start of training:
$$\Delta\mathbf{W} = \mathbf{B}\mathbf{A} = \mathbf{0} \cdot \mathbf{A} = \mathbf{0}$$

The model starts **exactly** at the pretrained weights and smoothly evolves from there. This is crucial: any random initialization of $\Delta\mathbf{W}$ would immediately disrupt the pretrained representations.

### The $\alpha$ scaling factor

In practice, LoRA includes a scaling factor:

$$\mathbf{h} = \mathbf{W}_0 \mathbf{x} + \frac{\alpha}{r} \mathbf{B}\mathbf{A}\mathbf{x}$$

where $\alpha$ is a hyperparameter (often $\alpha = r$ or $\alpha = 2r$).

**Why**: $\frac{\alpha}{r}$ normalizes the contribution of the low-rank update. When increasing $r$ (more parameters), we don't want the update magnitude to grow proportionally. The scaling keeps the effective learning rate independent of rank.

With $\alpha = r$: the scaling factor is 1, and $r$ only controls the number of parameters. With $\alpha = 16, r = 8$: the scaling is 2, amplifying the low-rank update.

### Which layers to apply LoRA to?

| Target modules | Effect | Common choice |
|---|---|---|
| $\mathbf{W}_Q, \mathbf{W}_K$ only | Modifies attention patterns | Minimal adaptation |
| $\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V, \mathbf{W}_O$ | Full attention adaptation | **Recommended** |
| All attention + FFN | Maximum adaptation | Best quality |
| Embedding + LM head | Adapts input/output | For domain shift |

Empirical finding: applying LoRA to **all linear layers** (attention + FFN) typically gives the best results for a given total parameter budget, as opposed to higher rank on fewer layers.

### LoRA implementation

```python
class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank=16, alpha=16):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.linear.weight.requires_grad = False  # Freeze

        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.scaling = alpha / rank

    def forward(self, x):
        # Frozen base + trainable low-rank
        base_out = self.linear(x)
        lora_out = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return base_out + lora_out

    def merge_weights(self):
        """Merge LoRA into base weights for deployment (no extra latency)."""
        self.linear.weight.data += (self.lora_B @ self.lora_A * self.scaling)
```

### Key property: merge at deployment

After training, merge $\Delta\mathbf{W} = \frac{\alpha}{r}\mathbf{B}\mathbf{A}$ into $\mathbf{W}_0$:

$$\mathbf{W}_{\text{deployed}} = \mathbf{W}_0 + \frac{\alpha}{r}\mathbf{B}\mathbf{A}$$

This produces a standard model with **zero inference overhead**. The LoRA structure is only needed during training.

---

## 18.4 The Effect of LoRA Rank ($r$)

### What rank controls

$r$ determines the expressiveness of the adaptation:

| $r$ | Trainable params (% of full) | Quality | Use case |
|---|---|---|---|
| 1 | ~0.05% | Low | Extreme compression, simple tasks |
| 4 | ~0.2% | Fair | Simple classification/extraction |
| 8 | ~0.4% | Good | General instruction following |
| 16 | ~0.8% | Very good | **Default recommendation** |
| 32 | ~1.6% | Excellent | Complex/creative tasks |
| 64 | ~3.1% | Near-full FT | When quality is paramount |
| 256 | ~12.5% | ≈ Full FT | Diminishing returns |

### The rank-performance curve

```
Quality (% of full FT)
 100% │                    ●───────●──────
      │               ●
  95% │          ●
      │     ●
  90% │  ●
      │●
  85% │
      └──┬──┬──┬──┬──┬──┬──→ Rank
         1  4  8  16 32 64 128
```

**Key observation**: quality saturates quickly. Going from $r=4$ to $r=16$ gives a big improvement. Going from $r=16$ to $r=64$ gives diminishing returns.

---

## 18.5 QLoRA: Quantized LoRA

### The memory problem LoRA doesn't solve

LoRA reduces **trainable** parameters, but you still need the **frozen base model** in memory. For LLaMA-70B, the base model alone is 140 GB in bf16.

### The QLoRA solution (Dettmers et al., 2023)

**Quantize** the frozen base model to 4-bit precision, then add LoRA adapters in bf16:

$$\mathbf{h} = \text{Dequant}(\mathbf{W}_{\text{4bit}}) \cdot \mathbf{x} + \mathbf{B}\mathbf{A}\mathbf{x}$$

### 4-bit NormalFloat (NF4) quantization

The key innovation: quantize using a data type optimized for normally-distributed weights.

**Observation**: neural network weights are approximately normally distributed $\mathcal{N}(0, \sigma^2)$.

**NF4**: choose 16 quantization levels ($2^4 = 16$) at the quantiles of the standard normal distribution, so that each quantization bin has equal probability mass.

| Standard uniform 4-bit | NF4 (normal-optimized) |
|---|---|
| Evenly spaced: -8, -7, ..., 0, ..., 7 | Quantile-spaced: -1.08, -0.86, ..., 0, ..., 0.86, 1.08 |
| Poor for normal distribution | Optimal for normal distribution |

### Double quantization

QLoRA also quantizes the **quantization constants** (the per-block scaling factors) to 8-bit, saving additional memory.

### QLoRA memory savings

For LLaMA-70B:

| Method | Base model | LoRA adapters | Optimizer | Total |
|---|---|---|---|---|
| Full FT (bf16) | 140 GB | — | 560 GB | ~700 GB |
| LoRA (bf16) | 140 GB | ~300 MB | ~2.4 GB | ~143 GB |
| QLoRA (4-bit) | ~35 GB | ~300 MB | ~2.4 GB | **~38 GB** |

QLoRA enables fine-tuning a 70B model on a **single 48GB GPU** (A6000) or even a consumer GPU.

### Quantization impact on quality

| Method | Perplexity (LLaMA-65B) | Notes |
|---|---|---|
| Full FT (bf16) | Best | Baseline |
| LoRA (bf16) | -0.1 | Negligible loss |
| QLoRA (4-bit) | -0.2 to -0.5 | Small loss, remarkable given 4-bit |

**Finding**: QLoRA matches full 16-bit fine-tuning quality on most benchmarks, despite the base model being in 4-bit.

---

## 18.6 Other PEFT Methods

### Prefix Tuning (Li & Liang, 2021)

Prepend learnable "virtual tokens" to the key and value matrices in each attention layer:

$$\mathbf{K}' = [\mathbf{P}_K; \mathbf{K}], \quad \mathbf{V}' = [\mathbf{P}_V; \mathbf{V}]$$

where $\mathbf{P}_K, \mathbf{P}_V \in \mathbb{R}^{l \times d}$ are the learnable prefix (length $l$).

**Effect**: the model can attend to $l$ extra "virtual context tokens" that encode task information.

**Parameters**: $2 \cdot l \cdot d \cdot N$ (prefix for K and V at each layer)

**Pros**: elegant, no modification to model weights
**Cons**: reduces effective context length by $l$ tokens, often underperforms LoRA

### Adapters (Houlsby et al., 2019)

Insert small bottleneck layers inside each transformer block:

$$\text{Adapter}(\mathbf{x}) = \mathbf{x} + f(\mathbf{x}\mathbf{W}_{\text{down}})\mathbf{W}_{\text{up}}$$

where $\mathbf{W}_{\text{down}} \in \mathbb{R}^{d \times r}$ and $\mathbf{W}_{\text{up}} \in \mathbb{R}^{r \times d}$.

**Pros**: flexible, well-studied
**Cons**: adds inference latency (cannot be merged like LoRA)

### Prompt Tuning (Lester et al., 2021)

Learn soft prompt tokens prepended to the input:

$$\text{input} = [\mathbf{p}_1, \mathbf{p}_2, \ldots, \mathbf{p}_l, \text{actual input tokens}]$$

where $\mathbf{p}_i \in \mathbb{R}^d$ are learnable embedding vectors.

**Parameters**: $l \times d$ (very few — just the soft prompt embeddings)

**Pros**: extremely parameter-efficient
**Cons**: limited capacity, works best for large models (>10B)

### Comparison table

| Method | Trainable params | Inference overhead | Quality (relative) | Merging |
|---|---|---|---|---|
| Full fine-tuning | 100% | None | Best | N/A |
| LoRA | 0.1-3% | None (after merge) | Near-best | ✓ |
| QLoRA | 0.1-3% | Small (dequant) | Good | ✓ |
| Adapters | 1-5% | Yes (extra layers) | Good | ✗ |
| Prefix tuning | 0.1-1% | Slight (longer seq) | Fair | ✗ |
| Prompt tuning | 0.01% | Slight (longer seq) | Fair | ✗ |

**Winner**: LoRA (and QLoRA) have become the dominant PEFT methods due to the combination of good quality, zero inference overhead, and simplicity.

---

## 18.7 DoRA: Weight-Decomposed Low-Rank Adaptation

### The insight

Liu et al. (2024) decompose weight matrices into **magnitude** and **direction** components:

$$\mathbf{W} = m \cdot \frac{\mathbf{W}_0 + \mathbf{B}\mathbf{A}}{\|\mathbf{W}_0 + \mathbf{B}\mathbf{A}\|_c}$$

where $m$ is a learnable magnitude scalar per output neuron and $\|\cdot\|_c$ is the column-wise norm.

**Why**: full fine-tuning changes both the magnitude and direction of weight vectors. Standard LoRA couples these changes through the low-rank matrix. DoRA separates them:
- **Direction** adapted by LoRA ($\mathbf{B}\mathbf{A}$)
- **Magnitude** adapted by a separate learnable scalar ($m$)

**Result**: DoRA matches or exceeds LoRA quality with the same rank, especially at low ranks ($r \leq 8$).

---

## 18.8 Practical Fine-Tuning Recipe

### Step 1: Choose your method

| Your situation | Recommended method |
|---|---|
| Single consumer GPU (24GB) | QLoRA, $r=16$ |
| A few GPUs (4-8 × 80GB) | LoRA, $r=16-32$ |
| Large GPU cluster | Full fine-tuning or LoRA with high rank |
| Need zero inference latency | LoRA (merge after training) |
| Multiple tasks, one base model | LoRA (swap adapters) |

### Step 2: Hyperparameters

| Hyperparameter | Recommended value | Notes |
|---|---|---|
| LoRA rank $r$ | 16 | Increase to 32-64 for complex tasks |
| LoRA $\alpha$ | $r$ or $2r$ | Keep $\alpha/r \in [1, 2]$ |
| Target modules | All linear layers | QKV, O, FFN (gate, up, down) |
| Learning rate | $2 \times 10^{-4}$ | 10-100× higher than full FT |
| Weight decay | 0.01 | Lower than pretraining |
| Batch size | 16-128 examples | With gradient accumulation |
| Epochs | 1-5 | More epochs for small datasets |
| Warmup | 3-10% of total steps | |
| LR schedule | Cosine or linear decay | |
| Dropout | 0.05-0.1 | LoRA dropout on A, B matrices |

### Step 3: Data preparation

```python
# Instruction fine-tuning dataset format
{
    "instruction": "Translate to French",
    "input": "Hello, how are you?",
    "output": "Bonjour, comment allez-vous?"
}

# Chat format (multi-turn)
{
    "conversations": [
        {"role": "user", "content": "What is photosynthesis?"},
        {"role": "assistant", "content": "Photosynthesis is..."},
        {"role": "user", "content": "Why is it important?"},
        {"role": "assistant", "content": "It's important because..."}
    ]
}
```

### Step 4: Training

```python
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3-8B",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Apply LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# trainable params: 83,886,080 || all params: 8,114,212,864 || trainable%: 1.03%

# Training
trainer = Trainer(
    model=model,
    train_dataset=train_dataset,
    args=TrainingArguments(
        output_dir="./output",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        num_train_epochs=3,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        bf16=True,
        logging_steps=10,
        save_strategy="epoch"
    )
)
trainer.train()
```

---

## 18.9 Multi-Task Fine-Tuning with LoRA

### Adapter stacking and switching

One powerful property of LoRA: you can train **multiple adapters** for different tasks and switch between them at inference:

```python
# Task 1: Medical QA
model.load_adapter("medical_lora")
response = model.generate(medical_question)

# Task 2: Code generation
model.load_adapter("code_lora")
response = model.generate(coding_prompt)

# Task 3: Base model (no adapter)
model.disable_adapter()
response = model.generate(general_prompt)
```

**Storage**: Each adapter is only ~100-300 MB. You can store hundreds of task-specific adapters against one base model.

### Adapter merging

You can also merge multiple adapters with learned weights:

$$\mathbf{W} = \mathbf{W}_0 + \lambda_1 \Delta\mathbf{W}_1 + \lambda_2 \Delta\mathbf{W}_2 + \ldots$$

Methods for choosing $\lambda_i$:
- **Linear combination**: manually weighted average
- **TIES merging**: resolve sign conflicts between adapters
- **DARE**: randomly drop elements before merging

---

## 18.10 Common Fine-Tuning Failures and Solutions

### Catastrophic forgetting

**Symptom**: model loses general capabilities after fine-tuning.

**Causes**:
- Learning rate too high
- Too many epochs
- Dataset too narrow

**Solutions**:
- Use LoRA (frozen base preserves general knowledge)
- Lower learning rate
- Add general instruction data to the fine-tuning mix
- Early stopping based on validation loss

### Overfitting

**Symptom**: training loss decreases but validation loss increases.

**Causes**:
- Dataset too small
- Too many epochs
- LoRA rank too high for the data

**Solutions**:
- Add LoRA dropout
- Reduce rank
- Reduce epochs
- Augment data

### Mode collapse

**Symptom**: model produces very similar responses to different inputs.

**Causes**:
- Dataset lacks diversity
- Learning rate too high in preference optimization
- KL penalty too weak (in RLHF/DPO)

**Solutions**:
- Diversify training data
- Increase $\beta$ in DPO
- Lower learning rate
- Add regularization

---

## 18.11 Key Takeaways

1. **LoRA is the default PEFT method**: low-rank updates to frozen weights, <1% trainable parameters, zero inference overhead after merging.

2. **QLoRA enables fine-tuning on consumer hardware**: 4-bit base model + bf16 LoRA adapters.

3. **Rank 16 is a good default**: quality/efficiency sweet spot for most tasks.

4. **Apply LoRA to all linear layers**: better than higher rank on fewer layers.

5. **Learning rate for LoRA is 10-100× higher** than for full fine-tuning ($2 \times 10^{-4}$ vs. $2 \times 10^{-5}$).

6. **Weight tying on initialization**: $\mathbf{B}=\mathbf{0}$ ensures the model starts exactly at pretrained weights.

7. **LoRA adapters are tiny and swappable**: enabling multi-task deployment with one base model.

8. **The low-rank hypothesis is real**: fine-tuning weight changes are inherently low-dimensional.

9. **Full fine-tuning is still king for quality** when you have the compute budget and data.

10. **The field is evolving**: DoRA, GaLore, and other methods continuously improve the quality/efficiency trade-off.
