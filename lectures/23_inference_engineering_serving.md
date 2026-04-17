# 23. Inference Engineering & Serving: From Model to Product

## Motivation

Training a great model is only half the battle. Serving it to millions of users at low latency and reasonable cost is an entirely different engineering discipline. This lecture covers the systems-level understanding of LLM inference that interviewers at Anthropic, OpenAI, and Google test for: memory bandwidth bottlenecks, batching strategies, quantization tradeoffs, cost modeling, and the full serving stack.

---

## 23.1 Why Inference Is Hard: The Memory Bandwidth Wall

### The fundamental bottleneck

LLM inference (token generation) is **memory-bandwidth bound**, not compute-bound.

For each generated token, the model must:
1. Load all model weights from GPU memory → compute units
2. Load the KV cache for the entire context
3. Compute a single forward pass (one token)
4. Write the new KV cache entry

**Arithmetic intensity** = FLOPs / bytes loaded.

For a single token in a 7B model (bf16):

- **FLOPs**: $\approx 2 \times 7 \times 10^9 = 14$ GFLOPs (2 FLOPs per parameter for matmul)
- **Bytes loaded**: $2 \times 7 \times 10^9 = 14$ GB (all weights in bf16)
- **Arithmetic intensity**: $14 \times 10^9 / 14 \times 10^9 = 1$ FLOP/byte

An A100 has:
- Compute: 312 TFLOPS (bf16)
- Memory bandwidth: 2 TB/s

To be compute-bound, you need arithmetic intensity > $312/2 = 156$ FLOP/byte. Single-token generation has intensity ~1. You're using **0.6%** of the GPU's compute capacity.

**This is why single-user LLM inference is so expensive**: you're paying for a 312 TFLOP GPU but only using 2 TFLOPS.

### The batch size cure

Batching multiple requests together amortizes weight loading:

| Batch size | Bytes loaded | FLOPs | Intensity | Utilization |
|---|---|---|---|---|
| 1 | 14 GB | 14 GF | 1.0 | 0.6% |
| 8 | 14 GB | 112 GF | 8.0 | 5% |
| 64 | 14 GB | 896 GF | 64 | 41% |
| 156 | 14 GB | 2.2 TF | 156 | 100% (compute bound) |

But there's a catch: **KV cache memory limits batch size**.

### KV cache memory per request

For a model with $N$ layers, $n_{kv}$ KV heads, head dimension $d_k$, context length $n$, in bf16:

$$\text{KV cache per request} = 2 \times N \times n_{kv} \times d_k \times n \times 2 \text{ bytes}$$

The factor of $2 \times$ at the start: one for K, one for V. The $\times 2$ at the end: bf16 = 2 bytes.

| Model | Layers | KV heads | $d_k$ | Context | KV cache/req |
|---|---|---|---|---|---|
| 7B (GQA) | 32 | 8 | 128 | 4K | 512 MB |
| 7B (GQA) | 32 | 8 | 128 | 128K | 16 GB |
| 70B (GQA) | 80 | 8 | 128 | 4K | 1.3 GB |
| 70B (MHA) | 80 | 64 | 128 | 4K | 10 GB |

**This is why GQA matters for inference**: 8× fewer KV heads = 8× smaller KV cache = 8× more requests per GPU.

**This is why long context is expensive**: 128K context = 32× more KV cache than 4K.

---

## 23.2 The Two Phases of Inference

### Prefill (prompt processing)

- Process the entire prompt in parallel (like training)
- **Compute-bound**: large matrix multiplications, high arithmetic intensity
- Latency = time-to-first-token (TTFT)
- TTFT scales linearly with prompt length (at fixed batch size)

### Decode (token generation)

- Generate one token at a time, using KV cache
- **Memory-bandwidth-bound**: load all weights for each token
- Latency per token = inter-token latency (ITL)
- ITL is roughly constant regardless of generated length (until KV cache pressure)

### The prefill-decode tension

These two phases have opposite requirements:

| Property | Prefill | Decode |
|---|---|---|
| Compute pattern | Large batch matmul | Single-vector matmul |
| Bottleneck | Compute | Memory bandwidth |
| Ideal batch size | Small | Large |
| GPU utilization | High | Low (without batching) |

**Solution: Disaggregated serving** — use separate GPU pools for prefill and decode, each optimized for its workload.

---

## 23.3 Continuous Batching

### The problem with static batching

Static batching: wait for $B$ requests, process them together, return all results.

- Request A needs 50 tokens → done quickly
- Request B needs 500 tokens → still generating
- Request A's GPU slot is wasted for 450 tokens

**GPU utilization**: terrible. Short requests hold up slots.

### Continuous batching (Orca)

Instead of waiting for all requests to finish:

1. As soon as a request finishes generating, **remove it** from the batch
2. Immediately **insert** a waiting request into the freed slot
3. The batch is always full (or as full as possible)

```
Time →
Slot 0: [A A A A A _ B B B B B B B B B B _ C C C]
Slot 1: [D D D D D D D D _ E E E E E _ F F F F F]
Slot 2: [G G G _ H H H H H H H H H _ I I I I I I]
```

Each slot generates tokens independently. When a request completes, a new one joins.

**Impact**: 2-5× throughput improvement with no latency penalty.

### Handling variable-length KV caches

Each request in the batch has a different context length → different KV cache size. Two approaches:

1. **Pad to max**: Allocate max context length for every request. Wasteful.
2. **PagedAttention (vLLM)**: Allocate KV cache in pages (like OS virtual memory).

---

## 23.4 PagedAttention: Virtual Memory for KV Cache

### The fragmentation problem

Without paging:
- Request needs 1000 tokens of KV cache
- Allocate a contiguous 1000-token block
- Request actually generates 200 tokens
- 800 tokens of memory wasted (80% waste!)

Average waste across a batch: ~60-70% of allocated KV memory.

### How PagedAttention works

Split KV cache into fixed-size **pages** (e.g., 16 tokens each):

```
Physical memory:
Page 0: [KV for tokens 0-15 of Request A]
Page 1: [KV for tokens 0-15 of Request B]
Page 2: [KV for tokens 16-31 of Request A]
Page 3: [KV for tokens 16-31 of Request B]
...

Page table (per request):
Request A: [0, 2, 5, 8, ...]
Request B: [1, 3, 6, 9, ...]
```

- Pages are allocated on demand as tokens are generated
- No internal fragmentation (waste = at most 1 page per request)
- Memory utilization: ~96-99%

### Copy-on-write for beam search / parallel sampling

When multiple beams share a prefix, they share the same physical KV cache pages. A page is only copied when a beam modifies it (writes a different token). This makes beam search nearly free in memory.

---

## 23.5 Quantization for Inference

### Why quantize?

- Model weights dominate memory → fewer bits = smaller model = faster weight loading
- Since inference is memory-bandwidth-bound, quantization directly increases throughput
- A 4-bit model loads 4× faster than a 16-bit model

### Quantization formats

| Format | Bits | Method | Quality loss | Speed gain |
|---|---|---|---|---|
| FP16/BF16 | 16 | Baseline | None | 1× |
| INT8 (per-tensor) | 8 | Round to nearest int8 | Minimal | ~1.8× |
| INT8 (per-channel) | 8 | Scale per output channel | Very small | ~1.8× |
| GPTQ (INT4) | 4 | Layer-wise optimal rounding | Small | ~3× |
| AWQ (INT4) | 4 | Activation-aware, protect salient weights | Small | ~3× |
| GGUF (k-quant) | 2-6 | Mixed precision per block | Varies | ~3-4× |
| FP8 (E4M3) | 8 | Native hardware support (H100+) | Minimal | ~1.9× |

### The "outlier" problem

Transformer activations have **outlier features**: a small number of hidden dimensions with values 10-100× larger than the rest. These appear consistently across tokens and layers.

**Why it matters for quantization**: If you quantize uniformly, the range is dominated by outliers → most values clustered in a few quantization bins → huge information loss.

**Solutions**:
- **Per-channel quantization**: Different scale per output dimension (handles per-channel outliers)
- **SmoothQuant**: Redistribute magnitude from activations to weights (mathematically equivalent, but activations become easier to quantize)
- **AWQ**: Identify and protect the ~1% of weight channels that correspond to outlier activations
- **SpQR**: Keep outlier weights in higher precision, quantize the rest aggressively

### Quality degradation profile

Not all layers degrade equally under quantization:

| Component | Sensitivity | Why |
|---|---|---|
| Embedding | Low | High-dimensional, redundant |
| Attention QKV | Medium | Precision matters for attention scores |
| Attention output | Low | Projected to residual stream |
| FFN up/gate | Medium | SwiGLU gating is sensitive |
| FFN down | Low | Projects back to residual |
| LM head | **High** | Directly affects token probabilities |
| Layer norms | **High** | Small parameters, big impact |

**Pro tip**: Never quantize layer norms or the LM head below INT8. Quantize FFN weights more aggressively (they're the largest and least sensitive).

---

## 23.6 Speculative Decoding: Getting 2-3× Speedup for Free

### The idea (revisited from Doc 19.7)

Use a small "draft" model ($M_d$) to generate $K$ candidate tokens. Then verify all $K$ in parallel with the large "target" model ($M_t$).

### Why it works: the acceptance probability

For token $x$ at position $t$:

$$P(\text{accept}) = \min\left(1, \frac{P_{M_t}(x | x_{<t})}{P_{M_d}(x | x_{<t})}\right)$$

If the draft model is good (close to target), most tokens are accepted. Expected speedup:

$$\text{speedup} = \frac{K \cdot P(\text{accept})^K}{1 + K \cdot c}$$

where $c = \text{cost of draft model} / \text{cost of target model}$.

### Key implementation details:

1. **Draft model selection**: Same architecture, ~10-20× fewer parameters. Or: use the target model's early layers (self-speculative decoding).

2. **Verification is parallel**: The target model processes all $K$ draft tokens in one forward pass (prefill-style). This is compute-bound → GPU is well-utilized.

3. **Quality guarantee**: The output distribution is mathematically identical to the target model. No approximation.

4. **The "rejection sampling" step**: If token $k$ is rejected, resample from an adjusted distribution:
$$P_{\text{adjusted}}(x) = \text{normalize}\left(\max(0, P_{M_t}(x) - P_{M_d}(x))\right)$$

### When speculative decoding helps most:

- Low-concurrency serving (few simultaneous users → single-request latency matters)
- Tasks where draft model accuracy is high (continuation, repetitive text)
- Models where decode is memory-bandwidth-dominated (no batching benefit)

### When it doesn't help:

- High-throughput serving (already batching → GPU is utilized)
- Tasks where draft model is bad (creative writing, reasoning)

---

## 23.7 Tensor Parallelism for Inference

### Why model parallelism at inference

Large models don't fit on one GPU:
- 70B model (bf16): 140 GB — doesn't fit on A100 (80 GB)
- With KV cache: even more memory needed

### Tensor parallelism (TP) for inference

Split each weight matrix across GPUs. For a linear layer $\mathbf{y} = \mathbf{x}\mathbf{W}$:

**Column-parallel**: Split $\mathbf{W}$ by columns across $P$ GPUs:
$$\mathbf{y}_p = \mathbf{x} \mathbf{W}_p \quad \text{(each GPU computes a slice)}$$
Results are concatenated (or used independently in attention heads).

**Row-parallel**: Split $\mathbf{W}$ by rows:
$$\mathbf{y} = \sum_{p=1}^{P} \mathbf{x}_p \mathbf{W}_p \quad \text{(all-reduce to sum)}$$

### Communication overhead

Each transformer block requires:
- 2 all-reduce operations (one per sub-layer: attention and FFN)
- Each all-reduce: $2 \times d_{\text{model}} \times B \times (\text{P-1}/\text{P})$ bytes

For 70B on 8 GPUs with NVLink (900 GB/s bidirectional):
- Data per all-reduce: $2 \times 8192 \times 1 \times (7/8) \times 2 \approx 28$ KB
- Time: $28\text{KB} / 900\text{GB/s} \approx 31$ ns

This is negligible! TP on NVLink adds <5% overhead for single-request inference.

**But**: Cross-node TP (over network, not NVLink) is much slower. Rule of thumb: only use TP within a single node (8 GPUs); use pipeline parallelism across nodes.

---

## 23.8 Cost Modeling: Dollars Per Token

### The formula

For a model with $P$ parameters, batch size $B$, on GPUs costing $\$/\text{GPU-hour}$:

**Tokens per second per GPU** (decode, memory-bandwidth-bound):

$$\text{tok/s/GPU} = \frac{\text{memory bandwidth}}{2P / B} = \frac{B \times \text{BW}}{2P}$$

(Each token needs to load all $2P$ bytes of bf16 weights. With batch $B$, amortized cost is $2P/B$ bytes per token.)

For A100 (2 TB/s BW), 7B model, batch 64:

$$\text{tok/s} = \frac{64 \times 2 \times 10^{12}}{2 \times 7 \times 10^9} = 9143 \text{ tok/s}$$

**Cost per 1M tokens** (at $\$2/\text{GPU-hour}$):

$$\text{cost} = \frac{10^6}{9143 \times 3600} \times 2 = \$0.061$$

### Cost scaling laws

| Factor | Effect on cost |
|---|---|
| 2× model size | ~2× cost (linear in params, bandwidth-bound) |
| 2× batch size | ~0.5× cost per token (until compute-bound) |
| 2× context | ~1× cost per token (KV cache memory limits batch) |
| INT4 quantization | ~0.25-0.5× cost (less memory, higher batch fits) |
| Speculative decode | ~0.4-0.6× cost (for decode-heavy workloads) |

### The prefill vs decode cost split

For a request with $n_{\text{prompt}}$ prompt tokens and $n_{\text{gen}}$ generated tokens:

- **Prefill cost**: $\propto n_{\text{prompt}} \times P$ (compute-bound, efficient)
- **Decode cost**: $\propto n_{\text{gen}} \times P$ (memory-bound, inefficient)

Prefill is ~10× cheaper per token than decode (because it's batched and compute-bound).

**Implication**: Long prompts + short outputs = cheap. Short prompts + long outputs = expensive.

This is why API pricing often charges less for input tokens than output tokens (e.g., 3:1 ratio).

---

## 23.9 Production Serving Stack

### A complete serving system:

```
[Load Balancer]
       │
       ▼
[Request Router] → assigns to GPU pool based on model, priority
       │
       ▼
[Prefill Pool]  ←─── Disaggregated prefill (compute-optimized GPUs)
       │
       ▼ (KV cache transfer)
[Decode Pool]   ←─── Disaggregated decode (memory-bandwidth-optimized)
       │
       ▼
[Token Streamer] → streams tokens to client as they're generated
       │
       ▼
[Safety Filter] → checks output before sending to user
```

### Key components:

1. **Request queue with priority**: Paid users get priority. Requests sorted by estimated cost.
2. **Dynamic batching**: Adjust batch size based on current load and KV cache memory.
3. **Token streaming**: Send each token to the client as it's generated (SSE/WebSocket). Don't wait for completion.
4. **Timeout handling**: If a request takes too long, return partial output.
5. **Model redundancy**: Multiple replicas for fault tolerance. If one GPU fails, route to another.
6. **A/B testing**: Route % of traffic to new model versions for evaluation.

### Latency targets:

| Metric | Target (consumer) | Target (API) |
|---|---|---|
| TTFT (time to first token) | < 1s | < 500ms |
| ITL (inter-token latency) | < 50ms (~20 tok/s) | < 30ms |
| Total for 500-token response | < 25s | < 15s |

---

## 23.10 Advanced: Prefix Caching and Prompt Caching

### System prompt caching

Most requests share the same system prompt. Instead of computing the KV cache for the system prompt on every request:

1. Precompute the KV cache for the system prompt
2. Store it in GPU memory
3. For each request, start with the cached KV and only process the user message

**Savings**: If system prompt is 1000 tokens and user message is 100 tokens, prefill is 10× faster.

### Radix attention (SGLang)

Generalize prefix caching to arbitrary prefix sharing:

- Build a **radix tree** of all active KV caches
- When a new request arrives, find the longest matching prefix
- Reuse that prefix's KV cache, only compute the remaining tokens

This is especially powerful for:
- Multi-turn conversations (each turn shares the full conversation history)
- Tree-of-thought / branching generation
- Batch evaluation of variations on the same prompt

---

## 23.11 The Inference Optimization Decision Tree

```
Is single-request latency the priority?
├─ Yes: Speculative decoding + TP across fast interconnect
│       Quantize to INT4 (AWQ/GPTQ) for weight loading speed
│       Disaggregate prefill/decode
│
└─ No (throughput is priority):
    Can you batch many requests?
    ├─ Yes (>64): Continuous batching + PagedAttention
    │              Quantize conservatively (INT8 or FP8)
    │              Maximize batch size within GPU memory
    │
    └─ No (low traffic):
        Use smallest viable model
        Quantize aggressively (INT4)
        Consider CPU/edge inference
```
