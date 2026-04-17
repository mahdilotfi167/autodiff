# 20. Building Your Own GPT from Scratch: Complete Implementation

## Motivation

Documents 13-19 covered the theory: architecture, tokenization, hyperparameters, pretraining, alignment, fine-tuning, and scaling laws. Now we **build it**. This document provides a complete, annotated implementation of a GPT model from scratch — every component derived in the previous lectures, translated into working PyTorch code.

We build a modern GPT with: RMSNorm, RoPE, SwiGLU, GQA, causal masking, and the full training loop. The code is production-style but readable — every design choice references the lecture where it was derived.

---

## 20.1 The Configuration

Every hyperparameter from Doc 15 in one place:

```python
from dataclasses import dataclass

@dataclass
class GPTConfig:
    # Architecture (Doc 13, 15)
    d_model: int = 768           # Model dimension (width of residual stream)
    n_layers: int = 12           # Number of transformer blocks (depth)
    n_heads: int = 12            # Number of query heads
    n_kv_heads: int = 4          # Number of KV heads (GQA, Doc 19)
    d_ff: int = 2048             # FFN inner dimension (≈ 8/3 * d_model for SwiGLU)
    vocab_size: int = 32000      # Vocabulary size (Doc 14)
    max_seq_len: int = 2048      # Maximum context length (Doc 15)

    # Regularization (Doc 15)
    dropout: float = 0.0         # Dropout (0 for pretraining, >0 for fine-tuning)

    # RoPE (Doc 14)
    rope_theta: float = 10000.0  # RoPE base frequency

    # Derived
    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def n_rep(self) -> int:
        """Number of query heads per KV head (for GQA)."""
        return self.n_heads // self.n_kv_heads
```

### Example configurations

```python
# ~124M parameters (GPT-2 Small scale)
small_config = GPTConfig(
    d_model=768, n_layers=12, n_heads=12, n_kv_heads=12,
    d_ff=3072, vocab_size=32000, max_seq_len=2048
)

# ~1.3B parameters (GPT-2 XL scale, with modern architecture)
medium_config = GPTConfig(
    d_model=2048, n_layers=24, n_heads=16, n_kv_heads=4,
    d_ff=5504, vocab_size=32000, max_seq_len=4096
)

# ~7B parameters (LLaMA-7B scale)
large_config = GPTConfig(
    d_model=4096, n_layers=32, n_heads=32, n_kv_heads=8,
    d_ff=11008, vocab_size=32000, max_seq_len=4096, rope_theta=10000.0
)
```

---

## 20.2 RMSNorm (Doc 13.9)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Doc 13.9).

    Unlike LayerNorm, does NOT subtract the mean or learn a bias.
    Just divides by RMS and applies a learned scale.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  # γ (scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n, d)
        # RMS(x) = sqrt(mean(x^2) + eps)
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight
```

---

## 20.3 Rotary Position Embeddings — RoPE (Doc 14.9)

```python
class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (Doc 14.9).

    Encodes position by rotating Q and K vectors. The dot product
    Q_m · K_n then naturally depends on relative position (m - n).
    """
    def __init__(self, head_dim: int, max_seq_len: int, theta: float = 10000.0):
        super().__init__()
        # Precompute frequency bands: θ_i = theta^(-2i/d) for i = 0, ..., d/2 - 1
        freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        # Precompute position × frequency for positions 0..max_seq_len-1
        positions = torch.arange(max_seq_len)
        angles = positions[:, None] * freqs[None, :]  # (max_seq_len, head_dim/2)
        # Store cos and sin (not learnable parameters, but persistent buffers)
        self.register_buffer('cos_cached', torch.cos(angles), persistent=False)
        self.register_buffer('sin_cached', torch.sin(angles), persistent=False)

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        """Apply RoPE to x.

        x: (B, n_heads, seq_len, head_dim)
        start_pos: position offset (for KV cache during generation)
        """
        seq_len = x.shape[2]
        cos = self.cos_cached[start_pos:start_pos + seq_len]  # (seq_len, head_dim/2)
        sin = self.sin_cached[start_pos:start_pos + seq_len]

        # Reshape for broadcasting: (1, 1, seq_len, head_dim/2)
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        # Split into even/odd pairs and apply rotation (Doc 14.9)
        x_even = x[..., 0::2]  # (B, h, n, head_dim/2)
        x_odd  = x[..., 1::2]

        # Rotation: [x_even, x_odd] → [x_even·cos - x_odd·sin, x_even·sin + x_odd·cos]
        out_even = x_even * cos - x_odd * sin
        out_odd  = x_even * sin + x_odd * cos

        # Interleave back to original shape
        out = torch.stack([out_even, out_odd], dim=-1).flatten(-2)
        return out
```

---

## 20.4 Grouped-Query Attention with KV Cache (Doc 9, 10, 13, 19)

```python
class CausalGroupedQueryAttention(nn.Module):
    """Multi-head attention with Grouped-Query Attention (GQA).

    - n_heads query heads, n_kv_heads KV heads (Doc 19.2)
    - Causal masking (Doc 13.2)
    - RoPE applied to Q and K (Doc 14.9)
    - KV cache for efficient autoregressive generation (Doc 19.4)
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.n_rep = config.n_rep
        self.head_dim = config.head_dim

        # Q/K/V projections (Doc 9.5)
        # Q: full n_heads. K, V: only n_kv_heads (GQA saves parameters + KV cache)
        self.wq = nn.Linear(config.d_model, config.n_heads * config.head_dim, bias=False)
        self.wk = nn.Linear(config.d_model, config.n_kv_heads * config.head_dim, bias=False)
        self.wv = nn.Linear(config.d_model, config.n_kv_heads * config.head_dim, bias=False)
        self.wo = nn.Linear(config.n_heads * config.head_dim, config.d_model, bias=False)

        self.rope = RotaryEmbedding(config.head_dim, config.max_seq_len, config.rope_theta)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int = 0,
        k_cache: torch.Tensor | None = None,
        v_cache: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: (B, seq_len, d_model)
        start_pos: position offset for KV cache
        Returns: (output, new_k_cache, new_v_cache)
        """
        B, seq_len, _ = x.shape

        # Compute Q, K, V projections (Doc 9.5, 10.2)
        q = self.wq(x).view(B, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        # q: (B, n_heads, seq_len, head_dim)
        # k, v: (B, n_kv_heads, seq_len, head_dim)

        # Apply RoPE to Q and K (NOT V — Doc 14.9)
        q = self.rope(q, start_pos)
        k = self.rope(k, start_pos)

        # Update KV cache for autoregressive generation (Doc 19.4)
        if k_cache is not None:
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)
        new_k_cache = k
        new_v_cache = v

        # Repeat KV heads to match number of query heads (GQA, Doc 19.2)
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)  # (B, n_heads, kv_len, head_dim)
            v = v.repeat_interleave(self.n_rep, dim=1)

        kv_len = k.shape[2]

        # Scaled dot-product attention (Doc 9.4-9.6)
        # scores[i,j] = q_i · k_j / sqrt(d_k)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # scores: (B, n_heads, seq_len, kv_len)

        # Causal mask: prevent attending to future positions (Doc 13.2)
        # During generation (seq_len=1), no masking needed (attending to all past tokens)
        if seq_len > 1:
            causal_mask = torch.triu(
                torch.full((seq_len, kv_len), float('-inf'), device=x.device),
                diagonal=kv_len - seq_len + 1
            )
            scores = scores + causal_mask.unsqueeze(0).unsqueeze(0)

        # Softmax → attention weights (Doc 9.3)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Weighted sum of values (Doc 9.6)
        out = attn_weights @ v  # (B, n_heads, seq_len, head_dim)

        # Concatenate heads and project (Doc 10.5)
        out = out.transpose(1, 2).contiguous().view(B, seq_len, -1)
        out = self.wo(out)  # (B, seq_len, d_model)

        return out, new_k_cache, new_v_cache
```

---

## 20.5 SwiGLU Feed-Forward Network (Doc 13.7)

```python
class SwiGLUFFN(nn.Module):
    """Feed-Forward Network with SwiGLU activation (Doc 13.7).

    FFN(x) = (x·W1 ⊙ SiLU(x·W_gate)) · W2

    Three weight matrices instead of two, with d_ff ≈ 8/3 · d_model
    to match parameter count of standard 4d FFN.
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.w1 = nn.Linear(config.d_model, config.d_ff, bias=False)       # "up" projection
        self.w_gate = nn.Linear(config.d_model, config.d_ff, bias=False)   # gate projection
        self.w2 = nn.Linear(config.d_ff, config.d_model, bias=False)       # "down" projection
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n, d_model)
        # SwiGLU: element-wise product of linear transform and gated transform
        return self.dropout(self.w2(self.w1(x) * F.silu(self.w_gate(x))))
```

---

## 20.6 The Transformer Block (Doc 11, 13.4)

```python
class TransformerBlock(nn.Module):
    """Single transformer block with Pre-Norm residual connections (Doc 13.4).

    y = x + Attention(RMSNorm(x))    # Attention sub-layer
    y = y + FFN(RMSNorm(y))          # FFN sub-layer

    Pre-Norm: gradient has clean identity path through residual stream.
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = RMSNorm(config.d_model)
        self.attn = CausalGroupedQueryAttention(config)
        self.ln2 = RMSNorm(config.d_model)
        self.ffn = SwiGLUFFN(config)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int = 0,
        k_cache: torch.Tensor | None = None,
        v_cache: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Attention with residual (Doc 11.1, 13.4)
        residual = x
        x = self.ln1(x)
        attn_out, k_cache, v_cache = self.attn(x, start_pos, k_cache, v_cache)
        x = residual + attn_out

        # FFN with residual
        x = x + self.ffn(self.ln2(x))

        return x, k_cache, v_cache
```

---

## 20.7 The Complete GPT Model (Doc 13.10)

```python
class GPT(nn.Module):
    """Complete GPT model (Doc 13.10).

    Architecture: Embedding → N × TransformerBlock → RMSNorm → Linear(→vocab)
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        # Token embedding (Doc 14.5)
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])

        # Final normalization (Doc 13.4 — needed because Pre-Norm leaves
        # the output of the last block un-normalized)
        self.ln_f = RMSNorm(config.d_model)

        # LM head: project to vocabulary (Doc 13.5)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying (optional, Doc 13.5): share embedding and unembedding weights
        # self.lm_head.weight = self.tok_emb.weight

        # Initialize weights (Doc 15.13)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """Initialize weights (Doc 15.13).

        Linear layers: N(0, 0.02). Embeddings: N(0, 0.02).
        Output projections (wo, w2): scaled by 1/sqrt(2*n_layers) to keep
        residual stream variance bounded.
        """
        if isinstance(module, nn.Linear):
            std = 0.02
            # Scale output projections (Doc 15.13)
            if hasattr(module, '_is_residual_proj'):
                std *= (2 * self.config.n_layers) ** -0.5
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        token_ids: torch.Tensor,
        start_pos: int = 0,
        kv_caches: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """
        token_ids: (B, seq_len) — integer token IDs
        start_pos: position offset for KV cache (generation mode)
        kv_caches: list of (k_cache, v_cache) per layer, or None

        Returns: (logits, new_kv_caches)
          logits: (B, seq_len, vocab_size)
          new_kv_caches: updated KV caches for all layers
        """
        B, seq_len = token_ids.shape

        # Token embedding (Doc 14.5) — positions handled by RoPE inside attention
        x = self.tok_emb(token_ids)  # (B, seq_len, d_model)

        # Pass through N transformer blocks
        new_kv_caches = []
        for i, block in enumerate(self.blocks):
            k_cache = kv_caches[i][0] if kv_caches is not None else None
            v_cache = kv_caches[i][1] if kv_caches is not None else None
            x, new_k, new_v = block(x, start_pos, k_cache, v_cache)
            new_kv_caches.append((new_k, new_v))

        # Final normalization + LM head
        x = self.ln_f(x)                # (B, seq_len, d_model)
        logits = self.lm_head(x)        # (B, seq_len, vocab_size)

        return logits, new_kv_caches

    def count_parameters(self) -> dict:
        """Count parameters by component (Doc 13.8)."""
        counts = {
            'embedding': sum(p.numel() for p in self.tok_emb.parameters()),
            'attention': sum(
                sum(p.numel() for p in block.attn.parameters())
                for block in self.blocks
            ),
            'ffn': sum(
                sum(p.numel() for p in block.ffn.parameters())
                for block in self.blocks
            ),
            'norm': sum(
                sum(p.numel() for p in block.ln1.parameters()) +
                sum(p.numel() for p in block.ln2.parameters())
                for block in self.blocks
            ) + sum(p.numel() for p in self.ln_f.parameters()),
            'lm_head': sum(p.numel() for p in self.lm_head.parameters()),
        }
        counts['total'] = sum(counts.values())
        return counts
```

---

## 20.8 The Training Loop (Doc 16.7)

```python
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


class TextDataset(Dataset):
    """Simple packed text dataset (Doc 16.9).

    Loads pre-tokenized data and returns fixed-length chunks.
    """
    def __init__(self, token_ids: torch.Tensor, seq_len: int):
        self.seq_len = seq_len
        # Truncate to multiple of seq_len
        n_tokens = (len(token_ids) // seq_len) * seq_len
        self.data = token_ids[:n_tokens].view(-1, seq_len)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        return x


def get_cosine_schedule(optimizer, warmup_steps, total_steps, min_lr_ratio=0.1):
    """Cosine learning rate schedule with warmup (Doc 15.8)."""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps  # Linear warmup
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return min_lr_ratio + 0.5 * (1 - min_lr_ratio) * (1 + math.cos(math.pi * progress))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train(
    model: GPT,
    train_tokens: torch.Tensor,
    config: GPTConfig,
    # Training hyperparameters (Doc 15)
    lr: float = 3e-4,
    min_lr_ratio: float = 0.1,
    weight_decay: float = 0.1,
    beta1: float = 0.9,
    beta2: float = 0.95,
    max_grad_norm: float = 1.0,
    batch_size: int = 32,
    total_steps: int = 10000,
    warmup_steps: int = 500,
    eval_interval: int = 250,
    log_interval: int = 10,
    device: str = 'cuda',
    gradient_accumulation_steps: int = 1,
):
    """Complete training loop (Doc 16.7)."""
    model = model.to(device)
    model.train()

    # Dataset and dataloader (Doc 16.9)
    dataset = TextDataset(train_tokens, config.max_seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    data_iter = iter(dataloader)

    # Optimizer: AdamW with decoupled weight decay (Doc 15.9, 16.3)
    # Separate weight-decay and no-weight-decay groups
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.dim() >= 2:  # Weight matrices get weight decay
                decay_params.append(param)
            else:  # Biases, norms don't (Doc 15.9)
                no_decay_params.append(param)

    optimizer = optim.AdamW([
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0},
    ], lr=lr, betas=(beta1, beta2))

    # Learning rate schedule (Doc 15.8)
    scheduler = get_cosine_schedule(optimizer, warmup_steps, total_steps, min_lr_ratio)

    # Training loop
    step = 0
    total_tokens = 0

    while step < total_steps:
        optimizer.zero_grad()
        accum_loss = 0.0

        for micro_step in range(gradient_accumulation_steps):
            # Get batch (loop dataloader if exhausted)
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            batch = batch.to(device)

            # Forward pass (Doc 13.10, 16.7)
            # Input: tokens[:-1], Labels: tokens[1:]
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits, _ = model(batch[:, :-1])
                loss = F.cross_entropy(
                    logits.reshape(-1, config.vocab_size),
                    batch[:, 1:].reshape(-1)
                )
                loss = loss / gradient_accumulation_steps

            loss.backward()
            accum_loss += loss.item()

        # Gradient clipping (Doc 15.12)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        # Optimizer step
        optimizer.step()
        scheduler.step()
        step += 1
        total_tokens += batch_size * gradient_accumulation_steps * (config.max_seq_len - 1)

        # Logging
        if step % log_interval == 0:
            current_lr = scheduler.get_last_lr()[0]
            ppl = math.exp(min(accum_loss, 20))  # Cap to avoid overflow
            print(f"Step {step:>6d} | Loss {accum_loss:.4f} | PPL {ppl:.1f} | "
                  f"LR {current_lr:.2e} | Grad norm {grad_norm:.2f} | "
                  f"Tokens {total_tokens:,}")

    return model
```

---

## 20.9 Text Generation (Inference) with KV Cache (Doc 19.4)

```python
@torch.no_grad()
def generate(
    model: GPT,
    prompt_tokens: torch.Tensor,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    device: str = 'cuda',
) -> torch.Tensor:
    """Autoregressive text generation with KV cache (Doc 19.4).

    Uses top-k and top-p (nucleus) sampling for quality generation.
    """
    model.eval()
    model = model.to(device)

    # Start with the prompt
    tokens = prompt_tokens.unsqueeze(0).to(device)  # (1, prompt_len)

    # Prefill: process the entire prompt at once
    logits, kv_caches = model(tokens, start_pos=0)
    # logits: (1, prompt_len, V)

    # Sample the first new token from the last position's logits
    next_token = _sample(logits[:, -1, :], temperature, top_k, top_p)
    generated = [next_token.item()]
    current_pos = tokens.shape[1]

    # Decode: generate one token at a time using KV cache
    for _ in range(max_new_tokens - 1):
        # Feed only the new token (KV cache has all history)
        logits, kv_caches = model(
            next_token.unsqueeze(0),  # (1, 1)
            start_pos=current_pos,
            kv_caches=kv_caches,
        )
        current_pos += 1

        next_token = _sample(logits[:, -1, :], temperature, top_k, top_p)
        generated.append(next_token.item())

        # Stop on EOS token
        if next_token.item() == 2:  # Assuming EOS = 2
            break

    return torch.tensor(generated)


def _sample(
    logits: torch.Tensor,
    temperature: float,
    top_k: int,
    top_p: float,
) -> torch.Tensor:
    """Sample from logits with temperature, top-k, and top-p filtering."""
    # Temperature scaling
    logits = logits / temperature

    # Top-k filtering
    if top_k > 0:
        top_k_values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < top_k_values[..., -1:]] = float('-inf')

    # Top-p (nucleus) filtering
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        # Remove tokens with cumulative probability above threshold
        sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
        sorted_logits[sorted_mask] = float('-inf')
        # Scatter back to original ordering
        logits = sorted_logits.scatter(-1, sorted_indices, sorted_logits)

    # Sample
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)
```

---

## 20.10 Putting It All Together: A Complete Training Script

```python
def main():
    """Train a small GPT on sample data."""
    import os

    # ─── Configuration ───────────────────────────────────────────────────
    # Small model for demonstration (fits on a single GPU)
    config = GPTConfig(
        d_model=512,
        n_layers=8,
        n_heads=8,
        n_kv_heads=4,      # GQA: 2 query heads per KV head
        d_ff=1376,          # ≈ 8/3 * 512, rounded
        vocab_size=32000,
        max_seq_len=512,
        dropout=0.0,        # No dropout for pretraining (Doc 15.10)
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ─── Model ───────────────────────────────────────────────────────────
    model = GPT(config)
    param_counts = model.count_parameters()
    print("Parameter counts:")
    for k, v in param_counts.items():
        print(f"  {k:>12s}: {v:>12,d}")
    print(f"  {'TOTAL':>12s}: {param_counts['total']:>12,d}")

    # ─── Data ────────────────────────────────────────────────────────────
    # In production, load pre-tokenized data (Doc 16.9)
    # Here, generate random data for demonstration
    print("\nGenerating synthetic training data...")
    n_tokens = 1_000_000  # 1M tokens
    train_tokens = torch.randint(0, config.vocab_size, (n_tokens,))

    # ─── Train ───────────────────────────────────────────────────────────
    print(f"\nTraining on {n_tokens:,} tokens...")
    model = train(
        model=model,
        train_tokens=train_tokens,
        config=config,
        lr=3e-4,
        weight_decay=0.1,
        batch_size=16,
        total_steps=1000,
        warmup_steps=100,
        device=device,
        gradient_accumulation_steps=4,
    )

    # ─── Generate ────────────────────────────────────────────────────────
    print("\nGenerating text...")
    prompt = torch.tensor([1, 100, 200, 300])  # Example prompt tokens
    output = generate(model, prompt, max_new_tokens=50, device=device)
    print(f"Generated tokens: {output.tolist()}")

    # ─── Save ────────────────────────────────────────────────────────────
    save_path = "gpt_checkpoint.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
    }, save_path)
    print(f"\nModel saved to {save_path}")


if __name__ == '__main__':
    main()
```

---

## 20.11 Extending to Real Training

### What this demo lacks (and how to add it)

| Feature | What to add | Reference |
|---|---|---|
| Real tokenizer | `sentencepiece` or `tiktoken` library | Doc 14.2 |
| Real data | Web crawl + filtering + dedup pipeline | Doc 16.1 |
| Multi-GPU | `torch.distributed` + FSDP/DDP | Doc 16.4 |
| Mixed precision | `torch.autocast` + GradScaler (already included) | Doc 15.14 |
| FlashAttention | `flash_attn` library or `F.scaled_dot_product_attention` | Doc 19.3 |
| Activation checkpointing | `torch.utils.checkpoint` | Doc 16.6 |
| Wandb logging | `wandb.log()` for loss, LR, grad norm | Doc 16.11 |
| Evaluation | Periodic eval on held-out data | Doc 16.11 |

### Using PyTorch's built-in FlashAttention

Replace the manual attention implementation with:

```python
# Inside CausalGroupedQueryAttention.forward():
# Replace the manual scores → softmax → matmul with:
out = F.scaled_dot_product_attention(
    q, k, v,
    attn_mask=None,
    dropout_p=self.dropout.p if self.training else 0.0,
    is_causal=True,  # Automatically applies causal mask
)
```

This automatically uses FlashAttention-2 on supported hardware (A100+, H100+).

### Adding LoRA for fine-tuning (Doc 18.3)

```python
class LoRALinear(nn.Module):
    """Add LoRA to an existing linear layer (Doc 18.3)."""
    def __init__(self, linear: nn.Linear, rank: int = 16, alpha: int = 16):
        super().__init__()
        self.linear = linear
        self.linear.weight.requires_grad = False  # Freeze base

        in_features = linear.in_features
        out_features = linear.out_features
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.scaling = alpha / rank

    def forward(self, x):
        base = self.linear(x)
        lora = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return base + lora


def add_lora(model: GPT, rank: int = 16, alpha: int = 16):
    """Apply LoRA to all attention and FFN projections."""
    for block in model.blocks:
        block.attn.wq = LoRALinear(block.attn.wq, rank, alpha)
        block.attn.wk = LoRALinear(block.attn.wk, rank, alpha)
        block.attn.wv = LoRALinear(block.attn.wv, rank, alpha)
        block.attn.wo = LoRALinear(block.attn.wo, rank, alpha)
        block.ffn.w1 = LoRALinear(block.ffn.w1, rank, alpha)
        block.ffn.w_gate = LoRALinear(block.ffn.w_gate, rank, alpha)
        block.ffn.w2 = LoRALinear(block.ffn.w2, rank, alpha)
    return model
```

---

## 20.12 Verification: Checking Your Implementation

### Shape checks

Always verify shapes at every stage:

```python
def verify_shapes(config):
    model = GPT(config)
    B, n = 2, config.max_seq_len
    x = torch.randint(0, config.vocab_size, (B, n))

    logits, caches = model(x)

    assert logits.shape == (B, n - 0, config.vocab_size), \
        f"Expected ({B}, {n}, {config.vocab_size}), got {logits.shape}"
    assert len(caches) == config.n_layers
    for k_cache, v_cache in caches:
        assert k_cache.shape == (B, config.n_kv_heads, n, config.head_dim)
        assert v_cache.shape == (B, config.n_kv_heads, n, config.head_dim)
    print("All shape checks passed!")
```

### Gradient check

Verify gradients flow through the entire model:

```python
def verify_gradients(config):
    model = GPT(config)
    x = torch.randint(0, config.vocab_size, (1, 32))
    logits, _ = model(x[:, :-1])
    loss = F.cross_entropy(logits.reshape(-1, config.vocab_size), x[:, 1:].reshape(-1))
    loss.backward()

    for name, param in model.named_parameters():
        if param.grad is None:
            print(f"WARNING: {name} has no gradient!")
        elif param.grad.abs().sum() == 0:
            print(f"WARNING: {name} has zero gradient!")
    print("Gradient check complete!")
```

### Overfitting a single batch

The most important sanity check: the model should be able to memorize a single small batch:

```python
def overfit_single_batch(config, steps=200):
    model = GPT(config).cuda()
    x = torch.randint(0, config.vocab_size, (4, 64)).cuda()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    for step in range(steps):
        logits, _ = model(x[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, config.vocab_size), x[:, 1:].reshape(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step % 50 == 0:
            print(f"Step {step}: loss = {loss.item():.4f}")

    assert loss.item() < 0.1, f"Failed to overfit! Final loss: {loss.item()}"
    print("✓ Successfully overfit a single batch — model can learn!")
```

---

## 20.13 Summary: From Theory to Code

This document translated every component from the lecture series into working code:

| Lecture | Component | Code section |
|---|---|---|
| Doc 13 | GPT architecture, causal mask | 20.4, 20.7 |
| Doc 13.7 | SwiGLU FFN | 20.5 |
| Doc 13.9 | RMSNorm | 20.2 |
| Doc 14.9 | RoPE | 20.3 |
| Doc 15 | Hyperparameters, initialization | 20.1, 20.7 |
| Doc 16 | Training loop, optimizer, schedule | 20.8 |
| Doc 18.3 | LoRA fine-tuning | 20.11 |
| Doc 19.2 | GQA | 20.4 |
| Doc 19.4 | KV cache | 20.4, 20.9 |

### The complete model in ~300 lines

The entire GPT — from embedding to generation — is implemented in roughly 300 lines of Python. The architecture is simple. The complexity in real-world LLMs comes from:

1. **Scale**: thousands of GPUs, petabytes of data, months of training
2. **Engineering**: distributed training, fault tolerance, efficient kernels
3. **Data**: the curation pipeline is more complex than the model
4. **Alignment**: post-training is where raw capability becomes usefulness

But the core model — the mathematical object you're training — is exactly what's written here.
