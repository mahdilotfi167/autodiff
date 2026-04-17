# Appendix C: Key Derivations & Proofs

## Purpose

This appendix collects the essential mathematical derivations that are referenced throughout the lecture series. For each derivation, we provide the **full step-by-step math** — no hand-waving, no "it can be shown that." These are the derivations you should be able to reproduce on a whiteboard at an interview.

---

## C.1 Softmax Gradient

### Setup

Given logits $z \in \mathbb{R}^V$ and the softmax function:

$$p_i = \frac{\exp(z_i)}{\sum_{k=1}^{V} \exp(z_k)}$$

We want $\frac{\partial p_i}{\partial z_j}$.

### Derivation

**Case 1: $i = j$**

$$\frac{\partial p_i}{\partial z_i} = \frac{\partial}{\partial z_i}\left[\frac{\exp(z_i)}{S}\right] \quad \text{where } S = \sum_k \exp(z_k)$$

By the quotient rule:

$$= \frac{\exp(z_i) \cdot S - \exp(z_i) \cdot \exp(z_i)}{S^2} = \frac{\exp(z_i)}{S} - \frac{\exp(z_i)^2}{S^2} = p_i - p_i^2 = p_i(1 - p_i)$$

**Case 2: $i \neq j$**

$$\frac{\partial p_i}{\partial z_j} = \frac{\partial}{\partial z_j}\left[\frac{\exp(z_i)}{S}\right] = \frac{0 - \exp(z_i) \cdot \exp(z_j)}{S^2} = -\frac{\exp(z_i)}{S} \cdot \frac{\exp(z_j)}{S} = -p_i p_j$$

**Compact form (Jacobian)**:

$$\frac{\partial p_i}{\partial z_j} = p_i(\delta_{ij} - p_j)$$

Or in matrix form:

$$\frac{\partial \mathbf{p}}{\partial \mathbf{z}} = \text{diag}(\mathbf{p}) - \mathbf{p}\mathbf{p}^T$$

### Verification

- Row sums: $\sum_j \frac{\partial p_i}{\partial z_j} = p_i(1-p_i) + \sum_{j \neq i}(-p_i p_j) = p_i - p_i \sum_j p_j = p_i - p_i = 0$ ✓

This makes sense: since $\sum_i p_i = 1$ is constant, perturbing any $z_j$ must redistribute probability (total change sums to zero).

---

## C.2 Cross-Entropy Gradient (Combined with Softmax)

### Setup

Cross-entropy loss for true class $y$ (one-hot encoded):

$$\mathcal{L} = -\sum_{i} y_i \log p_i = -\log p_y$$

where $p = \text{softmax}(z)$.

### Derivation

We want $\frac{\partial \mathcal{L}}{\partial z_j}$ — the gradient of the loss with respect to the logits.

By chain rule:

$$\frac{\partial \mathcal{L}}{\partial z_j} = \sum_i \frac{\partial \mathcal{L}}{\partial p_i} \cdot \frac{\partial p_i}{\partial z_j}$$

First: $\frac{\partial \mathcal{L}}{\partial p_i} = -\frac{y_i}{p_i}$

Then:

$$\frac{\partial \mathcal{L}}{\partial z_j} = \sum_i \left(-\frac{y_i}{p_i}\right) \cdot p_i(\delta_{ij} - p_j) = -\sum_i y_i(\delta_{ij} - p_j)$$

$$= -y_j + p_j \sum_i y_i = -y_j + p_j \cdot 1 = p_j - y_j$$

### Result

$$\boxed{\frac{\partial \mathcal{L}}{\partial z_j} = p_j - y_j}$$

This is remarkably simple: the gradient at each logit is just **(predicted probability − true probability)**.

For the true class $y$: gradient $= p_y - 1$ (negative, pushes logit up)
For all other classes: gradient $= p_j$ (positive, pushes logit down)

### Why this matters

This simplicity is why softmax + cross-entropy is universally used. The gradient is:
1. **Numerically stable** (no division by small probabilities in the final expression)
2. **Bounded** (always in $[-1, 1]$)
3. **Naturally calibrated** (the gradient is proportional to the error)

---

## C.3 Attention Backward Pass: Gradients Through Q, K, V

### Setup

Single-head attention: $\text{Attn} = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V = PV$

where $P = \text{softmax}(S/\sqrt{d_k})$ and $S = QK^T$.

Given $\frac{\partial \mathcal{L}}{\partial O}$ (gradient flowing from downstream), compute $\frac{\partial \mathcal{L}}{\partial Q}$, $\frac{\partial \mathcal{L}}{\partial K}$, $\frac{\partial \mathcal{L}}{\partial V}$.

Dimensions: $Q, K \in \mathbb{R}^{N \times d_k}$, $V \in \mathbb{R}^{N \times d_v}$, $O \in \mathbb{R}^{N \times d_v}$, $P \in \mathbb{R}^{N \times N}$.

### Step 1: Gradient w.r.t. V

$O = PV$, so:

$$\frac{\partial \mathcal{L}}{\partial V} = P^T \frac{\partial \mathcal{L}}{\partial O}$$

This is straightforward matrix calculus: if $O = PV$, then $dO = P \, dV$, so $\text{tr}(\frac{\partial \mathcal{L}}{\partial O}^T dO) = \text{tr}(\frac{\partial \mathcal{L}}{\partial O}^T P \, dV) = \text{tr}((P^T \frac{\partial \mathcal{L}}{\partial O})^T dV)$.

### Step 2: Gradient w.r.t. P

$O = PV$, so:

$$\frac{\partial \mathcal{L}}{\partial P} = \frac{\partial \mathcal{L}}{\partial O} V^T$$

### Step 3: Gradient through softmax

This is the hard part. For each row $i$ of $P$:

$$P_{i:} = \text{softmax}(S_{i:} / \sqrt{d_k})$$

From C.1, the Jacobian of softmax for row $i$ is:

$$J_i = \text{diag}(P_{i:}) - P_{i:} P_{i:}^T$$

So the gradient of the loss w.r.t. row $i$ of $S$:

$$\frac{\partial \mathcal{L}}{\partial S_{i:}} = \frac{1}{\sqrt{d_k}} \cdot \frac{\partial \mathcal{L}}{\partial P_{i:}} \cdot J_i$$

Let $g_i = \frac{\partial \mathcal{L}}{\partial P_{i:}}$ (row $i$ of $\frac{\partial \mathcal{L}}{\partial P}$). Then:

$$\frac{\partial \mathcal{L}}{\partial S_{i:}} = \frac{1}{\sqrt{d_k}}(g_i \odot P_{i:} - P_{i:} \cdot (g_i \cdot P_{i:}^T))$$

$$= \frac{1}{\sqrt{d_k}} P_{i:} \odot (g_i - \langle g_i, P_{i:} \rangle \mathbf{1})$$

In matrix form for all rows:

$$\frac{\partial \mathcal{L}}{\partial S} = \frac{1}{\sqrt{d_k}} P \odot \left(\frac{\partial \mathcal{L}}{\partial P} - \text{rowsum}\left(P \odot \frac{\partial \mathcal{L}}{\partial P}\right) \cdot \mathbf{1}^T\right)$$

where $\text{rowsum}(\cdot)$ sums each row into a column vector.

### Step 4: Gradients w.r.t. Q and K

$S = QK^T$, so:

$$\frac{\partial \mathcal{L}}{\partial Q} = \frac{\partial \mathcal{L}}{\partial S} \cdot K$$

$$\frac{\partial \mathcal{L}}{\partial K} = \left(\frac{\partial \mathcal{L}}{\partial S}\right)^T \cdot Q$$

### Complete backward pass summary

1. $\frac{\partial \mathcal{L}}{\partial V} = P^T \frac{\partial \mathcal{L}}{\partial O}$ — $O(N^2 d_v)$
2. $\frac{\partial \mathcal{L}}{\partial P} = \frac{\partial \mathcal{L}}{\partial O} V^T$ — $O(N^2 d_v)$
3. Softmax backward: $\frac{\partial \mathcal{L}}{\partial S}$ from $\frac{\partial \mathcal{L}}{\partial P}$ and $P$ — $O(N^2)$
4. $\frac{\partial \mathcal{L}}{\partial Q} = \frac{\partial \mathcal{L}}{\partial S} K$ — $O(N^2 d_k)$
5. $\frac{\partial \mathcal{L}}{\partial K} = (\frac{\partial \mathcal{L}}{\partial S})^T Q$ — $O(N^2 d_k)$

Total: $O(N^2(d_k + d_v))$ — same order as the forward pass. The backward of attention costs roughly $2\times$ the forward.

---

## C.4 LayerNorm Gradient

### Setup

$$\text{LayerNorm}(x) = \gamma \odot \frac{x - \mu}{\sigma + \epsilon} + \beta$$

where $\mu = \frac{1}{d}\sum_i x_i$ and $\sigma = \sqrt{\frac{1}{d}\sum_i(x_i - \mu)^2}$.

Let $\hat{x} = \frac{x - \mu}{\sigma + \epsilon}$ (normalized input).

### Derivation

Given $\frac{\partial \mathcal{L}}{\partial y}$ (upstream gradient), where $y = \gamma \odot \hat{x} + \beta$:

**Step 1**: $\frac{\partial \mathcal{L}}{\partial \gamma_i} = \sum_{\text{batch}} \frac{\partial \mathcal{L}}{\partial y_i} \hat{x}_i$ and $\frac{\partial \mathcal{L}}{\partial \beta_i} = \sum_{\text{batch}} \frac{\partial \mathcal{L}}{\partial y_i}$

**Step 2**: $\frac{\partial \mathcal{L}}{\partial \hat{x}_i} = \frac{\partial \mathcal{L}}{\partial y_i} \gamma_i$

**Step 3**: Now we need $\frac{\partial \hat{x}}{\partial x}$, which is the hard part because $\mu$ and $\sigma$ both depend on all elements of $x$.

$$\frac{\partial \hat{x}_i}{\partial x_j} = \frac{1}{\sigma}\left(\delta_{ij} - \frac{1}{d} - \frac{\hat{x}_i \hat{x}_j}{d}\right)$$

Derivation of this:

$$\hat{x}_i = \frac{x_i - \mu}{\sigma}$$

$$\frac{\partial \hat{x}_i}{\partial x_j} = \frac{1}{\sigma}\frac{\partial (x_i - \mu)}{\partial x_j} - \frac{x_i - \mu}{\sigma^2}\frac{\partial \sigma}{\partial x_j}$$

$$= \frac{1}{\sigma}\left(\delta_{ij} - \frac{1}{d}\right) - \frac{\hat{x}_i}{\sigma} \cdot \frac{x_j - \mu}{d\sigma}$$

$$= \frac{1}{\sigma}\left(\delta_{ij} - \frac{1}{d} - \frac{\hat{x}_i \hat{x}_j}{d}\right)$$

**Step 4**: Putting it together:

$$\frac{\partial \mathcal{L}}{\partial x_i} = \frac{1}{\sigma}\left(\frac{\partial \mathcal{L}}{\partial \hat{x}_i} - \frac{1}{d}\sum_j \frac{\partial \mathcal{L}}{\partial \hat{x}_j} - \frac{\hat{x}_i}{d}\sum_j \frac{\partial \mathcal{L}}{\partial \hat{x}_j}\hat{x}_j\right)$$

### PyTorch-style implementation

```python
def layernorm_backward(dy, x, gamma, epsilon=1e-5):
    d = x.shape[-1]
    mu = x.mean(-1, keepdim=True)
    sigma = x.std(-1, keepdim=True, unbiased=False)
    x_hat = (x - mu) / (sigma + epsilon)
    
    # Parameter gradients
    dgamma = (dy * x_hat).sum(0)  # sum over batch
    dbeta = dy.sum(0)
    
    # Input gradient
    dx_hat = dy * gamma
    dsigma = -(dx_hat * x_hat).sum(-1, keepdim=True) / (sigma + epsilon)
    dmu = -dx_hat.sum(-1, keepdim=True) / (sigma + epsilon) + dsigma * (-2/d) * (x - mu).sum(-1, keepdim=True)
    dx = dx_hat / (sigma + epsilon) + dsigma * 2 * (x - mu) / d + dmu / d
    
    return dx, dgamma, dbeta
```

---

## C.5 LoRA Update Equivalence

### Setup

LoRA (Low-Rank Adaptation) replaces a weight update $\Delta W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ with a low-rank factorization:

$$\Delta W = BA, \quad B \in \mathbb{R}^{d_{\text{out}} \times r}, \quad A \in \mathbb{R}^{r \times d_{\text{in}}}$$

where $r \ll \min(d_{\text{out}}, d_{\text{in}})$.

### Why this is (approximately) sufficient

**Claim**: For fine-tuning on a specific task, the full update $\Delta W^*$ that would be learned by full fine-tuning has low (or approximately low) rank.

**Evidence** (Aghajanyan et al., 2020; Hu et al., 2021): When you fine-tune a large model and compute SVD of $\Delta W^*$:

$$\Delta W^* = U \Sigma V^T$$

The singular values decay rapidly. Most of the "energy" is in the top-$r$ singular values:

$$\frac{\sum_{i=1}^{r} \sigma_i^2}{\sum_{i=1}^{d} \sigma_i^2} > 0.9 \quad \text{for } r \sim 1\text{-}64$$

This means a rank-$r$ approximation captures >90% of the full update.

### The intrinsic dimensionality argument

Aghajanyan et al. showed that fine-tuning operates in a low-dimensional subspace. If you project the gradient into a random $k$-dimensional subspace and optimize only there, you recover most of the performance for surprisingly small $k$.

For a model with $D$ parameters, the "intrinsic dimensionality" of fine-tuning is $d_{\text{intrinsic}} \ll D$:

| Model size $D$ | $d_{\text{intrinsic}}$ | Ratio |
|---|---|---|
| 100M | ~200 | 0.0002% |
| 1B | ~500 | 0.00005% |
| 10B | ~1000 | 0.00001% |

LoRA's rank-$r$ factorization per layer is a structured way to implement this low-dimensional update.

### The initialization and scaling

LoRA initializes:
- $A \sim \mathcal{N}(0, \sigma^2)$: Random Gaussian (or Kaiming)
- $B = 0$: Zero initialization

So $\Delta W = BA = 0$ at the start — the model begins at the pre-trained weights.

The forward pass becomes:

$$h = (W_0 + \frac{\alpha}{r} BA)x = W_0 x + \frac{\alpha}{r} B(Ax)$$

The factor $\frac{\alpha}{r}$ keeps the magnitude of the update invariant when changing rank:
- Doubling $r$ roughly halves the per-component magnitude of $A$ and $B$
- The $\frac{\alpha}{r}$ factor counteracts this, making the total update roughly constant

### Gradient flow through LoRA

$$\frac{\partial \mathcal{L}}{\partial A} = \frac{\alpha}{r} B^T \frac{\partial \mathcal{L}}{\partial h} x^T$$

$$\frac{\partial \mathcal{L}}{\partial B} = \frac{\alpha}{r} \frac{\partial \mathcal{L}}{\partial h} (Ax)^T$$

The gradients flow through both $A$ and $B$. Since $B$ is initialized to zero:
- At the start, $\frac{\partial \mathcal{L}}{\partial A} = 0$ (because $B=0$), so $A$ doesn't update
- But $\frac{\partial \mathcal{L}}{\partial B} \neq 0$, so $B$ starts changing first

This means in practice, $B$ adapts first, then $A$ follows. This asymmetry is fine — the product $BA$ has full rank-$r$ expressiveness.

---

## C.6 DPO Derivation from RLHF

### Setup

RLHF optimizes:

$$\max_{\pi} \mathbb{E}_{x \sim \mathcal{D}} \mathbb{E}_{y \sim \pi(\cdot|x)} [r(x, y)] - \beta D_{\text{KL}}[\pi(\cdot|x) \| \pi_{\text{ref}}(\cdot|x)]$$

where $\pi$ is the policy (model), $r$ is the reward function, $\pi_{\text{ref}}$ is the reference (SFT) model, and $\beta$ controls the KL penalty.

### Step 1: Optimal policy in closed form

The optimal policy for this KL-constrained optimization has a closed-form solution:

$$\pi^*(y|x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y|x) \exp\left(\frac{r(x,y)}{\beta}\right)$$

where $Z(x) = \sum_y \pi_{\text{ref}}(y|x) \exp(r(x,y)/\beta)$ is the partition function.

**Proof**: Write the Lagrangian and differentiate w.r.t. $\pi(y|x)$ for each $y$. The KL penalty turns the optimization into a Gibbs/Boltzmann distribution.

### Step 2: Express reward in terms of optimal policy

Rearranging the optimal policy:

$$r(x, y) = \beta \log \frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)$$

### Step 3: Bradley-Terry preference model

Human preferences follow:

$$P(y_w \succ y_l | x) = \sigma(r(x, y_w) - r(x, y_l))$$

where $\sigma$ is the sigmoid and $y_w, y_l$ are the preferred and dispreferred responses.

### Step 4: Substitute reward expression

$$r(x, y_w) - r(x, y_l) = \beta \log \frac{\pi^*(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi^*(y_l|x)}{\pi_{\text{ref}}(y_l|x)}$$

Note: $\beta \log Z(x)$ cancels! This is the key insight — the intractable partition function disappears.

### Step 5: DPO loss

$$\mathcal{L}_{\text{DPO}}(\pi_\theta) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right]$$

This is the final DPO objective. No reward model needed — just the policy $\pi_\theta$, the reference policy $\pi_{\text{ref}}$, and preference pairs $(y_w, y_l)$.

### Gradient of DPO

$$\nabla_\theta \mathcal{L}_{\text{DPO}} = -\beta \mathbb{E}\left[\sigma(-\hat{r}) \left(\nabla_\theta \log \pi_\theta(y_w|x) - \nabla_\theta \log \pi_\theta(y_l|x)\right)\right]$$

where $\hat{r} = \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}$.

**Interpretation**: The gradient:
1. Increases probability of $y_w$ and decreases probability of $y_l$
2. Weighted by $\sigma(-\hat{r})$: if the model already correctly ranks the pair (large $\hat{r}$), the gradient is small. If it incorrectly ranks them (small $\hat{r}$), the gradient is large.

This is **self-weighting**: examples the model gets wrong contribute more to the gradient.

---

## C.7 Scaling Law Derivation (Chinchilla)

### Setup

The loss as a function of model size $N$ (parameters) and data $D$ (tokens):

$$L(N, D) = \frac{A}{N^\alpha} + \frac{B}{D^\beta} + L_\infty$$

where $L_\infty$ is the irreducible loss (entropy of natural language), and $A, B, \alpha, \beta$ are fitted constants.

Empirically: $\alpha \approx 0.34$, $\beta \approx 0.28$, $L_\infty \approx 1.69$ nats.

### The compute-optimal allocation

Total compute: $C \approx 6ND$ (for a transformer, approximate FLOPs ≈ 6 × parameters × tokens).

**Optimization**: Given a fixed compute budget $C$, how should we split between $N$ and $D$?

$$\min_{N, D} L(N, D) \quad \text{subject to} \quad 6ND = C$$

Substitute $D = \frac{C}{6N}$:

$$L(N) = \frac{A}{N^\alpha} + \frac{B \cdot (6N)^\beta}{C^\beta} + L_\infty$$

Take derivative and set to zero:

$$\frac{dL}{dN} = -\frac{\alpha A}{N^{\alpha+1}} + \frac{\beta B \cdot 6^\beta}{C^\beta} N^{\beta-1} = 0$$

$$\frac{\alpha A}{N^{\alpha+1}} = \frac{\beta B \cdot 6^\beta \cdot N^{\beta-1}}{C^\beta}$$

$$N^{\alpha + \beta} = \frac{\alpha A C^\beta}{\beta B \cdot 6^\beta}$$

$$\boxed{N^* \propto C^{\beta/(\alpha+\beta)}}$$

And from $D = C/(6N)$:

$$\boxed{D^* \propto C^{\alpha/(\alpha+\beta)}}$$

With $\alpha \approx 0.34$, $\beta \approx 0.28$:
- $N^* \propto C^{0.45}$ and $D^* \propto C^{0.55}$
- $D^*/N^* \propto C^{0.1}$ — the data-to-parameter ratio grows slowly with compute

For practical values, this gives $D^* \approx 20N^*$ (the Chinchilla ratio).

### Why Chinchilla overturned Kaplan

Kaplan et al. (2020) fitted $\alpha \approx 0.076$, $\beta \approx 0.095$ (different experimental setup, limited range). Their ratio favored much larger models with less data (~5 tokens/parameter instead of ~20).

Chinchilla's key difference: training to convergence on different data amounts, rather than extrapolating from short training runs. The lesson: **scaling law exponents depend sensitively on the experimental methodology**.

---

## C.8 Rotary Position Encoding (RoPE) Derivation

### The goal

Design a position encoding such that the dot product between query and key **depends only on their relative position**:

$$\langle f(q, m), f(k, n) \rangle = g(q, k, m - n)$$

where $m, n$ are absolute positions and $f$ is the encoding function.

### The 2D case (one pair of dimensions)

Consider a query and key in 2D: $q = (q_1, q_2)$, $k = (k_1, k_2)$.

**The RoPE approach**: Apply a rotation by angle $m\theta$ to the query at position $m$:

$$f(q, m) = R(m\theta) \cdot q = \begin{pmatrix} \cos m\theta & -\sin m\theta \\ \sin m\theta & \cos m\theta \end{pmatrix} \begin{pmatrix} q_1 \\ q_2 \end{pmatrix}$$

**Proof that the dot product depends only on $m-n$**:

$$\langle f(q, m), f(k, n) \rangle = (R(m\theta) q)^T (R(n\theta) k) = q^T R(m\theta)^T R(n\theta) k$$

Since $R(\alpha)^T = R(-\alpha)$ and $R(\alpha)R(\beta) = R(\alpha + \beta)$:

$$= q^T R((n-m)\theta) k = q^T R(-(m-n)\theta) k$$

This depends only on $m - n$. ✓

### Extension to $d$ dimensions

Group the $d$ dimensions into $d/2$ pairs: $(x_1, x_2), (x_3, x_4), \ldots, (x_{d-1}, x_d)$.

Each pair $i$ gets a different frequency: $\theta_i = \theta_{\text{base}}^{-2i/d}$ where $\theta_{\text{base}} = 10000$.

The rotation matrix for position $m$:

$$R_m = \text{blockdiag}\left(R(m\theta_1), R(m\theta_2), \ldots, R(m\theta_{d/2})\right)$$

This gives a **multi-frequency** rotation — different dimension pairs encode position at different scales, similar to how sinusoidal encodings have multiple frequencies.

### Why RoPE works for context extension

The key to extending context beyond training length: the rotation frequencies $\theta_i$ determine the "wavelength" of each dimension pair:

$$\lambda_i = \frac{2\pi}{\theta_i} = 2\pi \cdot \theta_{\text{base}}^{2i/d}$$

For $\theta_{\text{base}} = 10000$ and $d = 128$:
- Dimension pair 0: $\lambda \approx 6.28$ (high frequency, encodes local position)
- Dimension pair $d/4$: $\lambda \approx 628$ (medium frequency)
- Dimension pair $d/2$: $\lambda \approx 62,800$ (low frequency, encodes global position)

To extend from context $L$ to $L' > L$, increase $\theta_{\text{base}}$ (NTK-aware RoPE):

$$\theta'_{\text{base}} = \theta_{\text{base}} \cdot \left(\frac{L'}{L}\right)^{d/(d-2)}$$

This stretches all wavelengths proportionally, allowing the model to handle longer contexts without retraining.

---

## C.9 The KV-Cache Memory Derivation

### Setup

During autoregressive generation, at step $t$ we need:
- All past keys: $K_{1:t} \in \mathbb{R}^{t \times d_k}$
- All past values: $V_{1:t} \in \mathbb{R}^{t \times d_v}$

For each attention head in each layer.

### Memory calculation

Per head, per layer:
$$M_{\text{KV}} = 2 \times t \times d_{\text{head}} \times \text{bytes\_per\_param}$$

For $H$ heads and $L$ layers:
$$M_{\text{total}} = 2 \times L \times H \times t \times d_{\text{head}} \times \text{bytes}$$

Since $H \times d_{\text{head}} = d_{\text{model}}$:

$$\boxed{M_{\text{KV}} = 2 \times L \times d_{\text{model}} \times t \times \text{bytes\_per\_param}}$$

### Concrete example (LLaMA 70B with bf16)

$L = 80$, $d_{\text{model}} = 8192$, $\text{bytes} = 2$ (bf16), $t = 8192$ context:

$$M_{\text{KV}} = 2 \times 80 \times 8192 \times 8192 \times 2 = 21.5 \text{ GB}$$

Per request! For a batch of 32 concurrent requests: $32 \times 21.5 = 688$ GB just for KV cache.

### GQA memory savings

With Grouped Query Attention (GQA), $n_{\text{kv\_heads}} < n_{\text{heads}}$:

$$M_{\text{GQA}} = 2 \times L \times n_{\text{kv\_heads}} \times d_{\text{head}} \times t \times \text{bytes}}$$

LLaMA 70B uses $n_{\text{kv\_heads}} = 8$ (vs $n_{\text{heads}} = 64$):

$$\text{Savings ratio} = \frac{n_{\text{kv\_heads}}}{n_{\text{heads}}} = \frac{8}{64} = 8\times$$

KV cache drops from 21.5 GB to ~2.7 GB per request. This is the primary motivation for GQA.

---

## C.10 Why $1/\sqrt{d_k}$ Scaling in Attention

### The problem without scaling

For random unit-variance vectors $q, k \in \mathbb{R}^{d_k}$ with entries $q_i, k_i \sim \mathcal{N}(0, 1)$:

$$q^T k = \sum_{i=1}^{d_k} q_i k_i$$

Each $q_i k_i$ has $\text{Var}(q_i k_i) = \text{Var}(q_i)\text{Var}(k_i) = 1$ (for independent zero-mean variables).

By independence: $\text{Var}(q^T k) = d_k$.

So $q^T k \sim \mathcal{N}(0, d_k)$ — the variance grows with $d_k$.

### Why this is dangerous

For large $d_k$ (e.g., 128), the logits before softmax have standard deviation $\sqrt{128} \approx 11.3$. The softmax of values with std=11.3 produces **extremely peaked distributions** — almost all mass on one key.

This means:
1. **Gradients vanish**: The softmax saturates (like sigmoid at extreme values), gradients are ~0
2. **No information mixing**: Attention becomes a hard lookup rather than soft aggregation

### The fix

Scale by $1/\sqrt{d_k}$:

$$\text{Var}\left(\frac{q^T k}{\sqrt{d_k}}\right) = \frac{\text{Var}(q^T k)}{d_k} = \frac{d_k}{d_k} = 1$$

Now the logits have unit variance regardless of $d_k$, keeping softmax in its useful (non-saturated) regime.

### Connection to initialization

This is the same principle as Xavier/He initialization: keep the variance of activations constant across layers. The $1/\sqrt{d_k}$ factor in attention is the initialization equivalent for the $QK^T$ operation.

---

## C.11 AdamW vs. Adam: The Weight Decay Distinction

### Adam update

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$
$$\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t}$$
$$\theta_{t+1} = \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

### L2 regularization in Adam (wrong approach)

Adding L2 to the loss: $\tilde{\mathcal{L}} = \mathcal{L} + \frac{\lambda}{2}\|\theta\|^2$

The gradient becomes $\tilde{g}_t = g_t + \lambda \theta_t$. Adam processes this modified gradient:

$$\tilde{m}_t = \beta_1 \tilde{m}_{t-1} + (1-\beta_1)(g_t + \lambda\theta_t)$$

The problem: the $\lambda\theta_t$ term gets divided by $\sqrt{\hat{v}_t}$, which **adapts the regularization strength per-parameter**. Parameters with large gradients get less regularization; parameters with small gradients get more. This is unintended and breaks the regularization.

### AdamW (correct approach)

Decouple weight decay from the adaptive step:

$$\theta_{t+1} = \theta_t - \eta\left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_t\right)$$

Or equivalently:

$$\theta_{t+1} = (1 - \eta\lambda)\theta_t - \eta\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

Now every parameter gets exactly the same decay rate $\eta\lambda$, regardless of its gradient history. This is the intended behavior of weight decay.

### Why this matters for large models

With SGD, L2 regularization and weight decay are equivalent. With Adam/AdaGrad, they're not. AdamW consistently outperforms Adam + L2 for transformer training, especially at large scale where the regularization difference compounds over many steps.
