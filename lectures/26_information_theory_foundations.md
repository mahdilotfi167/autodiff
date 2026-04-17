# 26. The Information-Theoretic View: Why Next-Token Prediction Is All You Need

## Motivation

Why does predicting the next token — a seemingly trivial objective — produce systems that can reason, code, translate, and explain physics? This lecture connects language modeling to information theory and shows that next-token prediction is, fundamentally, **optimal compression**, and that optimal compression requires understanding. This is the deepest "why" behind everything in this series.

---

## 26.1 The Core Insight: Prediction = Compression

### Shannon's source coding theorem (1948)

Given a source that produces symbols $x_1, x_2, \ldots$ from distribution $P$, the minimum average number of bits to encode each symbol is:

$$H(X) = -\sum_x P(x) \log_2 P(x)$$

This is the **entropy** of the source — the irreducible information content.

A model $Q$ that assigns probabilities to symbols achieves a code length of:

$$L_Q = -\sum_x P(x) \log_2 Q(x) = H(X) + D_{\text{KL}}(P \| Q)$$

The code length equals the true entropy *plus* the KL divergence (how far $Q$ is from $P$). **Minimizing code length = minimizing KL divergence = learning the true distribution.**

### Cross-entropy loss IS code length

The cross-entropy loss we minimize during training:

$$\mathcal{L} = -\frac{1}{T}\sum_{t=1}^{T} \log_2 P_\theta(x_t | x_{<t})$$

is literally the **average number of bits per token** our model needs to encode the training data. When loss decreases, the model is learning to compress text more efficiently.

### The fundamental equation of language modeling

$$\underbrace{\text{Cross-entropy loss}}_{\text{what we minimize}} = \underbrace{H(\text{language})}_{\text{irreducible entropy}} + \underbrace{D_{\text{KL}}(P_{\text{true}} \| P_\theta)}_{\text{model imperfection}}$$

- $H(\text{language})$: The true entropy of natural language (~1.0-1.5 bits per character for English). Even a perfect model can't compress below this.
- $D_{\text{KL}}$: How much our model deviates from the true distribution. Training drives this toward zero.

---

## 26.2 Compression Requires Understanding

### The Hutter Prize argument

Marcus Hutter offered a cash prize for the best compression of a 1GB Wikipedia text file. The insight: **a better compressor is a better language model, and vice versa.**

To compress "The capital of France is Paris" efficiently, a compressor must:
1. Know that this is a factual statement about geography
2. Assign high probability to "Paris" given the prefix (short code)
3. Assign low probability to "banana" (long code)

This requires **world knowledge**. A compressor that "knows" geography compresses geography-related text better than one that doesn't.

### Kolmogorov complexity: the ultimate compression

The Kolmogorov complexity $K(x)$ of string $x$ is the length of the shortest program that produces $x$:

$$K(x) = \min_{p: U(p) = x} |p|$$

where $U$ is a universal Turing machine. This is the theoretical minimum description length.

Key properties:
- **Incomputable**: You can never verify you've found the shortest program (halting problem)
- **Approximable**: LLMs approximate $K(x)$ by learning statistical regularities
- **Universal**: Up to a constant, $K(x)$ is independent of the choice of programming language

**The connection**: Language modeling loss approximates $K(x)$ from above. Better models → tighter approximation. A perfect language model would achieve the Kolmogorov complexity of natural language.

---

## 26.3 Why Larger Models Compress Better

### The Minimum Description Length (MDL) principle

The total description length of data $D$ with model $M$ is:

$$L(D, M) = \underbrace{L(M)}_{\text{model complexity}} + \underbrace{L(D | M)}_{\text{data given model}}$$

- A complex model (many parameters) has large $L(M)$ but small $L(D|M)$ (fits data well)
- A simple model has small $L(M)$ but large $L(D|M)$ (fits data poorly)
- The optimal model minimizes the **total** length

### How this explains scaling laws

For a model with $N$ parameters encoding the model costs roughly $L(M) \propto N \log N$ bits.

The data compression improves with $N$ as a power law:

$$L(D|M) \propto N^{-\alpha} \cdot |D|$$

The total:

$$L(D, M) \propto N \log N + N^{-\alpha} \cdot |D|$$

Minimizing over $N$ given a compute budget $C \propto N \cdot |D|$ recovers the **Chinchilla scaling law**: the optimal allocation is $N^* \propto C^{0.5}$, $D^* \propto C^{0.5}$.

**Scaling laws aren't empirical accidents — they follow from information-theoretic principles.**

---

## 26.4 The Information Bottleneck

### The principle

The Information Bottleneck (Tishby, 1999) says: a good representation $Z$ of input $X$ for predicting output $Y$ should:

1. **Maximize** information about the target: $I(Z; Y)$
2. **Minimize** information about the input: $I(Z; X)$

$$\max_{Z} \left[ I(Z; Y) - \beta \cdot I(Z; X) \right]$$

This is a **compression** objective: keep only the information in $X$ that's relevant for predicting $Y$, discard everything else.

### How transformers implement this

In a transformer with $N$ layers:

- **Early layers**: Retain most input information ($I(Z_\ell; X)$ is high). The representation is "close to the input."
- **Middle layers**: Compress away irrelevant details. The representation becomes more abstract.
- **Late layers**: The representation is optimized for prediction ($I(Z_\ell; Y)$ is maximized).

This is observable with the logit lens (Doc 25.5):
- Early layers: predictions are diffuse (high entropy = many bits about input, few about output)
- Late layers: predictions are peaked (low entropy = few bits about input, many about output)

### The phase transition in training

Shwartz-Ziv & Tishby (2017) observed two phases of training:

1. **Fitting phase**: Network rapidly increases $I(Z; Y)$ (learning to predict)
2. **Compression phase**: Network slowly decreases $I(Z; X)$ (forgetting irrelevant input details)

The compression phase is where **generalization** happens. The model transitions from memorizing the training set to learning the underlying distribution. This may relate to grokking (Doc 21.7).

---

## 26.5 Bits-Back Coding & Why Latent Variables Help

### The bits-back argument

A model with latent variables $Z$ can achieve better compression than a model without:

$$\log P(x) = \log \sum_z P(x|z)P(z)$$

The latent variables provide a "code" for organizing the data. The encoder assigns data points to latent codes, and the decoder reconstructs from those codes. The entropy of the latent code is "bits back" — free bits you get from the structure.

### Application to transformers

The attention mechanism computes a **soft assignment** of each token to all other tokens. This is analogous to a latent variable that encodes "which other tokens are relevant for predicting the next token."

The multi-head structure provides multiple **channels** for encoding different types of relevance (syntactic, semantic, positional, etc.). Each head can be seen as a different latent code for the data.

---

## 26.6 Temperature, Sampling & Information

### Temperature as information control

Sampling with temperature $\tau$:

$$P_\tau(x) = \frac{\exp(z_x / \tau)}{\sum_j \exp(z_j / \tau)}$$

| $\tau$ | Behavior | Information-theoretic view |
|---|---|---|
| $\tau \to 0$ | Greedy (argmax) | Minimum entropy output (most compressed) |
| $\tau = 1$ | Calibrated sampling | True model distribution |
| $\tau > 1$ | High-entropy sampling | Adds noise = increases output entropy |
| $\tau \to \infty$ | Uniform sampling | Maximum entropy = no information |

### Top-p (nucleus) sampling as entropy thresholding

Top-p sampling keeps the smallest set of tokens whose cumulative probability exceeds $p$:

$$\text{Nucleus}(p) = \min \{S \subseteq V : \sum_{x \in S} P(x) \geq p\}$$

This is equivalent to setting an **entropy threshold**: include tokens until the cumulative information content reaches a threshold. Tokens in the nucleus are "informative" (the model thinks they're plausible). Tokens outside are "noise."

### Why temperature matters for reasoning

Lower temperature → model outputs its highest-probability token → follows the most likely reasoning path.

Higher temperature → model explores alternative paths → may find better solutions but also makes more errors.

**The test-time compute connection** (Doc 24.3): Sampling $N$ times at temperature $\tau > 0$ and taking the best result is more effective than greedy decoding because diversity in samples explores the solution landscape. The optimal temperature depends on the task:
- Factual recall: $\tau \to 0$ (there's one right answer)
- Creative writing: $\tau \in [0.7, 1.2]$ (diversity is valuable)
- Math/code: $\tau \in [0.2, 0.8]$ (need correctness but benefit from exploration)

---

## 26.7 Mutual Information & Feature Learning

### What neural networks learn

A trained network $f_\theta$ transforms input $X$ into representation $Z = f_\theta(X)$. The quality of this representation is measured by:

$$I(Z; Y) = H(Y) - H(Y|Z)$$

A representation that preserves all information about $Y$ satisfies $I(Z; Y) = H(Y)$ — knowing $Z$ tells you everything about $Y$.

### The data processing inequality

For any transformation $Z = f(X)$:

$$I(Z; Y) \leq I(X; Y)$$

You can never create information about $Y$ that wasn't in $X$. Every layer can only **preserve or destroy** information, never create it.

**Implication for depth**: Each layer must be careful not to destroy information that later layers need. This is why:
- Residual connections are essential (they preserve information via the identity path)
- Pre-Norm is better than Post-Norm (normalizing after the residual preserves the identity path)
- The residual stream IS the information — the layers are perturbations to it

### Sufficient statistics

A representation $Z$ is a **sufficient statistic** for $Y$ given $X$ if:

$$I(Z; Y) = I(X; Y)$$

The representation contains ALL the information about $Y$ that was in $X$. The final layer of a well-trained model should approximate a sufficient statistic for the next token.

---

## 26.8 The "Unreasonable Effectiveness" Explained

### Why predicting the next token produces understanding

Consider what's needed to predict the next token in these contexts:

- "The boiling point of water at sea level is ___" → Requires physics knowledge
- "If all A are B, and all B are C, then all A are ___" → Requires logical reasoning
- "def fibonacci(n): return 1 if n <= 1 else ___" → Requires programming knowledge
- "She felt a wave of sadness wash over ___" → Requires emotional understanding

Each prediction task requires a different type of "understanding." A model that predicts well across all contexts must have representations that encode physics, logic, code, emotion, and more.

### The formal argument

Let $X$ be natural language and $Y = X_{t+1}$ be the next token. The mutual information $I(X_{\leq t}; X_{t+1})$ captures ALL statistical dependencies in the language.

Natural language is generated by humans who:
- Describe the physical world → $I$ contains physics
- Report logical arguments → $I$ contains logic
- Express emotions → $I$ contains psychology
- Write code → $I$ contains algorithms

Therefore: $I(X_{\leq t}; X_{t+1})$ encodes a compressed representation of human knowledge. Minimizing cross-entropy loss = maximizing $I(Z; X_{t+1})$ = learning this compressed representation.

**This is why next-token prediction works**: it's not that prediction is "intelligent." It's that natural language is an **encoding of human knowledge**, and a good compressor of that encoding must capture the knowledge itself.

---

## 26.9 Perplexity as a Communication Rate

Perplexity has a beautiful information-theoretic interpretation:

$$\text{PPL} = 2^{H_Q(X)} = 2^{-\frac{1}{T}\sum_t \log_2 P_\theta(x_t | x_{<t})}$$

**Interpretation**: Perplexity is the effective vocabulary size the model is "choosing from" at each step. 

| PPL | Meaning |
|---|---|
| 1 | Model is certain — perfect prediction |
| 10 | Model is choosing among ~10 plausible tokens |
| 100 | Model is choosing among ~100 plausible tokens |
| $V$ | Model is guessing uniformly (no knowledge) |

A model with PPL = 20 on English text is "as uncertain" as someone choosing uniformly from a 20-word vocabulary. The lower the PPL, the more the model "knows" about what comes next.

### The channel capacity interpretation

Shannon's channel coding theorem: a channel with capacity $C$ bits per use can reliably transmit at rate $R \leq C$.

An LLM with cross-entropy $H$ bits per token on a vocabulary of size $V$ has a "channel capacity" of $\log_2 V - H$ bits per token. This is the amount of **meaningful information** per token.

For GPT-4 on English ($V \approx 100K$, $H \approx 2$ bits/token):
$$C \approx \log_2(100000) - 2 \approx 17 - 2 = 15 \text{ bits/token}$$

Each token carries approximately 15 bits of meaningful information. A 1000-token response contains ~15,000 bits = ~2 KB of information. (This is why LLM outputs feel "dense with meaning" — they nearly saturate the channel.)

---

## 26.10 The Theoretical Landscape: Key Connections

### 1. Cross-entropy = KL divergence = compression gap

$$\mathcal{L}_{\text{CE}} = H(P) + D_{\text{KL}}(P \| Q)$$

Training minimizes $D_{\text{KL}}$. The minimum achievable loss is $H(P)$, the true entropy.

### 2. Scaling laws = MDL tradeoff

More parameters → better data compression → lower loss. But model encoding cost grows with parameters. The optimum balances these.

### 3. Generalization = compression beyond the training set

A model that compresses training data well by learning **patterns** (not memorizing) will also compress test data well. Compression on unseen data = generalization.

### 4. In-context learning = adaptive compression

Given examples in the prompt, the model adapts its "codebook" on the fly — assigning shorter codes to tokens that fit the demonstrated pattern. This is Bayesian prediction: the posterior over tasks provides a better code.

### 5. Chain-of-thought = serial decompression

A single forward pass compresses one reasoning step. Chain-of-thought allows the model to decompress multi-step reasoning sequentially, each step building on the previous output.

### 6. Emergence = phase transitions in compression

When a model crosses a scale threshold, new statistical regularities become learnable — the model suddenly "discovers" a new way to compress the data, causing apparent capability emergence.

This is the unified view. Everything in this lecture series — from attention to alignment — serves one goal: **compressing human knowledge into an efficiently-decodable representation.**
