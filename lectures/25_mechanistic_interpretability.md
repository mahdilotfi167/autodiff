# 25. Mechanistic Interpretability: Reverse-Engineering Neural Networks

## Motivation

Mechanistic interpretability (mech interp) is the field of reverse-engineering trained neural networks to understand what they compute and how. This is one of Anthropic's core research areas and increasingly important at all top labs. If you're interviewing for a research role that involves alignment, safety, or understanding model behavior, this is essential knowledge.

---

## 25.1 The Core Question

A trained transformer is a function $f: \mathbb{R}^{V \times n} \to \mathbb{R}^{V}$ with billions of parameters. We know the weights. We know the architecture. But we don't understand *what algorithm the model implements*.

Mechanistic interpretability aims to answer: **What is the model doing, mechanistically, at the level of individual neurons, attention heads, and circuits?**

This is analogous to reverse-engineering a compiled binary: you have the machine code (weights) and the instruction set (architecture), and you want to recover the source code (algorithm).

---

## 25.2 Features, Neurons & Superposition

### The neuron doctrine (and why it's wrong)

Naive assumption: each neuron encodes a single concept (the "grandmother neuron" hypothesis).

**Reality**: Individual neurons in modern networks respond to **multiple unrelated concepts**. Neuron 374 in layer 12 might activate for both "legal terminology" and "the color blue." This is **polysemanticity**.

### Superposition

Superposition: the model encodes more features than it has neurons, by using **almost-orthogonal** directions in activation space.

If the model has $d$ neurons but needs to represent $m \gg d$ features, it can store them as $m$ directions $\mathbf{v}_1, \ldots, \mathbf{v}_m \in \mathbb{R}^d$ where:

$$\mathbf{v}_i \cdot \mathbf{v}_j \approx 0 \quad \text{for } i \neq j$$

In $d$ dimensions, you can pack $\gg d$ nearly-orthogonal vectors (the Johnson-Lindenstrauss lemma guarantees $\exp(\mathcal{O}(d))$ such vectors exist).

**This is why looking at individual neurons is misleading**: the meaningful units are **directions** (linear combinations of neurons), not individual neurons.

### Sparse Autoencoders (SAEs)

To find the actual features from superposed representations, train a sparse autoencoder:

$$\mathbf{h} = \text{Enc}(\mathbf{x}) = \text{ReLU}(\mathbf{W}_{\text{enc}} \mathbf{x} + \mathbf{b}_{\text{enc}})$$
$$\hat{\mathbf{x}} = \text{Dec}(\mathbf{h}) = \mathbf{W}_{\text{dec}} \mathbf{h} + \mathbf{b}_{\text{dec}}$$

$$\mathcal{L} = \|\mathbf{x} - \hat{\mathbf{x}}\|^2 + \lambda \|\mathbf{h}\|_1$$

Key properties:
- $\mathbf{h} \in \mathbb{R}^{m}$ where $m \gg d$ (overcomplete dictionary)
- $L_1$ penalty enforces sparsity — each input activates only a few features
- Each column of $\mathbf{W}_{\text{dec}}$ is a "feature direction" in activation space

**Anthropic's key finding (2023-2024)**: SAE features in Claude are **interpretable**. They found features for:
- Specific entities (Golden Gate Bridge, DNA, etc.)
- Abstract concepts (deception, uncertainty, code bugs)
- Syntactic patterns (list items, quotations)
- Safety-relevant concepts (harmful requests, refusals)

```python
class SparseAutoencoder(nn.Module):
    def __init__(self, d_model, n_features, k=32):
        super().__init__()
        self.encoder = nn.Linear(d_model, n_features, bias=True)
        self.decoder = nn.Linear(n_features, d_model, bias=True)
        # TopK activation: only keep top k activations
        self.k = k

    def forward(self, x):
        # Encode
        h = self.encoder(x)
        # TopK sparsity (alternative to L1)
        topk_values, topk_indices = h.topk(self.k, dim=-1)
        h_sparse = torch.zeros_like(h).scatter(-1, topk_indices,
                                                F.relu(topk_values))
        # Decode
        x_hat = self.decoder(h_sparse)
        return x_hat, h_sparse
```

---

## 25.3 Circuits: How Models Compute

### What is a circuit?

A circuit is a subgraph of the model's computation graph that implements a specific function. It consists of:
- Specific attention heads and MLP neurons
- The connections between them (via the residual stream)
- The function they collectively compute

### The Residual Stream as Information Highway

The residual stream $\mathbf{x}_\ell$ is the central data structure. Each attention head and MLP **reads from** and **writes to** the residual stream independently:

$$\mathbf{x}_\ell = \mathbf{x}_0 + \sum_{i=1}^{\ell} \text{Attn}_i(\mathbf{x}_{i-1}) + \text{MLP}_i(\mathbf{x}_{i-1})$$

Think of it as a shared memory bus:
- Attention heads read information from other positions and write it to the current position
- MLPs process the information at each position independently
- Each component's contribution is **additive** — they can be studied independently

### Famous circuits discovered:

**1. Induction heads (Olsson et al., 2022)**

The single most important circuit discovered in transformers.

Two-head circuit:
- **Head 1 (previous token head)**: Copies the token at position $t$ into the residual stream at position $t+1$
- **Head 2 (induction head)**: Searches for the pattern "X was followed by Y" in context, and predicts Y will follow X again

```
Context: "...The cat sat on the mat. The cat sat on the ___"
                                                        ↑
Head 1: copies "cat" → "sat" → "on" → "the" pattern
Head 2: finds "the mat" after previous "on the" → predicts "mat"
```

This implements **in-context pattern matching** — the most basic form of in-context learning!

**Key finding**: Induction heads emerge during a specific phase of training (the "induction bump"). Before this phase, the model can't do in-context learning. After, it can. This is a genuine phase transition.

**2. Indirect Object Identification (Wang et al., 2023)**

For sentences like "John gave Mary a book. Mary gave ___":

A circuit of ~26 attention heads across multiple layers:
1. **Duplicate token heads**: Detect that "Mary" appears twice
2. **S-inhibition heads**: Suppress the subject "Mary" at the output
3. **Name mover heads**: Move the remaining name "John" to the output position

This is a complete mechanistic explanation of how the model solves a specific task.

**3. Factual recall circuits**

For "The Eiffel Tower is located in ___":
1. Early layers: Identify "Eiffel Tower" as an entity
2. Middle layers: MLP neurons that store the fact "Eiffel Tower → Paris"
3. Late layers: Project "Paris" to the output vocabulary

The MLP acts as a **key-value memory**: the first matrix ($\mathbf{W}_{\text{up}}$) acts as keys, the second ($\mathbf{W}_{\text{down}}$) acts as values.

---

## 25.4 Attribution & Patching Methods

### Activation patching (causal tracing)

The gold standard for identifying which components matter for a behavior.

**Method**:
1. Run the model on a clean input → clean activations
2. Run the model on a corrupted input (e.g., different entity) → corrupted activations
3. **Patch**: Replace one component's activation in the corrupted run with the clean activation
4. Measure: did the output recover?

If patching component $C$ recovers the output → $C$ is causally important.

```python
def activation_patch(model, clean_input, corrupt_input, component, position):
    """Patch activation of `component` at `position` from clean into corrupt."""
    # Get clean activation
    with torch.no_grad():
        clean_cache = model.run_with_cache(clean_input)
        clean_act = clean_cache[component][0, position]

    # Run corrupt with one patched activation
    def patch_hook(activation, hook):
        activation[0, position] = clean_act
        return activation

    corrupt_logits = model.run_with_hooks(corrupt_input, hooks=[(component, patch_hook)])
    return corrupt_logits
```

### Path patching

More fine-grained than activation patching: patch the **connection between two specific components**, not just one component's output.

This lets you trace the flow of information:
- "Head 3.2 sends information to Head 7.1 via the residual stream"
- "MLP 5 writes a fact that Head 8.4 reads"

### Attribution patching (gradient-based approximation)

Full activation patching requires $\mathcal{O}(C)$ forward passes (one per component). Gradient-based approximation:

$$\text{Attribution}(C) \approx (\mathbf{a}_{\text{clean}}^C - \mathbf{a}_{\text{corrupt}}^C) \cdot \nabla_{\mathbf{a}^C} \text{Logit}$$

One forward + one backward pass → approximate all component attributions simultaneously.

---

## 25.5 The Logit Lens & Tuned Lens

### Logit lens (nostalgebraist, 2020)

At each layer $\ell$, take the intermediate residual stream $\mathbf{x}_\ell$ and project it through the final layer norm + unembedding matrix:

$$\text{prediction}_\ell = \text{softmax}(\text{LN}(\mathbf{x}_\ell) \cdot \mathbf{W}_U)$$

This shows **what the model would predict if it stopped processing at layer $\ell$**.

**What you observe**:
- Early layers: predictions are essentially random
- Middle layers: the correct answer starts emerging
- Late layers: high-confidence correct prediction

**The insight**: Information is gradually refined through the layers. You can pinpoint exactly where a fact enters the residual stream (which layer "adds" the knowledge).

### Tuned lens

The logit lens assumes intermediate representations are in the same space as the final layer. This isn't quite true — each layer operates in a slightly different subspace.

**Tuned lens**: Learn a small affine transformation per layer:

$$\text{prediction}_\ell = \text{softmax}((\mathbf{A}_\ell \mathbf{x}_\ell + \mathbf{b}_\ell) \cdot \mathbf{W}_U)$$

Train $\mathbf{A}_\ell, \mathbf{b}_\ell$ to predict the final output from layer $\ell$'s representation. This gives a much cleaner view of per-layer processing.

---

## 25.6 Steering & Feature Intervention

### Activation steering

If you know the **direction** corresponding to a concept (e.g., from SAE), you can steer the model by adding or subtracting that direction:

$$\mathbf{x}_\ell \leftarrow \mathbf{x}_\ell + \alpha \cdot \mathbf{v}_{\text{concept}}$$

**Examples**:
- Adding the "Golden Gate Bridge" feature direction → model talks about the Golden Gate Bridge constantly
- Adding the "honesty" direction → model becomes more truthful
- Subtracting the "sycophancy" direction → model argues more

**This is a form of inference-time control without any fine-tuning.**

### Representation engineering

Find directions in activation space that correspond to high-level properties:

1. Collect activations for prompts that are "truthful" vs "untruthful"
2. Compute the difference in mean activation: $\mathbf{v}_{\text{truth}} = \bar{\mathbf{h}}_{\text{true}} - \bar{\mathbf{h}}_{\text{false}}$
3. This direction acts as a "truth probe" — project new activations onto it to predict truthfulness

```python
def find_concept_direction(model, positive_prompts, negative_prompts, layer):
    """Find the direction corresponding to a concept."""
    pos_acts = []
    neg_acts = []
    for p in positive_prompts:
        acts = model.get_activations(p, layer=layer)
        pos_acts.append(acts.mean(dim=1))  # Average over positions
    for n in negative_prompts:
        acts = model.get_activations(n, layer=layer)
        neg_acts.append(acts.mean(dim=1))

    pos_mean = torch.stack(pos_acts).mean(dim=0)
    neg_mean = torch.stack(neg_acts).mean(dim=0)

    direction = pos_mean - neg_mean
    direction = direction / direction.norm()  # Unit vector
    return direction
```

---

## 25.7 What Mech Interp Has Taught Us (The Big Picture)

### Key takeaways:

1. **Models are more structured than expected.** They develop clean, identifiable circuits for specific tasks. It's not undifferentiated soup.

2. **Superposition is the fundamental challenge.** Models represent more features than they have dimensions. Disentangling this is the core technical problem.

3. **MLPs are memory, attention is routing.** MLPs store facts (Eiffel Tower → Paris). Attention heads move information between positions (copy, induction, inhibition).

4. **Features are linear.** The model represents concepts as directions in activation space. This is why linear probes work and why SAEs find interpretable features.

5. **The model's "explanation" isn't its mechanism.** Chain-of-thought text is generated by the same forward pass, not by an interpretable reasoning process. The mechanism is in the weights and activations, not the output tokens.

6. **Safety implications are concrete.** If you can find the "deception" feature direction, you can potentially monitor or control it. This is why interpretability is a core part of alignment research.

### Open questions:

- Can we scale SAEs to find ALL features in a large model? (Current: ~1M features found in Claude 3 Sonnet)
- Can we verify that a circuit explanation is complete? (Not just a subset)
- Can we use mech interp to predict model behavior on unseen inputs?
- Is there a "universal" set of features that all LLMs learn?

---

## 25.8 Practical Tools

The main library for mech interp is **TransformerLens** (by Neel Nanda):

```python
import transformer_lens as tl

# Load a model with hooks
model = tl.HookedTransformer.from_pretrained("gpt2-small")

# Run with cache (stores all intermediate activations)
logits, cache = model.run_with_cache("The Eiffel Tower is in")

# Inspect attention patterns
attn_patterns = cache["blocks.5.attn.hook_pattern"]  # (B, heads, q, k)

# Logit lens: what does each layer predict?
for layer in range(model.cfg.n_layers):
    residual = cache[f"blocks.{layer}.hook_resid_post"]
    logits_at_layer = model.unembed(model.ln_final(residual))
    top_token = logits_at_layer[0, -1].argmax()
    print(f"Layer {layer}: predicts '{model.to_string(top_token)}'")
```

### Why interviewers ask about this:

At Anthropic especially, understanding mechanistic interpretability signals:
- You can think about models as objects of study, not just tools
- You understand the alignment problem at a technical level
- You can reason about what it means for a model to "know" or "intend" something
- You appreciate the gap between behavior and mechanism
