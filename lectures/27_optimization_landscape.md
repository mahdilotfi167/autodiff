# 27. The Optimization Landscape: Why Transformers Train At All

## Motivation

It's a miracle that gradient descent on a non-convex loss surface with billions of parameters converges to anything useful. Classical optimization theory says this should fail catastrophically — local minima, saddle points, and plateaus should trap the optimizer. Yet transformers train reliably. Why? This lecture explores the geometry of the loss landscape and explains why large neural networks are surprisingly easy to optimize.

---

## 27.1 The Non-Convex Paradox

### The classical worry

For a model with $N$ parameters, the loss $\mathcal{L}(\theta)$ is a function $\mathbb{R}^N \to \mathbb{R}$. For neural networks, this function is:
- **Non-convex**: Multiple local minima
- **Non-smooth**: ReLU kinks, discontinuities
- **Extremely high-dimensional**: $N \sim 10^9 - 10^{12}$

Classical optimization theory provides no convergence guarantees for this setting. Yet in practice, training works. The key insight is that **high-dimensional non-convex is qualitatively different from low-dimensional non-convex**.

---

## 27.2 Loss Landscape Geometry in High Dimensions

### Why local minima are rare (at large scale)

Consider a critical point $\theta^*$ where $\nabla \mathcal{L} = 0$. The Hessian matrix $\mathbf{H} = \nabla^2 \mathcal{L}$ at this point has $N$ eigenvalues $\lambda_1, \ldots, \lambda_N$.

- **Local minimum**: All eigenvalues positive ($\lambda_i > 0$ for all $i$)
- **Saddle point**: Some positive, some negative
- **Local maximum**: All eigenvalues negative

For a "random" function in $N$ dimensions, the probability that all $N$ eigenvalues are positive is approximately:

$$P(\text{local min}) \approx 2^{-N}$$

For $N = 10^9$, this is $2^{-10^9}$ — essentially zero. **Almost all critical points in high-dimensional spaces are saddle points, not local minima.**

### The saddle point structure

At a saddle point with $k$ negative eigenvalues ("index $k$"):

$$\mathcal{L}(\theta^* + \epsilon) \approx \mathcal{L}(\theta^*) + \frac{1}{2}\sum_{i=1}^{N} \lambda_i \epsilon_i^2$$

There are $k$ directions where the loss **decreases** (the negative eigenvalue directions). SGD will eventually find one of these escape routes.

### The high-loss ↔ high-index correlation

Baldi & Hornik (1989), extended by Choromanska et al. (2015):

Critical points with high loss tend to be high-index saddle points (many escape directions). Critical points with low loss tend to be (nearly) true local minima.

**Implication**: The optimizer naturally descends toward the "basin" of good critical points. High-loss saddle points are easy to escape; low-loss near-minima are where you want to be.

---

## 27.3 The Loss Surface Is Surprisingly Benign

### Loss surface visualization

Li et al. (2018) introduced a technique to visualize loss surfaces by projecting onto random 2D planes through parameter space. Key findings:

- **Small models**: The surface is rough, with many distinct local minima separated by high barriers. Training is sensitive to initialization.
- **Large models**: The surface is smooth, with wide valleys. Different initializations converge to regions with similar loss.

**The over-parameterization smoothing effect**: Adding parameters to a model smooths the loss landscape. This is counterintuitive — more parameters means a higher-dimensional, more complex function — but the extra dimensions provide more "paths" around obstacles.

### Why wider is smoother

For a network with width $d$, the number of critical points scales as:

$$\text{Number of critical points} \propto \exp(\text{poly}(d))$$

But the number of **bad** critical points (high loss) grows much more slowly than the number of **good** ones (low loss). As $d$ increases, the fraction of bad minima → 0.

Moreover, the barriers between good minima shrink. For sufficiently wide networks, **all good minima are connected by paths of low loss** (mode connectivity, Sec 27.5).

---

## 27.4 Sharp vs. Flat Minima

### The generalization connection

Consider two minima $\theta_1$ and $\theta_2$ with equal training loss:

- **Sharp minimum** ($\theta_1$): Loss increases rapidly when you move away from $\theta_1$. The Hessian has large eigenvalues.
- **Flat minimum** ($\theta_2$): Loss stays low in a large neighborhood around $\theta_2$. The Hessian has small eigenvalues.

**Claim** (Hochreiter & Schmidhuber, 1997; Keskar et al., 2017): Flat minima generalize better.

### Why flat minima generalize (intuition)

The training loss landscape and the test loss landscape are slightly different (because the test data is a different sample from the same distribution). A sharp minimum on the training loss might be a **different** point on the test loss. A flat minimum is likely to still be good on the test loss because the neighborhood is all low-loss.

Formally, if $\mathcal{L}_{\text{test}}(\theta) = \mathcal{L}_{\text{train}}(\theta) + \delta(\theta)$ where $\|\delta\| \leq \epsilon$:

- At a sharp minimum: $\mathcal{L}_{\text{test}}$ could be much higher (the perturbation moves you out of the narrow valley)
- At a flat minimum: $\mathcal{L}_{\text{test}}$ stays similar (the perturbation keeps you in the wide valley)

### PAC-Bayesian bound (the formal version)

$$\mathcal{L}_{\text{test}}(\theta) \leq \mathcal{L}_{\text{train}}(\theta) + \sqrt{\frac{D_{\text{KL}}(Q \| P) + \ln(n/\delta)}{2n}}$$

where $Q$ is a distribution around the found minimum and $P$ is the prior. A flat minimum allows $Q$ to be broad (low KL) → tighter bound → better generalization.

### How SGD finds flat minima

**SGD noise as implicit regularization**: SGD's gradient noise has covariance proportional to:

$$\mathbf{C} \approx \frac{1}{B}\left(\frac{1}{n}\sum_{i} \nabla \ell_i \nabla \ell_i^T - \nabla \mathcal{L} \nabla \mathcal{L}^T\right)$$

where $B$ is batch size. This noise:
1. Helps escape sharp minima (large Hessian eigenvalues → unstable under noise)
2. Settles in flat minima (small Hessian eigenvalues → stable under noise)

**The noise strength scales as $\eta / B$** (learning rate / batch size). This is why:
- Larger learning rates → flatter minima → better generalization
- Larger batch sizes → less noise → sharper minima → worse generalization (without LR adjustment)
- The "critical batch size" (Doc 15.7, Appendix A.12) is where gradient noise becomes too small to escape sharp minima

### The Sharpness-Aware Minimization (SAM) approach

SAM explicitly seeks flat minima:

$$\min_\theta \max_{\|\epsilon\| \leq \rho} \mathcal{L}(\theta + \epsilon)$$

Find parameters where even the **worst-case perturbation** within radius $\rho$ has low loss. This is directly optimizing for flatness.

Implementation:
```python
# SAM step
loss = model(batch).loss
loss.backward()
# First step: ascend to worst-case perturbation
with torch.no_grad():
    for p in model.parameters():
        e = rho * p.grad / p.grad.norm()
        p.add_(e)  # θ + ε
# Second step: compute gradient at perturbed point
loss2 = model(batch).loss
loss2.backward()
with torch.no_grad():
    for p in model.parameters():
        p.sub_(e)  # Undo perturbation
    optimizer.step()  # Step using gradient at θ + ε
```

---

## 27.5 Mode Connectivity

### The discovery

Garipov et al. (2018), Draxler et al. (2018): Two independently trained networks (different random seeds, same architecture) can be connected by a path in parameter space along which the loss stays low.

$$\exists \gamma: [0,1] \to \mathbb{R}^N \text{ s.t. } \gamma(0) = \theta_1, \gamma(1) = \theta_2, \text{ and } \mathcal{L}(\gamma(t)) \approx \mathcal{L}(\theta_1) \forall t$$

This means the loss landscape has the structure of a single connected **basin** (or a small number of basins), not a rugged terrain of isolated minima.

### What this means

1. **Initialization doesn't matter much** (for large models). Different initializations converge to the same connected basin.
2. **Averaging works**: Linear interpolation between good models often produces a good model: $\theta_{\text{avg}} = \frac{1}{2}(\theta_1 + \theta_2)$ has similar or better loss.
3. **The "right architecture" creates a benign landscape**: Residual connections and normalization layers are crucial for mode connectivity. Without them, the landscape fragments into disconnected basins.

### Linear mode connectivity and loss barriers

Two models are **linearly mode connected** if:

$$\mathcal{L}(\alpha \theta_1 + (1-\alpha) \theta_2) \leq \max(\mathcal{L}(\theta_1), \mathcal{L}(\theta_2)) + \epsilon \quad \forall \alpha \in [0,1]$$

The **loss barrier** is the maximum loss along the linear path minus the endpoint losses. For modern transformers trained from the same initialization but with different data order, the loss barrier is essentially zero. This is remarkable and enables model merging.

---

## 27.6 Double Descent

### The classical U-curve (bias-variance tradeoff)

Classical statistics says: as model complexity increases, test error first decreases (less bias) then increases (more variance). The optimal model is at the minimum of this U-curve.

### The modern observation

Belkin et al. (2019): Test error follows a **double descent** curve:

```
Test error
    │
    │  ╲
    │   ╲          Classical         Modern
    │    ╲        U-curve         "more is better"
    │     ╲      regime           regime
    │      ╲   ╱╲
    │       ╲╱   ╲
    │    ↓        ╲↓peak          ╲
    │  good       bad               ╲  good again
    │                                 ╲──────
    └──────────────────────────────────────→
          Model complexity (parameters)
                    ↑
             Interpolation
              threshold
```

At the **interpolation threshold** (model has exactly enough capacity to memorize the training set), test error peaks. Adding more parameters past this point causes test error to drop again.

### Why it happens

At the interpolation threshold:
- The model uses ALL its capacity to memorize → no room for generalization
- The solution is extremely complex (high "norm") → poor generalization
- There's essentially ONE way to fit the data → no choice of solution

Past the threshold:
- Many solutions fit the data → the optimizer can choose a "simple" one
- Implicit regularization (SGD noise, weight decay) biases toward flat, generalizing solutions
- Extra capacity allows the model to learn the **pattern** rather than memorize individual examples

### Implications for LLMs

- **Never train a model that's "barely large enough" for your data.** You'll be at the interpolation threshold.
- **Over-parameterize**, then regularize (weight decay, dropout, data augmentation).
- **Chinchilla scaling** avoids the threshold by ensuring the model is always well into the over-parameterized regime.

---

## 27.7 The Edge of Stability

### The observation (Cohen et al., 2021)

When training neural networks with gradient descent at a fixed learning rate $\eta$:

1. The largest eigenvalue of the Hessian, $\lambda_{\max}(\mathbf{H})$, increases during training
2. It rises until it reaches $2/\eta$ (the "edge of stability")
3. It then oscillates around $2/\eta$, never stably exceeding it

### Why $2/\eta$?

For gradient descent on a quadratic $f(x) = \frac{1}{2}\lambda x^2$:

$$x_{t+1} = x_t - \eta \lambda x_t = (1 - \eta\lambda)x_t$$

This converges if $|1 - \eta\lambda| < 1$, i.e., $\lambda < 2/\eta$. At $\lambda = 2/\eta$, the update magnitude equals the current value — the "edge" of stability.

### What happens at the edge

The model **self-tunes** its loss landscape to be at the edge of stability:
- If sharpness exceeds $2/\eta$ → gradients cause oscillation → the model moves to a flatter region
- If sharpness is below $2/\eta$ → stable descent → sharpness can increase
- Equilibrium: sharpness hovers at $2/\eta$

**Implication**: The learning rate implicitly controls the sharpness of the minimum you find.
- Large $\eta$ → small $2/\eta$ → flat minima (better generalization)
- Small $\eta$ → large $2/\eta$ → sharper minima (worse generalization)

This explains why **large learning rates generalize better** and why **learning rate warmup** is important: it allows the model to gradually increase the sharpness it can tolerate.

---

## 27.8 The Lottery Ticket Hypothesis

### The claim (Frankle & Carlin, 2019)

A randomly initialized dense network contains a **sparse subnetwork** (the "winning ticket") that, when trained in isolation from the same initialization, achieves comparable accuracy to the full network.

### Key findings:

1. The winning ticket is typically 10-20% of the full network
2. The winning ticket must be trained from its **original initialization** (not a random re-initialization)
3. Finding the ticket requires training the full network first (then pruning), so it doesn't save training compute

### Implications for understanding:

1. **Over-parameterization aids optimization, not just representation.** The full network needs to be large so that gradient descent can *find* the winning ticket. The extra parameters are "scaffolding."

2. **Pruning is post-hoc model compression.** Train big, find the ticket, deploy small. This is practical for inference.

3. **Connection to mode connectivity**: The winning ticket and the full network likely occupy the same connected basin — pruning removes dimensions that don't contribute to the solution.

### For LLMs:

The lottery ticket hypothesis has complicated results at LLM scale:
- Structured pruning (removing entire heads or layers) works better than unstructured pruning
- Many attention heads CAN be removed with minimal loss (Doc 25.3 — dead heads)
- But the remaining heads need to be the "right" ones — random pruning fails

---

## 27.9 The Natural Gradient & Adam's Approximation

### Why vanilla gradient descent is "wrong"

Gradient descent steps in the direction of steepest descent in **parameter space**: $\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}$.

But the same parameter change $\Delta\theta$ can have very different effects on the **function** the model computes, depending on the local curvature. A step of 0.01 in one direction might change the output by 1%, while a step of 0.01 in another direction changes it by 100%.

### The Fisher Information Matrix

The natural gradient uses the **Fisher Information Matrix** $\mathbf{F}$ to account for the geometry of the function space:

$$\mathbf{F} = \mathbb{E}_{x \sim P_\theta}\left[\nabla \log P_\theta(x) \cdot \nabla \log P_\theta(x)^T\right]$$

$\mathbf{F}$ measures how much the distribution changes per unit parameter change. The natural gradient:

$$\theta_{t+1} = \theta_t - \eta \mathbf{F}^{-1} \nabla_\theta \mathcal{L}$$

This steps in the direction that changes the **distribution** most, not the **parameters** most.

### Adam ≈ diagonal natural gradient

Adam's update:

$$\theta_{t+1} = \theta_t - \eta \frac{m_t}{\sqrt{v_t} + \epsilon}$$

The $1/\sqrt{v_t}$ term is a **diagonal approximation** to $\mathbf{F}^{-1}$. For each parameter, $v_t \approx \mathbb{E}[g_t^2] \approx F_{ii}$ (the diagonal of the Fisher). So:

$$\frac{1}{\sqrt{v_t}} \approx \frac{1}{\sqrt{F_{ii}}}$$

This is why Adam works so well for transformers: it approximates the natural gradient, which is the "correct" way to optimize in distribution space.

**The limitation**: Adam only uses the diagonal. Off-diagonal terms of $\mathbf{F}$ capture parameter interactions. K-FAC and other second-order methods approximate these, but are too expensive for LLM-scale training.

---

## 27.10 Why Residual Connections Make Everything Work

### The gradient flow argument (revisited with depth)

Without residual connections, a network of depth $N$ has gradient:

$$\frac{\partial \mathcal{L}}{\partial \theta_1} = \prod_{\ell=1}^{N} \mathbf{J}_\ell \cdot \frac{\partial \mathcal{L}}{\partial \mathbf{x}_N}$$

Each Jacobian $\mathbf{J}_\ell = \partial \mathbf{x}_\ell / \partial \mathbf{x}_{\ell-1}$ has spectral radius $\rho_\ell$. If $\rho_\ell < 1$ on average → **vanishing**. If $\rho_\ell > 1$ → **exploding**.

With residual connections: $\mathbf{x}_{\ell+1} = \mathbf{x}_\ell + f_\ell(\mathbf{x}_\ell)$

$$\frac{\partial \mathbf{x}_N}{\partial \mathbf{x}_1} = \mathbf{I} + \sum_{\text{paths}} \prod_{\ell \in \text{path}} \frac{\partial f_\ell}{\partial \mathbf{x}_\ell}$$

The identity term guarantees gradient magnitude $\geq 1$. The gradient never vanishes, regardless of depth.

### The ensemble interpretation

A ResNet of depth $N$ is equivalent to an **ensemble** of $2^N$ paths of different lengths (each path either includes or skips each residual block). Most information flows through short paths. The model is effectively an implicit ensemble, which explains its robustness and generalization.

### The loss landscape connection

Residual connections dramatically smooth the loss landscape:
- Without residuals: loss surface is chaotic, with many bad local minima
- With residuals: loss surface is nearly convex in most directions
- The identity path ensures that the model always has the option to "do nothing" at each layer, preventing catastrophic representations

**This is arguably the single most important architectural innovation in deep learning.** More important than attention, normalization, or any activation function.

---

## 27.11 Summary: Why Training Works

The answer is a conjunction of several factors:

| Factor | Mechanism | Without it |
|---|---|---|
| High dimensionality | Most critical points are saddle points, not minima | Stuck in local minima |
| Over-parameterization | Smooths loss surface, enables mode connectivity | Rough landscape, isolated minima |
| Residual connections | Gradient = I + corrections, ensemble of paths | Vanishing/exploding gradients |
| SGD noise | Escapes sharp minima, finds flat minima | Sharp minima, poor generalization |
| Adam (adaptive LR) | ≈ natural gradient, handles curvature | Slow convergence, scale sensitivity |
| Normalization layers | Stabilizes activation magnitudes | Activation drift, training instability |
| Learning rate warmup | Allows optimizer state to stabilize | Early instability, bad basins |
| Weight decay | Implicit bias toward simple solutions | Memorization, complex solutions |

No single factor is sufficient. Remove any one and training degrades. Together, they create a remarkably reliable optimization procedure for a problem that should, by all theoretical accounts, be impossibly hard.
