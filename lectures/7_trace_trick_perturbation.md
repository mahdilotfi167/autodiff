# 7. The Trace Trick, Matrix Differentials, and Perturbation Framework

## Motivation

In Document 4, we derived the matmul gradient element-by-element: write $Y_{ij} = \sum_k A_{ik}B_{kj}$, compute each partial derivative, then recognize the result as a matrix product. This works but is tedious, error-prone, and doesn't scale to complex operations.

There is a far more elegant approach: **matrix differential calculus with the trace trick**. Instead of reasoning about individual elements, you work directly with matrices, using the **differential** $dL$ and **trace identities** to extract gradients in closed form.

This is the method that framework authors (PyTorch, JAX, TensorFlow) actually use to derive gradient formulas for new operations. It is also the standard approach in the optimization and machine learning research literature.

---

## 7.1 Prerequisites: The Matrix Trace

### Definition

The trace of a square matrix $\mathbf{M} \in \mathbb{R}^{n \times n}$ is the sum of its diagonal:

$$\text{tr}(\mathbf{M}) = \sum_{i=1}^{n} M_{ii}$$

### Properties (all are critical — memorize these)

| Property | Formula | Proof sketch |
|----------|---------|-------------|
| Linearity | $\text{tr}(\mathbf{A} + \mathbf{B}) = \text{tr}(\mathbf{A}) + \text{tr}(\mathbf{B})$ | Sum of diagonals |
| Scalar factor | $\text{tr}(c\mathbf{A}) = c \cdot \text{tr}(\mathbf{A})$ | Factor out of sum |
| Transpose | $\text{tr}(\mathbf{A}) = \text{tr}(\mathbf{A}^T)$ | Diagonal is unchanged |
| **Cyclic** | $\text{tr}(\mathbf{ABC}) = \text{tr}(\mathbf{BCA}) = \text{tr}(\mathbf{CAB})$ | Index manipulation |
| **Inner product** | $\text{tr}(\mathbf{A}^T\mathbf{B}) = \sum_{ij} A_{ij}B_{ij}$ | Frobenius inner product |
| Scalar = trace | $a = \text{tr}(a)$ for $a \in \mathbb{R}$ | 1×1 matrix |

### Proof of the cyclic property

$$\text{tr}(\mathbf{ABC}) = \sum_i (\mathbf{ABC})_{ii} = \sum_i \sum_j \sum_k A_{ij}B_{jk}C_{ki}$$

$$\text{tr}(\mathbf{BCA}) = \sum_j (\mathbf{BCA})_{jj} = \sum_j \sum_k \sum_i B_{jk}C_{ki}A_{ij}$$

These are the same triple sum, just with the summation indices relabeled. ∎

**Warning**: cyclic permutation only — you cannot arbitrarily reorder. $\text{tr}(\mathbf{ABC}) \neq \text{tr}(\mathbf{BAC})$ in general.

### Proof of the inner product identity

$$\text{tr}(\mathbf{A}^T\mathbf{B}) = \sum_i (\mathbf{A}^T\mathbf{B})_{ii} = \sum_i \sum_j A^T_{ij} B_{ji} = \sum_i \sum_j A_{ji} B_{ji} = \sum_{j,i} A_{ji}B_{ji}$$

This is exactly $\sum_{\text{all elements}} A_{\alpha\beta}B_{\alpha\beta}$, which is the Frobenius inner product $\langle \mathbf{A}, \mathbf{B} \rangle_F$. ∎

**This identity is the bridge between traces and element-wise sums — it is the heart of the trace trick.**

---

## 7.2 Matrix Differentials

### What is a differential?

For a scalar function $f(x)$, the differential is:
$$df = f'(x) \, dx$$

For a function of a matrix $L(\mathbf{X})$ where $L$ is scalar:
$$dL = \sum_{ij} \frac{\partial L}{\partial X_{ij}} dX_{ij}$$

**What this means:**
- $dL$ = a tiny change in $L$ (the loss)
- $\frac{\partial L}{\partial X_{ij}}$ = how much $L$ changes per unit change in element $X_{ij}$ (the sensitivity)
- $dX_{ij}$ = a tiny perturbation to element $X_{ij}$
- **The equation says**: The total change in $L$ equals the sum over all matrix elements of: (sensitivity to that element) × (perturbation of that element)

**Example:** For $L = \text{tr}(\mathbf{X})$, only diagonal elements matter: $\frac{\partial L}{\partial X_{ii}} = 1$ and $\frac{\partial L}{\partial X_{ij}} = 0$ for $i \neq j$. So $dL = dX_{11} + dX_{22} + dX_{33}$ (perturb only the diagonals).

Using the trace inner product identity:
$$dL = \text{tr}\left(\frac{\partial L}{\partial \mathbf{X}}^T d\mathbf{X}\right)$$

**This is the key equation.** If we can manipulate $dL$ into the form $\text{tr}(\mathbf{G}^T \, d\mathbf{X})$, then $\mathbf{G} = \frac{\partial L}{\partial \mathbf{X}}$ is the gradient. The trace form is elegant because instead of summing element-by-element (tedious!), we work with entire matrices at once.

### Differential rules

These parallel the standard calculus product rule, chain rule, etc.:

| Function | Differential |
|----------|-------------|
| $\mathbf{Y} = \mathbf{A}\mathbf{X}$ | $d\mathbf{Y} = \mathbf{A} \, d\mathbf{X}$ |
| $\mathbf{Y} = \mathbf{X}\mathbf{B}$ | $d\mathbf{Y} = d\mathbf{X} \, \mathbf{B}$ |
| $\mathbf{Y} = \mathbf{X}^T$ | $d\mathbf{Y} = (d\mathbf{X})^T$ |
| $\mathbf{Y} = \mathbf{X} + \mathbf{Z}$ | $d\mathbf{Y} = d\mathbf{X} + d\mathbf{Z}$ |
| $\mathbf{Y} = \mathbf{X} \odot \mathbf{Z}$ | $d\mathbf{Y} = d\mathbf{X} \odot \mathbf{Z} + \mathbf{X} \odot d\mathbf{Z}$ |
| $y = \text{tr}(\mathbf{X})$ | $dy = \text{tr}(d\mathbf{X})$ |
| $\mathbf{Y} = f(\mathbf{X})$ (elem-wise) | $d\mathbf{Y} = f'(\mathbf{X}) \odot d\mathbf{X}$ |
| $\mathbf{Y} = \mathbf{X}\mathbf{X}^T$ | $d\mathbf{Y} = d\mathbf{X} \, \mathbf{X}^T + \mathbf{X} \, (d\mathbf{X})^T$ |
| $\mathbf{Y} = \mathbf{X}^{-1}$ | $d\mathbf{Y} = -\mathbf{X}^{-1} (d\mathbf{X}) \mathbf{X}^{-1}$ |

Key: if $\mathbf{A}$ is a constant (not the variable we're differentiating w.r.t.), then $d\mathbf{A} = 0$.

---

## 7.3 The Trace Trick: Worked Example — Matmul

**Goal**: Derive $\frac{\partial L}{\partial \mathbf{A}}$ for $\mathbf{Y} = \mathbf{A}\mathbf{B}$ where $L$ is a scalar loss.

### Step 1: Write the differential of $L$

We know that upstream, $L$ depends on $\mathbf{Y}$, and we have $\bar{\mathbf{Y}} = \frac{\partial L}{\partial \mathbf{Y}}$. So:

$$dL = \text{tr}\left(\bar{\mathbf{Y}}^T \, d\mathbf{Y}\right)$$

### Step 2: Express $d\mathbf{Y}$ in terms of $d\mathbf{A}$

Since $\mathbf{Y} = \mathbf{A}\mathbf{B}$ and $\mathbf{B}$ is treated as constant when differentiating w.r.t. $\mathbf{A}$:

$$d\mathbf{Y} = (d\mathbf{A})\mathbf{B}$$

### Step 3: Substitute

$$dL = \text{tr}\left(\bar{\mathbf{Y}}^T (d\mathbf{A}) \mathbf{B}\right)$$

### Step 4: Use trace properties to isolate $d\mathbf{A}$

Apply cyclic property: $\text{tr}(\mathbf{P}\mathbf{Q}\mathbf{R}) = \text{tr}(\mathbf{R}\mathbf{P}\mathbf{Q})$

$$dL = \text{tr}\left(\mathbf{B} \bar{\mathbf{Y}}^T (d\mathbf{A})\right) = \text{tr}\left((\bar{\mathbf{Y}} \mathbf{B}^T)^T d\mathbf{A}\right)$$

Wait — let me be more careful. We have $\text{tr}(\bar{\mathbf{Y}}^T (d\mathbf{A}) \mathbf{B})$.

Apply cyclic: move $\mathbf{B}$ to the front:

$$= \text{tr}\left(\mathbf{B} \bar{\mathbf{Y}}^T (d\mathbf{A})\right)$$

Now note that $\mathbf{B}\bar{\mathbf{Y}}^T = (\bar{\mathbf{Y}}\mathbf{B}^T)^T$. So:

$$= \text{tr}\left((\bar{\mathbf{Y}}\mathbf{B}^T)^T d\mathbf{A}\right)$$

### Step 5: Read off the gradient

Comparing with $dL = \text{tr}\left(\frac{\partial L}{\partial \mathbf{A}}^T d\mathbf{A}\right)$:

$$\boxed{\frac{\partial L}{\partial \mathbf{A}} = \bar{\mathbf{Y}} \mathbf{B}^T}$$

**This matches Document 4!** But we derived it in 5 lines instead of a page of index manipulation.

---

## 7.4 The Trace Trick: Gradient w.r.t. $\mathbf{B}$

**Goal**: $\frac{\partial L}{\partial \mathbf{B}}$ for $\mathbf{Y} = \mathbf{A}\mathbf{B}$.

$$dL = \text{tr}(\bar{\mathbf{Y}}^T \, d\mathbf{Y}) = \text{tr}(\bar{\mathbf{Y}}^T \mathbf{A} \, d\mathbf{B})$$

We need the form $\text{tr}(\mathbf{G}^T d\mathbf{B})$. Note that:

$$\text{tr}(\bar{\mathbf{Y}}^T \mathbf{A} \, d\mathbf{B}) = \text{tr}\left((\mathbf{A}^T \bar{\mathbf{Y}})^T d\mathbf{B}\right)$$

because $(\bar{\mathbf{Y}}^T \mathbf{A})^T = \mathbf{A}^T \bar{\mathbf{Y}}$, and $\text{tr}(\mathbf{M}^T \mathbf{N}) = \text{tr}(\mathbf{N}^T \mathbf{M})$ (transpose of product + trace transpose invariance).

Actually, let's be more explicit:

$$\text{tr}(\bar{\mathbf{Y}}^T \mathbf{A} \, d\mathbf{B})$$

Let $\mathbf{C} = \bar{\mathbf{Y}}^T \mathbf{A}$. Then we have $\text{tr}(\mathbf{C} \, d\mathbf{B})$.

Using $\text{tr}(\mathbf{C} \, d\mathbf{B}) = \text{tr}(\mathbf{C}^T d\mathbf{B})^?$ — No, that's wrong.

The identity we need: $\text{tr}(\mathbf{C} \, d\mathbf{B}) = \text{tr}(d\mathbf{B} \, \mathbf{C}) = \text{tr}(\mathbf{C}^T (d\mathbf{B})^T)$... 

Let me use the correct approach. The gradient identification rule is:

$$dL = \sum_{ij} G_{ij} \, dB_{ij} = \text{tr}(\mathbf{G}^T d\mathbf{B})$$

We have:

$$dL = \text{tr}(\bar{\mathbf{Y}}^T \mathbf{A} \, d\mathbf{B})$$

Let $\mathbf{G}^T = \bar{\mathbf{Y}}^T \mathbf{A}$, so $\mathbf{G} = \mathbf{A}^T \bar{\mathbf{Y}}$.

$$\boxed{\frac{\partial L}{\partial \mathbf{B}} = \mathbf{A}^T \bar{\mathbf{Y}}}$$

---

## 7.5 More Trace Trick Examples

### Example 1: $L = \text{tr}(\mathbf{A}\mathbf{X}\mathbf{B})$

$$dL = \text{tr}(\mathbf{A} \, d\mathbf{X} \, \mathbf{B})$$

Cyclic: $= \text{tr}(\mathbf{B}\mathbf{A} \, d\mathbf{X}) = \text{tr}((\mathbf{A}^T\mathbf{B}^T)^T d\mathbf{X})$

So: $\frac{\partial L}{\partial \mathbf{X}} = \mathbf{A}^T \mathbf{B}^T = (\mathbf{B}\mathbf{A})^T$

$$\boxed{\frac{\partial}{\partial \mathbf{X}} \text{tr}(\mathbf{A}\mathbf{X}\mathbf{B}) = \mathbf{A}^T\mathbf{B}^T}$$

### Example 2: $L = \text{tr}(\mathbf{X}^T\mathbf{A}\mathbf{X})$ (quadratic form, symmetric-like)

$$dL = \text{tr}((d\mathbf{X})^T \mathbf{A}\mathbf{X} + \mathbf{X}^T\mathbf{A} \, d\mathbf{X})$$

(Product rule: $d(\mathbf{X}^T\mathbf{A}\mathbf{X}) = (d\mathbf{X}^T)\mathbf{A}\mathbf{X} + \mathbf{X}^T\mathbf{A}(d\mathbf{X})$)

First term: $\text{tr}((d\mathbf{X})^T \mathbf{A}\mathbf{X}) = \text{tr}((\mathbf{A}\mathbf{X})^T d\mathbf{X})^?$

Actually: $\text{tr}(\mathbf{M}^T \mathbf{N}) = \text{tr}(\mathbf{N}^T \mathbf{M})$. With $\mathbf{M} = d\mathbf{X}$ and $\mathbf{N} = \mathbf{A}\mathbf{X}$:

$$\text{tr}((d\mathbf{X})^T \mathbf{A}\mathbf{X}) = \text{tr}((\mathbf{A}\mathbf{X})^T d\mathbf{X})$$

Wait, that uses $\text{tr}(\mathbf{P}^T\mathbf{Q}) = \text{tr}(\mathbf{Q}^T\mathbf{P})$... but actually $\text{tr}(\mathbf{P}^T\mathbf{Q})$ is the Frobenius inner product which is symmetric: $\langle \mathbf{P}, \mathbf{Q} \rangle_F = \langle \mathbf{Q}, \mathbf{P} \rangle_F$. So yes:

$$\text{tr}((d\mathbf{X})^T \mathbf{A}\mathbf{X}) = \text{tr}((\mathbf{A}\mathbf{X})^T d\mathbf{X})$$

This gives gradient contribution $\mathbf{A}\mathbf{X}$.

Second term: $\text{tr}(\mathbf{X}^T\mathbf{A} \, d\mathbf{X}) = \text{tr}((\mathbf{A}^T\mathbf{X})^T d\mathbf{X})$

(because $\mathbf{X}^T\mathbf{A} = (\mathbf{A}^T\mathbf{X})^T$)

This gives gradient contribution $\mathbf{A}^T\mathbf{X}$.

Total:
$$\boxed{\frac{\partial}{\partial \mathbf{X}} \text{tr}(\mathbf{X}^T\mathbf{A}\mathbf{X}) = (\mathbf{A} + \mathbf{A}^T)\mathbf{X}}$$

If $\mathbf{A}$ is symmetric ($\mathbf{A} = \mathbf{A}^T$), this simplifies to $2\mathbf{A}\mathbf{X}$.

### Example 3: $L = \|\mathbf{X}\mathbf{W} - \mathbf{T}\|_F^2$ (linear regression loss)

$$L = \text{tr}((\mathbf{X}\mathbf{W} - \mathbf{T})^T(\mathbf{X}\mathbf{W} - \mathbf{T}))$$

Let $\mathbf{R} = \mathbf{X}\mathbf{W} - \mathbf{T}$ (residual). Then $L = \text{tr}(\mathbf{R}^T\mathbf{R})$.

$$dL = \text{tr}((d\mathbf{R})^T\mathbf{R} + \mathbf{R}^T d\mathbf{R}) = 2\text{tr}(\mathbf{R}^T d\mathbf{R})$$

(using the symmetry of the Frobenius inner product: $\text{tr}((d\mathbf{R})^T\mathbf{R}) = \text{tr}(\mathbf{R}^T d\mathbf{R})$)

Now $d\mathbf{R} = \mathbf{X} \, d\mathbf{W}$ (since $\mathbf{T}$ is constant):

$$dL = 2\text{tr}(\mathbf{R}^T \mathbf{X} \, d\mathbf{W}) = 2\text{tr}((\mathbf{X}^T\mathbf{R})^T d\mathbf{W})$$

$$\boxed{\frac{\partial L}{\partial \mathbf{W}} = 2\mathbf{X}^T(\mathbf{X}\mathbf{W} - \mathbf{T})}$$

This is the well-known normal equations gradient! Setting to zero: $\mathbf{X}^T\mathbf{X}\mathbf{W} = \mathbf{X}^T\mathbf{T}$ → $\mathbf{W} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{T}$.

---

## 7.6 General Recipe: The Trace Trick Method

For any scalar loss $L$ that depends on matrix $\mathbf{X}$:

1. **Write** $dL = \text{tr}(\bar{\mathbf{Y}}^T \, d\mathbf{Y})$ where $\mathbf{Y}$ is the immediate output
2. **Express** $d\mathbf{Y}$ in terms of $d\mathbf{X}$ using differential rules
3. **Substitute** into the trace expression
4. **Manipulate** using trace properties (cyclic, transpose) to get the form $\text{tr}(\mathbf{G}^T d\mathbf{X})$
5. **Read off** $\frac{\partial L}{\partial \mathbf{X}} = \mathbf{G}$

The crucial step is Step 4: rearranging terms inside the trace to isolate $d\mathbf{X}$.

---

## 7.7 Perturbation Perspective

The trace trick is deeply connected to the idea of **perturbation**.

### The perturbation viewpoint

Suppose we perturb the input: $\mathbf{X} \to \mathbf{X} + \epsilon \, \mathbf{E}$ where $\mathbf{E}$ is an arbitrary direction matrix and $\epsilon$ is small. The change in the loss is:

$$L(\mathbf{X} + \epsilon\mathbf{E}) - L(\mathbf{X}) = \epsilon \cdot \text{tr}\left(\frac{\partial L}{\partial \mathbf{X}}^T \mathbf{E}\right) + O(\epsilon^2)$$

This is the **first-order Taylor expansion** of the loss around $\mathbf{X}$.

- The perturbation direction is $\mathbf{E}$
- The rate of change is $\text{tr}\left(\frac{\partial L}{\partial \mathbf{X}}^T \mathbf{E}\right)$
- This is the Frobenius inner product $\left\langle \frac{\partial L}{\partial \mathbf{X}}, \mathbf{E} \right\rangle_F$

### Connection to differentials

Setting $d\mathbf{X} = \epsilon \, \mathbf{E}$ and $dL = \epsilon \cdot \text{directional derivative}$:

$$dL = \text{tr}\left(\frac{\partial L}{\partial \mathbf{X}}^T d\mathbf{X}\right)$$

This IS the differential equation from Section 7.2. The trace trick is just the **algebraic formalization of perturbation analysis for matrices**.

### Connection to numerical gradients (Document 6)

Numerical gradient checking uses $\mathbf{E} = \mathbf{E}_{ij}$ (the single-element perturbation):

$$\frac{\partial L}{\partial X_{ij}} \approx \frac{L(\mathbf{X} + \epsilon \mathbf{E}_{ij}) - L(\mathbf{X} - \epsilon \mathbf{E}_{ij})}{2\epsilon}$$

The trace trick computes the SAME quantity analytically for ALL $(i,j)$ simultaneously.

---

## 7.8 Forward-Mode vs. Reverse-Mode: The Perturbation View

### Forward-mode AD (Jacobian-Vector Product, JVP)

**Idea**: Push a perturbation **forward** through the graph.

Given a perturbation $d\mathbf{X}$ at the input (called a "tangent"), compute the resulting perturbation $d\mathbf{Y}$ at the output.

For each operation $\mathbf{Y} = f(\mathbf{X})$:
$$d\mathbf{Y} = \mathbf{J}_f \cdot d\mathbf{X} \quad \text{(JVP: Jacobian × tangent vector)}$$

You propagate tangents **forward** through the graph, operation by operation.

**Cost**: One forward pass gives the directional derivative along ONE direction $d\mathbf{X}$. To get the full gradient (all partial derivatives), you need $n$ forward passes with $d\mathbf{X} = \mathbf{e}_1, \mathbf{e}_2, \ldots, \mathbf{e}_n$.

### Reverse-mode AD (Vector-Jacobian Product, VJP)

**Idea**: Push a "sensitivity" **backward** through the graph.

Given a sensitivity $\bar{\mathbf{Y}} = \frac{\partial L}{\partial \mathbf{Y}}$ at the output (called a "cotangent" or "adjoint"), compute the resulting sensitivity $\bar{\mathbf{X}}$ at the input.

For each operation $\mathbf{Y} = f(\mathbf{X})$:
$$\bar{\mathbf{X}} = \mathbf{J}_f^T \cdot \bar{\mathbf{Y}} \quad \text{(VJP: Jacobian}^T \times \text{cotangent vector)}$$

You propagate cotangents **backward** through the graph.

**Cost**: One backward pass gives the gradient w.r.t. ALL inputs. Start with $\bar{L} = 1$ (scalar loss) and one pass gives EVERY parameter gradient.

### Why neural networks use reverse mode

| Scenario | Forward passes needed | Backward passes needed |
|-----------|----------------------|----------------------|
| 1 output, $n$ inputs | $n$ (one per input dim) | **1** |
| $m$ outputs, 1 input | **1** | $m$ (one per output dim) |

Neural networks have 1 scalar loss and millions of parameters → reverse mode wins by factor of $10^6$.

---

## 7.9 JVP and VJP: Concrete Example

Consider $f(\mathbf{x}) = \mathbf{A}\mathbf{x}$ where $\mathbf{A} \in \mathbb{R}^{3 \times 2}$, $\mathbf{x} \in \mathbb{R}^2$.

Jacobian: $\mathbf{J} = \mathbf{A} \in \mathbb{R}^{3 \times 2}$.

**JVP** (forward mode): Given tangent $\mathbf{v} \in \mathbb{R}^2$:
$$\text{JVP}(\mathbf{v}) = \mathbf{A}\mathbf{v} \in \mathbb{R}^3$$

"If I perturb $\mathbf{x}$ by $\mathbf{v}$, the output changes by $\mathbf{A}\mathbf{v}$."

**VJP** (reverse mode): Given cotangent $\bar{\mathbf{y}} \in \mathbb{R}^3$:
$$\text{VJP}(\bar{\mathbf{y}}) = \mathbf{A}^T \bar{\mathbf{y}} \in \mathbb{R}^2$$

"If the loss is sensitive to $\mathbf{y}$ by $\bar{\mathbf{y}}$, then it's sensitive to $\mathbf{x}$ by $\mathbf{A}^T\bar{\mathbf{y}}$."

```python
import torch
from torch.func import jvp, vjp

A = torch.randn(3, 2)
x = torch.randn(2)

# Forward mode: JVP
tangent = torch.randn(2)  # perturbation direction
output, jvp_result = jvp(lambda x: A @ x, (x,), (tangent,))
print(f"JVP result: {jvp_result}")              # shape: (3,)
print(f"Manual:     {A @ tangent}")              # same
print(f"Match: {torch.allclose(jvp_result, A @ tangent)}")  # True

# Reverse mode: VJP
output, vjp_fn = vjp(lambda x: A @ x, x)
cotangent = torch.randn(3)  # sensitivity of loss w.r.t. output
(vjp_result,) = vjp_fn(cotangent)
print(f"\nVJP result: {vjp_result}")             # shape: (2,)
print(f"Manual:     {A.T @ cotangent}")          # same
print(f"Match: {torch.allclose(vjp_result, A.T @ cotangent)}")  # True
```

---

## 7.10 The Trace Trick for Non-Trivial Functions

### Softmax gradient via trace trick

Let $\mathbf{s} = \text{softmax}(\mathbf{x})$. From Document 5, the Jacobian is $\text{diag}(\mathbf{s}) - \mathbf{s}\mathbf{s}^T$.

Using the VJP directly:

$$\bar{\mathbf{x}} = (\text{diag}(\mathbf{s}) - \mathbf{s}\mathbf{s}^T)\bar{\mathbf{s}} = \mathbf{s} \odot \bar{\mathbf{s}} - \mathbf{s}(\mathbf{s}^T\bar{\mathbf{s}})$$

With the trace trick, we can derive this differently. Let's work with the differential:

$$ds_i = s_i \left(dx_i - \sum_j s_j \, dx_j\right) = s_i \, dx_i - s_i \sum_j s_j \, dx_j$$

(This comes from differentiating $s_i = \frac{e^{x_i}}{\sum_j e^{x_j}}$.)

Then:

$$dL = \sum_i \bar{s}_i \, ds_i = \sum_i \bar{s}_i s_i \, dx_i - \sum_i \bar{s}_i s_i \sum_j s_j dx_j$$

$$= \sum_i (\bar{s}_i s_i) dx_i - \left(\sum_i \bar{s}_i s_i\right)\left(\sum_j s_j dx_j\right)$$

Let $c = \sum_i \bar{s}_i s_i = \mathbf{s}^T\bar{\mathbf{s}}$:

$$dL = \sum_i s_i(\bar{s}_i - c) \, dx_i$$

Reading off: $\frac{\partial L}{\partial x_i} = s_i(\bar{s}_i - c)$, i.e., $\bar{\mathbf{x}} = \mathbf{s} \odot (\bar{\mathbf{s}} - c)$.

Same result as Document 5, derived more systematically.

---

## 7.11 The Trace Trick for Convolution

Using the im2col decomposition (Document 5), convolution becomes matmul: $\mathbf{Y}_{col} = \mathbf{X}_{col}\mathbf{K}_{col}$.

The trace trick for matmul immediately gives:

$$\frac{\partial L}{\partial \mathbf{K}_{col}} = \mathbf{X}_{col}^T \bar{\mathbf{Y}}_{col}$$
$$\frac{\partial L}{\partial \mathbf{X}_{col}} = \bar{\mathbf{Y}}_{col} \mathbf{K}_{col}^T$$

Then col2im maps $\frac{\partial L}{\partial \mathbf{X}_{col}}$ back to $\frac{\partial L}{\partial \mathbf{X}}$.

**This is exactly how framework authors handle convolution gradients** — reduce to matmul, apply known matmul gradient formulas.

---

## 7.12 Common Trace Trick Identities (Reference Table)

Useful results derived via the trace trick:

| Function $L$ | Gradient $\frac{\partial L}{\partial \mathbf{X}}$ |
|-------------|--------------------------------------------------|
| $\text{tr}(\mathbf{X})$ | $\mathbf{I}$ |
| $\text{tr}(\mathbf{A}\mathbf{X})$ | $\mathbf{A}^T$ |
| $\text{tr}(\mathbf{X}^T\mathbf{A})$ | $\mathbf{A}$ |
| $\text{tr}(\mathbf{A}\mathbf{X}\mathbf{B})$ | $\mathbf{A}^T\mathbf{B}^T$ |
| $\text{tr}(\mathbf{X}^T\mathbf{X})$ = $\|\mathbf{X}\|_F^2$ | $2\mathbf{X}$ |
| $\text{tr}(\mathbf{X}^T\mathbf{A}\mathbf{X})$ | $(\mathbf{A} + \mathbf{A}^T)\mathbf{X}$ |
| $\text{tr}(\mathbf{A}\mathbf{X}^T\mathbf{B})$ | $\mathbf{B}\mathbf{A}$ |

### Derivation practice: $L = \text{tr}(\mathbf{A}\mathbf{X}^T\mathbf{B})$

$$dL = \text{tr}(\mathbf{A} \, (d\mathbf{X})^T \mathbf{B})$$

Cyclic: $= \text{tr}(\mathbf{B}\mathbf{A} \, (d\mathbf{X})^T)$

Using $\text{tr}(\mathbf{M}(d\mathbf{X})^T) = \text{tr}(\mathbf{M}^T d\mathbf{X})^?$

Actually: $\text{tr}(\mathbf{M}\mathbf{N}^T) = \text{tr}(\mathbf{N}\mathbf{M}^T)$ (from $\text{tr}(\mathbf{P}) = \text{tr}(\mathbf{P}^T)$ applied to $\mathbf{P} = \mathbf{M}\mathbf{N}^T$).

Hmm, let's use the index form to be rigorous:

$$\text{tr}(\mathbf{B}\mathbf{A} (d\mathbf{X})^T) = \sum_i [\mathbf{B}\mathbf{A}(d\mathbf{X})^T]_{ii} = \sum_i \sum_j (BA)_{ij} (dX)^T_{ji} = \sum_{ij} (BA)_{ij} (dX)_{ij}$$

$$= \text{tr}((\mathbf{B}\mathbf{A})^T d\mathbf{X})$$

So: $\frac{\partial L}{\partial \mathbf{X}} = \mathbf{B}\mathbf{A}$ (not $(\mathbf{B}\mathbf{A})^T$ — the transpose is absorbed into the identification rule). 

Wait, let me be more careful. From the identification:

$$dL = \text{tr}(\mathbf{G}^T d\mathbf{X}) \implies \mathbf{G} = \frac{\partial L}{\partial \mathbf{X}}$$

We have $dL = \text{tr}((\mathbf{B}\mathbf{A})^T d\mathbf{X})$, so $\mathbf{G}^T = (\mathbf{B}\mathbf{A})^T$, thus $\mathbf{G} = \mathbf{B}\mathbf{A}$.

Actually: $\text{tr}((\mathbf{BA})^T d\mathbf{X})$ already has the form $\text{tr}(\mathbf{G}^T d\mathbf{X})$ where $\mathbf{G} = \mathbf{BA}$.

$$\boxed{\frac{\partial}{\partial \mathbf{X}} \text{tr}(\mathbf{A}\mathbf{X}^T\mathbf{B}) = \mathbf{B}\mathbf{A}}$$

---

## 7.13 Connection Between All the Concepts

```
Perturbation (physical idea)
    "What happens if I wiggle the input slightly?"
         │
         ▼
Differential (algebraic formalization)
    dL = tr(G^T dX)
         │
         ▼
Trace Trick (efficient algebraic manipulation)
    Use cyclic property + transpose to isolate dX
         │
         ▼
Gradient Formula (result)
    G = ∂L/∂X
         │
         ├──→ Forward-mode AD (JVP): push perturbation forward
         │    dY = J @ dX, one direction at a time
         │
         └──→ Reverse-mode AD (VJP): push sensitivity backward
              dX = J^T @ dY, ALL inputs in one pass
                   │
                   ▼
              Backpropagation (reverse AD on a graph)
              Chain VJPs through the computation graph
                   │
                   ▼
              Numerical check (Document 6)
              Verify by actual perturbation + finite difference
```

**Everything is connected.** The trace trick derives the formula. Reverse-mode AD (backprop) evaluates it efficiently on a graph. Numerical gradient checking validates it by literal perturbation.

---

## 7.14 PyTorch's `torch.func`: JVP and VJP Directly

```python
import torch
from torch.func import jvp, vjp, jacrev, jacfwd

def f(W, x):
    """Simple linear + squared loss."""
    y = W @ x
    return (y ** 2).sum()

W = torch.randn(3, 4)
x = torch.randn(4)

# === VJP (reverse mode, what .backward() does) ===
output, vjp_fn = vjp(f, W, x)
# Cotangent for scalar output is just 1.0
grad_W, grad_x = vjp_fn(torch.tensor(1.0))
print(f"grad_W shape: {grad_W.shape}")  # (3, 4)
print(f"grad_x shape: {grad_x.shape}")  # (4,)

# === JVP (forward mode) ===
tangent_W = torch.randn_like(W)  # random perturbation direction
tangent_x = torch.zeros_like(x)  # no perturbation in x
output, jvp_result = jvp(f, (W, x), (tangent_W, tangent_x))
print(f"\nJVP result (scalar): {jvp_result}")
# This is the directional derivative along tangent_W

# Verify: JVP result should equal tr(grad_W^T @ tangent_W)
manual = (grad_W * tangent_W).sum()
print(f"Manual:              {manual}")
print(f"Match: {torch.allclose(jvp_result, manual)}")  # True

# === Full Jacobian (via reverse mode) ===
# For vector-valued functions:
def g(x):
    return torch.stack([x[0]**2 + x[1], x[0]*x[1] + x[1]**2])

x = torch.tensor([2.0, 3.0])
J_reverse = jacrev(g)(x)   # Uses reverse-mode AD
J_forward = jacfwd(g)(x)   # Uses forward-mode AD
print(f"\nJacobian (reverse): \n{J_reverse}")
print(f"Jacobian (forward): \n{J_forward}")
# Both should give [[2*x0, 1], [x1, x0+2*x1]] = [[4, 1], [3, 8]]
```

---

## 7.15 Summary

| Concept | What It Is | When to Use |
|---------|-----------|-------------|
| **Matrix differential** | $dL = \text{tr}(\mathbf{G}^T d\mathbf{X})$ | Derive gradient formulas algebraically |
| **Trace trick** | Use cyclic/transpose properties to isolate $d\mathbf{X}$ inside trace | Any matrix-valued operation |
| **Perturbation** | Wiggle input, observe output change | Physical intuition; numerical validation |
| **JVP** (forward mode) | $\mathbf{J}\mathbf{v}$: push tangent forward | Few outputs, many inputs NOT typical of NN |
| **VJP** (reverse mode) | $\mathbf{J}^T\bar{\mathbf{y}}$: push cotangent backward | Many inputs, one output: ALL of neural network training |
| **Backpropagation** | Chain VJPs through computation graph | What `.backward()` does |

### The workflow of a framework author

1. **New operation** $f$: define forward semantics
2. **Derive gradient** using trace trick (or element-wise if simpler)
3. **Simplify** to efficient formula avoiding explicit Jacobian
4. **Implement** backward function (VJP)
5. **Validate** with `torch.autograd.gradcheck` (numerical perturbation)
6. **Optimize** with CUDA/C++ kernel

---

**Next**: Document 8 puts it all together — implementing a custom operation in PyTorch with hand-derived forward and backward passes, validated via numerical gradient checking.
