# 3. Matrix Calculus & the Backpropagation Algorithm

## Motivation

You know single-variable calculus: $\frac{d}{dx}(x^2) = 2x$. You know multivariate partial derivatives: $\frac{\partial}{\partial x}(xy) = y$. But deep learning operates on **matrices and tensors**, and the chain rule must compose through **matrix-valued functions**. This document builds the mathematical framework — Jacobians, the multivariate chain rule, and reverse-mode automatic differentiation — needed to derive gradients for **any** operation.

---

## 3.1 Derivatives of Vector-Valued Functions: The Jacobian

### Scalar → Scalar
$f: \mathbb{R} \to \mathbb{R}$, derivative is $f'(x) \in \mathbb{R}$.

### Vector → Scalar
$f: \mathbb{R}^n \to \mathbb{R}$, gradient is $\nabla f \in \mathbb{R}^n$:
$$\nabla f = \begin{pmatrix} \frac{\partial f}{\partial x_1} \\ \vdots \\ \frac{\partial f}{\partial x_n} \end{pmatrix}$$

### Vector → Vector (The Jacobian)
$\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$, the **Jacobian matrix** $\mathbf{J} \in \mathbb{R}^{m \times n}$:

$$\mathbf{J} = \frac{\partial \mathbf{f}}{\partial \mathbf{x}} = \begin{pmatrix}
\frac{\partial f_1}{\partial x_1} & \frac{\partial f_1}{\partial x_2} & \cdots & \frac{\partial f_1}{\partial x_n} \\
\frac{\partial f_2}{\partial x_1} & \frac{\partial f_2}{\partial x_2} & \cdots & \frac{\partial f_2}{\partial x_n} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial f_m}{\partial x_1} & \frac{\partial f_m}{\partial x_2} & \cdots & \frac{\partial f_m}{\partial x_n}
\end{pmatrix}$$

**Row $i$**: how all inputs affect output $i$ — the gradient of $f_i$.
**Column $j$**: how input $j$ affects all outputs — the "influence" of $x_j$.

**Shape rule**: If $\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$, the Jacobian is $m \times n$.

### Example: $\mathbf{f}(\mathbf{x}) = \begin{pmatrix} x_1 x_2 \\ x_1 + x_2^2 \end{pmatrix}$

$$\mathbf{J} = \begin{pmatrix} \frac{\partial(x_1 x_2)}{\partial x_1} & \frac{\partial(x_1 x_2)}{\partial x_2} \\ \frac{\partial(x_1 + x_2^2)}{\partial x_1} & \frac{\partial(x_1 + x_2^2)}{\partial x_2} \end{pmatrix} = \begin{pmatrix} x_2 & x_1 \\ 1 & 2x_2 \end{pmatrix}$$

At $\mathbf{x} = (3, 2)^T$:
$$\mathbf{J} = \begin{pmatrix} 2 & 3 \\ 1 & 4 \end{pmatrix}$$

---

## 3.2 The Multivariate Chain Rule

### Scalar chain rule (review)
If $y = f(g(x))$, then $\frac{dy}{dx} = f'(g(x)) \cdot g'(x)$.

### Vector chain rule (Jacobian multiplication)
If $\mathbf{y} = \mathbf{f}(\mathbf{g}(\mathbf{x}))$ where:
- $\mathbf{g}: \mathbb{R}^n \to \mathbb{R}^k$ with Jacobian $\mathbf{J}_g \in \mathbb{R}^{k \times n}$
- $\mathbf{f}: \mathbb{R}^k \to \mathbb{R}^m$ with Jacobian $\mathbf{J}_f \in \mathbb{R}^{m \times k}$

Then:
$$\mathbf{J}_{f \circ g} = \mathbf{J}_f \cdot \mathbf{J}_g \in \mathbb{R}^{m \times n}$$

**The chain rule for vectors is matrix multiplication of Jacobians.**

### For a composition of 3 functions:
$$\mathbf{h} = \mathbf{f}_3(\mathbf{f}_2(\mathbf{f}_1(\mathbf{x})))$$

$$\mathbf{J}_{total} = \mathbf{J}_{f_3} \cdot \mathbf{J}_{f_2} \cdot \mathbf{J}_{f_1}$$

---

## 3.3 The Gradient of a Scalar Loss (What Backprop Actually Computes)

In deep learning, we have:

$$L = \text{loss}(\mathbf{f}_K(\mathbf{f}_{K-1}(\cdots \mathbf{f}_1(\mathbf{x}) \cdots)))$$

Where $L \in \mathbb{R}$ is a scalar. We want $\frac{\partial L}{\partial \mathbf{x}} \in \mathbb{R}^n$ (same shape as $\mathbf{x}$).

Let's define intermediate variables:
$$\mathbf{z}_0 = \mathbf{x}, \quad \mathbf{z}_k = \mathbf{f}_k(\mathbf{z}_{k-1}), \quad L = \ell(\mathbf{z}_K)$$

By the chain rule:
$$\frac{\partial L}{\partial \mathbf{z}_0} = \frac{\partial L}{\partial \mathbf{z}_K} \cdot \mathbf{J}_{f_K} \cdot \mathbf{J}_{f_{K-1}} \cdots \mathbf{J}_{f_1}$$

Where $\frac{\partial L}{\partial \mathbf{z}_K}$ is a row vector (gradient of scalar w.r.t. vector, transposed).

### Two ways to evaluate this product

The expression $\mathbf{v}^T \cdot \mathbf{J}_K \cdot \mathbf{J}_{K-1} \cdots \mathbf{J}_1$ can be evaluated:

**Left-to-right (reverse mode = backpropagation)**:
$$\underbrace{(\underbrace{(\underbrace{\mathbf{v}^T \cdot \mathbf{J}_K}_{\text{vector}}) \cdot \mathbf{J}_{K-1}}_{\text{vector}}) \cdots \cdot \mathbf{J}_1}_{\text{vector}}$$

Each step is a **vector-Jacobian product (VJP)**: multiply a row vector by a matrix. Result: always a row vector. Cost: $O(n \cdot k)$ per step.

**Right-to-left (forward mode)**:
$$\mathbf{J}_K \cdot (\mathbf{J}_{K-1} \cdot (\cdots (\mathbf{J}_1 \cdot \mathbf{v}) \cdots))$$

Each step is a **Jacobian-vector product (JVP)**: multiply a matrix by a column vector. Result: always a column vector. Cost: $O(m \cdot k)$ per step.

### Why reverse mode wins for neural networks

In neural networks:
- The output is a **scalar** loss ($m = 1$)
- The inputs are **many** parameters ($n$ could be millions)

Reverse mode starts from the 1-dimensional output and propagates backward. Each VJP takes a vector and produces a vector — no matrices are ever fully materialized.

Forward mode would need to start from each of the $n$ input dimensions separately — requiring $n$ passes. That's $10^6$ times more expensive for a network with $10^6$ parameters.

**Backpropagation IS reverse-mode automatic differentiation. Nothing more.**

---

## 3.4 The VJP: The Fundamental Operation of Backprop

The **vector-Jacobian product** (VJP) is the core primitive:

Given:
- upstream gradient $\bar{\mathbf{z}} = \frac{\partial L}{\partial \mathbf{z}} \in \mathbb{R}^m$ (called `grad_output` in PyTorch)
- operation $\mathbf{z} = \mathbf{f}(\mathbf{x})$ with Jacobian $\mathbf{J}_f \in \mathbb{R}^{m \times n}$

Compute:
$$\bar{\mathbf{x}} = \bar{\mathbf{z}}^T \mathbf{J}_f \quad \in \mathbb{R}^{1 \times n}$$

Or equivalently (treating gradients as column vectors as PyTorch does):
$$\bar{\mathbf{x}} = \mathbf{J}_f^T \bar{\mathbf{z}} \quad \in \mathbb{R}^{n}$$

This gives us $\frac{\partial L}{\partial \mathbf{x}}$ without ever forming the full Jacobian $\mathbf{J}_f$.

**The "backward" function of each operation computes exactly this VJP.**

---

## 3.5 The Backprop Algorithm (Pseudocode)

```
Algorithm: Backpropagation (Reverse-Mode AD)

Input: Computation graph G, scalar loss L
Output: Gradient of L w.r.t. every leaf parameter

1. Run forward pass:
     For each operation in topological order:
       z_k = f_k(inputs_k)
       Store z_k and inputs_k for later use

2. Initialize:
     grad[L] = 1.0                    # dL/dL = 1

3. For each operation in REVERSE topological order:
     # Get upstream gradient
     upstream = grad[z_k]              # = dL/dz_k

     # Compute VJP for each input of operation k
     For each input x_j of f_k:
       local_grad = VJP(f_k, x_j, upstream)   # = J_f_k^T @ upstream
       grad[x_j] += local_grad         # ACCUMULATE (for fan-out)

4. Return grad[parameter] for each leaf parameter
```

### Key details:

**Topological order**: ensures every output is computed before it's needed as input (forward), and every upstream gradient is available before computing downstream (backward).

**Accumulation** (`+=`): handles the case where a tensor feeds into multiple operations (fan-out). Gradients from all consumers are summed.

**Storage**: intermediate values ($\mathbf{z}_k$ and their inputs) must be stored during the forward pass for use in backward. This is why training uses more memory than inference.

---

## 3.6 Worked Example: 3-Layer Computation

Consider: $L = \|\sigma(\mathbf{W}_2 \cdot \sigma(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2) - \mathbf{t}\|^2$

Where $\sigma$ is an element-wise activation (e.g., ReLU).

**Decompose into operations:**

| Operation | Formula | Shapes |
|-----------|---------|--------|
| $\mathbf{z}_1 = \mathbf{W}_1 \mathbf{x} + \mathbf{b}_1$ | Linear | $\mathbf{x} \in \mathbb{R}^3, \mathbf{W}_1 \in \mathbb{R}^{4 \times 3}, \mathbf{z}_1 \in \mathbb{R}^4$ |
| $\mathbf{a}_1 = \sigma(\mathbf{z}_1)$ | Activation | $\mathbf{a}_1 \in \mathbb{R}^4$ |
| $\mathbf{z}_2 = \mathbf{W}_2 \mathbf{a}_1 + \mathbf{b}_2$ | Linear | $\mathbf{W}_2 \in \mathbb{R}^{2 \times 4}, \mathbf{z}_2 \in \mathbb{R}^2$ |
| $\mathbf{a}_2 = \sigma(\mathbf{z}_2)$ | Activation | $\mathbf{a}_2 \in \mathbb{R}^2$ |
| $L = \|\mathbf{a}_2 - \mathbf{t}\|^2$ | Loss | $\mathbf{t} \in \mathbb{R}^2, L \in \mathbb{R}$ |

**Forward pass** (compute and store all intermediates).

**Backward pass**:

**Step 1**: $\frac{\partial L}{\partial \mathbf{a}_2} = 2(\mathbf{a}_2 - \mathbf{t}) \in \mathbb{R}^2$

**Step 2**: $\frac{\partial L}{\partial \mathbf{z}_2} = \frac{\partial L}{\partial \mathbf{a}_2} \odot \sigma'(\mathbf{z}_2) \in \mathbb{R}^2$

  (Element-wise because $\sigma$ acts element-wise → diagonal Jacobian → Hadamard product)

**Step 3**: $\frac{\partial L}{\partial \mathbf{W}_2} = \frac{\partial L}{\partial \mathbf{z}_2} \cdot \mathbf{a}_1^T \in \mathbb{R}^{2 \times 4}$ (outer product)

$$\frac{\partial L}{\partial \mathbf{b}_2} = \frac{\partial L}{\partial \mathbf{z}_2} \in \mathbb{R}^2$$

$$\frac{\partial L}{\partial \mathbf{a}_1} = \mathbf{W}_2^T \frac{\partial L}{\partial \mathbf{z}_2} \in \mathbb{R}^4$$

**Step 4**: $\frac{\partial L}{\partial \mathbf{z}_1} = \frac{\partial L}{\partial \mathbf{a}_1} \odot \sigma'(\mathbf{z}_1) \in \mathbb{R}^4$

**Step 5**: $\frac{\partial L}{\partial \mathbf{W}_1} = \frac{\partial L}{\partial \mathbf{z}_1} \cdot \mathbf{x}^T \in \mathbb{R}^{4 \times 3}$

$$\frac{\partial L}{\partial \mathbf{b}_1} = \frac{\partial L}{\partial \mathbf{z}_1} \in \mathbb{R}^4$$

$$\frac{\partial L}{\partial \mathbf{x}} = \mathbf{W}_1^T \frac{\partial L}{\partial \mathbf{z}_1} \in \mathbb{R}^3$$

**Observe the pattern for linear layers ($\mathbf{z} = \mathbf{W}\mathbf{a} + \mathbf{b}$)**:
- $\frac{\partial L}{\partial \mathbf{W}} = \frac{\partial L}{\partial \mathbf{z}} \cdot \mathbf{a}^T$ (outer product of upstream gradient and input)
- $\frac{\partial L}{\partial \mathbf{b}} = \frac{\partial L}{\partial \mathbf{z}}$ (just passes through)
- $\frac{\partial L}{\partial \mathbf{a}} = \mathbf{W}^T \frac{\partial L}{\partial \mathbf{z}}$ (transpose of weight times upstream)

---

## 3.7 Toy Numerical Example

Let's compute everything with actual numbers. A 2-layer network:

$$\mathbf{x} = \begin{pmatrix} 1 \\ 2 \end{pmatrix}, \quad \mathbf{W}_1 = \begin{pmatrix} 0.1 & 0.2 \\ 0.3 & 0.4 \end{pmatrix}, \quad \mathbf{b}_1 = \begin{pmatrix} 0 \\ 0 \end{pmatrix}$$

Using ReLU activation, and $\mathbf{W}_2 = \begin{pmatrix} 0.5 & 0.6 \end{pmatrix}$, $b_2 = 0$, target $t = 1$.

**Forward pass:**

$\mathbf{z}_1 = \mathbf{W}_1 \mathbf{x} + \mathbf{b}_1 = \begin{pmatrix} 0.1 \cdot 1 + 0.2 \cdot 2 \\ 0.3 \cdot 1 + 0.4 \cdot 2 \end{pmatrix} = \begin{pmatrix} 0.5 \\ 1.1 \end{pmatrix}$

$\mathbf{a}_1 = \text{ReLU}(\mathbf{z}_1) = \begin{pmatrix} 0.5 \\ 1.1 \end{pmatrix}$ (both positive, so ReLU is identity)

$z_2 = \mathbf{W}_2 \mathbf{a}_1 + b_2 = 0.5 \cdot 0.5 + 0.6 \cdot 1.1 = 0.25 + 0.66 = 0.91$

$a_2 = \text{ReLU}(z_2) = 0.91$

$L = (a_2 - t)^2 = (0.91 - 1)^2 = (-0.09)^2 = 0.0081$

**Backward pass:**

$\frac{\partial L}{\partial a_2} = 2(a_2 - t) = 2(-0.09) = -0.18$

$\frac{\partial L}{\partial z_2} = \frac{\partial L}{\partial a_2} \cdot \sigma'(z_2) = -0.18 \cdot 1 = -0.18$ (ReLU'(0.91) = 1)

$\frac{\partial L}{\partial \mathbf{W}_2} = \frac{\partial L}{\partial z_2} \cdot \mathbf{a}_1^T = -0.18 \cdot \begin{pmatrix} 0.5 & 1.1 \end{pmatrix} = \begin{pmatrix} -0.09 & -0.198 \end{pmatrix}$

$\frac{\partial L}{\partial b_2} = \frac{\partial L}{\partial z_2} = -0.18$

$\frac{\partial L}{\partial \mathbf{a}_1} = \mathbf{W}_2^T \frac{\partial L}{\partial z_2} = \begin{pmatrix} 0.5 \\ 0.6 \end{pmatrix} \cdot (-0.18) = \begin{pmatrix} -0.09 \\ -0.108 \end{pmatrix}$

$\frac{\partial L}{\partial \mathbf{z}_1} = \frac{\partial L}{\partial \mathbf{a}_1} \odot \sigma'(\mathbf{z}_1) = \begin{pmatrix} -0.09 \\ -0.108 \end{pmatrix} \odot \begin{pmatrix} 1 \\ 1 \end{pmatrix} = \begin{pmatrix} -0.09 \\ -0.108 \end{pmatrix}$

$\frac{\partial L}{\partial \mathbf{W}_1} = \frac{\partial L}{\partial \mathbf{z}_1} \cdot \mathbf{x}^T = \begin{pmatrix} -0.09 \\ -0.108 \end{pmatrix} \begin{pmatrix} 1 & 2 \end{pmatrix} = \begin{pmatrix} -0.09 & -0.18 \\ -0.108 & -0.216 \end{pmatrix}$

$\frac{\partial L}{\partial \mathbf{b}_1} = \frac{\partial L}{\partial \mathbf{z}_1} = \begin{pmatrix} -0.09 \\ -0.108 \end{pmatrix}$

---

## 3.8 Verify with PyTorch

```python
import torch

x = torch.tensor([1.0, 2.0])
W1 = torch.tensor([[0.1, 0.2], [0.3, 0.4]], requires_grad=True)
b1 = torch.tensor([0.0, 0.0], requires_grad=True)
W2 = torch.tensor([[0.5, 0.6]], requires_grad=True)
b2 = torch.tensor([0.0], requires_grad=True)
t = torch.tensor([1.0])

# Forward
z1 = W1 @ x + b1
a1 = torch.relu(z1)
z2 = W2 @ a1 + b2
a2 = torch.relu(z2)
L = (a2 - t) ** 2

L.backward()

print(f"L = {L.item():.4f}")            # 0.0081
print(f"W2.grad = {W2.grad}")           # tensor([[-0.0900, -0.1980]])
print(f"b2.grad = {b2.grad}")           # tensor([-0.1800])
print(f"W1.grad =\n{W1.grad}")          # [[-0.09, -0.18], [-0.108, -0.216]]
print(f"b1.grad = {b1.grad}")           # tensor([-0.0900, -0.1080])
```

**Output matches our hand calculation exactly.**

---

## 3.9 Jacobian Structure for Common Operations

Understanding Jacobian structure tells you how complex the backward pass is:

| Operation | Jacobian Structure | VJP Cost |
|-----------|--------------------|----------|
| Element-wise $f(x_i)$ | **Diagonal** ($J_{ii} = f'(x_i)$) | $O(n)$ — just Hadamard product |
| Linear $\mathbf{y} = \mathbf{W}\mathbf{x}$ | **Dense** ($J = \mathbf{W}$) | $O(mn)$ — matrix-vector product |
| Sum $y = \sum x_i$ | **Row of ones** | $O(n)$ — broadcast |
| Softmax | **Dense** ($J_{ij} = s_i(\delta_{ij} - s_j)$) | $O(n^2)$ or $O(n)$ via identity |
| Convolution | **Sparse, structured** (Toeplitz-like) | $O(\text{input} \times \text{kernel})$ |

**The Jacobian is never explicitly formed in practice.** Only VJPs are computed.

---

## 3.10 Memory Cost of Backpropagation

Why does training use 2-3x more memory than inference?

During the forward pass, we must **store** intermediate activations for use in the backward pass:

- $\mathbf{z}_1, \mathbf{a}_1, \mathbf{z}_2, \mathbf{a}_2, \ldots$ — all needed for gradient computation
- Example: computing $\frac{\partial L}{\partial \mathbf{W}_1} = \frac{\partial L}{\partial \mathbf{z}_1} \cdot \mathbf{x}^T$ requires $\mathbf{x}$
- Computing $\frac{\partial L}{\partial \mathbf{z}_1} = \frac{\partial L}{\partial \mathbf{a}_1} \odot \sigma'(\mathbf{z}_1)$ requires $\mathbf{z}_1$

**Gradient checkpointing** is a technique that trades compute for memory: discard some intermediates during forward, recompute them during backward. PyTorch supports this via `torch.utils.checkpoint`.

---

## 3.11 Summary: The Mathematical Core

| Concept | Formula | Significance |
|---------|---------|-------------|
| Jacobian | $\mathbf{J} \in \mathbb{R}^{m \times n}$, $J_{ij} = \frac{\partial f_i}{\partial x_j}$ | Encodes all first-order information |
| Vector chain rule | $\mathbf{J}_{f \circ g} = \mathbf{J}_f \cdot \mathbf{J}_g$ | Chain rule = matrix multiplication |
| VJP | $\bar{\mathbf{x}} = \mathbf{J}^T \bar{\mathbf{z}}$ | The one operation backprop needs |
| Reverse mode | Evaluate $\bar{\mathbf{z}}^T \mathbf{J}_K \cdots \mathbf{J}_1$ left-to-right | One pass gives ALL parameter gradients |
| Forward mode | Evaluate $\mathbf{J}_K \cdots \mathbf{J}_1 \mathbf{v}$ right-to-left | One pass gives ONE directional derivative |

### The golden equation of backpropagation:

$$\boxed{\frac{\partial L}{\partial \text{input}} = \mathbf{J}_{op}^T \cdot \frac{\partial L}{\partial \text{output}}}$$

Every `backward()` function in PyTorch computes exactly this. The only question for each new operation is: **what is $\mathbf{J}_{op}^T$ and how do you multiply by it efficiently without forming it explicitly?**

That is the subject of the next document.

---

**Next**: Document 4 derives the specific VJP formulas for matmul, element-wise operations, and reductions — the building blocks of all neural networks.
