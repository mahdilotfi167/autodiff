# 1. Computation Graphs in Deep Learning

## Motivation

Every modern deep learning framework (PyTorch, TensorFlow, JAX) represents computation as a **directed acyclic graph (DAG)**. This is not just an implementation detail — it is **the** core abstraction that makes automatic differentiation possible. Without understanding the graph, you cannot understand how gradients flow, why `.backward()` works, or how to debug gradient issues.

---

## 1.1 What Is a Computation Graph?

A computation graph is a DAG where:

- **Nodes** represent **operations** (add, multiply, matmul, ReLU, etc.)
- **Edges** represent **tensors** — the data flowing between operations
- **Leaf nodes** are inputs (parameters, data) — they have no incoming operation
- **Root node** is typically the scalar loss value

### Example: $L = (x \cdot w + b)^2$

```
  x    w        b
   \  /         |
   [mul]        |
     \         /
      [add]---
        |
      [square]
        |
        L
```

Each box `[op]` is an operation node. Each arrow carries a tensor (possibly a scalar).

**Key property**: the graph records the **recipe** of how the output was computed from the inputs. Given this recipe, we can mechanically compute derivatives by walking the graph **backwards**.

---

## 1.2 Forward Pass: Building the Graph

During the **forward pass**, you compute the output from inputs. As a side effect, the framework **records** every operation into the graph.

### Step-by-step for $L = (xw + b)^2$ with $x=2, w=3, b=1$:

| Step | Operation | Result | Stored in graph |
|------|-----------|--------|-----------------|
| 1 | $t_1 = x \cdot w$ | $t_1 = 6$ | `MulBackward`: inputs were $x, w$ |
| 2 | $t_2 = t_1 + b$ | $t_2 = 7$ | `AddBackward`: inputs were $t_1, b$ |
| 3 | $L = t_2^2$ | $L = 49$ | `PowBackward`: input was $t_2$ |

After this forward pass, we have:
- The **value** of every intermediate tensor ($t_1=6, t_2=7, L=49$)
- The **graph structure**: which operation produced each tensor, and from what inputs

---

## 1.3 Backward Pass: Traversing the Graph in Reverse

To compute gradients, we walk the graph **from the loss back to the leaves**:

1. Start at $L$ with $\frac{\partial L}{\partial L} = 1$
2. At each operation node, compute the **local gradient** (derivative of output w.r.t. each input)
3. Multiply the incoming gradient by the local gradient (chain rule)
4. Pass the result to the next node upstream

### Backward pass for our example:

**Step 3 (reverse)**: $L = t_2^2$
$$\frac{\partial L}{\partial t_2} = 2t_2 = 2 \times 7 = 14$$

**Step 2 (reverse)**: $t_2 = t_1 + b$
$$\frac{\partial L}{\partial t_1} = \frac{\partial L}{\partial t_2} \cdot \frac{\partial t_2}{\partial t_1} = 14 \cdot 1 = 14$$
$$\frac{\partial L}{\partial b} = \frac{\partial L}{\partial t_2} \cdot \frac{\partial t_2}{\partial b} = 14 \cdot 1 = 14$$

**Step 1 (reverse)**: $t_1 = x \cdot w$
$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial t_1} \cdot \frac{\partial t_1}{\partial x} = 14 \cdot w = 14 \cdot 3 = 42$$
$$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial t_1} \cdot \frac{\partial t_1}{\partial w} = 14 \cdot x = 14 \cdot 2 = 28$$

---

## 1.4 The Graph Is Dynamic in PyTorch

PyTorch uses a **dynamic computation graph** (also called "define-by-run"):

- The graph is built **during execution** of the forward pass
- Every time you run forward, a **new graph** is created
- After `.backward()`, the graph is **destroyed** (by default)

This means:
- You can use Python `if/else`, loops, etc. — the graph adapts each run
- Each forward pass can have a **different** graph structure
- You must call forward again before calling backward again

Contrast with TensorFlow 1.x which used **static graphs** (define-then-run): you declared the graph first, then fed data into it. TensorFlow 2.x moved toward dynamic graphs via eager mode.

---

## 1.5 PyTorch's Graph Internals

Every `torch.Tensor` that participates in computation has:

| Attribute | Meaning |
|-----------|---------|
| `.data` | The actual numerical values |
| `.grad` | Accumulated gradient (populated after `.backward()`) |
| `.grad_fn` | The operation that **created** this tensor (points to graph node) |
| `.requires_grad` | Whether this tensor is tracked by the graph |
| `.is_leaf` | Whether this is a leaf node (no `grad_fn`) |

### Graph node chain

When you do `c = a * b`, then `c.grad_fn` is a `MulBackward0` object. That object has references to the `grad_fn` of `a` and `b`. This forms a **linked list** (actually a DAG) from the output back to the leaves.

```
L.grad_fn → PowBackward0
               ↓ (next_functions)
             AddBackward0
               ↓ (next_functions)
             MulBackward0
               ↓ (next_functions)
            (AccumulateGrad for x, AccumulateGrad for w)
```

`AccumulateGrad` is a special node for leaf tensors — it accumulates (adds) gradients into the `.grad` attribute.

---

## 1.6 PyTorch Code Walkthrough

```python
import torch

# Create leaf tensors (requires_grad=True to track them)
x = torch.tensor(2.0, requires_grad=True)
w = torch.tensor(3.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)

# Forward pass — builds the graph
t1 = x * w          # MulBackward0
t2 = t1 + b         # AddBackward0
L = t2 ** 2          # PowBackward0

# Inspect the graph
print(f"L.grad_fn        = {L.grad_fn}")           # PowBackward0
print(f"t2.grad_fn       = {t2.grad_fn}")           # AddBackward0
print(f"t1.grad_fn       = {t1.grad_fn}")           # MulBackward0
print(f"x.grad_fn        = {x.grad_fn}")            # None (leaf)
print(f"x.is_leaf        = {x.is_leaf}")            # True

# Walk the graph manually
print("\n--- Graph structure ---")
print(f"L created by: {L.grad_fn}")
print(f"  inputs: {L.grad_fn.next_functions}")
print(f"  t2 created by: {t2.grad_fn}")
print(f"    inputs: {t2.grad_fn.next_functions}")

# Backward pass — computes gradients
L.backward()

# Check gradients
print(f"\ndL/dx = {x.grad}")   # 42.0
print(f"dL/dw = {w.grad}")     # 28.0
print(f"dL/db = {b.grad}")     # 14.0
```

**Output:**
```
L.grad_fn        = <PowBackward0 object at 0x...>
t2.grad_fn       = <AddBackward0 object at 0x...>
t1.grad_fn       = <MulBackward0 object at 0x...>
x.grad_fn        = None
x.is_leaf        = True

--- Graph structure ---
L created by: <PowBackward0 object at 0x...>
  inputs: ((<AddBackward0 object at 0x...>, 0),)
  t2 created by: <AddBackward0 object at 0x...>
    inputs: ((<MulBackward0 object at 0x...>, 0), (<AccumulateGrad object at 0x...>, 0))

dL/dx = 42.0
dL/dw = 28.0
dL/db = 14.0
```

---

## 1.7 Multi-Path Graphs (Fan-Out)

When a tensor is used in **multiple** operations, the graph **fans out**, and gradients from all paths are **summed**.

### Example: $L = x^2 + x^3$

```
       x
      / \
  [square] [cube]
      \     /
      [add]
        |
        L
```

Here $x$ feeds into two operations. During backward, gradients from both paths arrive at $x$ and are **added**:

$$\frac{\partial L}{\partial x} = \frac{\partial (x^2)}{\partial x} + \frac{\partial (x^3)}{\partial x} = 2x + 3x^2$$

This is **not** a special rule — it follows directly from the multivariable chain rule. If $L = f(u, v)$ where both $u$ and $v$ depend on $x$:

$$\frac{dL}{dx} = \frac{\partial L}{\partial u}\frac{du}{dx} + \frac{\partial L}{\partial v}\frac{dv}{dx}$$

```python
x = torch.tensor(2.0, requires_grad=True)
L = x**2 + x**3
L.backward()
print(f"dL/dx = {x.grad}")  # 2*2 + 3*4 = 4 + 12 = 16.0
```

---

## 1.8 Graph with Matrices (Toward Real Networks)

Real neural networks operate on matrices (tensors). The graph structure is identical — only the **local gradient computations** change.

### Example: Single linear layer

$$\mathbf{y} = \mathbf{W}\mathbf{x} + \mathbf{b}, \quad L = \|\mathbf{y}\|^2 = \sum_i y_i^2$$

Where $\mathbf{W} \in \mathbb{R}^{2 \times 3}$, $\mathbf{x} \in \mathbb{R}^{3}$, $\mathbf{b} \in \mathbb{R}^{2}$.

```
  W       x        b
   \     /         |
   [matmul]        |
       \          /
       [add]-----
         |
       [norm_sq]
         |
         L (scalar)
```

```python
import torch

W = torch.randn(2, 3, requires_grad=True)
x = torch.randn(3, requires_grad=True)
b = torch.randn(2, requires_grad=True)

# Forward
y = W @ x + b        # shape: (2,)
L = (y ** 2).sum()   # scalar loss

L.backward()

print(f"W.grad shape: {W.grad.shape}")  # (2, 3) — same as W
print(f"x.grad shape: {x.grad.shape}")  # (3,)   — same as x
print(f"b.grad shape: {b.grad.shape}")  # (2,)   — same as b
```

**Critical observation**: the gradient of a tensor **always has the same shape as the tensor itself**. This is a fundamental invariant:

$$\frac{\partial L}{\partial \mathbf{W}} \in \mathbb{R}^{2 \times 3}, \quad \frac{\partial L}{\partial \mathbf{x}} \in \mathbb{R}^{3}, \quad \frac{\partial L}{\partial \mathbf{b}} \in \mathbb{R}^{2}$$

Why? Because we need one gradient value per parameter to update it via $\theta \leftarrow \theta - \alpha \nabla_\theta L$.

---

## 1.9 `torch.no_grad()` and Detaching

Sometimes you want to **stop** the graph from being built (inference, evaluation):

```python
# Method 1: Context manager
with torch.no_grad():
    y = W @ x + b  # No graph built, no grad_fn

# Method 2: Detach a tensor from the graph
z = y.detach()  # z shares data with y but has no grad_fn
```

Use `torch.no_grad()` during inference to save memory (no intermediates stored).

Use `.detach()` when you want to use a value but stop gradient flow through it (e.g., target networks in RL).

---

## 1.10 Key Takeaways

| Concept | What to Remember |
|---------|-----------------|
| **Graph = DAG** | Nodes are ops, edges are tensors. Walk backward for gradients. |
| **Built during forward** | Every operation appends to the graph (in PyTorch, dynamically). |
| **`grad_fn` chain** | Each tensor points to the op that created it → linked list back to leaves. |
| **Fan-out → sum** | If a tensor feeds multiple ops, gradients from all paths are summed. |
| **Gradient shape = tensor shape** | Always. No exceptions. |
| **Destroyed after backward** | Graph is freed after `.backward()` (unless `retain_graph=True`). |

---

## Quick Reference

```
Forward:  input → [op1] → intermediate → [op2] → ... → loss

Backward: loss → [op2.backward] → intermediate.grad → [op1.backward] → input.grad
                 (dL/d_out * d_out/d_in)
```

**Next**: Document 2 covers how batches interact with this graph — does the framework build one graph per sample or one for the whole batch?
