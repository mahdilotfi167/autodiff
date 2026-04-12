# 6. Numerical Gradients via Finite Differences

## Motivation

You've derived a gradient formula by hand. How do you know it's correct? You can't just "trust the math" — sign errors, transposition mistakes, and index bugs are extremely common.

The answer: **numerical gradient checking**. Perturb each input by a tiny $\epsilon$, observe the output change, and compare against your analytical gradient. This is how PyTorch authors validate every new backward function, and how you should validate custom gradients.

---

## 6.1 The Finite Difference Idea

The derivative of $f$ at $x$ is defined as:

$$f'(x) = \lim_{\epsilon \to 0} \frac{f(x + \epsilon) - f(x)}{\epsilon}$$

For small finite $\epsilon$, we get **approximations**:

### Forward difference (one-sided)

$$f'(x) \approx \frac{f(x + \epsilon) - f(x)}{\epsilon}$$

Error: $O(\epsilon)$ — first-order accurate.

### Central difference (two-sided)

$$f'(x) \approx \frac{f(x + \epsilon) - f(x - \epsilon)}{2\epsilon}$$

Error: $O(\epsilon^2)$ — second-order accurate. **Always prefer this.**

### Why central difference is better

Taylor expand both:

$$f(x + \epsilon) = f(x) + \epsilon f'(x) + \frac{\epsilon^2}{2}f''(x) + \frac{\epsilon^3}{6}f'''(x) + \ldots$$

$$f(x - \epsilon) = f(x) - \epsilon f'(x) + \frac{\epsilon^2}{2}f''(x) - \frac{\epsilon^3}{6}f'''(x) + \ldots$$

**Forward difference:**
$$\frac{f(x+\epsilon) - f(x)}{\epsilon} = f'(x) + \frac{\epsilon}{2}f''(x) + O(\epsilon^2)$$

Error term: $\frac{\epsilon}{2}f''(x)$. This is $O(\epsilon)$.

**Central difference:**
$$\frac{f(x+\epsilon) - f(x-\epsilon)}{2\epsilon} = f'(x) + \frac{\epsilon^2}{6}f'''(x) + O(\epsilon^4)$$

The first-order error terms **cancel** (even powers survive, odd powers cancel). Error is $O(\epsilon^2)$.

---

## 6.2 Choosing $\epsilon$: The Precision Tradeoff

Two competing error sources:

1. **Truncation error**: from dropping higher-order Taylor terms. Decreases as $\epsilon \to 0$.
2. **Floating-point roundoff**: machine precision $\delta \approx 10^{-7}$ (float32) or $10^{-16}$ (float64). Creates error $\sim \delta/\epsilon$ which **increases** as $\epsilon \to 0$.

### Optimal $\epsilon$

For central difference with float64:

$$\text{Total error} \approx \frac{\delta}{\epsilon} + \epsilon^2$$

Minimizing: $\frac{d}{d\epsilon}\left(\frac{\delta}{\epsilon} + \epsilon^2\right) = -\frac{\delta}{\epsilon^2} + 2\epsilon = 0$

$$\epsilon^* = \left(\frac{\delta}{2}\right)^{1/3} \approx (5 \times 10^{-17})^{1/3} \approx 3.7 \times 10^{-6}$$

### Practical recommendation

| Precision | $\epsilon$ | Expected accuracy |
|-----------|-----------|-------------------|
| float64 | $10^{-5}$ to $10^{-7}$ | ~8-10 matching digits |
| float32 | $10^{-3}$ to $10^{-4}$ | ~3-4 matching digits |

**Always use float64 for gradient checking.** This is validation code, not training code.

---

## 6.3 Scalar Function Example

```python
import torch

def f(x):
    return x ** 3 + 2 * x ** 2 - x + 1

x = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
y = f(x)
y.backward()
analytical_grad = x.grad.item()

# Numerical gradient (central difference)
eps = 1e-6
numerical_grad = (f(torch.tensor(2.0 + eps)) - f(torch.tensor(2.0 - eps))) / (2 * eps)
numerical_grad = numerical_grad.item()

# True derivative: 3x^2 + 4x - 1 at x=2 = 12 + 8 - 1 = 19
print(f"Analytical:  {analytical_grad:.10f}")    # 19.0000000000
print(f"Numerical:   {numerical_grad:.10f}")     # 19.0000000000 (very close)
print(f"Abs error:   {abs(analytical_grad - numerical_grad):.2e}")  # ~1e-10
```

---

## 6.4 Extending to Vectors and Matrices

For $f: \mathbb{R}^n \to \mathbb{R}$ (scalar loss, multi-dimensional input), we compute partial derivatives one at a time:

$$\frac{\partial f}{\partial x_i} \approx \frac{f(\mathbf{x} + \epsilon \mathbf{e}_i) - f(\mathbf{x} - \epsilon \mathbf{e}_i)}{2\epsilon}$$

where $\mathbf{e}_i$ is the $i$-th standard basis vector (perturb only the $i$-th element).

This requires **$2n$ function evaluations** — one pair of $\pm\epsilon$ for each element.

### For matrices

Same idea, but flatten conceptually. For $\mathbf{W} \in \mathbb{R}^{m \times n}$, perturb each $W_{ij}$:

$$\frac{\partial L}{\partial W_{ij}} \approx \frac{L(\mathbf{W} + \epsilon \mathbf{E}_{ij}) - L(\mathbf{W} - \epsilon \mathbf{E}_{ij})}{2\epsilon}$$

where $\mathbf{E}_{ij}$ is the matrix with 1 at position $(i,j)$ and 0 elsewhere.

---

## 6.5 Implementation: General Numerical Gradient Checker

```python
import torch

def numerical_gradient(f, inputs, param_idx, eps=1e-6):
    """
    Compute numerical gradient of scalar function f w.r.t. inputs[param_idx].
    
    Args:
        f: callable that takes *inputs and returns a scalar tensor
        inputs: tuple of tensors
        param_idx: which input to differentiate w.r.t.
        eps: perturbation size
    
    Returns:
        Tensor of same shape as inputs[param_idx] containing numerical gradients
    """
    param = inputs[param_idx]
    grad = torch.zeros_like(param, dtype=torch.float64)
    
    # Flatten for iteration
    flat_param = param.data.view(-1)
    flat_grad = grad.view(-1)
    
    for i in range(flat_param.numel()):
        # Save original value
        orig = flat_param[i].item()
        
        # f(x + eps)
        flat_param[i] = orig + eps
        loss_plus = f(*inputs).item()
        
        # f(x - eps)
        flat_param[i] = orig - eps
        loss_minus = f(*inputs).item()
        
        # Central difference
        flat_grad[i] = (loss_plus - loss_minus) / (2 * eps)
        
        # Restore
        flat_param[i] = orig
    
    return grad


def gradient_check(f, inputs, param_idx, eps=1e-6, rtol=1e-4, atol=1e-6):
    """
    Compare analytical and numerical gradients.
    Returns True if they match within tolerance.
    """
    # Convert to float64 for numerical precision
    inputs_64 = tuple(
        x.detach().clone().double().requires_grad_(x.requires_grad)
        for x in inputs
    )
    
    # Analytical gradient
    loss = f(*inputs_64)
    loss.backward()
    analytical = inputs_64[param_idx].grad.clone()
    
    # Numerical gradient
    # Need fresh inputs (no grad history)
    inputs_fresh = tuple(
        x.detach().clone().double().requires_grad_(False)
        for x in inputs
    )
    numerical = numerical_gradient(f, inputs_fresh, param_idx, eps)
    
    # Compare
    max_diff = (analytical - numerical).abs().max().item()
    max_val = max(analytical.abs().max().item(), numerical.abs().max().item(), 1e-8)
    relative_error = max_diff / max_val
    
    print(f"Max absolute difference: {max_diff:.2e}")
    print(f"Max relative error:      {relative_error:.2e}")
    print(f"PASS: {relative_error < rtol}")
    
    return relative_error < rtol
```

---

## 6.6 Testing Our Known Gradients

### Test matmul gradient

```python
import torch

def matmul_loss(A, B):
    Y = A @ B
    return (Y ** 2).sum()

A = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
B = torch.randn(4, 2, dtype=torch.float64, requires_grad=True)

print("Checking gradient w.r.t. A:")
gradient_check(matmul_loss, (A, B), param_idx=0)
# Expected: PASS with relative error ~1e-10

print("\nChecking gradient w.r.t. B:")
gradient_check(matmul_loss, (A, B), param_idx=1)
# Expected: PASS with relative error ~1e-10
```

### Test ReLU gradient

```python
def relu_loss(x):
    return torch.relu(x).sum()

x = torch.randn(5, dtype=torch.float64, requires_grad=True)
# Avoid x ≈ 0 (ReLU is non-differentiable there)
x.data = x.data.clamp(min=0.1)

gradient_check(relu_loss, (x,), param_idx=0)
```

### Test softmax + cross-entropy gradient

```python
def ce_loss(logits):
    target = torch.tensor([2])
    return torch.nn.functional.cross_entropy(logits, target)

logits = torch.randn(1, 5, dtype=torch.float64, requires_grad=True)
gradient_check(ce_loss, (logits,), param_idx=0)
```

---

## 6.7 The `torch.autograd.gradcheck` Utility

PyTorch provides this built-in:

```python
import torch
from torch.autograd import gradcheck

def my_function(A, B):
    return A @ B

A = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
B = torch.randn(4, 2, dtype=torch.float64, requires_grad=True)

# Returns True if gradients match, raises error if not
result = gradcheck(my_function, (A, B), eps=1e-6, atol=1e-4, rtol=1e-3)
print(f"gradcheck passed: {result}")  # True
```

Under the hood, `gradcheck` does exactly what we implemented above: perturbs each element, computes central difference, compares with autograd.

### `gradgradcheck` for second-order gradients

```python
from torch.autograd import gradgradcheck
# Verifies that gradients of gradients are also correct
```

---

## 6.8 Common Pitfalls

### 1. Using float32

```python
# BAD: float32 has only ~7 digits of precision
x = torch.randn(3, requires_grad=True)  # float32 by default

# GOOD: always use float64 for gradient checking
x = torch.randn(3, dtype=torch.float64, requires_grad=True)
```

### 2. Non-differentiable points

Functions like ReLU, abs, max have kinks where the derivative is undefined. Numerical gradient at these points may not match analytical gradient. **Solution**: avoid testing at these points (add small offset).

### 3. Randomness in the function

If $f$ uses dropout, random sampling, etc., forward passes with $+\epsilon$ and $-\epsilon$ will give different random draws. **Solution**: set seed or disable stochastic components during checking.

### 4. $\epsilon$ too small or too large

- Too small ($< 10^{-10}$): roundoff dominates, garbage results
- Too large ($> 10^{-2}$): truncation dominates, inaccurate

### 5. Forgetting to check all inputs

A function $f(\mathbf{A}, \mathbf{B})$ needs gradient checked w.r.t. BOTH $\mathbf{A}$ and $\mathbf{B}$.

---

## 6.9 Relationship to Perturbation Analysis

Numerical gradient checking IS perturbation analysis:

> Perturb an input by a small amount → observe the change in output → ratio gives the derivative.

This is the physical intuition behind all of differential calculus. The finite difference method makes it computational.

In the next document, we'll see how this perturbation idea, combined with clever algebraic tricks involving the **matrix trace**, gives us an elegant framework for deriving gradients of matrix operations — the **trace trick**.

---

## 6.10 Summary

| Concept | Formula / Rule |
|---------|---------------|
| Central difference | $f'(x) \approx \frac{f(x+\epsilon) - f(x-\epsilon)}{2\epsilon}$, error $O(\epsilon^2)$ |
| Optimal $\epsilon$ (float64) | $\sim 10^{-5}$ to $10^{-7}$ |
| For matrices | Perturb each element independently, $2nm$ evaluations |
| Comparison metric | Relative error: $\frac{|\text{analytical} - \text{numerical}|}{\max(|\text{analytical}|, |\text{numerical}|)}$ |
| Accept threshold | Relative error $< 10^{-5}$ (float64), $< 10^{-3}$ (float32) |
| PyTorch built-in | `torch.autograd.gradcheck(fn, inputs)` |
| **Always use float64** | Non-negotiable for gradient checking |

---

**Next**: Document 7 derives the **trace trick** — an elegant mathematical framework for computing matrix gradients using differentials and trace properties, avoiding the tedious element-by-element derivation.
