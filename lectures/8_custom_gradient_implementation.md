# 8. Custom Gradient Implementation in PyTorch

## Motivation

You now know how to derive gradient formulas (Documents 3-5, 7) and validate them numerically (Document 6). The final step: **implementing** a custom operation with hand-written forward and backward passes in PyTorch.

This is what framework authors do for every operation in the library. It is also what YOU do when:
- You invent a new operation not in PyTorch
- You need a fused/optimized kernel
- You want to understand exactly what happens inside `.backward()`
- You need to save memory by not storing certain intermediates

---

## 8.1 The `torch.autograd.Function` API

Every custom differentiable operation inherits from `torch.autograd.Function`. You must implement two static methods:

```python
class MyOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *inputs):
        """
        Compute the forward pass.
        - ctx: context object to save tensors for backward
        - inputs: input tensors
        Returns: output tensor(s)
        """
        # Save what backward needs
        ctx.save_for_backward(...)
        return output
    
    @staticmethod
    def backward(ctx, *grad_outputs):
        """
        Compute the VJP (backward pass).
        - ctx: context from forward
        - grad_outputs: upstream gradients (dL/d_output for each output)
        Returns: tuple of gradients, one per input to forward
        """
        # Retrieve saved tensors
        ... = ctx.saved_tensors
        return grad_input1, grad_input2, ...
```

Call it via: `output = MyOp.apply(input1, input2, ...)`

---

## 8.2 Example 1: Custom Matrix Multiply

Let's implement matmul from scratch, with hand-derived gradients from Document 4.

### The math (review)

Forward: $\mathbf{Y} = \mathbf{A}\mathbf{B}$

Backward (from Document 4):
- $\frac{\partial L}{\partial \mathbf{A}} = \bar{\mathbf{Y}} \mathbf{B}^T$
- $\frac{\partial L}{\partial \mathbf{B}} = \mathbf{A}^T \bar{\mathbf{Y}}$

### Implementation

```python
import torch
from torch.autograd import Function, gradcheck

class CustomMatmul(Function):
    @staticmethod
    def forward(ctx, A, B):
        # Save inputs needed for backward
        ctx.save_for_backward(A, B)
        return A @ B
    
    @staticmethod
    def backward(ctx, grad_output):
        # grad_output is dL/dY (same shape as Y)
        A, B = ctx.saved_tensors
        
        # Apply our derived formulas
        grad_A = grad_output @ B.T       # dL/dA = dL/dY @ B^T
        grad_B = A.T @ grad_output       # dL/dB = A^T @ dL/dY
        
        return grad_A, grad_B

# Test
A = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
B = torch.randn(4, 2, dtype=torch.float64, requires_grad=True)

# Use our custom op
Y = CustomMatmul.apply(A, B)
L = (Y ** 2).sum()
L.backward()

print(f"A.grad shape: {A.grad.shape}")  # (3, 4)
print(f"B.grad shape: {B.grad.shape}")  # (4, 2)

# Validate with gradcheck
print("\nGradient check:")
result = gradcheck(CustomMatmul.apply, (A, B), eps=1e-6, atol=1e-4)
print(f"  Passed: {result}")  # True
```

---

## 8.3 Example 2: Custom ReLU

### The math

Forward: $y_i = \max(0, x_i)$

Backward: $\bar{x}_i = \bar{y}_i \cdot \mathbb{1}[x_i > 0]$

### Implementation

```python
class CustomReLU(Function):
    @staticmethod
    def forward(ctx, x):
        # Save what backward needs: a mask of positive elements
        ctx.save_for_backward(x)
        return x.clamp(min=0)
    
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        grad_input = grad_output * (x > 0).to(grad_output.dtype)
        return grad_input

# Test
x = torch.randn(10, dtype=torch.float64, requires_grad=True)
# Avoid values near zero for gradient check (non-differentiable point)
x.data = x.data + 0.5 * x.data.sign()

y = CustomReLU.apply(x)
L = y.sum()
L.backward()

print(f"x.grad: {x.grad}")

# Validate
result = gradcheck(CustomReLU.apply, (x,), eps=1e-6)
print(f"Gradient check passed: {result}")
```

### Memory optimization note

We saved the entire input tensor `x`. But we only need the **mask** `x > 0` (a boolean tensor, 8× smaller than float64). A more efficient implementation:

```python
class EfficientReLU(Function):
    @staticmethod
    def forward(ctx, x):
        mask = x > 0
        ctx.save_for_backward(mask)  # Save boolean mask instead of full tensor
        return x.clamp(min=0)
    
    @staticmethod
    def backward(ctx, grad_output):
        mask, = ctx.saved_tensors
        return grad_output * mask
```

This is why custom implementations can be more memory-efficient than autograd's default decomposition.

---

## 8.4 Example 3: Custom Sigmoid (with Efficient Backward)

### The math

Forward: $\sigma(x) = \frac{1}{1 + e^{-x}}$

Backward: $\bar{x} = \bar{y} \cdot \sigma(x)(1 - \sigma(x))$

Key insight: the backward only needs the **output** $\sigma(x)$, not the input $x$.

### Implementation

```python
class CustomSigmoid(Function):
    @staticmethod
    def forward(ctx, x):
        output = torch.sigmoid(x)
        # Save OUTPUT, not input — more efficient for backward
        ctx.save_for_backward(output)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        output, = ctx.saved_tensors
        # sigma'(x) = sigma(x) * (1 - sigma(x))
        grad_input = grad_output * output * (1 - output)
        return grad_input

# Test
x = torch.randn(5, dtype=torch.float64, requires_grad=True)
result = gradcheck(CustomSigmoid.apply, (x,))
print(f"Gradient check passed: {result}")
```

---

## 8.5 Example 4: Linear Layer (Matmul + Bias + Batching)

Now let's combine everything: a full linear layer with batch support.

### The math

Forward: $\mathbf{Y} = \mathbf{X}\mathbf{W} + \mathbf{b}$

Where $\mathbf{X} \in \mathbb{R}^{B \times D_{in}}$, $\mathbf{W} \in \mathbb{R}^{D_{in} \times D_{out}}$, $\mathbf{b} \in \mathbb{R}^{D_{out}}$ (broadcast).

Backward:
- $\frac{\partial L}{\partial \mathbf{X}} = \bar{\mathbf{Y}} \mathbf{W}^T$
- $\frac{\partial L}{\partial \mathbf{W}} = \mathbf{X}^T \bar{\mathbf{Y}}$
- $\frac{\partial L}{\partial \mathbf{b}} = \sum_{i=1}^{B} \bar{Y}_{i,:}$ (sum over batch, undo broadcast)

### Implementation

```python
class CustomLinear(Function):
    @staticmethod
    def forward(ctx, X, W, b):
        ctx.save_for_backward(X, W)
        return X @ W + b  # b is broadcast from (D_out,) to (B, D_out)
    
    @staticmethod
    def backward(ctx, grad_output):
        X, W = ctx.saved_tensors
        
        grad_X = grad_output @ W.T                # (B, D_in)
        grad_W = X.T @ grad_output                # (D_in, D_out)
        grad_b = grad_output.sum(dim=0)           # (D_out,) — sum over batch
        
        return grad_X, grad_W, grad_b

# Test
B, D_in, D_out = 4, 3, 2
X = torch.randn(B, D_in, dtype=torch.float64, requires_grad=True)
W = torch.randn(D_in, D_out, dtype=torch.float64, requires_grad=True)
b = torch.randn(D_out, dtype=torch.float64, requires_grad=True)

# Forward + backward
Y = CustomLinear.apply(X, W, b)
L = (Y ** 2).sum()
L.backward()

print(f"X.grad shape: {X.grad.shape}")  # (4, 3)
print(f"W.grad shape: {W.grad.shape}")  # (3, 2)
print(f"b.grad shape: {b.grad.shape}")  # (2,)

# Validate
result = gradcheck(CustomLinear.apply, (X, W, b))
print(f"Gradient check passed: {result}")
```

---

## 8.6 Example 5: Custom Cross-Entropy (Numerically Stable)

This is a more complex example showing how fused operations have simpler gradients.

### The math (from Document 5)

Forward: $L = -x_t + \log\left(\sum_j e^{x_j}\right)$ where $t$ is the target class.

Backward: $\frac{\partial L}{\partial x_i} = s_i - \mathbb{1}[i = t]$ where $s_i = \text{softmax}(x)_i$.

### Implementation

```python
class CustomCrossEntropy(Function):
    @staticmethod
    def forward(ctx, logits, target):
        """
        logits: (B, C) — raw logits for B samples, C classes
        target: (B,) — integer class labels
        """
        B, C = logits.shape
        
        # Numerically stable softmax
        x_max = logits.max(dim=1, keepdim=True).values
        log_sum_exp = (logits - x_max).exp().sum(dim=1, keepdim=True).log() + x_max
        
        # Log-softmax
        log_probs = logits - log_sum_exp  # (B, C)
        
        # NLL loss: pick log_prob of correct class, negate, average
        loss = -log_probs[torch.arange(B), target].mean()
        
        # Save for backward
        softmax_probs = log_probs.exp()
        ctx.save_for_backward(softmax_probs)
        ctx.target = target
        ctx.B = B
        
        return loss
    
    @staticmethod
    def backward(ctx, grad_output):
        softmax_probs, = ctx.saved_tensors
        target = ctx.target
        B = ctx.B
        
        # Gradient: softmax_probs - one_hot(target), averaged over batch
        grad_logits = softmax_probs.clone()
        grad_logits[torch.arange(B), target] -= 1
        grad_logits = grad_logits / B  # mean reduction
        grad_logits = grad_logits * grad_output  # chain rule with upstream
        
        return grad_logits, None  # None for target (not differentiable)

# Test
B, C = 4, 5
logits = torch.randn(B, C, dtype=torch.float64, requires_grad=True)
target = torch.randint(C, (B,))

# Our custom implementation
loss_custom = CustomCrossEntropy.apply(logits, target)
print(f"Custom loss: {loss_custom.item():.6f}")

# PyTorch's implementation
loss_pytorch = torch.nn.functional.cross_entropy(logits, target)
print(f"PyTorch loss: {loss_pytorch.item():.6f}")
print(f"Match: {torch.allclose(loss_custom, loss_pytorch)}")

# Gradient check (only check logits, not target)
result = gradcheck(
    lambda x: CustomCrossEntropy.apply(x, target),
    (logits,),
    eps=1e-6
)
print(f"Gradient check passed: {result}")
```

---

## 8.7 Example 6: Custom Bilinear Layer (Trace Trick Derivation)

A more complex operation to show the complete derivation-to-implementation pipeline.

### Forward definition

$$y = \mathbf{x}_1^T \mathbf{W} \mathbf{x}_2$$

Where $\mathbf{x}_1 \in \mathbb{R}^m$, $\mathbf{W} \in \mathbb{R}^{m \times n}$, $\mathbf{x}_2 \in \mathbb{R}^n$, $y \in \mathbb{R}$ (scalar).

### Gradient derivation using trace trick

$$dy = d(\mathbf{x}_1^T \mathbf{W} \mathbf{x}_2)$$

Product rule (3 terms, but we'll do one variable at a time):

**w.r.t. $\mathbf{W}$**: hold $\mathbf{x}_1, \mathbf{x}_2$ constant:

$$dy = \mathbf{x}_1^T (d\mathbf{W}) \mathbf{x}_2$$

This is a scalar, so $dy = \text{tr}(dy) = \text{tr}(\mathbf{x}_1^T (d\mathbf{W}) \mathbf{x}_2)$.

Cyclic: $= \text{tr}(\mathbf{x}_2 \mathbf{x}_1^T (d\mathbf{W}))$

Now $\mathbf{x}_2 \mathbf{x}_1^T \in \mathbb{R}^{n \times m}$, and we need $\text{tr}(\mathbf{G}^T d\mathbf{W})$ where $\mathbf{G} \in \mathbb{R}^{m \times n}$.

$$\text{tr}(\mathbf{x}_2 \mathbf{x}_1^T d\mathbf{W}) = \text{tr}((\mathbf{x}_1 \mathbf{x}_2^T)^T d\mathbf{W})$$

So: $\frac{\partial y}{\partial \mathbf{W}} = \mathbf{x}_1 \mathbf{x}_2^T$ (outer product).

With upstream scalar $\bar{y}$: $\frac{\partial L}{\partial \mathbf{W}} = \bar{y} \cdot \mathbf{x}_1 \mathbf{x}_2^T$

**w.r.t. $\mathbf{x}_1$**: $dy = (d\mathbf{x}_1)^T \mathbf{W} \mathbf{x}_2 = (\mathbf{W}\mathbf{x}_2)^T d\mathbf{x}_1$

So: $\frac{\partial y}{\partial \mathbf{x}_1} = \mathbf{W}\mathbf{x}_2$

With upstream: $\frac{\partial L}{\partial \mathbf{x}_1} = \bar{y} \cdot \mathbf{W}\mathbf{x}_2$

**w.r.t. $\mathbf{x}_2$**: $dy = \mathbf{x}_1^T \mathbf{W} (d\mathbf{x}_2) = (\mathbf{W}^T\mathbf{x}_1)^T d\mathbf{x}_2$

So: $\frac{\partial y}{\partial \mathbf{x}_2} = \mathbf{W}^T \mathbf{x}_1$

With upstream: $\frac{\partial L}{\partial \mathbf{x}_2} = \bar{y} \cdot \mathbf{W}^T \mathbf{x}_1$

### Implementation

```python
class CustomBilinear(Function):
    @staticmethod
    def forward(ctx, x1, W, x2):
        ctx.save_for_backward(x1, W, x2)
        # y = x1^T @ W @ x2 (scalar)
        return x1 @ W @ x2
    
    @staticmethod
    def backward(ctx, grad_output):
        x1, W, x2 = ctx.saved_tensors
        
        # grad_output is dL/dy (scalar)
        grad_x1 = grad_output * (W @ x2)           # (m,)
        grad_W  = grad_output * torch.outer(x1, x2) # (m, n)
        grad_x2 = grad_output * (W.T @ x1)          # (n,)
        
        return grad_x1, grad_W, grad_x2

# Test
m, n = 3, 4
x1 = torch.randn(m, dtype=torch.float64, requires_grad=True)
W = torch.randn(m, n, dtype=torch.float64, requires_grad=True)
x2 = torch.randn(n, dtype=torch.float64, requires_grad=True)

result = gradcheck(CustomBilinear.apply, (x1, W, x2))
print(f"Bilinear gradient check passed: {result}")
```

---

## 8.8 Rules and Gotchas

### 1. Return `None` for non-differentiable inputs

If an input is an integer, boolean, or otherwise not a floating-point tensor:

```python
@staticmethod
def backward(ctx, grad_output):
    return grad_tensor, None  # None for non-differentiable input
```

The number of return values must **exactly match** the number of inputs to `forward` (excluding `ctx`).

### 2. Use `ctx.save_for_backward()` (not attributes)

```python
# GOOD: PyTorch manages memory correctly
ctx.save_for_backward(A, B)

# BAD: memory leak, no safety checks
ctx.A = A  # only use for non-tensor metadata
```

`save_for_backward` only accepts tensors. For non-tensor data (integers, shapes), use attributes: `ctx.shape = x.shape`.

### 3. Don't modify saved tensors

Saved tensors must not be modified in-place after saving. PyTorch will detect this and raise an error (version counter check).

### 4. Handle the case where gradient is not needed

```python
@staticmethod
def backward(ctx, grad_output):
    A, B = ctx.saved_tensors
    
    # Only compute what's needed
    grad_A = grad_output @ B.T if ctx.needs_input_grad[0] else None
    grad_B = A.T @ grad_output if ctx.needs_input_grad[1] else None
    
    return grad_A, grad_B
```

`ctx.needs_input_grad` is a tuple of booleans indicating which inputs need gradients.

### 5. Use `.apply()`, not direct instantiation

```python
# CORRECT
output = MyOp.apply(input1, input2)

# WRONG
output = MyOp()(input1, input2)  # Don't do this
```

---

## 8.9 Complete End-to-End: Custom Two-Layer MLP

Combining all custom operations into a trainable network:

```python
import torch
from torch.autograd import Function, gradcheck

class LinearReLU(Function):
    """Fused Linear + ReLU for efficiency."""
    
    @staticmethod
    def forward(ctx, X, W, b):
        Z = X @ W + b
        mask = Z > 0
        Y = Z * mask
        ctx.save_for_backward(X, W, mask)
        return Y
    
    @staticmethod
    def backward(ctx, grad_output):
        X, W, mask = ctx.saved_tensors
        
        # ReLU backward: mask the gradient
        grad_Z = grad_output * mask
        
        # Linear backward
        grad_X = grad_Z @ W.T
        grad_W = X.T @ grad_Z
        grad_b = grad_Z.sum(dim=0)
        
        return grad_X, grad_W, grad_b

# Build a 2-layer MLP using custom ops
torch.manual_seed(42)
B, D_in, D_hidden, D_out = 8, 4, 6, 3

X = torch.randn(B, D_in, dtype=torch.float64)
targets = torch.randn(B, D_out, dtype=torch.float64)

W1 = torch.randn(D_in, D_hidden, dtype=torch.float64, requires_grad=True)
b1 = torch.randn(D_hidden, dtype=torch.float64, requires_grad=True)
W2 = torch.randn(D_hidden, D_out, dtype=torch.float64, requires_grad=True)
b2 = torch.randn(D_out, dtype=torch.float64, requires_grad=True)

# Forward pass using custom ops
H = LinearReLU.apply(X, W1, b1)       # Hidden layer
Y = H @ W2 + b2                        # Output layer (using standard ops)
loss = ((Y - targets) ** 2).mean()

print(f"Loss: {loss.item():.4f}")

# Backward pass
loss.backward()

print(f"W1.grad shape: {W1.grad.shape}")  # (4, 6)
print(f"b1.grad shape: {b1.grad.shape}")  # (6,)
print(f"W2.grad shape: {W2.grad.shape}")  # (6, 3)
print(f"b2.grad shape: {b2.grad.shape}")  # (3,)

# Gradient check for the fused LinearReLU
print("\nGradient check for LinearReLU:")
# Avoid inputs near zero (ReLU non-differentiable)
X_test = torch.randn(2, 4, dtype=torch.float64, requires_grad=True)
W_test = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
b_test = torch.randn(3, dtype=torch.float64, requires_grad=True)

# Shift to avoid values near zero after linear transform
result = gradcheck(
    LinearReLU.apply,
    (X_test, W_test, b_test),
    eps=1e-6,
    atol=1e-4,
    nondet_tol=1e-5
)
print(f"  Passed: {result}")

# Training loop
print("\nTraining for 100 steps:")
lr = 0.01
params = [W1, W2, b1, b2]

for step in range(100):
    # Zero grads
    for p in params:
        if p.grad is not None:
            p.grad.zero_()
    
    # Forward
    H = LinearReLU.apply(X, W1, b1)
    Y = H @ W2 + b2
    loss = ((Y - targets) ** 2).mean()
    
    # Backward
    loss.backward()
    
    # SGD update
    with torch.no_grad():
        for p in params:
            p -= lr * p.grad
    
    if step % 20 == 0:
        print(f"  Step {step:3d}: loss = {loss.item():.4f}")
```

---

## 8.10 How PyTorch Registers Built-in Operations

For reference, here's how PyTorch's own operations are structured:

1. **Forward**: Implemented in C++ (ATen library), registered via `TORCH_LIBRARY`
2. **Backward**: Defined in `derivatives.yaml` — a YAML file that specifies the VJP formula for each operation
3. **Code generation**: `tools/autograd/gen_autograd.py` reads `derivatives.yaml` and generates C++ backward functions
4. **Example from `derivatives.yaml`**:

```yaml
- name: mm(Tensor self, Tensor mat2) -> Tensor
  self: grad.mm(mat2.t())
  mat2: self.t().mm(grad)
```

This says: for `mm` (matrix multiply):
- Gradient w.r.t. `self` (left matrix): `grad @ mat2.T`
- Gradient w.r.t. `mat2` (right matrix): `self.T @ grad`

Exactly our formulas from Document 4!

### Where to find it

In the PyTorch source code: `torch/csrc/autograd/FunctionsManual.cpp` and `tools/autograd/derivatives.yaml`.

---

## 8.11 Summary: The Complete Pipeline

```
1. DEFINE the forward operation
        ↓
2. DERIVE the gradient formula
   (Element-wise method OR Trace trick, Document 4/7)
        ↓
3. IMPLEMENT as torch.autograd.Function
   - forward(): compute output, save tensors
   - backward(): compute VJP using derived formula
        ↓
4. VALIDATE with gradcheck (Document 6)
   - torch.autograd.gradcheck(fn, inputs)
   - Uses central finite differences internally
        ↓
5. OPTIMIZE (optional)
   - Fuse operations to reduce memory
   - Save only what backward needs (e.g., mask vs full tensor)
   - Write CUDA kernel for GPU efficiency
        ↓
6. USE in your model
   - output = MyOp.apply(input1, input2)
   - Integrates seamlessly with all PyTorch autograd
```

| Step | Skills Used | Document Reference |
|------|-------------|-------------------|
| Define forward | Domain knowledge | — |
| Derive gradient | Matrix calculus, trace trick | Documents 3, 4, 5, 7 |
| Implement | PyTorch `Function` API | This document |
| Validate | Numerical gradient checking | Document 6 |
| Understand graph | Computation graph structure | Documents 1, 2 |

---

## What You Can Now Do

After studying all 8 documents, you can:

1. **Draw** the computation graph for any neural network architecture
2. **Predict** gradient shapes and understand batch dimension behavior
3. **Derive** the gradient formula for any differentiable operation using:
   - Element-wise Jacobian computation
   - The trace trick (matrix differential calculus)
4. **Implement** the forward and backward passes in PyTorch
5. **Validate** your implementation using numerical gradient checking
6. **Understand** why backpropagation is efficient (reverse-mode AD, VJP chaining)
7. **Read** PyTorch source code (`derivatives.yaml`) and understand what each backward formula means
8. **Debug** gradient issues: shape mismatches, NaN gradients, incorrect accumulation
