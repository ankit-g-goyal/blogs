# Blog 5: Deep Learning Frameworks — PyTorch
## From Scratch to Framework

**Reading time:** 60–75 minutes
**Coding time:** 90–120 minutes
**Total investment:** 2.5–3 hours

---

### 🧭 Reading Guide

**Prerequisites**: Blog 4 (neural networks from scratch — backpropagation, training loops), Python fluency. You should understand forward/backward passes and gradient descent before starting.

| Your Background | Recommended Path |
|----------------|-----------------|
| Completed Blog 4 | Read all parts in order — see how framework automates what you built |
| Know PyTorch basics | Skim Parts 1-3, focus on Parts 4-6 (training pipeline, patterns, debugging) |
| Evaluating frameworks | Read "Why PyTorch" section and Manager's Summary |

**Dependencies**: `pip install torch torchvision matplotlib scikit-learn`

---

## What You'll Walk Away With

By the end of this blog, you will:

1. **Understand** tensors and how they relate to NumPy arrays
2. **Implement** automatic differentiation with `autograd`
3. **Build** neural networks using `nn.Module`
4. **Train** models with proper training loops, validation, and checkpointing
5. **Compare** PyTorch vs TensorFlow to make informed framework choices
6. **Debug** common PyTorch issues: device mismatches, gradient problems, memory leaks

After building everything from scratch in Blog 4, you'll now see how PyTorch automates the tedious parts while giving you control where it matters.

---

## Why PyTorch?

You could continue building everything from scratch, but:

| Task | From Scratch | PyTorch |
|------|-------------|---------|
| Backpropagation | Derive and implement for each layer | `loss.backward()` |
| GPU support | Write CUDA kernels | `model.to('cuda')` |
| Common layers | Implement each one | `nn.Linear`, `nn.Conv2d`, etc. |
| Optimizers | Implement SGD, Adam, etc. | `torch.optim.Adam` |
| Data loading | Write batching, shuffling | `DataLoader` |

PyTorch handles the mechanics so you can focus on architecture and experimentation.

### PyTorch vs TensorFlow

| Factor | PyTorch | TensorFlow |
|--------|---------|------------|
| Debugging | Eager execution, Python debugger works | Historically graph-based, now eager too |
| Research | Dominant in academia, papers | Strong enterprise adoption |
| Deployment | TorchScript, ONNX | TensorFlow Serving, TFLite |
| Learning curve | More Pythonic, intuitive | More boilerplate |
| Industry | Meta, OpenAI, Anthropic | Google, many enterprises |

**Recommendation**: Learn PyTorch first. Converting to TensorFlow later is straightforward; the concepts transfer.

---

## Part 1: Tensors — NumPy with Superpowers

### Creating Tensors

```python
import torch
import numpy as np

# From Python lists
t1 = torch.tensor([1, 2, 3, 4])
print(f"From list: {t1}, dtype: {t1.dtype}")

# From NumPy (shares memory by default!)
arr = np.array([1.0, 2.0, 3.0])
t2 = torch.from_numpy(arr)
print(f"From NumPy: {t2}, dtype: {t2.dtype}")

# Verify shared memory
arr[0] = 99
print(f"After modifying NumPy: {t2}")  # Also changed!

# To avoid shared memory, use .clone()
t3 = torch.from_numpy(arr).clone()
arr[0] = 1
print(f"Cloned tensor unchanged: {t3}")

# Common initialization
zeros = torch.zeros(3, 4)        # 3x4 matrix of zeros
ones = torch.ones(2, 3)          # 2x3 matrix of ones
rand_uniform = torch.rand(3, 3)  # Uniform [0, 1)
rand_normal = torch.randn(3, 3)  # Standard normal
eye = torch.eye(4)               # Identity matrix

# Specify dtype
float_tensor = torch.tensor([1, 2, 3], dtype=torch.float32)
int_tensor = torch.tensor([1.5, 2.5], dtype=torch.int64)

print(f"\nShapes:")
print(f"zeros: {zeros.shape}")
print(f"rand_normal: {rand_normal.shape}")
```

### Tensor Operations

```python
import torch

# Arithmetic
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

print(f"a + b = {a + b}")
print(f"a * b = {a * b}")  # Element-wise
print(f"a @ b = {a @ b}")  # Dot product (scalar)

# Matrix operations
A = torch.randn(3, 4)
B = torch.randn(4, 5)

C = A @ B  # Matrix multiplication
print(f"A @ B shape: {C.shape}")  # (3, 5)

# Transpose
print(f"A.T shape: {A.T.shape}")  # (4, 3)

# Reshaping
x = torch.arange(12)  # [0, 1, ..., 11]
print(f"Original: {x.shape}")

reshaped = x.reshape(3, 4)
print(f"Reshaped: {reshaped.shape}")

viewed = x.view(2, 6)  # view shares memory, reshape may copy
print(f"Viewed: {viewed.shape}")

# Squeeze and unsqueeze
t = torch.randn(1, 3, 1, 4)
print(f"Before squeeze: {t.shape}")
print(f"After squeeze: {t.squeeze().shape}")  # Remove size-1 dims

t = torch.randn(3, 4)
print(f"Before unsqueeze: {t.shape}")
print(f"After unsqueeze(0): {t.unsqueeze(0).shape}")  # Add batch dim
```

### Device Management (CPU/GPU)

```python
import torch

# Check GPU availability
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Device selection
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create tensor on specific device
t_cpu = torch.randn(3, 3)
t_gpu = torch.randn(3, 3, device=device)

print(f"CPU tensor device: {t_cpu.device}")
print(f"GPU tensor device: {t_gpu.device}")

# Move tensor between devices
t_moved = t_cpu.to(device)
print(f"Moved tensor device: {t_moved.device}")

# Common error: mixing devices
try:
    result = t_cpu + t_gpu  # Fails if GPU tensor!
except RuntimeError as e:
    print(f"Device mismatch error: {e}")

# Solution: ensure same device
result = t_cpu.to(device) + t_gpu  # Works
print(f"Same device operation: {result.device}")
```

> **✅ Checkpoint**: At this point you should be able to: (1) create tensors from NumPy arrays and explain the shared-memory pitfall, (2) perform matrix multiplication with `@` and predict output shapes, (3) move tensors between CPU and GPU with `.to(device)`, (4) explain why mixing devices causes RuntimeError.

---

## Part 2: Automatic Differentiation (Autograd)

This is PyTorch's killer feature: automatic gradient computation.

### How Autograd Works

```python
import torch

# Create tensor with gradient tracking
x = torch.tensor([2.0, 3.0], requires_grad=True)
print(f"x requires grad: {x.requires_grad}")

# Forward pass (builds computation graph)
y = x ** 2
z = y.sum()

print(f"y = x^2 = {y}")
print(f"z = sum(y) = {z}")

# Backward pass (computes gradients)
z.backward()

# Gradients: dz/dx = d(x1^2 + x2^2)/dx = [2*x1, 2*x2] = [4, 6]
print(f"x.grad (dz/dx) = {x.grad}")

# The computation graph
print(f"\nComputation graph:")
print(f"z.grad_fn: {z.grad_fn}")
print(f"y.grad_fn: {y.grad_fn}")
```

**What the computation graph looks like:**
```
# For z = sum(x^2) where x = [2.0, 3.0]:
#
#   x [leaf tensor, requires_grad=True]
#   │
#   ▼
#   PowBackward0  (y = x^2)
#   │
#   ▼
#   SumBackward0  (z = sum(y))
#   │
#   ▼
#   z.backward()  → walks graph in reverse, computes dz/dx = 2x = [4, 6]
#
# Key: Each operation node stores references to its inputs and knows how to
# compute its local gradient. backward() applies the chain rule through
# these nodes. This is a DAG (directed acyclic graph) — no cycles allowed,
# which is why in-place operations on graph tensors are forbidden.
#
# TensorFlow 1.x used STATIC graphs (define-then-run): build graph first,
# then execute. PyTorch uses DYNAMIC graphs (define-by-run): graph is built
# during forward pass and destroyed after backward(). This makes debugging
# easier (print statements work, Python control flow works) but means the
# graph is rebuilt every iteration.
```

### Gradient Accumulation (Autograd)

```python
import torch

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

# First backward
y1 = (x ** 2).sum()
y1.backward()
print(f"After first backward: x.grad = {x.grad}")

# Second backward - gradients ACCUMULATE!
y2 = (x ** 3).sum()
y2.backward()
print(f"After second backward (accumulated): x.grad = {x.grad}")

# This is usually not what you want. Zero gradients first:
x.grad.zero_()
y3 = (x ** 2).sum()
y3.backward()
print(f"After zeroing and new backward: x.grad = {x.grad}")
```

### Neural Network Gradients

```python
import torch
import torch.nn as nn

# Simple network
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 2)

    def forward(self, x):
        return self.linear(x)

net = SimpleNet()
x = torch.randn(4, 3)  # Batch of 4, 3 features each
y_true = torch.randn(4, 2)

# Forward pass
y_pred = net(x)
loss = ((y_pred - y_true) ** 2).mean()

print(f"Loss: {loss.item():.4f}")
print(f"Weight grad before backward: {net.linear.weight.grad}")

# Backward pass
loss.backward()

print(f"Weight grad after backward: {net.linear.weight.grad}")
print(f"Bias grad: {net.linear.bias.grad}")
```

### Disabling Gradient Tracking

```python
import torch

x = torch.randn(3, requires_grad=True)

# Method 1: torch.no_grad() context
with torch.no_grad():
    y = x * 2
    print(f"In no_grad: y.requires_grad = {y.requires_grad}")

# Method 2: .detach()
y = x.detach() * 2
print(f"With detach: y.requires_grad = {y.requires_grad}")

# Method 3: For evaluation mode
model = torch.nn.Linear(3, 2)
model.eval()  # Sets evaluation mode (affects dropout, batchnorm)

with torch.no_grad():  # Still need this to save memory/computation
    output = model(x)

# When to use:
# - Inference: torch.no_grad() saves memory and computation
# - Validation: model.eval() + torch.no_grad()
# - Freezing layers: param.requires_grad = False
```

> **✅ Checkpoint**: At this point you should be able to: (1) explain what `requires_grad=True` does and draw the DAG for `z = sum(x^2)`, (2) explain why gradients accumulate and why `zero_grad()` is needed, (3) list three ways to disable gradient tracking and when to use each, (4) explain the difference between PyTorch's dynamic graph and TensorFlow 1.x's static graph.

---

## Part 3: Building Neural Networks with nn.Module

### The nn.Module Pattern

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MNISTClassifier(nn.Module):
    """
    A proper PyTorch neural network.

    Key rules:
    1. Inherit from nn.Module
    2. Define layers in __init__
    3. Define forward pass in forward()
    4. Never define backward() - autograd handles it
    """

    def __init__(self, input_dim=784, hidden_dims=None, num_classes=10, dropout=0.2):
        super().__init__()  # Always call parent __init__
        if hidden_dims is None:
            hidden_dims = [256, 128]  # Avoid mutable default argument

        # Build layers dynamically
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_classes))

        # nn.Sequential wraps multiple layers
        self.network = nn.Sequential(*layers)

        # Track architecture for debugging
        self.architecture = f"{input_dim} → {' → '.join(map(str, hidden_dims))} → {num_classes}"

    def forward(self, x):
        """
        Forward pass. Input shape: (batch_size, 784)
        """
        # Flatten if needed (e.g., from image shape)
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        return self.network(x)

    def predict(self, x):
        """Get class predictions."""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.argmax(logits, dim=1)

    def __repr__(self):
        return f"MNISTClassifier({self.architecture})"


# Create model
model = MNISTClassifier(hidden_dims=[512, 256, 128])
print(model)

# Inspect parameters
print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# Forward pass
x = torch.randn(32, 784)  # Batch of 32 images
output = model(x)
print(f"\nOutput shape: {output.shape}")
```

### Common Layers

```python
import torch
import torch.nn as nn

# Linear (fully connected)
linear = nn.Linear(in_features=784, out_features=256)
print(f"Linear weight shape: {linear.weight.shape}")

# Convolutional
conv = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
print(f"Conv weight shape: {conv.weight.shape}")

# BatchNorm (stabilizes training)
bn = nn.BatchNorm1d(num_features=256)

# Dropout (regularization)
dropout = nn.Dropout(p=0.5)

# Embedding (for NLP)
embedding = nn.Embedding(num_embeddings=10000, embedding_dim=256)
print(f"Embedding weight shape: {embedding.weight.shape}")

# LSTM (sequence modeling)
lstm = nn.LSTM(input_size=256, hidden_size=512, num_layers=2, batch_first=True)

# Transformer components
attention = nn.MultiheadAttention(embed_dim=256, num_heads=8)

# Example forward passes
x_linear = torch.randn(32, 784)
x_conv = torch.randn(32, 1, 28, 28)  # (batch, channels, height, width)
x_embed = torch.randint(0, 10000, (32, 100))  # (batch, seq_len)

print(f"\nLinear output: {linear(x_linear).shape}")
print(f"Conv output: {conv(x_conv).shape}")
print(f"Embedding output: {embedding(x_embed).shape}")
```

### Saving and Loading Models

```python
import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 5)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

model = SimpleModel()

# Method 1: Save entire model (not recommended)
# torch.save(model, 'model_full.pt')
# loaded = torch.load('model_full.pt')  # Requires original class definition

# Method 2: Save state dict (recommended)
torch.save(model.state_dict(), 'model_state.pt')

# Load state dict
# ⚠️ SECURITY: Always use weights_only=True to prevent arbitrary code
# execution from untrusted checkpoint files (pickle deserialization attack)
new_model = SimpleModel()  # Create architecture first
new_model.load_state_dict(torch.load('model_state.pt', weights_only=True))
new_model.eval()  # Set to evaluation mode

# Method 3: Save checkpoint (for resuming training)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
checkpoint = {
    'epoch': 10,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),  # Preserve optimizer momentum/state
    'loss': 0.5,
    'architecture': 'SimpleModel',
}
torch.save(checkpoint, 'checkpoint.pt')

# Load checkpoint (use weights_only=False for dicts with non-tensor data)
# Only do this for checkpoints you trust (your own training runs)
checkpoint = torch.load('checkpoint.pt', weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
print(f"Resumed from epoch {checkpoint['epoch']}, loss={checkpoint['loss']}")

# Cleanup
import os
for f in ['model_state.pt', 'checkpoint.pt']:
    if os.path.exists(f):
        os.remove(f)
```

---

## Part 4: Training Loop

### Complete Training Pipeline

```python
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import time

# Load MNIST
print("Loading MNIST...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist.data.astype(np.float32) / 255.0, mnist.target.astype(np.int64)

# Split: train (60%), validation (20%), test (20%)
# ⚠️ Use validation for early stopping/hyperparameter tuning; test ONLY for final eval
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.25, random_state=42)
# 0.25 of 80% = 20%, so: 60% train, 20% val, 20% test

# Convert to tensors
X_train = torch.tensor(X_train)
y_train = torch.tensor(y_train)
X_val = torch.tensor(X_val)
y_val = torch.tensor(y_val)
X_test = torch.tensor(X_test)
y_test = torch.tensor(y_test)

# Create datasets and dataloaders
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

print(f"Training batches: {len(train_loader)}")
print(f"Validation batches: {len(val_loader)}")
print(f"Test batches: {len(test_loader)}")


class Trainer:
    """
    A complete training pipeline with best practices.
    """

    def __init__(self, model, device='cpu', learning_rate=0.001):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3
        )

        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }

    def train_epoch(self, dataloader):
        """Train for one epoch."""
        self.model.train()  # Set training mode

        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(self.device), target.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()  # Clear gradients
            output = self.model(data)
            loss = self.criterion(output, target)

            # Backward pass
            loss.backward()

            # Gradient clipping (prevents exploding gradients)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # Update weights
            self.optimizer.step()

            # Track metrics
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)

        return total_loss / len(dataloader), correct / total

    @torch.no_grad()
    def evaluate(self, dataloader):
        """Evaluate model on a dataset."""
        self.model.eval()  # Set evaluation mode

        total_loss = 0
        correct = 0
        total = 0

        for data, target in dataloader:
            data, target = data.to(self.device), target.to(self.device)

            output = self.model(data)
            loss = self.criterion(output, target)

            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)

        return total_loss / len(dataloader), correct / total

    def train(self, train_loader, val_loader, epochs=10, early_stopping_patience=5):
        """
        Full training loop with validation and early stopping.
        """
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        for epoch in range(epochs):
            start_time = time.time()

            # Train
            train_loss, train_acc = self.train_epoch(train_loader)

            # Validate
            val_loss, val_acc = self.evaluate(val_loader)

            # Learning rate scheduling
            self.scheduler.step(val_loss)

            # Track history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            epoch_time = time.time() - start_time
            current_lr = self.optimizer.param_groups[0]['lr']

            print(f"Epoch {epoch+1:2d}/{epochs}: "
                  f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                  f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}, "
                  f"lr={current_lr:.6f}, time={epoch_time:.1f}s")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # Deep copy required: .copy() only shallow-copies the dict,
                # not the tensor values — use copy.deepcopy for safety
                best_model_state = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"Restored best model with val_loss={best_val_loss:.4f}")

        return self.history

    def plot_history(self, save_path='training_history.png'):
        """Plot training history."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Loss
        axes[0].plot(self.history['train_loss'], label='Train')
        axes[0].plot(self.history['val_loss'], label='Validation')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)

        # Accuracy
        axes[1].plot(self.history['train_acc'], label='Train')
        axes[1].plot(self.history['val_acc'], label='Validation')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.show()


# Create and train model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}")

model = MNISTClassifier(hidden_dims=[512, 256, 128], dropout=0.3)
trainer = Trainer(model, device=device, learning_rate=0.001)

history = trainer.train(train_loader, val_loader, epochs=20, early_stopping_patience=5)

# Final evaluation on HELD-OUT test set (never seen during training/validation)
test_loss, test_acc = trainer.evaluate(test_loader)
print(f"\nFinal test accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
print("(This is the unbiased estimate — test set was never used for early stopping)")

trainer.plot_history()

# === Per-Class Metrics (always go beyond aggregate accuracy) ===
print("\n=== Per-Class Evaluation ===")
all_preds = []
all_targets = []
with torch.no_grad():
    model.eval()
    for data, target in test_loader:
        data = data.to(device)
        preds = model(data).argmax(dim=1).cpu()
        all_preds.append(preds)
        all_targets.append(target)

all_preds = torch.cat(all_preds).numpy()
all_targets = torch.cat(all_targets).numpy()

print(f"{'Digit':>6} {'Precision':>10} {'Recall':>8} {'F1':>6} {'Support':>8}")
for digit in range(10):
    tp = np.sum((all_preds == digit) & (all_targets == digit))
    fp = np.sum((all_preds == digit) & (all_targets != digit))
    fn = np.sum((all_preds != digit) & (all_targets == digit))
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    support = np.sum(all_targets == digit)
    print(f"{digit:>6d} {precision:>10.3f} {recall:>8.3f} {f1:>6.3f} {support:>8d}")

# === Baseline Comparison ===
from sklearn.linear_model import LogisticRegression
print("\n=== Baseline Comparison ===")
lr_model = LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='multinomial')
lr_model.fit(X_trainval, y_trainval)
lr_acc = lr_model.score(X_test.numpy(), y_test.numpy())
print(f"Logistic Regression: {lr_acc:.4f} ({lr_acc*100:.2f}%)")
print(f"PyTorch Neural Net:  {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"Random chance:       0.1000 (10.00%)")
print(f"→ Neural net improvement over LR baseline: +{(test_acc - lr_acc)*100:.2f} pp")
```

> **✅ Checkpoint**: At this point you should be able to: (1) build a Trainer class with train/eval loops, early stopping, and LR scheduling, (2) implement proper 3-way split (train/val/test) and explain why each is needed, (3) compute per-class precision/recall/F1 and anchor results against a baseline, (4) save and load model checkpoints with `weights_only=True`.

---

## Part 5: Common PyTorch Patterns

### Custom Datasets

```python
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

class CustomDataset(Dataset):
    """
    Template for custom datasets.

    Must implement:
    - __len__: Return dataset size
    - __getitem__: Return single sample by index
    """

    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]

        if self.transform:
            x = self.transform(x)

        return x, y


# Example: CSV dataset
class CSVDataset(Dataset):
    def __init__(self, csv_path, feature_cols, target_col):
        df = pd.read_csv(csv_path)
        self.features = torch.tensor(df[feature_cols].values, dtype=torch.float32)
        self.targets = torch.tensor(df[target_col].values, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


# DataLoader options
dataset = TensorDataset(torch.randn(1000, 10), torch.randint(0, 2, (1000,)))

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,           # Shuffle for training
    num_workers=4,          # Parallel data loading (see note below)
    pin_memory=True,        # Faster GPU transfer (only useful with CUDA)
    drop_last=True          # Drop incomplete final batch
)
# num_workers guidelines:
# - Start with 0 (single process) for debugging
# - Try 2-4 for typical workloads; use os.cpu_count() as upper bound
# - ⚠️ On Windows: must guard with if __name__ == '__main__' or workers will crash
# - ⚠️ High num_workers + large dataset = high RAM usage (each worker loads data)
# - pin_memory=True only helps when training on GPU (CUDA pinned memory)
```

### Learning Rate Schedulers

```python
import torch.optim as optim

model = nn.Linear(10, 2)
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Step decay: multiply LR by gamma every step_size epochs
scheduler1 = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Exponential decay
scheduler2 = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

# Reduce on plateau (common for validation loss)
scheduler3 = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5
)

# Cosine annealing
scheduler4 = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

# One cycle (great for fast training)
scheduler5 = optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=0.01, total_steps=1000
)

# Usage in training loop
for epoch in range(100):
    # ... train ...
    val_loss = 0.5  # placeholder

    # For ReduceLROnPlateau
    scheduler3.step(val_loss)

    # For others
    # scheduler1.step()
```

### Mixed Precision Training

Mixed precision uses FP16 (or BF16) for most operations and FP32 for loss scaling, giving **1.5-3x speedup** on modern GPUs (Volta/Ampere+) with minimal accuracy loss.

```python
import torch
from torch.amp import autocast, GradScaler  # Modern API (PyTorch 2.4+)
# Note: torch.cuda.amp.autocast is DEPRECATED — use torch.amp.autocast instead

scaler = GradScaler('cuda')

for data, target in dataloader:
    optimizer.zero_grad()

    # Forward pass: FP16 for matmuls, FP32 for reductions (loss, softmax)
    with autocast('cuda'):  # Specify device type explicitly
        output = model(data)
        loss = criterion(output, target)

    # GradScaler prevents underflow in FP16 gradients
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

**When to use mixed precision:**
- Training large models (saves ~50% GPU memory)
- GPU has Tensor Cores (V100, A100, RTX 3000+, H100)
- Training is compute-bound (not data-loading-bound)

**FP16 vs BF16:**
- FP16: Wider range of hardware support, but prone to overflow/underflow → needs GradScaler
- BF16: Same exponent range as FP32 → no scaler needed, but requires Ampere+ GPU
- Use BF16 when available: `with autocast('cuda', dtype=torch.bfloat16):`

**Common pitfall:** NaN loss with FP16 usually means the scaler isn't working. Check that `scaler.get_scale()` isn't stuck at a very small value.

### GPU Memory Estimation

Before training, estimate if your model fits in VRAM:

```python
import torch
import torch.nn as nn

def estimate_memory(model, batch_size, input_shape, dtype=torch.float32):
    """Estimate GPU memory usage for training.

    Components:
    1. Model parameters: sum of all weight tensors
    2. Optimizer state: Adam stores m (momentum) and v (variance) per parameter = 2× params
    3. Gradients: same size as parameters = 1× params
    4. Activations: stored for backprop, depends on batch size and architecture
    Total ≈ 4× params (FP32) + activations
    """
    bytes_per = {torch.float32: 4, torch.float16: 2, torch.bfloat16: 2}[dtype]
    param_bytes = sum(p.numel() * bytes_per for p in model.parameters())
    grad_bytes = param_bytes  # One gradient per parameter
    adam_bytes = 2 * param_bytes  # m + v (always FP32 in Adam)

    # Activations: rough estimate (exact depends on architecture)
    # Each layer stores output for backprop: batch_size × layer_output_size × bytes
    # For feedforward: ≈ batch_size × sum(hidden_dims) × bytes
    total_hidden = sum(m.out_features for m in model.modules() if isinstance(m, nn.Linear))
    activation_bytes = batch_size * total_hidden * bytes_per

    total = param_bytes + grad_bytes + adam_bytes + activation_bytes
    print(f"=== Memory Estimate ({dtype}) ===")
    print(f"  Parameters:   {param_bytes/1e6:>8.1f} MB")
    print(f"  Gradients:    {grad_bytes/1e6:>8.1f} MB")
    print(f"  Adam state:   {adam_bytes/1e6:>8.1f} MB  (m + v, always FP32)")
    print(f"  Activations:  {activation_bytes/1e6:>8.1f} MB  (batch={batch_size})")
    print(f"  ─────────────────────────")
    print(f"  Total:        {total/1e6:>8.1f} MB")
    print(f"  + PyTorch overhead: ~300-500 MB (CUDA context, kernels)")
    return total

# Example: our MNIST model
model_test = MNISTClassifier(hidden_dims=[512, 256, 128])
estimate_memory(model_test, batch_size=128, input_shape=(128, 784))

# To measure ACTUAL usage during training:
# torch.cuda.reset_peak_memory_stats()
# ... run one training step ...
# print(f"Peak memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
```

### Gradient Accumulation

When your model is too large for a single batch to fit in GPU memory:

**Why gradient accumulation is mathematically equivalent to larger batches:**
The gradient of the loss over N samples is `(1/N) Σ ∇L_i`. If we split N into K mini-batches of size B (so N = K×B), each mini-batch gradient is `(1/B) Σ ∇L_i`. Summing K of these and dividing by K gives `(1/N) Σ ∇L_i` — the same gradient. The `loss / accumulation_steps` scaling below achieves this division.

```python
# Assumes: model, criterion, optimizer, dataloader, autocast, scaler
# are defined as in the mixed precision section above
accumulation_steps = 4  # Effective batch = actual_batch * accumulation_steps
optimizer.zero_grad()

for i, (data, target) in enumerate(dataloader):
    with autocast('cuda'):
        output = model(data)
        loss = criterion(output, target) / accumulation_steps  # Scale loss

    scaler.scale(loss).backward()  # Gradients accumulate

    if (i + 1) % accumulation_steps == 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

# Effective batch size: 32 (actual) × 4 (accumulation) = 128
```

---

### Reproducibility

```python
import torch
import numpy as np
import random

def set_seed(seed=42):
    """Set all random seeds for reproducibility.

    ⚠️ Full reproducibility on GPU requires cuDNN deterministic mode,
    which may slow training by 10-20%.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    np.random.seed(seed)  # For NumPy operations in data loading
    random.seed(seed)      # For Python random module

    # Deterministic algorithms (slower but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Note: Some operations (e.g., scatter_add) have no deterministic
    # implementation — set torch.use_deterministic_algorithms(True)
    # to catch these at runtime (will raise errors)

set_seed(42)

# For DataLoader: seed workers for reproducible data loading
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(42)

loader = DataLoader(dataset, batch_size=32, shuffle=True,
                    worker_init_fn=seed_worker, generator=g)
```

### Modern PyTorch: torch.compile (PyTorch 2.x)

```python
import torch

# torch.compile() fuses operations and generates optimized kernels
# Typical speedup: 1.3-2x on GPU, especially for transformer models
model = MNISTClassifier()

# Compile the model (one-time cost, then faster inference/training)
compiled_model = torch.compile(model)  # Default: 'inductor' backend

# Usage is identical to uncompiled model
output = compiled_model(x)

# Modes:
# torch.compile(model, mode='default')     # Balanced speed/compile time
# torch.compile(model, mode='reduce-overhead')  # Minimal overhead for small models
# torch.compile(model, mode='max-autotune')     # Best speed, longer compile
```

### Cross-Device Model Loading

A common production scenario: train on GPU, deploy on CPU (or different GPU).

```python
import torch

# Save on GPU
# torch.save(model.state_dict(), 'model.pt')

# Load on CPU (even if saved from GPU)
# state_dict = torch.load('model.pt', map_location='cpu', weights_only=True)
# model.load_state_dict(state_dict)

# Load on specific GPU
# state_dict = torch.load('model.pt', map_location='cuda:0', weights_only=True)

# Without map_location: if the checkpoint was saved on GPU and you don't
# have a GPU, torch.load will fail. Always specify map_location explicitly.
```

> **Industry note**: Most PyTorch code in production (2024+) uses either `torch.compile()` or the **Hugging Face ecosystem** (`transformers`, `accelerate`, `peft`) which handles distributed training, mixed precision, and gradient accumulation automatically. Learning raw PyTorch first (as in this blog) gives you the understanding to debug when these abstractions break.

> **✅ Checkpoint**: At this point you should be able to: (1) write a custom Dataset with `__len__` and `__getitem__`, (2) choose a LR scheduler and explain when ReduceLROnPlateau vs CosineAnnealing is better, (3) implement mixed precision training with GradScaler and explain FP16 vs BF16, (4) implement gradient accumulation and justify the `loss / accumulation_steps` scaling mathematically, (5) estimate GPU memory usage for a model before training.

---

## Part 6: Debugging PyTorch

### Common Issues and Solutions

```python
import torch
import torch.nn as nn

# Issue 1: Device mismatch
def debug_device_mismatch():
    """Most common PyTorch error."""
    model = nn.Linear(10, 2).cuda()
    x = torch.randn(5, 10)  # On CPU!

    try:
        output = model(x)
    except RuntimeError as e:
        print(f"Error: {e}")
        print("Solution: Move input to same device as model")
        x = x.cuda()
        output = model(x)
        print(f"Success! Output device: {output.device}")


# Issue 2: Gradient not computed
def debug_no_gradient():
    """Gradient is None after backward."""
    x = torch.tensor([1.0, 2.0])  # requires_grad=False by default!

    y = x ** 2
    try:
        y.sum().backward()
        print(f"x.grad: {x.grad}")  # None!
    except RuntimeError:
        pass

    # Solution: enable gradient tracking
    x = torch.tensor([1.0, 2.0], requires_grad=True)
    y = x ** 2
    y.sum().backward()
    print(f"x.grad (fixed): {x.grad}")


# Issue 3: In-place operation breaks gradient
def debug_inplace_error():
    """In-place operations can break autograd."""
    x = torch.tensor([1.0, 2.0], requires_grad=True)
    y = x ** 2

    # This modifies y in-place after it's used in computation graph
    try:
        y += 1  # In-place add
        y.sum().backward()
    except RuntimeError as e:
        print(f"Error: in-place operation")

    # Solution: use out-of-place operations
    x = torch.tensor([1.0, 2.0], requires_grad=True)
    y = x ** 2
    y = y + 1  # Creates new tensor
    y.sum().backward()
    print(f"x.grad (fixed): {x.grad}")


# Issue 4: Memory leak in evaluation
def debug_memory_leak():
    """Forgetting torch.no_grad() causes memory buildup."""
    model = nn.Linear(1000, 1000)

    # Bad: keeps building computation graph
    for i in range(100):
        x = torch.randn(100, 1000)
        y = model(x)  # Graph stored!

    # Good: no gradient tracking during inference
    with torch.no_grad():
        for i in range(100):
            x = torch.randn(100, 1000)
            y = model(x)  # No graph stored


# Issue 5: Model not in training/eval mode
def debug_train_eval_mode():
    """Dropout/BatchNorm behave differently in train vs eval."""
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.Dropout(0.5),
        nn.Linear(20, 2)
    )

    x = torch.randn(5, 10)

    model.train()  # Dropout active
    out_train = model(x)

    model.eval()  # Dropout inactive
    out_eval = model(x)

    print(f"Training mode output varies due to dropout")
    print(f"Eval mode output is deterministic")


# Run diagnostics
print("=== PyTorch Debugging Examples ===\n")

print("1. Device mismatch:")
if torch.cuda.is_available():
    debug_device_mismatch()
else:
    print("Skipped (no GPU)")

print("\n2. No gradient:")
debug_no_gradient()

print("\n3. In-place operation:")
debug_inplace_error()

print("\n4. Train/Eval mode:")
debug_train_eval_mode()
```

### Profiling and Optimization

```python
import torch
import torch.nn as nn
import time

def profile_model(model, input_shape, device='cpu', num_iterations=100):
    """Profile model forward pass performance."""
    device = torch.device(device)  # Handle both str and torch.device
    model = model.to(device)
    model.eval()

    x = torch.randn(input_shape, device=device)

    # Warmup (important for GPU — CUDA kernels compile on first call)
    for _ in range(10):
        with torch.no_grad():
            _ = model(x)

    # Time forward pass
    if device.type == 'cuda':
        torch.cuda.synchronize()

    start = time.time()
    for _ in range(num_iterations):
        with torch.no_grad():
            _ = model(x)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    elapsed = time.time() - start
    avg_time = elapsed / num_iterations * 1000  # ms

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())

    print(f"Model: {model.__class__.__name__}")
    print(f"Device: {device}")
    print(f"Parameters: {num_params:,}")
    print(f"Avg forward pass: {avg_time:.2f} ms")
    print(f"Throughput: {1000/avg_time:.1f} samples/sec (batch size {input_shape[0]})")


# Profile
model = nn.Sequential(
    nn.Linear(784, 512),
    nn.ReLU(),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

profile_model(model, (128, 784), device='cpu')
```

---

## 📊 Manager's Summary

### Framework Choice Impact

| Factor | Impact |
|--------|--------|
| Team expertise | Training time, bug frequency |
| Deployment target | Mobile (TensorFlow Lite), Server (either), Browser (TensorFlow.js) |
| Research vs Production | PyTorch dominates research; both work for production |
| Existing infrastructure | Cloud provider preferences, existing code |

### Key Metrics to Track

1. **Training speed**: Iterations per second
2. **GPU utilization**: Should be >80% for efficiency
3. **Memory usage**: Batch size vs available VRAM
4. **Convergence**: Loss should decrease; accuracy should increase

### Questions to Ask Your Team

1. "Why did you choose PyTorch over TensorFlow?"
2. "Are we using GPU? What's the utilization?"
3. "How long does one training epoch take?"
4. "Are we using mixed precision training?"
5. "What's our checkpoint strategy?"

---

## Interview Preparation

### Job Role Mapping

| Section | MLE / ML Engineer | Data Scientist | AI/ML Architect | Engineering Manager |
|---------|:-:|:-:|:-:|:-:|
| Part 1: Tensors | ✅ Must know | ⚡ Understand basics | ✅ Must know | — |
| Part 2: Autograd | ✅ Must know | ⚡ Understand concept | ✅ Must know | — |
| Part 3: nn.Module | ✅ Must implement | ⚡ Modify existing | ✅ Must review | — |
| Part 4: Training Pipeline | ✅ Must implement | ✅ Must run/modify | ✅ Must review | 📊 Manager's Summary |
| Part 5: Patterns (Mixed Precision) | ✅ Must implement | ⚡ Know tradeoffs | ✅ Must choose | 📊 Cost impact |
| Part 5: GPU Memory Estimation | ✅ Must estimate | ⚡ Understand limits | ✅ Must estimate | 📊 Budget decisions |
| Part 6: Debugging | ✅ Must master | ⚡ Recognize symptoms | ✅ Must diagnose | 📊 Questions to Ask |

**Interview context**: MLE coding rounds test training loop implementation and debugging. Data science interviews focus on evaluation metrics and experiment design. Architect interviews ask about GPU memory budgets, mixed precision tradeoffs, and framework selection. Manager interviews test "Questions to Ask Your Team."

### Likely Questions

**Q: What's the difference between .train() and .eval() in PyTorch?**
A: They set the model's mode for layers that behave differently during training vs inference. Dropout is active only in train mode. BatchNorm uses batch statistics in train mode but running statistics in eval mode. Always call the appropriate mode, and use torch.no_grad() during evaluation to save memory.

**Q: Explain autograd in PyTorch.**
A: Autograd is PyTorch's automatic differentiation engine. When you create tensors with requires_grad=True, PyTorch builds a computation graph during the forward pass. Calling .backward() traverses this graph in reverse, computing gradients via the chain rule. Gradients accumulate in .grad attributes.

**Q: How do you prevent overfitting in PyTorch?**
A: Multiple strategies:
- Dropout layers (nn.Dropout)
- Weight decay in optimizer (L2 regularization)
- Data augmentation (transforms)
- Early stopping (monitor validation loss)
- Batch normalization (also helps training stability)

**Q: What's the purpose of optimizer.zero_grad()?**
A: PyTorch accumulates gradients by default. If you don't zero them before each backward pass, gradients from multiple batches add up, leading to incorrect updates. Always call optimizer.zero_grad() at the start of each training step.

**Q: How would you debug a model that's not learning?**
A: Systematic approach:
1. Check learning rate (try 10x higher and lower)
2. Verify data pipeline (visualize batches)
3. Check for NaN/Inf in loss
4. Ensure model is in train() mode
5. Verify gradients flow (check .grad is not None)
6. Try overfitting on single batch first
7. Check loss function matches task (CE for classification, MSE for regression)

**Q: Explain mixed precision training. When would you use it?**
A: Mixed precision uses FP16/BF16 for most operations (matmuls) and FP32 for numerically sensitive operations (loss, softmax, reductions). GradScaler prevents FP16 gradient underflow by scaling the loss before backward() and unscaling before optimizer.step(). Use when: training on GPUs with Tensor Cores (V100+), model is compute-bound, and VRAM is a bottleneck. BF16 (Ampere+) doesn't need GradScaler because it has the same exponent range as FP32.

**Q: How do you estimate GPU memory requirements before training?**
A: Total VRAM ≈ parameters × dtype_bytes × 4 (weights + gradients + Adam m + Adam v) + batch_size × activations_per_layer × dtype_bytes + ~500MB CUDA overhead. For a 100M parameter model in FP32 with Adam: ~100M × 4B × 4 = 1.6GB + activations. Mixed precision roughly halves the parameter/activation portion but Adam state stays FP32.

**Q: What is torch.compile and when would you use it?**
A: torch.compile() (PyTorch 2.x) traces the model's forward pass and generates optimized GPU kernels using the Inductor backend. It fuses operations, eliminates Python overhead, and typically gives 1.3-2× speedup. Use for production inference and long training runs. Avoid for models with highly dynamic control flow (torch.compile handles simple branching but struggles with data-dependent shapes).

---

## Exercises (Do These)

1. **GPU speedup**: Train MNIST on CPU vs GPU. Measure speedup factor.

2. **Custom layer**: Implement a custom activation function as an nn.Module with proper backward.

3. **Checkpoint resume**: Save a checkpoint mid-training, then resume from it.

4. **Learning rate finder**: Implement exponentially increasing LR to find optimal range.

5. **TensorBoard**: Add TensorBoard logging to track training metrics.

---

## What This Blog Does NOT Cover

This blog covers PyTorch fundamentals for single-GPU training. Topics NOT covered include:

- **Distributed training**: `DistributedDataParallel`, FSDP, DeepSpeed — needed for multi-GPU/multi-node
- **Custom CUDA kernels**: Writing `.cu` files, Triton kernels for custom operations
- **Quantization**: INT8/INT4 inference, `torch.quantization`, GPTQ, AWQ
- **TorchScript & ONNX export**: Production deployment serialization (only mentioned, not demonstrated)
- **Custom autograd Functions**: Extending `torch.autograd.Function` for novel operations
- **TensorFlow in depth**: Only compared at a high level; no TF implementation provided
- **Hugging Face ecosystem**: `transformers`, `accelerate`, `peft` — covered in later NLP blogs
- **MLOps/experiment tracking**: Weights & Biases, MLflow, TensorBoard integration

---

## What's Next

You now have:
- ✅ PyTorch tensor fundamentals
- ✅ Automatic differentiation understanding
- ✅ Complete training pipeline
- ✅ Best practices for production
- ✅ Debugging skills for common issues

**Blog 6** starts Module 2: NLP. We'll cover text preprocessing and representation—tokenization, TF-IDF, and word embeddings. This is where AI meets language.

**[→ Blog 6: Text Preprocessing and Representation](#)**

---

---

## Self-Assessment Rubric

| Criteria | Excellent (9-10) | Good (7-8) | Needs Work (5-6) |
|----------|------------------|------------|------------------|
| **Tensor Operations** | Masters broadcasting, views, and GPU transfers | Comfortable with basic operations | Struggles with shapes |
| **Autograd Understanding** | Can trace computation graph and optimize memory | Knows .backward() and requires_grad | Treats autograd as black box |
| **Model Architecture** | Builds custom nn.Module with flexible layers | Uses Sequential for standard architectures | Cannot define custom modules |
| **Training Pipeline** | Implements checkpointing, early stopping, and logging | Has working train/eval loop | Training loop has bugs |
| **Performance Optimization** | Uses mixed precision, gradient accumulation, and profiling | Knows basic GPU usage | Ignores performance |
| **Overall Score** | See assessment below |

### Where This Blog Does Well
- Complete MNIST training pipeline with early stopping, LR scheduling, validation monitoring, and per-class precision/recall/F1
- Baseline comparison anchoring accuracy against logistic regression and random chance
- GPU memory estimation formula with component breakdown (params, gradients, optimizer, activations)
- Computation graph DAG explanation with static vs dynamic graph comparison
- Gradient accumulation with mathematical justification for loss scaling
- Modern mixed precision training with correct torch.amp API and FP16 vs BF16 tradeoff
- Cross-device model loading with `map_location`
- Security-aware model loading (weights_only=True explanation)
- Reproducibility setup with seed management and worker seeding
- torch.compile() coverage for PyTorch 2.x optimization
- Job role mapping for MLE, Data Scientist, AI Architect, and Manager
- 4 section checkpoints for self-verification

### Where This Blog Falls Short
- No side-by-side performance comparison: CPU vs GPU, compiled vs uncompiled
- No distributed training (DDP, FSDP) — deferred to later blogs
- No custom autograd Function implementation
- CSVDataset example references a file that isn't provided

---

## Architect Sanity Checks

### ✅ Check 1: Framework Proficiency
**Question**: Can you convert the from-scratch MNIST classifier to PyTorch?
**Answer: YES** — A complete PyTorch reimplementation uses nn.Module with nn.Sequential, torch.optim.Adam, CrossEntropyLoss, and autograd for automatic gradient computation. The MNISTClassifier demonstrates flexible architecture definition, proper train/eval mode handling, and achieves equivalent accuracy to the Blog 4 from-scratch version.

### ✅ Check 2: GPU Utilization
**Question**: How do you move computation to GPU efficiently?
**Answer: YES** — GPU training is demonstrated with `.to(device)` for model and tensors, DataLoader with `pin_memory=True` for async transfer, mixed precision training (`torch.amp.autocast`) for memory savings, and `torch.compile()` for kernel fusion. Device detection handles CUDA/CPU fallback gracefully.

### ✅ Check 3: Production Patterns
**Question**: How do you save and load models correctly?
**Answer: YES** — The blog demonstrates `state_dict()` saving (portable), `weights_only=True` for security, checkpoint saving with optimizer state, best-model tracking with `copy.deepcopy`, and cross-device loading with `map_location`. It also includes GPU memory estimation for capacity planning. TorchScript and ONNX export are NOT covered — these are deferred to deployment-focused blogs.

---

*Questions? Found an error? Comments are open. Technical corrections get priority.*
