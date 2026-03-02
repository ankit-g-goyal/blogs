# Blog 7: Sequence Models — RNNs and LSTMs
## When Order Matters

**Reading time:** 60–75 minutes
**Coding time:** 90–120 minutes
**Total investment:** 2.5–3 hours

---

## Prerequisites & Reading Guide

Before starting this blog, you should be comfortable with:

- **Blog 4** (Neural Networks from Scratch): Backpropagation, gradient descent, weight matrices
- **Blog 5** (PyTorch Deep Learning): `nn.Module`, training loops, `DataLoader`
- **Blog 6** (Text Preprocessing): Tokenization, vocabularies, bag-of-words representations
- **Linear algebra basics**: Matrix multiplication, vector operations (Blog 3)

**How to read this blog:**
- **Parts 1–2** (RNN + vanishing gradients): Read carefully — these build foundational intuition
- **Part 3** (LSTM): The core payoff — understand gates before moving on
- **Part 4** (BiLSTM classifier): Hands-on build — run the code yourself
- **Part 5** (RNN vs Transformer): Skim for decision-making context

---

## What This Blog Does NOT Cover

- **GRU (Gated Recurrent Unit)**: Mentioned briefly but not implemented — see exercises
- **Seq2Seq models**: Encoder-decoder architectures come in Blog 8 (Attention)
- **Transformer architectures**: Covered in Blog 9 — this blog provides the historical "why"
- **Production deployment**: Model serving, ONNX export, etc. are in Blog 24
- **Pre-trained language models**: BERT/GPT fine-tuning is covered in later blogs
- **Backpropagation Through Time (BPTT) derivation**: We show the gradient problem empirically, not the full calculus

---

## What You'll Walk Away With

By the end of this blog, you will:

1. **Understand** why sequences require different architectures than static data
2. **Implement** an RNN from scratch to see the core mechanism
3. **Explain** the vanishing gradient problem and why LSTMs exist
4. **Build** a bidirectional LSTM sentiment classifier with PyTorch
5. **Know** when RNNs make sense vs when to use transformers

This blog provides historical context for understanding why transformers were invented. You'll rarely use raw RNNs today, but understanding them makes attention mechanisms intuitive.

---

## Why Sequence Modeling?

### The Problem with Bag-of-Words

In Blog 6, we treated documents as bags of words—order didn't matter:

```
"The dog bit the man" → {the: 2, dog: 1, bit: 1, man: 1}
"The man bit the dog" → {the: 2, dog: 1, bit: 1, man: 1}

Same representation, VERY different meanings!
```

For many tasks, order matters:

| Task | Why Order Matters |
|------|------------------|
| Sentiment | "Not bad" ≠ "Bad, not good" |
| Translation | Word order differs between languages |
| Summarization | Need to understand narrative flow |
| Question Answering | "Who did what to whom?" |
| Time Series | Past values predict future |

### What Makes Sequences Special

Sequences have:
1. **Variable length**: Sentences can be 3 words or 300 words
2. **Positional meaning**: Position changes meaning
3. **Long-range dependencies**: "The cat, which was..." [many words] "...ate"
4. **Context**: Meaning of a word depends on surrounding words

Feedforward networks can't naturally handle these properties.

---

## Part 1: Recurrent Neural Networks (RNN)

### The Core Idea

An RNN processes sequences one element at a time, maintaining a **hidden state** that carries information from previous steps:

```
      Input:     x₁      x₂      x₃      x₄
                 ↓       ↓       ↓       ↓
Hidden:    h₀ → RNN → h₁ → RNN → h₂ → RNN → h₃ → RNN → h₄
                                                   ↓
Output:                                          y (e.g., classification)
```

The hidden state `h` acts as "memory" of what came before.

### Mathematical Formulation

At each timestep t:
```
hₜ = tanh(Wₓₓ · xₜ + Wₕₕ · hₜ₋₁ + b)
```

Where:
- `xₜ`: Input at time t
- `hₜ₋₁`: Hidden state from previous time step
- `Wₓₓ`: Input-to-hidden weights
- `Wₕₕ`: Hidden-to-hidden weights (the "recurrence")
- `b`: Bias
- `tanh`: Non-linearity (output range [-1, 1])

**Why tanh, not sigmoid or ReLU?**
- **Sigmoid** outputs [0, 1] — hidden states would always be positive, losing the ability to represent "negative" features. Also suffers from vanishing gradients when saturated.
- **ReLU** is unbounded — without the [-1, 1] constraint, hidden states can explode across time steps since they're repeatedly multiplied by `Wₕₕ`. ReLU works in feedforward nets (single pass) but is dangerous in recurrent architectures (many passes).
- **tanh** constrains values to [-1, 1], centering around zero (better for gradient flow than sigmoid), while preventing unbounded growth. Its derivative peaks at 1.0 (at tanh(0)) and approaches 0 at extremes — this is exactly the source of the vanishing gradient problem we'll explore in Part 2.

### RNN from Scratch

```python
import numpy as np

class SimpleRNN:
    """
    A minimal RNN implementation to understand the mechanism.

    This processes sequences one step at a time, maintaining a hidden state.
    """

    def __init__(self, input_size, hidden_size, output_size):
        self.hidden_size = hidden_size

        # Weight initialization (Xavier)
        scale = np.sqrt(2.0 / (input_size + hidden_size))

        # Input to hidden
        self.Wxh = np.random.randn(input_size, hidden_size) * scale
        # Hidden to hidden (the "recurrent" connection)
        self.Whh = np.random.randn(hidden_size, hidden_size) * scale
        # Hidden to output
        self.Why = np.random.randn(hidden_size, output_size) * scale

        # Biases
        self.bh = np.zeros(hidden_size)
        self.by = np.zeros(output_size)

    def forward(self, inputs, h_prev=None):
        """
        Forward pass through the RNN.

        Args:
            inputs: Sequence of inputs, shape (seq_len, input_size)
            h_prev: Initial hidden state, shape (hidden_size,)

        Returns:
            outputs: All outputs, shape (seq_len, output_size)
            hidden_states: All hidden states, shape (seq_len, hidden_size)
            final_hidden: Last hidden state
        """
        seq_len = len(inputs)

        if h_prev is None:
            h_prev = np.zeros(self.hidden_size)

        hidden_states = []
        outputs = []

        h = h_prev
        for t in range(seq_len):
            x = inputs[t]

            # Recurrent computation
            h = np.tanh(x @ self.Wxh + h @ self.Whh + self.bh)

            # Output computation
            y = h @ self.Why + self.by

            hidden_states.append(h)
            outputs.append(y)

        return np.array(outputs), np.array(hidden_states), h

    def __repr__(self):
        return f"SimpleRNN(hidden_size={self.hidden_size})"


# Example: Process a sequence
rnn = SimpleRNN(input_size=10, hidden_size=20, output_size=5)

# Sequence of 15 time steps, each with 10 features
sequence = np.random.randn(15, 10)

outputs, hidden_states, final_hidden = rnn.forward(sequence)

print(f"Input sequence shape: {sequence.shape}")
print(f"Outputs shape: {outputs.shape}")
print(f"Hidden states shape: {hidden_states.shape}")
print(f"Final hidden shape: {final_hidden.shape}")

# Visualize hidden state evolution
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Hidden state over time (first 5 dimensions)
axes[0].imshow(hidden_states[:, :5].T, aspect='auto', cmap='RdBu')
axes[0].set_xlabel('Time step')
axes[0].set_ylabel('Hidden dimension')
axes[0].set_title('Hidden State Evolution (5 dims)')
plt.colorbar(axes[0].images[0], ax=axes[0])

# Output over time
axes[1].imshow(outputs.T, aspect='auto', cmap='viridis')
axes[1].set_xlabel('Time step')
axes[1].set_ylabel('Output dimension')
axes[1].set_title('Output Evolution')
plt.colorbar(axes[1].images[0], ax=axes[1])

plt.tight_layout()
plt.savefig('rnn_evolution.png', dpi=150)
plt.show()
```

### RNN for Text Classification

```python
import numpy as np
import torch
import torch.nn as nn

class TextRNN(nn.Module):
    """
    RNN for text classification.

    Architecture:
    - Embedding layer: words → vectors
    - RNN: processes sequence
    - Linear: final hidden state → class scores
    """

    def __init__(self, vocab_size, embedding_dim, hidden_size, num_classes,
                 padding_idx=0):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,  # (batch, seq, features)
            bidirectional=False
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: (batch_size, seq_len)

        # Embed
        embedded = self.embedding(x)  # (batch, seq, embed_dim)

        # RNN
        output, hidden = self.rnn(embedded)
        # output: (batch, seq, hidden)
        # hidden: (1, batch, hidden)

        # Use final hidden state for classification
        final_hidden = hidden.squeeze(0)  # (batch, hidden)

        # Classify
        logits = self.fc(final_hidden)  # (batch, num_classes)

        return logits


# Test
model = TextRNN(vocab_size=10000, embedding_dim=100, hidden_size=128, num_classes=2)

# Batch of 32 sequences, each 50 tokens
x = torch.randint(0, 10000, (32, 50))
output = model(x)

print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
```

### ✅ Checkpoint: After Part 1

You should now be able to answer:
1. What is the hidden state in an RNN and what role does it play?
2. Why is tanh used instead of ReLU for the recurrent non-linearity?
3. What are the three weight matrices in a basic RNN (input-to-hidden, hidden-to-hidden, hidden-to-output)?
4. Why can't a feedforward network naturally handle variable-length sequences?

If you can't answer all four, re-read Part 1 before continuing.

---

## Part 2: The Vanishing Gradient Problem

### Why RNNs Struggle with Long Sequences

During backpropagation through time (BPTT), gradients are multiplied at each time step:

```python
import numpy as np
import matplotlib.pyplot as plt

def simulate_gradient_flow(seq_len, gradient_scale=0.9):
    """
    Simulate gradient flow through an RNN.

    If gradient_scale < 1: vanishing gradients
    If gradient_scale > 1: exploding gradients
    """
    gradient = 1.0
    gradients = [gradient]

    for t in range(seq_len):
        gradient *= gradient_scale
        gradients.append(gradient)

    return gradients

# Compare different scales
seq_lengths = range(50)

plt.figure(figsize=(12, 5))

for scale, label in [(0.9, 'Vanishing (0.9)'), (1.0, 'Stable (1.0)'),
                      (0.5, 'Severe vanishing (0.5)'), (1.1, 'Exploding (1.1)')]:
    grads = simulate_gradient_flow(50, scale)
    plt.semilogy(grads, label=label, linewidth=2)

plt.xlabel('Time steps from output')
plt.ylabel('Gradient magnitude (log scale)')
plt.title('Gradient Flow in RNNs')
plt.legend()
plt.grid(True)
plt.savefig('gradient_flow.png', dpi=150)
plt.show()

print("""
The Problem:
- With gradient scale < 1: Gradients vanish exponentially
- Early words have nearly zero gradient → don't learn long-range dependencies
- "The movie that I watched last night with my friends was" → "good"
  The word "movie" barely affects gradient of "good" 10 steps later
""")
```

### Mathematical Demonstration

```python
import numpy as np

# Simulate gradient through 20 RNN steps
rng = np.random.default_rng(42)
seq_len = 20

# Initialize hidden-to-hidden weight matrix
hidden_size = 50
W_hh = rng.standard_normal((hidden_size, hidden_size)) * 0.5

# Compute gradient flow
gradient = np.eye(hidden_size)
gradient_norms = [np.linalg.norm(gradient)]

for t in range(seq_len):
    # Gradient multiplied by W_hh at each step (simplified)
    # Also multiplied by derivative of tanh, max 1.0
    gradient = gradient @ W_hh * 0.8  # 0.8 represents tanh derivative
    gradient_norms.append(np.linalg.norm(gradient))

print(f"Initial gradient norm: {gradient_norms[0]:.4f}")
print(f"After 5 steps: {gradient_norms[5]:.4f}")
print(f"After 10 steps: {gradient_norms[10]:.4f}")
print(f"After 20 steps: {gradient_norms[20]:.4f}")

# Ratio
print(f"\nGradient decay ratio: {gradient_norms[20]/gradient_norms[0]:.6f}")
print("This means early inputs barely affect the loss!")
```

### The BPTT Chain Rule (Why Gradients Vanish Mathematically)

```python
"""
Backpropagation Through Time (BPTT) — The Chain Rule Applied to Sequences

For a loss L computed at the final time step T, the gradient with respect
to hidden state at time t is:

    ∂L/∂hₜ = ∂L/∂hₜ × Π_{k=t}^{T-1} (∂h_{k+1}/∂hₖ)

Each factor ∂h_{k+1}/∂hₖ involves:
    ∂h_{k+1}/∂hₖ = diag(tanh'(zₖ₊₁)) × Wₕₕ

where tanh'(z) = 1 - tanh²(z), and zₖ₊₁ = Wₓₓ·xₖ₊₁ + Wₕₕ·hₖ + b.

So the full gradient is a PRODUCT of (T-t) matrices:

    ∂L/∂hₜ = ∂L/∂hₜ × Π_{k=t}^{T-1} [diag(tanh'(zₖ₊₁)) × Wₕₕ]

KEY INSIGHT: This is a product of (T-t) terms.
- If the largest singular value of Wₕₕ × tanh' < 1 → product shrinks exponentially → VANISHING
- If the largest singular value of Wₕₕ × tanh' > 1 → product grows exponentially → EXPLODING

Since tanh'(z) ∈ (0, 1], the derivative always shrinks the gradient further.
For a sequence of length 100, you're multiplying ~100 matrices together.
Even a singular value of 0.9 gives 0.9^100 ≈ 2.6 × 10⁻⁵ — effectively zero.

PRACTICAL CONSEQUENCE:
- Gradients for time steps far from the loss are near zero
- The model can't learn dependencies spanning more than ~10-20 steps
- This is NOT a bug — it's a structural property of vanilla RNNs

SOLUTIONS:
1. LSTM cell state (additive updates, not multiplicative) — covered in Part 3
2. Gradient clipping (prevents exploding, doesn't fix vanishing)
3. Truncated BPTT (only backprop through a fixed window, e.g., 35 steps)
4. Skip connections / residual connections (shortcut gradient paths)
"""
```

### Real Impact on Learning

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def check_gradient_flow(model, seq_len):
    """Check gradient magnitudes at each position in sequence."""
    # Create input
    x = torch.randn(1, seq_len, model.input_size, requires_grad=True)

    # Forward pass
    h = None
    hidden_states = []
    for t in range(seq_len):
        h = torch.tanh(x[:, t, :] @ model.W_xh + (h @ model.W_hh if h is not None else 0) + model.b_h)
        hidden_states.append(h)

    # Backward from final hidden state
    loss = hidden_states[-1].sum()
    loss.backward()

    # Check gradient at each position
    gradients = []
    for t in range(seq_len):
        grad_norm = x.grad[:, t, :].norm().item()
        gradients.append(grad_norm)

    return gradients


class SimpleRNNModule(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.W_xh = nn.Parameter(torch.randn(input_size, hidden_size) * 0.1)
        self.W_hh = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.1)
        self.b_h = nn.Parameter(torch.zeros(hidden_size))


model = SimpleRNNModule(input_size=10, hidden_size=20)
gradients = check_gradient_flow(model, seq_len=30)

plt.figure(figsize=(10, 5))
plt.plot(range(30), gradients, 'b-o')
plt.xlabel('Position in sequence (0 = first word)')
plt.ylabel('Gradient magnitude')
plt.title('Vanishing Gradients: Earlier positions have smaller gradients')
plt.grid(True)
plt.savefig('vanishing_gradient_demo.png', dpi=150)
plt.show()
```

### ✅ Checkpoint: After Part 2

You should now be able to answer:
1. Write the BPTT gradient chain rule for ∂L/∂hₜ (product of Jacobians).
2. Why does a singular value of Wₕₕ < 1 cause vanishing gradients?
3. What is the practical consequence for learning long-range dependencies?
4. Name three solutions to the vanishing gradient problem and which ones fix vanishing vs exploding.

If you can't answer all four, re-read Part 2 before continuing.

---

## Part 3: LSTM — Long Short-Term Memory

### The LSTM Solution

LSTMs add **gates** that control information flow, allowing gradients to flow more easily:

```
Standard RNN: hₜ = tanh(W·[hₜ₋₁, xₜ])
                   └── All information goes through this bottleneck

LSTM: Adds a "cell state" cₜ that can carry information across many steps
      Gates control what to add/remove from cell state
```

### LSTM Architecture

```
       forget     input       cell        output
       gate       gate        update      gate
         ↓          ↓           ↓           ↓
        [f]        [i]         [c̃]         [o]
         │          │           │           │
         └────┬─────┴───────────┴─────┬─────┘
              │                       │
              ↓                       ↓
         ┌────────────────────────────────┐
cₜ₋₁ ──→ │  cₜ = f·cₜ₋₁ + i·c̃           │ ──→ cₜ
         └────────────────────────────────┘
                      │
                      ↓
                 hₜ = o·tanh(cₜ)
```

**Gates**:
- **Forget gate (f)**: What to remove from cell state
- **Input gate (i)**: What new information to add
- **Output gate (o)**: What to output from cell state

### LSTM from Scratch

```python
import numpy as np

class LSTMCell:
    """
    Single LSTM cell implementation.

    Shows exactly how gates control information flow.
    """

    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Combined weight matrices for efficiency
        # [input gate, forget gate, cell update, output gate]
        combined_size = input_size + hidden_size
        self.W = np.random.randn(combined_size, 4 * hidden_size) * 0.1
        self.b = np.zeros(4 * hidden_size)

        # CRITICAL: Initialize forget gate bias to 1.0 (not 0.0)
        # With bias=0, sigmoid(0)=0.5 → model starts by forgetting 50% of cell state
        # With bias=1, sigmoid(1)≈0.73 → model starts by REMEMBERING most cell state
        # This was shown by Jozefowicz et al. (2015) to significantly improve learning.
        # PyTorch's nn.LSTM does this automatically since v1.8.
        self.b[hidden_size:2*hidden_size] = 1.0  # Forget gate bias

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def forward(self, x, h_prev, c_prev):
        """
        Single LSTM step.

        Args:
            x: Input, shape (input_size,)
            h_prev: Previous hidden state, shape (hidden_size,)
            c_prev: Previous cell state, shape (hidden_size,)

        Returns:
            h: New hidden state
            c: New cell state
        """
        H = self.hidden_size

        # Concatenate input and previous hidden state
        combined = np.concatenate([x, h_prev])

        # Compute all gates at once
        gates = combined @ self.W + self.b

        # Split into individual gates
        i = self.sigmoid(gates[:H])      # Input gate
        f = self.sigmoid(gates[H:2*H])   # Forget gate
        g = np.tanh(gates[2*H:3*H])      # Cell candidate
        o = self.sigmoid(gates[3*H:])    # Output gate

        # Update cell state
        c = f * c_prev + i * g

        # Compute output
        h = o * np.tanh(c)

        # Store for visualization
        self.gates = {'input': i, 'forget': f, 'output': o, 'cell_candidate': g}

        return h, c


class LSTM:
    """Full LSTM for sequence processing."""

    def __init__(self, input_size, hidden_size):
        self.cell = LSTMCell(input_size, hidden_size)
        self.hidden_size = hidden_size

    def forward(self, inputs):
        """
        Process entire sequence.

        Args:
            inputs: Shape (seq_len, input_size)

        Returns:
            outputs: All hidden states, shape (seq_len, hidden_size)
            final_state: (h, c) tuple
        """
        seq_len = len(inputs)

        h = np.zeros(self.hidden_size)
        c = np.zeros(self.hidden_size)

        outputs = []
        gate_history = {'input': [], 'forget': [], 'output': []}

        for t in range(seq_len):
            h, c = self.cell.forward(inputs[t], h, c)
            outputs.append(h)

            # Track gate activations
            for gate_name in gate_history:
                gate_history[gate_name].append(self.cell.gates[gate_name].mean())

        return np.array(outputs), (h, c), gate_history


# Demonstrate LSTM gate behavior
import matplotlib.pyplot as plt

lstm = LSTM(input_size=10, hidden_size=20)

# Process a sequence
sequence = np.random.randn(30, 10)
outputs, final_state, gate_history = lstm.forward(sequence)

# Visualize gate activations
plt.figure(figsize=(12, 6))
for gate_name, values in gate_history.items():
    plt.plot(values, label=gate_name, linewidth=2)

plt.xlabel('Time step')
plt.ylabel('Average gate activation')
plt.title('LSTM Gate Activations Over Time')
plt.legend()
plt.grid(True)
plt.savefig('lstm_gates.png', dpi=150)
plt.show()

print("Gate interpretations:")
print("- Forget gate ≈ 1: Keep previous cell state")
print("- Forget gate ≈ 0: Reset cell state")
print("- Input gate ≈ 1: Write new information")
print("- Output gate ≈ 1: Use cell state for output")
```

### Why LSTMs Handle Long Sequences Better

```python
import numpy as np
import matplotlib.pyplot as plt

def lstm_gradient_flow(seq_len, forget_gate_bias=1.0):
    """
    Simulate gradient flow through LSTM.

    Key insight: When forget gate ≈ 1, gradients flow directly through
    the cell state without multiplication by weight matrices.
    """
    # Gradient through cell state path
    gradient_cell = 1.0
    cell_gradients = [gradient_cell]

    # Gradient through hidden state path (like RNN)
    gradient_hidden = 1.0
    hidden_gradients = [gradient_hidden]

    for t in range(seq_len):
        # Cell state gradient: multiplied by forget gate (can be close to 1)
        forget_gate = 0.9  # Biased toward keeping information
        gradient_cell *= forget_gate
        cell_gradients.append(gradient_cell)

        # Hidden state gradient: multiplied by weights (shrinks like RNN)
        gradient_hidden *= 0.5  # Weight multiplication effect
        hidden_gradients.append(gradient_hidden)

    return cell_gradients, hidden_gradients


cell_grads, hidden_grads = lstm_gradient_flow(50)

plt.figure(figsize=(10, 6))
plt.semilogy(cell_grads, label='Cell state path (LSTM)', linewidth=2)
plt.semilogy(hidden_grads, label='Hidden state path (RNN-like)', linewidth=2)
plt.xlabel('Time steps from output')
plt.ylabel('Gradient magnitude (log scale)')
plt.title('LSTM vs RNN Gradient Flow')
plt.legend()
plt.grid(True)
plt.savefig('lstm_gradient_flow.png', dpi=150)
plt.show()

print("""
Why LSTM works better:
1. Cell state provides a "highway" for gradients
2. Forget gate controls this highway (biased toward open)
3. Gradients don't have to go through weight matrices at every step
4. Information can persist across many time steps
""")
```

### GRU: The Simplified Alternative

```python
"""
GRU (Gated Recurrent Unit) — Cho et al., 2014

GRU simplifies LSTM by:
1. Combining forget + input gates into a single "update gate" z
2. Merging cell state and hidden state into one state h
3. Using a "reset gate" r to control how much past state to forget

Architecture:
    zₜ = σ(Wz · [hₜ₋₁, xₜ])          # Update gate: how much to keep from past
    rₜ = σ(Wr · [hₜ₋₁, xₜ])          # Reset gate: how much past to expose
    h̃ₜ = tanh(W · [rₜ * hₜ₋₁, xₜ])   # Candidate hidden state
    hₜ = (1 - zₜ) * hₜ₋₁ + zₜ * h̃ₜ   # Final: interpolate old and new

vs LSTM (3 gates + cell state):
    fₜ, iₜ, oₜ = sigmoid(...)         # Forget, Input, Output gates
    cₜ = fₜ * cₜ₋₁ + iₜ * c̃ₜ         # Cell state update
    hₜ = oₜ * tanh(cₜ)                # Hidden state from cell

TRADEOFF:
- GRU: 3N² parameters per layer (2 gates × N² + candidate × N²)
- LSTM: 4N² parameters per layer (3 gates × N² + candidate × N²)
- GRU trains ~15-20% faster (fewer parameters, simpler gradient path)
- LSTM has slightly more capacity (separate cell state = more memory)
- In practice: performance is similar for most tasks (within 1-2% accuracy)

WHEN TO PREFER GRU:
- Smaller datasets (fewer parameters → less overfitting)
- Faster prototyping (quicker training cycles)
- Shorter sequences (GRU's simpler gradient path is sufficient)

WHEN TO PREFER LSTM:
- Very long sequences (separate cell state preserves more information)
- Large datasets (more parameters can be utilized)
- When you need to inspect cell state separately from output
"""

class GRUCell:
    """GRU cell from scratch for comparison with LSTM."""

    def __init__(self, input_size, hidden_size):
        self.hidden_size = hidden_size
        combined = input_size + hidden_size

        # Update gate, Reset gate, Candidate (3 weight matrices vs LSTM's 4)
        self.W_z = np.random.randn(combined, hidden_size) * 0.1
        self.W_r = np.random.randn(combined, hidden_size) * 0.1
        self.W_h = np.random.randn(combined, hidden_size) * 0.1

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def forward(self, x, h_prev):
        combined = np.concatenate([h_prev, x])

        z = self.sigmoid(combined @ self.W_z)          # Update gate
        r = self.sigmoid(combined @ self.W_r)          # Reset gate

        combined_reset = np.concatenate([r * h_prev, x])
        h_candidate = np.tanh(combined_reset @ self.W_h)  # Candidate

        h = (1 - z) * h_prev + z * h_candidate        # Interpolate
        return h


# Parameter count comparison
hidden = 256
input_dim = 128
lstm_params = 4 * (input_dim + hidden) * hidden  # 4 gates
gru_params = 3 * (input_dim + hidden) * hidden   # 3 gates
print(f"LSTM parameters: {lstm_params:,} ({lstm_params/1e6:.1f}M)")
print(f"GRU parameters:  {gru_params:,} ({gru_params/1e6:.1f}M)")
print(f"GRU is {1 - gru_params/lstm_params:.0%} smaller")
```

### ✅ Checkpoint: After Part 3

You should now be able to answer:
1. Name the three LSTM gates and explain what each controls.
2. Why is the forget gate bias initialized to 1.0 instead of 0.0?
3. How does the cell state provide a "gradient highway" that vanilla RNNs lack?
4. What's the structural difference between GRU and LSTM? When would you choose each?

If you can't answer all four, re-read Part 3 before continuing.

---

## Part 4: Building a Sentiment Classifier with Bidirectional LSTM

### Why Bidirectional?

A forward LSTM only sees past context. But understanding often requires future context too:

```
"The movie was not as bad as I expected"
                 ↑
    To understand "not", we need to see "bad" (future)
    and "expected" (even more future)
```

Bidirectional LSTM processes the sequence in both directions:

```
Forward:   x₁ → x₂ → x₃ → x₄ → h_forward
Backward:  x₁ ← x₂ ← x₃ ← x₄ ← h_backward
Output:    [h_forward; h_backward]
```

### Complete Implementation

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from collections import Counter
import re

class SentimentDataset(Dataset):
    """Dataset for sentiment classification."""

    def __init__(self, texts, labels, vocab=None, max_len=100):
        self.texts = texts
        self.labels = labels
        self.max_len = max_len

        if vocab is None:
            self.vocab = self._build_vocab(texts)
        else:
            self.vocab = vocab

    def _build_vocab(self, texts, min_freq=2):
        """Build vocabulary from texts."""
        word_counts = Counter()
        for text in texts:
            words = self._tokenize(text)
            word_counts.update(words)

        vocab = {'<PAD>': 0, '<UNK>': 1}
        for word, count in word_counts.most_common():
            if count >= min_freq:
                vocab[word] = len(vocab)

        return vocab

    def _tokenize(self, text):
        """Simple tokenization."""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text.split()

    def _encode(self, text):
        """Convert text to indices."""
        words = self._tokenize(text)
        indices = [self.vocab.get(w, self.vocab['<UNK>']) for w in words]

        # Pad or truncate
        if len(indices) < self.max_len:
            indices = indices + [self.vocab['<PAD>']] * (self.max_len - len(indices))
        else:
            indices = indices[:self.max_len]

        return indices

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        indices = self._encode(text)
        return torch.tensor(indices), torch.tensor(label)


class BiLSTMClassifier(nn.Module):
    """
    Bidirectional LSTM for text classification.

    Architecture:
    - Embedding layer
    - Bidirectional LSTM (captures both past and future context)
    - Attention pooling (weighted average of hidden states)
    - Classification head
    """

    def __init__(self, vocab_size, embedding_dim=128, hidden_size=256,
                 num_layers=2, num_classes=2, dropout=0.3, padding_idx=0):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=padding_idx
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Attention for pooling
        self.attention = nn.Linear(hidden_size * 2, 1)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )

    def attention_pool(self, lstm_output, mask=None):
        """
        Attention-weighted pooling of LSTM outputs.

        Args:
            lstm_output: (batch, seq_len, hidden*2)
            mask: (batch, seq_len) - 1 for real tokens, 0 for padding
        """
        # Compute attention scores
        scores = self.attention(lstm_output).squeeze(-1)  # (batch, seq_len)

        # Mask padding tokens
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Softmax to get weights
        weights = torch.softmax(scores, dim=1)  # (batch, seq_len)

        # Weighted sum
        pooled = torch.bmm(weights.unsqueeze(1), lstm_output)  # (batch, 1, hidden*2)
        pooled = pooled.squeeze(1)  # (batch, hidden*2)

        return pooled, weights

    def forward(self, x, return_attention=False):
        """
        Forward pass.

        Args:
            x: Token indices, shape (batch, seq_len)
            return_attention: Whether to return attention weights
        """
        # Create padding mask
        mask = (x != 0).float()

        # Embed
        embedded = self.embedding(x)  # (batch, seq, embed)

        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(embedded)
        # lstm_out: (batch, seq, hidden*2)
        # h_n: (num_layers*2, batch, hidden)

        # Attention pooling
        pooled, attention_weights = self.attention_pool(lstm_out, mask)

        # Classify
        logits = self.classifier(pooled)

        if return_attention:
            return logits, attention_weights
        return logits


# Training function
def train_model(model, train_loader, val_loader, epochs=10, device='cpu'):
    """Train the BiLSTM model."""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()

            # Gradient clipping (important for RNNs!)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += loss.item()
            train_correct += (outputs.argmax(1) == batch_y).sum().item()
            train_total += batch_y.size(0)

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)

                val_loss += loss.item()
                val_correct += (outputs.argmax(1) == batch_y).sum().item()
                val_total += batch_y.size(0)

        # Record metrics
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{epochs}: "
              f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
              f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

    return history


# ---- Synthetic demo data ----
# WARNING: This is a trivially small synthetic dataset repeated to fill batches.
# A real sentiment classifier needs thousands of diverse examples.
# For production work, use datasets like IMDB (50K reviews), SST-2, or Yelp.
#
# To use IMDB instead:
#   from torchtext.datasets import IMDB
#   or: import datasets; ds = datasets.load_dataset("imdb")
train_texts = [
    "This movie was absolutely wonderful and amazing",
    "I loved every minute of this fantastic film",
    "A terrible waste of time, completely boring",
    "The worst movie I have ever seen in my life",
    "Great acting and a compelling story throughout",
    "Dull, lifeless, and utterly disappointing film",
    "An incredible masterpiece of modern cinema",
    "I fell asleep halfway through this disaster",
    "Brilliant direction and stellar performances all around",
    "Predictable plot with wooden dialogue and bad pacing",
    "One of the best films released this year hands down",
    "Could not wait for this painfully slow movie to end",
] * 80  # Repeat to fill batches — NOT a substitute for real data

train_labels = [1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0] * 80

# Create dataset and dataloader
dataset = SentimentDataset(train_texts, train_labels)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Create and train model
model = BiLSTMClassifier(
    vocab_size=len(dataset.vocab),
    embedding_dim=128,
    hidden_size=128,
    num_layers=2,
    num_classes=2,
    dropout=0.3
)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

history = train_model(model, train_loader, val_loader, epochs=5)

# Visualize attention
def visualize_attention(model, text, vocab, device='cpu'):
    """Visualize what the model attends to."""
    model.eval()

    # Tokenize and encode
    words = text.lower().split()
    indices = [vocab.get(w, vocab['<UNK>']) for w in words]
    indices = indices + [vocab['<PAD>']] * (100 - len(indices))
    x = torch.tensor([indices]).to(device)

    with torch.no_grad():
        logits, attention = model(x, return_attention=True)

    pred = torch.softmax(logits, dim=1)
    pred_label = "Positive" if pred[0, 1] > 0.5 else "Negative"

    # Get attention for actual words (not padding)
    attention = attention[0, :len(words)].cpu().numpy()

    plt.figure(figsize=(12, 4))
    plt.bar(range(len(words)), attention)
    plt.xticks(range(len(words)), words, rotation=45, ha='right')
    plt.ylabel('Attention weight')
    plt.title(f'Attention Visualization | Prediction: {pred_label} ({pred[0, 1]:.2f})')
    plt.tight_layout()
    plt.savefig('attention_visualization.png', dpi=150)
    plt.show()


visualize_attention(model, "This movie was absolutely terrible and boring", dataset.vocab)
```

### Packed Sequences: Efficient Variable-Length Processing

```python
"""
PROBLEM: Padding wastes computation. If your batch has sequences of length
[5, 12, 8, 3], padding to max_len=12 means the LSTM processes 7 padding
tokens for the first sequence — wasted FLOPs and potentially misleading hidden states.

SOLUTION: PyTorch's pack_padded_sequence / pad_packed_sequence.
"""
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class EfficientBiLSTM(nn.Module):
    """BiLSTM that uses packed sequences for variable-length inputs."""

    def __init__(self, vocab_size, embedding_dim, hidden_size, num_classes, padding_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_size, num_layers=2,
            batch_first=True, bidirectional=True, dropout=0.3
        )
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x, lengths):
        """
        Args:
            x: Padded token indices (batch, max_seq_len)
            lengths: Actual sequence lengths (batch,)
        """
        embedded = self.embedding(x)

        # Pack: tells LSTM to skip padding tokens
        packed = pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        # LSTM processes only real tokens
        packed_output, (h_n, c_n) = self.lstm(packed)

        # Unpack back to padded form (if you need per-token outputs)
        # output, _ = pad_packed_sequence(packed_output, batch_first=True)

        # Use final hidden states (forward + backward)
        # h_n shape: (num_layers * 2, batch, hidden) for bidirectional
        forward_final = h_n[-2]   # Last layer, forward direction
        backward_final = h_n[-1]  # Last layer, backward direction
        pooled = torch.cat([forward_final, backward_final], dim=1)

        return self.fc(pooled)


# Example usage with collate function for DataLoader
def collate_fn(batch):
    """Custom collate that returns lengths for packing."""
    texts, labels = zip(*batch)
    lengths = torch.tensor([t.nonzero().size(0) for t in texts])  # Non-padding length
    texts = torch.stack(texts)
    labels = torch.stack(labels)
    return texts, labels, lengths

# train_loader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)
# In training loop: outputs = model(batch_x, lengths)

print("Why packed sequences matter:")
print("  Batch: [len=5, len=12, len=8, len=3], max_len=12")
print("  Without packing: 4 × 12 = 48 LSTM steps (16 wasted on padding)")
print("  With packing: 5 + 12 + 8 + 3 = 28 LSTM steps (0 waste)")
print("  Speedup depends on length variance — higher variance = more savings")
```

### Baseline Comparison: Is BiLSTM Better Than Bag-of-Words?

```python
"""
CRITICAL: Never deploy a complex model without comparing to a simple baseline.
If logistic regression on TF-IDF achieves 85% and your BiLSTM gets 86%,
the complexity is not justified.
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def baseline_comparison(train_texts, train_labels, val_texts, val_labels):
    """Compare BiLSTM against bag-of-words baseline."""

    # Baseline 1: Random chance
    import numpy as np
    random_preds = np.random.randint(0, 2, len(val_labels))
    random_acc = accuracy_score(val_labels, random_preds)

    # Baseline 2: TF-IDF + Logistic Regression
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
    X_train = tfidf.fit_transform(train_texts)
    X_val = tfidf.transform(val_texts)

    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, train_labels)
    lr_preds = lr.predict(X_val)
    lr_acc = accuracy_score(val_labels, lr_preds)

    print("="*60)
    print("BASELINE COMPARISON")
    print("="*60)
    print(f"Random chance:              {random_acc:.4f}")
    print(f"TF-IDF + LogisticRegression: {lr_acc:.4f}")
    print(f"BiLSTM (from training):     [insert your val_acc here]")
    print()
    print("If BiLSTM < TF-IDF baseline, check:")
    print("  1. Is the dataset too small for LSTM to learn?")
    print("  2. Is the LSTM overfitting (train_acc >> val_acc)?")
    print("  3. Are hyperparameters reasonable (lr, hidden_size, dropout)?")
    print()
    print("TF-IDF baseline classification report:")
    print(classification_report(val_labels, lr_preds, target_names=["Negative", "Positive"]))

    return {'random': random_acc, 'tfidf_lr': lr_acc}

# Usage (with real data like IMDB):
# baselines = baseline_comparison(train_texts, train_labels, val_texts, val_labels)
```

### ✅ Checkpoint: After Part 4

You should now be able to answer:
1. Why does a bidirectional LSTM need both forward and backward passes?
2. What does `pack_padded_sequence` do and why is it important for efficiency?
3. Why should you always compare against a simple baseline (e.g., TF-IDF + LogReg)?
4. What does the attention pooling layer do and why is it better than using just the final hidden state?

If you can't answer all four, re-read Part 4 before continuing.

---

## Part 5: When to Use RNNs/LSTMs vs Transformers

### Comparison

| Factor | RNN/LSTM | Transformer |
|--------|----------|-------------|
| Sequence length | Struggles > 100-500 tokens | Handles thousands |
| Training speed | Sequential (slow) | Parallel (fast) |
| Memory per token | O(1) | O(n²) |
| Long-range deps | Difficult | Easy |
| Interpretability | Hidden state unclear | Attention is interpretable |
| Model size | Typically smaller | Typically larger |

### When to Use RNNs/LSTMs

1. **Resource constraints**: Mobile, embedded devices (LSTM ~1-4M params vs BERT ~110M)
2. **Streaming data**: Real-time processing where you can't wait for full sequence — LSTM processes token-by-token
3. **Very long sequences where O(n²) attention is prohibitive**: LSTM uses O(1) memory per step vs transformer's O(n²). For a 10K-token sequence, transformer attention needs ~380 MB (10K² × 4 bytes), while LSTM needs ~2 MB regardless of length. **Caveat:** LSTM still *struggles to learn* long-range dependencies > 100-500 steps — the advantage here is purely memory/compute, not modeling quality
4. **Simple sequence tasks**: When transformer is overkill (e.g., basic time-series, short text classification)

### Truncated BPTT: Practical Long-Sequence Training

```python
"""
For sequences longer than ~100-200 tokens, full BPTT is impractical:
- Memory: Must store all intermediate hidden states for backprop
- Gradients: Vanish/explode over long chains

TRUNCATED BPTT: Only backpropagate through a fixed window (e.g., 35 steps).

How it works:
1. Process the full sequence forward (hidden state carries information)
2. But only compute gradients for the last `bptt_len` steps
3. Detach hidden state at the boundary (stop gradient flow further back)

This is how language models (pre-transformer era) were trained on long texts.
"""

def train_with_truncated_bptt(model, data, bptt_len=35, device='cpu'):
    """Train LSTM with truncated BPTT for long sequences."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    hidden = None
    total_loss = 0

    for i in range(0, len(data) - bptt_len, bptt_len):
        # Get chunk
        inputs = data[i:i + bptt_len].to(device)
        targets = data[i + 1:i + 1 + bptt_len].to(device)

        # CRITICAL: Detach hidden state from previous chunk's computation graph
        # This stops gradients from flowing back more than bptt_len steps
        if hidden is not None:
            hidden = tuple(h.detach() for h in hidden)

        output, hidden = model(inputs.unsqueeze(0), hidden)
        loss = criterion(output.view(-1, output.size(-1)), targets.view(-1))

        optimizer.zero_grad()
        loss.backward()  # Gradients only flow back bptt_len steps
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss

# NOTE: hidden.detach() is the key line. Without it, PyTorch would try to
# backprop through the entire sequence history, causing OOM errors.
```

### Memory Estimation for LSTM Models

```python
def estimate_lstm_memory(vocab_size, embedding_dim, hidden_size, num_layers,
                         bidirectional, batch_size, seq_len, dtype_bytes=4):
    """
    Estimate GPU memory for LSTM training.

    Components:
    1. Embedding table: vocab_size × embedding_dim
    2. LSTM parameters: 4 × (input + hidden) × hidden × num_layers × directions
    3. Hidden states: batch × seq × hidden × directions (for backprop)
    4. Cell states: same as hidden states
    5. Gradients: ~same as parameters
    6. Optimizer states (Adam): 2× parameters (momentum + variance)
    """
    directions = 2 if bidirectional else 1

    # Embedding
    embed_params = vocab_size * embedding_dim

    # LSTM parameters (4 gates × (input + hidden) × hidden per layer)
    lstm_params_layer0 = 4 * (embedding_dim + hidden_size) * hidden_size * directions
    lstm_params_other = 4 * (hidden_size * directions + hidden_size) * hidden_size * directions
    lstm_params = lstm_params_layer0 + max(0, num_layers - 1) * lstm_params_other

    # Classifier head (approximate)
    classifier_params = hidden_size * directions * 2  # small

    total_params = embed_params + lstm_params + classifier_params

    # Activations for backprop (hidden + cell states at each step)
    activations = batch_size * seq_len * hidden_size * directions * 2  # h and c

    # Total memory
    params_memory = total_params * dtype_bytes
    gradient_memory = total_params * dtype_bytes
    optimizer_memory = total_params * dtype_bytes * 2  # Adam m and v
    activation_memory = activations * dtype_bytes

    total_mb = (params_memory + gradient_memory + optimizer_memory + activation_memory) / (1024**2)

    print(f"LSTM Memory Estimation:")
    print(f"  Parameters:   {total_params:>12,} ({params_memory/1024**2:.1f} MB)")
    print(f"  Gradients:    {' ':>12}  ({gradient_memory/1024**2:.1f} MB)")
    print(f"  Adam states:  {' ':>12}  ({optimizer_memory/1024**2:.1f} MB)")
    print(f"  Activations:  {' ':>12}  ({activation_memory/1024**2:.1f} MB)")
    print(f"  TOTAL:        {' ':>12}  ({total_mb:.1f} MB)")
    print(f"\n  Fits on: {'Any GPU' if total_mb < 2000 else '8GB+ GPU' if total_mb < 6000 else '16GB+ GPU'}")

    return total_mb

# Example: Our BiLSTM classifier
estimate_lstm_memory(
    vocab_size=10000, embedding_dim=128, hidden_size=128,
    num_layers=2, bidirectional=True, batch_size=32, seq_len=100
)

# Comparison: Larger model
print()
estimate_lstm_memory(
    vocab_size=50000, embedding_dim=256, hidden_size=512,
    num_layers=3, bidirectional=True, batch_size=64, seq_len=200
)
```

### Debugging Checklist: LSTM Training Not Converging

If your LSTM loss plateaus or diverges, work through this list:

1. **Check gradient norms**: Add `torch.nn.utils.clip_grad_norm_` and print the returned norm. If norms are >100, gradients are exploding. If <1e-6, they are vanishing.
2. **Learning rate**: Start with 1e-3 for Adam. If loss oscillates, reduce to 1e-4. If loss is flat, try 3e-3.
3. **Embedding scale**: Verify embeddings are initialized with reasonable variance (~0.01–0.1). Large embeddings can destabilize LSTM gates.
4. **Gate saturation**: Monitor forget/input gate values. If all gates are near 0 or 1 for all inputs, the model has saturated and cannot learn. Reduce learning rate or re-initialize.
5. **Sequence length vs. batch size**: Very long sequences with small batches cause noisy gradients. Try truncating or increasing batch size.
6. **Data leakage**: Ensure train/val split happened *before* any preprocessing that touches the full dataset.
7. **Label balance**: If 95% of labels are one class, the model will predict that class for everything. Use weighted loss or oversample the minority class.

### When to Use Transformers

1. **Most NLP tasks**: They're just better
2. **Pre-training available**: BERT, GPT, etc.
3. **Parallelizable training**: You have GPU resources
4. **Long-range dependencies matter**: Document-level understanding

### ✅ Checkpoint: After Part 5

You should now be able to answer:
1. Why does LSTM have O(1) memory per step while transformer has O(n²)?
2. What does `hidden.detach()` do in truncated BPTT and why is it necessary?
3. Estimate the GPU memory needed for a 2-layer BiLSTM with hidden_size=256, batch_size=32, seq_len=100.
4. When would you choose LSTM over transformer despite transformers being "better"?

If you can't answer all four, re-read Part 5 before continuing.

### The Historical Context

```
Timeline of Sequence Models:

1990s: RNNs introduced (struggle with long sequences)
   ↓
1997: LSTM invented (solves vanishing gradient)
   ↓
2014: GRU invented (simpler alternative to LSTM)
   ↓
2014: Attention mechanism (helps with long sequences)
   ↓
2017: Transformer ("Attention Is All You Need")
   ↓
2018+: BERT, GPT, etc. dominate NLP

RNNs are historical context, not the future.
But understanding them makes transformers intuitive.
```

---

## Part 6: Evaluating Your Sequence Model

Building a model is only half the work — you need to know whether it actually works. Here is a minimal evaluation template for sequence classifiers.

### Classification Metrics

```python
import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_model(model, data_loader, device='cpu'):
    """
    Evaluate a trained classifier and return detailed metrics.

    Returns predictions, labels, and a classification report.
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch_y.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Classification report: precision, recall, F1 per class
    print("Classification Report:")
    print(classification_report(
        all_labels, all_preds,
        target_names=["Negative", "Positive"]
    ))

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print(f"Confusion Matrix:\n{cm}")
    print(f"  True Negatives:  {cm[0,0]}")
    print(f"  False Positives: {cm[0,1]}")
    print(f"  False Negatives: {cm[1,0]}")
    print(f"  True Positives:  {cm[1,1]}")

    return all_preds, all_labels

# Wire evaluation into training — run this AFTER train_model() completes:
print("\n" + "="*60)
print("FINAL EVALUATION ON VALIDATION SET")
print("="*60)
evaluate_model(model, val_loader)

# NOTE: On the synthetic toy data above, accuracy will be near 100% (memorized).
# On a real dataset (IMDB), expect BiLSTM: ~87-90%, TF-IDF+LR baseline: ~85-87%.
# The gap matters — if BiLSTM doesn't beat the baseline, simplify.
```

### What to Look For

| Metric | What It Tells You | When It Matters Most |
|--------|-------------------|---------------------|
| **Accuracy** | Overall correctness | Balanced datasets |
| **Precision** | Of predicted positives, how many are correct | When false positives are costly (spam filter) |
| **Recall** | Of actual positives, how many were found | When false negatives are costly (medical) |
| **F1** | Harmonic mean of precision and recall | Imbalanced datasets |

### Common Failure Modes for Sequence Models

1. **Overfitting on short sequences**: Model memorizes token combinations instead of learning sequential patterns. Check: compare train vs. val accuracy over epochs.
2. **Padding dominance**: If most tokens are `<PAD>`, the model may learn to classify based on sequence length rather than content. Check: use attention visualization to verify the model focuses on real tokens.
3. **Sentiment keyword shortcut**: The model may learn "terrible" = negative without understanding negation ("not terrible"). Check: test with adversarial examples like "not bad at all."

---

## Manager's Summary

### What This Means for Your Projects

**RNNs/LSTMs**:
- Process sequences sequentially (slow training)
- Good for real-time/streaming applications
- Smaller models, lower resource requirements
- Declining use in modern NLP

**Practical recommendation**: Unless you have specific constraints (mobile, streaming), use transformers. They're better for almost everything.

### Resource Comparison (Approximate)

| Model | Parameters | Relative Training Cost | IMDB Accuracy Range |
|-------|-----------|----------------------|---------------------|
| LSTM (1 layer) | ~500K–1M | Low (single GPU, minutes–hours) | ~85–88% |
| BiLSTM (2 layer) | ~2–4M | Moderate (single GPU, hours) | ~87–90% |
| BERT-base (fine-tuned) | ~110M | Higher (single GPU, hours for fine-tuning) | ~93–95% |

*Numbers are approximate ranges from published benchmarks and common practitioner experience. Actual results depend on hyperparameters, preprocessing, and hardware. BERT accuracy reflects fine-tuning a pre-trained checkpoint, not training from scratch.*

### Questions to Ask Your Team

1. "Why LSTM instead of a transformer?"
2. "What's our maximum sequence length?"
3. "Do we need real-time processing?"
4. "Have we tried pre-trained models?"
5. "What's the gradient clipping value?"

---

## Interview Preparation

### Where This Knowledge Maps to Job Roles

| Blog Section | ML Engineer | Data Scientist | AI Architect | Engineering Manager |
|---|---|---|---|---|
| Part 1: RNN Mechanics | Implement custom RNN layers, debug gradient flow | Understand sequential feature extraction | Choose RNN vs transformer architecture | Understand why sequential processing is slower |
| Part 2: Vanishing Gradients | Diagnose training failures, implement gradient clipping | Know when model can't learn long dependencies | Design gradient-friendly architectures | Ask "what's our effective context window?" |
| Part 3: LSTM/GRU | Implement gates, tune forget bias, choose LSTM vs GRU | Use LSTM for time-series, text classification | Select architecture for resource constraints | Compare LSTM vs transformer resource costs |
| Part 4: BiLSTM Classifier | Build packed-sequence pipelines, attention pooling | Evaluate per-class metrics, compare baselines | Design end-to-end NLP pipeline | Ask "does BiLSTM beat the TF-IDF baseline?" |
| Part 5: Architecture Decisions | Estimate memory, implement truncated BPTT | Profile latency vs accuracy tradeoffs | Make build-vs-buy decisions, migration plans | Budget GPU resources, plan LSTM→transformer migration |
| Part 6: Evaluation | Wire evaluation into CI/CD, monitor per-class metrics | Identify failure modes, design test sets | Set quality gates for deployment | Require baseline comparison before deployment |

### Likely Questions

**Q: What's the vanishing gradient problem in RNNs?**
A: During backpropagation through time, gradients are multiplied at each step. If these multiplications are < 1, gradients shrink exponentially, meaning early inputs barely affect learning. LSTMs solve this with a cell state that provides a direct path for gradients.

**Q: How do LSTM gates work?**
A: LSTMs have three gates: Forget (what to remove from cell state), Input (what to add), and Output (what to expose). Each gate is a sigmoid layer (0-1) that controls information flow. The cell state acts as a highway for information, with gates as on-ramps and off-ramps.

**Q: What's the difference between LSTM and GRU?**
A: GRU combines forget and input gates into an "update gate" and merges cell/hidden states. It has fewer parameters (2 gates vs 3), trains faster, but may have slightly less capacity. In practice, performance is similar for most tasks.

**Q: Why use bidirectional RNNs?**
A: Forward RNNs only see past context. Many tasks require future context too ("The movie was not as bad as expected" - understanding "not" requires seeing "bad" later). Bidirectional processes both directions and concatenates representations.

**Q: When would you choose LSTM over Transformer?**
A: LSTMs when: (1) resource-constrained (mobile), (2) real-time streaming (can't wait for full sequence), (3) very long sequences where transformer attention is prohibitive, (4) simple tasks where transformer is overkill. Otherwise, transformers usually win.

**Q: Explain Backpropagation Through Time (BPTT).**
A: BPTT unrolls the RNN across time steps and applies the chain rule. The gradient ∂L/∂hₜ involves a product of Jacobians: Π_{k=t}^{T-1} [diag(tanh'(z)) × Wₕₕ]. If the spectral radius of this product < 1, gradients vanish exponentially; if > 1, they explode. Truncated BPTT limits backpropagation to a fixed window (e.g., 35 steps) to manage this.

**Q: What is gradient clipping and how does it work?**
A: Gradient clipping rescales the gradient vector if its norm exceeds a threshold. `clip_grad_norm_(params, max_norm=1.0)` computes the global gradient norm across all parameters and scales it down to `max_norm` if it's larger. This prevents exploding gradients but does NOT fix vanishing gradients — it only caps the maximum, not the minimum.

**Q: What's the purpose of `pack_padded_sequence` in PyTorch?**
A: It tells the LSTM to skip padding tokens during forward computation. Without packing, the LSTM processes padding tokens wastefully and their hidden state updates can dilute the real signal. Packing requires knowing actual sequence lengths and sorting (or using `enforce_sorted=False`). It's essential for efficient batched training with variable-length sequences.

---

## Where RNNs/LSTMs Still Appear in Industry

While transformers dominate headlines, RNN-family models remain relevant in specific niches:

- **On-device ML (mobile, IoT)**: LSTMs power keyboard prediction (e.g., GBoard), wake-word detection ("Hey Siri"), and on-device speech recognition where model size and latency matter more than peak accuracy.
- **Time-series forecasting**: Financial modeling, sensor data, and anomaly detection often use LSTMs because sequences are very long and attention-based models may not justify the compute overhead.
- **Streaming / real-time systems**: When you cannot buffer the full input (live audio transcription, real-time trading signals), RNNs process one step at a time without needing the full sequence.
- **Legacy systems**: Many production NLP systems built before 2019 still run LSTM-based models. Understanding them is essential for maintenance, debugging, and migration planning.

**Career implication**: Even if you build new systems with transformers, you will encounter LSTM-based code in production. The ability to read, debug, and migrate these models is a practical skill that distinguishes junior from mid-level ML engineers.

---

## Exercises (Do These)

1. **Gradient flow**: Implement gradient checking for your LSTM. Verify gradients flow better than basic RNN.

2. **Gate visualization**: Track gate activations during training. Do patterns emerge for different sentiment words?

3. **Sequence length**: Test your BiLSTM on sequences of length 50, 100, 200, 500. Where does it break down?

4. **GRU comparison**: Implement a GRU model. Compare training speed and final accuracy.

5. **Pre-trained embeddings**: Initialize your embedding layer with GloVe. Does it improve convergence?

---

## What's Next

You now have:
- Understanding of why sequence order matters
- RNN implementation and intuition
- Knowledge of the vanishing gradient problem
- LSTM gates and mechanisms
- A working bidirectional sentiment classifier with evaluation

**Blog 8** introduces the **attention mechanism**—the key innovation that makes transformers possible. You'll implement attention from scratch and see how it solves the long-range dependency problem more elegantly than LSTMs.

**[→ Blog 8: Attention Mechanism — Focus on What Matters](#)**

---

---

## Self-Assessment: What This Blog Does Well and Where It Falls Short

### Does Well

- **Conceptual progression**: Builds a clear narrative from "why order matters" through RNN limitations to the LSTM solution — the vanishing gradient explanation with simulation, matrix demonstration, and BPTT chain rule derivation is comprehensive
- **From-scratch implementations**: RNN, LSTM cell, and GRU cell are all implemented in NumPy with clear comments, making the mechanisms transparent
- **BPTT chain rule**: The mathematical derivation of why gradients vanish (product of Jacobians, spectral radius analysis) closes the gap between intuition and formal understanding
- **Practical BiLSTM classifier**: Attention-pooled BiLSTM with training loop, gradient clipping, packed sequences, baseline comparison, and wired evaluation
- **GRU coverage**: Architecture, formulas, parameter comparison, and when-to-use guidance — readers can make informed LSTM vs GRU decisions
- **Engineering depth**: Memory estimation, truncated BPTT, packed sequences, and forget gate bias initialization are production-relevant patterns
- **Honest positioning**: Upfront that RNNs/LSTMs are historical context, with specific niches where they remain relevant

### Falls Short

- **No real dataset integration**: The training example uses a synthetic 12-sentence dataset repeated 80 times — readers must substitute IMDB or SST-2 themselves. Instructions are provided but not automated.
- **Streaming/real-time use case is theoretical**: The blog argues RNNs suit streaming data but never demonstrates token-by-token inference
- **No teacher forcing discussion**: A standard technique for training sequence-to-sequence models with RNNs, mentioned nowhere
- **Attention mechanism in BiLSTM is simple**: Single-layer attention pooling is shown but the connection to the full attention mechanism (Blog 8) is not bridged

### Reader Self-Check

| Criteria | Excellent | Good | Needs Work |
|----------|-----------|------|------------|
| **Sequence Intuition** | Can explain why word order matters for sentiment | Understands sequences carry temporal info | Views text as bag of words |
| **RNN Mechanics** | Can implement forward pass, explain tanh choice, write BPTT chain rule | Understands recurrence conceptually | Cannot trace information flow |
| **Vanishing Gradients** | Can derive BPTT gradient product, diagnose with spectral radius, implement clipping + truncated BPTT | Knows the problem exists and LSTM solves it | Unaware of training issues |
| **LSTM/GRU Architecture** | Can implement all gates, explain forget bias init, compare LSTM vs GRU parameter counts | Understands forget/input/output gates | Cannot distinguish LSTM from RNN |
| **Bidirectional Models** | Implements BiLSTM with packed sequences, attention pooling, and baseline comparison | Understands concept of future context | Only uses unidirectional |
| **Engineering** | Estimates memory, uses truncated BPTT, profiles latency vs transformer | Knows LSTM is smaller than transformer | No resource awareness |

---

## Architect Sanity Checks

### Check 1: Architecture Understanding
**Question**: Why did LSTMs replace vanilla RNNs for most NLP tasks?
**Answer: YES** — The blog explains how LSTM gate mechanisms (input, forget, output gates) selectively control information flow through memory cells, preventing vanishing gradients through additive cell state updates. The gradient flow simulation in Part 2 concretely shows how vanilla RNN gradients decay exponentially while LSTM cell-state gradients remain viable.

### Check 2: Implementation Capability
**Question**: Can you implement a sentiment classifier with LSTM?
**Answer: YES** — The blog provides a complete BiLSTM classifier with attention pooling, packed sequences for efficiency, gradient clipping, baseline comparison (TF-IDF + LogReg), and wired evaluation with per-class metrics. The training data is synthetic (acknowledged as a limitation), but the architecture, training loop, and evaluation pipeline are production-pattern quality. Instructions for IMDB substitution are provided.

### Check 3: Debugging Skills
**Question**: Your LSTM loss is stuck. What do you check?
**Answer: YES** — The blog includes a 7-point debugging checklist (gradient norms, learning rate, embedding scale, gate saturation, sequence length vs. batch size, data leakage, label balance) in Part 5, plus gradient clipping in the training loop and failure modes in Part 6. A reader following this checklist could systematically diagnose most common LSTM training issues.

---

*Questions? Found an error? Comments are open. Technical corrections get priority.*
