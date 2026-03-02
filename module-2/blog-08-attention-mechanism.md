# Blog 8: Attention Mechanism
## Focus on What Matters

**Reading time:** 60–75 minutes
**Coding time:** 90–120 minutes
**Total investment:** 2.5–3 hours

---

## Prerequisites and Reading Guide

**Required background (from this series):**
- Blog 4: Neural networks fundamentals (forward pass, backpropagation)
- Blog 5: PyTorch basics (tensors, nn.Module, autograd)
- Blog 7: RNNs and LSTMs (hidden states, sequence processing, encoder-decoder)

**Required knowledge:**
- Matrix multiplication and transpose operations
- Softmax function and its properties
- Basic Python and NumPy fluency

**How to read this blog:**
- Parts 1-3 build the concept from scratch in NumPy — work through the code by hand
- Part 4 covers attention variants — understand the taxonomy before moving on
- Part 5 provides the PyTorch production implementation — compare it to your NumPy version
- Part 6 covers interpretability — this is where intuition solidifies
- Part 7 covers performance and inference considerations critical for production

---

## What This Blog Does NOT Cover

- **Full Transformer architecture** — that is Blog 9. This blog covers only the attention mechanism itself.
- **Positional encodings** — we discuss how they interact with attention, but the full treatment is in Blog 9.
- **Training attention models** — all implementations here use random (untrained) weights for demonstration. Training is covered in Blogs 9 and 23.
- **Sparse attention and long-context methods** — we mention FlashAttention and linear attention but do not implement them.
- **Attention in vision models (ViT, DETR)** — this blog focuses on sequence/text attention. Vision attention is covered in Blog 21.

---

## What You'll Walk Away With

By the end of this blog, you will:

1. **Understand** the Query-Key-Value framework that powers all modern transformers
2. **Implement** scaled dot-product attention from scratch
3. **Build** self-attention and cross-attention mechanisms
4. **Visualize** attention patterns to see what models "focus on"
5. **Explain** why attention solves problems that RNNs couldn't
6. **Know** how KV-caching speeds up inference and why it matters

This blog is the bridge between LSTMs and transformers. Master attention, and transformers become intuitive.

---

## Why Attention?

### The RNN Bottleneck

In Blog 7, we saw that RNNs compress all past information into a fixed-size hidden state:

```
"The cat sat on the mat because it was comfortable"

After processing:
h = [0.2, -0.1, 0.5, ...] <- ALL context squeezed into this vector
                            "it" needs to know "it" refers to "cat" or "mat"?
                            This information may be lost!
```

The problem: **fixed-size bottleneck** can't capture all relevant information for long sequences.

### The Attention Solution

Instead of compressing everything, attention **directly looks at all previous words** and decides which ones are relevant:

```
Query: "it" is asking "what am I referring to?"
Keys: All words say "here's what I am"
Values: All words provide their information

"it" attends to:
  "cat"  -> high attention (animate, could be comfortable)
  "mat"  -> high attention (could be comfortable)
  "sat"  -> low attention (not relevant to "it")
  "the"  -> very low attention (function word)
```

Attention is **selective memory access** rather than sequential compression.

### Historical Context

Attention was introduced by Bahdanau et al. (2014) for machine translation, originally as an add-on to RNN encoder-decoder models. The key insight: instead of forcing the encoder to compress everything into one vector, let the decoder look back at all encoder states. Vaswani et al. (2017) took this further with "Attention Is All You Need," removing RNNs entirely and using only attention — creating the Transformer.

---

## Part 1: The Query-Key-Value Framework

### Intuition

Think of attention like a **search engine**:

| Component | Search Analogy | What It Does |
|-----------|---------------|--------------|
| **Query (Q)** | Search query | "What am I looking for?" |
| **Key (K)** | Document titles | "What does each word offer?" |
| **Value (V)** | Document content | "What information to return?" |

The process:
1. Compare query to all keys (compute relevance scores)
2. Convert scores to weights (softmax)
3. Return weighted sum of values

### Mathematical Formulation

```
Attention(Q, K, V) = softmax(Q . K^T / sqrt(d_k)) . V
```

Where:
- `Q . K^T`: Similarity between query and all keys
- `sqrt(d_k)`: Scaling factor (prevents large values before softmax)
- `softmax`: Converts scores to probabilities (weights)
- `. V`: Weighted combination of values

**Why sqrt(d_k)?** When d_k is large, the dot products Q.K^T tend to grow in magnitude (variance proportional to d_k), pushing the softmax into regions where it has extremely small gradients. Dividing by sqrt(d_k) keeps the variance of the dot products at approximately 1, regardless of dimension. This is not optional — without it, training becomes unstable for any non-trivial d_k.

### Implementation from Scratch

```python
import numpy as np
import matplotlib.pyplot as plt

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Scaled dot-product attention.

    Args:
        Q: Queries, shape (batch, seq_len_q, d_k)
        K: Keys, shape (batch, seq_len_k, d_k)
        V: Values, shape (batch, seq_len_k, d_v)
        mask: Optional mask, shape (batch, seq_len_q, seq_len_k)

    Returns:
        output: Attended values, shape (batch, seq_len_q, d_v)
        attention_weights: Attention distribution, shape (batch, seq_len_q, seq_len_k)
    """
    d_k = K.shape[-1]

    # Step 1: Compute attention scores
    # Q @ K^T gives similarity between each query and all keys
    scores = np.matmul(Q, K.transpose(0, 2, 1))  # (batch, seq_q, seq_k)

    # Step 2: Scale by sqrt(d_k)
    # Without scaling, large d_k leads to large dot products,
    # pushing softmax into regions with tiny gradients
    scores = scores / np.sqrt(d_k)

    # Step 3: Apply mask (optional)
    # Used for padding and causal (autoregressive) attention
    if mask is not None:
        scores = np.where(mask == 0, -1e9, scores)

    # Step 4: Softmax to get attention weights
    # Each query has a probability distribution over all keys
    attention_weights = softmax(scores, axis=-1)

    # Step 5: Weighted sum of values
    # Each query gets a weighted combination of all values
    output = np.matmul(attention_weights, V)  # (batch, seq_q, d_v)

    return output, attention_weights


def softmax(x, axis=-1):
    """Numerically stable softmax."""
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


# Example: Simple attention
rng = np.random.default_rng(42)

# Single sequence, 5 positions, dimension 4
batch_size = 1
seq_len = 5
d_model = 4

# In self-attention, Q, K, V come from the same sequence
X = rng.standard_normal((batch_size, seq_len, d_model))

# Simple case: Q, K, V are all the same (self-attention without projections)
Q = K = V = X

output, attention = scaled_dot_product_attention(Q, K, V)

print(f"Input shape: {X.shape}")
print(f"Output shape: {output.shape}")
print(f"Attention weights shape: {attention.shape}")

print(f"\nAttention weights (each row sums to 1):")
print(attention[0].round(3))
print(f"Row sums: {attention[0].sum(axis=1)}")
```

### Visualizing Attention

```python
def visualize_attention(attention_weights, tokens_q, tokens_k, title="Attention"):
    """Visualize attention as a heatmap."""
    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(attention_weights, cmap='Blues')

    ax.set_xticks(range(len(tokens_k)))
    ax.set_xticklabels(tokens_k, rotation=45, ha='right')
    ax.set_yticks(range(len(tokens_q)))
    ax.set_yticklabels(tokens_q)

    ax.set_xlabel('Keys (attending to)')
    ax.set_ylabel('Queries (from)')
    ax.set_title(title)

    # Add colorbar
    plt.colorbar(im, ax=ax)

    # Add value annotations
    for i in range(len(tokens_q)):
        for j in range(len(tokens_k)):
            text = ax.text(j, i, f'{attention_weights[i, j]:.2f}',
                          ha='center', va='center',
                          color='white' if attention_weights[i, j] > 0.5 else 'black',
                          fontsize=8)

    plt.tight_layout()
    return fig


# Example with real words
tokens = ['The', 'cat', 'sat', 'on', 'the']

# IMPORTANT: These embeddings are random, not learned.
# The attention patterns below do NOT reflect real linguistic relationships.
# They only demonstrate the mechanics of how attention scores are computed.
rng = np.random.default_rng(42)
embeddings = rng.standard_normal((1, 5, 64))
Q = K = V = embeddings

output, attention = scaled_dot_product_attention(Q, K, V)

fig = visualize_attention(attention[0], tokens, tokens,
    "Self-Attention Weights (random embeddings — not linguistically meaningful)")
plt.savefig('attention_heatmap.png', dpi=150)
plt.show()
```

### Attention Scoring Functions: Dot-Product vs Additive

```python
"""
There are multiple ways to compute attention scores. The choice matters:

1. SCALED DOT-PRODUCT (Vaswani et al., 2017) — used in Transformers
   score(Q, K) = Q · K^T / sqrt(d_k)

   Pros: Fast (matrix multiplication), parallelizable on GPU
   Cons: Assumes Q and K are in the same vector space
   Complexity: O(n * d_k) per query-key pair

2. ADDITIVE / BAHDANAU (Bahdanau et al., 2014) — used in original attention
   score(Q, K) = v^T · tanh(W_q · Q + W_k · K)

   Pros: Can compare Q and K in different spaces (no shared dimension needed)
   Cons: Slower (requires learned weight matrices and tanh), not easily parallelized
   Complexity: O(d_attn) per query-key pair, where d_attn is the attention hidden dim

3. MULTIPLICATIVE / LUONG (Luong et al., 2015)
   score(Q, K) = Q^T · W · K  (general form)
   score(Q, K) = Q^T · K      (dot form, special case of scaled dot-product)

   Pros: More expressive than pure dot-product (learnable W)
   Cons: O(d_q × d_k) parameters for general form

WHY TRANSFORMERS USE SCALED DOT-PRODUCT:
- Matrix multiplication is the most hardware-optimized operation on GPUs
- With learned Q/K projections (W_Q, W_K), dot-product attention has the same
  expressiveness as additive — the projections subsume the role of W_q, W_k
- Additive attention was designed for RNN encoder-decoder where Q and K had
  different dimensionalities. In transformers, Q and K always share d_k.

When you still see additive attention:
- Legacy models (pre-2017 seq2seq)
- Cross-modal attention where Q and K have genuinely different dimensions
- Some speech recognition systems
"""
```

### ✅ Checkpoint: After Part 1

You should now be able to answer:
1. What do Q, K, and V represent in the attention framework?
2. Why does dividing by sqrt(d_k) prevent softmax saturation?
3. What is the difference between scaled dot-product and additive (Bahdanau) attention?
4. Why do transformers use dot-product attention instead of additive?

If you can't answer all four, re-read Part 1 before continuing.

---

## Part 2: Self-Attention with Learned Projections

### Why Projections?

In real transformers, Q, K, V aren't just the input — they're **learned projections**:

```python
Q = X @ W_Q  # Project input to query space
K = X @ W_K  # Project input to key space
V = X @ W_V  # Project input to value space
```

This lets the model learn:
- **What to query for**: W_Q learns what information each position needs
- **What to advertise**: W_K learns what each position offers
- **What to provide**: W_V learns what information to return

Without projections, self-attention on raw embeddings can only compute similarity between the same vectors. With projections, the model can learn asymmetric relationships: "I'm a pronoun looking for a noun" (query) is different from "I'm a noun that can be referred to" (key).

### Implementation

```python
import numpy as np

class SelfAttention:
    """
    Self-attention with learned Q, K, V projections.

    NOTE: Weights are randomly initialized (not trained).
    Output patterns are NOT linguistically meaningful.
    This class demonstrates the forward-pass mechanics only.
    """

    def __init__(self, d_model, d_k=None, d_v=None, rng=None):
        """
        Args:
            d_model: Input/output dimension
            d_k: Key/query dimension (default: d_model)
            d_v: Value dimension (default: d_model)
            rng: NumPy random generator (default: creates new one)
        """
        self.d_model = d_model
        self.d_k = d_k or d_model
        self.d_v = d_v or d_model

        if rng is None:
            rng = np.random.default_rng()

        # Initialize projection matrices (Xavier-like initialization)
        scale = np.sqrt(2.0 / d_model)
        self.W_Q = rng.standard_normal((d_model, self.d_k)) * scale
        self.W_K = rng.standard_normal((d_model, self.d_k)) * scale
        self.W_V = rng.standard_normal((d_model, self.d_v)) * scale
        self.W_O = rng.standard_normal((self.d_v, d_model)) * scale

    def forward(self, X, mask=None):
        """
        Forward pass.

        Args:
            X: Input, shape (batch, seq_len, d_model)
            mask: Optional attention mask

        Returns:
            output: Attended representation, shape (batch, seq_len, d_model)
            attention: Attention weights, shape (batch, seq_len, seq_len)
        """
        # Project to Q, K, V
        Q = X @ self.W_Q  # (batch, seq, d_k)
        K = X @ self.W_K  # (batch, seq, d_k)
        V = X @ self.W_V  # (batch, seq, d_v)

        # Compute attention
        output, attention = scaled_dot_product_attention(Q, K, V, mask)

        # Project output
        output = output @ self.W_O  # (batch, seq, d_model)

        return output, attention


# Test
rng = np.random.default_rng(42)
attention_layer = SelfAttention(d_model=64, rng=rng)
X = rng.standard_normal((2, 10, 64))  # Batch of 2, sequence length 10

output, attention_weights = attention_layer.forward(X)

print(f"Input shape: {X.shape}")
print(f"Output shape: {output.shape}")
print(f"Attention shape: {attention_weights.shape}")
```

### ✅ Checkpoint: After Part 2

You should now be able to answer:
1. Why do we need learned W_Q, W_K, W_V projections instead of using raw embeddings?
2. What asymmetric relationships can projections learn that raw dot-product can't?
3. What is the output projection W_O for?
4. How many parameter matrices does a single self-attention layer have?

If you can't answer all four, re-read Part 2 before continuing.

---

## Part 3: Multi-Head Attention

### Why Multiple Heads?

Single attention can only focus on one type of relationship. Multi-head attention runs **multiple attention operations in parallel**, each learning different relationships:

```
Head 1: Learns syntactic relationships (subject-verb)
Head 2: Learns semantic relationships (pronoun-antecedent)
Head 3: Learns positional relationships (adjacent words)
...
```

Each head operates on a lower-dimensional subspace (d_model / num_heads), so the total computation is similar to single-head attention with full dimensionality. The benefit is specialization: different heads can attend to different relationship types simultaneously.

### Implementation

```python
import numpy as np

class MultiHeadAttention:
    """
    Multi-head attention mechanism.

    Runs multiple attention heads in parallel, each with its own projections.

    NOTE: Weights are randomly initialized. Attention patterns will NOT
    reflect real linguistic relationships. This demonstrates mechanics only.
    """

    def __init__(self, d_model, num_heads, rng=None):
        """
        Args:
            d_model: Model dimension (must be divisible by num_heads)
            num_heads: Number of attention heads
            rng: NumPy random generator
        """
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
            )

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        if rng is None:
            rng = np.random.default_rng()

        # All projections combined for efficiency
        scale = np.sqrt(2.0 / d_model)
        self.W_Q = rng.standard_normal((d_model, d_model)) * scale
        self.W_K = rng.standard_normal((d_model, d_model)) * scale
        self.W_V = rng.standard_normal((d_model, d_model)) * scale
        self.W_O = rng.standard_normal((d_model, d_model)) * scale

    def split_heads(self, x):
        """
        Split the last dimension into (num_heads, d_k).

        Input: (batch, seq_len, d_model)
        Output: (batch, num_heads, seq_len, d_k)
        """
        batch_size, seq_len, _ = x.shape
        x = x.reshape(batch_size, seq_len, self.num_heads, self.d_k)
        return x.transpose(0, 2, 1, 3)

    def combine_heads(self, x):
        """
        Combine heads back together.

        Input: (batch, num_heads, seq_len, d_k)
        Output: (batch, seq_len, d_model)
        """
        batch_size, _, seq_len, _ = x.shape
        x = x.transpose(0, 2, 1, 3)
        return x.reshape(batch_size, seq_len, self.d_model)

    def forward(self, X, mask=None):
        """
        Forward pass through multi-head attention.

        Args:
            X: Input, shape (batch, seq_len, d_model)
            mask: Optional attention mask

        Returns:
            output: Attended representation
            attention: Attention weights from each head
        """
        batch_size = X.shape[0]

        # Linear projections
        Q = X @ self.W_Q
        K = X @ self.W_K
        V = X @ self.W_V

        # Split into heads
        Q = self.split_heads(Q)  # (batch, heads, seq, d_k)
        K = self.split_heads(K)
        V = self.split_heads(V)

        # Scaled dot-product attention for each head
        d_k = self.d_k
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(d_k)

        if mask is not None:
            # Expand mask for heads: (batch, 1, seq, seq)
            if mask.ndim == 3:
                mask = np.expand_dims(mask, 1)
            scores = np.where(mask == 0, -1e9, scores)

        attention = softmax(scores, axis=-1)
        context = np.matmul(attention, V)

        # Combine heads
        context = self.combine_heads(context)

        # Final projection
        output = context @ self.W_O

        return output, attention


# Test multi-head attention
rng = np.random.default_rng(42)
mha = MultiHeadAttention(d_model=64, num_heads=8, rng=rng)
X = rng.standard_normal((2, 10, 64))

output, attention = mha.forward(X)

print(f"Input shape: {X.shape}")
print(f"Output shape: {output.shape}")
print(f"Attention shape: {attention.shape}")  # (batch, heads, seq, seq)
```

### Visualizing Multiple Heads

```python
def visualize_multihead_attention(attention_weights, tokens, num_heads_to_show=4):
    """
    Visualize attention patterns from multiple heads.

    NOTE: With random (untrained) weights, head patterns will look similar
    and will NOT show specialized linguistic behavior. In trained models,
    different heads learn to specialize in different relationship types.
    """
    num_heads = min(attention_weights.shape[1], num_heads_to_show)

    fig, axes = plt.subplots(1, num_heads, figsize=(4*num_heads, 4))

    for head_idx in range(num_heads):
        ax = axes[head_idx] if num_heads > 1 else axes
        weights = attention_weights[0, head_idx]

        im = ax.imshow(weights, cmap='Blues')
        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha='right', fontsize=8)
        ax.set_yticks(range(len(tokens)))
        ax.set_yticklabels(tokens, fontsize=8)
        ax.set_title(f'Head {head_idx + 1}')

    plt.suptitle('Multi-Head Attention Patterns (random weights — not linguistically meaningful)')
    plt.tight_layout()
    return fig


# Create attention with meaningful pattern
tokens = ['The', 'cat', 'sat', 'on', 'the', 'mat', '.']
rng = np.random.default_rng(42)
X = rng.standard_normal((1, 7, 64))

mha = MultiHeadAttention(d_model=64, num_heads=8, rng=rng)
_, attention = mha.forward(X)

fig = visualize_multihead_attention(attention, tokens, num_heads_to_show=4)
plt.savefig('multihead_attention.png', dpi=150)
plt.show()
```

### ✅ Checkpoint: After Part 3

You should now be able to answer:
1. Why does multi-head attention use d_k = d_model / num_heads per head?
2. How does split_heads reshape a (batch, seq, d_model) tensor into (batch, heads, seq, d_k)?
3. Why is multi-head attention computationally similar to single-head full-dimensional attention?
4. What different types of relationships might different heads specialize in?

If you can't answer all four, re-read Part 3 before continuing.

---

## Part 4: Attention Types

### Self-Attention vs Cross-Attention

```python
class CrossAttention:
    """
    Cross-attention: Queries from one sequence, Keys/Values from another.

    Used in:
    - Encoder-decoder models (decoder attends to encoder)
    - Vision-language models (text attends to image)
    - Retrieval-augmented generation (query attends to retrieved docs)

    NOTE: Weights are randomly initialized (untrained).
    """

    def __init__(self, d_model, num_heads, rng=None):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        if rng is None:
            rng = np.random.default_rng()

        scale = np.sqrt(2.0 / d_model)
        self.W_Q = rng.standard_normal((d_model, d_model)) * scale
        self.W_K = rng.standard_normal((d_model, d_model)) * scale
        self.W_V = rng.standard_normal((d_model, d_model)) * scale
        self.W_O = rng.standard_normal((d_model, d_model)) * scale

    def forward(self, X_query, X_context, mask=None):
        """
        Cross-attention forward pass.

        Args:
            X_query: Query source, shape (batch, seq_q, d_model)
            X_context: Key/Value source, shape (batch, seq_k, d_model)
            mask: Optional attention mask

        Returns:
            output: Attended representation
            attention: Attention weights
        """
        # Queries from X_query
        Q = X_query @ self.W_Q

        # Keys and Values from X_context
        K = X_context @ self.W_K
        V = X_context @ self.W_V

        # Compute attention (simplified: no head splitting for clarity)
        d_k = self.d_k
        scores = Q @ K.transpose(0, 2, 1) / np.sqrt(d_k)

        if mask is not None:
            scores = np.where(mask == 0, -1e9, scores)

        attention = softmax(scores, axis=-1)
        context = attention @ V

        output = context @ self.W_O

        return output, attention


# Example: Decoder attending to encoder
rng = np.random.default_rng(42)
encoder_output = rng.standard_normal((1, 20, 64))  # Source sequence
decoder_input = rng.standard_normal((1, 10, 64))    # Target sequence

cross_attn = CrossAttention(d_model=64, num_heads=8, rng=rng)
output, attention = cross_attn.forward(decoder_input, encoder_output)

print(f"Decoder input: {decoder_input.shape}")
print(f"Encoder output: {encoder_output.shape}")
print(f"Cross-attention output: {output.shape}")
print(f"Attention shape: {attention.shape}")  # (1, 10, 20) - decoder attends to encoder
```

### Causal (Masked) Attention

For autoregressive models (GPT), each position can only attend to previous positions:

```python
def create_causal_mask(seq_len):
    """
    Create causal attention mask.

    Position i can only attend to positions 0, 1, ..., i
    """
    mask = np.tril(np.ones((seq_len, seq_len)))
    return mask


def causal_attention(Q, K, V):
    """Attention with causal masking."""
    seq_len = Q.shape[1]
    mask = create_causal_mask(seq_len)

    return scaled_dot_product_attention(Q, K, V, mask)


# Visualize causal mask
mask = create_causal_mask(8)

plt.figure(figsize=(8, 6))
plt.imshow(mask, cmap='Blues')
plt.xticks(range(8), [f'Pos {i}' for i in range(8)])
plt.yticks(range(8), [f'Pos {i}' for i in range(8)])
plt.xlabel('Key positions (can attend to)')
plt.ylabel('Query positions (from)')
plt.title('Causal Attention Mask\n(1 = can attend, 0 = masked)')
plt.colorbar()
plt.savefig('causal_mask.png', dpi=150)
plt.show()

print("Causal mask ensures position 3 can only see positions 0, 1, 2, 3")
print("This enables autoregressive generation (predict next token)")
```

### ✅ Checkpoint: After Part 4

You should now be able to answer:
1. What's the structural difference between self-attention and cross-attention?
2. How does causal masking prevent "seeing the future" in autoregressive models?
3. Name three use cases for cross-attention.
4. Why is the causal mask a lower-triangular matrix?

If you can't answer all four, re-read Part 4 before continuing.

---

## Part 5: PyTorch Implementation

### Complete Attention Module

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    """
    Production-quality multi-head attention in PyTorch.
    """

    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
            )

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Linear projections
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

        # WHY attention dropout?
        # Without it, heads can co-adapt: head 3 always relies on head 1's output.
        # Dropout on attention weights forces each head to independently learn useful
        # patterns. It also regularizes attention distributions — preventing the model
        # from putting 100% weight on a single token (which is brittle to input noise).
        # Typical values: 0.1 for base models, 0.0 for inference.
        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x, batch_size):
        """Split into heads: (batch, seq, d_model) -> (batch, heads, seq, d_k)"""
        x = x.view(batch_size, -1, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def forward(self, query, key, value, mask=None):
        """
        Forward pass.

        Args:
            query: (batch, seq_q, d_model)
            key: (batch, seq_k, d_model)
            value: (batch, seq_k, d_model)
            mask: Optional (batch, 1, seq_q, seq_k) or (batch, 1, 1, seq_k)

        Returns:
            output: (batch, seq_q, d_model)
            attention_weights: (batch, num_heads, seq_q, seq_k)
        """
        batch_size = query.size(0)

        # Linear projections
        Q = self.W_Q(query)
        K = self.W_K(key)
        V = self.W_V(value)

        # Split into heads
        Q = self.split_heads(Q, batch_size)
        K = self.split_heads(K, batch_size)
        V = self.split_heads(V, batch_size)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        context = torch.matmul(attention_weights, V)

        # Combine heads
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # Final projection
        output = self.W_O(context)

        return output, attention_weights


# Test
mha = MultiHeadAttention(d_model=256, num_heads=8)
x = torch.randn(4, 32, 256)  # batch=4, seq=32, dim=256

output, attention = mha(x, x, x)  # Self-attention

print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
print(f"Attention shape: {attention.shape}")

# Count parameters
total_params = sum(p.numel() for p in mha.parameters())
print(f"Total parameters: {total_params:,}")
# Expected: 4 * (256*256 + 256) = 263,168
```

### Attention with Real Text

```python
from transformers import AutoTokenizer, AutoModel
import torch

# Load pre-trained BERT
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased', output_attentions=True)

# Tokenize a sentence
sentence = "The cat sat on the mat because it was comfortable."
inputs = tokenizer(sentence, return_tensors='pt')

# Get attention weights
with torch.no_grad():
    outputs = model(**inputs)

# outputs.attentions is a tuple of attention weights from each layer
# Each has shape (batch, num_heads, seq_len, seq_len)
attentions = outputs.attentions

print(f"Number of layers: {len(attentions)}")
print(f"Attention shape per layer: {attentions[0].shape}")

# Visualize attention for "it" token
tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
print(f"Tokens: {tokens}")

# Find "it" token position
it_position = tokens.index('it')
print(f"'it' is at position {it_position}")

# Get attention from "it" to all other tokens (last layer, average over heads)
last_layer_attention = attentions[-1][0].mean(dim=0)  # Average over heads
it_attention = last_layer_attention[it_position]

# Visualize
fig, ax = plt.subplots(figsize=(12, 4))
ax.bar(range(len(tokens)), it_attention.numpy())
ax.set_xticks(range(len(tokens)))
ax.set_xticklabels(tokens, rotation=45, ha='right')
ax.set_ylabel('Attention weight')
ax.set_title("What does 'it' attend to? (BERT last layer)")
plt.tight_layout()
plt.savefig('bert_attention.png', dpi=150)
plt.show()

print("\nInterpretation: 'it' should attend strongly to 'cat' or 'mat' (its referent)")
print("NOTE: Averaging over heads obscures per-head specialization.")
print("Individual heads may show clearer coreference patterns than the average.")
```

---

## Part 6: Attention Patterns and Interpretability

### Common Attention Patterns

```python
def analyze_attention_patterns(attention_weights, tokens):
    """Analyze common attention patterns."""
    seq_len = len(tokens)

    # Pattern 1: Diagonal (self-attention to same position)
    diagonal = np.diag(attention_weights)
    avg_self_attention = np.mean(diagonal)

    # Pattern 2: First token attention (often CLS or BOS)
    first_token_attention = np.mean(attention_weights[:, 0])

    # Pattern 3: Previous token attention
    prev_attention = np.mean([attention_weights[i, i-1] for i in range(1, seq_len)])

    # Pattern 4: Attention entropy (spread vs focused)
    entropies = []
    for row in attention_weights:
        # Avoid log(0)
        row_clipped = np.clip(row, 1e-10, None)
        entropy = -np.sum(row_clipped * np.log(row_clipped))
        entropies.append(entropy)
    avg_entropy = np.mean(entropies)
    max_entropy = np.log(seq_len)  # Maximum entropy = uniform distribution

    print("=== Attention Pattern Analysis ===")
    print(f"Self-attention (diagonal): {avg_self_attention:.3f}")
    print(f"First token attention: {first_token_attention:.3f}")
    print(f"Previous token attention: {prev_attention:.3f}")
    print(f"Attention entropy: {avg_entropy:.3f} / {max_entropy:.3f} (focused if low)")
    print(f"Entropy ratio: {avg_entropy / max_entropy:.2%} of maximum")

    return {
        'self_attention': avg_self_attention,
        'first_token': first_token_attention,
        'previous_token': prev_attention,
        'entropy': avg_entropy,
        'entropy_ratio': avg_entropy / max_entropy,
    }


# Analyze the BERT attention
analysis = analyze_attention_patterns(
    last_layer_attention.numpy(),
    tokens
)
```

### What Different Heads Learn

```python
def compare_attention_heads(attentions, tokens, layer_idx=-1):
    """Compare what different heads in a layer focus on."""
    layer_attention = attentions[layer_idx][0]  # (num_heads, seq, seq)
    num_heads = layer_attention.shape[0]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for head_idx in range(min(8, num_heads)):
        ax = axes[head_idx]
        head_attn = layer_attention[head_idx].numpy()

        ax.imshow(head_attn, cmap='Blues')
        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha='right', fontsize=6)
        ax.set_yticks(range(len(tokens)))
        ax.set_yticklabels(tokens, fontsize=6)
        ax.set_title(f'Head {head_idx + 1}')

    plt.suptitle(f'Layer {layer_idx}: Different Heads Learn Different Patterns')
    plt.tight_layout()
    plt.savefig('head_comparison.png', dpi=150)
    plt.show()


compare_attention_heads(attentions, tokens, layer_idx=-1)
```

### Caution: Attention is Not Explanation

A critical caveat for interpretability: **attention weights do not necessarily indicate importance.** Research (Jain & Wallace, 2019; Wiegreffe & Pinter, 2019) has shown that:

1. Alternative attention distributions can produce the same output
2. Attention weights do not always correlate with gradient-based feature importance
3. Heads can be "pruned" (zeroed out) without significant performance loss

Attention visualizations are useful for building intuition and debugging, but should not be treated as faithful explanations of model reasoning. For rigorous interpretability, combine attention analysis with gradient-based methods (e.g., integrated gradients) or probing classifiers.

---

## Part 7: Performance — KV-Caching and Efficient Attention

### The Quadratic Cost Problem

Attention computes pairwise scores between all positions, making its time and memory complexity **O(n^2)** in sequence length. This matters enormously in practice:

| Sequence Length | Attention Operations | Memory (float32, per head) |
|----------------|---------------------|---------------------------|
| 512            | 262K                | 1 MB                      |
| 2,048          | 4.2M                | 16 MB                     |
| 8,192          | 67M                 | 256 MB                    |
| 32,768         | 1.07B               | 4 GB                      |
| 128,000        | 16.4B               | 64 GB                     |

This is why context length is the primary scaling bottleneck and why efficient attention is an active research area.

### KV-Caching: Essential for Inference

During autoregressive generation (e.g., GPT producing text), the model generates one token at a time. Without caching, generating token N requires recomputing attention over all N-1 previous tokens from scratch. **KV-caching** stores the Key and Value tensors from previous steps so they are not recomputed:

```python
class CachedCausalAttention:
    """
    Causal attention with KV-caching for efficient autoregressive generation.

    Without cache: Generating N tokens requires O(N^3) total operations
    With cache:    Generating N tokens requires O(N^2) total operations

    NOTE: Uses random (untrained) weights for demonstration.
    """

    def __init__(self, d_model, num_heads, rng=None):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        if rng is None:
            rng = np.random.default_rng()

        scale = np.sqrt(2.0 / d_model)
        self.W_Q = rng.standard_normal((d_model, d_model)) * scale
        self.W_K = rng.standard_normal((d_model, d_model)) * scale
        self.W_V = rng.standard_normal((d_model, d_model)) * scale
        self.W_O = rng.standard_normal((d_model, d_model)) * scale

        # KV cache: stores Keys and Values from all previous positions
        self.k_cache = None  # Will be (batch, seq_so_far, d_model)
        self.v_cache = None

    def reset_cache(self):
        """Clear the cache (call at start of each new sequence)."""
        self.k_cache = None
        self.v_cache = None

    def forward(self, x_new):
        """
        Process one new token (or a batch of new tokens).

        Args:
            x_new: New input, shape (batch, new_seq_len, d_model)
                   During generation, new_seq_len is typically 1

        Returns:
            output: Attended output for new positions only
        """
        # Compute Q, K, V for new positions only
        Q_new = x_new @ self.W_Q
        K_new = x_new @ self.W_K
        V_new = x_new @ self.W_V

        # Append new K, V to cache
        if self.k_cache is None:
            self.k_cache = K_new
            self.v_cache = V_new
        else:
            self.k_cache = np.concatenate([self.k_cache, K_new], axis=1)
            self.v_cache = np.concatenate([self.v_cache, V_new], axis=1)

        # Attend: new queries attend to ALL cached keys/values
        d_k = self.d_k
        scores = Q_new @ self.k_cache.transpose(0, 2, 1) / np.sqrt(d_k)

        # Causal mask: new positions can attend to all cached positions
        # (cache only contains past + current, so no future leakage)
        attention = softmax(scores, axis=-1)
        context = attention @ self.v_cache

        output = context @ self.W_O
        return output, attention


# Demonstrate KV-caching
rng = np.random.default_rng(42)
cached_attn = CachedCausalAttention(d_model=64, num_heads=8, rng=rng)

# Simulate autoregressive generation: process one token at a time
tokens_so_far = rng.standard_normal((1, 1, 64))  # First token

print("Autoregressive generation with KV-cache:")
for step in range(5):
    new_token = rng.standard_normal((1, 1, 64))
    output, attn = cached_attn.forward(new_token)
    print(f"  Step {step}: cache size = {cached_attn.k_cache.shape[1]}, "
          f"attention shape = {attn.shape}")

# Key insight: at step N, we only compute Q for 1 new token,
# but attend to all N cached K/V pairs. Without cache, we'd
# recompute K and V for all N tokens every step.
```

### Memory Trade-off

KV-caching trades memory for compute. For a model with L layers, H heads, and d_k per head, the cache stores:

```
Cache memory per token = 2 * L * H * d_k * sizeof(dtype)

Example (LLaMA-7B-scale):
  L=32, H=32, d_k=128, float16
  = 2 * 32 * 32 * 128 * 2 bytes = 512 KB per token
  For 4096 context: ~2 GB of KV-cache alone
```

This is why long-context models require careful memory management, and techniques like **Grouped-Query Attention (GQA)** — where multiple query heads share fewer key/value heads — reduce the KV-cache size substantially. LLaMA 2 70B uses GQA with 8 KV heads shared across 64 query heads, reducing cache by 8x.

### Memory Estimation for Attention Layers

```python
def estimate_attention_memory(d_model, num_heads, seq_len, batch_size,
                              num_layers=1, kv_cache=False, dtype_bytes=4):
    """
    Estimate GPU memory for attention computation.

    Components:
    1. Parameters: W_Q, W_K, W_V, W_O per layer
    2. Attention matrix: batch × heads × seq × seq (the O(n²) part)
    3. KV-cache (if inference): 2 × batch × seq × d_model per layer
    4. Gradients + optimizer states (training only)
    """
    d_k = d_model // num_heads

    # Parameters per layer: 4 × (d_model² + d_model)  [weights + biases]
    params_per_layer = 4 * (d_model * d_model + d_model)
    total_params = params_per_layer * num_layers

    # Attention matrix: batch × heads × seq × seq × dtype
    attn_matrix = batch_size * num_heads * seq_len * seq_len * dtype_bytes
    attn_total = attn_matrix * num_layers

    # KV-cache (inference mode): 2 × layers × batch × seq × d_model × dtype
    kv_cache_mem = 0
    if kv_cache:
        kv_cache_mem = 2 * num_layers * batch_size * seq_len * d_model * dtype_bytes

    # Training overhead: gradients (~1x params) + Adam states (~2x params)
    training_overhead = total_params * dtype_bytes * 3 if not kv_cache else 0

    param_mem = total_params * dtype_bytes
    total_mb = (param_mem + attn_total + kv_cache_mem + training_overhead) / (1024**2)

    print(f"Attention Memory Estimation:")
    print(f"  Parameters:       {total_params:>12,} ({param_mem/1024**2:.1f} MB)")
    print(f"  Attention matrix: {' ':>12}  ({attn_total/1024**2:.1f} MB) [O(n²) — this dominates]")
    if kv_cache:
        print(f"  KV-cache:         {' ':>12}  ({kv_cache_mem/1024**2:.1f} MB)")
    else:
        print(f"  Training overhead:{' ':>12}  ({training_overhead/1024**2:.1f} MB)")
    print(f"  TOTAL:            {' ':>12}  ({total_mb:.1f} MB)")

    return total_mb

# Example 1: BERT-base training
print("=== BERT-base (training, batch=32) ===")
estimate_attention_memory(d_model=768, num_heads=12, seq_len=512, batch_size=32, num_layers=12)

# Example 2: GPT-2 inference with KV-cache
print("\n=== GPT-2 (inference with KV-cache, batch=1) ===")
estimate_attention_memory(d_model=768, num_heads=12, seq_len=1024, batch_size=1,
                          num_layers=12, kv_cache=True, dtype_bytes=2)  # float16

# Example 3: Why long context is expensive
print("\n=== Long context attention matrix only (seq=32K, 32 heads) ===")
for seq in [512, 2048, 8192, 32768]:
    mem = 1 * 32 * seq * seq * 2 / (1024**2)  # 1 batch, 32 heads, float16
    print(f"  seq={seq:>6}: attention matrix = {mem:>8.1f} MB")
```

### FlashAttention (Conceptual)

FlashAttention (Dao et al., 2022) does not change the attention math but changes how it is computed in hardware. Standard attention materializes the full N x N attention matrix in GPU HBM (slow memory). FlashAttention:

1. Tiles the computation into blocks that fit in SRAM (fast on-chip memory)
2. Computes softmax incrementally using the online softmax trick
3. Never materializes the full attention matrix

Result: same outputs, but 2-4x faster and uses O(N) memory instead of O(N^2). In PyTorch 2.0+, `torch.nn.functional.scaled_dot_product_attention` uses FlashAttention automatically when available.

### ✅ Checkpoint: After Part 7

You should now be able to answer:
1. What is the memory complexity of attention and why does it matter for long sequences?
2. How does KV-caching reduce autoregressive generation from O(N³) to O(N²)?
3. Estimate the KV-cache size for a 7B-parameter model with 4K context.
4. What does FlashAttention change about the computation (not the math)?

If you can't answer all four, re-read Part 7 before continuing.

---

## Manager's Summary

### Why Attention Matters

**Before attention**: Models compressed all information into fixed-size vectors, losing details for long sequences.

**With attention**: Models can directly access any part of the input, focusing on what's relevant for each output.

### Business Impact

| Capability | Without Attention | With Attention |
|------------|------------------|----------------|
| Document understanding | Limited to short text | Handles thousands of tokens |
| Translation quality | Degrades with length | Maintains quality across long documents |
| Question answering | Struggles with long context | Can locate relevant information in long documents |
| Interpretability | Opaque | Attention maps provide some visibility into model focus |

### Questions to Ask Your Team

1. "Can you show me attention visualizations for our model?"
2. "Are we using multi-head attention? How many heads?"
3. "What's our maximum sequence length? What drives that limit?"
4. "Are we using KV-caching during inference? What's our cache memory budget?"
5. "Have we tried FlashAttention or other efficient attention implementations?"

### Cost Considerations

Attention has O(n^2) complexity in sequence length:
- 1,000 tokens: ~1M attention computations
- 10,000 tokens: ~100M attention computations (100x more)
- 100,000 tokens: ~10B attention computations (10,000x more)

This is why context length is expensive and why efficient attention variants (FlashAttention, GQA, sliding-window attention) matter for production deployments.

---

## Interview Preparation

### Where This Knowledge Maps to Job Roles

| Blog Section | ML Engineer | Data Scientist | AI Architect | Engineering Manager |
|---|---|---|---|---|
| Part 1: QKV Framework | Implement custom attention, debug gradient flow | Understand what attention computes | Design attention-based architectures | Understand why attention replaced RNNs |
| Part 2-3: Self/Multi-Head | Build efficient MHA, optimize split_heads | Use attention layers in models | Choose num_heads, d_model tradeoffs | Ask "how many heads? what dimension?" |
| Part 4: Cross/Causal | Implement encoder-decoder, causal masks | Understand seq2seq and generation | Design RAG cross-attention pipelines | Understand generation vs encoding |
| Part 5: PyTorch MHA | Production implementation, torch.compile | Use nn.MultiheadAttention | Code review attention implementations | Budget GPU for attention-heavy models |
| Part 6: Interpretability | Build attention visualization tools | Analyze model behavior on tasks | Design interpretability pipelines | Ask "can we explain model decisions?" |
| Part 7: KV-Cache/Efficiency | Implement KV-cache, FlashAttention integration | Profile inference latency | Choose GQA vs MHA, FlashAttention strategy | Budget inference memory and latency |

### Likely Questions

**Q: Explain the Query-Key-Value framework.**
A: Queries represent what each position is "looking for." Keys represent what each position "offers." Values represent what information to return. Attention computes Q.K similarity, applies softmax to get weights, then returns weighted sum of values. This lets each position selectively access relevant information from anywhere in the sequence.

**Q: Why do we scale by sqrt(d_k)?**
A: Without scaling, when d_k is large, the dot products Q.K can have high variance (proportional to d_k), pushing softmax into extreme regions with very small gradients (saturation). Scaling by sqrt(d_k) normalizes the variance to approximately 1 regardless of dimension, keeping gradients well-behaved. This was first noted in the original Transformer paper (Vaswani et al., 2017).

**Q: What's the difference between self-attention and cross-attention?**
A: Self-attention: Q, K, V all come from the same sequence. Each position attends to all positions in the same sequence. Cross-attention: Q from one sequence (e.g., decoder), K and V from another (e.g., encoder). This allows one sequence to gather information from another.

**Q: Why use multiple heads?**
A: Single attention can only learn one type of relationship. Multiple heads let the model attend to different aspects in parallel: syntactic relationships in one head, semantic in another, positional in another. The outputs are concatenated and projected. Each head operates in a d_model/num_heads subspace, so total compute is comparable to single-head full-dimensional attention.

**Q: How does causal masking work and why is it needed?**
A: Causal masking prevents each position from attending to future positions. Position i can only see positions 0 to i. This is essential for autoregressive generation — when predicting the next token, we can't peek at future tokens that don't exist yet during inference.

**Q: What is KV-caching and why does it matter?**
A: During autoregressive generation, KV-caching stores the Key and Value projections from all previous tokens so they are not recomputed at each generation step. Without it, generating N tokens costs O(N^3) total attention operations; with it, O(N^2). For a 7B-parameter model with 4K context, the KV-cache alone is about 2 GB, which is why memory management is critical at scale.

**Q: Does attention weight magnitude mean a token is "important"?**
A: Not necessarily. Research has shown that attention weights do not always correlate with feature importance. Alternative attention distributions can produce the same output. Attention visualizations are useful for building intuition but should be combined with gradient-based methods for rigorous interpretability.

**Q: What's the computational complexity of self-attention and why is it a problem?**
A: O(n²) in sequence length for both time and memory (n² attention scores per head per layer). For 512 tokens this is manageable (~262K operations), but at 32K tokens it becomes ~1 billion operations per head. This is the primary bottleneck for long-context models. Solutions: FlashAttention (same math, better memory access), sliding-window attention, sparse attention, or linear attention approximations.

**Q: Compare dot-product attention vs additive (Bahdanau) attention.**
A: Dot-product: score = Q·K^T/sqrt(d_k). Fast GPU matrix multiplication. Requires Q and K to share dimension. Used in all transformers. Additive: score = v^T·tanh(W_q·Q + W_k·K). Can handle different Q/K dimensions. Slower. Used in pre-2017 seq2seq models. With learned projections, dot-product has equivalent expressiveness.

**Q: What is Grouped-Query Attention (GQA) and why does it matter?**
A: GQA shares K/V heads across multiple Q heads. In standard MHA, 32 query heads each have their own K/V heads (32 KV heads). In GQA, you might have 32 query heads but only 8 KV heads (4:1 ratio). This reduces KV-cache size by 4x with minimal quality loss. LLaMA 2 70B uses GQA. It's essential for deploying large models with long contexts where KV-cache dominates memory.

**Q: Why do we apply dropout to attention weights?**
A: Attention dropout prevents heads from co-adapting (head 3 relying on head 1's output) and regularizes attention distributions. Without it, the model may put 100% weight on a single token, making it brittle to input noise. Typical rate: 0.1 during training, 0.0 during inference.

---

## Exercises (Do These)

1. **Attention from scratch**: Implement scaled dot-product attention without using any ML libraries. Verify with numerical gradient checking.

2. **Visualize BERT**: Load a pre-trained BERT model, run sentences through it, and visualize attention from pronouns to their antecedents.

3. **Compare patterns**: Create sentences where attention should show clear patterns (e.g., "The dog that bit the cat ran away"). Do the learned patterns match your expectations?

4. **KV-cache benchmark**: Implement autoregressive generation with and without KV-caching. Measure wall-clock time for generating 100 tokens with sequence lengths of 128, 512, and 2048. Plot the speedup.

5. **Cross-attention**: Build a simple encoder-decoder with cross-attention. Train it on a toy task (e.g., reversing sequences) and visualize the cross-attention pattern.

6. **Head pruning**: Take a pre-trained BERT model, zero out one attention head at a time, and measure the impact on a downstream task (e.g., SST-2 sentiment). Which heads matter most?

---

## What's Next

You now have:
- Query-Key-Value framework understanding
- Scaled dot-product attention implementation
- Multi-head attention mechanics
- Self-attention vs cross-attention knowledge
- Attention visualization and interpretation skills
- KV-caching understanding for efficient inference

**Blog 9** assembles these pieces into the **Transformer architecture** — the foundation of GPT, BERT, and all modern language models. You'll build a mini-transformer from scratch, including positional encodings, feed-forward layers, and layer normalization.

**[Blog 9: Transformers — The Architecture Behind Everything](#)**

---

---

## Honest Self-Assessment

### What This Blog Does Well
- Builds the QKV framework from first principles with clear analogies and rigorous sqrt(d_k) variance argument
- Provides working NumPy implementations with step-by-step commentary, progressing to PyTorch production code
- Covers the full taxonomy: self-attention, cross-attention, causal attention, multi-head — with implementations for each
- Compares attention scoring functions (dot-product vs additive/Bahdanau vs multiplicative/Luong) with clear guidance on when each applies
- Includes real BERT attention visualization with pre-trained weights and "attention is not explanation" caveat
- KV-caching with concrete memory formula, GQA explanation, and FlashAttention conceptual coverage
- Memory estimation function for attention layers with O(n²) breakdown
- Attention dropout explanation (prevents co-adaptation, regularizes distributions)
- 6 section checkpoints with self-check questions
- Job role mapping table and 12 interview questions

### Where This Blog Falls Short
- All NumPy implementations use random weights — attention patterns in those demos are not linguistically meaningful, despite warnings
- No training loop is included — readers cannot see attention weights evolve from random to meaningful
- Cross-attention class omits head splitting for simplicity, which could mislead readers
- Efficient attention (FlashAttention, linear attention) is discussed conceptually but not implemented
- Positional encoding interaction with attention is deferred entirely to Blog 9

---

## Architect Sanity Checks

### Check 1: Mechanism Understanding
**Question**: Why do we scale by 1/sqrt(d_k) in attention?
**Verdict: YES** — Scaling prevents softmax saturation by keeping query-key dot products in an appropriate numerical range, ensuring gradients remain stable during backpropagation. The blog explains the variance argument and connects it to training stability.

### Check 2: Implementation Capability
**Question**: Can you implement multi-head attention from scratch?
**Verdict: YES** — A complete multi-head attention implementation from raw tensor operations includes proper head splitting and reshaping, separate projection matrices, scaled dot-product attention per head, and head concatenation. The implementation correctly handles batch and sequence dimensions.

### Check 3: Practical Application
**Question**: What do attention weights tell us about the model?
**Verdict: YES** — The blog provides visualization tools, pattern analysis (diagonal, first-token, entropy), and explicitly warns that attention weights ≠ importance with citations (Jain & Wallace, 2019). The caveat to combine attention with gradient-based methods for rigorous interpretability is clear. Readers won't make the common mistake of over-interpreting attention heatmaps.

### Check 4: Production Readiness
**Question**: Does the reader understand inference-time attention considerations?
**Verdict: YES** — KV-caching is covered with a working implementation, concrete memory formula (512 KB/token for 7B model), and GQA explanation with LLaMA 2 70B specifics. Memory estimation function covers training and inference modes. FlashAttention is explained conceptually (tiling, online softmax, O(N) memory). The O(n²) complexity table gives concrete numbers from 512 to 128K tokens. Remaining gap: sliding-window and quantized KV-caches are mentioned but not implemented, which is appropriate for this blog's scope.

---

*Questions? Found an error? Comments are open. Technical corrections get priority.*
