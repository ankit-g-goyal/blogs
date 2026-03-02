# Blog 9: Transformers
## The Architecture Behind Everything

**Reading time:** 75–90 minutes
**Coding time:** 120–150 minutes
**Total investment:** 3.5–4 hours

---

## Prerequisites & Reading Guide

**Required knowledge from earlier blogs:**
- Blog 8: Attention mechanism and multi-head attention (we build on this directly)
- Blog 7: RNNs and LSTMs (understanding why transformers replaced them)
- Blog 5: PyTorch basics (nn.Module, training loops, autograd)
- Blog 4: Neural network fundamentals (backpropagation, gradient descent)

**Environment setup:**
- Python 3.9+, PyTorch 2.0+, NumPy, Matplotlib
- GPU recommended for Shakespeare training (CPU works but takes ~10x longer)
- ~2GB disk space for Shakespeare dataset and saved model

**How to read this blog:**
- Parts 1-2: Architecture overview and building blocks — read carefully, run all code
- Parts 3-4: Encoder and decoder implementation — type the code yourself, do not copy-paste
- Part 5: GPT implementation — this is the core deliverable; study it thoroughly
- Part 6: Shakespeare training — run this end-to-end; experiment with hyperparameters
- Parts 7-8: Variants and inference — conceptual understanding; reference material

---

## What This Blog Does NOT Cover

- **Tokenization algorithms** (BPE, WordPiece, SentencePiece) — covered in Blog 10
- **Pre-training at scale** (distributed training, data pipelines) — covered in Blog 10 and Blog 23
- **Fine-tuning techniques** (LoRA, QLoRA, adapter layers) — covered in Blog 23
- **Efficient attention variants** (FlashAttention internals, linear attention, sparse attention) — mentioned but not implemented
- **Mixture of Experts (MoE)** architectures — beyond scope
- **Multimodal transformers** (vision transformers, cross-modal attention) — covered in Blog 21
- **RLHF and alignment** — covered in Blog 23

---

## What You'll Walk Away With

By the end of this blog, you will:

1. **Understand** the complete Transformer architecture from the 2017 paper
2. **Implement** positional encoding, layer normalization, and the feed-forward network
3. **Build** a complete Transformer encoder and decoder from scratch
4. **Train** a mini-GPT on Shakespeare that generates text
5. **Explain** why transformers dominated and what their limitations are

This is the most important blog in the series. Transformers power GPT, Claude, BERT, LLaMA, and virtually every frontier AI model.

---

## Why Transformers Won

### The 2017 Revolution

"Attention Is All You Need" (Vaswani et al., 2017) made a bold claim: **you don't need recurrence or convolution for sequence modeling**. Pure attention is enough.

The results were staggering:

| Model | BLEU (Translation) | Training Time |
|-------|-------------------|---------------|
| LSTM (previous SOTA) | 26.0 | ~3 weeks |
| Transformer | 28.4 | ~3.5 days |

Better quality AND 5x faster training.

### Why Transformers Beat RNNs

| RNN | Transformer |
|-----|-------------|
| Sequential processing | Parallel processing |
| O(n) steps to connect distant positions | O(1) steps (direct attention) |
| Fixed hidden state size | Attention over all positions |
| Vanishing gradients | Constant-length gradient paths |
| Hard to parallelize | Massively parallelizable |

The key insight: **attention provides direct connections between all positions**, eliminating the information bottleneck of sequential processing.

---

## Part 1: The Complete Architecture

### High-Level View

```
                    ┌─────────────────────┐
                    │   Output Probabilities│
                    └──────────┬──────────┘
                               │
                    ┌──────────┴──────────┐
                    │    Linear + Softmax  │
                    └──────────┬──────────┘
                               │
         ┌─────────────────────┴─────────────────────┐
         │                                           │
    ┌────┴────┐                               ┌──────┴──────┐
    │ Encoder │                               │   Decoder   │
    │  Stack  │  ──── Cross-Attention ────>   │    Stack    │
    │ (N=6)   │                               │   (N=6)     │
    └────┬────┘                               └──────┬──────┘
         │                                           │
    ┌────┴────┐                               ┌──────┴──────┐
    │Positional│                               │  Positional │
    │Encoding │                               │  Encoding   │
    └────┬────┘                               └──────┬──────┘
         │                                           │
    ┌────┴────┐                               ┌──────┴──────┐
    │  Input  │                               │   Output    │
    │Embedding│                               │  Embedding  │
    └─────────┘                               └─────────────┘
         ↑                                           ↑
      "Hello"                                    "<start>"
```

### Component Breakdown

| Component | Purpose | Key Operation |
|-----------|---------|---------------|
| Embedding | Convert tokens to vectors | Lookup table |
| Positional Encoding | Add position information | Sinusoidal/learned |
| Multi-Head Attention | Relate positions | Q·K similarity, weighted V |
| Feed-Forward | Transform representations | Two linear layers with ReLU |
| Layer Norm | Stabilize training | Normalize across features |
| Residual Connections | Gradient flow | Add input to output |

### ✅ Checkpoint: After Part 1

You should now be able to answer:
1. What are the six components of a transformer layer (embedding, positional encoding, attention, FFN, layer norm, residual)?
2. Why do transformers beat RNNs on both quality AND training speed?
3. What is the role of residual connections in enabling deep transformer training?
4. Why does the encoder use bidirectional attention while the decoder uses causal attention?

If you can't answer all four, re-read Part 1 before continuing.

---

## Part 2: Building Blocks

### Positional Encoding

Since attention has no inherent notion of position, we must add it explicitly:

```python
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding from the original paper.

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Each position gets a unique encoding.
    Each dimension captures different frequencies.
    """

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Compute the div_term: 10000^(2i/d_model)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )

        # Apply sin to even indices, cos to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension and register as buffer (not a parameter)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Add positional encoding to input embeddings.

        Args:
            x: (batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# Visualize positional encodings
pe = PositionalEncoding(d_model=128, max_len=100)
encodings = pe.pe[0, :50, :].numpy()

plt.figure(figsize=(12, 8))
plt.imshow(encodings.T, aspect='auto', cmap='RdBu')
plt.xlabel('Position')
plt.ylabel('Dimension')
plt.title('Sinusoidal Positional Encodings')
plt.colorbar(label='Value')
plt.savefig('positional_encoding.png', dpi=150)
plt.show()

# Show how different positions have different patterns
print("Key properties:")
print("1. Each position has a unique encoding")
print("2. Low dimensions change slowly (capture long patterns)")
print("3. High dimensions change quickly (capture short patterns)")
print("4. Relative positions can be computed via dot product")
```

### Layer Normalization

```python
class LayerNorm(nn.Module):
    """
    Layer normalization: normalize across features (not batch).

    For each position in the sequence:
    1. Compute mean and variance across feature dimensions
    2. Normalize to zero mean, unit variance
    3. Scale and shift with learned parameters

    Why Layer Norm instead of Batch Norm?
    - Works with variable sequence lengths
    - Independent of batch size
    - More stable for transformers
    """

    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        """
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


# Comparison: Layer Norm vs Batch Norm
x = torch.randn(32, 10, 64)  # (batch, seq, features)

layer_norm = LayerNorm(64)
batch_norm = nn.BatchNorm1d(64)

ln_out = layer_norm(x)
# For batch norm, need to reshape: (batch*seq, features)
bn_out = batch_norm(x.view(-1, 64)).view(32, 10, 64)

print(f"Layer Norm: normalizes across {x.shape[-1]} features for each position")
print(f"Batch Norm: normalizes across {x.shape[0]*x.shape[1]} samples for each feature")
```

### Feed-Forward Network

```python
class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.

    Applied identically to each position.
    FFN(x) = max(0, xW₁ + b₁)W₂ + b₂

    The inner dimension (d_ff) is typically 4x the model dimension.
    WHY 4x? Attention captures inter-position relationships but applies
    only linear transformations per position. The FFN adds non-linear
    capacity at each position independently. The 4x expansion gives the
    model a large "working space" for these transformations before
    compressing back to d_model. Think of it as: attention decides WHAT
    to look at, FFN decides HOW to transform what was gathered.

    The 4x ratio is empirical (Vaswani et al., 2017) but has held up:
    GPT-2/3: d_ff = 4 * d_model. LLaMA uses d_ff ≈ 2.67 * d_model with
    SwiGLU activation (which adds a gating parameter, so total params
    are similar). The ratio trades parameters for per-position capacity.
    """

    def __init__(self, d_model, d_ff=None, dropout=0.1):
        super().__init__()
        d_ff = d_ff or d_model * 4

        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()  # Modern choice; original used ReLU

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model)
        """
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


ff = FeedForward(d_model=64, d_ff=256)
x = torch.randn(2, 10, 64)
print(f"Input: {x.shape} → Output: {ff(x).shape}")
print(f"Parameters: {sum(p.numel() for p in ff.parameters()):,}")
```

### Rotary Positional Embeddings (RoPE) — The Modern Standard

```python
"""
Sinusoidal (original paper) and learned (GPT-2) positional encodings are
ADDITIVE: they add position info to the embedding before attention.

RoPE (Su et al., 2021) is MULTIPLICATIVE: it encodes position by ROTATING
query and key vectors in 2D subspaces. This is the standard in modern LLMs
(LLaMA, Mistral, Qwen, Gemma).

WHY RoPE is better:
1. RELATIVE position information: the dot product Q·K depends only on the
   DIFFERENCE between positions, not absolute positions. This means the model
   naturally generalizes to different positions.
2. Length extrapolation: RoPE handles sequences longer than training context
   better than learned positional embeddings (with techniques like NTK-aware
   scaling or YaRN).
3. No additional parameters: unlike learned positional embeddings, RoPE is
   a deterministic function (no trainable weights).

HOW IT WORKS (simplified):
For each pair of dimensions (2i, 2i+1) in the query/key vectors, RoPE
applies a 2D rotation by angle θ_i * position:

    [q_2i]     [cos(θ*pos)  -sin(θ*pos)] [q_2i]
    [q_2i+1] = [sin(θ*pos)   cos(θ*pos)] [q_2i+1]

where θ_i = 10000^(-2i/d_k), same base as sinusoidal encoding.

The key insight: when computing Q·K^T, the rotation angles SUBTRACT:
    rotated_q · rotated_k = f(q, k, pos_q - pos_k)
This naturally encodes relative position.
"""

import torch

def apply_rope(x, seq_len, d_model, base=10000.0):
    """
    Apply Rotary Positional Embedding to query or key tensor.

    Args:
        x: (batch, heads, seq_len, d_k)
        seq_len: sequence length
        d_model: dimension per head
    """
    # Compute rotation frequencies
    freqs = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
    positions = torch.arange(seq_len).float()

    # Outer product: (seq_len, d_model/2)
    angles = torch.outer(positions, freqs)

    # Create rotation matrices as complex numbers
    cos_vals = torch.cos(angles)  # (seq_len, d_model/2)
    sin_vals = torch.sin(angles)

    # Reshape x into pairs
    x_pairs = x.float().reshape(*x.shape[:-1], -1, 2)
    x_even = x_pairs[..., 0]  # (batch, heads, seq, d_k/2)
    x_odd = x_pairs[..., 1]

    # Apply rotation
    rotated_even = x_even * cos_vals - x_odd * sin_vals
    rotated_odd = x_even * sin_vals + x_odd * cos_vals

    # Interleave back
    rotated = torch.stack([rotated_even, rotated_odd], dim=-1)
    return rotated.reshape(*x.shape).type_as(x)


# Demonstrate RoPE property: dot product depends on relative position
d_k = 64
q = torch.randn(1, 1, 1, d_k)  # query at position 5
k = torch.randn(1, 1, 1, d_k)  # key at position 3

# Apply RoPE at different absolute positions but same relative distance
for pos_q, pos_k in [(5, 3), (10, 8), (100, 98)]:
    q_rot = apply_rope(q, pos_q + 1, d_k)[..., pos_q:pos_q+1, :]
    k_rot = apply_rope(k, pos_k + 1, d_k)[..., pos_k:pos_k+1, :]
    dot = (q_rot * k_rot).sum().item()
    print(f"  pos_q={pos_q}, pos_k={pos_k}, diff={pos_q-pos_k}: dot={dot:.4f}")

# All three should give similar dot products (same relative distance = 2)
print("(Values should be similar — RoPE encodes relative position)")
```

### ✅ Checkpoint: After Part 2

You should now be able to answer:
1. Why is the embedding multiplied by sqrt(d_model) before adding positional encoding?
2. Why is d_ff = 4 × d_model (and how does LLaMA's SwiGLU change this)?
3. Why does Layer Norm work better than Batch Norm for transformers?
4. How does RoPE encode relative position through rotation, and why is it better than additive encoding?

If you can't answer all four, re-read Part 2 before continuing.

---

## Part 3: Transformer Encoder

### Single Encoder Layer

```python
import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    """Multi-head attention (from Blog 8)."""

    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        Q = self.W_Q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_K(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_V(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attention = torch.softmax(scores, dim=-1)
        attention = self.dropout(attention)

        context = torch.matmul(attention, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        return self.W_O(context), attention


class EncoderLayer(nn.Module):
    """
    Single Transformer encoder layer.

    Components:
    1. Multi-head self-attention
    2. Add & Norm (residual + layer norm)
    3. Feed-forward network
    4. Add & Norm
    """

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()

        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: Optional attention mask
        """
        # Self-attention with residual and norm
        attn_output, _ = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Feed-forward with residual and norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x


class TransformerEncoder(nn.Module):
    """
    Full Transformer encoder: stack of encoder layers.
    """

    def __init__(self, vocab_size, d_model=512, num_heads=8, num_layers=6,
                 d_ff=2048, max_len=5000, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)

        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        """
        Args:
            x: Token indices, (batch, seq_len)
            mask: Optional padding mask
        """
        # Embed and add positional encoding
        # WHY multiply by sqrt(d_model)?
        # Embeddings are initialized with variance ~1/d_model (nn.Embedding default).
        # Positional encodings have values in [-1, 1] (sin/cos).
        # Without scaling, the positional signal dominates as d_model grows.
        # Multiplying by sqrt(d_model) brings embedding magnitudes to ~O(1),
        # matching the positional encoding scale.
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)

        # Pass through encoder layers
        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)


# Test encoder
encoder = TransformerEncoder(vocab_size=10000, d_model=256, num_heads=8, num_layers=4)
x = torch.randint(0, 10000, (4, 32))  # batch=4, seq_len=32

output = encoder(x)
print(f"Input shape: {x.shape}")
print(f"Encoder output shape: {output.shape}")
print(f"Total parameters: {sum(p.numel() for p in encoder.parameters()):,}")
```

### ✅ Checkpoint: After Part 3

You should now be able to answer:
1. What are the two sub-layers in an encoder layer (self-attention + FFN) and what does each do?
2. Where do residual connections go and why are they critical?
3. What is the difference between pre-norm and post-norm architectures?
4. How many parameters does a single encoder layer have? (Express in terms of d_model, d_ff, num_heads)

If you can't answer all four, re-read Part 3 before continuing.

---

## Part 4: Transformer Decoder

### Decoder Layer with Causal Masking

```python
class DecoderLayer(nn.Module):
    """
    Single Transformer decoder layer.

    Components:
    1. Masked multi-head self-attention (causal)
    2. Add & Norm
    3. Multi-head cross-attention (to encoder output)
    4. Add & Norm
    5. Feed-forward network
    6. Add & Norm
    """

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()

        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        """
        Args:
            x: Decoder input, (batch, tgt_seq_len, d_model)
            encoder_output: Encoder output, (batch, src_seq_len, d_model)
            src_mask: Padding mask for encoder output
            tgt_mask: Causal mask for decoder (prevents looking ahead)
        """
        # Masked self-attention
        attn_output, _ = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Cross-attention to encoder
        attn_output, _ = self.cross_attention(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))

        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))

        return x


def create_causal_mask(size):
    """Create causal (look-ahead) mask."""
    mask = torch.triu(torch.ones(size, size), diagonal=1)
    return mask == 0  # True where attention is allowed


class TransformerDecoder(nn.Module):
    """
    Full Transformer decoder: stack of decoder layers.
    """

    def __init__(self, vocab_size, d_model=512, num_heads=8, num_layers=6,
                 d_ff=2048, max_len=5000, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)

        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        """
        Args:
            x: Target token indices, (batch, tgt_seq_len)
            encoder_output: Encoder output
            src_mask: Source padding mask
            tgt_mask: Target causal mask
        """
        seq_len = x.size(1)

        # Create causal mask if not provided
        if tgt_mask is None:
            tgt_mask = create_causal_mask(seq_len).to(x.device)

        # Embed and add positional encoding
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)

        # Pass through decoder layers
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)

        x = self.norm(x)
        return self.output_projection(x)


# Test decoder
decoder = TransformerDecoder(vocab_size=10000, d_model=256, num_heads=8, num_layers=4)
encoder_output = torch.randn(4, 32, 256)  # From encoder
tgt = torch.randint(0, 10000, (4, 20))  # Target sequence

output = decoder(tgt, encoder_output)
print(f"Target shape: {tgt.shape}")
print(f"Decoder output shape: {output.shape}")  # (batch, tgt_seq_len, vocab_size)
```

### ✅ Checkpoint: After Part 4

You should now be able to answer:
1. How does the decoder differ from the encoder (3 sub-layers vs 2, causal masking)?
2. What is cross-attention and where does it connect encoder to decoder?
3. Why does the causal mask use `torch.triu` (upper triangular)?
4. Where in the decoder is the output vocabulary projection applied?

If you can't answer all four, re-read Part 4 before continuing.

---

## Part 5: GPT-Style Decoder-Only Transformer

For language modeling (like GPT), we only need the decoder (no encoder):

```python
# NOTE: GPTBlock is defined first because GPT references it.

class GPTBlock(nn.Module):
    """Single GPT transformer block."""

    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()

        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Pre-norm architecture (GPT-2 style)
        attn_input = self.norm1(x)
        attn_output, _ = self.attention(attn_input, attn_input, attn_input, mask)
        x = x + self.dropout(attn_output)

        ff_input = self.norm2(x)
        ff_output = self.feed_forward(ff_input)
        x = x + self.dropout(ff_output)

        return x


class GPT(nn.Module):
    """
    GPT-style decoder-only transformer.

    This is the architecture behind GPT-2, GPT-3, GPT-4, and similar models.
    No encoder, no cross-attention - just causal self-attention.
    """

    def __init__(self, vocab_size, d_model=256, num_heads=8, num_layers=6,
                 d_ff=1024, max_len=512, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.max_len = max_len

        # Token and position embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)

        self.dropout = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            GPTBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying: share embedding and output weights
        self.output_projection.weight = self.token_embedding.weight

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, targets=None):
        """
        Args:
            x: Token indices, (batch, seq_len)
            targets: Optional target indices for computing loss

        Returns:
            logits: (batch, seq_len, vocab_size)
            loss: Cross-entropy loss if targets provided
        """
        batch_size, seq_len = x.shape
        device = x.device

        # Get embeddings
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        x = self.token_embedding(x) + self.position_embedding(positions)
        x = self.dropout(x)

        # Create causal mask
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device),
            diagonal=1
        ).bool()

        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x, causal_mask)

        x = self.norm(x)
        logits = self.output_projection(x)

        # Compute loss if targets provided
        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generate text autoregressively.

        Args:
            idx: Starting token indices, (batch, seq_len)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: If set, only sample from top k tokens

        Returns:
            Generated token indices, (batch, seq_len + max_new_tokens)
        """
        for _ in range(max_new_tokens):
            # Crop to max_len if necessary
            idx_cond = idx if idx.size(1) <= self.max_len else idx[:, -self.max_len:]

            # Forward pass
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            # Optional top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float('-inf')

            # Optional nucleus (top-p) sampling
            # Top-p keeps the smallest set of tokens whose cumulative probability >= p
            # This adapts the number of candidates dynamically:
            #   - High-confidence predictions → few candidates (peaked distribution)
            #   - Low-confidence predictions → many candidates (flat distribution)
            # Production standard: top_p=0.9-0.95 with temperature=0.7-1.0
            if hasattr(self, '_top_p') and self._top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > self._top_p
                # Shift to keep the first token above threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')

            # Sample
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append
            idx = torch.cat([idx, idx_next], dim=1)

        return idx
```

### ✅ Checkpoint: After Part 5

You should now be able to answer:
1. Why does GPT use learned positional embeddings instead of sinusoidal?
2. What is weight tying and why does it reduce parameters AND improve quality?
3. How does pre-norm (GPT-2 style) differ from post-norm (original Transformer)?
4. What is nucleus (top-p) sampling and why is it preferred over pure top-k in production?

If you can't answer all four, re-read Part 5 before continuing.

---

## Part 6: Training a Mini-GPT on Shakespeare

```python
# shakespeare_gpt.py
"""
Train a mini-GPT on Shakespeare's works.

This demonstrates that our transformer implementation can actually learn!
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import matplotlib.pyplot as plt

# Download Shakespeare text
import urllib.request

def download_shakespeare():
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    response = urllib.request.urlopen(url)
    return response.read().decode('utf-8')

print("Downloading Shakespeare...")
text = download_shakespeare()
print(f"Downloaded {len(text):,} characters")
print(f"Sample:\n{text[:500]}")

# Create character-level tokenizer
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"\nVocabulary size: {vocab_size}")
print(f"Characters: {''.join(chars)}")

# Mappings
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

def encode(text):
    return [char_to_idx[ch] for ch in text]

def decode(indices):
    return ''.join([idx_to_char[i] for i in indices])

# Dataset
class ShakespeareDataset(Dataset):
    def __init__(self, text, block_size=128):
        self.data = torch.tensor(encode(text), dtype=torch.long)
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.block_size]
        y = self.data[idx + 1:idx + self.block_size + 1]
        return x, y


# Create datasets
block_size = 128
train_dataset = ShakespeareDataset(text[:int(0.9*len(text))], block_size)
val_dataset = ShakespeareDataset(text[int(0.9*len(text)):], block_size)

print(f"Training samples: {len(train_dataset):,}")
print(f"Validation samples: {len(val_dataset):,}")

# Create dataloaders
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Create model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}")

model = GPT(
    vocab_size=vocab_size,
    d_model=128,       # Small for demo
    num_heads=4,
    num_layers=4,
    d_ff=512,
    max_len=block_size,
    dropout=0.1
).to(device)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Training
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)

# WHY WARMUP + COSINE DECAY?
# Transformers are sensitive to learning rate at the start of training:
# - Too high initially → attention scores diverge, loss spikes, sometimes unrecoverable
# - The warmup phase gradually increases LR so the model can find a stable region
# - After warmup, cosine decay smoothly reduces LR for fine-grained convergence
# This schedule is standard across GPT-2, GPT-3, LLaMA, and most modern LLMs.
# Typical warmup: 1-10% of total training steps.

total_steps = epochs * len(train_loader)
warmup_steps = total_steps // 10  # 10% warmup

def lr_lambda(step):
    if step < warmup_steps:
        return step / warmup_steps  # Linear warmup
    # Cosine decay after warmup
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * progress))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def train_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        _, loss = model(x, y)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()  # Step per batch for warmup + cosine schedule

        total_loss += loss.item()
    return total_loss / len(loader)

@torch.no_grad()
def evaluate(model, loader):
    """Evaluate model and return average loss and perplexity.

    Perplexity = exp(cross_entropy_loss)
    A perplexity of P means the model is as uncertain as if it were
    choosing uniformly among P candidates at each step.
    Lower is better. Random baseline = vocab_size.
    """
    model.eval()
    total_loss = 0
    num_batches = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        total_loss += loss.item()
        num_batches += 1
    avg_loss = total_loss / num_batches
    perplexity = math.exp(avg_loss) if avg_loss < 100 else float('inf')
    return avg_loss, perplexity

def generate_sample(model, start_text="ROMEO:", max_tokens=200):
    model.eval()
    context = torch.tensor([encode(start_text)], device=device)
    generated = model.generate(context, max_new_tokens=max_tokens, temperature=0.8)
    return decode(generated[0].tolist())


# Training loop
history = {'train_loss': [], 'val_loss': [], 'val_perplexity': []}
epochs = 10

# Baseline: random model perplexity = vocab_size
print(f"\nRandom baseline perplexity: {vocab_size} (model should beat this quickly)")

print("\nTraining...")
for epoch in range(epochs):
    train_loss = train_epoch(model, train_loader, optimizer)
    val_loss, val_ppl = evaluate(model, val_loader)

    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['val_perplexity'].append(val_ppl)

    print(f"Epoch {epoch+1}/{epochs}: train_loss={train_loss:.4f}, "
          f"val_loss={val_loss:.4f}, val_perplexity={val_ppl:.1f}")

    # Generate sample every few epochs
    if (epoch + 1) % 3 == 0:
        sample = generate_sample(model)
        print(f"\n--- Generated Sample ---")
        print(sample[:300])
        print("---\n")

# Plot training: loss and perplexity
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(history['train_loss'], label='Train Loss')
ax1.plot(history['val_loss'], label='Val Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Cross-Entropy Loss')
ax1.set_title('Training Loss')
ax1.legend()
ax1.grid(True)

ax2.plot(history['val_perplexity'], label='Val Perplexity', color='green')
ax2.axhline(y=vocab_size, color='red', linestyle='--', label=f'Random Baseline ({vocab_size})')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Perplexity')
ax2.set_title('Validation Perplexity (lower is better)')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig('shakespeare_training.png', dpi=150)
plt.show()

print(f"\nFinal validation perplexity: {history['val_perplexity'][-1]:.1f}")
print(f"Random baseline perplexity: {vocab_size}")
print(f"Improvement over random: {vocab_size / history['val_perplexity'][-1]:.1f}x")

# Final generation
print("\n=== Final Generated Text ===")
prompts = ["ROMEO:", "To be or not to", "HAMLET:\nWhat"]

for prompt in prompts:
    print(f"\nPrompt: {prompt}")
    print(generate_sample(model, prompt, max_tokens=200))
    print("-" * 50)

# Save model
torch.save({
    'model_state_dict': model.state_dict(),
    'char_to_idx': char_to_idx,
    'idx_to_char': idx_to_char,
}, 'shakespeare_gpt.pt')
print("\nModel saved!")
```

### Memory Estimation for Transformer Models

```python
def estimate_transformer_memory(vocab_size, d_model, num_heads, num_layers,
                                 d_ff, max_len, batch_size, seq_len,
                                 dtype_bytes=4):
    """
    Estimate GPU memory for transformer training.

    Components:
    1. Embeddings: token + position
    2. Attention: 4 × d_model² per layer (Q, K, V, O projections)
    3. FFN: 2 × d_model × d_ff per layer (two linear layers)
    4. LayerNorm: 2 × d_model per norm (gamma + beta)
    5. Attention matrix: batch × heads × seq × seq per layer
    6. Activations for backprop
    7. Gradients + optimizer states (Adam: 2× parameters)
    """
    # Embedding parameters
    embed_params = vocab_size * d_model + max_len * d_model

    # Per-layer parameters
    attn_params = 4 * d_model * d_model + 4 * d_model  # Q,K,V,O + biases
    ffn_params = 2 * d_model * d_ff + d_model + d_ff   # 2 linear + biases
    norm_params = 4 * d_model  # 2 layer norms × (gamma + beta)
    layer_params = attn_params + ffn_params + norm_params

    total_params = embed_params + num_layers * layer_params + d_model  # final norm

    # Attention matrices (the O(n²) part)
    attn_matrix = batch_size * num_heads * seq_len * seq_len * dtype_bytes * num_layers

    # Activations (rough estimate: ~2× params × batch_size × seq_len / d_model)
    activation_mem = 2 * num_layers * batch_size * seq_len * d_model * dtype_bytes

    # Parameter memory + gradients + Adam states (m, v)
    param_mem = total_params * dtype_bytes * 4  # params + grads + 2× optimizer

    total_mb = (param_mem + attn_matrix + activation_mem) / (1024**2)

    print(f"Transformer Memory Estimation:")
    print(f"  Total parameters: {total_params:,} ({total_params/1e6:.1f}M)")
    print(f"  Parameter memory (w/ optimizer): {param_mem/1024**2:.1f} MB")
    print(f"  Attention matrices:              {attn_matrix/1024**2:.1f} MB")
    print(f"  Activations:                     {activation_mem/1024**2:.1f} MB")
    print(f"  TOTAL:                           {total_mb:.1f} MB")
    print(f"  Fits on: {'CPU/any GPU' if total_mb < 2000 else '8GB+ GPU' if total_mb < 6000 else '16GB+ GPU' if total_mb < 14000 else '24GB+ GPU'}")
    return total_mb

# Our Shakespeare model
print("=== Shakespeare Mini-GPT ===")
estimate_transformer_memory(
    vocab_size=65, d_model=128, num_heads=4, num_layers=4,
    d_ff=512, max_len=128, batch_size=64, seq_len=128
)

# Scaling up: what if we wanted GPT-2 Small?
print("\n=== GPT-2 Small scale ===")
estimate_transformer_memory(
    vocab_size=50257, d_model=768, num_heads=12, num_layers=12,
    d_ff=3072, max_len=1024, batch_size=8, seq_len=1024
)
```

### Perplexity Interpretation Guide

```python
"""
What perplexity values mean for character-level language models:

| Perplexity | Interpretation |
|------------|----------------|
| vocab_size (65 for Shakespeare) | Random baseline — model learned nothing |
| 10-20      | Model captures basic character patterns (common letters) |
| 3-10       | Model captures word-level patterns and common words |
| 1.5-3      | Good: model captures vocabulary, grammar, some style |
| < 1.5      | Excellent: near human-level character prediction |

For WORD-LEVEL models (e.g., GPT-2 on WikiText-103):
| Perplexity | Interpretation |
|------------|----------------|
| > 100      | Poor — model is guessing |
| 30-100     | Reasonable baseline (bigram/trigram territory) |
| 15-30      | Good (LSTM-level performance) |
| < 20       | Strong (fine-tuned transformer territory) |
| < 10       | State-of-the-art (GPT-3 class) |

IMPORTANT: Perplexity is NOT comparable across different tokenizations.
Character-level perplexity of 2.0 is NOT the same as word-level perplexity of 2.0.
Always compare within the same tokenization scheme.
"""
```

### ✅ Checkpoint: After Part 6

You should now be able to answer:
1. Why does transformer training need a warmup phase?
2. What is perplexity and how do you interpret it for a character-level model?
3. Estimate the memory needed for a 4-layer, 128-dim transformer with batch_size=64.
4. What does the random baseline perplexity tell you about your model's progress?

If you can't answer all four, re-read Part 6 before continuing.

---

## Part 7: Transformer Variants

### Encoder-Only (BERT-style)

```python
# For classification, understanding, embeddings
# Bidirectional attention (see all tokens)
# Used for: sentiment, NER, question answering

class BERTClassifier(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, num_classes):
        super().__init__()
        self.encoder = TransformerEncoder(vocab_size, d_model, num_heads, num_layers)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        encoded = self.encoder(x)
        # Use [CLS] token representation (first token)
        cls_output = encoded[:, 0, :]
        return self.classifier(cls_output)
```

### Decoder-Only (GPT-style)

```python
# For generation, completion
# Causal attention (only see past tokens)
# Used for: text generation, code completion, chat

# Our GPT class above is decoder-only
```

### Encoder-Decoder (Original Transformer)

```python
# For sequence-to-sequence tasks
# Encoder: bidirectional, Decoder: causal + cross-attention
# Used for: translation, summarization

class Seq2SeqTransformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model, num_heads, num_layers):
        super().__init__()
        self.encoder = TransformerEncoder(src_vocab, d_model, num_heads, num_layers)
        self.decoder = TransformerDecoder(tgt_vocab, d_model, num_heads, num_layers)

    def forward(self, src, tgt):
        encoder_output = self.encoder(src)
        return self.decoder(tgt, encoder_output)
```

### Comparison

| Architecture | Attention Type | Use Cases | Examples |
|-------------|---------------|-----------|----------|
| Encoder-only | Bidirectional | Understanding, classification | BERT, RoBERTa |
| Decoder-only | Causal | Generation | GPT, LLaMA, Claude |
| Encoder-decoder | Both | Seq2seq | T5, BART |

---

## Part 8: Inference Optimization — KV-Caching

Our `generate()` method has a critical inefficiency: it recomputes the full forward pass for every new token. For a sequence of length L, generating one new token requires recomputing attention for all L previous positions — even though their key and value projections have not changed.

### The Problem

```
Generate token 1: compute attention over positions [0]
Generate token 2: compute attention over positions [0, 1]       ← position 0 recomputed
Generate token 3: compute attention over positions [0, 1, 2]    ← positions 0,1 recomputed
...
Generate token N: compute attention over positions [0..N-1]     ← all previous recomputed
```

Total work: O(N^2) attention computations. For N=1000 tokens, this means ~500,000 redundant computations.

### The Solution: KV-Cache

Cache the key (K) and value (V) projections from previous positions. When generating a new token, only compute Q, K, V for the new position, then concatenate K and V with the cached values.

```python
class CachedMultiHeadAttention(nn.Module):
    """Multi-head attention with KV-cache for efficient autoregressive generation."""

    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, kv_cache=None):
        """
        Args:
            x: (batch, seq_len, d_model) — during generation, seq_len=1
            mask: Optional attention mask
            kv_cache: Tuple of (cached_K, cached_V) or None

        Returns:
            output: (batch, seq_len, d_model)
            attention: attention weights
            new_kv_cache: Updated (K, V) cache
        """
        batch_size = x.size(0)

        # Project current input
        Q = self.W_Q(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_K(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Append to cache if it exists
        if kv_cache is not None:
            cached_K, cached_V = kv_cache
            K = torch.cat([cached_K, K], dim=2)  # Extend along sequence dim
            V = torch.cat([cached_V, V], dim=2)

        new_kv_cache = (K, V)

        # Standard attention computation
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attention = torch.softmax(scores, dim=-1)
        attention = self.dropout(attention)

        context = torch.matmul(attention, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        return self.W_O(context), attention, new_kv_cache


# Performance comparison (conceptual)
# Without KV-cache: generating N tokens costs O(N^2 * d_model) total work
# With KV-cache:    generating N tokens costs O(N * d_model) total work
# For N=1000 tokens, this is a ~500x speedup
#
# Memory tradeoff: KV-cache stores (2 * num_layers * batch * seq_len * d_model) floats
# For a 7B model with 32 layers, d_model=4096, seq_len=2048:
#   Cache size = 2 * 32 * 1 * 2048 * 4096 * 2 bytes (fp16) ≈ 1 GB per sequence
```

**Why this matters in production**: Every commercial LLM API (OpenAI, Anthropic, Google) uses KV-caching. Without it, inference costs would be orders of magnitude higher. This is also why "context length" has a direct memory cost — longer contexts require larger KV-caches.

---

## Manager's Summary

### Why Transformers Matter

**Business impact**: Transformers are the foundation of ChatGPT, Claude, Gemini, and every major AI assistant. Understanding them is understanding modern AI.

### Scale Considerations

| Model | Parameters | Approximate Training Cost |
|-------|-----------|---------------------------|
| GPT-2 Small | 117M | Not publicly disclosed |
| GPT-3 | 175B | ~$4.6M (estimated from compute) |
| GPT-4 | Not publicly disclosed | Not publicly disclosed |

*Note: Training cost estimates are based on reported compute usage and hardware pricing at the time. GPT-4's architecture and parameter count have not been officially disclosed by OpenAI. Claims of specific numbers circulating online are speculation. Inference costs vary by provider and change frequently — check current API pricing pages.*

### Key Tradeoffs

| Decision | Tradeoff |
|----------|----------|
| More layers | Better quality, slower/more expensive |
| More heads | More relationship types, more memory |
| Larger context | More info access, O(n²) cost |
| Pre-training | Expensive once, amortized across uses |

### Questions to Ask Your Team

1. "Are we using encoder-only, decoder-only, or encoder-decoder? Why?"
2. "What's our context length limit?"
3. "Are we fine-tuning or prompting a pre-trained model?"
4. "What's our inference latency and cost per query?"
5. "Have we considered smaller models for our use case?"

---

## Interview Preparation

### Where This Knowledge Maps to Job Roles

| Blog Section | ML Engineer | Data Scientist | AI Architect | Engineering Manager |
|---|---|---|---|---|
| Part 1-2: Architecture & Building Blocks | Implement custom transformers, debug gradient flow | Understand what each component does | Design transformer configurations for tasks | Understand why transformers are the standard |
| Part 3-4: Encoder & Decoder | Build encoder-decoder for seq2seq, implement causal masking | Use encoder for embeddings, decoder for generation | Choose encoder-only vs decoder-only vs enc-dec | Ask "which architecture and why?" |
| Part 5: GPT Implementation | Implement weight tying, pre-norm, nucleus sampling | Use GPT for text generation tasks | Design generation pipeline (temperature, top-p) | Budget GPU for training and inference |
| Part 6: Training | Implement warmup schedulers, gradient clipping, checkpointing | Track perplexity, compare baselines | Design training pipelines, set quality gates | Ask "what's our perplexity vs baseline?" |
| Part 7: Variants | Implement BERT/GPT/T5 architectures | Select model architecture for task | Design multi-model system architecture | Compare cost/quality of different architectures |
| Part 8: KV-Caching | Implement cached inference, optimize generation | Profile inference latency | Design serving infrastructure, memory budgets | Budget inference costs per query |

### Likely Questions

**Q: Explain the Transformer architecture at a high level.**
A: Transformers use self-attention instead of recurrence. Each position attends to all other positions (weighted by relevance). This enables parallelization and direct long-range connections. The architecture has encoders (bidirectional) and/or decoders (causal). Each layer has multi-head attention + feed-forward network with residual connections and layer normalization.

**Q: What is the computational complexity of attention?**
A: O(n²) in sequence length for both time and memory. This is because each of n positions attends to all n positions. For 1000 tokens, that's 1 million attention computations per layer. This is why context length is expensive and why efficient attention methods (FlashAttention, sparse attention) are important.

**Q: What's the difference between pre-LayerNorm and post-LayerNorm?**
A: Post-LayerNorm (original): norm after residual addition. Pre-LayerNorm (GPT-2+): norm before attention/FFN. Pre-LayerNorm is more stable for deep networks and is now the standard. The gradient flow is cleaner with pre-norm.

**Q: Why does GPT use decoder-only while BERT uses encoder-only?**
A: GPT is designed for generation—it predicts the next token, so it can only see past tokens (causal mask). BERT is designed for understanding—it fills in masked tokens, so it benefits from seeing both past and future (bidirectional). The task determines the architecture.

**Q: How does weight tying work in transformers?**
A: The embedding matrix (vocab_size x d_model) and the output projection matrix (d_model x vocab_size) are transposed versions of each other. Sharing these weights reduces parameters and provides a useful inductive bias: words that embed similarly should also have similar output probabilities.

**Q: What is KV-caching and why is it essential for inference?**
A: During autoregressive generation, each new token only needs to attend to all previous tokens. Without caching, we recompute K and V projections for all previous positions at every step — O(N^2) total work. KV-caching stores previously computed K and V tensors, so each step only computes projections for the new token and concatenates with cached values — O(N) total work. The tradeoff is memory: the cache scales as O(layers x seq_len x d_model). For large models with long contexts, this cache can consume gigabytes of GPU memory per sequence.

**Q: What is perplexity and how do you interpret it?**
A: Perplexity = exp(cross-entropy loss). It represents how "surprised" the model is on average. A perplexity of P means the model is as uncertain as if it were choosing uniformly among P options at each step. Lower is better. A character-level model on English text might achieve perplexity ~1.5-3.0 (good) vs. vocabulary size (random baseline). For word-level models, perplexities under 20 on standard benchmarks (WikiText-103) are considered strong.

**Q: Why do transformers need learning rate warmup?**
A: At initialization, attention scores are nearly random. A high learning rate can push softmax into extreme regions (near 0 or 1), causing loss spikes or divergence. Warmup gradually increases LR so the model finds a stable loss region before taking large steps. Typical warmup: 1-10% of total training steps. After warmup, cosine decay smoothly reduces LR. This schedule is standard across GPT-2, GPT-3, and LLaMA.

**Q: What is RoPE and why has it replaced sinusoidal encodings?**
A: Rotary Positional Embedding encodes position by rotating Q/K vectors in 2D subspaces. Unlike additive encodings (sinusoidal, learned), RoPE makes the Q·K dot product depend only on relative position difference, not absolute position. This improves length extrapolation — models can handle sequences longer than training context. RoPE adds no trainable parameters and is used by LLaMA, Mistral, and most modern LLMs.

**Q: Why is the embedding scaled by sqrt(d_model)?**
A: Embedding weights are initialized with variance ~1/d_model, so their magnitude decreases with dimension. Positional encodings (sin/cos) have values in [-1, 1]. Without scaling, positional signal would dominate for large d_model. Multiplying embeddings by sqrt(d_model) brings their magnitude to ~O(1), matching positional encoding scale.

**Q: What is nucleus (top-p) sampling?**
A: Top-p sampling keeps the smallest set of tokens whose cumulative probability exceeds threshold p (e.g., 0.9). Unlike top-k (fixed number of candidates), top-p adapts dynamically: peaked distributions get few candidates, flat distributions get many. This produces more coherent text than top-k alone. Production standard: top_p=0.9-0.95 with temperature=0.7-1.0.

---

## Exercises (Do These)

1. **Attention visualization**: Add code to your GPT to extract and visualize attention patterns. What do different heads learn? Do some heads attend to the previous token? Do others attend to specific characters?

2. **Positional encoding comparison**: Implement learned positional embeddings (like GPT) vs sinusoidal (like original Transformer). Train both on Shakespeare for 10 epochs and compare validation perplexity. Which converges faster?

3. **Architecture ablation**: Train variants with different numbers of layers (2, 4, 6), heads (2, 4, 8), and dimensions (64, 128, 256). Plot validation perplexity vs. total parameter count. Is there a clear compute-quality frontier?

4. **Temperature and sampling experiment**: Generate from your trained model using: (a) greedy decoding (temperature=0), (b) temperatures 0.5, 0.8, 1.0, 1.5, (c) top-k with k=5, 20, 50, (d) nucleus/top-p sampling with p=0.9, 0.95. Rate each output for coherence and diversity.

5. **KV-cache implementation**: Modify the GPT `generate()` method to use KV-caching. Benchmark generation speed (tokens/second) with and without caching for sequences of length 100, 500, and 1000. How much speedup do you observe?

6. **Evaluation rigor**: Compute perplexity on a held-out test set (last 5% of Shakespeare). Compare your trained model against: (a) a random baseline (perplexity = vocab_size), (b) a bigram model, (c) your model at epoch 1 vs. epoch 10. Present results in a table.

---

## What's Next

You now have:
- Complete Transformer architecture understanding
- All components implemented from scratch
- Working GPT that generates Shakespeare, evaluated with perplexity
- Knowledge of variants (encoder-only, decoder-only, encoder-decoder)
- Understanding of KV-caching for efficient inference
- Foundation for understanding GPT, BERT, Claude, and other modern LLMs

**Blog 10** covers **pre-trained language models**—how to use BERT, GPT-2, and others from Hugging Face instead of training from scratch. This is where theory meets practical deployment.

**[→ Blog 10: Pre-trained Language Models — Standing on Giants](#)**

---

---

## Self-Assessment: What This Blog Does Well and Where It Falls Short

### What This Blog Does Well

- **Complete from-scratch implementation**: Every transformer component (positional encoding, RoPE, layer norm, multi-head attention, feed-forward, encoder, decoder, GPT) is implemented in working PyTorch code with clear docstrings.
- **Architecture variants**: Covers encoder-only, decoder-only, and encoder-decoder patterns with concrete examples and a comparison table.
- **End-to-end training with perplexity**: Shakespeare mini-GPT with warmup + cosine decay scheduler, perplexity tracking against random baseline, and interpretation guide.
- **RoPE implementation**: Demonstrates rotary positional embeddings with proof that dot product depends on relative position — bridges the gap to modern LLMs.
- **Nucleus (top-p) sampling**: Production-standard generation method implemented alongside temperature and top-k.
- **Memory estimation**: Function that breaks down parameter, attention matrix, and activation memory for any transformer configuration.
- **Engineering depth**: sqrt(d_model) embedding scaling, d_ff=4x rationale, warmup necessity, pre-norm vs post-norm, weight tying.
- **KV-caching**: Working implementation with O(N²) → O(N) complexity reduction and memory formula.

### Where This Blog Falls Short

- **Toy-scale only**: The Shakespeare model is character-level with 128-dim embeddings. The gap to real LLMs (BPE, 4096+ dim, billions of params) is bridged by memory estimation but not by actual scaling experiments.
- **No gradient accumulation**: Essential when training larger models that don't fit in a single batch. Covered in Blog 7 but not repeated here.
- **No beam search**: Only sampling methods (temperature, top-k, top-p) are covered. Beam search remains relevant for translation and summarization.
- **No gradient checkpointing**: Important for memory-efficient training of large models, not covered.
- **No scaling laws**: How perplexity improves with model size and data (Chinchilla/Kaplan) is not discussed.

---

## Architect Sanity Checks

### Check 1: Architecture Mastery
**Question**: What are the key differences between encoder-only, decoder-only, and encoder-decoder transformers?
**Answer: YES** — Encoder-only models (BERT) use bidirectional context for understanding tasks, decoder-only models (GPT) generate sequences autoregressively, and encoder-decoder models (T5) process input context then generate output. A comparison table shows attention patterns and use cases. All three variants are implemented in code.

### Check 2: Implementation Capability
**Question**: Can you build a working transformer from scratch?
**Answer: YES** — A complete implementation covers all components: embeddings, positional encoding, multi-head attention, feed-forward layers, residual connections, and layer normalization. The mini-GPT trains on Shakespeare and generates multi-token sequences.

### Check 3: Foundation Completeness
**Question**: How does this connect to modern LLMs like GPT-4 and Claude?
**Answer: YES** — The blog covers: (1) KV-caching with working implementation and memory formula, (2) RoPE with implementation and relative-position proof, (3) pre-norm architecture (GPT-2+ standard), (4) nucleus sampling (production generation method), (5) warmup + cosine decay training schedule, (6) memory estimation for scaling analysis. Remaining gaps (GQA, FlashAttention internals, scaling laws) are acknowledged but non-blocking — they are covered in Blog 8 and later blogs.

### Check 4: Evaluation Rigor
**Question**: Does the blog teach how to properly evaluate a language model?
**Answer: YES** — The training loop computes perplexity with clear interpretation (exp(CE loss)), compares against a random baseline (perplexity = vocab_size), and includes a perplexity interpretation guide for both character-level and word-level models. The evaluate function reports both loss and perplexity. Train/val loss is plotted alongside baseline.

---

*Questions? Found an error? Comments are open. Technical corrections get priority.*
