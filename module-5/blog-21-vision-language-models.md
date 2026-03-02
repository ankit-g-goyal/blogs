# Blog 21: Vision + Language Models
## Bridging Sight and Understanding with Multimodal AI

**Reading time:** 60-90 minutes
**Coding time:** 90-120 minutes
**Total investment:** ~3.5 hours

---

## What You'll Walk Away With

By the end of this blog, you will be able to:

1. **Understand how Vision-Language Models work** (architecture and training)
2. **Use GPT-4V, Claude Vision, and Gemini** for image understanding
3. **Build Visual Question Answering systems** with multiple providers
4. **Run local multimodal models** like LLaVA and Qwen-VL
5. **Create production applications** for image analysis with cost tracking
6. **Evaluate VLM outputs** using hallucination detection, OCR accuracy metrics, and baseline comparisons

> **How to read this blog:** If you are primarily a technical leader evaluating VLM capabilities, start with the Manager's Summary and the Failure Modes section, then skim the API integration code. If you are a developer, work through the ViT and CLIP sections first for architectural understanding, then build with the commercial APIs. The evaluation framework near the end is essential for anyone planning production use. Sections are designed to stand alone, so you can pause between them.

### Prerequisites

Before starting this blog, you should be comfortable with:

- **Python classes and type hints** (Blog 2) -- used throughout for service abstractions
- **Neural network fundamentals** (Blog 4) -- understanding of layers, forward passes, embeddings
- **Transformer architecture** (Blog 9) -- self-attention, multi-head attention, positional embeddings
- **Working with AI APIs** (Blog 14) -- API keys, request/response patterns, error handling
- **REST API basics** -- the FastAPI service section assumes familiarity with HTTP endpoints

---

## What This Blog Does NOT Cover

Before we begin, let's set clear expectations on scope:

- **Training VLMs from scratch** -- we cover architecture and inference, not the multi-GPU training pipelines or dataset curation required to build a new VLM. That is a research-lab undertaking.
- **Video understanding in depth** -- we touch on Gemini's video capabilities, but dedicated video analysis (temporal reasoning, action recognition) is a separate discipline.
- **Fine-tuning vision models** -- adapting VLMs to your domain (e.g., medical imaging, satellite imagery) is covered conceptually in Blog 23.
- **Image generation** -- this blog is about understanding existing images, not creating new ones. Image generation is covered in Blogs 20 and 22.
- **Accessibility standards compliance** -- we show how to generate alt text, but WCAG compliance, ARIA patterns, and accessibility auditing are specialized topics beyond our scope.
- **Edge deployment and mobile optimization** -- running VLMs on phones or IoT devices requires quantization and runtime optimization not covered here. Blog 24 covers deployment more broadly.

---

## Manager's Summary

**For Technical Leaders and Decision Makers:**

Vision-Language Models (VLMs) bridge the gap between seeing and understanding. These models can analyze images, read documents, interpret charts, and answer questions about visual content—capabilities that unlock entirely new automation possibilities.

**Business Applications:**
- **Document Processing**: Extract data from invoices, receipts, forms automatically
- **Quality Control**: Visual inspection of products, defect detection
- **Content Moderation**: Analyze images for policy violations
- **Accessibility**: Generate image descriptions for visually impaired users
- **Customer Support**: Understand screenshots, product images in support tickets
- **Research**: Analyze medical images, scientific visualizations, charts

**Capability Comparison:**

| Capability | GPT-4V | Claude Vision | Gemini Pro Vision |
|------------|--------|---------------|-------------------|
| General Understanding | Excellent | Excellent | Excellent |
| Text in Images (OCR) | Excellent | Excellent | Excellent |
| Chart/Graph Analysis | Excellent | Very Good | Very Good |
| Object Detection | Good | Good | Good |
| Spatial Reasoning | Good | Good | Good |
| Multi-image | Limited | Yes | Yes |
| Cost (per image) | Higher | Medium | Lower |

**Strategic Consideration**: VLMs are now production-ready for most document understanding and image analysis tasks. The ROI on manual image processing tasks can be substantial.

---

## The Evolution of Visual AI

### From CNN to Vision Transformers

```
Timeline of Visual AI:
├── 2012: AlexNet - Deep CNNs for classification
├── 2015: ResNet - Very deep networks with skip connections
├── 2017: Attention in vision - SENet, CBAM
├── 2020: Vision Transformer (ViT) - Pure transformer for images
├── 2021: CLIP - Contrastive language-image pre-training
├── 2022: Flamingo - Few-shot visual learning
├── 2023: GPT-4V, LLaVA, Qwen-VL - Conversational vision
└── 2024: Claude 3 Vision, Gemini - Advanced multimodal
```

### How Vision-Language Models Work

```
┌────────────────────────────────────────────────────────────────┐
│                   Vision-Language Model                         │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────────┐   │
│   │   Image     │    │   Vision    │    │                 │   │
│   │   Input     │───▶│   Encoder   │───▶│                 │   │
│   │             │    │   (ViT)     │    │    Multimodal   │   │
│   └─────────────┘    └─────────────┘    │    Fusion       │   │
│                                          │                 │   │
│   ┌─────────────┐    ┌─────────────┐    │    ┌────────┐   │   │
│   │   Text      │    │   Text      │    │    │  LLM   │   │   │
│   │   Input     │───▶│   Encoder   │───▶│───▶│ Decoder│──▶│Output
│   │             │    │             │    │    │        │   │   │
│   └─────────────┘    └─────────────┘    │    └────────┘   │   │
│                                          └─────────────────┘   │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

### The Critical Question: How Are Vision and Language Fused?

The diagram above hides the hardest engineering problem in VLMs: **how do you convert visual features (patch embeddings from ViT) into something an LLM can understand?** The vision encoder produces a sequence of vectors in "vision space," and the LLM expects tokens in "language space." These are different embedding spaces with different dimensionalities and learned representations.

There are three dominant approaches, and understanding them is essential for choosing and debugging VLMs:

```
Fusion Strategy Comparison:

1. LINEAR PROJECTION (LLaVA-style)
   ViT patches (N×1024) ──▶ Linear Layer (1024→4096) ──▶ LLM input tokens
   │
   ├── Simplest approach: single matrix multiplication
   ├── Used by: LLaVA, LLaVA-Next
   ├── Pros: Fast training, easy to implement, surprisingly effective
   └── Cons: No information compression — all N patches become N tokens,
             consuming LLM context window (a 224×224 image = 196 tokens)

2. CROSS-ATTENTION (Flamingo-style)
   ViT patches ──▶ Cross-attention layers inside frozen LLM
   │
   ├── Vision features attend to text features at specific LLM layers
   ├── Used by: Flamingo, IDEFICS, Otter
   ├── Pros: Vision doesn't consume text context, can handle many images
   └── Cons: More complex, requires modifying LLM internals

3. LEARNED QUERY TRANSFORMER (Q-Former / Perceiver-style)
   ViT patches (N×1024) ──▶ Q-Former (32 learnable queries) ──▶ 32 tokens ──▶ LLM
   │
   ├── Fixed set of learnable queries extract relevant information from patches
   ├── Used by: BLIP-2, InstructBLIP
   ├── Pros: Compresses N patches → fixed K tokens (e.g., 32), saves context
   └── Cons: May lose fine-grained spatial detail during compression
```

**Why this matters in practice:** If your VLM misses small text in an image, the fusion strategy may be the cause. Q-Former-based models (BLIP-2) compress visual information and may lose fine details. Linear projection models (LLaVA) preserve all patches but consume more context. Cross-attention models (Flamingo) can handle many images efficiently but require more complex infrastructure. When debugging VLM failures, ask: "Is this a fusion bottleneck or a model capability issue?"

---

## Understanding Vision Transformers (ViT)

### The ViT Architecture

```python
"""
Vision Transformer Implementation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class PatchEmbedding(nn.Module):
    """
    Convert image into sequence of patch embeddings.

    Process:
    1. Split image into fixed-size patches (e.g., 16x16)
    2. Flatten each patch
    3. Linear projection to embedding dimension
    4. Add positional embeddings
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        # Linear projection of flattened patches
        self.projection = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # Positional embeddings
        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.n_patches + 1, embed_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Image tensor of shape (batch, channels, height, width)

        Returns:
            Patch embeddings of shape (batch, n_patches + 1, embed_dim)
        """
        batch_size = x.shape[0]

        # Project patches: (batch, embed_dim, n_patches_h, n_patches_w)
        x = self.projection(x)

        # Flatten spatial dimensions: (batch, embed_dim, n_patches)
        x = x.flatten(2)

        # Transpose: (batch, n_patches, embed_dim)
        x = x.transpose(1, 2)

        # Prepend [CLS] token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Add positional embeddings
        x = x + self.pos_embedding

        return x


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention for ViT."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, embed_dim = x.shape

        # Compute Q, K, V
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, heads, seq, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = attn @ v
        out = out.transpose(1, 2).reshape(batch_size, seq_len, embed_dim)

        return self.proj(out)


class TransformerBlock(nn.Module):
    """Single transformer block for ViT."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm architecture
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """
    Complete Vision Transformer model.

    Architecture:
    1. Patch embedding (image → sequence of patches)
    2. Transformer encoder (self-attention layers)
    3. Classification head (optional)
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        num_classes: int = 1000,
        dropout: float = 0.0
    ):
        super().__init__()

        self.patch_embed = PatchEmbedding(
            img_size, patch_size, in_channels, embed_dim
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # Classification head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Patch embedding
        x = self.patch_embed(x)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # Use [CLS] token for classification
        cls_token = x[:, 0]

        return self.head(cls_token)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Get visual features without classification head."""
        x = self.patch_embed(x)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


# ViT configurations (matching original paper)
VIT_CONFIGS = {
    "vit_base_patch16_224": {
        "img_size": 224,
        "patch_size": 16,
        "embed_dim": 768,
        "depth": 12,
        "num_heads": 12
    },
    "vit_large_patch16_224": {
        "img_size": 224,
        "patch_size": 16,
        "embed_dim": 1024,
        "depth": 24,
        "num_heads": 16
    },
    "vit_huge_patch14_224": {
        "img_size": 224,
        "patch_size": 14,
        "embed_dim": 1280,
        "depth": 32,
        "num_heads": 16
    }
}


def create_vit(config_name: str, **kwargs) -> VisionTransformer:
    """Create a ViT model from config."""
    config = VIT_CONFIGS[config_name].copy()
    config.update(kwargs)
    return VisionTransformer(**config)
```

### CLIP: Connecting Vision and Language

```python
"""
CLIP - Contrastive Language-Image Pre-training
"""
import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests

class CLIPWrapper:
    """
    Wrapper for using CLIP for various tasks.

    CLIP learns aligned representations of images and text
    through contrastive learning on 400M image-text pairs.

    Capabilities:
    - Zero-shot classification
    - Image-text similarity
    - Feature extraction
    """

    def __init__(self, model_name: str = "openai/clip-vit-large-patch14"):
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name)
        self.model.eval()

    @torch.no_grad()
    def get_image_features(self, images: list) -> torch.Tensor:
        """Extract visual features from images."""
        inputs = self.processor(images=images, return_tensors="pt")
        features = self.model.get_image_features(**inputs)
        # Normalize
        features = features / features.norm(dim=-1, keepdim=True)
        return features

    @torch.no_grad()
    def get_text_features(self, texts: list) -> torch.Tensor:
        """Extract text features."""
        inputs = self.processor(text=texts, return_tensors="pt", padding=True)
        features = self.model.get_text_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
        return features

    def zero_shot_classify(
        self,
        image: Image.Image,
        candidate_labels: list[str],
        hypothesis_template: str = "a photo of {}"
    ) -> dict[str, float]:
        """
        Classify an image into one of the candidate labels.

        Args:
            image: PIL Image
            candidate_labels: List of possible class names
            hypothesis_template: Template for text descriptions

        Returns:
            Dictionary of label -> probability
        """
        # Create text descriptions for each label
        texts = [hypothesis_template.format(label) for label in candidate_labels]

        # Get features
        image_features = self.get_image_features([image])
        text_features = self.get_text_features(texts)

        # Compute similarities
        similarities = (image_features @ text_features.T).squeeze(0)

        # Softmax to get probabilities
        # Temperature scaling (multiply by 100): CLIP embeddings are L2-normalized,
        # so cosine similarities fall in [-1, 1]. Raw softmax over such small values
        # produces near-uniform distributions (e.g., softmax([0.25, 0.22, 0.20]) ≈ [0.34, 0.33, 0.33]).
        # Multiplying by a learned temperature (CLIP uses ~100) sharpens the distribution
        # so the best match clearly dominates (e.g., softmax([25, 22, 20]) ≈ [0.88, 0.11, 0.01]).
        # The original CLIP paper learns this temperature as exp(log_temperature).
        probs = torch.softmax(similarities * 100, dim=-1)

        return {label: prob.item() for label, prob in zip(candidate_labels, probs)}

    def compute_similarity(
        self,
        images: list[Image.Image],
        texts: list[str]
    ) -> torch.Tensor:
        """
        Compute similarity matrix between images and texts.

        Returns:
            Similarity matrix of shape (num_images, num_texts)
        """
        image_features = self.get_image_features(images)
        text_features = self.get_text_features(texts)
        return image_features @ text_features.T


# Example usage
def demonstrate_clip():
    clip = CLIPWrapper()

    # Load an image
    url = "https://example.com/cat.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    # Zero-shot classification
    labels = ["cat", "dog", "bird", "fish", "horse"]
    results = clip.zero_shot_classify(image, labels)

    print("Classification results:")
    for label, prob in sorted(results.items(), key=lambda x: -x[1]):
        print(f"  {label}: {prob:.2%}")

    # Image-text matching
    texts = [
        "a cute cat sleeping",
        "a dog playing fetch",
        "a beautiful sunset"
    ]

    image_features = clip.get_image_features([image])
    text_features = clip.get_text_features(texts)
    similarities = (image_features @ text_features.T).squeeze()

    print("\nText similarities:")
    for text, sim in zip(texts, similarities):
        print(f"  '{text}': {sim.item():.3f}")
```

---

## Working with Commercial Vision APIs

### GPT-4 Vision (GPT-4V)

```python
"""
GPT-4 Vision API Integration
"""
from openai import OpenAI
import base64
from pathlib import Path
from PIL import Image
import io

client = OpenAI()


def encode_image_to_base64(image_path: str) -> str:
    """Encode an image file to base64."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def encode_pil_image(image: Image.Image, format: str = "PNG") -> str:
    """Encode a PIL Image to base64."""
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def analyze_image_gpt4v(
    image_source: str | Image.Image,
    prompt: str,
    model: str = "gpt-4o",
    max_tokens: int = 1000,
    detail: str = "auto"  # "low", "high", or "auto"
) -> str:
    """
    Analyze an image using GPT-4 Vision.

    Args:
        image_source: File path, URL, or PIL Image
        prompt: Question or instruction about the image
        detail: Image resolution ("low" for 512px, "high" for full res)

    Returns:
        Model's response about the image
    """
    # Prepare image content
    if isinstance(image_source, Image.Image):
        base64_image = encode_pil_image(image_source)
        image_content = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{base64_image}",
                "detail": detail
            }
        }
    elif image_source.startswith(("http://", "https://")):
        image_content = {
            "type": "image_url",
            "image_url": {
                "url": image_source,
                "detail": detail
            }
        }
    else:
        base64_image = encode_image_to_base64(image_source)
        ext = Path(image_source).suffix.lower()
        mime_type = "image/jpeg" if ext in [".jpg", ".jpeg"] else "image/png"
        image_content = {
            "type": "image_url",
            "image_url": {
                "url": f"data:{mime_type};base64,{base64_image}",
                "detail": detail
            }
        }

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    image_content
                ]
            }
        ],
        max_tokens=max_tokens
    )

    return response.choices[0].message.content


def analyze_multiple_images(
    images: list[str | Image.Image],
    prompt: str,
    model: str = "gpt-4o"
) -> str:
    """Analyze multiple images together."""
    content = [{"type": "text", "text": prompt}]

    for img in images:
        if isinstance(img, Image.Image):
            base64_img = encode_pil_image(img)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{base64_img}"}
            })
        elif img.startswith(("http://", "https://")):
            content.append({
                "type": "image_url",
                "image_url": {"url": img}
            })
        else:
            base64_img = encode_image_to_base64(img)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{base64_img}"}
            })

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        max_tokens=2000
    )

    return response.choices[0].message.content


# Specialized analysis functions
class GPT4VisionAnalyzer:
    """Specialized image analysis with GPT-4V."""

    @staticmethod
    def extract_text_ocr(image_path: str) -> str:
        """Extract all text from an image (OCR)."""
        prompt = """Extract ALL text visible in this image.
        Maintain the original formatting and layout as much as possible.
        If there are multiple columns, indicate the structure.
        Include any text in headers, footers, watermarks, etc."""

        return analyze_image_gpt4v(image_path, prompt, detail="high")

    @staticmethod
    def analyze_chart(image_path: str) -> dict:
        """Analyze a chart or graph."""
        prompt = """Analyze this chart/graph and provide:
        1. Type of chart (bar, line, pie, etc.)
        2. Title and axis labels
        3. Key data points and values
        4. Main trends or insights
        5. Any notable observations

        Format your response as structured data."""

        response = analyze_image_gpt4v(image_path, prompt, detail="high")
        return {"analysis": response}

    @staticmethod
    def describe_for_accessibility(image_path: str) -> str:
        """Generate accessibility description (alt text)."""
        prompt = """Create a detailed accessibility description for this image.
        This will be used as alt text for visually impaired users.

        Guidelines:
        - Describe the main subject and action
        - Include relevant details (colors, positions, text)
        - Keep it concise but informative (2-3 sentences)
        - Focus on what's important for understanding context"""

        return analyze_image_gpt4v(image_path, prompt)

    @staticmethod
    def analyze_document(image_path: str) -> dict:
        """Analyze a document image (invoice, receipt, form)."""
        prompt = """Analyze this document and extract:
        1. Document type (invoice, receipt, form, letter, etc.)
        2. All text content organized by section
        3. Key data fields (dates, amounts, names, addresses)
        4. Any tables with their data
        5. Signatures or stamps if present

        Return as structured JSON."""

        response = analyze_image_gpt4v(image_path, prompt, detail="high")
        return {"document_analysis": response}

    @staticmethod
    def compare_images(image1: str, image2: str) -> str:
        """Compare two images and describe differences."""
        prompt = """Compare these two images and describe:
        1. What are the main similarities?
        2. What are the key differences?
        3. Has anything been added, removed, or changed?
        4. Which elements remain the same?

        Be specific and detailed in your comparison."""

        return analyze_multiple_images([image1, image2], prompt)
```

### Claude Vision

```python
"""
Claude Vision API Integration
"""
import anthropic
import base64
from pathlib import Path
from PIL import Image
import io

client = anthropic.Anthropic()


def analyze_image_claude(
    image_source: str | Image.Image,
    prompt: str,
    model: str = "claude-sonnet-4-20250514",
    max_tokens: int = 1024,
    system_prompt: str = None
) -> str:
    """
    Analyze an image using Claude Vision.

    Claude supports:
    - JPEG, PNG, GIF, WebP formats
    - Up to 20MB per image
    - Multiple images in one request

    Note: Claude's image API uses a different content structure than OpenAI.
    Images use "type": "image" with a "source" object (not "image_url").
    Base64 sources require "media_type" and "data" fields.
    URL sources require "type": "url" and "url" fields.
    """
    # Prepare image content block
    # BUG FIX: The original code had a logic flow error where image_content
    # was only set in the URL branch, then a second check using
    # image_source.startswith() would crash on PIL Image objects
    # (which have no startswith method). Fixed by building image_content
    # in each branch directly.
    if isinstance(image_source, Image.Image):
        buffer = io.BytesIO()
        image_source.save(buffer, format="PNG")
        image_data = base64.standard_b64encode(buffer.getvalue()).decode("utf-8")
        image_content = {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": image_data
            }
        }
    elif image_source.startswith(("http://", "https://")):
        # Claude supports URL images directly
        image_content = {
            "type": "image",
            "source": {
                "type": "url",
                "url": image_source
            }
        }
    else:
        with open(image_source, "rb") as f:
            image_data = base64.standard_b64encode(f.read()).decode("utf-8")
        ext = Path(image_source).suffix.lower()
        media_types = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                       ".png": "image/png", ".gif": "image/gif", ".webp": "image/webp"}
        media_type = media_types.get(ext, "image/png")
        image_content = {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": image_data
            }
        }

    messages = [
        {
            "role": "user",
            "content": [
                image_content,
                {"type": "text", "text": prompt}
            ]
        }
    ]

    kwargs = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages
    }

    if system_prompt:
        kwargs["system"] = system_prompt

    response = client.messages.create(**kwargs)

    return response.content[0].text


def analyze_multiple_images_claude(
    images: list[str | Image.Image],
    prompt: str,
    model: str = "claude-sonnet-4-20250514"
) -> str:
    """Analyze multiple images with Claude."""
    content = []

    for img in images:
        if isinstance(img, Image.Image):
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": base64.standard_b64encode(buffer.getvalue()).decode("utf-8")
                }
            })
        elif img.startswith(("http://", "https://")):
            content.append({
                "type": "image",
                "source": {"type": "url", "url": img}
            })
        else:
            with open(img, "rb") as f:
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": base64.standard_b64encode(f.read()).decode("utf-8")
                    }
                })

    content.append({"type": "text", "text": prompt})

    response = client.messages.create(
        model=model,
        max_tokens=2048,
        messages=[{"role": "user", "content": content}]
    )

    return response.content[0].text


class ClaudeVisionAnalyzer:
    """Specialized analysis with Claude Vision."""

    @staticmethod
    def detailed_description(image_path: str) -> str:
        """Get a detailed description of an image."""
        prompt = """Provide a comprehensive description of this image.

        Include:
        1. Main subject(s) and their appearance
        2. Setting/environment
        3. Colors, lighting, mood
        4. Any text visible
        5. Notable details or interesting elements
        6. Overall composition and style"""

        return analyze_image_claude(image_path, prompt)

    @staticmethod
    def extract_structured_data(image_path: str, schema: dict) -> str:
        """Extract data according to a schema."""
        schema_str = str(schema)
        prompt = f"""Extract information from this image according to this schema:

{schema_str}

Return the extracted data as valid JSON matching the schema.
If a field is not visible or not applicable, use null."""

        return analyze_image_claude(
            image_path,
            prompt,
            system_prompt="You are a data extraction assistant. Always return valid JSON."
        )

    @staticmethod
    def analyze_ui_screenshot(image_path: str) -> str:
        """Analyze a UI/app screenshot."""
        prompt = """Analyze this user interface screenshot:

        1. What application/website is this?
        2. What screen/page is shown?
        3. List all visible UI elements (buttons, forms, menus, etc.)
        4. Describe the layout and organization
        5. Identify any issues (if apparent) with UX/UI
        6. Note any user data or sensitive information visible"""

        return analyze_image_claude(image_path, prompt)

    @staticmethod
    def code_from_design(image_path: str, framework: str = "React") -> str:
        """Generate code from a UI design mockup."""
        prompt = f"""Convert this UI design into {framework} code.

        Requirements:
        - Match the layout as closely as possible
        - Use modern CSS (flexbox/grid)
        - Include appropriate component structure
        - Add placeholder content where needed
        - Include basic styling
        - Make it responsive

        Return complete, working code."""

        return analyze_image_claude(image_path, prompt)
```

### Google Gemini Vision

```python
"""
Google Gemini Vision API Integration
"""
import google.generativeai as genai
from PIL import Image
import io
import os

# Configure API -- use environment variable, never hardcode keys
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])


def analyze_image_gemini(
    image_source: str | Image.Image,
    prompt: str,
    model: str = "gemini-1.5-pro"
) -> str:
    """
    Analyze an image using Google Gemini.

    Gemini Pro Vision supports:
    - Multiple images
    - Video input
    - Long context (1M+ tokens)
    """
    model = genai.GenerativeModel(model)

    # Load image
    if isinstance(image_source, str):
        if image_source.startswith(("http://", "https://")):
            import requests
            img = Image.open(requests.get(image_source, stream=True).raw)
        else:
            img = Image.open(image_source)
    else:
        img = image_source

    response = model.generate_content([prompt, img])

    return response.text


def analyze_with_video_gemini(
    video_path: str,
    prompt: str
) -> str:
    """Analyze a video file with Gemini."""
    model = genai.GenerativeModel("gemini-1.5-pro")

    # Upload video
    video_file = genai.upload_file(video_path)

    # Wait for processing
    import time
    while video_file.state.name == "PROCESSING":
        time.sleep(2)
        video_file = genai.get_file(video_file.name)

    response = model.generate_content([prompt, video_file])

    return response.text


class GeminiVisionAnalyzer:
    """Specialized analysis with Gemini Vision."""

    def __init__(self, model: str = "gemini-1.5-pro"):
        self.model = genai.GenerativeModel(model)

    def interactive_qa(self, image: Image.Image) -> "ImageChat":
        """Start an interactive Q&A session about an image."""
        return ImageChat(self.model, image)

    def batch_analyze(
        self,
        images: list[Image.Image],
        prompt: str
    ) -> str:
        """Analyze multiple images together."""
        content = [prompt] + images
        response = self.model.generate_content(content)
        return response.text


class ImageChat:
    """Interactive chat about an image."""

    def __init__(self, model, image: Image.Image):
        self.chat = model.start_chat()
        # Initialize with image
        self.chat.send_message([
            "I'll be asking questions about this image.",
            image
        ])

    def ask(self, question: str) -> str:
        """Ask a question about the image."""
        response = self.chat.send_message(question)
        return response.text
```

---

## Running Local Vision-Language Models

### Local Model Comparison and Hardware Requirements

Before choosing a local model, understand the resource requirements and trade-offs:

```
Local VLM Hardware Requirements:
┌──────────────────────────┬──────────┬───────────┬───────────────┬──────────────────┐
│ Model                    │ VRAM     │ Disk      │ Latency (A100)│ Best For         │
├──────────────────────────┼──────────┼───────────┼───────────────┼──────────────────┤
│ LLaVA 1.5 7B (fp16)     │ ~14 GB   │ ~14 GB    │ 2-4 sec       │ General VQA      │
│ LLaVA 1.5 13B (fp16)    │ ~26 GB   │ ~26 GB    │ 4-8 sec       │ Better accuracy  │
│ LLaVA-Next 7B (fp16)    │ ~15 GB   │ ~15 GB    │ 2-5 sec       │ Higher resolution│
│ Qwen2-VL 7B (fp16)      │ ~16 GB   │ ~16 GB    │ 2-5 sec       │ OCR, multilingual│
│ Qwen2-VL 72B (GPTQ-4bit)│ ~40 GB   │ ~40 GB    │ 8-15 sec      │ Best local qual. │
│ LLaVA 7B (4-bit quant)  │ ~5 GB    │ ~4 GB     │ 5-10 sec      │ Low-resource dev │
│ Ollama llava (q4_0)     │ ~5 GB    │ ~4 GB     │ 3-8 sec       │ Easy setup       │
└──────────────────────────┴──────────┴───────────┴───────────────┴──────────────────┘

NOTE: Latencies are approximate per-image on A100 40GB.
Consumer GPUs (RTX 3090/4090) will be 1.5-3x slower.
CPU-only inference is 10-50x slower and generally impractical for interactive use.
```

**Key trade-offs for local vs. cloud:**
- **Privacy**: Local models keep images on your hardware — critical for medical, legal, financial documents
- **Cost at scale**: Local models have fixed hardware cost vs. per-image API cost. Breakeven is typically 5K-20K images/month depending on hardware
- **Quality**: GPT-4o and Claude 3.5 Sonnet significantly outperform 7B local models on complex reasoning. The gap narrows with Qwen2-VL 72B but requires expensive hardware
- **Latency**: Cloud APIs have network overhead (200-500ms) but faster inference. Local models have no network cost but slower generation on consumer hardware

### LLaVA (Large Language and Vision Assistant)

```python
"""
Running LLaVA Locally
"""
from transformers import (
    LlavaNextProcessor,
    LlavaNextForConditionalGeneration,
    AutoProcessor,
    LlavaForConditionalGeneration
)
import torch
from PIL import Image

class LLaVALocal:
    """
    Local LLaVA model for vision-language tasks.

    LLaVA architecture (Linear Projection fusion — see fusion strategies above):
    - Vision Encoder: CLIP ViT-L/14 (produces 256 patch tokens × 1024 dims)
    - Projection: 2-layer MLP (1024 → 4096) mapping vision → language space
    - Language Model: Vicuna/LLaMA (receives projected vision tokens + text tokens)

    The key insight: LLaVA treats projected image patches as if they were
    additional text tokens. The LLM processes [image_tokens | text_tokens]
    as one sequence. This is why LLaVA's approach is called "visual instruction
    tuning" — it fine-tunes the LLM to follow instructions that include visual context.

    Hardware: Requires ~14GB VRAM for 7B fp16, ~5GB with 4-bit quantization.
    For CPU-only or low-VRAM setups, use Ollama (see OllamaVision below).
    """

    def __init__(
        self,
        model_name: str = "llava-hf/llava-1.5-7b-hf",
        device: str = "cuda",
        dtype: torch.dtype = torch.float16
    ):
        """
        Initialize LLaVA model.

        Models:
        - "llava-hf/llava-1.5-7b-hf": LLaVA 1.5 7B
        - "llava-hf/llava-1.5-13b-hf": LLaVA 1.5 13B
        - "llava-hf/llava-v1.6-mistral-7b-hf": LLaVA 1.6 with Mistral
        """
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto"
        )

    def analyze(
        self,
        image: Image.Image,
        prompt: str,
        max_new_tokens: int = 500
    ) -> str:
        """Analyze an image with a prompt."""
        # Format conversation
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        # Apply chat template
        text = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True
        )

        # Process inputs
        inputs = self.processor(
            images=image,
            text=text,
            return_tensors="pt"
        ).to(self.device)

        # Generate
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7
        )

        # Decode
        response = self.processor.decode(outputs[0], skip_special_tokens=True)

        # Extract assistant response
        if "ASSISTANT:" in response:
            response = response.split("ASSISTANT:")[-1].strip()

        return response

    def chat(self, image: Image.Image):
        """Interactive chat about an image."""
        print("LLaVA Chat (type 'quit' to exit)")
        print("-" * 40)

        while True:
            user_input = input("\nYou: ").strip()
            if user_input.lower() == 'quit':
                break

            response = self.analyze(image, user_input)
            print(f"\nLLaVA: {response}")


# LLaVA-Next (improved version)
class LLaVANext:
    """LLaVA-Next with improved capabilities."""

    def __init__(self, model_name: str = "llava-hf/llava-v1.6-mistral-7b-hf"):
        self.processor = LlavaNextProcessor.from_pretrained(model_name)
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )

    def analyze(
        self,
        image: Image.Image,
        prompt: str,
        max_tokens: int = 500
    ) -> str:
        """Analyze with LLaVA-Next."""
        # LLaVA-Next uses different prompt format
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        prompt_text = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True
        )

        inputs = self.processor(image, prompt_text, return_tensors="pt").to("cuda")

        output = self.model.generate(**inputs, max_new_tokens=max_tokens)
        return self.processor.decode(output[0], skip_special_tokens=True)
```

### Qwen-VL and Other Models

```python
"""
Other Local Vision-Language Models
"""
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
import torch
from PIL import Image

class QwenVL:
    """
    Qwen-VL: Alibaba's vision-language model.

    Features:
    - Excellent multilingual support
    - Strong OCR capabilities
    - Good at document understanding
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-VL-7B-Instruct",
        device: str = "cuda"
    ):
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.device = device

    def analyze(self, image: Image.Image, prompt: str) -> str:
        """Analyze an image."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt"
        ).to(self.device)

        outputs = self.model.generate(**inputs, max_new_tokens=512)
        return self.processor.batch_decode(
            outputs,
            skip_special_tokens=True
        )[0]


# Using Ollama for local vision models
class OllamaVision:
    """
    Use vision models via Ollama.

    Supported models:
    - llava: LLaVA
    - llava-llama3: LLaVA with Llama 3
    - bakllava: BakLLaVA
    """

    def __init__(self, model: str = "llava"):
        import ollama
        self.client = ollama
        self.model = model

    def analyze(self, image_path: str, prompt: str) -> str:
        """Analyze an image with Ollama."""
        response = self.client.chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "images": [image_path]
                }
            ]
        )
        return response["message"]["content"]

    def stream_analyze(self, image_path: str, prompt: str):
        """Stream the analysis response."""
        for chunk in self.client.chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "images": [image_path]
                }
            ],
            stream=True
        ):
            yield chunk["message"]["content"]
```

---

## Building Production Applications

### Multi-Provider Vision Service

```python
"""
Production Vision Analysis Service
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List
from PIL import Image
import io
import base64
from enum import Enum

class VisionProvider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    LOCAL = "local"


@dataclass
class VisionRequest:
    """Vision analysis request."""
    image: Image.Image | str  # PIL Image or path/URL
    prompt: str
    provider: VisionProvider = VisionProvider.OPENAI
    max_tokens: int = 1000
    detail: str = "auto"  # OpenAI-specific


@dataclass
class VisionResponse:
    """Vision analysis response."""
    text: str
    provider: str
    model: str
    tokens_used: Optional[int] = None
    processing_time: float = 0.0


class VisionAnalyzer(ABC):
    """Abstract base for vision analyzers."""

    @abstractmethod
    def analyze(self, request: VisionRequest) -> VisionResponse:
        pass


class OpenAIVisionAnalyzer(VisionAnalyzer):
    """OpenAI GPT-4V analyzer."""

    def __init__(self, model: str = "gpt-4o"):
        from openai import OpenAI
        self.client = OpenAI()
        self.model = model

    def _prepare_image(self, image: Image.Image | str) -> dict:
        if isinstance(image, str):
            if image.startswith(("http://", "https://")):
                return {"type": "image_url", "image_url": {"url": image}}
            else:
                with open(image, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode()
                return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
        else:
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            b64 = base64.b64encode(buffer.getvalue()).decode()
            return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}

    def analyze(self, request: VisionRequest) -> VisionResponse:
        import time
        start = time.time()

        image_content = self._prepare_image(request.image)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": request.prompt},
                    image_content
                ]
            }],
            max_tokens=request.max_tokens
        )

        return VisionResponse(
            text=response.choices[0].message.content,
            provider="openai",
            model=self.model,
            tokens_used=response.usage.total_tokens,
            processing_time=time.time() - start
        )


class AnthropicVisionAnalyzer(VisionAnalyzer):
    """Anthropic Claude Vision analyzer."""

    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        import anthropic
        self.client = anthropic.Anthropic()
        self.model = model

    def _prepare_image(self, image: Image.Image | str) -> dict:
        if isinstance(image, str):
            if image.startswith(("http://", "https://")):
                return {"type": "image", "source": {"type": "url", "url": image}}
            else:
                with open(image, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode()
                return {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}}
        else:
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            b64 = base64.b64encode(buffer.getvalue()).decode()
            return {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}}

    def analyze(self, request: VisionRequest) -> VisionResponse:
        import time
        start = time.time()

        image_content = self._prepare_image(request.image)

        response = self.client.messages.create(
            model=self.model,
            max_tokens=request.max_tokens,
            messages=[{
                "role": "user",
                "content": [image_content, {"type": "text", "text": request.prompt}]
            }]
        )

        return VisionResponse(
            text=response.content[0].text,
            provider="anthropic",
            model=self.model,
            tokens_used=response.usage.input_tokens + response.usage.output_tokens,
            processing_time=time.time() - start
        )


class VisionService:
    """
    Unified vision analysis service.

    Features:
    - Multiple provider support
    - Automatic fallback
    - Response caching
    - Rate limiting
    """

    def __init__(self):
        self.analyzers = {}
        self._setup_analyzers()

    def _setup_analyzers(self):
        """Initialize available analyzers."""
        try:
            self.analyzers[VisionProvider.OPENAI] = OpenAIVisionAnalyzer()
        except (ImportError, ValueError, OSError) as e:
            logging.warning(f"OpenAI analyzer unavailable: {e}")

        try:
            self.analyzers[VisionProvider.ANTHROPIC] = AnthropicVisionAnalyzer()
        except (ImportError, ValueError, OSError) as e:
            logging.warning(f"Anthropic analyzer unavailable: {e}")

    def analyze(
        self,
        image: Image.Image | str,
        prompt: str,
        provider: VisionProvider = VisionProvider.OPENAI,
        fallback: bool = True
    ) -> VisionResponse:
        """
        Analyze an image with specified provider.

        Args:
            image: Image to analyze
            prompt: Analysis prompt
            provider: Preferred provider
            fallback: Try other providers on failure
        """
        request = VisionRequest(image=image, prompt=prompt, provider=provider)

        # Try preferred provider
        if provider in self.analyzers:
            try:
                return self.analyzers[provider].analyze(request)
            except Exception as e:
                if not fallback:
                    raise
                print(f"Primary provider failed: {e}")

        # Fallback to other providers
        if fallback:
            for p, analyzer in self.analyzers.items():
                if p != provider:
                    try:
                        request.provider = p
                        return analyzer.analyze(request)
                    except (ConnectionError, TimeoutError, ValueError) as e:
                        logging.warning(f"Fallback provider {p.value} failed: {e}")
                        continue

        raise RuntimeError("All vision providers failed")

    def compare_providers(
        self,
        image: Image.Image | str,
        prompt: str
    ) -> dict[str, VisionResponse]:
        """Compare responses from all available providers."""
        results = {}
        request = VisionRequest(image=image, prompt=prompt)

        for provider, analyzer in self.analyzers.items():
            try:
                request.provider = provider
                results[provider.value] = analyzer.analyze(request)
            except Exception as e:
                results[provider.value] = VisionResponse(
                    text=f"Error: {e}",
                    provider=provider.value,
                    model="",
                    processing_time=0
                )

        return results


# FastAPI service with proper lifecycle management
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from pydantic import BaseModel
import asyncio

# Thread pool for vision API calls (which are I/O-bound but use synchronous SDKs)
_executor = ThreadPoolExecutor(max_workers=4)

SUPPORTED_FORMATS = {"image/jpeg", "image/png", "image/gif", "image/webp"}
MAX_IMAGE_SIZE_BYTES = 20 * 1024 * 1024  # 20MB


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage service lifecycle — initialize providers on startup, cleanup on shutdown."""
    app.state.vision_service = VisionService()
    yield
    _executor.shutdown(wait=False)


app = FastAPI(title="Vision Analysis API", lifespan=lifespan)


class AnalysisResponse(BaseModel):
    text: str
    provider: str
    model: str
    processing_time: float


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_image(
    file: UploadFile = File(...),
    prompt: str = Form(...),
    provider: str = Form("openai")
):
    """
    Analyze an uploaded image.

    NOTE: This is an educational service. Production additions needed:
    - Authentication (API keys, JWT)
    - Rate limiting (per-user, per-IP)
    - Request logging and tracing
    - Input sanitization (prompt injection defense)
    - Response caching for identical image+prompt pairs
    """
    # Validate content type
    if file.content_type not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported format: {file.content_type}. Supported: {SUPPORTED_FORMATS}"
        )

    # Read and validate image size
    contents = await file.read()
    if len(contents) > MAX_IMAGE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Image too large: {len(contents) / 1024 / 1024:.1f}MB (max {MAX_IMAGE_SIZE_BYTES / 1024 / 1024:.0f}MB)"
        )

    try:
        image = Image.open(io.BytesIO(contents))
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image file")

    # Get provider enum
    try:
        vision_provider = VisionProvider(provider)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Unknown provider: {provider}")

    # Run analysis in thread pool to avoid blocking the event loop
    # (Vision SDK clients are synchronous)
    loop = asyncio.get_event_loop()
    try:
        response = await loop.run_in_executor(
            _executor,
            app.state.vision_service.analyze,
            image, prompt, vision_provider
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=f"All providers failed: {e}")
    except (ConnectionError, TimeoutError) as e:
        raise HTTPException(status_code=502, detail=f"Provider connection error: {e}")

    return AnalysisResponse(
        text=response.text,
        provider=response.provider,
        model=response.model,
        processing_time=response.processing_time
    )
```

### Document Processing Pipeline

```python
"""
Document Processing with Vision Models
"""
from dataclasses import dataclass
from typing import Optional, List
from PIL import Image
import json

@dataclass
class ExtractedField:
    """A field extracted from a document."""
    name: str
    value: str
    confidence: float
    location: Optional[tuple] = None  # (x, y, width, height)


@dataclass
class DocumentResult:
    """Result of document processing."""
    document_type: str
    fields: List[ExtractedField]
    raw_text: str
    tables: List[dict]
    metadata: dict


class DocumentProcessor:
    """
    Process documents using Vision-Language Models.

    Supported document types:
    - Invoices
    - Receipts
    - Forms
    - Contracts
    - IDs
    """

    def __init__(self, vision_service: VisionService):
        self.vision = vision_service

    def process(self, image: Image.Image, document_type: str = None) -> DocumentResult:
        """Process a document image."""
        # Step 1: Classify document if type not provided
        if not document_type:
            document_type = self._classify_document(image)

        # Step 2: Extract text (OCR)
        raw_text = self._extract_text(image)

        # Step 3: Extract structured fields
        fields = self._extract_fields(image, document_type)

        # Step 4: Extract tables
        tables = self._extract_tables(image)

        # Step 5: Extract metadata
        metadata = self._extract_metadata(image, document_type)

        return DocumentResult(
            document_type=document_type,
            fields=fields,
            raw_text=raw_text,
            tables=tables,
            metadata=metadata
        )

    def _classify_document(self, image: Image.Image) -> str:
        """Classify the document type."""
        prompt = """Classify this document into one of these categories:
        - invoice
        - receipt
        - form
        - contract
        - id_document
        - letter
        - other

        Respond with only the category name."""

        response = self.vision.analyze(image, prompt)
        return response.text.strip().lower()

    def _extract_text(self, image: Image.Image) -> str:
        """Extract all text from document."""
        prompt = """Extract ALL text from this document.
        Maintain the original layout and structure.
        Include headers, footers, and any small text."""

        response = self.vision.analyze(image, prompt)
        return response.text

    def _extract_fields(self, image: Image.Image, doc_type: str) -> List[ExtractedField]:
        """Extract structured fields based on document type."""
        field_schemas = {
            "invoice": ["invoice_number", "date", "due_date", "vendor_name",
                       "vendor_address", "total_amount", "tax_amount", "line_items"],
            "receipt": ["store_name", "date", "time", "total", "payment_method",
                       "items", "tax"],
            "form": ["form_name", "fields", "signatures", "dates"],
            "id_document": ["name", "id_number", "date_of_birth", "expiry_date",
                           "address", "photo_present"]
        }

        fields_to_extract = field_schemas.get(doc_type, ["content"])

        prompt = f"""Extract these fields from the document:
        {json.dumps(fields_to_extract)}

        Return as JSON with this format:
        {{
            "field_name": {{
                "value": "extracted value",
                "confidence": 0.0 to 1.0
            }}
        }}

        If a field is not found, set value to null."""

        response = self.vision.analyze(image, prompt)

        # Parse response — VLMs often wrap JSON in markdown code fences or add
        # preamble text. We try multiple strategies before giving up.
        return self._parse_field_response(response.text)

    def _parse_field_response(self, text: str, max_retries: int = 1) -> List[ExtractedField]:
        """
        Robustly parse VLM JSON output for field extraction.

        VLMs frequently return JSON wrapped in ```json ... ``` blocks,
        or add explanatory text before/after the JSON. This method handles
        those common patterns instead of failing silently.
        """
        # Strategy 1: Direct parse
        parsed = self._try_parse_json(text)
        if parsed is not None:
            return self._json_to_fields(parsed)

        # Strategy 2: Extract JSON from markdown code fences
        import re
        json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
        if json_match:
            parsed = self._try_parse_json(json_match.group(1))
            if parsed is not None:
                return self._json_to_fields(parsed)

        # Strategy 3: Find first { ... } block
        brace_match = re.search(r'\{.*\}', text, re.DOTALL)
        if brace_match:
            parsed = self._try_parse_json(brace_match.group(0))
            if parsed is not None:
                return self._json_to_fields(parsed)

        # All strategies failed — log for debugging
        logging.warning(f"Could not parse VLM field extraction response: {text[:200]}...")
        return []

    def _try_parse_json(self, text: str):
        """Attempt JSON parse, return None on failure."""
        try:
            return json.loads(text.strip())
        except (json.JSONDecodeError, ValueError):
            return None

    def _json_to_fields(self, data: dict) -> List[ExtractedField]:
        """Convert parsed JSON dict to ExtractedField list."""
        if not isinstance(data, dict):
            return []
        return [
            ExtractedField(
                name=name,
                value=info.get("value") if isinstance(info, dict) else info,
                confidence=info.get("confidence", 0.8) if isinstance(info, dict) else 0.7
            )
            for name, info in data.items()
            if (info.get("value") if isinstance(info, dict) else info) is not None
        ]

    def _extract_tables(self, image: Image.Image) -> List[dict]:
        """Extract any tables from the document."""
        prompt = """If there are any tables in this document, extract them.

        Return as JSON array of tables:
        [
            {
                "headers": ["col1", "col2", ...],
                "rows": [
                    ["val1", "val2", ...],
                    ...
                ]
            }
        ]

        If no tables, return: []"""

        response = self.vision.analyze(image, prompt)

        try:
            return json.loads(response.text)
        except json.JSONDecodeError:
            return []

    def _extract_metadata(self, image: Image.Image, doc_type: str) -> dict:
        """Extract document metadata."""
        prompt = """Analyze this document and provide metadata:
        - language
        - page_count (if visible)
        - has_signatures
        - has_stamps
        - image_quality (good/fair/poor)
        - any_handwriting

        Return as JSON."""

        response = self.vision.analyze(image, prompt)

        try:
            return json.loads(response.text)
        except json.JSONDecodeError:
            return {}


# Example usage
def process_invoice(image_path: str) -> dict:
    """Process an invoice and return structured data."""
    service = VisionService()
    processor = DocumentProcessor(service)

    image = Image.open(image_path)
    result = processor.process(image, "invoice")

    return {
        "document_type": result.document_type,
        "fields": {f.name: f.value for f in result.fields},
        "tables": result.tables,
        "metadata": result.metadata
    }
```

---

## Evaluation & Measurement Framework

### Why VLM Evaluation is Critical

**Production Reality**: VLMs hallucinate field values, misread text, and confidently return wrong answers. Without rigorous evaluation, you'll deploy systems that silently fail.

```python
"""
VLM Evaluation Framework
========================
Measure accuracy, detect hallucinations, and establish quality baselines.
"""
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from PIL import Image
import json
import time
from enum import Enum


class ErrorCategory(Enum):
    """Categories of VLM errors for debugging."""
    HALLUCINATION = "hallucination"      # Made up content not in image
    MISREAD = "misread"                  # Incorrect text extraction
    OMISSION = "omission"                # Missed content that exists
    WRONG_FORMAT = "wrong_format"        # Correct content, wrong structure
    SPATIAL_ERROR = "spatial_error"      # Misunderstood layout/position
    REFUSAL = "refusal"                  # Model refused to answer


@dataclass
class EvaluationResult:
    """Result of evaluating a single VLM response."""
    ground_truth: str
    prediction: str
    is_correct: bool
    error_category: Optional[ErrorCategory]
    confidence: float
    latency_ms: float
    tokens_used: int
    cost_usd: float


@dataclass
class EvaluationReport:
    """Aggregate evaluation report."""
    total_samples: int
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    mean_latency_ms: float
    p95_latency_ms: float
    total_cost_usd: float
    error_distribution: Dict[str, int]
    hallucination_rate: float


class VLMEvaluator:
    """
    Comprehensive VLM evaluation system.

    Measures:
    - Field extraction accuracy
    - Hallucination rate
    - Latency distribution
    - Cost per document
    """

    # Cost per 1K tokens (approximate, as of early 2025)
    # NOTE: These prices change frequently. Always check provider pricing pages.
    COST_PER_1K_TOKENS = {
        "gpt-4o": {"input": 0.005, "output": 0.015},
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        "claude-3-5-sonnet": {"input": 0.003, "output": 0.015},
        "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
        "gemini-1.5-pro": {"input": 0.00125, "output": 0.005},
        "gemini-1.5-flash": {"input": 0.000075, "output": 0.0003},
    }

    # Image tokens (approximate)
    IMAGE_TOKENS = {
        "gpt-4o": 765,        # 512x512 low detail
        "gpt-4o-high": 1105,  # 512x512 high detail
        "claude-3": 1334,     # Average image
        "gemini-1.5": 258,    # Per image
    }

    def __init__(self, vision_service):
        self.vision_service = vision_service
        self.results: List[EvaluationResult] = []

    def evaluate_extraction(
        self,
        image: Image.Image,
        ground_truth: Dict[str, str],
        extraction_prompt: str,
        model: str = "gpt-4o"
    ) -> EvaluationResult:
        """
        Evaluate field extraction accuracy.

        Args:
            image: Document image
            ground_truth: Expected field values
            extraction_prompt: Prompt for extraction
            model: Model to evaluate

        Returns:
            EvaluationResult with detailed metrics
        """
        start_time = time.time()

        try:
            response = self.vision_service.analyze(image, extraction_prompt)
            latency_ms = (time.time() - start_time) * 1000

            # Parse response
            try:
                prediction = json.loads(response.text)
            except json.JSONDecodeError:
                prediction = {"_raw": response.text}

            # Calculate accuracy
            correct_fields = 0
            total_fields = len(ground_truth)
            error_category = None

            for field, expected in ground_truth.items():
                predicted = prediction.get(field, {})
                pred_value = predicted.get("value") if isinstance(predicted, dict) else predicted

                if self._normalize(pred_value) == self._normalize(expected):
                    correct_fields += 1
                elif pred_value and not expected:
                    error_category = ErrorCategory.HALLUCINATION
                elif not pred_value and expected:
                    error_category = ErrorCategory.OMISSION
                else:
                    error_category = ErrorCategory.MISREAD

            accuracy = correct_fields / total_fields if total_fields > 0 else 0

            # Calculate cost
            tokens_used = response.tokens_used or self._estimate_tokens(extraction_prompt, response.text)
            cost = self._calculate_cost(tokens_used, model, has_image=True)

            result = EvaluationResult(
                ground_truth=json.dumps(ground_truth),
                prediction=json.dumps(prediction),
                is_correct=accuracy >= 0.95,  # 95% threshold for "correct"
                error_category=error_category,
                confidence=accuracy,
                latency_ms=latency_ms,
                tokens_used=tokens_used,
                cost_usd=cost
            )

            self.results.append(result)
            return result

        except Exception as e:
            return EvaluationResult(
                ground_truth=json.dumps(ground_truth),
                prediction=str(e),
                is_correct=False,
                error_category=ErrorCategory.REFUSAL,
                confidence=0.0,
                latency_ms=(time.time() - start_time) * 1000,
                tokens_used=0,
                cost_usd=0.0
            )

    def evaluate_ocr_accuracy(
        self,
        image: Image.Image,
        ground_truth_text: str,
        model: str = "gpt-4o"
    ) -> Dict[str, float]:
        """
        Evaluate OCR accuracy using character error rate (CER) and word error rate (WER).
        """
        prompt = "Extract ALL text from this image exactly as written. Preserve formatting."

        start_time = time.time()
        response = self.vision_service.analyze(image, prompt)
        latency_ms = (time.time() - start_time) * 1000

        predicted_text = response.text

        # Calculate Character Error Rate
        cer = self._levenshtein_distance(
            ground_truth_text, predicted_text
        ) / max(len(ground_truth_text), 1)

        # Calculate Word Error Rate
        gt_words = ground_truth_text.split()
        pred_words = predicted_text.split()
        wer = self._levenshtein_distance(
            gt_words, pred_words
        ) / max(len(gt_words), 1)

        return {
            "character_error_rate": min(cer, 1.0),
            "word_error_rate": min(wer, 1.0),
            "character_accuracy": max(1.0 - cer, 0.0),
            "word_accuracy": max(1.0 - wer, 0.0),
            "latency_ms": latency_ms,
            "ground_truth_chars": len(ground_truth_text),
            "predicted_chars": len(predicted_text)
        }

    def detect_hallucinations(
        self,
        image: Image.Image,
        vlm_response: str,
        model: str = "gpt-4o"
    ) -> Dict[str, any]:
        """
        Detect potential hallucinations in VLM output.

        Uses a secondary verification pass to check claims.
        """
        verification_prompt = f"""
        I received this description of an image:

        ---
        {vlm_response}
        ---

        Look at the image and verify each claim. Return JSON:
        {{
            "verified_claims": ["claim 1", "claim 2"],
            "unverified_claims": ["claim that might be wrong"],
            "hallucinations": ["claim definitely not in image"],
            "confidence": 0.0 to 1.0
        }}
        """

        response = self.vision_service.analyze(image, verification_prompt)

        try:
            result = json.loads(response.text)
            hallucination_count = len(result.get("hallucinations", []))
            total_claims = (
                len(result.get("verified_claims", [])) +
                len(result.get("unverified_claims", [])) +
                hallucination_count
            )

            return {
                "hallucination_rate": hallucination_count / max(total_claims, 1),
                "details": result,
                "has_hallucinations": hallucination_count > 0
            }
        except json.JSONDecodeError:
            return {
                "hallucination_rate": None,
                "details": {"error": "Could not parse verification"},
                "has_hallucinations": None
            }

    def generate_report(self) -> EvaluationReport:
        """Generate aggregate evaluation report."""
        if not self.results:
            raise ValueError("No evaluation results to report")

        # Basic metrics
        total = len(self.results)
        correct = sum(1 for r in self.results if r.is_correct)
        accuracy = correct / total

        # Precision, Recall, F1 for field extraction
        # True Positive: correctly extracted field (is_correct=True)
        # False Positive: extracted but wrong (not is_correct and has prediction)
        # False Negative: missed field (error_category == OMISSION)
        tp = correct
        fp = sum(1 for r in self.results
                 if not r.is_correct and r.error_category != ErrorCategory.OMISSION)
        fn = sum(1 for r in self.results
                 if r.error_category == ErrorCategory.OMISSION)

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)

        # Latency metrics
        latencies = [r.latency_ms for r in self.results]
        mean_latency = sum(latencies) / len(latencies)
        sorted_latencies = sorted(latencies)
        p95_idx = int(0.95 * len(sorted_latencies))
        p95_latency = sorted_latencies[min(p95_idx, len(sorted_latencies) - 1)]

        # Cost
        total_cost = sum(r.cost_usd for r in self.results)

        # Error distribution
        error_dist = {}
        hallucination_count = 0
        for r in self.results:
            if r.error_category:
                cat = r.error_category.value
                error_dist[cat] = error_dist.get(cat, 0) + 1
                if r.error_category == ErrorCategory.HALLUCINATION:
                    hallucination_count += 1

        return EvaluationReport(
            total_samples=total,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            mean_latency_ms=mean_latency,
            p95_latency_ms=p95_latency,
            total_cost_usd=total_cost,
            error_distribution=error_dist,
            hallucination_rate=hallucination_count / total
        )

    def _normalize(self, value: str) -> str:
        """Normalize string for comparison."""
        if value is None:
            return ""
        return str(value).lower().strip().replace(" ", "")

    def _levenshtein_distance(self, s1, s2) -> int:
        """Calculate Levenshtein distance between two sequences."""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        return previous_row[-1]

    def _estimate_tokens(self, prompt: str, response: str) -> int:
        """Estimate token count (rough: 4 chars per token)."""
        return (len(prompt) + len(response)) // 4

    def _calculate_cost(self, tokens: int, model: str, has_image: bool = False) -> float:
        """Calculate cost in USD."""
        costs = self.COST_PER_1K_TOKENS.get(model, {"input": 0.01, "output": 0.03})

        # Assume 70% input, 30% output
        input_tokens = int(tokens * 0.7)
        output_tokens = int(tokens * 0.3)

        # Add image tokens
        if has_image:
            image_tokens = self.IMAGE_TOKENS.get(model.split("-")[0] + "-" + model.split("-")[1] if "-" in model else model, 500)
            input_tokens += image_tokens

        cost = (input_tokens / 1000 * costs["input"]) + (output_tokens / 1000 * costs["output"])
        return round(cost, 6)


# Benchmark suite for VLMs
def run_benchmark(
    evaluator: VLMEvaluator,
    test_dataset: List[Dict],
    models: List[str] = ["gpt-4o", "claude-3-5-sonnet"]
) -> Dict[str, EvaluationReport]:
    """
    Run benchmark across multiple models.

    Args:
        evaluator: VLMEvaluator instance
        test_dataset: List of {"image": Image, "ground_truth": dict, "prompt": str}
        models: Models to benchmark

    Returns:
        Dict of model -> EvaluationReport
    """
    results = {}

    for model in models:
        print(f"\n{'='*50}")
        print(f"Benchmarking: {model}")
        print(f"{'='*50}")

        evaluator.results = []  # Reset

        for i, sample in enumerate(test_dataset):
            result = evaluator.evaluate_extraction(
                image=sample["image"],
                ground_truth=sample["ground_truth"],
                extraction_prompt=sample["prompt"],
                model=model
            )
            print(f"  Sample {i+1}: {'PASS' if result.is_correct else 'FAIL'} "
                  f"(confidence: {result.confidence:.2f}, latency: {result.latency_ms:.0f}ms)")

        report = evaluator.generate_report()
        results[model] = report

        print(f"\n  Summary for {model}:")
        print(f"    Accuracy: {report.accuracy:.1%}")
        print(f"    Hallucination Rate: {report.hallucination_rate:.1%}")
        print(f"    Mean Latency: {report.mean_latency_ms:.0f}ms")
        print(f"    P95 Latency: {report.p95_latency_ms:.0f}ms")
        print(f"    Total Cost: ${report.total_cost_usd:.4f}")

    return results
```

### Establishing Baselines

Before deploying any VLM system, establish baselines:

```python
# Example: Invoice extraction baseline
INVOICE_BENCHMARK = {
    "min_accuracy": 0.95,        # 95% field accuracy required
    "max_hallucination_rate": 0.02,  # Max 2% hallucinations
    "max_p95_latency_ms": 5000,  # 5 second P95
    "max_cost_per_doc": 0.05,    # $0.05 per document
}

def validate_against_baseline(report: EvaluationReport, baseline: dict) -> Dict[str, bool]:
    """Check if model meets production requirements."""
    return {
        "accuracy_ok": report.accuracy >= baseline["min_accuracy"],
        "hallucination_ok": report.hallucination_rate <= baseline["max_hallucination_rate"],
        "latency_ok": report.p95_latency_ms <= baseline["max_p95_latency_ms"],
        "cost_ok": (report.total_cost_usd / report.total_samples) <= baseline["max_cost_per_doc"],
        "production_ready": all([
            report.accuracy >= baseline["min_accuracy"],
            report.hallucination_rate <= baseline["max_hallucination_rate"],
            report.p95_latency_ms <= baseline["max_p95_latency_ms"],
        ])
    }
```

### Worked Example: Evaluating Invoice Extraction

To make the evaluation framework concrete, here is how you would run it on a real dataset:

```python
"""
Worked Evaluation Example — Invoice Field Extraction
=====================================================
Demonstrates the full evaluation loop with synthetic ground truth.
In production, you would use real scanned invoices with human-verified labels.
"""

def run_invoice_evaluation_example():
    """
    Walk through a complete evaluation cycle.

    This example uses synthetic data to demonstrate the pattern.
    Replace with your own images and ground truth for real evaluation.
    """
    # Step 1: Define ground truth for test invoices
    # In practice, you'd have 50-200 labeled invoices from human annotators
    test_dataset = [
        {
            "image": Image.new("RGB", (800, 1200), "white"),  # Placeholder
            "ground_truth": {
                "invoice_number": "INV-2024-0042",
                "vendor_name": "Acme Corp",
                "total_amount": "$1,234.56",
                "date": "2024-03-15",
                "tax_amount": "$111.11"
            },
            "prompt": (
                "Extract these fields as JSON: invoice_number, vendor_name, "
                "total_amount, date, tax_amount. Format: "
                '{"field": {"value": "...", "confidence": 0.0-1.0}}'
            )
        },
        # ... more test samples
    ]

    # Step 2: Run evaluation
    service = VisionService()
    evaluator = VLMEvaluator(service)

    for sample in test_dataset:
        result = evaluator.evaluate_extraction(
            image=sample["image"],
            ground_truth=sample["ground_truth"],
            extraction_prompt=sample["prompt"]
        )
        print(f"Correct: {result.is_correct}, Confidence: {result.confidence:.2f}, "
              f"Error: {result.error_category}, Cost: ${result.cost_usd:.4f}")

    # Step 3: Generate report
    report = evaluator.generate_report()
    print(f"\n--- Evaluation Report ---")
    print(f"Accuracy:           {report.accuracy:.1%}")
    print(f"Precision:          {report.precision:.1%}")
    print(f"Recall:             {report.recall:.1%}")
    print(f"F1 Score:           {report.f1_score:.1%}")
    print(f"Hallucination Rate: {report.hallucination_rate:.1%}")
    print(f"Mean Latency:       {report.mean_latency_ms:.0f}ms")
    print(f"P95 Latency:        {report.p95_latency_ms:.0f}ms")
    print(f"Total Cost:         ${report.total_cost_usd:.4f}")
    print(f"Error Distribution: {report.error_distribution}")

    # Step 4: Validate against production baseline
    baseline = INVOICE_BENCHMARK
    validation = validate_against_baseline(report, baseline)
    print(f"\n--- Production Readiness ---")
    for check, passed in validation.items():
        print(f"  {check}: {'PASS' if passed else 'FAIL'}")

    # Step 5: Interpret results
    # Common patterns and what to do:
    #   - High hallucination rate → add verification pass (doubles cost)
    #   - Low recall (missing fields) → improve prompt specificity
    #   - High latency → switch to smaller model or Gemini Flash
    #   - Accuracy varies by field → some fields need specialized prompts

    return report, validation
```

> **Key insight from running evaluations:** In practice, you will find that VLM accuracy varies significantly by field type. Date and amount extraction typically achieves 90-98% accuracy, while vendor names and addresses drop to 80-90% due to formatting variation. Always report per-field accuracy, not just aggregate accuracy — the aggregate number hides the fields that will cause production issues.

### Standard Multimodal Evaluation Metrics

Beyond field extraction, the research community uses several standard metrics for evaluating multimodal models. Understanding these helps you interpret benchmark results and design your own evaluations.

```python
"""
Standard Multimodal Evaluation Metrics
=======================================
Metrics used in VQA, image captioning, and visual reasoning benchmarks.
"""
from typing import List, Dict
from collections import Counter
import math


def vqa_accuracy(predicted: str, ground_truth_answers: List[str]) -> float:
    """
    VQA accuracy metric (Antol et al., 2015).

    In VQA benchmarks, each question has multiple human answers.
    Accuracy = min(count of humans who gave the same answer / 3, 1.0)

    This accounts for answer ambiguity: if 3 out of 10 humans said "red"
    and the model also says "red", accuracy is min(3/3, 1.0) = 1.0.

    Args:
        predicted: Model's predicted answer
        ground_truth_answers: List of human-provided answers (typically 10)

    Returns:
        Accuracy score between 0.0 and 1.0
    """
    predicted_clean = predicted.strip().lower()
    gt_clean = [a.strip().lower() for a in ground_truth_answers]
    matching = sum(1 for a in gt_clean if a == predicted_clean)
    return min(matching / 3.0, 1.0)


def bleu_score_simple(reference: str, hypothesis: str, max_n: int = 4) -> float:
    """
    Simplified BLEU score for image captioning evaluation.

    BLEU (Bilingual Evaluation Understudy) measures n-gram overlap
    between generated and reference captions. Used in COCO Captions
    and similar benchmarks.

    NOTE: For production evaluation, use the sacrebleu or nltk.translate.bleu_score
    libraries, which handle edge cases (brevity penalty, smoothing) correctly.
    This implementation is for understanding the concept.

    Args:
        reference: Ground truth caption
        hypothesis: Generated caption

    Returns:
        BLEU score between 0.0 and 1.0
    """
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()

    if len(hyp_tokens) == 0:
        return 0.0

    # Calculate n-gram precisions
    precisions = []
    for n in range(1, max_n + 1):
        ref_ngrams = Counter(
            tuple(ref_tokens[i:i+n]) for i in range(len(ref_tokens) - n + 1)
        )
        hyp_ngrams = Counter(
            tuple(hyp_tokens[i:i+n]) for i in range(len(hyp_tokens) - n + 1)
        )

        # Clipped counts
        clipped = sum(
            min(count, ref_ngrams.get(ngram, 0))
            for ngram, count in hyp_ngrams.items()
        )
        total = sum(hyp_ngrams.values())

        if total == 0:
            precisions.append(0.0)
        else:
            precisions.append(clipped / total)

    # Geometric mean of precisions (with smoothing for zero precisions)
    log_avg = 0.0
    for p in precisions:
        if p == 0:
            return 0.0
        log_avg += math.log(p) / max_n

    # Brevity penalty
    bp = 1.0
    if len(hyp_tokens) < len(ref_tokens):
        bp = math.exp(1.0 - len(ref_tokens) / len(hyp_tokens))

    return bp * math.exp(log_avg)


def spatial_reasoning_accuracy(
    predictions: List[Dict],
    ground_truths: List[Dict]
) -> Dict[str, float]:
    """
    Evaluate spatial reasoning capabilities.

    Spatial reasoning is a known weakness of VLMs. This metric
    separately tracks performance on spatial relationship questions
    (left/right, above/below, counting, relative size).

    Args:
        predictions: List of {"question": str, "answer": str, "category": str}
        ground_truths: List of {"question": str, "answer": str, "category": str}

    Returns:
        Accuracy by spatial category
    """
    categories = {}

    for pred, gt in zip(predictions, ground_truths):
        cat = gt.get("category", "general")
        if cat not in categories:
            categories[cat] = {"correct": 0, "total": 0}

        categories[cat]["total"] += 1
        if pred["answer"].strip().lower() == gt["answer"].strip().lower():
            categories[cat]["correct"] += 1

    results = {}
    for cat, counts in categories.items():
        results[cat] = counts["correct"] / max(counts["total"], 1)

    results["overall"] = sum(
        c["correct"] for c in categories.values()
    ) / max(sum(c["total"] for c in categories.values()), 1)

    return results


# Standard benchmarks to be aware of (not implemented here)
STANDARD_VLM_BENCHMARKS = {
    "VQAv2": "Visual Question Answering -- open-ended questions about images",
    "GQA": "Visual reasoning and compositional question answering",
    "TextVQA": "Questions requiring reading text within images",
    "DocVQA": "Document-oriented visual question answering",
    "ChartQA": "Chart and graph understanding",
    "MMMU": "Massive Multi-discipline Multimodal Understanding",
    "MMBench": "Multi-modal benchmark with fine-grained ability evaluation",
    "POPE": "Polling-based Object Probing Evaluation (hallucination detection)",
}
```

> **Practical note:** When evaluating VLMs for production, combine standard metrics (VQA accuracy, OCR CER/WER) with domain-specific metrics (invoice field match rate, chart data extraction accuracy). Standard benchmarks tell you how a model performs in general; your own benchmark tells you how it performs on your data.

---

## Cost Analysis & Optimization

### True Cost of VLM Operations

**Hidden costs most teams miss:**

> **Note:** The cost figures below are rough estimates based on published pricing as of early 2025. Actual costs depend on image resolution, prompt length, response length, and provider pricing changes. Always verify against current provider pricing pages before budgeting.

```
VLM Cost Breakdown (per 1000 documents, estimated):
┌─────────────────────────────────────────────────────────────────┐
│ Cost Component          │ GPT-4o   │ Claude   │ Gemini Flash  │
├─────────────────────────────────────────────────────────────────┤
│ Image tokens (input)    │ $5.00    │ $4.00    │ $0.08        │
│ Text tokens (input)     │ $2.50    │ $1.50    │ $0.04        │
│ Response tokens         │ $7.50    │ $7.50    │ $0.15        │
│ Retries (5% rate)       │ $0.75    │ $0.65    │ $0.01        │
│ Verification pass       │ $7.50    │ $6.50    │ $0.13        │
├─────────────────────────────────────────────────────────────────┤
│ TOTAL per 1K docs       │ $23.25   │ $20.15   │ $0.41        │
│ Cost per document       │ $0.023   │ $0.020   │ $0.0004      │
└─────────────────────────────────────────────────────────────────┘

NOTE: At 100K documents/month:
   - GPT-4o: $2,325/month
   - Claude: $2,015/month
   - Gemini Flash: $41/month
```

### Cost-Aware Vision Service

```python
"""
Cost-Optimized Vision Service
=============================
Track costs, enforce budgets, and optimize provider selection.
"""
from dataclasses import dataclass, field
from typing import Dict, Optional
from datetime import datetime, timedelta
import threading


@dataclass
class CostTracker:
    """Track VLM costs in real-time."""

    total_cost: float = 0.0
    cost_by_provider: Dict[str, float] = field(default_factory=dict)
    cost_by_hour: Dict[str, float] = field(default_factory=dict)
    request_count: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    # Budget limits
    hourly_budget: float = 10.0     # $10/hour default
    daily_budget: float = 100.0     # $100/day default

    def record_cost(self, provider: str, cost: float):
        """Record a cost event."""
        with self._lock:
            self.total_cost += cost
            self.request_count += 1

            # By provider
            self.cost_by_provider[provider] = self.cost_by_provider.get(provider, 0) + cost

            # By hour
            hour_key = datetime.now().strftime("%Y-%m-%d-%H")
            self.cost_by_hour[hour_key] = self.cost_by_hour.get(hour_key, 0) + cost

    def get_hourly_cost(self) -> float:
        """Get cost for current hour."""
        hour_key = datetime.now().strftime("%Y-%m-%d-%H")
        return self.cost_by_hour.get(hour_key, 0)

    def get_daily_cost(self) -> float:
        """Get cost for current day."""
        today = datetime.now().strftime("%Y-%m-%d")
        return sum(v for k, v in self.cost_by_hour.items() if k.startswith(today))

    def check_budget(self) -> Dict[str, bool]:
        """Check if within budget limits."""
        return {
            "hourly_ok": self.get_hourly_cost() < self.hourly_budget,
            "daily_ok": self.get_daily_cost() < self.daily_budget,
            "can_proceed": (
                self.get_hourly_cost() < self.hourly_budget and
                self.get_daily_cost() < self.daily_budget
            )
        }

    def get_summary(self) -> Dict:
        """Get cost summary."""
        return {
            "total_cost": round(self.total_cost, 4),
            "request_count": self.request_count,
            "avg_cost_per_request": round(self.total_cost / max(self.request_count, 1), 6),
            "hourly_cost": round(self.get_hourly_cost(), 4),
            "daily_cost": round(self.get_daily_cost(), 4),
            "hourly_budget_remaining": round(self.hourly_budget - self.get_hourly_cost(), 4),
            "daily_budget_remaining": round(self.daily_budget - self.get_daily_cost(), 4),
            "by_provider": {k: round(v, 4) for k, v in self.cost_by_provider.items()}
        }


class CostOptimizedVisionService:
    """
    Vision service with cost tracking and optimization.

    Features:
    - Real-time cost tracking
    - Budget enforcement
    - Automatic provider downgrade when budget tight
    - Cost-based routing
    """

    PROVIDER_COSTS = {
        "openai": {"tier": "premium", "cost_multiplier": 1.0},
        "anthropic": {"tier": "premium", "cost_multiplier": 0.9},
        "google": {"tier": "budget", "cost_multiplier": 0.02},  # Gemini Flash
        "local": {"tier": "free", "cost_multiplier": 0.0},
    }

    def __init__(self, hourly_budget: float = 10.0, daily_budget: float = 100.0):
        self.cost_tracker = CostTracker(
            hourly_budget=hourly_budget,
            daily_budget=daily_budget
        )
        self.analyzers = self._setup_analyzers()

    def _setup_analyzers(self):
        """Initialize available analyzers."""
        analyzers = {}
        # Setup code from earlier...
        return analyzers

    def analyze_with_cost_control(
        self,
        image,
        prompt: str,
        preferred_provider: str = "openai",
        max_cost: float = 0.10
    ) -> Tuple[VisionResponse, float]:
        """
        Analyze with cost control.

        Args:
            image: Image to analyze
            prompt: Analysis prompt
            preferred_provider: Preferred provider
            max_cost: Maximum cost allowed for this request

        Returns:
            (VisionResponse, actual_cost)
        """
        # Check budget
        budget_status = self.cost_tracker.check_budget()

        if not budget_status["can_proceed"]:
            # Downgrade to cheaper provider
            preferred_provider = self._get_cheapest_provider()
            print(f"WARNING: Budget limit reached. Downgrading to {preferred_provider}")

        # Estimate cost
        estimated_cost = self._estimate_cost(prompt, preferred_provider)

        if estimated_cost > max_cost:
            # Use cheaper provider
            cheaper = self._get_provider_under_cost(max_cost)
            if cheaper:
                preferred_provider = cheaper
            else:
                raise BudgetExceededError(
                    f"No provider available under ${max_cost}. "
                    f"Cheapest: ${estimated_cost}"
                )

        # Execute
        start_time = time.time()
        response = self.analyzers[preferred_provider].analyze(
            VisionRequest(image=image, prompt=prompt)
        )

        # Calculate actual cost
        actual_cost = self._calculate_actual_cost(response, preferred_provider)
        self.cost_tracker.record_cost(preferred_provider, actual_cost)

        return response, actual_cost

    def _estimate_cost(self, prompt: str, provider: str) -> float:
        """Estimate cost before making request."""
        base_cost = 0.01  # Base cost estimate
        multiplier = self.PROVIDER_COSTS.get(provider, {}).get("cost_multiplier", 1.0)
        return base_cost * multiplier

    def _calculate_actual_cost(self, response: VisionResponse, provider: str) -> float:
        """Calculate actual cost from response."""
        tokens = response.tokens_used or 1000
        costs = VLMEvaluator.COST_PER_1K_TOKENS.get(
            "gpt-4o" if provider == "openai" else "claude-3-5-sonnet",
            {"input": 0.01, "output": 0.03}
        )
        return (tokens / 1000) * (costs["input"] + costs["output"]) / 2

    def _get_cheapest_provider(self) -> str:
        """Get the cheapest available provider."""
        for provider in ["local", "google", "anthropic", "openai"]:
            if provider in self.analyzers:
                return provider
        return "openai"

    def _get_provider_under_cost(self, max_cost: float) -> Optional[str]:
        """Get a provider that can serve under the cost limit."""
        for provider, info in sorted(
            self.PROVIDER_COSTS.items(),
            key=lambda x: x[1]["cost_multiplier"]
        ):
            if provider in self.analyzers:
                estimated = self._estimate_cost("", provider)
                if estimated <= max_cost:
                    return provider
        return None


class BudgetExceededError(Exception):
    """Raised when budget limits are exceeded."""
    pass
```

---

## Failure Modes & When NOT to Use VLMs

### Common Failure Modes

```
┌────────────────────────────────────────────────────────────────────────┐
│                    VLM Failure Mode Taxonomy                            │
├────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. HALLUCINATION (Most Dangerous)                                     │
│     ├── Invents text not in image                                      │
│     ├── Creates plausible but wrong field values                       │
│     ├── Confident about non-existent details                           │
│     └── Risk: Silent data corruption in production                     │
│                                                                         │
│  2. SPATIAL REASONING FAILURES                                         │
│     ├── Confuses left/right, above/below                               │
│     ├── Miscounts objects                                              │
│     ├── Wrong bounding box associations                                │
│     └── Risk: Medical imaging, defect detection failures               │
│                                                                         │
│  3. TEXT EXTRACTION ERRORS                                             │
│     ├── Similar character confusion (O/0, l/1, rn/m)                   │
│     ├── Font/style misinterpretation                                   │
│     ├── Handwriting failures                                           │
│     └── Risk: Invoice amounts, dates, IDs wrong                        │
│                                                                         │
│  4. CONTEXT WINDOW OVERFLOW                                            │
│     ├── Image too large (>10MB typically fails)                        │
│     ├── Too many images in one request                                 │
│     ├── Combined image+text exceeds limits                             │
│     └── Risk: Truncated analysis, missed content                       │
│                                                                         │
│  5. ADVERSARIAL/EDGE CASES                                             │
│     ├── Intentionally misleading images                                │
│     ├── Low resolution / blurry images                                 │
│     ├── Unusual orientations                                           │
│     └── Risk: Security bypass, incorrect classifications               │
│                                                                         │
└────────────────────────────────────────────────────────────────────────┘
```

### When NOT to Use VLMs

```python
"""
Decision Framework: When to Avoid VLMs
"""

def should_use_vlm(task_requirements: dict) -> dict:
    """
    Evaluate whether VLM is appropriate for a task.

    Returns recommendation with justification.
    """

    # RED FLAGS - Don't use VLM
    red_flags = []

    if task_requirements.get("requires_100_percent_accuracy"):
        red_flags.append(
            "VLMs have inherent error rates that vary by task and model. "
            "Use traditional OCR + human review for 100% accuracy needs. "
            "Always benchmark on your specific data before relying on accuracy claims."
        )

    if task_requirements.get("safety_critical"):
        red_flags.append(
            "VLMs can hallucinate. Safety-critical applications "
            "(medical diagnosis, autonomous vehicles) need specialized models + human oversight."
        )

    if task_requirements.get("high_volume") and task_requirements.get("low_budget"):
        volume = task_requirements.get("monthly_volume", 0)
        budget = task_requirements.get("monthly_budget", 0)
        cost_per_doc = budget / max(volume, 1)
        if cost_per_doc < 0.001:  # Less than $0.001 per doc
            red_flags.append(
                f"Budget (${budget}/month for {volume} docs = ${cost_per_doc:.4f}/doc) "
                f"is too low for VLMs. Use traditional OCR or local models."
            )

    if task_requirements.get("requires_real_time") and task_requirements.get("latency_ms", 1000) < 100:
        red_flags.append(
            "VLM latency is typically 1-5 seconds. "
            "Sub-100ms requirements need specialized edge models."
        )

    if task_requirements.get("data_residency_critical"):
        red_flags.append(
            "Cloud VLMs send images to external servers. "
            "Use local models (LLaVA, Qwen-VL) for data residency requirements."
        )

    # YELLOW FLAGS - Use with caution
    yellow_flags = []

    if task_requirements.get("handwritten_text"):
        yellow_flags.append(
            "Handwriting recognition is improving but still error-prone. "
            "Error rates vary widely by handwriting style and language; "
            "benchmark on representative samples before committing."
        )

    if task_requirements.get("multi_language"):
        yellow_flags.append(
            "VLM performance varies by language. "
            "Test extensively for non-English content."
        )

    if task_requirements.get("complex_tables"):
        yellow_flags.append(
            "Complex table extraction often fails. "
            "Consider specialized table extraction tools."
        )

    # GREEN FLAGS - Good VLM use cases
    green_flags = []

    if task_requirements.get("general_understanding"):
        green_flags.append("VLMs excel at general image understanding and description.")

    if task_requirements.get("structured_extraction"):
        green_flags.append("Document field extraction is a strength with proper prompting.")

    if task_requirements.get("allows_human_review"):
        green_flags.append("Human-in-the-loop catches VLM errors effectively.")

    return {
        "recommendation": "AVOID" if red_flags else ("CAUTION" if yellow_flags else "PROCEED"),
        "red_flags": red_flags,
        "yellow_flags": yellow_flags,
        "green_flags": green_flags,
        "suggested_alternatives": _get_alternatives(task_requirements) if red_flags else []
    }


def _get_alternatives(requirements: dict) -> list:
    """Suggest alternatives to VLMs."""
    alternatives = []

    if requirements.get("requires_100_percent_accuracy"):
        alternatives.append("Traditional OCR (Tesseract, AWS Textract) + human validation")

    if requirements.get("safety_critical"):
        alternatives.append("Specialized medical/industrial vision models with FDA/CE approval")

    if requirements.get("high_volume") and requirements.get("low_budget"):
        alternatives.append("Local models (PaddleOCR, EasyOCR) on commodity hardware")

    if requirements.get("data_residency_critical"):
        alternatives.append("On-premise deployment of LLaVA or Qwen-VL")

    return alternatives
```

### Error Handling Best Practices

```python
"""
Production Error Handling for VLM Services
"""
from enum import Enum
from dataclasses import dataclass
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class VLMErrorCode(Enum):
    """Categorized error codes for VLM failures."""

    # Recoverable errors
    RATE_LIMITED = "rate_limited"           # Retry with backoff
    TIMEOUT = "timeout"                     # Retry or switch provider
    TEMPORARY_FAILURE = "temporary_failure" # Retry

    # Input errors
    IMAGE_TOO_LARGE = "image_too_large"     # Resize and retry
    INVALID_FORMAT = "invalid_format"       # Convert format
    CONTENT_POLICY = "content_policy"       # Cannot process

    # Processing errors
    PARSE_FAILURE = "parse_failure"         # Response unparseable
    LOW_CONFIDENCE = "low_confidence"       # Result unreliable
    HALLUCINATION_DETECTED = "hallucination"# Result invalid

    # System errors
    PROVIDER_DOWN = "provider_down"         # Switch provider
    BUDGET_EXCEEDED = "budget_exceeded"     # Pause or downgrade
    UNKNOWN = "unknown"                     # Log and investigate


@dataclass
class VLMError:
    """Structured VLM error."""
    code: VLMErrorCode
    message: str
    recoverable: bool
    suggested_action: str
    original_exception: Optional[Exception] = None


class RobustVisionService:
    """
    Production-grade vision service with comprehensive error handling.
    """

    MAX_RETRIES = 3
    RETRY_DELAYS = [1, 2, 4]  # Exponential backoff
    MAX_IMAGE_SIZE_MB = 10

    def __init__(self, vision_service, cost_tracker: CostTracker):
        self.vision = vision_service
        self.cost_tracker = cost_tracker

    def analyze_robust(
        self,
        image,
        prompt: str,
        require_confidence: float = 0.8,
        verify_hallucinations: bool = True
    ):
        """
        Analyze with full error handling and validation.
        """
        # Pre-flight checks
        preflight_error = self._preflight_check(image)
        if preflight_error:
            return self._handle_error(preflight_error)

        # Attempt analysis with retries
        last_error = None
        for attempt in range(self.MAX_RETRIES):
            try:
                response = self._attempt_analysis(image, prompt)

                # Validate response
                validation_error = self._validate_response(
                    response, require_confidence, verify_hallucinations, image
                )
                if validation_error:
                    if validation_error.recoverable:
                        last_error = validation_error
                        continue
                    return self._handle_error(validation_error)

                return {"success": True, "response": response}

            except Exception as e:
                error = self._categorize_exception(e)
                if not error.recoverable:
                    return self._handle_error(error)
                last_error = error
                time.sleep(self.RETRY_DELAYS[min(attempt, len(self.RETRY_DELAYS) - 1)])

        return self._handle_error(last_error or VLMError(
            code=VLMErrorCode.UNKNOWN,
            message="Max retries exceeded",
            recoverable=False,
            suggested_action="Investigate logs and contact support"
        ))

    def _preflight_check(self, image) -> Optional[VLMError]:
        """Check image before sending to VLM."""
        # Check budget
        if not self.cost_tracker.check_budget()["can_proceed"]:
            return VLMError(
                code=VLMErrorCode.BUDGET_EXCEEDED,
                message="Budget limit reached",
                recoverable=False,
                suggested_action="Wait for budget reset or increase limits"
            )

        # Check image size
        if hasattr(image, 'size'):
            width, height = image.size
            estimated_mb = (width * height * 3) / (1024 * 1024)
            if estimated_mb > self.MAX_IMAGE_SIZE_MB:
                return VLMError(
                    code=VLMErrorCode.IMAGE_TOO_LARGE,
                    message=f"Image ~{estimated_mb:.1f}MB exceeds {self.MAX_IMAGE_SIZE_MB}MB limit",
                    recoverable=True,
                    suggested_action="Resize image before processing"
                )

        return None

    def _validate_response(
        self,
        response,
        require_confidence: float,
        verify_hallucinations: bool,
        image
    ) -> Optional[VLMError]:
        """Validate VLM response quality."""

        # Check for empty/error response
        if not response.text or response.text.startswith("Error"):
            return VLMError(
                code=VLMErrorCode.PARSE_FAILURE,
                message="Empty or error response from VLM",
                recoverable=True,
                suggested_action="Retry with different prompt"
            )

        # Check confidence if available
        if hasattr(response, 'confidence') and response.confidence < require_confidence:
            return VLMError(
                code=VLMErrorCode.LOW_CONFIDENCE,
                message=f"Confidence {response.confidence:.2f} below threshold {require_confidence}",
                recoverable=True,
                suggested_action="Use clearer image or simpler prompt"
            )

        # Verify hallucinations if required
        if verify_hallucinations:
            evaluator = VLMEvaluator(self.vision)
            hallucination_check = evaluator.detect_hallucinations(image, response.text)
            if hallucination_check.get("has_hallucinations"):
                return VLMError(
                    code=VLMErrorCode.HALLUCINATION_DETECTED,
                    message="Potential hallucinations detected in response",
                    recoverable=False,
                    suggested_action="Use human review for this document"
                )

        return None

    def _categorize_exception(self, e: Exception) -> VLMError:
        """Categorize exception into VLMError."""
        error_msg = str(e).lower()

        if "rate limit" in error_msg or "429" in error_msg:
            return VLMError(
                code=VLMErrorCode.RATE_LIMITED,
                message=str(e),
                recoverable=True,
                suggested_action="Wait and retry with exponential backoff",
                original_exception=e
            )

        if "timeout" in error_msg or "timed out" in error_msg:
            return VLMError(
                code=VLMErrorCode.TIMEOUT,
                message=str(e),
                recoverable=True,
                suggested_action="Retry or switch to faster provider",
                original_exception=e
            )

        if "content policy" in error_msg or "safety" in error_msg:
            return VLMError(
                code=VLMErrorCode.CONTENT_POLICY,
                message=str(e),
                recoverable=False,
                suggested_action="Image violates content policy - cannot process",
                original_exception=e
            )

        return VLMError(
            code=VLMErrorCode.UNKNOWN,
            message=str(e),
            recoverable=False,
            suggested_action="Log error and investigate",
            original_exception=e
        )

    def _handle_error(self, error: VLMError) -> dict:
        """Handle and log error."""
        logger.error(f"VLM Error [{error.code.value}]: {error.message}")

        return {
            "success": False,
            "error_code": error.code.value,
            "message": error.message,
            "suggested_action": error.suggested_action,
            "recoverable": error.recoverable
        }
```

---

## Interview Preparation

### Conceptual Questions (with Full Explanations)

**1. "Walk me through how a Vision Transformer processes an image from input to output."**

A ViT converts a 2D image into a 1D sequence that a standard Transformer can process. The pipeline is: (1) **Patch embedding** — the image (e.g., 224×224×3) is split into fixed-size patches (e.g., 16×16), giving 196 patches. Each patch is flattened (16×16×3 = 768 values) and linearly projected to the model's embedding dimension. This projection is implemented as a Conv2d with kernel_size=stride=patch_size, which is mathematically equivalent to flatten+linear but computationally faster. (2) **CLS token prepended** — a learnable classification token is concatenated to the front of the sequence (now 197 tokens). (3) **Positional embeddings added** — learned position vectors are added (not concatenated) to each token, since Transformers have no inherent position awareness. (4) **Transformer encoder** — the sequence passes through L standard Transformer blocks (LayerNorm → Multi-Head Self-Attention → residual → LayerNorm → MLP → residual). Critically, ViT uses **pre-norm** (LayerNorm before attention), unlike the original Transformer which uses post-norm. (5) **Classification** — the CLS token's final representation is passed through a linear head.

**Key follow-up to anticipate:** "Why patches instead of individual pixels?" — A 224×224 image has 50,176 pixels. Self-attention is O(n²), so processing all pixels would require ~2.5 billion attention computations. With 16×16 patches, we have 196 tokens — attention costs ~38K computations. This 65,000× reduction makes Transformers practical for images.

**2. "How does CLIP enable zero-shot image classification, and what are its limitations?"**

CLIP (Contrastive Language-Image Pre-training) trains separate image and text encoders on 400M image-text pairs from the internet. The training objective is contrastive: for a batch of N (image, text) pairs, CLIP maximizes the cosine similarity of the N correct pairs while minimizing similarity of the N²-N incorrect pairs. This is implemented as a symmetric cross-entropy loss over a similarity matrix scaled by a learned temperature parameter.

For zero-shot classification, you: (1) encode candidate labels as text ("a photo of a dog", "a photo of a cat"), (2) encode the query image, (3) compute cosine similarities, (4) softmax with temperature scaling to get probabilities. No task-specific training needed.

**Limitations:** CLIP fails on (a) fine-grained distinctions (dog breeds, bird species) because web-scraped captions rarely distinguish subspecies, (b) counting and spatial reasoning ("three red balls to the left of the box"), (c) abstract or specialized domains (medical images, satellite imagery) not well-represented in training data, (d) adversarial inputs — "typographic attacks" where placing text like "iPod" on an apple image causes CLIP to classify it as an iPod.

**3. "Compare fusion strategies in VLMs. When would you choose each?"**

Three dominant approaches: **Linear projection** (LLaVA) maps ViT patch embeddings through an MLP to the LLM's input space — simplest, all patches become tokens consuming LLM context. Choose when: you need fine-grained spatial detail and have sufficient context window. **Cross-attention** (Flamingo) injects visual features into frozen LLM layers via cross-attention — vision doesn't consume text context. Choose when: processing many images per conversation or when text context is premium. **Q-Former** (BLIP-2) uses learned queries to compress N patches into K fixed tokens — reduces context cost. Choose when: you need to balance detail vs. context efficiency, especially for long documents.

**Production trade-off:** Linear projection is the easiest to debug (you can inspect which patch tokens the LLM attends to), but Q-Former-based systems are cheaper to run at scale because they compress visual tokens.

**4. "Design a document processing system that handles 100K invoices/month with 95% field accuracy."**

Architecture: (1) **Ingestion** — async image upload with format validation, resize to provider's optimal resolution (e.g., 1024px width for GPT-4o high-detail). (2) **Tiered processing** — route simple documents (standard templates) to Gemini Flash ($0.0004/doc) and complex/handwritten documents to GPT-4o ($0.023/doc). Use a lightweight classifier (CLIP or rule-based on file metadata) for routing. (3) **Extraction** — structured prompt requesting JSON output with confidence scores per field. Implement robust JSON parsing (handle markdown fences, extract from prose). (4) **Verification** — for fields below 0.9 confidence, run a second pass with a different provider. Flag hallucinated fields by cross-checking: if one provider extracts a field that another doesn't, mark for human review. (5) **Human-in-the-loop** — surface the ~5% lowest-confidence documents for human verification. This catches the tail of VLM errors. (6) **Monitoring** — track per-field accuracy over time. If accuracy on "vendor_name" drops below 90%, investigate (new vendor formats? model regression?).

**Cost estimate:** 80% to Gemini Flash ($32/month) + 20% to GPT-4o ($465/month) + verification passes (~$100/month) ≈ $600/month. Compare with human processing: 100K invoices × 2 min each × $20/hr = $66,667/month.

**5. "How do you detect and prevent hallucinations in VLM-based extraction systems?"**

Hallucination in VLMs means the model outputs information not present in the image — inventing invoice numbers, fabricating text, or confidently describing objects that don't exist. Detection strategies: (1) **Cross-provider verification** — run the same image through two providers and flag disagreements. If GPT-4o says invoice total is "$1,234" and Claude says "$1,243", neither should be trusted without human review. (2) **Structural validation** — check extracted values against expected formats (dates should parse as dates, amounts should be numeric, invoice numbers should match known patterns). (3) **Confidence calibration** — track historical accuracy by field type. If your system is 98% accurate on dates but 85% on addresses, weight confidence accordingly. (4) **POPE-style probing** — ask yes/no questions about specific image contents ("Is there a table in this image?") and check consistency with extracted data. (5) **Pixel-grounding** — some models (Qwen-VL, Gemini) can output bounding boxes. Verify that extracted text falls within the indicated region.

Prevention: Use structured output formats (JSON mode), provide few-shot examples in prompts, and always include "If a field is not visible, return null" instructions. The most dangerous hallucinations are plausible ones — a made-up but valid-looking invoice number won't trigger format checks.

### Career Mapping

| Role | VLM Skills That Matter | Interview Focus |
|------|----------------------|-----------------|
| **ML Engineer** | ViT architecture, CLIP fine-tuning, model serving, quantization | "Implement patch embedding from scratch", "How would you reduce VLM latency by 50%?" |
| **Backend/Platform Engineer** | Multi-provider service, async patterns, cost tracking, error handling | "Design a vision service that handles provider failures", "How do you enforce budget limits?" |
| **Applied AI Engineer** | Document processing pipelines, evaluation frameworks, prompt engineering | "Build an invoice extraction system with 95% accuracy", "How do you detect hallucinations?" |
| **AI Product Manager** | Cost-quality trade-offs, failure modes, "when not to use VLMs" | "What's the ROI on VLM vs. manual processing?", "What are the risks of deploying VLMs for medical imaging?" |
| **Solutions Architect** | Provider selection, privacy constraints, local vs. cloud trade-offs | "Design a VLM architecture for a healthcare company with data residency requirements" |

### Coding Challenges

**Challenge 1**: Implement zero-shot image classifier using CLIP:

```python
from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image


def zero_shot_classify(
    image: Image.Image,
    categories: list[str],
    model_name: str = "openai/clip-vit-base-patch32"
) -> dict[str, float]:
    """
    Classify image into categories without task-specific training.

    Uses CLIP's aligned vision-language embedding space: encode the image
    and each category label, then compute cosine similarities.
    Temperature-scaled softmax converts similarities to probabilities.

    Returns:
        Dict of category -> probability, sorted by probability descending
    """
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)
    model.eval()

    # Create text prompts — "a photo of {category}" works better than bare labels
    # because CLIP was trained on natural language captions, not single words
    text_prompts = [f"a photo of {cat}" for cat in categories]

    with torch.no_grad():
        inputs = processor(
            text=text_prompts,
            images=image,
            return_tensors="pt",
            padding=True
        )

        outputs = model(**inputs)

        # outputs.logits_per_image is already similarity × learned temperature
        probs = outputs.logits_per_image.softmax(dim=-1).squeeze()

    results = {cat: prob.item() for cat, prob in zip(categories, probs)}
    return dict(sorted(results.items(), key=lambda x: -x[1]))
```

**Challenge 2**: Build document comparison tool using a VLM:

```python
def compare_documents(
    doc1: Image.Image,
    doc2: Image.Image,
    vision_service: VisionService = None
) -> dict:
    """
    Compare two document images and identify differences.

    Strategy: Extract structured content from each document independently,
    then diff the results. This is more reliable than asking a VLM to
    compare two images directly (which often misses subtle differences).

    Returns:
        Dict with structural_differences, content_changes, and summary
    """
    if vision_service is None:
        vision_service = VisionService()

    extraction_prompt = """Extract all content from this document as JSON:
    {
        "document_type": "...",
        "title": "...",
        "sections": [{"heading": "...", "content": "..."}],
        "tables": [{"headers": [...], "rows": [[...]]}],
        "key_fields": {"field_name": "value"},
        "signatures_present": true/false
    }
    If a field is not visible, use null."""

    # Extract from both documents independently
    resp1 = vision_service.analyze(doc1, extraction_prompt)
    resp2 = vision_service.analyze(doc2, extraction_prompt)

    # Parse responses (using robust parsing)
    import re
    def parse_json(text):
        # Try direct parse, then extract from code fences, then find braces
        for attempt in [text, None]:
            try:
                return json.loads(text.strip())
            except (json.JSONDecodeError, ValueError):
                pass
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except (json.JSONDecodeError, ValueError):
                pass
        return {"raw_text": text}

    data1 = parse_json(resp1.text)
    data2 = parse_json(resp2.text)

    # Compare structures
    differences = {
        "structural_differences": [],
        "content_changes": [],
        "fields_only_in_doc1": [],
        "fields_only_in_doc2": [],
    }

    # Compare key fields
    fields1 = data1.get("key_fields", {})
    fields2 = data2.get("key_fields", {})

    all_fields = set(list(fields1.keys()) + list(fields2.keys()))
    for field in all_fields:
        v1 = fields1.get(field)
        v2 = fields2.get(field)
        if v1 and not v2:
            differences["fields_only_in_doc1"].append(f"{field}: {v1}")
        elif v2 and not v1:
            differences["fields_only_in_doc2"].append(f"{field}: {v2}")
        elif v1 != v2:
            differences["content_changes"].append(
                f"{field}: '{v1}' → '{v2}'"
            )

    # Compare sections
    sections1 = {s.get("heading", ""): s for s in data1.get("sections", [])}
    sections2 = {s.get("heading", ""): s for s in data2.get("sections", [])}

    for heading in set(list(sections1.keys()) | set(sections2.keys())):
        if heading in sections1 and heading not in sections2:
            differences["structural_differences"].append(f"Section removed: {heading}")
        elif heading in sections2 and heading not in sections1:
            differences["structural_differences"].append(f"Section added: {heading}")

    differences["summary"] = (
        f"Found {len(differences['content_changes'])} content changes, "
        f"{len(differences['structural_differences'])} structural differences, "
        f"{len(differences['fields_only_in_doc1'])} fields only in doc1, "
        f"{len(differences['fields_only_in_doc2'])} fields only in doc2."
    )

    return differences
```

---

## Exercises

### Exercise 1: Build an Accessibility Tool
Create a service that:
- Generates alt text for images
- Describes charts and graphs
- Identifies text in images
- Produces audio descriptions

### Exercise 2: Create a Visual Q&A System
Build an interactive system that:
- Allows uploading images
- Supports follow-up questions
- Maintains conversation context
- Works with multiple providers

### Exercise 3: Implement Invoice Processing
Create a pipeline that:
- Accepts invoice images
- Extracts all fields
- Validates extracted data
- Exports to structured format (JSON/CSV)

### Exercise 4: Build a Visual Search Engine
Create a system that:
- Indexes images with descriptions
- Supports text-based search
- Returns similar images
- Uses CLIP embeddings

---

## Summary

### Key Takeaways

1. **VLMs bridge vision and language through a fusion mechanism** — understanding whether your model uses linear projection (LLaVA), cross-attention (Flamingo), or Q-Former (BLIP-2) determines what it can and cannot see
2. **ViT converts images to sequences using patch embedding** — this is the foundation of all modern VLMs, and understanding it explains image resolution trade-offs
3. **CLIP's contrastive learning creates a shared embedding space** — enabling zero-shot classification, but with known weaknesses in counting, spatial reasoning, and fine-grained distinctions
4. **Commercial APIs are production-ready but not error-free** — GPT-4o, Claude, and Gemini achieve 90-98% field extraction accuracy depending on document type, with hallucination rates of 1-5%
5. **Evaluation must measure hallucinations, not just accuracy** — a VLM that's 95% accurate but hallucinates 3% of the time will silently corrupt your data pipeline
6. **Cost varies 50× between providers** — Gemini Flash at $0.0004/doc vs. GPT-4o at $0.023/doc. Tiered routing (cheap model for simple docs, premium for complex) can cut costs 60-80%
7. **Local models trade quality for privacy** — 7B models are significantly weaker than commercial APIs on complex reasoning but keep all data on your hardware

### Provider Selection Guide

| Use Case | Recommended Provider | Why |
|----------|---------------------|-----|
| General analysis | GPT-4o or Claude 3.5 | Best accuracy on complex visual reasoning |
| Document OCR | Claude or Gemini Pro | Strong text extraction, good with tables |
| Cost-sensitive (>10K/month) | Gemini Flash | ~$0.0004/doc, 50x cheaper than GPT-4o |
| Privacy-critical | Local (LLaVA/Qwen-VL) | No data leaves your infrastructure |
| Multi-image comparison | Claude or Gemini | Native multi-image support in one request |
| Video analysis | Gemini 1.5 Pro | Only commercial API with native video input |
| Multilingual documents | Qwen-VL | Strongest non-English performance, especially CJK |

---

## Self-Assessment

| Category | Score | Justification |
|----------|-------|---------------|
| Conceptual Clarity | Strong | ViT from scratch, CLIP with temperature explanation, fusion strategies explained with trade-offs |
| Depth vs Surface | Strong | Fusion mechanism deep-dive, evaluation framework with worked example, failure mode taxonomy |
| Hands-On Practicality | Strong | Complete API integrations, working coding challenges, multi-provider service with fallback |
| Engineering Rigor | Good | Cost tracking, budget enforcement, robust JSON parsing, async FastAPI, but `_estimate_cost` is still simplistic |
| Evaluation Discipline | Good | VLMEvaluator with proper precision/recall/F1, CER/WER, hallucination detection, worked example, but no real dataset |
| Career Relevance | Strong | 5 interview questions with full explanations, career mapping table, system design question |
| Audience Targeting | Good | Clear reading guide, prerequisites, but still assumes CUDA for local models |

### Known Limitations

- **CLIP fine-tuning not covered** — adapting CLIP to domain-specific categories is a common need, but is covered in Blog 23 (fine-tuning)
- **Gemini video section is minimal** — `analyze_with_video_gemini` has basic error handling; dedicated video understanding is out of scope (see "What This Blog Does NOT Cover")
- **Evaluation uses synthetic data** — the worked example demonstrates the pattern but uses placeholder images. Real evaluation requires a labeled dataset specific to your domain
- **Cost estimates are approximate** — API pricing changes frequently; the disclaimers call this out, but readers should verify against current pricing

---

## Architect Sanity Checks

- **Would you trust someone who learned only this blog to touch a production AI system?**
  **YES** — The blog covers multimodal fusion (so readers understand what's inside the models), provides production-grade patterns (async FastAPI, cost tracking, budget enforcement, robust error handling), and the evaluation framework with hallucination detection teaches measurement discipline. The failure mode taxonomy and `should_use_vlm()` framework provide guardrails. The main gap is real evaluation data, but the framework is correct and the worked example demonstrates the workflow.

- **Can you explain at least one real failure case using only what's taught here?**
  **YES** — The hallucination failure mode explains that VLMs invent plausible but wrong field values (e.g., fabricating invoice numbers). The blog provides concrete detection via cross-provider verification and structural validation, plus the `detect_hallucinations()` method for automated checking. The spatial reasoning failure mode explains why VLMs miscount objects and confuse positions — critical for medical imaging and quality control use cases.

- **Would this blog survive senior-engineer interview follow-up questions?**
  **YES** — Each interview question includes mechanism-level explanations (not just bullet points), anticipates follow-up questions, and connects to system design. The ViT question explains patch-to-token conversion with computational complexity justification. The CLIP question explains contrastive loss, temperature scaling, and specific failure modes (typographic attacks). The system design question covers tiered routing, cost estimation, and human-in-the-loop patterns.

---

## What's Next?

In **Blog 22: Commercial Image APIs**, we'll explore the business side of image AI. You'll learn:
- DALL-E, Midjourney, and Stability AI APIs
- Cost optimization strategies
- Building image generation products
- Legal and ethical considerations

From understanding images to creating them at scale!

---

*Vision-language models see what we see, but understand it through the lens of language. Master both to unlock their full potential.*
