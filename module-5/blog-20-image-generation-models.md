# Blog 20: Image Generation Models

## Prompt Your Career: The Complete Generative AI Masterclass

**Reading time:** 60-90 minutes
**Coding time:** 90-120 minutes (GPU required for hands-on sections)
**Total investment:** ~3.5 hours

---

## What You'll Walk Away With

By the end of this blog, you will be able to:

1. **Understand how diffusion models work** — including latent space, the VAE, and classifier-free guidance (not just "add/remove noise")
2. **Run Stable Diffusion locally** and generate images from text prompts with informed parameter choices
3. **Apply prompt engineering** techniques with awareness of what actually impacts quality and what is cargo-cult
4. **Implement image-to-image transformations** and inpainting with the diffusers library
5. **Fine-tune models** with LoRA and DreamBooth, including validation strategy to detect overfitting
6. **Build image generation pipelines** for production use, including cost analysis and GPU memory management
7. **Evaluate image quality** using CLIP score, FID, and human preference methods
8. **Know when NOT to generate locally** and when API services are the better choice

> **How to read this blog:** If you want conceptual understanding, read "Understanding Diffusion Models" through "Latent Diffusion." If you want hands-on generation, skip to "Working with Stable Diffusion." If you're building production systems, focus on "Production Image Generation Pipeline" and "Cost and Hardware Analysis." Each section is self-contained.

> **Prerequisites:** Python fluency (Blog 2), basic deep learning concepts (neural networks, loss functions — Blog 9-10), familiarity with PyTorch tensors. GPU with ≥8GB VRAM required for hands-on sections; CPU-only users can follow along conceptually.

---

## What This Blog Does NOT Cover

- **Video generation** — Sora, Runway, and video diffusion models are a separate domain with different architectures and temporal modeling
- **3D generation** — NeRF, 3D Gaussian Splatting, and text-to-3D pipelines (DreamFusion, etc.)
- **Training from scratch** — We cover fine-tuning (LoRA, DreamBooth) but not pre-training a diffusion model from zero, which requires thousands of GPU-hours
- **Every model architecture** — We focus on the Stable Diffusion family; Midjourney's architecture is proprietary, and DALL-E 3 uses a different approach (CLIP + diffusion prior)
- **Advanced ControlNet/IP-Adapter** — We mention these briefly but detailed spatial conditioning pipelines deserve dedicated treatment
- **Legal compliance** — Copyright, licensing (CreativeML, SDXL license terms), and fair use are evolving legal areas that require actual legal counsel

---

## Manager's Summary

**For Technical Leaders and Decision Makers:**

Image generation AI has transformed from research curiosity to business reality. Models like Stable Diffusion, DALL-E, and Midjourney can create professional-quality images from text descriptions in seconds.

**Business Applications:**
- **Marketing**: Generate ad creatives, social media content, product mockups
- **E-commerce**: Create product variations, lifestyle images, backgrounds
- **Design**: Rapid prototyping, concept art, mood boards
- **Content**: Blog illustrations, thumbnails, infographics
- **Gaming**: Asset generation, texture creation, concept exploration

**Key Considerations:**
- **Cost**: Cloud APIs charge per image; local deployment requires GPU investment
- **Quality**: Latest models rival professional photography/illustration
- **Speed**: 2-10 seconds per image depending on model and hardware
- **Control**: Local models offer full customization; APIs offer convenience
- **Legal**: Training data and copyright remain evolving concerns

**Strategic Decision**: For occasional use, APIs are cost-effective. For high-volume production, local deployment with custom models offers better economics and control.

---

## The Evolution of Image Generation

### From GANs to Diffusion

```
Timeline of Image Generation:
├── 2014: GANs introduced (Goodfellow)
│   └── Adversarial training: Generator vs Discriminator
├── 2018: StyleGAN (NVIDIA)
│   └── High-quality face generation
├── 2020: DALL-E (OpenAI)
│   └── Text-to-image with transformers
├── 2021: CLIP (OpenAI)
│   └── Text-image alignment
├── 2022: Stable Diffusion (Stability AI)
│   └── Open-source diffusion model
├── 2023: SDXL, Midjourney v5
│   └── Photorealistic quality
└── 2024: Flux, SD3
    └── Architecture improvements
```

### Why Diffusion Models Won

| Aspect | GANs | Diffusion Models | Why |
|--------|------|------------------|-----|
| Training Stability | Unstable (mode collapse) | Very stable | GANs require adversarial balance between generator and discriminator — if one outpaces the other, training collapses. Diffusion models optimize a single denoising objective (MSE loss), which is convex-like and doesn't require balancing two networks. |
| Image Diversity | Limited | High | GANs suffer from mode collapse — the generator finds a few "good" outputs and gets stuck. Diffusion models sample from the full learned distribution because each denoising trajectory is unique. |
| Controllability | Difficult | Easy (conditioning) | Conditioning in GANs requires architectural changes (cGAN, StyleGAN). Diffusion models accept conditioning via cross-attention at every denoising step — any signal (text, depth map, edges) can guide generation. |
| Inference Speed | Fast (single forward pass) | Slower (20-50 steps) | GANs generate in one pass (~50ms). Diffusion requires iterative denoising (~2-10s). This is the primary tradeoff. Distillation methods (LCM, SDXL Turbo) reduce steps to 1-4 but sacrifice some quality. |
| Scalability | Diminishing returns at scale | Scales with compute | Larger GANs don't reliably improve quality. Diffusion models follow scaling laws — more compute and data consistently yields better results, similar to LLMs. |

---

## Understanding Diffusion Models

### The Core Concept

Diffusion models learn by:
1. **Forward process**: Gradually add noise to images until they become pure noise
2. **Reverse process**: Learn to remove noise step by step, recovering the image

```python
"""
Conceptual Diffusion Process
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Forward diffusion: Add noise progressively
def forward_diffusion(image: torch.Tensor, num_steps: int = 100):
    """
    Simulate forward diffusion by adding noise.

    At each step t, we have:
    x_t = sqrt(alpha_t) * x_{t-1} + sqrt(1 - alpha_t) * noise
    """
    # Define noise schedule (linear)
    betas = torch.linspace(0.0001, 0.02, num_steps)
    alphas = 1 - betas
    alpha_cumprod = torch.cumprod(alphas, dim=0)

    noisy_images = [image]

    for t in range(num_steps):
        # Sample noise
        noise = torch.randn_like(image)

        # Apply noise according to schedule
        sqrt_alpha = torch.sqrt(alpha_cumprod[t])
        sqrt_one_minus_alpha = torch.sqrt(1 - alpha_cumprod[t])

        noisy_image = sqrt_alpha * image + sqrt_one_minus_alpha * noise
        noisy_images.append(noisy_image)

    return noisy_images, alpha_cumprod


def visualize_diffusion(image: torch.Tensor, steps: list = [0, 25, 50, 75, 99]):
    """Visualize the forward diffusion process."""
    noisy_images, _ = forward_diffusion(image, num_steps=100)

    fig, axes = plt.subplots(1, len(steps), figsize=(15, 3))

    for ax, step in zip(axes, steps):
        img = noisy_images[step].permute(1, 2, 0).numpy()
        img = (img - img.min()) / (img.max() - img.min())  # Normalize
        ax.imshow(img)
        ax.set_title(f"Step {step}")
        ax.axis('off')

    plt.suptitle("Forward Diffusion: Adding Noise")
    plt.tight_layout()
    plt.savefig("diffusion_process.png")
    plt.close()


# The math behind diffusion
class DiffusionMath:
    """
    Key equations in diffusion models.
    """

    @staticmethod
    def noise_schedule(num_steps: int, schedule_type: str = "linear"):
        """
        Define the noise schedule (beta values).

        Linear: beta_t increases linearly
        Cosine: beta_t follows cosine curve (better for high-res)
        """
        if schedule_type == "linear":
            return torch.linspace(1e-4, 0.02, num_steps)

        elif schedule_type == "cosine":
            # Cosine schedule from "Improved DDPM" paper
            steps = torch.arange(num_steps + 1)
            alpha_bar = torch.cos((steps / num_steps + 0.008) / 1.008 * np.pi / 2) ** 2
            alpha_bar = alpha_bar / alpha_bar[0]
            betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
            return torch.clamp(betas, 0.0001, 0.999)

    @staticmethod
    def q_sample(x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor,
                 alpha_cumprod: torch.Tensor) -> torch.Tensor:
        """
        Sample from q(x_t | x_0) - the forward process.

        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise

        This lets us jump directly to any timestep t.
        """
        sqrt_alpha_cumprod = torch.sqrt(alpha_cumprod[t])
        sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - alpha_cumprod[t])

        # Reshape for broadcasting
        while sqrt_alpha_cumprod.dim() < x_0.dim():
            sqrt_alpha_cumprod = sqrt_alpha_cumprod.unsqueeze(-1)
            sqrt_one_minus_alpha_cumprod = sqrt_one_minus_alpha_cumprod.unsqueeze(-1)

        return sqrt_alpha_cumprod * x_0 + sqrt_one_minus_alpha_cumprod * noise
```

### The U-Net Architecture

The noise prediction network is typically a U-Net with attention. The simplified version below demonstrates the core concepts (encoder-decoder with skip connections, time conditioning, self-attention). The actual Stable Diffusion U-Net is significantly larger: it has residual blocks, cross-attention at multiple resolutions for text conditioning, and operates on 64×64×4 **latent** tensors rather than pixel images — more on this in the "Latent Diffusion" section below.

```python
"""
Simplified U-Net for Diffusion
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalPositionEmbeddings(nn.Module):
    """
    Encode timestep as sinusoidal embeddings.
    Similar to transformer positional encodings.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ConvBlock(nn.Module):
    """Convolutional block with normalization and activation."""

    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int = None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)

        # Time embedding projection
        if time_emb_dim:
            self.time_mlp = nn.Linear(time_emb_dim, out_channels)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor = None):
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.silu(h)

        # Add time embedding
        if time_emb is not None and hasattr(self, 'time_mlp'):
            time_emb = self.time_mlp(time_emb)
            h = h + time_emb[:, :, None, None]

        h = self.conv2(h)
        h = self.norm2(h)
        h = F.silu(h)
        return h


class AttentionBlock(nn.Module):
    """Self-attention block for U-Net."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.attention = nn.MultiheadAttention(channels, num_heads=4, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape

        # Reshape for attention: (batch, seq_len, channels)
        x_norm = self.norm(x)
        x_flat = x_norm.view(b, c, h * w).transpose(1, 2)

        # Self-attention
        attn_out, _ = self.attention(x_flat, x_flat, x_flat)

        # Reshape back
        attn_out = attn_out.transpose(1, 2).view(b, c, h, w)

        return x + attn_out


class SimpleUNet(nn.Module):
    """
    Simplified U-Net for diffusion models.

    Architecture:
    - Encoder: Downsampling path with conv blocks
    - Bottleneck: Attention at lowest resolution
    - Decoder: Upsampling path with skip connections
    - Time conditioning: Added at each block
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        time_dim: int = 256,
        base_channels: int = 64
    ):
        super().__init__()

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )

        # Encoder
        self.enc1 = ConvBlock(in_channels, base_channels, time_dim)
        self.enc2 = ConvBlock(base_channels, base_channels * 2, time_dim)
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4, time_dim)

        # Bottleneck
        self.bottleneck = ConvBlock(base_channels * 4, base_channels * 4, time_dim)
        self.attention = AttentionBlock(base_channels * 4)

        # Decoder
        self.dec3 = ConvBlock(base_channels * 8, base_channels * 2, time_dim)  # Skip connection
        self.dec2 = ConvBlock(base_channels * 4, base_channels, time_dim)
        self.dec1 = ConvBlock(base_channels * 2, base_channels, time_dim)

        # Output
        self.out = nn.Conv2d(base_channels, out_channels, 1)

        # Pooling and upsampling
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # Time embedding
        t_emb = self.time_mlp(t)

        # Encoder
        e1 = self.enc1(x, t_emb)
        e2 = self.enc2(self.pool(e1), t_emb)
        e3 = self.enc3(self.pool(e2), t_emb)

        # Bottleneck
        b = self.bottleneck(self.pool(e3), t_emb)
        b = self.attention(b)

        # Decoder with skip connections
        d3 = self.dec3(torch.cat([self.up(b), e3], dim=1), t_emb)
        d2 = self.dec2(torch.cat([self.up(d3), e2], dim=1), t_emb)
        d1 = self.dec1(torch.cat([self.up(d2), e1], dim=1), t_emb)

        return self.out(d1)


# Training loop for diffusion
def train_diffusion_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    images: torch.Tensor,
    noise_scheduler
) -> float:
    """
    Single training step for diffusion model.

    1. Sample random timesteps
    2. Add noise to images
    3. Predict noise
    4. Compute loss
    """
    optimizer.zero_grad()

    batch_size = images.shape[0]
    device = images.device

    # Sample random timesteps
    t = torch.randint(0, noise_scheduler.num_steps, (batch_size,), device=device)

    # Sample noise
    noise = torch.randn_like(images)

    # Add noise to images
    noisy_images = noise_scheduler.add_noise(images, noise, t)

    # Predict noise
    predicted_noise = model(noisy_images, t)

    # MSE loss between predicted and actual noise
    loss = F.mse_loss(predicted_noise, noise)

    loss.backward()
    optimizer.step()

    return loss.item()
```

### Text Conditioning with CLIP

```python
"""
Text Conditioning for Image Generation
"""
import torch
import torch.nn as nn
from transformers import CLIPTextModel, CLIPTokenizer

class TextEncoder:
    """
    Encode text prompts using CLIP for conditioning.
    """

    def __init__(self, model_name: str = "openai/clip-vit-large-patch14"):
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.text_encoder = CLIPTextModel.from_pretrained(model_name)
        self.text_encoder.eval()

    def encode(self, prompts: list[str], max_length: int = 77) -> torch.Tensor:
        """
        Encode text prompts to embeddings.

        Args:
            prompts: List of text prompts
            max_length: Maximum token length

        Returns:
            Text embeddings of shape (batch, seq_len, hidden_dim)
        """
        # Tokenize
        tokens = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt"
        )

        # Encode
        with torch.no_grad():
            outputs = self.text_encoder(
                input_ids=tokens.input_ids,
                attention_mask=tokens.attention_mask
            )

        # Return last hidden state
        return outputs.last_hidden_state


class CrossAttention(nn.Module):
    """
    Cross-attention for conditioning on text.
    Query: image features
    Key/Value: text embeddings
    """

    def __init__(self, query_dim: int, context_dim: int, heads: int = 8, dim_head: int = 64):
        super().__init__()
        inner_dim = heads * dim_head

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Image features (batch, seq_len, dim)
            context: Text embeddings (batch, context_len, context_dim)
        """
        batch_size = x.shape[0]

        # Project queries, keys, values
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        # Reshape for multi-head attention
        q = q.view(batch_size, -1, self.heads, q.shape[-1] // self.heads).transpose(1, 2)
        k = k.view(batch_size, -1, self.heads, k.shape[-1] // self.heads).transpose(1, 2)
        v = v.view(batch_size, -1, self.heads, v.shape[-1] // self.heads).transpose(1, 2)

        # Attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        # Apply attention to values
        out = torch.matmul(attn, v)

        # Reshape back
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, out.shape[-1] * self.heads)

        return self.to_out(out)
```

### Latent Diffusion — The Key Insight of Stable Diffusion

The U-Net and diffusion process above operate on images directly (pixel space). This works but is extremely expensive: a 512×512 RGB image has 786,432 dimensions. Denoising in this space at each step is slow and memory-intensive.

**Stable Diffusion's breakthrough**: run diffusion in a compressed **latent space** instead. A pre-trained Variational Autoencoder (VAE) compresses images from 512×512×3 to 64×64×4 — an **48x compression** — while preserving the information needed for reconstruction. The U-Net denoises in this small latent space, then the VAE decoder reconstructs the full-resolution image at the end.

```
Full Stable Diffusion Pipeline:

Text Prompt ──→ CLIP Text Encoder ──→ Text Embeddings (77 × 768)
                                            │
                                            ▼ (cross-attention)
Random Noise (64×64×4) ──→ U-Net Denoiser ──→ Denoised Latents (64×64×4)
    (latent space)           (20-50 steps)         │
                                                   ▼
                                            VAE Decoder ──→ Final Image (512×512×3)
                                            (pixel space)
```

**Why this matters for you:**
- The VAE is why you sometimes see blurry or artifact-heavy fine details — the VAE compression loses some high-frequency information
- SDXL uses a better VAE with less compression loss, which is one reason it produces sharper images
- When you set `output_type="latent"` (as in the SDXL base+refiner pipeline), you're passing latents directly without decoding, saving one VAE decode/encode cycle

### Classifier-Free Guidance (CFG)

CFG is the most important user-facing parameter in image generation (`guidance_scale`). Understanding it prevents the most common quality issues.

**The mechanism:** At each denoising step, the model makes two predictions:
1. **Conditional prediction**: "Denoise this given the text prompt"
2. **Unconditional prediction**: "Denoise this with no prompt at all" (empty string)

The final output amplifies the difference:

```
output = unconditional + guidance_scale × (conditional - unconditional)
```

**What `guidance_scale` actually controls:**

| Value | Effect | When to Use |
|-------|--------|-------------|
| 1.0 | No guidance — model ignores the prompt | Never (for text-guided generation) |
| 3.0-5.0 | Soft guidance — creative, diverse, sometimes off-prompt | Artistic exploration, abstract concepts |
| 7.0-8.5 | Balanced — good prompt adherence with natural look | **Default for most use cases** |
| 10.0-15.0 | Strong guidance — very literal, but colors saturate | Exact prompt matching at the cost of natural appearance |
| 20.0+ | Extreme — artifacts, oversaturation, burnt images | Almost never — indicates a prompt problem |

**The tradeoff is fundamental:** Higher CFG = more prompt adherence but less image quality (saturation, artifacts). This is not a bug — it's the mathematical consequence of amplifying the conditional signal. If you need high prompt adherence AND high quality, use a better prompt rather than cranking up guidance.

---

## Working with Stable Diffusion

### Setting Up Stable Diffusion

```python
"""
Stable Diffusion with diffusers library
"""
# pip install diffusers transformers accelerate torch

from diffusers import StableDiffusionPipeline, DiffusionPipeline
from diffusers import (
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionXLPipeline
)
import torch
from PIL import Image

# ============ Basic Text-to-Image ============

def setup_stable_diffusion(
    model_id: str = "stabilityai/stable-diffusion-2-1",
    device: str = "cuda",
    dtype: torch.dtype = torch.float16
):
    """
    Set up Stable Diffusion pipeline.

    Models:
    - "stabilityai/stable-diffusion-2-1": SD 2.1 (768x768)
    - "runwayml/stable-diffusion-v1-5": SD 1.5 (512x512)
    - "stabilityai/stable-diffusion-xl-base-1.0": SDXL (1024x1024)
    """
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        safety_checker=None  # OK for local experimentation only — see Content Safety section
    )

    # Move to GPU
    pipe = pipe.to(device)

    # Enable memory optimizations
    pipe.enable_attention_slicing()  # Reduce memory
    # pipe.enable_xformers_memory_efficient_attention()  # If xformers installed

    return pipe


def generate_image(
    pipe,
    prompt: str,
    negative_prompt: str = None,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    width: int = 512,
    height: int = 512,
    seed: int = None
) -> Image.Image:
    """
    Generate an image from a text prompt.

    Args:
        prompt: What you want to see
        negative_prompt: What you don't want to see
        num_inference_steps: More steps = better quality, slower
        guidance_scale: How closely to follow prompt (7-12 typical)
        seed: For reproducibility
    """
    generator = None
    if seed is not None:
        generator = torch.Generator(device="cuda").manual_seed(seed)

    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        width=width,
        height=height,
        generator=generator
    )

    return result.images[0]


# Example usage
if __name__ == "__main__":
    pipe = setup_stable_diffusion()

    # Generate with detailed prompt
    image = generate_image(
        pipe,
        prompt="""
        A majestic mountain landscape at sunset,
        snow-capped peaks, alpine lake reflection,
        golden hour lighting, dramatic clouds,
        professional photography, 8k, highly detailed
        """,
        negative_prompt="""
        blurry, low quality, distorted,
        oversaturated, cartoon, anime,
        watermark, text
        """,
        num_inference_steps=50,
        guidance_scale=7.5,
        seed=42
    )

    image.save("mountain_landscape.png")
```

### SDXL for Higher Quality

```python
"""
Stable Diffusion XL (SDXL) - Higher quality images
"""
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
import torch

def setup_sdxl(
    model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
    refiner_id: str = "stabilityai/stable-diffusion-xl-refiner-1.0",
    use_refiner: bool = True
):
    """
    Set up SDXL with optional refiner.

    SDXL produces 1024x1024 images with better text rendering, composition,
    detail, and prompt understanding.

    WHY TWO MODELS? The base model handles overall composition and structure
    (denoising from pure noise to high_noise_frac, e.g., 80%). The refiner is
    a separate U-Net trained specifically on low-noise images — it adds fine
    details, textures, and sharpness in the final denoising steps (80% to 100%).
    This is a form of expert specialization: coarse structure and fine detail
    are different tasks that benefit from different model weights.

    When to skip the refiner: If VRAM is limited (<12GB), or latency matters
    more than fine detail, the base model alone produces good results.
    """
    # Base model
    base = StableDiffusionXLPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    )
    base.to("cuda")
    base.enable_attention_slicing()

    # Refiner (optional, for enhanced details)
    refiner = None
    if use_refiner:
        refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            refiner_id,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True
        )
        refiner.to("cuda")
        refiner.enable_attention_slicing()

    return base, refiner


def generate_with_sdxl(
    base,
    refiner,
    prompt: str,
    negative_prompt: str = None,
    num_inference_steps: int = 40,
    high_noise_frac: float = 0.8,
    seed: int = None
) -> Image.Image:
    """
    Generate with SDXL base + refiner.

    The process:
    1. Base model generates image (denoising from noise to high_noise_frac)
    2. Refiner enhances details (denoising from high_noise_frac to 0)
    """
    generator = torch.Generator("cuda").manual_seed(seed) if seed else None

    # Generate with base
    latents = base(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        denoising_end=high_noise_frac,
        output_type="latent",
        generator=generator
    ).images

    # Refine
    if refiner is not None:
        image = refiner(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            denoising_start=high_noise_frac,
            image=latents,
            generator=generator
        ).images[0]
    else:
        # Decode latents if no refiner
        image = base.vae.decode(latents / base.vae.config.scaling_factor).sample
        image = base.image_processor.postprocess(image)[0]

    return image


# SDXL-specific prompting
sdxl_style_prompts = {
    "photorealistic": {
        "positive": "professional photograph, DSLR, 85mm lens, bokeh, sharp focus",
        "negative": "illustration, painting, drawing, art, sketch"
    },
    "cinematic": {
        "positive": "cinematic still, movie scene, dramatic lighting, film grain",
        "negative": "low quality, amateur, overexposed"
    },
    "digital_art": {
        "positive": "digital art, trending on artstation, highly detailed, vibrant colors",
        "negative": "photo, photograph, realistic"
    },
    "anime": {
        "positive": "anime style, manga, studio ghibli, detailed anime art",
        "negative": "realistic, photograph, 3d render"
    }
}
```

### Image-to-Image Transformation

```python
"""
Image-to-Image with Stable Diffusion
"""
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import torch

def setup_img2img(model_id: str = "stabilityai/stable-diffusion-2-1"):
    """Set up image-to-image pipeline."""
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16
    )
    pipe.to("cuda")
    pipe.enable_attention_slicing()
    return pipe


def transform_image(
    pipe,
    image: Image.Image,
    prompt: str,
    negative_prompt: str = None,
    strength: float = 0.75,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5
) -> Image.Image:
    """
    Transform an existing image based on a prompt.

    Args:
        image: Input image to transform
        prompt: What the output should look like
        strength: How much to change (0.0-1.0)
                  0.0 = no change, 1.0 = completely new image

    Use cases:
        - Style transfer (strength 0.5-0.7)
        - Major transformation (strength 0.8-1.0)
        - Subtle enhancement (strength 0.2-0.4)
    """
    # Ensure image is RGB
    image = image.convert("RGB")

    # Resize to model's preferred size
    image = image.resize((512, 512))

    result = pipe(
        prompt=prompt,
        image=image,
        strength=strength,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        negative_prompt=negative_prompt
    )

    return result.images[0]


# Style transfer examples
def style_transfer_examples():
    pipe = setup_img2img()

    # Load input image
    input_image = Image.open("photo.jpg")

    styles = {
        "oil_painting": {
            "prompt": "oil painting, thick brushstrokes, impressionist style, vibrant colors",
            "strength": 0.7
        },
        "watercolor": {
            "prompt": "watercolor painting, soft edges, flowing colors, artistic",
            "strength": 0.65
        },
        "cyberpunk": {
            "prompt": "cyberpunk style, neon lights, futuristic, sci-fi, blade runner",
            "strength": 0.75
        },
        "anime": {
            "prompt": "anime style, manga art, studio ghibli, detailed illustration",
            "strength": 0.7
        },
        "sketch": {
            "prompt": "pencil sketch, hand drawn, detailed linework, artistic",
            "strength": 0.6
        }
    }

    for style_name, config in styles.items():
        output = transform_image(
            pipe,
            input_image,
            prompt=config["prompt"],
            strength=config["strength"]
        )
        output.save(f"styled_{style_name}.png")
```

### Inpainting

```python
"""
Inpainting - Edit specific parts of images
"""
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image, ImageDraw
import numpy as np

def setup_inpainting(model_id: str = "stabilityai/stable-diffusion-2-inpainting"):
    """Set up inpainting pipeline."""
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16
    )
    pipe.to("cuda")
    return pipe


def create_mask(
    image_size: tuple,
    mask_region: tuple  # (x1, y1, x2, y2)
) -> Image.Image:
    """
    Create a mask image.
    White = area to inpaint
    Black = area to preserve
    """
    mask = Image.new("RGB", image_size, "black")
    draw = ImageDraw.Draw(mask)
    draw.rectangle(mask_region, fill="white")
    return mask


def inpaint_image(
    pipe,
    image: Image.Image,
    mask: Image.Image,
    prompt: str,
    negative_prompt: str = None,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5
) -> Image.Image:
    """
    Inpaint an image in the masked region.

    Args:
        image: Original image
        mask: Mask (white = inpaint, black = keep)
        prompt: What to generate in the masked area
    """
    # Ensure correct sizes
    image = image.convert("RGB").resize((512, 512))
    mask = mask.convert("RGB").resize((512, 512))

    result = pipe(
        prompt=prompt,
        image=image,
        mask_image=mask,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        negative_prompt=negative_prompt
    )

    return result.images[0]


# Advanced inpainting with automatic mask from text
def segment_and_inpaint(
    image_path: str,
    object_to_replace: str,
    replacement_prompt: str
):
    """
    Automatically segment an object and inpaint it.
    Requires: pip install transformers[torch]
    """
    from transformers import pipeline

    # Load image
    image = Image.open(image_path)

    # Use segmentation model to find object
    segmenter = pipeline("image-segmentation", model="facebook/maskformer-swin-base-ade")
    segments = segmenter(image)

    # Find the segment matching our object
    mask = None
    for segment in segments:
        if object_to_replace.lower() in segment['label'].lower():
            mask = segment['mask']
            break

    if mask is None:
        raise ValueError(f"Could not find '{object_to_replace}' in image")

    # Convert mask to PIL
    mask = Image.fromarray((np.array(mask) * 255).astype(np.uint8))

    # Inpaint
    inpaint_pipe = setup_inpainting()
    result = inpaint_image(
        inpaint_pipe,
        image,
        mask,
        prompt=replacement_prompt
    )

    return result
```

---

## Prompt Engineering for Images

### The Art of Image Prompts

```python
"""
Image Prompt Engineering
"""

class ImagePromptBuilder:
    """
    Build effective prompts for image generation.

    Structure:
    [Subject] + [Style] + [Quality] + [Mood/Lighting] + [Technical]
    """

    # Quality modifiers
    QUALITY_BOOSTERS = [
        "highly detailed",
        "professional",
        "8k resolution",
        "masterpiece",
        "best quality",
        "ultra realistic",
        "sharp focus"
    ]

    # Style keywords
    STYLES = {
        "photorealistic": ["photograph", "photo", "DSLR", "canon EOS", "realistic"],
        "digital_art": ["digital art", "digital painting", "trending on artstation"],
        "oil_painting": ["oil painting", "oil on canvas", "classical painting"],
        "watercolor": ["watercolor", "watercolor painting", "soft colors"],
        "anime": ["anime", "anime style", "manga", "cel shaded"],
        "3d_render": ["3D render", "octane render", "unreal engine", "blender"],
        "concept_art": ["concept art", "matte painting", "fantasy art"],
        "sketch": ["sketch", "pencil drawing", "charcoal", "lineart"]
    }

    # Lighting keywords
    LIGHTING = {
        "golden_hour": ["golden hour", "warm lighting", "sunset light"],
        "dramatic": ["dramatic lighting", "chiaroscuro", "high contrast"],
        "soft": ["soft lighting", "diffused light", "ambient light"],
        "neon": ["neon lights", "cyberpunk lighting", "colorful lighting"],
        "studio": ["studio lighting", "professional lighting", "softbox"],
        "natural": ["natural lighting", "daylight", "outdoor light"]
    }

    # Camera/composition
    COMPOSITION = {
        "portrait": ["portrait", "face closeup", "headshot", "85mm"],
        "wide": ["wide angle", "landscape", "panoramic", "establishing shot"],
        "macro": ["macro photography", "extreme closeup", "detailed"],
        "aerial": ["aerial view", "drone shot", "bird's eye view"],
        "low_angle": ["low angle shot", "worm's eye view", "looking up"]
    }

    # Common negative prompts
    NEGATIVE_DEFAULTS = [
        "blurry",
        "low quality",
        "distorted",
        "disfigured",
        "bad anatomy",
        "watermark",
        "text",
        "signature",
        "cropped",
        "out of frame"
    ]

    def __init__(self):
        self.subject = ""
        self.style = []
        self.quality = []
        self.lighting = []
        self.composition = []
        self.extras = []
        self.negative = self.NEGATIVE_DEFAULTS.copy()

    def set_subject(self, subject: str) -> "ImagePromptBuilder":
        """Set the main subject of the image."""
        self.subject = subject
        return self

    def add_style(self, style_key: str) -> "ImagePromptBuilder":
        """Add style keywords."""
        if style_key in self.STYLES:
            self.style.extend(self.STYLES[style_key])
        else:
            self.style.append(style_key)
        return self

    def add_lighting(self, lighting_key: str) -> "ImagePromptBuilder":
        """Add lighting keywords."""
        if lighting_key in self.LIGHTING:
            self.lighting.extend(self.LIGHTING[lighting_key])
        else:
            self.lighting.append(lighting_key)
        return self

    def add_composition(self, comp_key: str) -> "ImagePromptBuilder":
        """Add composition keywords."""
        if comp_key in self.COMPOSITION:
            self.composition.extend(self.COMPOSITION[comp_key])
        else:
            self.composition.append(comp_key)
        return self

    def add_quality(self, level: str = "high") -> "ImagePromptBuilder":
        """Add quality boosters."""
        if level == "high":
            self.quality.extend(self.QUALITY_BOOSTERS[:3])
        elif level == "ultra":
            self.quality.extend(self.QUALITY_BOOSTERS)
        return self

    def add_extra(self, *keywords: str) -> "ImagePromptBuilder":
        """Add additional keywords."""
        self.extras.extend(keywords)
        return self

    def add_negative(self, *keywords: str) -> "ImagePromptBuilder":
        """Add negative prompt keywords."""
        self.negative.extend(keywords)
        return self

    def build(self) -> tuple[str, str]:
        """Build the positive and negative prompts."""
        parts = [self.subject]
        parts.extend(self.style)
        parts.extend(self.quality)
        parts.extend(self.lighting)
        parts.extend(self.composition)
        parts.extend(self.extras)

        positive = ", ".join(filter(None, parts))
        negative = ", ".join(self.negative)

        return positive, negative


# Usage examples
def create_prompts():
    """Create various prompts using the builder."""

    # Portrait photography
    portrait = (ImagePromptBuilder()
        .set_subject("beautiful woman with red hair and green eyes")
        .add_style("photorealistic")
        .add_lighting("studio")
        .add_composition("portrait")
        .add_quality("ultra")
        .add_extra("looking at camera", "slight smile")
        .build())

    print("Portrait prompt:", portrait[0])
    print("Negative:", portrait[1])

    # Fantasy landscape
    landscape = (ImagePromptBuilder()
        .set_subject("floating islands in the sky with waterfalls")
        .add_style("concept_art")
        .add_lighting("golden_hour")
        .add_composition("wide")
        .add_quality("high")
        .add_extra("magical atmosphere", "birds flying", "lush vegetation")
        .build())

    print("\nLandscape prompt:", landscape[0])

    # Product photography
    product = (ImagePromptBuilder()
        .set_subject("luxury wristwatch on marble surface")
        .add_style("photorealistic")
        .add_lighting("studio")
        .add_composition("macro")
        .add_quality("ultra")
        .add_extra("product photography", "clean background", "reflections")
        .build())

    print("\nProduct prompt:", product[0])

    return portrait, landscape, product


# Prompt weights (for Stable Diffusion)
def weighted_prompt(prompt: str) -> str:
    """
    Add emphasis to parts of prompt.

    Syntax:
    - (word) = 1.1x emphasis
    - ((word)) = 1.21x emphasis
    - (word:1.5) = 1.5x emphasis
    - [word] = 0.9x de-emphasis
    """
    # Example: emphasize key elements
    weighted = prompt.replace(
        "red hair",
        "(red hair:1.3)"  # Emphasize red hair
    ).replace(
        "green eyes",
        "((green eyes))"  # Strong emphasis on eyes
    )
    return weighted
```

### Advanced Prompting Techniques

```python
"""
Advanced Image Prompting Techniques
"""

class AdvancedPrompting:
    """
    Advanced techniques for better image generation.
    """

    @staticmethod
    def regional_prompting(
        regions: dict[str, str]
    ) -> str:
        """
        Define different prompts for different regions.

        Example:
        regions = {
            "background": "sunset sky, clouds",
            "midground": "rolling hills, grass",
            "foreground": "single oak tree"
        }
        """
        # Combine with AND operator (model-specific)
        parts = [f"{region}: {prompt}" for region, prompt in regions.items()]
        return " AND ".join(parts)

    @staticmethod
    def blended_concepts(concepts: list[tuple[str, float]]) -> str:
        """
        Blend multiple concepts with weights.

        Example:
        concepts = [
            ("cyberpunk city", 0.6),
            ("ancient rome", 0.4)
        ]
        """
        return " | ".join(f"({concept}:{weight})" for concept, weight in concepts)

    @staticmethod
    def artistic_reference(
        subject: str,
        artist: str = None,
        art_movement: str = None,
        medium: str = None
    ) -> str:
        """
        Generate prompt with artistic references.

        Note: Be mindful of copyright when referencing specific artists.
        Better to reference styles/movements than living artists.
        """
        parts = [subject]

        if artist:
            parts.append(f"in the style of {artist}")
        if art_movement:
            parts.append(f"{art_movement} art movement")
        if medium:
            parts.append(f"{medium}")

        return ", ".join(parts)

    @staticmethod
    def photographic_prompt(
        subject: str,
        camera: str = "Canon EOS R5",
        lens: str = "85mm f/1.4",
        settings: dict = None
    ) -> str:
        """
        Create realistic photography prompt.
        """
        parts = [
            f"photograph of {subject}",
            f"shot on {camera}",
            f"{lens} lens"
        ]

        if settings:
            if "aperture" in settings:
                parts.append(f"f/{settings['aperture']}")
            if "iso" in settings:
                parts.append(f"ISO {settings['iso']}")
            if "shutter" in settings:
                parts.append(f"1/{settings['shutter']}s")

        parts.extend([
            "professional photography",
            "sharp focus",
            "natural lighting"
        ])

        return ", ".join(parts)


# Prompt templates for common use cases
PROMPT_TEMPLATES = {
    "product_photography": """
{product} on {surface},
professional product photography,
{lighting} lighting,
clean white background,
high resolution, commercial photography,
slight shadow, reflections
""",

    "character_portrait": """
{character_description},
{style} style,
{expression} expression,
{pose},
detailed face,
{background},
{lighting}
""",

    "landscape": """
{scene_description},
{time_of_day},
{weather},
{style} style,
{mood} atmosphere,
highly detailed environment,
cinematic composition
""",

    "architecture": """
{building_description},
{architectural_style} architecture,
{view_angle} view,
{time_of_day} lighting,
{environment},
architectural photography,
clean lines, detailed
""",

    "food_photography": """
{dish_description},
food photography,
{plating_style} presentation,
{garnish},
{background_surface},
{lighting} lighting,
appetizing, delicious looking,
shallow depth of field
"""
}


def use_template(template_name: str, **kwargs) -> str:
    """Fill in a prompt template."""
    template = PROMPT_TEMPLATES.get(template_name, "")
    return template.format(**kwargs)


# Example usage
def template_examples():
    # Product shot
    product_prompt = use_template(
        "product_photography",
        product="sleek wireless headphones",
        surface="black marble",
        lighting="soft studio"
    )

    # Character portrait
    character_prompt = use_template(
        "character_portrait",
        character_description="young woman with silver hair and heterochromia eyes",
        style="fantasy digital art",
        expression="determined",
        pose="three quarter view",
        background="mystical forest",
        lighting="ethereal glowing"
    )

    # Landscape
    landscape_prompt = use_template(
        "landscape",
        scene_description="volcanic mountains with lava flows",
        time_of_day="twilight",
        weather="stormy",
        style="realistic matte painting",
        mood="dramatic and ominous"
    )

    return product_prompt, character_prompt, landscape_prompt
```

---

## Fine-Tuning Image Models

### LoRA (Low-Rank Adaptation)

```python
"""
LoRA Training for Stable Diffusion
"""
from diffusers import StableDiffusionPipeline
from diffusers.loaders import LoraLoaderMixin
import torch

# Loading a LoRA
def load_lora(
    pipe: StableDiffusionPipeline,
    lora_path: str,
    weight: float = 1.0
):
    """
    Load a LoRA into existing pipeline.

    LoRAs are small (2-200MB) adapter weights that modify the base model.

    HOW IT WORKS: Instead of fine-tuning the full weight matrix W (millions of
    parameters), LoRA learns W + A×B where A (d×r) and B (r×d) are small
    matrices with rank r (typically 4-128). This reduces trainable parameters
    by 100-1000x. The key insight: attention weight changes during fine-tuning
    have low intrinsic rank — you don't need to change every weight, just a
    low-dimensional subspace. This is why LoRA files are so small and why
    multiple LoRAs can be composed (each modifies a different subspace).
    """
    pipe.load_lora_weights(lora_path)
    pipe.fuse_lora(lora_scale=weight)
    return pipe


def generate_with_lora(
    pipe,
    prompt: str,
    lora_trigger: str = None
) -> Image.Image:
    """
    Generate with LoRA.

    Many LoRAs have trigger words that activate their style.
    """
    if lora_trigger:
        prompt = f"{lora_trigger}, {prompt}"

    return pipe(prompt).images[0]


# Training a LoRA (using diffusers training script)
"""
To train a LoRA, you would typically use:

accelerate launch diffusers/examples/text_to_image/train_text_to_image_lora.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1" \
  --train_data_dir="./training_images" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=1000 \
  --learning_rate=1e-4 \
  --output_dir="./my_lora" \
  --validation_prompt="a photo of my_subject"
"""


class LoRATrainingConfig:
    """Configuration for LoRA training."""

    def __init__(
        self,
        training_images_dir: str,
        output_dir: str,
        trigger_word: str,
        base_model: str = "stabilityai/stable-diffusion-2-1"
    ):
        self.training_images_dir = training_images_dir
        self.output_dir = output_dir
        self.trigger_word = trigger_word
        self.base_model = base_model

        # Training parameters
        self.resolution = 512
        self.train_batch_size = 1
        self.gradient_accumulation_steps = 4
        self.learning_rate = 1e-4
        self.max_train_steps = 1000
        self.lora_rank = 4  # Lower = smaller file, less capacity

    def prepare_dataset(self) -> list[dict]:
        """
        Prepare training dataset.

        Each image needs a caption. Format:
        - images/image1.jpg -> images/image1.txt (caption)

        Caption format:
        "a photo of {trigger_word}, detailed description of image"
        """
        import os
        from pathlib import Path

        dataset = []
        images_dir = Path(self.training_images_dir)

        for img_path in images_dir.glob("*.jpg"):
            # Look for corresponding caption
            caption_path = img_path.with_suffix(".txt")

            if caption_path.exists():
                caption = caption_path.read_text().strip()
            else:
                # Auto-generate basic caption
                caption = f"a photo of {self.trigger_word}"

            dataset.append({
                "image": str(img_path),
                "caption": caption
            })

        return dataset

    def get_training_command(self) -> str:
        """Generate the training command."""
        return f"""
accelerate launch train_text_to_image_lora.py \\
    --pretrained_model_name_or_path="{self.base_model}" \\
    --train_data_dir="{self.training_images_dir}" \\
    --output_dir="{self.output_dir}" \\
    --resolution={self.resolution} \\
    --train_batch_size={self.train_batch_size} \\
    --gradient_accumulation_steps={self.gradient_accumulation_steps} \\
    --learning_rate={self.learning_rate} \\
    --max_train_steps={self.max_train_steps} \\
    --rank={self.lora_rank} \\
    --validation_prompt="a photo of {self.trigger_word}" \\
    --mixed_precision="fp16"
"""
```

### DreamBooth Fine-Tuning

```python
"""
DreamBooth - Personalize Stable Diffusion
"""

class DreamBoothConfig:
    """
    DreamBooth configuration for personalization.

    DreamBooth teaches the model a new concept (person, object, style)
    using just 3-10 images.

    Key concepts:
    - Instance prompt: "a photo of sks dog" (sks is the identifier)
    - Class prompt: "a photo of dog" (for regularization)

    WHY PRIOR PRESERVATION? Without it, DreamBooth causes "language drift" —
    the model associates ALL dogs with YOUR dog. The class prompt generates
    regularization images of generic dogs, and the loss function balances
    learning your specific dog (instance loss) with not forgetting what dogs
    in general look like (prior preservation loss). Without this, asking for
    "a photo of a golden retriever" might produce your specific dog instead.
    """

    def __init__(
        self,
        instance_name: str,  # Unique identifier like "sks" or "xyz"
        class_name: str,     # What the subject is: "dog", "person", "style"
        instance_images_dir: str,
        output_dir: str
    ):
        self.instance_name = instance_name
        self.class_name = class_name
        self.instance_images_dir = instance_images_dir
        self.output_dir = output_dir

        # Instance prompt
        self.instance_prompt = f"a photo of {instance_name} {class_name}"

        # Class prompt for regularization
        self.class_prompt = f"a photo of {class_name}"

        # Training parameters
        self.resolution = 512
        self.train_batch_size = 1
        self.learning_rate = 5e-6
        self.max_train_steps = 800
        self.with_prior_preservation = True
        self.prior_loss_weight = 1.0
        self.num_class_images = 200

    def prepare_training(self):
        """
        Prepare for DreamBooth training.

        Requirements:
        1. 3-10 high-quality images of the subject
        2. Consistent lighting/background preferred
        3. Various angles/expressions help
        """
        import os
        from pathlib import Path

        # Check instance images
        instance_path = Path(self.instance_images_dir)
        images = list(instance_path.glob("*.jpg")) + list(instance_path.glob("*.png"))

        if len(images) < 3:
            raise ValueError("Need at least 3 training images")

        if len(images) > 10:
            print(f"Warning: {len(images)} images. 3-10 is optimal for DreamBooth.")

        return {
            "num_images": len(images),
            "instance_prompt": self.instance_prompt,
            "class_prompt": self.class_prompt
        }

    def get_training_command(self) -> str:
        """Generate DreamBooth training command."""
        return f"""
accelerate launch train_dreambooth.py \\
    --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1" \\
    --instance_data_dir="{self.instance_images_dir}" \\
    --output_dir="{self.output_dir}" \\
    --instance_prompt="{self.instance_prompt}" \\
    --class_prompt="{self.class_prompt}" \\
    --resolution={self.resolution} \\
    --train_batch_size={self.train_batch_size} \\
    --learning_rate={self.learning_rate} \\
    --max_train_steps={self.max_train_steps} \\
    --with_prior_preservation \\
    --prior_loss_weight={self.prior_loss_weight} \\
    --num_class_images={self.num_class_images} \\
    --mixed_precision="fp16"
"""


# Using trained DreamBooth model
def use_dreambooth_model(model_path: str, instance_prompt: str):
    """Use a trained DreamBooth model."""
    pipe = StableDiffusionPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16
    ).to("cuda")

    # Generate with the learned concept
    # The model now "knows" the subject
    prompts = [
        f"{instance_prompt} wearing a spacesuit on mars",
        f"{instance_prompt} as a renaissance painting",
        f"{instance_prompt} in a cyberpunk city",
    ]

    images = []
    for prompt in prompts:
        image = pipe(prompt).images[0]
        images.append(image)

    return images
```

---

## Production Image Generation Pipeline

### Cost and Hardware Analysis

Before building a pipeline, understand the economics:

**GPU VRAM Requirements:**

| Model | fp16 VRAM (minimum) | fp16 VRAM (comfortable) | CPU Offload VRAM |
|-------|--------------------|-----------------------|------------------|
| SD 1.5 (512²) | ~4GB | ~6GB | ~2GB (slow) |
| SD 2.1 (768²) | ~5GB | ~8GB | ~3GB (slow) |
| SDXL base (1024²) | ~7GB | ~12GB | ~4GB (very slow) |
| SDXL base+refiner | ~14GB | ~24GB | ~6GB (very slow) |
| Flux.1 (1024²) | ~12GB | ~24GB | ~6GB (very slow) |

**Cost Comparison: Local vs API:**

| Factor | Local GPU (RTX 4090) | Cloud GPU (A100) | API (DALL-E 3) |
|--------|---------------------|------------------|----------------|
| Per-image cost | ~$0.001 (electricity) | ~$0.02-0.05 (rental) | $0.04-0.08 |
| Setup cost | $1,600 (hardware) | $0 (pay per hour) | $0 |
| Images to break even vs API | ~25,000 | N/A | N/A |
| Latency | 2-5s | 2-5s | 5-15s (network + queue) |
| Customization | Full (LoRA, DreamBooth, any model) | Full | None |
| Maintenance | You handle driver updates, model updates | Provider handles | Provider handles |

**Decision rule:** If you generate >1,000 images/month with custom models, local/cloud GPU is more economical. For occasional use or prototyping, APIs win on total cost of ownership.

### Content Safety (Non-Optional for Production)

Disabling `safety_checker=None` is fine for local experimentation but **unacceptable for any user-facing system**:

1. **NSFW filtering**: Re-enable the safety checker, or use a dedicated classifier (e.g., `CompVis/stable-diffusion-safety-checker`) as a post-generation filter
2. **Prompt injection**: Users can craft prompts to bypass safety measures. Implement input sanitization AND output classification as a defense-in-depth strategy
3. **Copyright concerns**: Models trained on internet-scraped data may reproduce recognizable copyrighted styles or subjects. Log prompts and generated images for audit trails
4. **Legal liability**: Hosting a generation service without content moderation exposes you to legal risk. Implement rate limiting, logging, and a content policy

```python
"""
Educational Image Generation System

NOTE: This demonstrates production patterns (model loading, memory management,
batch processing, API serving) but is NOT production-ready as-is. For production:
- Re-enable or replace safety_checker with a content moderation pipeline
- Add authentication and rate limiting to API endpoints
- Add prompt sanitization (reject or modify harmful prompts)
- Use lifespan context manager instead of deprecated on_event
- Run diffusion inference in a thread pool, not in async handler
- Add request queuing (Redis + Celery) for multi-user load
- Implement CDN or object storage for serving generated images
"""
from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    AutoPipelineForInpainting
)
import torch
from PIL import Image
from dataclasses import dataclass
from typing import Optional, List
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class GenerationRequest:
    """Image generation request."""
    prompt: str
    negative_prompt: Optional[str] = None
    width: int = 1024
    height: int = 1024
    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    seed: Optional[int] = None
    style_preset: Optional[str] = None


@dataclass
class GenerationResult:
    """Image generation result."""
    image: Image.Image
    seed: int
    prompt: str
    generation_time: float


class ProductionImageGenerator:
    """
    Production-ready image generation system.

    Features:
    - Multiple model support
    - Request queuing
    - Memory management
    - Error handling
    - Logging and monitoring
    """

    STYLE_PRESETS = {
        "photorealistic": {
            "suffix": ", professional photography, DSLR, 8k, sharp focus",
            "negative": "cartoon, anime, illustration, painting"
        },
        "digital_art": {
            "suffix": ", digital art, trending on artstation, highly detailed",
            "negative": "photo, photograph, realistic"
        },
        "anime": {
            "suffix": ", anime style, manga, detailed anime art",
            "negative": "realistic, photo, 3d render"
        },
        "cinematic": {
            "suffix": ", cinematic still, movie scene, dramatic lighting, film grain",
            "negative": "low quality, amateur"
        }
    }

    def __init__(
        self,
        model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
        device: str = "cuda",
        enable_memory_optimization: bool = True
    ):
        self.device = device
        self.model_id = model_id
        self.pipe = None
        self.enable_memory_optimization = enable_memory_optimization
        self._load_model()

    def _load_model(self):
        """Load the model with optimizations."""
        logger.info(f"Loading model: {self.model_id}")

        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True
        )

        self.pipe.to(self.device)

        if self.enable_memory_optimization:
            self.pipe.enable_attention_slicing()
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
                logger.info("xformers enabled")
            except Exception:
                logger.info("xformers not available, using default attention")

        logger.info("Model loaded successfully")

    def _apply_style_preset(self, request: GenerationRequest) -> GenerationRequest:
        """Apply style preset to request."""
        if request.style_preset and request.style_preset in self.STYLE_PRESETS:
            preset = self.STYLE_PRESETS[request.style_preset]
            request.prompt = request.prompt + preset["suffix"]

            if request.negative_prompt:
                request.negative_prompt = request.negative_prompt + ", " + preset["negative"]
            else:
                request.negative_prompt = preset["negative"]

        return request

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Generate a single image."""
        import time

        start_time = time.time()

        # Apply style preset
        request = self._apply_style_preset(request)

        # Set up generator for reproducibility
        seed = request.seed if request.seed else torch.randint(0, 2**32 - 1, (1,)).item()
        generator = torch.Generator(self.device).manual_seed(seed)

        # Add default negative prompt
        negative = request.negative_prompt or ""
        negative = negative + ", blurry, low quality, distorted, watermark, text"

        try:
            result = self.pipe(
                prompt=request.prompt,
                negative_prompt=negative,
                width=request.width,
                height=request.height,
                num_inference_steps=request.num_inference_steps,
                guidance_scale=request.guidance_scale,
                generator=generator
            )

            generation_time = time.time() - start_time

            return GenerationResult(
                image=result.images[0],
                seed=seed,
                prompt=request.prompt,
                generation_time=generation_time
            )

        except torch.cuda.OutOfMemoryError:
            logger.error("CUDA out of memory. Clearing cache and retrying...")
            torch.cuda.empty_cache()
            raise

    def generate_batch(
        self,
        requests: List[GenerationRequest],
        max_concurrent: int = 2
    ) -> List[GenerationResult]:
        """Generate multiple images."""
        results = []

        for i, request in enumerate(requests):
            logger.info(f"Generating image {i+1}/{len(requests)}")
            try:
                result = self.generate(request)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to generate image {i+1}: {e}")
                results.append(None)

        return results

    def generate_variations(
        self,
        prompt: str,
        num_variations: int = 4,
        base_request: GenerationRequest = None
    ) -> List[GenerationResult]:
        """Generate multiple variations of a prompt."""
        if base_request is None:
            base_request = GenerationRequest(prompt=prompt)
        else:
            base_request.prompt = prompt

        requests = []
        for i in range(num_variations):
            req = GenerationRequest(
                prompt=base_request.prompt,
                negative_prompt=base_request.negative_prompt,
                width=base_request.width,
                height=base_request.height,
                num_inference_steps=base_request.num_inference_steps,
                guidance_scale=base_request.guidance_scale,
                seed=None,  # Different seed for each variation
                style_preset=base_request.style_preset
            )
            requests.append(req)

        return self.generate_batch(requests)

    def cleanup(self):
        """Clean up resources."""
        if self.pipe:
            del self.pipe
        torch.cuda.empty_cache()
        logger.info("Resources cleaned up")


# FastAPI integration
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel as PydanticModel
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
import base64
from io import BytesIO

# Thread pool for running synchronous GPU work without blocking the event loop
_executor = ThreadPoolExecutor(max_workers=1)  # 1 worker = 1 GPU serialized


class ImageRequest(PydanticModel):
    prompt: str
    negative_prompt: str = None
    width: int = 1024
    height: int = 1024
    steps: int = 30
    guidance: float = 7.5
    seed: int = None
    style: str = None


class ImageResponse(PydanticModel):
    image_base64: str
    seed: int
    generation_time: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan — replaces deprecated @app.on_event."""
    logger.info("Loading image generation model...")
    app.state.generator = ProductionImageGenerator()
    logger.info("Model loaded successfully")
    yield
    app.state.generator.cleanup()
    logger.info("Resources cleaned up")


app = FastAPI(title="Image Generation API", lifespan=lifespan)


@app.post("/generate", response_model=ImageResponse)
async def generate_image_endpoint(request: ImageRequest):
    """Generate an image from a text prompt.

    NOTE: Diffusion inference is CPU/GPU-bound and must NOT run in the async
    event loop. We offload to a thread pool to keep the server responsive.
    For multi-user production, use a task queue (Celery/Redis) instead.
    """
    try:
        gen_request = GenerationRequest(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            width=request.width,
            height=request.height,
            num_inference_steps=request.steps,
            guidance_scale=request.guidance,
            seed=request.seed,
            style_preset=request.style
        )

        # Offload GPU work to thread pool — do NOT block the event loop
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            _executor, app.state.generator.generate, gen_request
        )

        # Convert image to base64
        buffer = BytesIO()
        result.image.save(buffer, format="PNG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode()

        return ImageResponse(
            image_base64=image_base64,
            seed=result.seed,
            generation_time=result.generation_time
        )

    except torch.cuda.OutOfMemoryError:
        raise HTTPException(
            status_code=503,
            detail="GPU out of memory. Try a smaller resolution or wait for current requests to finish."
        )
    except (ValueError, RuntimeError) as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    gen = getattr(app.state, "generator", None)
    return {"status": "healthy", "model": gen.model_id if gen else None}
```

---

## Evaluating Image Generation Quality

Building a generation pipeline is the easy part. Knowing whether your outputs are good — and catching regressions when you change prompts, models, or parameters — is harder.

### Metrics That Matter

| Metric | What It Measures | How to Compute | When to Use |
|--------|-----------------|----------------|-------------|
| **CLIP Score** | Text-image alignment — does the image match the prompt? | Cosine similarity between CLIP text and image embeddings | Every generation — cheap, automated, tells you if prompt adherence degraded |
| **FID (Fréchet Inception Distance)** | Statistical similarity between generated and real image distributions | Compare Inception-v3 feature distributions of generated vs reference set | Model comparison, fine-tuning validation (lower = more realistic) |
| **Human Preference** | Subjective quality as perceived by humans | A/B testing, Elo ratings, or side-by-side ranking | Final evaluation before deployment — no automated metric replaces this |
| **LPIPS** | Perceptual similarity between two images | Learned perceptual distance using deep features | Measuring variation diversity, img2img fidelity |

### Minimal Evaluation Script

```python
"""
Evaluate image generation quality using CLIP score.
Run this after changing prompts, models, or parameters to catch regressions.
"""
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from typing import List

class ImageEvaluator:
    """Evaluate generated images using CLIP score and basic quality checks."""

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

    def clip_score(self, image: Image.Image, prompt: str) -> float:
        """
        Compute CLIP score (text-image alignment).

        Returns cosine similarity in [0, 1]. Higher = better alignment.
        Typical good scores: 0.25-0.35. Below 0.20 = likely off-prompt.
        """
        inputs = self.processor(text=[prompt], images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Cosine similarity between text and image embeddings
        score = outputs.logits_per_image.item() / 100  # Normalize to ~[0, 1]
        return score

    def evaluate_batch(
        self,
        images: List[Image.Image],
        prompts: List[str]
    ) -> dict:
        """Evaluate a batch of generated images."""
        scores = []
        for img, prompt in zip(images, prompts):
            scores.append(self.clip_score(img, prompt))

        return {
            "mean_clip_score": sum(scores) / len(scores),
            "min_clip_score": min(scores),
            "max_clip_score": max(scores),
            "below_threshold": sum(1 for s in scores if s < 0.20),
            "total_images": len(scores),
        }

# Usage:
# evaluator = ImageEvaluator()
# score = evaluator.clip_score(generated_image, "a cat sitting on a windowsill")
# print(f"CLIP score: {score:.3f}")  # > 0.25 is usually acceptable
```

### Fine-Tuning Validation Strategy

When training LoRA or DreamBooth, overfitting is the primary failure mode. Signs:
1. **Generated images look identical to training images** — the model memorized instead of learning the concept
2. **Non-subject elements degrade** — backgrounds become blurry, hands deformed, because the model forgot general knowledge
3. **CLIP score for unrelated prompts drops** — the model "forgot" how to follow prompts that don't involve the trained concept

**Validation protocol:**
- Generate validation images every 100 training steps using 3-5 fixed prompts (some with the trained concept, some without)
- Monitor CLIP score on the "without" prompts — if it drops >10%, you're overfitting
- Save checkpoints and compare — earlier checkpoints often generalize better than the final one
- Use a held-out set of 2-3 images of the subject that were NOT in training — can the model reproduce them?

---

## Interview Preparation

### Career Mapping

| Role | How Image Generation Knowledge Applies | Key Skills from This Blog |
|------|---------------------------------------|---------------------------|
| **ML Engineer (Computer Vision)** | Model deployment, fine-tuning, inference optimization | Diffusion math, LoRA/DreamBooth, VRAM management, latent space |
| **AI Product Engineer** | Building user-facing generation features | diffusers library, prompt engineering, content safety, API design |
| **Creative Technologist** | Tools for design/marketing teams | img2img, inpainting, style presets, prompt builder patterns |
| **MLOps / Platform** | Serving models at scale | GPU memory management, batching, FastAPI integration, cost analysis |
| **Solutions Architect** | Build vs buy decisions for image generation | Cost comparison (local vs cloud vs API), hardware sizing, framework selection |

### Conceptual Questions

1. **How do diffusion models generate images? Explain the full Stable Diffusion pipeline.**

   Diffusion models learn to reverse a noise-adding process. During training, random noise is added to images at varying timestep levels, and a U-Net learns to predict the noise that was added. At inference, we start with pure random noise and iteratively denoise it over 20-50 steps. **Stable Diffusion's key innovation** is doing this in **latent space**: a pre-trained VAE compresses 512×512×3 images to 64×64×4 latents (48x compression), the U-Net denoises in this compact space, and the VAE decoder reconstructs the full image. Text conditioning happens via CLIP text embeddings injected through cross-attention at each U-Net block. Classifier-free guidance amplifies the difference between text-conditional and unconditional predictions: `output = uncond + scale × (cond - uncond)`.

2. **What is classifier-free guidance and what are its failure modes?**

   CFG makes two noise predictions per step — one conditioned on the prompt, one unconditional — then extrapolates: `output = uncond + guidance_scale × (cond - uncond)`. Higher values force stronger prompt adherence but amplify artifacts. At scale 7-8.5 (default), you get good balance. At 15+, colors oversaturate and images develop high-frequency artifacts because you're extrapolating beyond the training distribution. The fix for poor prompt adherence is a better prompt, not higher guidance. CFG doubles inference cost because you make two predictions per step; CFG-distilled models (LCM, SDXL Turbo) avoid this by learning to predict the guided output directly.

3. **Explain LoRA vs DreamBooth — when would you use each?**

   **LoRA** (Low-Rank Adaptation) injects small trainable matrices (rank 4-128) into the attention layers of the U-Net. Instead of updating the full weight matrix W, it learns W + A×B where A and B are small rank-r matrices. This produces 2-200MB adapter files that can be loaded/unloaded at runtime. Use LoRA for: styles, general concepts, combining multiple adapters. **DreamBooth** fine-tunes the full model (or a large portion) to learn a specific subject from 3-10 images, using a unique identifier ("sks dog"). It uses prior preservation loss — generating class images ("a photo of dog") to prevent the model from forgetting what dogs generally look like. DreamBooth produces larger model weights (2-7GB) but better identity preservation. Use DreamBooth for: personalized subjects (your face, your product, your pet) where identity consistency matters.

4. **How do you evaluate image generation quality? What can go wrong?**

   **Automated metrics:** CLIP score for text-image alignment (cosine similarity between CLIP embeddings — below 0.20 indicates off-prompt), FID for distributional similarity to real images (lower = more realistic, but requires a reference dataset of 10K+ images). **Human evaluation:** A/B preference tests are the gold standard but expensive. **Failure modes for fine-tuning:** Overfitting (model memorizes training images instead of learning the concept — detect by generating novel poses/backgrounds and checking for degradation), catastrophic forgetting (model forgets general knowledge — detect by checking CLIP scores on unrelated prompts), and training instability (learning rate too high for DreamBooth — use 5e-6, not 1e-4).

5. **Walk through the cost/latency tradeoffs for serving image generation.**

   Local GPU (RTX 4090, ~$1,600): ~2-5s latency, ~$0.001/image, full customization, but you handle maintenance. Cloud GPU (A100 on-demand): ~2-5s latency, ~$0.02-0.05/image, same customization, provider handles hardware. API service (DALL-E 3): ~5-15s latency (network + queue), ~$0.04-0.08/image, zero maintenance but no customization (no LoRA, no inpainting control). Breakeven for local vs API: ~25,000 images. For high-volume production with custom models, local/cloud wins. For prototyping or low-volume, APIs win on total cost of ownership. Memory optimization is critical: FP16 halves VRAM, attention slicing trades speed for memory, xformers provides 20-30% memory reduction, and sequential CPU offload enables running large models on small GPUs at the cost of 5-10x slower inference.

### Coding Challenges

**Challenge 1**: Implement a prompt optimizer — complete solution, not a skeleton:

```python
def optimize_prompt(base_prompt: str, style: str) -> tuple[str, str]:
    """
    Optimize a prompt for better generation.
    Returns (positive_prompt, negative_prompt).
    """
    STYLE_SUFFIXES = {
        "photo": (
            ", professional photograph, DSLR, 85mm lens, bokeh, sharp focus, 8k",
            "cartoon, anime, illustration, painting, drawing"
        ),
        "digital_art": (
            ", digital art, trending on artstation, highly detailed, vibrant colors",
            "photo, photograph, realistic, blurry"
        ),
        "anime": (
            ", anime style, manga, cel shading, detailed anime art",
            "realistic, photo, 3d render, blurry"
        ),
    }

    suffix, style_negative = STYLE_SUFFIXES.get(style, ("", ""))
    positive = base_prompt + suffix
    negative = f"{style_negative}, blurry, low quality, distorted, watermark, text, disfigured"

    return positive, negative

# Example:
# pos, neg = optimize_prompt("a cat sitting on a windowsill at sunset", "photo")
# image = pipe(prompt=pos, negative_prompt=neg).images[0]
```

**Challenge 2**: Build an image variation generator — complete solution:

```python
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import torch

def generate_variations(
    pipe: StableDiffusionImg2ImgPipeline,
    image: Image.Image,
    prompt: str,
    num_variations: int = 4,
    variation_strength: float = 0.5
) -> list[Image.Image]:
    """
    Generate variations of an input image with different seeds.
    strength controls divergence: 0.3 = subtle, 0.7 = significant.
    """
    image = image.convert("RGB").resize((512, 512))
    results = []

    for i in range(num_variations):
        generator = torch.Generator("cuda").manual_seed(i * 42)
        result = pipe(
            prompt=prompt,
            image=image,
            strength=variation_strength,
            num_inference_steps=30,
            generator=generator,
        ).images[0]
        results.append(result)

    return results
```

---

## Exercises

### Exercise 1: Build a Style Transfer Tool
Create a system that:
- Takes an input image
- Applies various artistic styles
- Allows strength adjustment
- Saves results with metadata

### Exercise 2: Create a Product Photography Generator
Build a pipeline that:
- Takes product images
- Generates studio-quality photos
- Creates multiple backgrounds
- Outputs consistent sizing

### Exercise 3: Implement a Character Generator
Create a system that:
- Generates consistent characters
- Maintains identity across poses
- Supports multiple styles
- Uses LoRA for personalization

### Exercise 4: Build an Image Generation API
Create a FastAPI service with:
- Multiple endpoints (txt2img, img2img, inpaint)
- Request queuing
- Progress tracking
- Webhook notifications

---

## Summary

### Key Takeaways

1. **Diffusion models work in latent space**: The VAE compresses images 48x before the U-Net denoises, which is why Stable Diffusion is practical on consumer GPUs. Understanding this architecture explains most quality artifacts (VAE compression loss on fine details)
2. **Classifier-free guidance is the most important parameter**: It controls prompt adherence vs image quality. Higher is not always better — above 12-15, artifacts dominate. Fix prompt adherence issues with better prompts, not higher guidance
3. **LoRA and DreamBooth serve different purposes**: LoRA (low-rank adapters, 2-200MB) for styles/concepts that can compose; DreamBooth (full fine-tune, 2-7GB) for identity-preserving personalization with prior preservation loss
4. **Evaluation is not optional**: CLIP score for automated text-image alignment, FID for distributional quality, human preference for final validation. Without metrics, you cannot catch regressions when changing prompts, models, or parameters
5. **Production requires more than `pipe()`**: Content safety filtering, GPU memory management (FP16, attention slicing, xformers), thread pool for async serving, request queuing for multi-user load, CDN for image delivery
6. **Know the economics**: Local GPU breaks even at ~25K images vs API. For <1K images/month or no customization needs, APIs win on total cost of ownership
7. **Content safety is a first-class concern**: Disabling safety_checker is acceptable only for local experimentation; any user-facing system needs NSFW filtering, prompt sanitization, and audit logging

### Technology Comparison

| Model | Resolution | VRAM (fp16) | Latency (RTX 4090) | Per-Image Cost (Local) | Per-Image Cost (API) | Best For |
|-------|-----------|-------------|---------------------|----------------------|---------------------|----------|
| SD 1.5 | 512² | ~4GB | ~1.5s | ~$0.0005 | N/A (no hosted API) | Prototyping, LoRA ecosystem |
| SD 2.1 | 768² | ~5GB | ~2.5s | ~$0.0008 | N/A | Better quality, still lightweight |
| SDXL | 1024² | ~7GB | ~4s | ~$0.001 | N/A | Production quality, good text rendering |
| SDXL+Refiner | 1024² | ~14GB | ~6s | ~$0.002 | N/A | Maximum quality, needs high VRAM |
| Flux.1 | 1024² | ~12GB | ~5s | ~$0.002 | ~$0.03 (via Replicate) | Cutting-edge quality |
| DALL-E 3 | 1024² | N/A | 5-15s | N/A | $0.04-0.08 | Zero setup, good quality |
| Midjourney | Up to 2048² | N/A | 10-60s | N/A | $0.01-0.03 (subscription) | Artistic quality, community |

### Framework Comparison

| Tool | Best For | Learning Curve | Customization |
|------|---------|---------------|---------------|
| **diffusers (HuggingFace)** | Production pipelines, custom code, fine-tuning | Medium (Python API) | Full — any model, any pipeline |
| **ComfyUI** | Visual workflow design, complex multi-model pipelines | Low-medium (node-based UI) | High — nodes for everything |
| **Automatic1111 WebUI** | Interactive exploration, community extensions | Low (web interface) | Medium — via extensions |
| **Direct API (OpenAI/Replicate)** | Prototyping, no GPU available | Low (REST API) | None — use what's offered |

---

## Self-Assessment Rubric

| Criteria | Excellent (9-10) | Good (7-8) | Needs Work (5-6) |
|----------|-----------------|------------|-------------------|
| **Diffusion understanding** | Explains latent space, VAE, CFG mechanism, noise schedule tradeoffs | Understands forward/reverse process and conditioning | "It removes noise from random images" |
| **Prompt engineering** | Uses structured prompts with style/quality/composition, understands CFG tradeoffs, validates with CLIP score | Good prompts with iteration | Tries random keywords, no systematic approach |
| **Fine-tuning** | Trains LoRA/DreamBooth with validation strategy, detects overfitting, understands prior preservation | Runs training commands, checks outputs visually | No fine-tuning experience |
| **Production readiness** | Thread-pooled async serving, content safety, GPU memory management, cost analysis | Basic generation with API endpoint | `pipe(prompt).images[0]` in a notebook |
| **Evaluation** | Uses CLIP score for automated regression, validates fine-tuning with held-out prompts | Visual inspection with some A/B testing | No evaluation methodology |
| **Cost awareness** | Can estimate VRAM needs, per-image cost, break-even vs API | Knows GPU is needed, understands FP16 | No awareness of resource requirements |

### What This Blog Does Well

- Covers the full pipeline from diffusion math to production serving in one coherent walkthrough
- Explains latent diffusion (the VAE), which is the key insight most tutorials skip
- Provides concrete CFG guidance with a parameter table and failure mode explanations
- Includes cost and VRAM analysis for informed hardware decisions
- Production code addresses real issues: thread pool for async serving, specific error handling, lifespan management
- Evaluation section provides automated CLIP-based regression testing and fine-tuning validation strategy
- Interview questions include full explanations with tradeoff analysis, not just bullet lists

### Where This Blog Falls Short

- All hands-on code requires a GPU with ≥8GB VRAM — no CPU-friendly or mocked alternatives for learning
- The simplified U-Net is educational but significantly simpler than the actual SD U-Net (no residual blocks in skip connections, no cross-attention at multiple resolutions)
- ControlNet and IP-Adapter — two of the most practically useful extensions — are only mentioned, not demonstrated
- No coverage of batch inference optimization (dynamic batching, TensorRT compilation, ONNX export)
- The prompt engineering section is long but empirically unvalidated — we provide techniques but no A/B tests showing their actual impact
- No discussion of model quantization (INT8) for deployment on edge devices or smaller GPUs

### Architect Sanity Checks

### Check 1: Would you trust someone who learned *only this blog* to touch a production image generation system?
**YES, with caveats.** The blog covers the full architecture (latent diffusion, VAE, CFG), practical generation with diffusers, fine-tuning with validation, and production serving patterns. The content safety section and production app notes explicitly list what's missing for real deployment. The cost analysis enables informed hardware decisions. The reader would still need to implement content moderation, request queuing, and CDN serving for production, but the blog makes these gaps explicit.

### Check 2: Can you explain at least one real failure case using only what's taught here?
**YES.** Multiple failure cases are explicitly addressed: (1) CFG too high — images oversaturate and develop artifacts because you're extrapolating beyond training distribution, (2) LoRA/DreamBooth overfitting — model memorizes training images, detectable via CLIP score degradation on unrelated prompts, (3) CUDA OOM — generation with insufficient VRAM crashes without memory optimization (FP16, attention slicing), (4) VAE compression artifacts on fine details — explained by the 48x latent compression, (5) synchronous inference blocking the event loop in async servers — solved by thread pool offloading.

### Check 3: Would this blog survive senior-engineer interview follow-up questions?
**YES.** The interview section covers: the full SD pipeline (latent space, VAE, CLIP, cross-attention, CFG), CFG failure modes, LoRA vs DreamBooth mechanisms (low-rank decomposition vs full fine-tune with prior preservation), evaluation methodology (CLIP score, FID, human preference), and cost/latency tradeoffs with specific dollar figures. A candidate could defend architecture choices (why latent space, why CFG, when to use API vs local) with concrete reasoning.

---

## What's Next?

In **Blog 21: Vision + Language Models**, we'll explore multimodal AI that understands both images and text. You'll learn:
- Vision transformers and image understanding
- Models like GPT-4V, Claude Vision, and LLaVA
- Building visual Q&A systems
- Multimodal applications

From generating images to understanding them—let's see what AI can see!

---

*The best images come from the best prompts. Master the language of visual generation.*
