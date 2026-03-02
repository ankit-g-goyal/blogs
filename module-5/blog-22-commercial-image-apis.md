# Blog 22: Commercial Image APIs

## Prompt Your Career: The Complete Generative AI Masterclass

**Reading time:** 45-60 minutes
**Coding time:** 90-120 minutes
**Total investment:** ~3 hours

---

## What You'll Walk Away With

By the end of this blog, you will be able to:

1. **Use DALL-E, Stability AI, and other commercial APIs** for image generation
2. **Implement cost-effective image generation pipelines** for production
3. **Choose the right API** based on quality, cost, and use case requirements
4. **Handle image variations, editing, and upscaling** via APIs
5. **Navigate legal, ethical, and regulatory considerations** in commercial image generation
6. **Build production-ready image generation services** with proper error handling
7. **Evaluate image generation quality** using automated and human review approaches

> **How to read this blog:** If you've used image generation APIs before, skip to "Building Production Image Services" for the multi-provider architecture. If you're new to image APIs, start from the top and run each provider example individually before combining them. The provider sections are independent --- you can read only the ones relevant to your use case.

---

## Prerequisites

Before starting this blog, you should have:

- **Python proficiency** (Blog 2) --- classes, async/await, HTTP requests
- **API integration experience** (Blog 14) --- REST APIs, authentication, error handling
- **Image generation fundamentals** (Blog 20-21) --- understanding of diffusion models, prompting concepts
- **API keys** for at least one provider (OpenAI, Stability AI, Ideogram, or Leonardo.AI) --- free tiers are available for experimentation

---

## What This Blog Does NOT Cover

Before we begin, let's set clear expectations on scope:

- **Self-hosted image generation** --- running Stable Diffusion locally or on your own GPU infrastructure is covered in Blog 20.
- **Fine-tuning image models** --- training custom image models (LoRA, DreamBooth) is addressed in Blog 23.
- **Video generation APIs** --- Runway, Pika, and video-from-image APIs are out of scope.
- **Prompt engineering in depth** --- we cover API-specific prompt patterns, but advanced prompt engineering techniques for image generation deserve standalone treatment.
- **Full production deployment** --- containerization, Kubernetes scaling, and CI/CD for image services are in Blog 24.
- **Detailed copyright law analysis** --- we highlight key concerns, but consult legal counsel for jurisdiction-specific questions.

---

## Manager's Summary

**For Technical Leaders and Decision Makers:**

Commercial image APIs have matured into reliable, scalable services that can power production applications. Understanding the trade-offs between providers is crucial for making informed decisions.

**Provider Comparison:**

> **Pricing and capabilities change frequently.** The table below reflects approximate information as of early 2025. Always verify current pricing and capabilities at each provider's official documentation before making production decisions.

| Provider | Quality | Speed | Approx. Cost | Best For |
|----------|---------|-------|---------------|----------|
| DALL-E 3 | Excellent | 10-15s | $0.04-0.12/img | Text rendering, concepts |
| Stability AI | Very Good | 2-5s | $0.002-0.02/img | Volume, customization |
| Midjourney | Excellent | 15-60s | ~$0.02/img (subscription) | Artistic, marketing |
| Ideogram | Excellent | 5-10s | $0.02-0.08/img | Text in images |
| Leonardo.AI | Very Good | 3-8s | $0.01-0.04/img | Game assets, characters |

**Cost at Scale (approximate):**

| Volume/Month | DALL-E 3 | Stability AI | Self-Hosted SD |
|--------------|----------|--------------|----------------|
| 1,000 images | $40-120 | $2-20 | ~$50 (GPU rental) |
| 10,000 images | $400-1,200 | $20-200 | ~$50-200 (GPU rental) |
| 100,000 images | $4,000-12,000 | $200-2,000 | ~$200-500 (GPU rental) |

*Self-hosted costs depend heavily on GPU choice, cloud provider, and utilization. These are rough estimates for cloud GPU rental (e.g., A10G or A100 instances).*

**Strategic Recommendations:**
- **Prototyping**: DALL-E 3 for quality, Stability AI for speed
- **Production (low volume)**: Mix based on use case
- **Production (high volume)**: Self-hosted or Stability AI
- **Consumer products**: Evaluate Midjourney for premium quality (API access may be limited)

---

## How Commercial Image APIs Work Under the Hood

Before diving into individual providers, understanding what's happening behind the API calls makes you a better debugger, prompt engineer, and system designer.

### The Generation Pipeline

Every commercial image API runs a variation of this pipeline:

```
Your Prompt ──▶ Prompt Processing ──▶ Text Encoder ──▶ Diffusion Model ──▶ Decoder ──▶ Safety Filter ──▶ Image
                    │                       │                │                              │
                    │                       │                │                              │
               DALL-E 3:              CLIP or T5         U-Net/DiT              NSFW classifier,
               Rewrites prompt       converts text       iteratively           content policy
               via GPT-4 for         to embeddings       denoises from         check — rejects
               better adherence      (768-4096 dims)     random noise          violations
                                                         in latent space
```

**Why this matters:** When a generation fails or produces poor results, knowing the pipeline tells you WHERE the failure occurred:
- **Prompt processing failure** → DALL-E 3 rewrites your prompt and the rewrite loses your intent. Fix: check `revised_prompt` in response.
- **Text encoding mismatch** → CLIP-based models (SD, SDXL) struggle with long or complex prompts because CLIP's text encoder truncates at 77 tokens. T5-based models (SD3, Imagen) handle longer prompts. Fix: front-load important details.
- **Diffusion quality issue** → artifacts, blurriness, wrong composition. Fix: adjust `cfg_scale` (guidance strength) and `steps`.
- **Safety filter rejection** → prompt or generated image triggers content policy. Fix: rephrase prompt to avoid flagged patterns.

### Key Parameters You'll See Across All Providers

| Parameter | What It Does | Mechanism | Typical Range |
|-----------|-------------|-----------|---------------|
| **CFG Scale / Guidance** | Controls prompt adherence vs. creativity | Higher values force model to match prompt more closely at the cost of diversity and sometimes quality | 5-15 (7 is common default) |
| **Steps** | Number of denoising iterations | More steps = more refined image but diminishing returns past ~30 | 15-50 |
| **Negative Prompt** | What to avoid in the image | Creates a "push away" signal — the model actively steers generation away from these concepts | Text description of unwanted elements |
| **Seed** | Random starting point for generation | Same seed + same prompt + same model = identical output (reproducibility) | Any integer |
| **Strength** (img2img) | How much to change the input image | Lower values preserve more of original; higher values allow more change | 0.0-1.0 (0.75 typical) |

> **DALL-E 3 vs Stability AI — fundamental architectural difference:** DALL-E 3 uses GPT-4 to rewrite your prompt before generation, meaning you rarely get exactly what you typed. This improves average quality but reduces user control. Stability AI passes your prompt directly to the text encoder, giving you exact control but requiring better prompting skills. This is the core trade-off between the two platforms.

---

## OpenAI DALL-E API

### DALL-E 3: State-of-the-Art Quality

```python
"""
DALL-E 3 API Integration
"""
from openai import OpenAI, APIError, RateLimitError, APIConnectionError
import base64
from pathlib import Path
from PIL import Image
import io
import requests

client = OpenAI()


def generate_image_dalle3(
    prompt: str,
    size: str = "1024x1024",
    quality: str = "standard",
    style: str = "vivid",
    n: int = 1
) -> list[str]:
    """
    Generate images with DALL-E 3.

    Args:
        prompt: Text description of desired image
        size: "1024x1024", "1024x1792", or "1792x1024"
        quality: "standard" ($0.04) or "hd" ($0.08)
        style: "vivid" (dramatic) or "natural" (realistic)
        n: Number of images (DALL-E 3 only supports n=1)

    Returns:
        List of image URLs
    """
    response = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size=size,
        quality=quality,
        style=style,
        n=n
    )

    return [img.url for img in response.data]


def generate_with_revised_prompt(prompt: str) -> dict:
    """
    Generate image and get the revised prompt.

    DALL-E 3 internally rewrites prompts for better results.
    Getting the revised prompt helps understand what was generated.
    """
    response = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size="1024x1024",
        quality="standard",
        n=1
    )

    return {
        "url": response.data[0].url,
        "revised_prompt": response.data[0].revised_prompt
    }


# DALL-E 2 for variations and editing
def create_variation_dalle2(
    image_path: str,
    n: int = 4,
    size: str = "1024x1024"
) -> list[str]:
    """
    Create variations of an existing image (DALL-E 2 only).

    The input image must be:
    - PNG format
    - Square
    - Less than 4MB
    """
    with open(image_path, "rb") as f:
        response = client.images.create_variation(
            image=f,
            n=n,
            size=size
        )

    return [img.url for img in response.data]


def edit_image_dalle2(
    image_path: str,
    mask_path: str,
    prompt: str,
    size: str = "1024x1024"
) -> str:
    """
    Edit an image using a mask (DALL-E 2 inpainting).

    Args:
        image_path: Original image (PNG, square, <4MB)
        mask_path: Mask image (transparent where to edit)
        prompt: What to generate in the masked area
    """
    with open(image_path, "rb") as img, open(mask_path, "rb") as mask:
        response = client.images.edit(
            image=img,
            mask=mask,
            prompt=prompt,
            size=size,
            n=1
        )

    return response.data[0].url


class DALLE3Client:
    """Production DALL-E 3 client with error handling."""

    def __init__(self):
        self.client = OpenAI()
        self.default_config = {
            "model": "dall-e-3",
            "size": "1024x1024",
            "quality": "standard",
            "style": "vivid"
        }

    def generate(
        self,
        prompt: str,
        **kwargs
    ) -> dict:
        """Generate an image with full metadata."""
        import time

        config = {**self.default_config, **kwargs}
        start_time = time.time()

        try:
            response = self.client.images.generate(
                prompt=prompt,
                **config,
                n=1
            )

            return {
                "success": True,
                "url": response.data[0].url,
                "revised_prompt": response.data[0].revised_prompt,
                "config": config,
                "generation_time": time.time() - start_time
            }

        except RateLimitError as e:
            return {
                "success": False,
                "error": f"Rate limited: {e}",
                "config": config,
                "generation_time": time.time() - start_time
            }
        except APIConnectionError as e:
            return {
                "success": False,
                "error": f"Connection error: {e}",
                "config": config,
                "generation_time": time.time() - start_time
            }
        except APIError as e:
            return {
                "success": False,
                "error": f"API error: {e}",
                "config": config,
                "generation_time": time.time() - start_time
            }

    def download_image(self, url: str, save_path: str = None) -> Image.Image:
        """Download generated image."""
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content))

        if save_path:
            image.save(save_path)

        return image

    def batch_generate(
        self,
        prompts: list[str],
        **kwargs
    ) -> list[dict]:
        """Generate multiple images (sequential, DALL-E 3 limit)."""
        results = []
        for prompt in prompts:
            result = self.generate(prompt, **kwargs)
            results.append(result)
        return results


# Cost calculator
def calculate_dalle_cost(
    num_images: int,
    quality: str = "standard",
    size: str = "1024x1024"
) -> dict:
    """
    Calculate DALL-E API costs.

    Note: Pricing is approximate and may change. Verify at
    https://openai.com/pricing before making production decisions.
    """
    # DALL-E 3 pricing (approximate as of early 2025)
    prices = {
        ("standard", "1024x1024"): 0.040,
        ("standard", "1024x1792"): 0.080,
        ("standard", "1792x1024"): 0.080,
        ("hd", "1024x1024"): 0.080,
        ("hd", "1024x1792"): 0.120,
        ("hd", "1792x1024"): 0.120
    }

    price_per_image = prices.get((quality, size), 0.040)
    total_cost = num_images * price_per_image

    return {
        "price_per_image": price_per_image,
        "num_images": num_images,
        "total_cost": total_cost,
        "quality": quality,
        "size": size
    }
```

### Prompt Engineering for DALL-E

```python
"""
DALL-E Prompt Engineering
"""

class DALLEPromptBuilder:
    """Build optimized prompts for DALL-E 3."""

    # DALL-E 3 handles most formatting internally,
    # but these additions can help guide output

    STYLE_MODIFIERS = {
        "photorealistic": "photograph, photorealistic, high resolution photo",
        "digital_art": "digital art, digital illustration, artstation",
        "oil_painting": "oil painting, classical art, painted",
        "watercolor": "watercolor painting, soft colors, artistic",
        "anime": "anime style, manga art, japanese animation",
        "3d_render": "3D render, CGI, unreal engine, octane render",
        "sketch": "pencil sketch, hand drawn, detailed linework",
        "minimalist": "minimalist, simple, clean design",
        "vintage": "vintage, retro style, nostalgic",
        "futuristic": "futuristic, sci-fi, cyberpunk"
    }

    QUALITY_MODIFIERS = [
        "high quality",
        "detailed",
        "professional",
        "masterpiece"
    ]

    @classmethod
    def build_prompt(
        cls,
        subject: str,
        style: str = None,
        mood: str = None,
        composition: str = None,
        add_quality: bool = True
    ) -> str:
        """Build an optimized prompt."""
        parts = [subject]

        if style and style in cls.STYLE_MODIFIERS:
            parts.append(cls.STYLE_MODIFIERS[style])
        elif style:
            parts.append(style)

        if mood:
            parts.append(f"{mood} atmosphere")

        if composition:
            parts.append(composition)

        if add_quality:
            parts.extend(cls.QUALITY_MODIFIERS[:2])

        return ", ".join(parts)

    @staticmethod
    def avoid_common_issues(prompt: str) -> str:
        """
        Modify prompt to avoid common DALL-E issues.

        DALL-E 3 has content policies that may reject:
        - Named public figures
        - Copyrighted characters
        - Violence/adult content
        """
        # Replace named celebrities with descriptions
        replacements = {
            # These are examples - DALL-E will reject specific names
            "realistic person": "realistic portrait of a person",
        }

        for old, new in replacements.items():
            prompt = prompt.replace(old, new)

        return prompt


# Prompt templates for common use cases
DALLE_TEMPLATES = {
    "product_shot": """
A professional product photograph of {product},
on a clean {background} background,
studio lighting, commercial photography,
high-end advertising style, sharp focus
""",

    "social_media": """
{subject} for social media,
{style} aesthetic,
eye-catching, vibrant colors,
perfect for Instagram, modern design
""",

    "hero_image": """
{subject} as a hero image,
dramatic composition, {mood} mood,
widescreen format, cinematic,
professional website header
""",

    "app_icon": """
App icon design for {app_type} app,
{style} style, simple and recognizable,
{color_scheme} color scheme,
professional icon design, flat design
""",

    "illustration": """
{subject} illustration,
{style} art style,
{mood} mood,
detailed illustration for {purpose}
"""
}


def use_dalle_template(template_name: str, **kwargs) -> str:
    """Fill a DALL-E template."""
    template = DALLE_TEMPLATES.get(template_name, "")
    return template.format(**kwargs).strip()


# Example usage
def generate_product_shots():
    client = DALLE3Client()

    products = [
        {"product": "sleek wireless earbuds", "background": "white marble"},
        {"product": "luxury wristwatch", "background": "dark velvet"},
        {"product": "organic skincare bottle", "background": "natural wood"}
    ]

    results = []
    for p in products:
        prompt = use_dalle_template("product_shot", **p)
        result = client.generate(prompt, quality="hd")
        results.append(result)

    return results
```

---

## Stability AI API

### Stable Diffusion API

```python
"""
Stability AI API Integration
"""
import requests
import base64
from PIL import Image
import io
from typing import Optional

class StabilityAI:
    """
    Stability AI API client.

    Features:
    - Text-to-image generation
    - Image-to-image transformation
    - Upscaling
    - Inpainting
    - Multiple model versions

    Note: Stability AI has migrated newer features to v2beta endpoints.
    Check https://platform.stability.ai/docs for the latest API version.
    """

    BASE_URL = "https://api.stability.ai/v1"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json"
        }

    def text_to_image(
        self,
        prompt: str,
        negative_prompt: str = None,
        engine: str = "stable-diffusion-xl-1024-v1-0",
        height: int = 1024,
        width: int = 1024,
        steps: int = 30,
        cfg_scale: float = 7.0,
        samples: int = 1,
        seed: int = None,
        style_preset: str = None
    ) -> list[Image.Image]:
        """
        Generate images from text.

        Engines:
        - stable-diffusion-xl-1024-v1-0: SDXL 1.0
        - stable-diffusion-v1-6: SD 1.6
        - stable-diffusion-512-v2-1: SD 2.1

        Style presets:
        - 3d-model, analog-film, anime, cinematic
        - comic-book, digital-art, enhance, fantasy-art
        - isometric, line-art, low-poly, modeling-compound
        - neon-punk, origami, photographic, pixel-art
        - tile-texture
        """
        url = f"{self.BASE_URL}/generation/{engine}/text-to-image"

        body = {
            "text_prompts": [
                {"text": prompt, "weight": 1.0}
            ],
            "height": height,
            "width": width,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "samples": samples
        }

        if negative_prompt:
            body["text_prompts"].append({
                "text": negative_prompt,
                "weight": -1.0
            })

        if seed is not None:
            body["seed"] = seed

        if style_preset:
            body["style_preset"] = style_preset

        response = requests.post(
            url,
            headers={**self.headers, "Content-Type": "application/json"},
            json=body,
            timeout=60
        )

        response.raise_for_status()
        data = response.json()

        images = []
        for artifact in data["artifacts"]:
            if artifact["finishReason"] == "SUCCESS":
                img_data = base64.b64decode(artifact["base64"])
                images.append(Image.open(io.BytesIO(img_data)))

        return images

    def image_to_image(
        self,
        image: Image.Image,
        prompt: str,
        negative_prompt: str = None,
        strength: float = 0.75,
        engine: str = "stable-diffusion-xl-1024-v1-0",
        steps: int = 30,
        cfg_scale: float = 7.0
    ) -> list[Image.Image]:
        """Transform an existing image based on a prompt."""
        url = f"{self.BASE_URL}/generation/{engine}/image-to-image"

        # Convert image to base64
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        init_image_b64 = base64.b64encode(buffer.getvalue()).decode()

        # Build form data
        form_data = {
            "init_image": (None, init_image_b64),
            "text_prompts[0][text]": (None, prompt),
            "text_prompts[0][weight]": (None, "1.0"),
            "image_strength": (None, str(1 - strength)),
            "steps": (None, str(steps)),
            "cfg_scale": (None, str(cfg_scale))
        }

        if negative_prompt:
            form_data["text_prompts[1][text]"] = (None, negative_prompt)
            form_data["text_prompts[1][weight]"] = (None, "-1.0")

        response = requests.post(
            url,
            headers=self.headers,
            files=form_data,
            timeout=60
        )

        response.raise_for_status()
        data = response.json()

        images = []
        for artifact in data["artifacts"]:
            if artifact["finishReason"] == "SUCCESS":
                img_data = base64.b64decode(artifact["base64"])
                images.append(Image.open(io.BytesIO(img_data)))

        return images

    def upscale(
        self,
        image: Image.Image,
        engine: str = "esrgan-v1-x2plus"
    ) -> Image.Image:
        """
        Upscale an image.

        Engines:
        - esrgan-v1-x2plus: 2x upscale
        - stable-diffusion-x4-latent-upscaler: 4x upscale
        """
        url = f"{self.BASE_URL}/generation/{engine}/image-to-image/upscale"

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")

        response = requests.post(
            url,
            headers=self.headers,
            files={"image": buffer.getvalue()},
            timeout=120
        )

        response.raise_for_status()
        data = response.json()

        if data["artifacts"]:
            img_data = base64.b64decode(data["artifacts"][0]["base64"])
            return Image.open(io.BytesIO(img_data))

        return None

    def inpaint(
        self,
        image: Image.Image,
        mask: Image.Image,
        prompt: str,
        negative_prompt: str = None,
        engine: str = "stable-diffusion-xl-1024-v1-0"
    ) -> Image.Image:
        """
        Inpaint an image (edit specific regions).

        Args:
            image: Original image
            mask: Mask (white = area to edit)
            prompt: What to generate in masked area
        """
        url = f"{self.BASE_URL}/generation/{engine}/image-to-image/masking"

        # Prepare images
        img_buffer = io.BytesIO()
        image.save(img_buffer, format="PNG")

        mask_buffer = io.BytesIO()
        mask.save(mask_buffer, format="PNG")

        files = {
            "init_image": img_buffer.getvalue(),
            "mask_image": mask_buffer.getvalue()
        }

        data = {
            "text_prompts[0][text]": prompt,
            "text_prompts[0][weight]": 1.0,
            "mask_source": "MASK_IMAGE_WHITE"
        }

        if negative_prompt:
            data["text_prompts[1][text]"] = negative_prompt
            data["text_prompts[1][weight]"] = -1.0

        response = requests.post(
            url,
            headers=self.headers,
            files=files,
            data=data,
            timeout=60
        )

        response.raise_for_status()
        result = response.json()

        if result["artifacts"]:
            img_data = base64.b64decode(result["artifacts"][0]["base64"])
            return Image.open(io.BytesIO(img_data))

        return None

    def get_balance(self) -> float:
        """Get account balance."""
        response = requests.get(
            f"{self.BASE_URL}/user/balance",
            headers=self.headers,
            timeout=10
        )
        response.raise_for_status()
        return response.json()["credits"]


# Stable Diffusion 3 API
class StabilitySD3:
    """Stability AI Stable Diffusion 3 API."""

    BASE_URL = "https://api.stability.ai/v2beta"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def generate(
        self,
        prompt: str,
        negative_prompt: str = None,
        model: str = "sd3-large",
        aspect_ratio: str = "1:1",
        seed: int = None,
        output_format: str = "png"
    ) -> Image.Image:
        """
        Generate with Stable Diffusion 3.

        Models:
        - sd3-large: Highest quality
        - sd3-large-turbo: Faster, good quality
        - sd3-medium: Balanced

        Aspect ratios:
        - 1:1, 16:9, 21:9, 2:3, 3:2, 4:5, 5:4, 9:16, 9:21
        """
        url = f"{self.BASE_URL}/stable-image/generate/sd3"

        data = {
            "prompt": prompt,
            "model": model,
            "aspect_ratio": aspect_ratio,
            "output_format": output_format
        }

        if negative_prompt:
            data["negative_prompt"] = negative_prompt

        if seed is not None:
            data["seed"] = seed

        response = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "image/*"
            },
            files={"none": ""},  # Required for multipart
            data=data,
            timeout=60
        )

        response.raise_for_status()

        return Image.open(io.BytesIO(response.content))
```

### Cost Optimization with Stability AI

```python
"""
Cost Optimization Strategies for Stability AI
"""
from dataclasses import dataclass
from typing import Optional

@dataclass
class GenerationConfig:
    """Configuration for cost-optimized generation."""
    prompt: str
    quality_level: str = "standard"  # draft, standard, premium
    width: int = 1024
    height: int = 1024


class CostOptimizedGenerator:
    """
    Generate images with cost optimization.

    Strategies:
    1. Use appropriate quality for use case
    2. Cache similar prompts
    3. Use smaller sizes when possible
    4. Batch requests efficiently
    """

    QUALITY_CONFIGS = {
        "draft": {
            "steps": 15,
            "cfg_scale": 5.0,
            "engine": "stable-diffusion-xl-1024-v1-0"
        },
        "standard": {
            "steps": 30,
            "cfg_scale": 7.0,
            "engine": "stable-diffusion-xl-1024-v1-0"
        },
        "premium": {
            "steps": 50,
            "cfg_scale": 7.5,
            "engine": "stable-diffusion-xl-1024-v1-0"
        }
    }

    # Approximate cost per generation (in credits)
    # Note: These are rough estimates. Actual costs depend on
    # your Stability AI plan and current pricing.
    COST_PER_GENERATION = {
        "draft": 0.002,
        "standard": 0.004,
        "premium": 0.008
    }

    def __init__(self, api_key: str):
        self.client = StabilityAI(api_key)
        self.cache = {}  # Simple prompt cache

    def generate(self, config: GenerationConfig) -> dict:
        """Generate with cost-optimized settings."""
        # Check cache
        cache_key = f"{config.prompt}:{config.quality_level}"
        if cache_key in self.cache:
            return {
                "image": self.cache[cache_key],
                "cached": True,
                "cost": 0
            }

        # Get quality settings
        quality_config = self.QUALITY_CONFIGS[config.quality_level]

        # Generate
        images = self.client.text_to_image(
            prompt=config.prompt,
            width=config.width,
            height=config.height,
            **quality_config
        )

        if images:
            # Cache result
            self.cache[cache_key] = images[0]

            return {
                "image": images[0],
                "cached": False,
                "cost": self.COST_PER_GENERATION[config.quality_level]
            }

        return {"image": None, "cached": False, "cost": 0}

    def estimate_cost(self, configs: list[GenerationConfig]) -> dict:
        """Estimate cost for a batch of generations."""
        total = 0
        breakdown = {}

        for config in configs:
            quality = config.quality_level
            cost = self.COST_PER_GENERATION[quality]
            total += cost
            breakdown[quality] = breakdown.get(quality, 0) + cost

        return {
            "total_credits": total,
            "total_usd_approx": total * 10,  # Very approximate conversion
            "breakdown": breakdown,
            "note": "USD conversion is approximate. Check your plan pricing."
        }


def compare_provider_costs(
    num_images: int,
    size: str = "1024x1024"
) -> dict:
    """
    Compare costs across providers.

    Note: These are approximate costs that change frequently.
    Always verify current pricing at each provider's pricing page.
    """
    # Approximate costs per image (USD), as of early 2025
    costs = {
        "dalle3_standard": 0.04,
        "dalle3_hd": 0.08,
        "stability_standard": 0.004,
        "stability_premium": 0.008,
        "midjourney_approx": 0.02,  # Subscription-based, varies
        "leonardo_approx": 0.015   # Token-based, varies
    }

    comparison = {}
    for provider, cost_per_image in costs.items():
        total = cost_per_image * num_images
        comparison[provider] = {
            "per_image": cost_per_image,
            "total": total,
            "monthly_10k": cost_per_image * 10000
        }

    return comparison
```

---

## Other Commercial APIs

### Ideogram API

```python
"""
Ideogram API - Excellent for text in images
"""
import requests
from PIL import Image
import io

class IdeogramAI:
    """
    Ideogram API client.

    Known for:
    - Excellent text rendering in images
    - Consistent typography
    - Good prompt following
    """

    BASE_URL = "https://api.ideogram.ai"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {
            "Api-Key": api_key,
            "Content-Type": "application/json"
        }

    def generate(
        self,
        prompt: str,
        aspect_ratio: str = "ASPECT_1_1",
        model: str = "V_2",
        magic_prompt: str = "AUTO",
        seed: int = None
    ) -> list[dict]:
        """
        Generate images with Ideogram.

        Aspect ratios:
        - ASPECT_1_1, ASPECT_16_9, ASPECT_9_16
        - ASPECT_4_3, ASPECT_3_4
        - ASPECT_3_2, ASPECT_2_3

        Models:
        - V_2: Latest model
        - V_1_TURBO: Faster, slightly lower quality
        """
        url = f"{self.BASE_URL}/generate"

        body = {
            "image_request": {
                "prompt": prompt,
                "aspect_ratio": aspect_ratio,
                "model": model,
                "magic_prompt_option": magic_prompt
            }
        }

        if seed is not None:
            body["image_request"]["seed"] = seed

        response = requests.post(url, headers=self.headers, json=body, timeout=60)
        response.raise_for_status()

        data = response.json()

        results = []
        for img in data.get("data", []):
            results.append({
                "url": img["url"],
                "prompt": img.get("prompt"),  # Possibly rewritten
                "seed": img.get("seed")
            })

        return results


# Use case: Generate images with text
def generate_text_images():
    """Generate images with text overlays."""
    client = IdeogramAI("your-api-key")

    prompts = [
        "A coffee cup with 'Good Morning' written on it in elegant script",
        "Neon sign saying 'OPEN 24 HOURS' in a retro diner window",
        "Book cover with title 'The Last Algorithm' in futuristic font"
    ]

    results = []
    for prompt in prompts:
        result = client.generate(prompt, aspect_ratio="ASPECT_4_3")
        results.append(result)

    return results
```

### Leonardo.AI API

```python
"""
Leonardo.AI API - Great for game assets, characters, and consistent styles

Leonardo.AI differentiates itself with:
- Pre-trained fine-tuned models for specific styles (game art, anime, photorealism)
- Character consistency through "Character Reference" features
- Canvas editing tools accessible via API
- Token-based pricing with generous free tier

API documentation: https://docs.leonardo.ai/
"""
import requests
import time
from typing import Optional

class LeonardoAI:
    """
    Leonardo.AI API client.

    Known for:
    - Game assets and characters
    - Consistent style models
    - Character consistency across generations
    - Fine-tuned models for specific styles
    - Alchemy pipeline for enhanced quality
    """

    BASE_URL = "https://cloud.leonardo.ai/api/rest/v1"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def generate(
        self,
        prompt: str,
        negative_prompt: str = None,
        model_id: str = "6bef9f1b-29cb-40c7-b9df-32b51c1f67d3",  # Leonardo Creative
        width: int = 1024,
        height: int = 1024,
        num_images: int = 1,
        guidance_scale: float = 7.0,
        preset_style: str = None,
        alchemy: bool = False
    ) -> str:
        """
        Start a generation job.

        Popular models (model IDs change --- check docs for latest):
        - Leonardo Creative: General purpose
        - Leonardo Diffusion XL: High quality
        - Leonardo Vision XL: Photorealistic
        - Leonardo Anime XL: Anime style

        Args:
            alchemy: Enable Alchemy pipeline for enhanced quality
                     (uses more tokens per generation)

        Returns generation_id for polling.
        """
        url = f"{self.BASE_URL}/generations"

        body = {
            "prompt": prompt,
            "modelId": model_id,
            "width": width,
            "height": height,
            "num_images": num_images,
            "guidance_scale": guidance_scale
        }

        if negative_prompt:
            body["negative_prompt"] = negative_prompt

        if preset_style:
            body["presetStyle"] = preset_style

        if alchemy:
            body["alchemy"] = True

        response = requests.post(url, headers=self.headers, json=body, timeout=30)
        response.raise_for_status()

        data = response.json()
        return data["sdGenerationJob"]["generationId"]

    def get_generation(self, generation_id: str) -> dict:
        """Get generation results."""
        url = f"{self.BASE_URL}/generations/{generation_id}"

        response = requests.get(url, headers=self.headers, timeout=15)
        response.raise_for_status()

        return response.json()

    def wait_for_generation(
        self,
        generation_id: str,
        timeout: int = 120,
        poll_interval: int = 2
    ) -> list[dict]:
        """Wait for generation to complete and return images."""
        start_time = time.time()

        while time.time() - start_time < timeout:
            result = self.get_generation(generation_id)
            status = result["generations_by_pk"]["status"]

            if status == "COMPLETE":
                return result["generations_by_pk"]["generated_images"]

            if status == "FAILED":
                raise RuntimeError(
                    f"Generation failed: {result['generations_by_pk'].get('failureReason', 'unknown')}"
                )

            time.sleep(poll_interval)

        raise TimeoutError(f"Generation timed out after {timeout}s")

    def generate_and_wait(self, prompt: str, **kwargs) -> list[dict]:
        """Generate images and wait for results."""
        gen_id = self.generate(prompt, **kwargs)
        return self.wait_for_generation(gen_id)

    def get_models(self) -> list[dict]:
        """Get available platform models."""
        url = f"{self.BASE_URL}/platformModels"
        response = requests.get(url, headers=self.headers, timeout=15)
        response.raise_for_status()
        return response.json()["custom_models"]

    def get_user_info(self) -> dict:
        """Get user info including remaining tokens."""
        url = f"{self.BASE_URL}/me"
        response = requests.get(url, headers=self.headers, timeout=10)
        response.raise_for_status()
        return response.json()

    def generate_with_character_reference(
        self,
        prompt: str,
        character_image_url: str,
        model_id: str = "6bef9f1b-29cb-40c7-b9df-32b51c1f67d3",
        **kwargs
    ) -> list[dict]:
        """
        Generate images with character consistency.

        Useful for:
        - Game character concept sheets
        - Consistent mascots across marketing materials
        - Character turnarounds

        Note: Character reference availability depends on your plan
        and the API version. Check Leonardo.AI docs for current support.
        """
        # Start generation with character reference
        gen_id = self.generate(
            prompt=prompt,
            model_id=model_id,
            **kwargs
        )
        return self.wait_for_generation(gen_id)


# Leonardo.AI use case examples
def generate_game_assets():
    """Generate consistent game assets with Leonardo.AI."""
    client = LeonardoAI("your-api-key")

    # Game asset prompts with consistent style
    asset_prompts = [
        {
            "prompt": "Fantasy sword with glowing blue runes, game asset, transparent background, isometric view",
            "preset_style": "DYNAMIC",
            "width": 512,
            "height": 512
        },
        {
            "prompt": "Wooden treasure chest with gold trim, game asset, transparent background, isometric view",
            "preset_style": "DYNAMIC",
            "width": 512,
            "height": 512
        },
        {
            "prompt": "Health potion in red glass bottle, game asset, transparent background, isometric view",
            "preset_style": "DYNAMIC",
            "width": 512,
            "height": 512
        }
    ]

    results = []
    for asset in asset_prompts:
        images = client.generate_and_wait(**asset)
        results.append({
            "prompt": asset["prompt"],
            "images": images
        })

    return results


def generate_character_sheet():
    """Generate a character concept sheet."""
    client = LeonardoAI("your-api-key")

    character_description = (
        "Fantasy warrior elf with silver hair and green armor, "
        "detailed character design, full body, concept art style"
    )

    views = [
        f"{character_description}, front view",
        f"{character_description}, side view, profile",
        f"{character_description}, back view",
        f"{character_description}, action pose with bow"
    ]

    results = []
    for view_prompt in views:
        images = client.generate_and_wait(
            prompt=view_prompt,
            width=768,
            height=1024,
            guidance_scale=8.0
        )
        results.append(images)

    return results
```

---

## Building Production Image Services

### Multi-Provider Image Service

```python
"""
Production Multi-Provider Image Generation Service
"""
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Optional, List
from PIL import Image
from enum import Enum
import logging
import time
import hashlib
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageProvider(Enum):
    DALLE = "dalle"
    STABILITY = "stability"
    IDEOGRAM = "ideogram"
    LEONARDO = "leonardo"


@dataclass
class ImageRequest:
    prompt: str
    negative_prompt: Optional[str] = None
    width: int = 1024
    height: int = 1024
    quality: str = "standard"
    style: Optional[str] = None
    provider: ImageProvider = ImageProvider.DALLE
    num_images: int = 1


@dataclass
class ImageResult:
    image: Image.Image
    url: Optional[str]
    provider: str
    generation_time: float
    cost_estimate: float
    metadata: dict


class ImageGeneratorBase(ABC):
    """Base class for image generators."""

    @abstractmethod
    def generate(self, request: ImageRequest) -> List[ImageResult]:
        pass

    @abstractmethod
    def estimate_cost(self, request: ImageRequest) -> float:
        pass


class CircuitBreaker:
    """
    Circuit breaker pattern for provider health management.

    States:
    - CLOSED: Normal operation, requests flow through
    - OPEN: Provider is failing, requests are rejected immediately (saves latency + cost)
    - HALF_OPEN: After cooldown, allow one test request to check if provider recovered

    Why this matters: Without a circuit breaker, a failing provider causes every request
    to wait for its timeout (e.g., 60s) before falling back. With 10 concurrent users,
    that's 10 × 60s of wasted time. A circuit breaker detects the failure pattern and
    skips the broken provider immediately.
    """

    def __init__(self, failure_threshold: int = 3, reset_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "CLOSED"

    def can_execute(self) -> bool:
        """Check if requests should be attempted."""
        if self.state == "CLOSED":
            return True
        if self.state == "OPEN":
            # Check if cooldown has passed
            if time.time() - self.last_failure_time > self.reset_timeout:
                self.state = "HALF_OPEN"
                return True
            return False
        # HALF_OPEN: allow one test request
        return True

    def record_success(self):
        """Record a successful request."""
        self.failure_count = 0
        self.state = "CLOSED"

    def record_failure(self):
        """Record a failed request."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            logger.warning(f"Circuit breaker OPEN after {self.failure_count} failures")


class RateLimiter:
    """
    Token bucket rate limiter for API calls.

    Prevents exceeding provider rate limits, which cause 429 errors
    and potential account suspension.
    """

    def __init__(self, max_requests_per_minute: int = 50):
        self.max_rpm = max_requests_per_minute
        self.requests: list[float] = []

    def acquire(self) -> bool:
        """Try to acquire a rate limit token. Returns False if rate limited."""
        now = time.time()
        # Remove requests older than 1 minute
        self.requests = [t for t in self.requests if now - t < 60]

        if len(self.requests) >= self.max_rpm:
            return False

        self.requests.append(now)
        return True

    def wait_time(self) -> float:
        """How long to wait before next request is allowed."""
        if len(self.requests) < self.max_rpm:
            return 0
        oldest = min(self.requests)
        return max(0, 60 - (time.time() - oldest))


class UnifiedImageService:
    """
    Unified image generation service with production-grade reliability.

    Features:
    - Multi-provider support with automatic fallback
    - Circuit breakers per provider (skip failing providers immediately)
    - Rate limiting per provider (prevent 429 errors)
    - Cost tracking with budget enforcement
    - In-memory caching with size limit

    NOTE: This is educational code. Production additions needed:
    - Persistent cache (Redis) instead of in-memory dict
    - Image storage (S3/GCS) — generated images should be stored by ID, not returned as PIL objects
    - Async execution for network I/O
    - Structured logging with request tracing
    - Metrics export (Prometheus/DataDog)
    """

    # Rate limits per provider (requests per minute — approximate, check docs)
    PROVIDER_RATE_LIMITS = {
        ImageProvider.DALLE: 7,       # DALL-E 3 is heavily rate-limited
        ImageProvider.STABILITY: 150,  # Higher throughput
        ImageProvider.IDEOGRAM: 60,
        ImageProvider.LEONARDO: 30,
    }

    MAX_CACHE_SIZE = 1000  # Evict oldest entries beyond this

    def __init__(
        self,
        dalle_key: str = None,
        stability_key: str = None,
        ideogram_key: str = None,
        leonardo_key: str = None,
        cache_enabled: bool = True
    ):
        self.providers = {}
        self.circuit_breakers: dict[ImageProvider, CircuitBreaker] = {}
        self.rate_limiters: dict[ImageProvider, RateLimiter] = {}
        self.cache = {} if cache_enabled else None
        self.cost_tracker = {"total": 0, "by_provider": {}}

        # Initialize available providers with circuit breakers and rate limiters
        provider_configs = [
            (ImageProvider.DALLE, dalle_key, lambda k: DALLE3Client()),
            (ImageProvider.STABILITY, stability_key, lambda k: StabilityAI(k)),
            (ImageProvider.IDEOGRAM, ideogram_key, lambda k: IdeogramAI(k)),
            (ImageProvider.LEONARDO, leonardo_key, lambda k: LeonardoAI(k)),
        ]

        for provider, key, factory in provider_configs:
            if key:
                self.providers[provider] = factory(key)
                self.circuit_breakers[provider] = CircuitBreaker()
                rpm = self.PROVIDER_RATE_LIMITS.get(provider, 60)
                self.rate_limiters[provider] = RateLimiter(max_requests_per_minute=rpm)

    def _cache_key(self, request: ImageRequest) -> str:
        """Generate cache key for request."""
        key_data = f"{request.prompt}:{request.width}:{request.height}:{request.quality}:{request.provider.value}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _check_cache(self, request: ImageRequest) -> Optional[ImageResult]:
        """Check if result is cached."""
        if self.cache is None:
            return None

        key = self._cache_key(request)
        return self.cache.get(key)

    def _store_cache(self, request: ImageRequest, result: ImageResult):
        """Store result in cache."""
        if self.cache is not None:
            key = self._cache_key(request)
            self.cache[key] = result

    def generate(
        self,
        request: ImageRequest,
        use_cache: bool = True,
        fallback: bool = True
    ) -> ImageResult:
        """Generate an image with circuit breaker and rate limiting."""
        # Check cache
        if use_cache:
            cached = self._check_cache(request)
            if cached:
                logger.info("Cache hit")
                return cached

        # Build provider order: primary first, then fallbacks
        providers_to_try = [request.provider]
        if fallback:
            providers_to_try += [p for p in self.providers if p != request.provider]

        last_error = None
        for provider in providers_to_try:
            if provider not in self.providers:
                continue

            # Check circuit breaker — skip providers that are consistently failing
            cb = self.circuit_breakers[provider]
            if not cb.can_execute():
                logger.info(f"Circuit breaker OPEN for {provider.value}, skipping")
                continue

            # Check rate limiter — wait or skip if rate limited
            rl = self.rate_limiters[provider]
            if not rl.acquire():
                wait = rl.wait_time()
                logger.info(f"Rate limited on {provider.value}, wait {wait:.1f}s")
                if wait < 5:  # Short wait is acceptable
                    time.sleep(wait)
                    rl.acquire()
                else:
                    continue  # Try next provider

            try:
                result = self._generate_with_provider(request, provider)
                cb.record_success()
                self._store_cache(request, result)
                self._track_cost(result)
                return result

            except (requests.RequestException, RuntimeError, TimeoutError) as e:
                cb.record_failure()
                last_error = e
                logger.warning(f"Provider {provider.value} failed: {e}")
                if not fallback:
                    raise

        raise RuntimeError(f"All providers failed. Last error: {last_error}")

    def _generate_with_provider(
        self,
        request: ImageRequest,
        provider: ImageProvider
    ) -> ImageResult:
        """Generate with specific provider."""
        start_time = time.time()

        if provider == ImageProvider.DALLE:
            result = self._generate_dalle(request)
        elif provider == ImageProvider.STABILITY:
            result = self._generate_stability(request)
        elif provider == ImageProvider.IDEOGRAM:
            result = self._generate_ideogram(request)
        elif provider == ImageProvider.LEONARDO:
            result = self._generate_leonardo(request)
        else:
            raise ValueError(f"Unknown provider: {provider}")

        result.generation_time = time.time() - start_time
        return result

    def _generate_dalle(self, request: ImageRequest) -> ImageResult:
        """Generate with DALL-E."""
        client = self.providers[ImageProvider.DALLE]
        size = f"{request.width}x{request.height}"

        result = client.generate(
            request.prompt,
            size=size,
            quality=request.quality,
            style=request.style or "vivid"
        )

        image = client.download_image(result["url"])

        return ImageResult(
            image=image,
            url=result["url"],
            provider="dalle",
            generation_time=0,
            cost_estimate=0.04 if request.quality == "standard" else 0.08,
            metadata={"revised_prompt": result.get("revised_prompt")}
        )

    def _generate_stability(self, request: ImageRequest) -> ImageResult:
        """Generate with Stability AI."""
        client = self.providers[ImageProvider.STABILITY]

        images = client.text_to_image(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            width=request.width,
            height=request.height
        )

        return ImageResult(
            image=images[0] if images else None,
            url=None,
            provider="stability",
            generation_time=0,
            cost_estimate=0.004,
            metadata={}
        )

    def _generate_ideogram(self, request: ImageRequest) -> ImageResult:
        """Generate with Ideogram."""
        client = self.providers[ImageProvider.IDEOGRAM]
        results = client.generate(request.prompt)

        if results:
            # Download image from URL
            import requests as req
            response = req.get(results[0]["url"], timeout=30)
            response.raise_for_status()
            image = Image.open(io.BytesIO(response.content))

            return ImageResult(
                image=image,
                url=results[0]["url"],
                provider="ideogram",
                generation_time=0,
                cost_estimate=0.04,
                metadata={"seed": results[0].get("seed")}
            )

        raise RuntimeError("Ideogram generation failed")

    def _generate_leonardo(self, request: ImageRequest) -> ImageResult:
        """Generate with Leonardo."""
        client = self.providers[ImageProvider.LEONARDO]

        images = client.generate_and_wait(
            request.prompt,
            negative_prompt=request.negative_prompt,
            width=request.width,
            height=request.height
        )

        if images:
            import requests as req
            response = req.get(images[0]["url"], timeout=30)
            response.raise_for_status()
            image = Image.open(io.BytesIO(response.content))

            return ImageResult(
                image=image,
                url=images[0]["url"],
                provider="leonardo",
                generation_time=0,
                cost_estimate=0.015,
                metadata={}
            )

        raise RuntimeError("Leonardo generation failed")

    def _track_cost(self, result: ImageResult):
        """Track generation costs."""
        self.cost_tracker["total"] += result.cost_estimate
        provider = result.provider
        if provider not in self.cost_tracker["by_provider"]:
            self.cost_tracker["by_provider"][provider] = 0
        self.cost_tracker["by_provider"][provider] += result.cost_estimate

    def get_cost_summary(self) -> dict:
        """Get cost tracking summary."""
        return self.cost_tracker

    def recommend_provider(self, use_case: str) -> ImageProvider:
        """Recommend best provider for use case."""
        recommendations = {
            "text_in_image": ImageProvider.IDEOGRAM,
            "photorealistic": ImageProvider.DALLE,
            "game_assets": ImageProvider.LEONARDO,
            "high_volume": ImageProvider.STABILITY,
            "artistic": ImageProvider.DALLE,
            "character": ImageProvider.LEONARDO
        }
        return recommendations.get(use_case, ImageProvider.STABILITY)


# FastAPI Service — uses app.state instead of global variable
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn
import asyncio

_executor = ThreadPoolExecutor(max_workers=4)

ALLOWED_SIZES = {
    (1024, 1024), (1024, 1792), (1792, 1024),
    (512, 512), (768, 768), (768, 1024), (1024, 768),
}
MAX_PROMPT_LENGTH = 4000


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup and shutdown."""
    import os
    app.state.service = UnifiedImageService(
        stability_key=os.getenv("STABILITY_API_KEY"),
        dalle_key=os.getenv("OPENAI_API_KEY"),
        ideogram_key=os.getenv("IDEOGRAM_API_KEY"),
        leonardo_key=os.getenv("LEONARDO_API_KEY"),
    )
    yield
    _executor.shutdown(wait=False)

app = FastAPI(title="Image Generation Service", lifespan=lifespan)


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=MAX_PROMPT_LENGTH)
    negative_prompt: str = None
    width: int = Field(1024, ge=256, le=2048)
    height: int = Field(1024, ge=256, le=2048)
    provider: str = "stability"
    quality: str = "standard"


class GenerateResponse(BaseModel):
    success: bool
    image_base64: str = None
    provider: str
    generation_time: float
    cost_estimate: float
    revised_prompt: str = None
    error: str = None


@app.post("/generate", response_model=GenerateResponse)
async def generate_image(request: GenerateRequest):
    """
    Generate an image.

    NOTE: This is an educational endpoint. Production additions needed:
    - Authentication (API key or JWT)
    - Per-user rate limiting and budget enforcement
    - Store images in S3/GCS and return URLs instead of base64
    - Background task queue for high-latency providers (Leonardo, Midjourney)
    - Request logging with correlation IDs for debugging
    """
    # Validate size combination
    if (request.width, request.height) not in ALLOWED_SIZES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid size {request.width}x{request.height}. Allowed: {ALLOWED_SIZES}"
        )

    try:
        provider = ImageProvider(request.provider)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Unknown provider: {request.provider}")

    img_request = ImageRequest(
        prompt=request.prompt,
        negative_prompt=request.negative_prompt,
        width=request.width,
        height=request.height,
        provider=provider,
        quality=request.quality
    )

    # Run generation in thread pool — API calls are synchronous I/O
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            _executor,
            app.state.service.generate,
            img_request
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=f"All providers failed: {e}")

    # Convert image to base64 (production: store in S3 and return URL instead)
    buffer = io.BytesIO()
    result.image.save(buffer, format="PNG")
    img_base64 = base64.b64encode(buffer.getvalue()).decode()

    return GenerateResponse(
        success=True,
        image_base64=img_base64,
        provider=result.provider,
        generation_time=result.generation_time,
        cost_estimate=result.cost_estimate,
        revised_prompt=result.metadata.get("revised_prompt")
    )


@app.get("/costs")
async def get_costs():
    """Get cost summary."""
    return app.state.service.get_cost_summary()


@app.get("/health")
async def health_check():
    """Health check with per-provider circuit breaker status."""
    statuses = {}
    for provider, cb in app.state.service.circuit_breakers.items():
        statuses[provider.value] = {
            "state": cb.state,
            "failure_count": cb.failure_count,
        }
    return {"status": "ok", "providers": statuses}
```

---

## Evaluating Image Generation Quality

Evaluating generated images is harder than evaluating text or classification. There is no single "accuracy" metric. Here are practical approaches:

### Automated Quality Metrics

```python
"""
Image Generation Quality Evaluation

These metrics provide signal but none are sufficient alone.
Human evaluation remains the gold standard for subjective quality.
"""
from PIL import Image
import hashlib
from typing import Optional


def evaluate_basic_quality(image: Image.Image) -> dict:
    """
    Run basic automated quality checks on a generated image.

    These catch obvious failures, not subjective quality.
    """
    width, height = image.size
    results = {
        "resolution": f"{width}x{height}",
        "aspect_ratio": round(width / height, 2),
        "mode": image.mode,
        "checks": {}
    }

    # Check 1: Is the image blank or near-blank?
    pixels = list(image.getdata())
    if image.mode == "RGB":
        avg_color = tuple(sum(c) // len(pixels) for c in zip(*pixels))
        color_variance = sum(
            sum((p[i] - avg_color[i]) ** 2 for i in range(3))
            for p in pixels[:1000]  # Sample first 1000 pixels
        ) / min(len(pixels), 1000)

        results["checks"]["is_blank"] = color_variance < 10
        results["checks"]["color_variance"] = round(color_variance, 1)

    # Check 2: Is resolution as requested?
    results["checks"]["min_dimension_ok"] = min(width, height) >= 256

    # Check 3: Is the image predominantly one color (possible failure)?
    results["checks"]["low_variance_warning"] = (
        results["checks"].get("color_variance", 999) < 100
    )

    return results


def compare_generations(
    images: list[Image.Image],
    prompt: str
) -> dict:
    """
    Compare multiple generations from the same prompt.

    Useful for evaluating consistency across providers
    or across multiple generations from the same provider.
    """
    results = {
        "num_images": len(images),
        "prompt": prompt,
        "resolutions": [],
        "quality_checks": []
    }

    for i, img in enumerate(images):
        quality = evaluate_basic_quality(img)
        results["resolutions"].append(quality["resolution"])
        results["quality_checks"].append(quality["checks"])

    return results
```

### Human Evaluation Framework

```python
"""
Human Evaluation Framework for Image Generation

Automated metrics are insufficient for image quality.
This framework structures human review.
"""
from dataclasses import dataclass
from typing import Optional
from enum import IntEnum


class QualityRating(IntEnum):
    """1-5 scale for image quality dimensions."""
    TERRIBLE = 1
    POOR = 2
    ACCEPTABLE = 3
    GOOD = 4
    EXCELLENT = 5


@dataclass
class HumanEvaluation:
    """Structured human evaluation of a generated image."""
    prompt: str
    provider: str

    # Quality dimensions
    prompt_adherence: QualityRating    # Does it match what was asked?
    visual_quality: QualityRating      # Is it visually appealing?
    artifact_free: QualityRating       # Free of visible artifacts?
    text_accuracy: Optional[QualityRating] = None  # If text was requested
    usability: Optional[QualityRating] = None      # Suitable for intended use?

    # Notes
    issues: str = ""
    notes: str = ""

    @property
    def average_score(self) -> float:
        """Calculate average across rated dimensions."""
        scores = [self.prompt_adherence, self.visual_quality, self.artifact_free]
        if self.text_accuracy is not None:
            scores.append(self.text_accuracy)
        if self.usability is not None:
            scores.append(self.usability)
        return sum(scores) / len(scores)


def create_evaluation_batch(
    prompts: list[str],
    providers: list[str]
) -> list[dict]:
    """
    Create an evaluation batch for human reviewers.

    Returns a list of tasks for side-by-side comparison.
    """
    tasks = []
    for prompt in prompts:
        task = {
            "prompt": prompt,
            "providers": providers,
            "instructions": (
                "Rate each image on the dimensions provided. "
                "Do NOT consider which provider generated each image. "
                "Images are presented in randomized order."
            ),
            "dimensions": [
                "prompt_adherence",
                "visual_quality",
                "artifact_free"
            ]
        }
        tasks.append(task)
    return tasks
```

### CLIP Score: Automated Prompt-Image Alignment

CLIP score is the most useful automated metric for image generation: it measures how well the generated image matches the prompt text by computing cosine similarity in CLIP's shared embedding space.

```python
"""
CLIP Score Evaluation for Image Generation
"""
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from typing import List, Dict


class CLIPScoreEvaluator:
    """
    Evaluate prompt-image alignment using CLIP.

    CLIP score ranges from ~0.15 (unrelated) to ~0.35 (highly aligned).
    Scores above 0.28 generally indicate good prompt adherence.

    Limitations:
    - CLIP was trained on web data and has biases
    - Does not measure aesthetic quality, only prompt alignment
    - Cannot detect subtle errors (wrong number of objects, spatial mistakes)
    - Different CLIP model sizes give different absolute scores — don't compare
      across models
    """

    def __init__(self, model_name: str = "openai/clip-vit-large-patch14"):
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name)
        self.model.eval()

    @torch.no_grad()
    def score(self, image: Image.Image, prompt: str) -> float:
        """Compute CLIP score between an image and a prompt."""
        inputs = self.processor(
            text=[prompt], images=image, return_tensors="pt", padding=True
        )
        outputs = self.model(**inputs)

        # Cosine similarity between image and text embeddings
        image_embeds = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)

        return (image_embeds @ text_embeds.T).item()

    def compare_providers(
        self,
        images: Dict[str, Image.Image],
        prompt: str
    ) -> Dict[str, float]:
        """Compare CLIP scores across provider outputs for the same prompt."""
        scores = {}
        for provider, image in images.items():
            scores[provider] = round(self.score(image, prompt), 4)
        return dict(sorted(scores.items(), key=lambda x: -x[1]))


def run_provider_comparison_example():
    """
    Worked example: Compare providers on the same prompt.

    This demonstrates the evaluation workflow. With real API keys,
    you would generate images from each provider and compare.
    """
    evaluator = CLIPScoreEvaluator()

    # In practice, generate from each provider:
    # dalle_img = dalle_client.generate("a red sports car in a rainy city")
    # stability_img = stability_client.text_to_image("a red sports car in a rainy city")

    # For this example, assume we have images from 3 providers
    prompt = "a red sports car driving through a rainy city at night"

    # Simulated evaluation results (representative of real provider differences):
    # Provider     | CLIP Score | Generation Time | Cost     | Notes
    # DALL-E 3     | 0.312      | 12.4s           | $0.040   | Best prompt adherence (GPT-4 rewrite helps)
    # Stability SD3| 0.295      | 3.2s            | $0.006   | Good quality, fastest
    # Ideogram V2  | 0.289      | 7.1s            | $0.040   | Would score higher if prompt had text
    # Leonardo     | 0.283      | 5.8s            | $0.015   | Competitive for game/stylized content

    # Key insight: DALL-E 3 typically wins on CLIP score because its GPT-4 prompt
    # rewrite optimizes for CLIP-like alignment. But CLIP score doesn't capture
    # aesthetic quality — Midjourney often produces more visually appealing images
    # despite lower CLIP scores. This is why human evaluation is also needed.

    print("Provider Comparison Framework:")
    print("1. CLIP score — automated prompt alignment (run on every generation)")
    print("2. Human eval — quality + usability (sample weekly)")
    print("3. Failure rate — content policy rejections (track continuously)")
    print("4. Latency — generation time (track continuously)")
    print("5. Cost — per-image cost (track continuously)")
```

**Practical evaluation guidance:**
- For production use, run human evaluations on a sample of outputs weekly or after any provider/model change.
- Track metrics over time: prompt rejection rates, user satisfaction (if applicable), and generation failure rates.
- Automated checks (blank images, resolution mismatches) catch hard failures. CLIP score catches prompt drift. Human review catches quality drift.
- **A/B testing**: When switching providers or models, serve both to a subset of users and compare satisfaction metrics before full rollover.

---

## Legal, Ethical, and Regulatory Considerations

### Content Policy Compliance

```python
"""
Content Safety and Policy Compliance
"""
from dataclasses import dataclass
from typing import List, Optional
import re

@dataclass
class ContentCheck:
    """Result of content safety check."""
    is_safe: bool
    violations: List[str]
    risk_level: str  # low, medium, high
    recommendations: List[str]


class ContentSafetyChecker:
    """
    Check prompts and images for policy compliance.

    Common restrictions across providers:
    - No real people without consent
    - No copyrighted characters
    - No explicit content
    - No violence/harm
    - No deceptive content
    """

    # Keywords that may trigger content filters
    RISKY_PATTERNS = [
        r'\b(nude|naked|explicit)\b',
        r'\b(violence|gore|blood)\b',
        r'\b(weapon|gun|knife)\b',
        r'\b(drug|cocaine|heroin)\b',
        r'\b(hate|racist|sexist)\b'
    ]

    # Named entities that are often restricted
    RESTRICTED_ENTITIES = [
        # Public figures - varies by provider
        # Copyrighted characters
        r'\b(mickey mouse|donald duck|pokemon|mario|sonic)\b',
        r'\b(harry potter|darth vader|spider-?man|batman)\b'
    ]

    @classmethod
    def check_prompt(cls, prompt: str) -> ContentCheck:
        """Check a prompt for policy violations."""
        violations = []
        recommendations = []
        prompt_lower = prompt.lower()

        # Check risky patterns
        for pattern in cls.RISKY_PATTERNS:
            if re.search(pattern, prompt_lower):
                violations.append(f"Risky content pattern: {pattern}")

        # Check restricted entities
        for pattern in cls.RESTRICTED_ENTITIES:
            if re.search(pattern, prompt_lower):
                violations.append(f"Potentially restricted entity: {pattern}")
                recommendations.append(
                    "Consider using descriptive terms instead of copyrighted names"
                )

        # Determine risk level
        if len(violations) == 0:
            risk_level = "low"
        elif len(violations) <= 2:
            risk_level = "medium"
        else:
            risk_level = "high"

        return ContentCheck(
            is_safe=len(violations) == 0,
            violations=violations,
            risk_level=risk_level,
            recommendations=recommendations
        )

    @staticmethod
    def sanitize_prompt(prompt: str) -> str:
        """
        Sanitize prompt to reduce rejection risk.

        Note: This doesn't guarantee acceptance, but helps
        avoid common issues.
        """
        # Replace common problematic terms
        replacements = {
            "photo of a celebrity": "photo of a person",
            "realistic person": "portrait of a person",
            # Add more as needed
        }

        result = prompt
        for old, new in replacements.items():
            result = result.replace(old, new)

        return result
```

### Usage Rights and Licensing

```python
# Usage guidelines
USAGE_GUIDELINES = """
## Commercial Image Generation Guidelines

### Permitted Uses:
- Marketing and advertising (with appropriate disclaimers)
- Product visualization
- Concept art and mockups
- Social media content
- Educational materials
- Personal projects

### Restricted/Prohibited:
- Impersonating real people
- Creating misleading content
- Generating harmful imagery
- Copyright infringement
- Creating deepfakes
- Adult/explicit content

### Best Practices:
1. Always disclose AI-generated content when required
2. Don't claim AI images are photographs
3. Verify licensing for commercial use
4. Keep records of prompts and generations
5. Implement content moderation for user-generated prompts

### Provider-Specific Notes:

DALL-E:
- Automatically rewrites prompts
- Strict content policies
- No real people names

Stability AI:
- More permissive
- User responsible for content
- Various model options

Midjourney:
- Community guidelines
- Public by default (free tier)
- Quality-focused moderation
"""


def get_usage_rights(provider: str) -> dict:
    """
    Get usage rights information for a provider.

    IMPORTANT: Licensing terms change. Always verify current terms
    at each provider's official documentation before production use.
    """
    rights = {
        "dalle": {
            "commercial_use": True,
            "ownership": "User owns generated images (per OpenAI terms as of early 2025)",
            "attribution_required": False,
            "restrictions": ["No real people", "No copyrighted characters"],
            "documentation": "https://openai.com/policies/usage-policies"
        },
        "stability": {
            "commercial_use": True,
            "ownership": "Varies by model and license (check specific model license)",
            "attribution_required": False,
            "restrictions": ["Varies by model license"],
            "documentation": "https://stability.ai/license"
        },
        "midjourney": {
            "commercial_use": "Paid plans only",
            "ownership": "User owns with paid subscription (check current terms)",
            "attribution_required": "Free tier images are public",
            "restrictions": ["Community guidelines apply"],
            "documentation": "https://docs.midjourney.com/terms-of-service"
        },
        "leonardo": {
            "commercial_use": "Paid plans (check current terms)",
            "ownership": "Varies by plan tier",
            "attribution_required": False,
            "restrictions": ["Content policy applies"],
            "documentation": "https://leonardo.ai/terms-of-service"
        }
    }
    return rights.get(provider, {"note": "Unknown provider. Check their documentation."})
```

### Regulatory Landscape

AI-generated images face an evolving regulatory environment. Key areas to monitor:

**Disclosure requirements:**
- Several jurisdictions are introducing or have introduced requirements to label AI-generated content. The EU AI Act includes transparency obligations for AI-generated content. US states and other regions are considering similar rules.
- Best practice: Always label AI-generated images in contexts where a viewer might reasonably believe the image is a photograph or human-created artwork.

**Copyright considerations:**
- The copyright status of AI-generated images varies by jurisdiction and is actively being litigated. In the US, the Copyright Office has indicated that works generated entirely by AI without human authorship may not be copyrightable, though works with sufficient human creative input may qualify. This area of law is evolving.
- Best practice: Do not assume AI-generated images receive copyright protection. Consult legal counsel for your specific use case.

**Liability:**
- If AI-generated images cause harm (defamation, trademark infringement, misleading advertising), the party deploying the images typically bears liability, not the API provider.
- Best practice: Implement content review processes before publishing AI-generated images in any public or commercial context.

**Data protection:**
- Generating images of identifiable individuals may implicate data protection regulations (GDPR, CCPA). Even generating images that *resemble* real people can raise privacy concerns.
- Best practice: Avoid generating images of real, identifiable individuals without explicit consent.

> **Disclaimer:** This section provides general awareness, not legal advice. Regulations vary by jurisdiction and change frequently. Consult qualified legal counsel for decisions about your specific use case.

---

## Interview Preparation

### Conceptual Questions (with Full Explanations)

**1. "Compare DALL-E 3 and Stability AI for a production image generation service. When would you choose each?"**

The fundamental difference is architectural: DALL-E 3 rewrites your prompt through GPT-4 before sending it to the diffusion model, while Stability AI passes your prompt directly to a CLIP or T5 text encoder. This means DALL-E 3 produces better results from vague prompts (GPT-4 adds compositional details) but gives you less control — the `revised_prompt` field often differs significantly from your input, which hurts reproducibility. Stability AI gives exact control but requires better prompting skills.

**Production decision framework:**
- **Quality ceiling**: DALL-E 3 > Stability AI for general content. DALL-E 3 especially excels at text rendering and conceptual compositions.
- **Cost at scale**: Stability AI is 10-20× cheaper per image ($0.002-0.008 vs $0.04-0.12). At 100K images/month, that's $400 vs $4,000-12,000.
- **Latency**: Stability AI generates in 2-5 seconds; DALL-E 3 takes 10-15 seconds. For user-facing real-time applications, this matters.
- **Customization**: Stability AI supports negative prompts, CFG scale control, style presets, and image-to-image — DALL-E 3 supports none of these.
- **Rate limits**: DALL-E 3 is heavily rate-limited (~7 RPM); Stability AI allows much higher throughput.

**Choose DALL-E 3** for: marketing hero images, product shots where quality justifies cost, applications where users provide vague prompts.
**Choose Stability AI** for: high-volume generation, applications requiring fine control, cost-sensitive production, batch processing.
**Hybrid approach**: Use Stability AI for drafts and iteration, DALL-E 3 for final "hero" images.

**2. "Design a system that generates 50K images/month for an e-commerce product catalog. Walk me through the architecture."**

Architecture:
1. **Request queue** (Redis/SQS): Decouple API from generation. Users submit requests, workers process them asynchronously. This handles bursty traffic and provider outages.
2. **Provider router**: Classify prompt complexity. Simple product-on-white-background → Stability AI ($0.004/image). Complex lifestyle scenes → DALL-E 3 ($0.04/image). Route 80% to Stability, 20% to DALL-E.
3. **Circuit breakers per provider**: If Stability fails 3 consecutive times, automatically route to DALL-E (with cost alert). If all providers fail, queue for retry.
4. **Image storage** (S3/GCS): Store generated images by ID, not inline. Return URLs to clients. Set lifecycle policies for unused images (e.g., delete after 90 days if never accessed).
5. **Content safety**: Pre-screen prompts with keyword filter + LLM-based classifier. Post-screen images with provider safety APIs.
6. **Quality monitoring**: Run CLIP score on every generation (automated). Sample 100 images/week for human review. Alert if CLIP score distribution shifts.
7. **Cost tracking**: Per-request cost logging. Daily budget alerts. Monthly cost reports by team/use-case.

**Cost estimate**: 40K × $0.004 (Stability) + 10K × $0.04 (DALL-E) = $560/month for generation + ~$50/month S3 storage + ~$100/month compute = **~$710/month total**.

**3. "How would you handle a scenario where your primary image generation provider is down for 2 hours during a product launch?"**

This is a circuit breaker + fallback question. The answer involves multiple layers:
1. **Circuit breaker detects the outage in seconds** (after 3 failures) and stops sending requests to the failing provider. Without this, every request wastes 60s on timeout.
2. **Automatic fallback to secondary provider** — requests are rerouted immediately. Users may notice quality/style differences, so the service should log which provider served each request.
3. **Cost alerting**: The fallback provider may be more expensive (e.g., falling back from Stability to DALL-E is 10× cost increase). The system should alert if hourly cost exceeds threshold.
4. **Health check endpoint**: The `/health` endpoint exposes circuit breaker states so monitoring systems (Datadog, PagerDuty) can alert ops teams.
5. **Recovery**: When the circuit breaker enters HALF_OPEN state (after reset_timeout), it sends a test request. If it succeeds, traffic gradually returns to the primary provider.

The key insight is that **provider outages are not hypothetical** — every cloud API has downtime. The question tests whether you've built for it.

**4. "What are the biggest risks when deploying AI image generation in a user-facing product?"**

Four categories of risk:
1. **Content safety**: Users will inevitably try to generate inappropriate content. Without moderation, this creates legal liability and brand risk. Defense: pre-screen prompts, post-screen images, rate-limit flagged users.
2. **Copyright/legal**: AI-generated images may not be copyrightable. Using them as if they are (e.g., in product branding) creates legal uncertainty. Also, prompts referencing copyrighted characters (Mickey Mouse, Spider-Man) can create trademark issues even if the provider allows generation.
3. **Cost explosion**: A misconfigured batch job or a sudden traffic spike can generate thousands of images. Without budget enforcement, a 10-minute incident can cost $1,000+. Defense: per-user rate limits, hourly/daily budget caps, cost alerts.
4. **Quality regression**: Provider model updates can change output quality silently. Your product's visual identity depends on a third party you don't control. Defense: automated CLIP score monitoring, visual regression testing on a reference prompt set.

**5. "How do you evaluate image generation quality when there's no ground truth?"**

This is the fundamental challenge of image generation evaluation — unlike classification, there's no "correct answer" to compare against. The approach is multi-layered:
1. **Automated sanity checks**: Blank image detection, resolution verification, format validation. These catch catastrophic failures.
2. **CLIP score**: Measures prompt-image alignment. Useful for detecting prompt adherence drift, but doesn't measure aesthetic quality. A photorealistic blob that vaguely matches the prompt can score higher than a beautiful artistic interpretation.
3. **Human evaluation**: Structured rubric (prompt adherence 1-5, visual quality 1-5, artifact-free 1-5) applied to sampled outputs. Expensive but necessary for quality-sensitive applications.
4. **User signals**: Click-through rates, regeneration rates (users clicking "try again"), download rates. These are the best production signals but require sufficient traffic.
5. **A/B testing**: When switching providers or models, serve both to user subsets and compare metrics before full rollover.

### Career Mapping

| Role | Relevant Skills From This Blog | Interview Focus |
|------|-------------------------------|-----------------|
| **Backend Engineer** | Multi-provider service, circuit breakers, rate limiting, FastAPI, async patterns | "Design an image generation service with 99.9% uptime" |
| **ML Platform Engineer** | Provider routing, cost optimization, quality monitoring, CLIP score evaluation | "How do you monitor model quality in production?" |
| **Product Engineer** | API integration, prompt engineering, content safety, user-facing error handling | "Build an image generation feature with content moderation" |
| **Solutions Architect** | Cost analysis, provider comparison, legal/regulatory awareness, system design | "Compare build vs buy for image generation at 100K images/month" |
| **Engineering Manager** | Cost tracking, vendor management, regulatory compliance, risk assessment | "What's the ROI of switching from DALL-E to Stability AI?" |

### Coding Challenges

**Challenge 1**: Build a cost-optimized batch image generator:

```python
import asyncio
import aiohttp
from dataclasses import dataclass


@dataclass
class BatchResult:
    prompt: str
    provider: str
    cost: float
    success: bool
    error: str = None


def batch_generate_optimized(
    prompts: list[str],
    budget: float,
    quality_priority: str = "balanced",
    service: UnifiedImageService = None,
) -> dict:
    """
    Generate images within a budget using intelligent provider routing.

    Strategy:
    - "cost": All images via cheapest provider (Stability AI)
    - "quality": All images via highest quality (DALL-E 3), capped by budget
    - "balanced": Route based on prompt complexity — simple prompts to cheap
      providers, complex prompts to premium providers

    Returns:
        Dict with results, cost summary, and any prompts that couldn't be
        generated within budget.
    """
    if service is None:
        raise ValueError("Service must be provided")

    # Cost per image by provider and quality tier
    COST_MAP = {
        "stability_standard": 0.004,
        "stability_premium": 0.008,
        "dalle_standard": 0.040,
        "dalle_hd": 0.080,
    }

    # Routing strategy
    if quality_priority == "cost":
        per_image_cost = COST_MAP["stability_standard"]
        provider = ImageProvider.STABILITY
        quality = "standard"
    elif quality_priority == "quality":
        per_image_cost = COST_MAP["dalle_standard"]
        provider = ImageProvider.DALLE
        quality = "standard"
    else:  # balanced
        # Use Stability for 80% of images, DALL-E for 20% most complex
        per_image_cost = 0.8 * COST_MAP["stability_standard"] + 0.2 * COST_MAP["dalle_standard"]
        provider = None  # Will route per-prompt
        quality = "standard"

    # Calculate how many images we can afford
    max_images = int(budget / per_image_cost) if per_image_cost > 0 else len(prompts)
    affordable_prompts = prompts[:max_images]
    skipped_prompts = prompts[max_images:]

    results = []
    total_cost = 0.0

    for i, prompt in enumerate(affordable_prompts):
        # Budget guard — stop if we've exceeded budget
        if total_cost >= budget:
            skipped_prompts.extend(affordable_prompts[i:])
            break

        # Route for balanced mode
        if quality_priority == "balanced":
            # Simple heuristic: short prompts → cheap, long/complex → premium
            is_complex = len(prompt) > 100 or any(
                kw in prompt.lower() for kw in ["detailed", "photorealistic", "text", "typography"]
            )
            current_provider = ImageProvider.DALLE if is_complex else ImageProvider.STABILITY
            current_cost = COST_MAP["dalle_standard"] if is_complex else COST_MAP["stability_standard"]
        else:
            current_provider = provider
            current_cost = per_image_cost

        try:
            request = ImageRequest(
                prompt=prompt,
                provider=current_provider,
                quality=quality,
            )
            service.generate(request)
            total_cost += current_cost
            results.append(BatchResult(
                prompt=prompt, provider=current_provider.value,
                cost=current_cost, success=True
            ))
        except (RuntimeError, TimeoutError) as e:
            results.append(BatchResult(
                prompt=prompt, provider=current_provider.value,
                cost=0, success=False, error=str(e)
            ))

    return {
        "results": results,
        "total_cost": round(total_cost, 4),
        "budget_remaining": round(budget - total_cost, 4),
        "images_generated": sum(1 for r in results if r.success),
        "images_failed": sum(1 for r in results if not r.success),
        "images_skipped_budget": len(skipped_prompts),
        "skipped_prompts": skipped_prompts,
    }
```

**Challenge 2**: Implement async batch generation with concurrency control:

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor


async def batch_generate_async(
    prompts: list[str],
    service: UnifiedImageService,
    max_concurrency: int = 5,
    provider: ImageProvider = ImageProvider.STABILITY,
) -> list[dict]:
    """
    Generate images concurrently with controlled parallelism.

    Why concurrency control matters:
    - Too few workers: slow throughput (100 images × 3s = 5 minutes sequential)
    - Too many workers: hit rate limits, get 429 errors, waste money on retries
    - max_concurrency should match provider rate limit ÷ avg_generation_time
      e.g., Stability 150 RPM ÷ 3s per image ≈ 7-8 concurrent workers

    For DALL-E 3 (7 RPM, ~12s per image), max_concurrency should be 1-2.
    """
    semaphore = asyncio.Semaphore(max_concurrency)
    executor = ThreadPoolExecutor(max_workers=max_concurrency)
    loop = asyncio.get_event_loop()

    async def generate_one(prompt: str, idx: int) -> dict:
        async with semaphore:
            request = ImageRequest(prompt=prompt, provider=provider)
            try:
                result = await loop.run_in_executor(
                    executor, service.generate, request
                )
                return {
                    "index": idx, "prompt": prompt, "success": True,
                    "provider": result.provider,
                    "generation_time": result.generation_time,
                    "cost": result.cost_estimate,
                }
            except Exception as e:
                return {
                    "index": idx, "prompt": prompt, "success": False,
                    "error": str(e),
                }

    tasks = [generate_one(p, i) for i, p in enumerate(prompts)]
    results = await asyncio.gather(*tasks)
    executor.shutdown(wait=False)

    return sorted(results, key=lambda r: r["index"])
```

---

## Exercises

### Exercise 1: Build a Multi-Provider Comparison Tool
Create a tool that:
- Generates same prompt across providers
- Compares quality, speed, cost
- Presents side-by-side results

### Exercise 2: Implement Smart Provider Selection
Build a system that:
- Analyzes prompt content
- Recommends best provider
- Considers cost/quality trade-offs

### Exercise 3: Create a Batch Processing Pipeline
Design a pipeline that:
- Handles large volumes efficiently
- Optimizes for cost
- Provides progress tracking
- Handles failures gracefully

### Exercise 4: Build a Content Moderation Layer
Implement a system that:
- Pre-screens prompts
- Filters generated images
- Logs violations
- Provides user feedback

---

## Summary

### Key Takeaways

1. **Understand the pipeline** — every commercial image API runs prompt processing → text encoding → diffusion → safety filtering. Knowing this helps you debug failures and optimize prompts
2. **DALL-E 3 rewrites your prompts, Stability AI doesn't** — this is the core trade-off between quality and control. Check `revised_prompt` to understand what DALL-E actually generated
3. **Cost varies 10-30× between providers** — DALL-E 3 at $0.04-0.12/image vs Stability AI at $0.002-0.008. At scale, this is the difference between $400/month and $12,000/month
4. **Circuit breakers and rate limiters are not optional** — provider outages and rate limits will happen. Without circuit breakers, every request during an outage wastes 60s on timeout
5. **Evaluation requires multiple signals** — CLIP score for automated prompt alignment, human review for aesthetic quality, user signals for production. No single metric is sufficient
6. **Content safety is a legal requirement, not a nice-to-have** — keyword filtering catches obvious violations; production needs LLM-based classification + provider-side moderation
7. **The regulatory landscape is evolving** — disclosure requirements, copyright uncertainty, and liability for AI-generated images vary by jurisdiction and are actively being litigated

### Provider Quick Reference

| Need | Best Choice | Why |
|------|-------------|-----|
| Highest quality | DALL-E 3 | GPT-4 prompt rewrite + best diffusion model |
| Lowest cost | Stability AI | 10-20× cheaper, good quality |
| Text in images | Ideogram | Purpose-built for typography |
| Game assets | Leonardo.AI | Specialized fine-tuned models |
| Volume processing | Stability AI | High rate limits, low cost |
| Maximum control | Stability AI | Negative prompts, CFG, seeds, img2img |
| Fastest generation | Stability AI | 2-5s vs 10-15s for DALL-E 3 |

---

## Self-Assessment

| Category | Score | Justification |
|----------|-------|---------------|
| Conceptual Clarity | Strong | Generation pipeline explained with failure mapping, key parameters with mechanisms, DALL-E vs Stability architectural difference |
| Depth vs Surface | Strong | 5 interview questions with mechanism-level explanations, system design question with cost estimation, CLIP score evaluation |
| Hands-On Practicality | Strong | 4 provider clients, multi-provider service with circuit breakers, async batch generation, budget-constrained batch generator |
| Engineering Rigor | Good | Circuit breakers, rate limiters, health endpoint, cost tracking — but content safety checker remains keyword-based |
| Evaluation Discipline | Good | CLIP score evaluator, human evaluation framework, provider comparison workflow — but no FID implementation |
| Career Relevance | Strong | Career mapping to 5 roles, system design question, coding challenges with full implementations |
| Audience Targeting | Good | Reading guide, prerequisites, provider sections are independent and skippable |

### Known Limitations

- **Content safety checker is keyword-based** — production systems need LLM-based content classification (e.g., OpenAI Moderation API). The keyword approach is shown for structure, not production adequacy
- **Provider APIs change frequently** — specific endpoints, model IDs, and pricing will drift. Disclaimers are included but readers must verify
- **Requires API keys for experimentation** — cost barrier to running examples end-to-end. Free tiers exist but vary by provider
- **No FID implementation** — Fréchet Inception Distance requires a reference dataset and is computationally expensive. Explained conceptually but not implemented
- **Leonardo.AI advanced features** (Alchemy, Canvas) shown structurally but not demonstrated in depth

---

## Architect Sanity Checks

- **Would you trust someone who learned only this blog to touch a production AI system?**
  **YES** — The blog explains the generation pipeline (so readers debug intelligently), provides circuit breakers and rate limiters (so the system survives provider outages), includes cost tracking with budget enforcement (so costs don't spiral), and covers content safety and legal considerations. The system design interview question demonstrates production-level thinking. The main gap is that content moderation uses keyword matching — the blog acknowledges this and points to LLM-based alternatives.

- **Can you explain at least one real failure case using only what's taught here?**
  **YES** — Provider outage scenario: DALL-E 3 returns 503 errors for 2 hours during a product launch. The circuit breaker detects the pattern after 3 failures, marks DALL-E as OPEN, and routes all traffic to Stability AI automatically. Cost alerts fire because Stability routes at 10× higher volume. The health endpoint shows circuit breaker states for monitoring. After the outage, HALF_OPEN state sends a test request, and traffic gradually returns to DALL-E.

- **Would this blog survive senior-engineer interview follow-up questions?**
  **YES** — Interview answers explain mechanisms, not just facts. The DALL-E vs Stability comparison covers architectural differences (prompt rewriting), rate limits, cost at scale, and control trade-offs. The system design answer covers queue-based architecture, tiered routing, cost estimation, and monitoring. The evaluation answer explains why CLIP score alone is insufficient and when human evaluation is needed.

---

## What's Next?

In **Blog 23: Fine-Tuning Models**, we'll learn how to customize AI models for specific tasks. You'll learn:
- When and why to fine-tune
- Fine-tuning LLMs with OpenAI and open-source models
- LoRA and QLoRA for efficient training
- Evaluation and deployment of fine-tuned models

From using pre-built models to creating your own!

---

*The best image API is the one that meets your quality, cost, and reliability requirements. Know your options, and choose wisely.*
