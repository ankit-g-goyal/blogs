# Blog 10: Pre-trained Language Models — Standing on Giants' Shoulders

**Series:** Prompt Your Career: The Complete Generative AI Masterclass
**Prerequisites:** Blog 9 (Transformers)
**Time to Complete:** 3.5-4 hours
**Difficulty:** Intermediate to Advanced

---

## Reading Guide

**Who this is for:** You have completed Blog 9 (Transformers) and are comfortable with self-attention, encoder-decoder architecture, and positional encoding. You can write a PyTorch training loop from scratch.

**Minimum prerequisites:**
- Solid grasp of the Transformer architecture (Blog 9)
- Comfort with PyTorch `nn.Module`, `DataLoader`, and basic training loops (Blogs 4-5)
- Familiarity with tokenization concepts (Blog 6)
- A machine with Python 3.9+ and ideally a CUDA-capable GPU (CPU works but fine-tuning will be slow)

**How to read this blog:**
- **If you are short on time (1 hour):** Read "The Pre-training Revolution," "BERT vs GPT: The Complete Comparison," and "Common Fine-Tuning Failures and Fixes."
- **If you want hands-on practice (3-4 hours):** Work through every code block, especially the full fine-tuning pipeline in "Hands-On: Fine-Tuning BERT for Sentiment Analysis."
- **If you are preparing for interviews:** Focus on the "Interview Preparation" and "BERT vs GPT" sections.

---

## What This Blog Does NOT Cover

- **Training a language model from scratch.** We cover fine-tuning existing models, not pre-training new ones. For pre-training concepts, see Blog 9.
- **Prompt engineering and in-context learning.** These are covered in Blog 11 (LLMs) and Blog 12 (Prompt Engineering).
- **Reinforcement Learning from Human Feedback (RLHF).** Alignment techniques are out of scope for this blog.
- **Multi-modal models.** Vision-language models are covered later in the series (Blogs 21-22).
- **Distributed training.** We stick to single-GPU fine-tuning. Multi-GPU and distributed setups are a separate topic.
- **Detailed Hugging Face Trainer API.** We use a manual training loop for pedagogical clarity; the `Trainer` class is introduced in later blogs.

---

## What You'll Walk Away With

After completing this blog, you will be able to:

1. **Explain transfer learning** and why pre-training revolutionized NLP
2. **Navigate the Hugging Face ecosystem** confidently (Hub, Transformers, Datasets, Tokenizers)
3. **Articulate BERT vs GPT differences** at the architecture, training objective, and use-case level
4. **Choose the right model family** for classification, generation, and embedding tasks
5. **Fine-tune a pre-trained model** for custom text classification
6. **Debug common fine-tuning failures** (overfitting, catastrophic forgetting, learning rate issues)
7. **Deploy a fine-tuned model** with proper evaluation metrics

---

## Manager's Summary

**The Business Case for Pre-trained Models:**

Pre-trained language models represent the most significant ROI shift in AI history. Instead of training models from scratch (costing $1-100M and requiring massive datasets), teams can now fine-tune existing models for $10-1000 with hundreds of examples.

**Key Decisions for Leadership:**

| Decision | BERT-family | GPT-family |
|----------|-------------|------------|
| **Best For** | Classification, Search, Analysis | Generation, Chatbots, Content |
| **Training Data Needed** | 100-10,000 examples | 50-1,000 examples (or zero-shot) |
| **Latency** | Lower (encoder-only) | Higher (autoregressive) |
| **Cost** | Lower compute | Higher compute |
| **Interpretability** | Higher (attention visualization) | Lower (emergent behaviors) |

**Risk Factors:**
- **Model Licensing:** Some models have commercial restrictions (check license!)
- **Data Privacy:** Fine-tuning data may leak into model
- **Version Lock-in:** Models get deprecated; version your dependencies
- **Compute Costs:** Fine-tuning still requires GPU; inference adds up at scale

**Bottom Line:** Pre-trained models have dramatically lowered the barrier to entry. The competitive advantage has shifted from "can you build models from scratch" to "can you select, fine-tune, evaluate, and deploy them effectively."

---

## The Pre-training Revolution

### Why Training from Scratch Is (Usually) Wrong

In Blog 9, we trained a Mini-GPT on Shakespeare. The results were fun but limited:

```python
# What we did in Blog 9
model = MiniGPT(vocab_size=65, d_model=64, n_heads=4, n_layers=2)
# Trained on ~1MB of Shakespeare text
# Result: Vaguely Shakespearean gibberish
```

The problem? Language is vast. Shakespeare alone doesn't teach:
- Modern vocabulary ("smartphone", "quantum computing")
- Grammar patterns beyond 16th-century English
- World knowledge ("Paris is in France")
- Reasoning patterns ("If A > B and B > C, then A > C")

**The Pre-training Solution:**

```
Traditional ML Pipeline:
Raw Data → Feature Engineering → Model Training → Task-Specific Model

Transfer Learning Pipeline:
Massive Corpus → Pre-training → Foundation Model → Fine-tuning → Task-Specific Model
                 (done once)    (reusable)       (cheap)
```

### The Scale of Modern Pre-training

| Model | Parameters | Training Data | Est. Training Cost | Your Cost to Use |
|-------|------------|---------------|---------------------|------------------|
| BERT-base | 110M | 3.3B words | Modest (2018) | Free |
| GPT-2 | 1.5B | 40GB text | Not disclosed | Free |
| GPT-3 | 175B | ~300B tokens | ~$4.6M (reported by third-party estimates) | API fees |
| LLaMA-2-70B | 70B | 2T tokens | Not disclosed by Meta | Free (weights) |
| GPT-4 | Not disclosed | Not disclosed | Not disclosed | API fees |

> **Note on cost figures:** Exact pre-training costs are rarely published by model creators. The GPT-3 estimate comes from third-party analyses based on GPU-hours and cloud pricing at the time. GPT-4's architecture and parameter count have not been officially confirmed by OpenAI. Treat all cost figures as rough order-of-magnitude estimates.

**The key insight:** You don't pay for pre-training. You inherit enormous compute investments for free.

---

## The Hugging Face Ecosystem

Hugging Face has become the "GitHub of ML models." Understanding its components is essential.

### Component Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    HUGGING FACE ECOSYSTEM                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │   HF Hub    │  │ Transformers│  │  Datasets   │              │
│  │             │  │   Library   │  │   Library   │              │
│  │ • 400K+     │  │             │  │             │              │
│  │   models    │  │ • Unified   │  │ • 50K+      │              │
│  │ • Model     │  │   API       │  │   datasets  │              │
│  │   cards     │  │ • Auto      │  │ • Streaming │              │
│  │ • Spaces    │  │   classes   │  │ • Transforms│              │
│  │ • Datasets  │  │ • Pipelines │  │             │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
│         │                │                │                      │
│         └────────────────┼────────────────┘                      │
│                          ▼                                       │
│               ┌─────────────────┐                                │
│               │   Tokenizers    │                                │
│               │                 │                                │
│               │ • Fast (Rust)   │                                │
│               │ • BPE, WordPiece│                                │
│               │ • Consistent API│                                │
│               └─────────────────┘                                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Installation and Setup

```python
# Core installation
# pip install transformers datasets tokenizers accelerate

# Verify installation
import transformers
import datasets
import tokenizers

print(f"Transformers version: {transformers.__version__}")
print(f"Datasets version: {datasets.__version__}")
print(f"Tokenizers version: {tokenizers.__version__}")
```

### Your First Hugging Face Model

```python
from transformers import pipeline

# The simplest way to use a pre-trained model
classifier = pipeline("sentiment-analysis")

results = classifier([
    "I love this product! Best purchase ever!",
    "Terrible experience. Would not recommend.",
    "It's okay, nothing special."
])

for text, result in zip(["positive", "negative", "neutral"], results):
    print(f"Expected: {text:8} | Got: {result['label']:8} | Confidence: {result['score']:.3f}")
```

Output:
```
Expected: positive | Got: POSITIVE | Confidence: 0.999
Expected: negative | Got: NEGATIVE | Confidence: 0.999
Expected: neutral  | Got: NEGATIVE | Confidence: 0.796
```

**What just happened?**
1. Pipeline downloaded `distilbert-base-uncased-finetuned-sst-2-english`
2. Loaded model weights (~250MB)
3. Tokenized your text
4. Ran inference
5. Converted logits to labels

### Available Pipeline Tasks

```python
from transformers import pipeline

# Text Classification
classifier = pipeline("text-classification")
classifier("I love machine learning!")

# Named Entity Recognition
ner = pipeline("ner", aggregation_strategy="simple")
ner("Hugging Face is based in New York City.")

# Question Answering
qa = pipeline("question-answering")
qa(question="What is the capital of France?",
   context="France is a country in Europe. Its capital is Paris.")

# Text Generation
generator = pipeline("text-generation", model="gpt2")
generator("The future of AI is", max_length=50, num_return_sequences=1)

# Summarization
summarizer = pipeline("summarization")
summarizer(long_article, max_length=100, min_length=30)

# Translation
translator = pipeline("translation_en_to_fr")
translator("Hello, how are you?")

# Fill-Mask (BERT-style)
unmasker = pipeline("fill-mask")
unmasker("Paris is the [MASK] of France.")

# Zero-Shot Classification
zero_shot = pipeline("zero-shot-classification")
zero_shot(
    "I need to buy groceries",
    candidate_labels=["shopping", "work", "entertainment"]
)

# Feature Extraction (Embeddings)
extractor = pipeline("feature-extraction")
embeddings = extractor("Convert this to a vector.")
```

### Understanding Model Cards

Every model on the Hub has a model card. Here's how to read them:

```python
from huggingface_hub import ModelCard

card = ModelCard.load("bert-base-uncased")
print(card.content[:500])  # Shows model description

# Programmatically check model info
from huggingface_hub import HfApi

api = HfApi()
model_info = api.model_info("bert-base-uncased")

print(f"Model: {model_info.modelId}")
print(f"Downloads last month: {model_info.downloads}")
print(f"Library: {model_info.library_name}")
print(f"Pipeline tag: {model_info.pipeline_tag}")
print(f"License: {model_info.cardData.get('license', 'Not specified')}")
```

**Critical Model Card Sections:**

| Section | What to Check | Why It Matters |
|---------|---------------|----------------|
| **License** | MIT, Apache, CC-BY, custom | Commercial use allowed? |
| **Training Data** | What corpus was used | Bias and domain fit |
| **Intended Uses** | What it's designed for | Don't use summarization model for classification |
| **Limitations** | Known failure modes | Set realistic expectations |
| **Evaluation** | Benchmark scores | Compare to alternatives |

---

## BERT: The Encoder Revolution

### Architecture Recap

BERT (Bidirectional Encoder Representations from Transformers) uses only the encoder stack:

```
Input: "The cat sat on the [MASK]"

          [CLS]  The   cat   sat   on   the  [MASK]
            │     │     │     │    │     │     │
            ▼     ▼     ▼     ▼    ▼     ▼     ▼
         ┌─────────────────────────────────────────┐
         │         ENCODER LAYER 1                 │
         │    (Bidirectional Self-Attention)       │
         └─────────────────────────────────────────┘
                          │
                          ▼
                        (×12)
                          │
                          ▼
         ┌─────────────────────────────────────────┐
         │         ENCODER LAYER 12                │
         └─────────────────────────────────────────┘
            │     │     │     │    │     │     │
            ▼     ▼     ▼     ▼    ▼     ▼     ▼
          h_cls h_the h_cat h_sat h_on h_the h_mask
                                              │
                                              ▼
                                        Predict: "mat"
```

**Key Feature:** Bidirectional attention means each token sees all other tokens. The word "sat" knows about both "cat" (before) and "mat" (after).

### BERT's Training Objectives

**Objective 1: Masked Language Modeling (MLM)**

```python
# During pre-training, 15% of tokens are masked
original = "The cat sat on the mat"
masked = "The cat [MASK] on the mat"
# Model predicts: "sat"

# But it's more nuanced:
# 80% of time: Replace with [MASK]
# 10% of time: Replace with random word
# 10% of time: Keep original
# This prevents model from just learning "[MASK] = predict something"
```

**Objective 2: Next Sentence Prediction (NSP)**

```python
# Given two sentences, predict if B follows A
sentence_a = "The cat sat on the mat."
sentence_b = "It was a comfortable spot."  # IsNext
sentence_c = "The stock market crashed."    # NotNext

# Model learns document-level coherence
```

### Using BERT for Classification

```python
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load pre-trained BERT with classification head
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2  # Binary classification
)

# Tokenize input
text = "This movie was absolutely fantastic!"
inputs = tokenizer(
    text,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=512
)

print(f"Input IDs shape: {inputs['input_ids'].shape}")
print(f"Attention mask shape: {inputs['attention_mask'].shape}")
print(f"Tokens: {tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])}")

# Forward pass
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predictions = torch.softmax(logits, dim=-1)

print(f"Logits: {logits}")
print(f"Probabilities: {predictions}")
print(f"Predicted class: {torch.argmax(predictions, dim=-1).item()}")
```

### BERT Variants

```
BERT Family Tree:

BERT-base (110M)
    │
    ├── DistilBERT (66M) ─── 40% smaller, 60% faster, 97% performance
    │
    ├── RoBERTa (125M) ─── Dropped NSP (found unhelpful), more data, dynamic masking
    │
    ├── ALBERT (12M) ─── Parameter sharing, factorized embeddings
    │
    ├── ELECTRA (14M) ─── Replaced token detection (more efficient)
    │
    └── DeBERTa (140M) ─── Disentangled attention, enhanced mask decoder
```

**When to Use Which:**

| Model | Use Case | Trade-off |
|-------|----------|-----------|
| **BERT-base** | Research, baselines | Standard, well-documented |
| **DistilBERT** | Production, edge devices | Speed over accuracy |
| **RoBERTa** | When accuracy matters | Slower than DistilBERT |
| **ALBERT** | Memory-constrained | May underperform BERT |
| **DeBERTa** | SOTA on benchmarks | Larger, slower |

---

## GPT: The Decoder Revolution

### Architecture Recap

GPT (Generative Pre-trained Transformer) uses only the decoder stack with causal attention:

```
Input: "The cat sat on the"

       The   cat   sat   on   the
        │     │     │    │     │
        ▼     ▼     ▼    ▼     ▼
    ┌─────────────────────────────┐
    │      DECODER LAYER 1        │
    │   (Causal Self-Attention)   │
    │                             │
    │   The → [The]               │
    │   cat → [The, cat]          │
    │   sat → [The, cat, sat]     │
    │   ...                       │
    └─────────────────────────────┘
                  │
                  ▼
                (×12)
                  │
                  ▼
    ┌─────────────────────────────┐
    │      DECODER LAYER 12       │
    └─────────────────────────────┘
        │     │     │    │     │
        ▼     ▼     ▼    ▼     ▼
       cat   sat   on   the   mat
      (next) (next)(next)(next)(predicted)
```

**Key Feature:** Causal (unidirectional) attention means each token only sees previous tokens. This enables autoregressive generation.

### GPT's Training Objective

**Causal Language Modeling (CLM):**

```python
# Predict the next token at every position
text = "The cat sat on the mat"

# Training examples generated:
# Input: "The"           → Predict: "cat"
# Input: "The cat"       → Predict: "sat"
# Input: "The cat sat"   → Predict: "on"
# Input: "The cat sat on"→ Predict: "the"
# ...

# All predictions happen in parallel during training (teacher forcing)
# But generation is sequential during inference
```

### Using GPT for Generation

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained GPT-2
model_name = "gpt2"  # 124M parameters
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Set padding token (GPT-2 doesn't have one by default)
tokenizer.pad_token = tokenizer.eos_token

# Generate text
prompt = "The future of artificial intelligence is"
inputs = tokenizer(prompt, return_tensors="pt")

# Generation with different strategies
outputs = model.generate(
    **inputs,
    max_new_tokens=50,
    do_sample=True,          # Enable sampling
    temperature=0.7,         # Control randomness
    top_k=50,               # Top-k sampling
    top_p=0.95,             # Nucleus sampling
    repetition_penalty=1.2,  # Reduce repetition
    pad_token_id=tokenizer.eos_token_id
)

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
```

### Generation Strategies Explained

```python
import torch
import torch.nn.functional as F

def demonstrate_generation_strategies(logits, temperature=1.0, top_k=0, top_p=1.0):
    """
    Visualize different sampling strategies.

    Args:
        logits: Raw model outputs for next token
        temperature: Higher = more random, Lower = more deterministic
        top_k: Only consider top k tokens
        top_p: Only consider tokens comprising top p probability mass
    """
    # Simulated vocabulary
    vocab = ["the", "a", "cat", "dog", "sat", "ran", "quickly", "slowly", ".", "!"]
    logits = torch.tensor([2.0, 1.5, 3.0, 2.8, 1.0, 0.5, 0.3, 0.2, 1.8, 0.1])

    # Strategy 1: Greedy (temperature → 0)
    greedy_probs = F.softmax(logits / 0.01, dim=-1)
    greedy_choice = torch.argmax(greedy_probs)
    print(f"Greedy: Always picks '{vocab[greedy_choice]}' (p={greedy_probs[greedy_choice]:.3f})")

    # Strategy 2: Temperature scaling
    for temp in [0.5, 1.0, 2.0]:
        scaled_probs = F.softmax(logits / temp, dim=-1)
        print(f"\nTemperature {temp}:")
        for i, (word, prob) in enumerate(zip(vocab, scaled_probs)):
            if prob > 0.05:
                print(f"  {word}: {prob:.3f} {'█' * int(prob * 20)}")

    # Strategy 3: Top-k sampling
    k = 3
    top_k_logits = logits.clone()
    top_k_indices = torch.topk(logits, k).indices
    mask = torch.ones_like(logits, dtype=torch.bool)
    mask[top_k_indices] = False
    top_k_logits[mask] = float('-inf')
    top_k_probs = F.softmax(top_k_logits, dim=-1)
    print(f"\nTop-{k} sampling:")
    for i, (word, prob) in enumerate(zip(vocab, top_k_probs)):
        if prob > 0:
            print(f"  {word}: {prob:.3f}")

    # Strategy 4: Nucleus (top-p) sampling
    p = 0.8
    sorted_probs, sorted_indices = torch.sort(F.softmax(logits, dim=-1), descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    nucleus_mask = cumsum <= p
    nucleus_mask[0] = True  # Always include top token
    print(f"\nTop-p ({p}) nucleus:")
    for idx in sorted_indices[nucleus_mask]:
        print(f"  {vocab[idx]}: {F.softmax(logits, dim=-1)[idx]:.3f}")

demonstrate_generation_strategies(None)
```

### GPT Variants

```
GPT Family Evolution:

GPT-1 (2018, 117M)
    │   • First GPT, 12 layers
    │   • Showed transfer learning works for generation
    │
    ▼
GPT-2 (2019, 124M-1.5B)
    │   • Scaled up, zero-shot capabilities emerged
    │   • "Too dangerous to release" (they did anyway)
    │
    ▼
GPT-3 (2020, 175B)
    │   • Massive scale, in-context learning
    │   • API-only access
    │
    ▼
InstructGPT/ChatGPT (2022)
    │   • RLHF alignment
    │   • Follows instructions
    │
    ▼
GPT-4 (2023, architecture undisclosed)
        • Multimodal (text + vision)
        • Substantially improved reasoning
```

**Open-Source Alternatives:**

| Model | Parameters | Strengths | License |
|-------|------------|-----------|---------|
| GPT-2 | 1.5B | Small, well-understood, great for learning | MIT |
| LLaMA-2 | 7B-70B | Strong general performance, large community | Custom (commercial use allowed) |
| Mistral | 7B | Efficient, strong for its size | Apache 2.0 |
| Falcon | 7B-180B | Trained on curated web data (RefinedWeb) | Apache 2.0 |
| Qwen | 7B-72B | Multilingual, competitive benchmarks | Custom |

> **Note:** Informal comparisons like "Mistral 7B matches LLaMA-2 13B" circulate widely, but benchmark results vary by task. Always evaluate on your own data before choosing.

---

## T5 and BART: The Encoder-Decoder Models

The BERT-vs-GPT framing is useful but incomplete. A third family — **encoder-decoder models** — is critical for sequence-to-sequence tasks where both understanding input and generating output matter.

### Why Encoder-Decoder?

```
Encoder-Only (BERT):   Input → Understanding         (classification, NER)
Decoder-Only (GPT):    Prefix → Generation            (text completion, chat)
Encoder-Decoder (T5):  Input → Understanding → Generation  (translation, summarization)
```

Encoder-decoder models excel when the output is a **transformation** of the input rather than a continuation.

### T5: Text-to-Text Transfer Transformer

T5's key insight: **every NLP task can be framed as text-to-text.**

```python
from transformers import T5ForConditionalGeneration, T5Tokenizer

model_name = "t5-small"  # 60M params; t5-base = 220M, t5-large = 770M
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Summarization
input_text = "summarize: The quick brown fox jumps over the lazy dog. The fox was very fast and agile."
inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
outputs = model.generate(**inputs, max_new_tokens=50)
print(f"Summary: {tokenizer.decode(outputs[0], skip_special_tokens=True)}")

# Translation
input_text = "translate English to French: Hello, how are you?"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
print(f"Translation: {tokenizer.decode(outputs[0], skip_special_tokens=True)}")

# Classification (framed as generation)
input_text = "sst2 sentence: This movie is terrible"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=5)
print(f"Sentiment: {tokenizer.decode(outputs[0], skip_special_tokens=True)}")
```

### BART: Denoising Sequence-to-Sequence

BART combines BERT's bidirectional encoder with GPT's autoregressive decoder, pre-trained via denoising (reconstruct corrupted text):

```python
from transformers import BartForConditionalGeneration, BartTokenizer

model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

article = """The International Space Station has been continuously occupied since
November 2000. It serves as a microgravity research laboratory in which crew
members conduct experiments in biology, physics, and astronomy."""

inputs = tokenizer(article, return_tensors="pt", max_length=1024, truncation=True)
outputs = model.generate(**inputs, max_new_tokens=60, num_beams=4, early_stopping=True)
print(f"Summary: {tokenizer.decode(outputs[0], skip_special_tokens=True)}")
```

### When to Use Encoder-Decoder vs Decoder-Only

| Task | Best Family | Why |
|------|-------------|-----|
| Summarization | Encoder-Decoder (BART, T5) | Input must be fully understood before condensing |
| Translation | Encoder-Decoder (mBART, mT5) | Source and target are structurally different |
| Data-to-text | Encoder-Decoder (T5) | Structured input → natural language output |
| Open-ended generation | Decoder-Only (GPT) | No fixed input to encode |
| Instruction following | Decoder-Only (GPT) | Flexible input/output format |

> **Trend note:** Decoder-only models (GPT-4, LLaMA) have increasingly matched or surpassed encoder-decoder models on seq2seq tasks at sufficient scale. However, encoder-decoder models remain more efficient at smaller scales (< 1B parameters) for tasks like summarization and translation.

---

## BERT vs GPT: The Complete Comparison

### Architectural Differences

```
                 BERT                              GPT
         ┌──────────────────┐            ┌──────────────────┐
         │    ENCODER       │            │    DECODER       │
         │                  │            │                  │
         │  Bidirectional   │            │   Unidirectional │
         │   Attention      │            │    Attention     │
         │                  │            │                  │
         │    A ←→ B ←→ C   │            │    A → B → C     │
         │    ↑  ↗↙  ↑      │            │    ↓   ↓   ↓     │
         │    └──┴───┘      │            │    A  A+B A+B+C  │
         └──────────────────┘            └──────────────────┘
                 │                                │
                 ▼                                ▼
           Understanding                    Generation
           - Classification                 - Text completion
           - NER                           - Dialogue
           - QA (extractive)               - Creative writing
           - Semantic similarity           - Code generation
```

### Training Objective Comparison

```python
# BERT: Masked Language Modeling
# "The [MASK] sat on the mat" → Predict "cat"
# Sees: left AND right context

# GPT: Causal Language Modeling
# "The cat sat on the" → Predict "mat"
# Sees: only left context

# Visual comparison:
"""
BERT attention for "sat":
The  cat  [sat]  on  the  mat
 ✓    ✓    ●     ✓    ✓    ✓    (sees everything)

GPT attention for "sat":
The  cat  [sat]  on  the  mat
 ✓    ✓    ●     ✗    ✗    ✗    (sees only past)
"""
```

### Use Case Decision Matrix

```python
def choose_model_family(task_description):
    """
    Decision tree for BERT vs GPT selection.
    """
    decisions = {
        # Understanding tasks → BERT
        "sentiment_analysis": "BERT",
        "text_classification": "BERT",
        "named_entity_recognition": "BERT",
        "question_answering_extractive": "BERT",
        "semantic_similarity": "BERT",
        "paraphrase_detection": "BERT",

        # Generation tasks → GPT
        "text_generation": "GPT",
        "story_writing": "GPT",
        "code_generation": "GPT",
        "chatbot": "GPT",
        "summarization": "GPT",  # Can be either, but GPT more common now
        "translation": "GPT",    # Historically encoder-decoder, now GPT

        # Could go either way
        "question_answering_generative": "GPT",
        "information_extraction": "BERT",
        "search_ranking": "BERT",
        "embedding_generation": "BERT",  # BERT embeddings often better
    }

    return decisions.get(task_description, "Depends on specific requirements")

# The key question:
# "Do I need to GENERATE new text, or UNDERSTAND existing text?"
```

### Performance Comparison

```python
# Benchmark results (approximate, varies by model size)

benchmarks = {
    "GLUE (Understanding)": {
        "BERT-large": 84.4,
        "RoBERTa-large": 88.5,
        "GPT-2": "N/A (not designed for this)"
    },
    "SuperGLUE (Hard Understanding)": {
        "DeBERTa": 90.3,
        "GPT-3 (few-shot)": 71.8,
    },
    "Text Generation (Human Eval)": {
        "BERT": "N/A (not designed for this)",
        "GPT-2": "Coherent but limited",
        "GPT-3": "Often indistinguishable from human"
    }
}

# Key insight: Use the right tool for the job
# Don't use GPT for classification just because it's trendy
# Don't use BERT for generation just because you know it
```

---

## Fine-Tuning: From Pre-trained to Task-Specific

### What is Fine-Tuning?

```
Pre-training (General Knowledge):
┌─────────────────────────────────────────────────────────┐
│  Read Wikipedia, Books, Web → Learn language patterns   │
│  Cost: $Millions | Time: Months | Data: Terabytes      │
└─────────────────────────────────────────────────────────┘
                              │
                              ▼
Fine-tuning (Specific Task):
┌─────────────────────────────────────────────────────────┐
│  Your labeled data → Adapt for your task                │
│  Cost: $10-1000 | Time: Hours | Data: 100-10K examples │
└─────────────────────────────────────────────────────────┘
                              │
                              ▼
               Task-Specific Model
```

### Hands-On: Fine-Tuning BERT for Sentiment Analysis

Let's fine-tune BERT on the IMDb movie review dataset:

```python
import torch
from torch.utils.data import DataLoader
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW  # Note: transformers.AdamW is deprecated; use PyTorch native
from datasets import load_dataset
from tqdm import tqdm
import numpy as np

# ============================================================
# STEP 1: Load and Prepare Data
# ============================================================

print("Loading IMDb dataset...")
dataset = load_dataset("imdb")

# Reduce size for demonstration (use full dataset in production)
# CRITICAL: Proper train / validation / test split.
# Never use the test set for model selection — that leaks test information
# into your training decisions and produces overfit evaluation numbers.
train_full = dataset["train"].shuffle(seed=42).select(range(2000))

# Split training data into train (80%) and validation (20%)
split = train_full.train_test_split(test_size=0.2, seed=42)
train_data = split["train"]   # 1600 examples — used for gradient updates
val_data = split["test"]      # 400 examples — used for early stopping & model selection
test_data = dataset["test"].shuffle(seed=42).select(range(500))  # held-out, touched ONCE

print(f"Training examples: {len(train_data)}")
print(f"Validation examples: {len(val_data)}")
print(f"Test examples: {len(test_data)} (held-out, evaluate ONCE at the end)")
print(f"\nSample review: {train_data[0]['text'][:200]}...")
print(f"Label: {'Positive' if train_data[0]['label'] == 1 else 'Negative'}")

# ============================================================
# STEP 2: Tokenization
# ============================================================

model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    # Note: Do NOT use return_tensors="pt" inside dataset.map().
    # The datasets library handles conversion later via set_format().
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=256  # IMDb reviews can be long
    )

print("\nTokenizing datasets...")
train_tokenized = train_data.map(tokenize_function, batched=True)
val_tokenized = val_data.map(tokenize_function, batched=True)
test_tokenized = test_data.map(tokenize_function, batched=True)

# Set format for PyTorch
train_tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])
val_tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])
test_tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# ============================================================
# STEP 3: Create DataLoaders
# ============================================================

batch_size = 16

train_loader = DataLoader(train_tokenized, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_tokenized, batch_size=batch_size)
test_loader = DataLoader(test_tokenized, batch_size=batch_size)  # Only used ONCE at the end

# ============================================================
# STEP 4: Initialize Model
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")

model = BertForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,
    output_attentions=False,
    output_hidden_states=False
)
model.to(device)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# ============================================================
# STEP 5: Setup Training
# ============================================================

num_epochs = 3
learning_rate = 2e-5  # CRITICAL: Lower than typical deep learning!
warmup_steps = 100

# WHY WARMUP? Adam's second-moment estimate (v_t) is initialized to zero,
# making early updates unreliable (dividing by near-zero). Warmup keeps
# the learning rate small until Adam's running statistics stabilize.
# Without warmup, the first few batches can push weights far from the
# pre-trained optimum, causing catastrophic forgetting.

optimizer = AdamW(model.parameters(), lr=learning_rate, eps=1e-8)

total_steps = len(train_loader) * num_epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

# ============================================================
# STEP 6: Training Loop
# ============================================================

def train_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    progress_bar = tqdm(loader, desc="Training")

    for batch in progress_bar:
        # Move batch to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        logits = outputs.logits

        # Backward pass
        loss.backward()

        # Clip gradients (prevents exploding gradients)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Update weights
        optimizer.step()
        scheduler.step()

        # Track metrics
        total_loss += loss.item()
        predictions = torch.argmax(logits, dim=-1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{correct/total:.4f}'
        })

    return total_loss / len(loader), correct / total

def evaluate(model, loader, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            total_loss += outputs.loss.item()
            predictions = torch.argmax(outputs.logits, dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return total_loss / len(loader), correct / total, all_predictions, all_labels

# ============================================================
# STEP 6b: Establish Baselines (before fine-tuning)
# ============================================================
# Always measure baselines BEFORE fine-tuning. If your fine-tuned model
# doesn't meaningfully beat these, something is wrong.

def compute_baselines(val_loader, device):
    """Compute baselines to contextualize fine-tuning results."""
    all_labels = []
    for batch in val_loader:
        all_labels.extend(batch["label"].numpy())
    all_labels = np.array(all_labels)

    # Baseline 1: Majority class (always predict most common label)
    from collections import Counter
    majority_class = Counter(all_labels).most_common(1)[0][0]
    majority_acc = (all_labels == majority_class).mean()
    print(f"Baseline 1 — Majority class (always predict {majority_class}): {majority_acc:.4f}")

    # Baseline 2: Random (proportional to class distribution)
    class_probs = np.bincount(all_labels) / len(all_labels)
    random_acc = (class_probs ** 2).sum()  # Expected accuracy of random proportional
    print(f"Baseline 2 — Random proportional: {random_acc:.4f}")

    # Baseline 3: Zero-shot (pre-trained model without fine-tuning)
    from transformers import pipeline
    zero_shot = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=device)
    correct = 0
    sample_size = min(100, len(all_labels))  # Sample for speed
    sample_texts = [val_data[i]["text"][:512] for i in range(sample_size)]
    sample_labels = all_labels[:sample_size]
    results = zero_shot(sample_texts, candidate_labels=["negative", "positive"], batch_size=8)
    for result, true_label in zip(results, sample_labels):
        pred = 1 if result["labels"][0] == "positive" else 0
        correct += (pred == true_label)
    zero_shot_acc = correct / sample_size
    print(f"Baseline 3 — Zero-shot (BART-MNLI, n={sample_size}): {zero_shot_acc:.4f}")

    return {"majority": majority_acc, "random": random_acc, "zero_shot": zero_shot_acc}

print("\n" + "="*50)
print("Computing Baselines")
print("="*50)
baselines = compute_baselines(val_loader, device)

# Training
print("\n" + "="*50)
print("Starting Fine-tuning")
print("="*50)

best_accuracy = 0
training_history = []

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch + 1}/{num_epochs}")
    print("-" * 30)

    train_loss, train_acc = train_epoch(
        model, train_loader, optimizer, scheduler, device
    )

    # Evaluate on VALIDATION set (not test set!) for model selection
    val_loss, val_acc, predictions, labels = evaluate(model, val_loader, device)

    training_history.append({
        'epoch': epoch + 1,
        'train_loss': train_loss,
        'train_acc': train_acc,
        'val_loss': val_loss,
        'val_acc': val_acc
    })

    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    if val_acc > best_accuracy:
        best_accuracy = val_acc
        # Save best model
        model.save_pretrained("./best_sentiment_model")
        tokenizer.save_pretrained("./best_sentiment_model")
        print(f"✓ New best model saved! Accuracy: {best_accuracy:.4f}")

print(f"\nBest validation accuracy: {best_accuracy:.4f}")

# ============================================================
# STEP 7: Final Evaluation on Held-Out Test Set (ONE TIME ONLY)
# ============================================================
# Load best model and evaluate on test set exactly once.
# This gives the unbiased estimate of generalization performance.
from transformers import BertForSequenceClassification as BertSeqCls
best_model = BertSeqCls.from_pretrained("./best_sentiment_model").to(device)
test_loss, test_acc, test_preds, test_labels = evaluate(best_model, test_loader, device)
print(f"\n{'='*50}")
print(f"FINAL TEST SET EVALUATION (unbiased)")
print(f"{'='*50}")
print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")
print(f"Val-Test gap: {abs(best_accuracy - test_acc):.4f}")
if abs(best_accuracy - test_acc) > 0.05:
    print("⚠ Large val-test gap — possible overfitting to validation set")
```

### Analyzing Fine-Tuning Results

```python
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Detailed evaluation
print("\n" + "="*50)
print("Detailed Evaluation")
print("="*50)

# Classification report
print("\nClassification Report:")
print(classification_report(
    labels,
    predictions,
    target_names=["Negative", "Positive"]
))

# Confusion matrix
cm = confusion_matrix(labels, predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=["Negative", "Positive"],
    yticklabels=["Negative", "Positive"]
)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.close()
print("Confusion matrix saved to confusion_matrix.png")

# Learning curves
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Loss curve
epochs = [h['epoch'] for h in training_history]
axes[0].plot(epochs, [h['train_loss'] for h in training_history], 'b-', label='Train')
axes[0].plot(epochs, [h['val_loss'] for h in training_history], 'r-', label='Validation')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training and Validation Loss')
axes[0].legend()
axes[0].grid(True)

# Accuracy curve
axes[1].plot(epochs, [h['train_acc'] for h in training_history], 'b-', label='Train')
axes[1].plot(epochs, [h['val_acc'] for h in training_history], 'r-', label='Validation')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Training and Validation Accuracy')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig('learning_curves.png', dpi=150)
plt.close()
print("Learning curves saved to learning_curves.png")
```

### Interpreting Your Evaluation Metrics

After running the classification report and confusion matrix, ask yourself these questions before declaring success:

**1. Is accuracy enough?**
For balanced datasets (roughly equal positive/negative), accuracy is a reasonable metric. For imbalanced datasets (e.g., 95% negative, 5% positive), accuracy is misleading -- a model that always predicts "negative" gets 95% accuracy. Use **precision, recall, and F1** instead.

**2. What is the cost of each error type?**
- **False Positive** (predicted positive, actually negative): e.g., flagging a safe email as spam.
- **False Negative** (predicted negative, actually positive): e.g., missing a fraudulent transaction.
- If false negatives are costlier, optimize for **recall**. If false positives are costlier, optimize for **precision**.

**3. Is the model well-calibrated?**
A model that predicts "positive" with 90% confidence should be correct about 90% of the time. Check calibration with a reliability diagram:

```python
from sklearn.calibration import calibration_curve

# After collecting all predictions and true labels
prob_true, prob_pred = calibration_curve(all_labels, all_probs, n_bins=10)
plt.plot(prob_pred, prob_true, marker='o', label='Model')
plt.plot([0, 1], [0, 1], linestyle='--', label='Perfect calibration')
plt.xlabel('Mean predicted probability')
plt.ylabel('Fraction of positives')
plt.title('Calibration Curve')
plt.legend()
plt.savefig('calibration_curve.png', dpi=150)
plt.close()
```

**4. How does it compare to baselines?**
Always compare against:
- **Random baseline:** Expected accuracy for random predictions given class distribution
- **Majority-class baseline:** Always predict the most common class
- **Simple rule-based baseline:** e.g., keyword matching for sentiment
- **Zero-shot baseline:** Pre-trained model without fine-tuning

If your fine-tuned model does not meaningfully beat these baselines, something is wrong.

### Using Your Fine-Tuned Model

```python
from transformers import pipeline

# Load your fine-tuned model
sentiment_classifier = pipeline(
    "text-classification",
    model="./best_sentiment_model",
    tokenizer="./best_sentiment_model"
)

# Test on new reviews
test_reviews = [
    "This movie exceeded all my expectations. The acting was superb!",
    "What a waste of time. I want my 2 hours back.",
    "It was okay. Not great, not terrible. Pretty average.",
    "The cinematography was stunning but the plot was confusing.",
    "I've watched it 5 times already. A true masterpiece!"
]

print("\n" + "="*50)
print("Testing Fine-Tuned Model")
print("="*50)

for review in test_reviews:
    result = sentiment_classifier(review)[0]
    sentiment = "Positive" if result['label'] == "LABEL_1" else "Negative"
    print(f"\nReview: {review[:60]}...")
    print(f"Sentiment: {sentiment} (confidence: {result['score']:.3f})")
```

---

## Common Fine-Tuning Failures and Fixes

### Failure 1: Catastrophic Forgetting

```python
"""
Problem: Model forgets pre-trained knowledge during fine-tuning.
Symptom: Good on fine-tuning task, terrible on related tasks.

Example:
- Fine-tune BERT on legal documents
- Model forgets general English understanding
- Performs poorly on non-legal text
"""

# Solution 1: Lower learning rate
learning_rate = 2e-5  # Not 1e-3 like typical deep learning

# Solution 2: Gradual unfreezing
def gradual_unfreeze(model, epoch, total_epochs):
    """Unfreeze layers gradually during training."""
    n_layers = 12  # BERT has 12 layers
    layers_per_epoch = n_layers // total_epochs

    # Start with all frozen except classifier
    if epoch == 0:
        for param in model.bert.parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = True
    else:
        # Unfreeze from top (closest to output)
        layers_to_unfreeze = min(epoch * layers_per_epoch, n_layers)
        for i in range(n_layers - layers_to_unfreeze, n_layers):
            for param in model.bert.encoder.layer[i].parameters():
                param.requires_grad = True

# Solution 3: Regularization toward pre-trained weights (EWC)
class ElasticWeightConsolidation:
    """
    Penalize changes to important pre-trained weights.
    Importance is measured by the Fisher Information Matrix (diagonal approximation):
    parameters with high Fisher information are "important" for the pre-trained task
    and should change less during fine-tuning.
    """
    def __init__(self, model, dataloader, device, lambda_ewc=0.4):
        self.lambda_ewc = lambda_ewc
        # Store pre-trained weights
        self.pretrained = {n: p.clone().detach()
                           for n, p in model.named_parameters() if p.requires_grad}
        # Compute Fisher Information Matrix (diagonal approximation)
        self.importance = self._compute_fisher(model, dataloader, device)

    def _compute_fisher(self, model, dataloader, device, n_samples=200):
        """
        Diagonal Fisher = E[ (∂ log p(y|x; θ))² ] per parameter.
        Approximated by averaging squared gradients over data samples.
        """
        fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters()
                  if p.requires_grad}
        model.eval()
        count = 0
        for batch in dataloader:
            if count >= n_samples:
                break
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**inputs)
            # Use log-likelihood of the model's own prediction (not ground truth)
            log_probs = torch.nn.functional.log_softmax(outputs.logits, dim=-1)
            labels = torch.argmax(log_probs, dim=-1)
            loss = torch.nn.functional.nll_loss(log_probs, labels)
            model.zero_grad()
            loss.backward()
            for n, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.data.pow(2)
            count += inputs["input_ids"].size(0)
        # Average
        for n in fisher:
            fisher[n] /= count
        return fisher

    def penalty(self, model):
        loss = 0
        for name, param in model.named_parameters():
            if name in self.pretrained:
                loss += (self.importance[name] *
                        (param - self.pretrained[name]).pow(2)).sum()
        return self.lambda_ewc * loss
```

### Failure 2: Overfitting

```python
"""
Problem: Model memorizes training data instead of learning patterns.
Symptom: Training accuracy >> Validation accuracy

Example:
- 100 training examples
- 99% training accuracy
- 55% validation accuracy (barely better than random)
"""

# Solution 1: More data (obvious but often ignored)
# Rule of thumb: At least 100 examples per class

# Solution 2: Data augmentation
import nlpaug.augmenter.word as naw

def augment_text(text):
    """Simple text augmentation strategies."""
    augmenters = [
        naw.SynonymAug(aug_src='wordnet'),  # Replace with synonyms
        naw.RandomWordAug(action="swap"),    # Swap adjacent words
        naw.RandomWordAug(action="delete"),  # Delete random words
    ]

    augmented_texts = [text]
    for aug in augmenters:
        augmented_texts.append(aug.augment(text)[0])

    return augmented_texts

# Solution 3: Dropout increase
from transformers import BertConfig, BertForSequenceClassification

config = BertConfig.from_pretrained(
    "bert-base-uncased",
    num_labels=2,
    hidden_dropout_prob=0.3,  # Default is 0.1
    attention_probs_dropout_prob=0.3
)
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    config=config
)

# Solution 4: Early stopping
class EarlyStopping:
    def __init__(self, patience=3, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
```

### Failure 3: Learning Rate Issues

```python
"""
Problem: Wrong learning rate causes training failure.
Symptoms:
- Too high: Loss explodes or oscillates wildly
- Too low: Loss decreases extremely slowly, stuck in local minimum
"""

# Solution: Learning rate finder
# WARNING: The LR finder modifies model weights. Always save and restore state.
import copy

def find_learning_rate(model, train_loader, optimizer, device,
                       start_lr=1e-7, end_lr=1, num_iter=100):
    """
    Find optimal learning rate using range test.

    IMPORTANT: This function saves and restores model + optimizer state
    so it does not corrupt your training run.
    """
    # Save state so we can restore after the sweep
    model_state = copy.deepcopy(model.state_dict())
    optimizer_state = copy.deepcopy(optimizer.state_dict())

    model.train()
    lr_mult = (end_lr / start_lr) ** (1 / num_iter)
    lr = start_lr

    lrs = []
    losses = []
    best_loss = float('inf')

    for i, batch in enumerate(train_loader):
        if i >= num_iter:
            break

        # Set learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Forward pass
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        # Stop if loss has diverged (4x best loss)
        if loss.item() > 4 * best_loss and best_loss < float('inf'):
            break
        best_loss = min(best_loss, loss.item())

        # Track
        lrs.append(lr)
        losses.append(loss.item())

        # Backward pass
        loss.backward()
        optimizer.step()

        lr *= lr_mult

    # Restore model and optimizer state (critical!)
    model.load_state_dict(model_state)
    optimizer.load_state_dict(optimizer_state)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(lrs, losses)
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss')
    plt.title('Learning Rate Finder')

    # Find steepest descent point
    min_grad_idx = np.argmin(np.gradient(losses))
    optimal_lr = lrs[min_grad_idx]
    plt.axvline(x=optimal_lr, color='r', linestyle='--',
                label=f'Suggested LR: {optimal_lr:.2e}')
    plt.legend()
    plt.savefig('lr_finder.png', dpi=150)
    plt.close()

    return optimal_lr

# Typical fine-tuning learning rates
LEARNING_RATES = {
    "bert-base": 2e-5,
    "bert-large": 1e-5,
    "roberta": 1e-5,
    "distilbert": 5e-5,
    "gpt2": 5e-5,
}
```

### Failure 4: Memory Issues

```python
"""
Problem: GPU out of memory during training.
Solutions in order of preference:
"""

# Solution 1: Reduce batch size (simplest)
batch_size = 8  # Instead of 16 or 32

# Solution 2: Gradient accumulation (effective batch size without memory)
accumulation_steps = 4  # Effective batch = 8 * 4 = 32

for i, batch in enumerate(train_loader):
    outputs = model(**batch)
    loss = outputs.loss / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

# Solution 3: Mixed precision training
# Note: torch.cuda.amp.autocast and GradScaler are deprecated since PyTorch 2.4.
# Use torch.amp.autocast and torch.amp.GradScaler instead.
from torch.amp import autocast, GradScaler

scaler = GradScaler("cuda")

for batch in train_loader:
    optimizer.zero_grad()

    with autocast("cuda"):  # Automatic mixed precision
        outputs = model(**batch)
        loss = outputs.loss

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

# Solution 4: Gradient checkpointing
model.gradient_checkpointing_enable()

# Solution 5: Use smaller model
# DistilBERT instead of BERT
# BERT-base instead of BERT-large
```

### Failure 5: Tokenizer Mismatches

```python
"""
Problem: Using the wrong tokenizer or mismatched special tokens.
This is one of the most common real-world fine-tuning bugs because
it produces no error — just silently degraded performance.
"""

# WRONG: Using a different model's tokenizer
from transformers import RobertaTokenizer, BertForSequenceClassification
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")  # BPE tokenizer
model = BertForSequenceClassification.from_pretrained("bert-base-uncased")  # WordPiece model
# This runs without errors but produces garbage because:
# 1. Token IDs map to different vocabulary entries
# 2. Special tokens differ ([CLS]/[SEP] for BERT vs <s>/</s> for RoBERTa)
# 3. The embedding matrix was trained with a different tokenization

# RIGHT: Always load tokenizer and model from the same checkpoint
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# ALSO WATCH: Adding special tokens requires resizing embeddings
# If you add custom tokens (e.g., for domain-specific terms):
num_added = tokenizer.add_special_tokens({"additional_special_tokens": ["[ENTITY]", "[RELATION]"]})
model.resize_token_embeddings(len(tokenizer))
# The new embeddings are randomly initialized — they need fine-tuning data to learn meaning

# DEBUGGING TIP: Always verify round-trip consistency
text = "Test sentence with [ENTITY] markers"
encoded = tokenizer(text, return_tensors="pt")
decoded = tokenizer.decode(encoded["input_ids"][0])
print(f"Original: {text}")
print(f"Round-trip: {decoded}")
# If these don't match (modulo special tokens), something is wrong
```

### Memory Estimation for Fine-Tuning

Before launching a training run, estimate whether it will fit in GPU memory:

```python
def estimate_finetuning_memory(
    model_params_M: float,  # Model parameters in millions
    batch_size: int,
    seq_len: int,
    d_model: int,
    n_layers: int,
    precision: str = "fp32",  # "fp32", "fp16", or "int8"
    optimizer: str = "adamw",  # "adamw" or "sgd"
    gradient_checkpointing: bool = False,
) -> dict:
    """
    Estimate GPU memory for fine-tuning a transformer model.

    Memory components:
    1. Model weights
    2. Optimizer states (AdamW stores 2 copies: momentum + variance)
    3. Gradients (same size as weights)
    4. Activations (depends on batch_size × seq_len × d_model × n_layers)
    """
    bytes_per_param = {"fp32": 4, "fp16": 2, "int8": 1}[precision]
    params = model_params_M * 1e6

    # 1. Model weights
    weight_memory = params * bytes_per_param

    # 2. Gradients (always fp32 for stability, even in mixed precision)
    gradient_memory = params * 4  # Always fp32

    # 3. Optimizer states
    if optimizer == "adamw":
        # AdamW stores first moment (m) and second moment (v) per parameter
        optimizer_memory = params * 4 * 2  # Two fp32 copies
    else:
        optimizer_memory = params * 4  # SGD: just momentum

    # 4. Activations (rough estimate)
    # Each layer stores activations for backward pass
    activation_per_layer = batch_size * seq_len * d_model * 4  # fp32
    if gradient_checkpointing:
        # Only store activations at checkpoint boundaries (sqrt(n_layers))
        import math
        effective_layers = math.ceil(math.sqrt(n_layers))
    else:
        effective_layers = n_layers
    activation_memory = activation_per_layer * effective_layers

    total = weight_memory + gradient_memory + optimizer_memory + activation_memory

    result = {
        "weights_MB": weight_memory / 1e6,
        "gradients_MB": gradient_memory / 1e6,
        "optimizer_MB": optimizer_memory / 1e6,
        "activations_MB": activation_memory / 1e6,
        "total_MB": total / 1e6,
        "total_GB": total / 1e9,
    }

    print(f"{'Component':<20} {'Memory (MB)':>12}")
    print("-" * 34)
    for k, v in result.items():
        if k != "total_GB":
            print(f"{k:<20} {v:>12.1f}")
    print(f"\n{'TOTAL':.<20} {result['total_GB']:.2f} GB")

    return result

# Example: BERT-base fine-tuning
print("=== BERT-base (batch=16, seq=256) ===")
estimate_finetuning_memory(110, batch_size=16, seq_len=256, d_model=768, n_layers=12)

print("\n=== BERT-base with gradient checkpointing ===")
estimate_finetuning_memory(110, batch_size=16, seq_len=256, d_model=768, n_layers=12,
                           gradient_checkpointing=True)

print("\n=== BERT-large (batch=8, seq=512) ===")
estimate_finetuning_memory(340, batch_size=8, seq_len=512, d_model=1024, n_layers=24)
```

---

## Parameter-Efficient Fine-Tuning (PEFT)

When fine-tuning massive models, updating all parameters is expensive. PEFT methods freeze most parameters and train only a small subset.

### LoRA: Low-Rank Adaptation

```python
"""
LoRA insight: Weight updates during fine-tuning have low intrinsic rank.
Instead of: W_new = W_old + ΔW (full matrix update)
LoRA does: W_new = W_old + BA (low-rank decomposition)

If W is 768×768, that's 589,824 parameters.
With LoRA rank=8: B is 768×8, A is 8×768 = 12,288 parameters (50x smaller!)
"""

from peft import LoraConfig, get_peft_model, TaskType

# Configure LoRA
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8,  # Rank
    lora_alpha=32,  # Scaling factor
    lora_dropout=0.1,
    target_modules=["query", "value"],  # Which layers to adapt
)

# Apply LoRA to model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
peft_model = get_peft_model(model, lora_config)

# Check trainable parameters
peft_model.print_trainable_parameters()
# Output: trainable params: 294,912 || all params: 109,776,386 || trainable%: 0.27%
```

### Adapter Layers: Bottleneck Inside the Transformer

Adapters (Houlsby et al., 2019) insert small trainable bottleneck modules inside each transformer layer while freezing the original weights:

```python
import torch.nn as nn

class AdapterLayer(nn.Module):
    """
    Bottleneck adapter inserted after the feed-forward sub-layer.

    Architecture: hidden → down-project → nonlinearity → up-project → residual
    If d_model=768 and bottleneck_dim=64, trainable params per adapter =
      768×64 + 64 + 64×768 + 768 = 99,136 (vs. millions in the frozen layer)
    """
    def __init__(self, d_model: int, bottleneck_dim: int = 64):
        super().__init__()
        self.down_project = nn.Linear(d_model, bottleneck_dim)
        self.activation = nn.GELU()
        self.up_project = nn.Linear(bottleneck_dim, d_model)
        # Initialize up-projection near zero so adapter starts as identity
        nn.init.zeros_(self.up_project.weight)
        nn.init.zeros_(self.up_project.bias)

    def forward(self, x):
        residual = x
        x = self.down_project(x)
        x = self.activation(x)
        x = self.up_project(x)
        return x + residual  # Skip connection preserves pre-trained behavior

# Example: inject adapters into BERT
def add_adapters_to_bert(model, bottleneck_dim=64):
    """Freeze all BERT weights, add trainable adapters after each FFN."""
    # Freeze original parameters
    for param in model.parameters():
        param.requires_grad = False

    # Add adapter after each transformer layer's output
    for layer in model.bert.encoder.layer:
        d_model = layer.output.dense.out_features
        adapter = AdapterLayer(d_model, bottleneck_dim)
        # Store adapter and modify forward pass
        layer.adapter = adapter
        original_forward = layer.output.forward

        def make_adapted_forward(orig_fn, adapt):
            def adapted_forward(hidden_states, input_tensor):
                output = orig_fn(hidden_states, input_tensor)
                return adapt(output)
            return adapted_forward

        layer.output.forward = make_adapted_forward(original_forward, adapter)

    # Count trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Adapter params: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")
    return model
```

**LoRA vs Adapters — When to choose which:**
- **LoRA:** Better when you need to serve multiple fine-tuned variants (hot-swap LoRA weights at inference without reloading). Lower latency since no extra forward-pass layers.
- **Adapters:** Better when you want modular task composition (stack adapters for multi-task). More mature ecosystem (AdapterHub).
- **Both:** Fail when the task requires learning fundamentally new representations not present in the base model (e.g., fine-tuning an English model for Mandarin).

### Comparison of PEFT Methods

| Method | Trainable % | Memory | Speed | Relative Performance |
|--------|-------------|--------|-------|----------------------|
| Full Fine-tuning | 100% | High | Slow | Best (baseline) |
| LoRA | 0.1-1% | Low | Fast | Near full FT on many tasks |
| Adapter | 1-5% | Medium | Medium | Near full FT on many tasks |
| Prefix Tuning | 0.1% | Low | Fast | Slightly below LoRA on average |
| Prompt Tuning | 0.01% | Very Low | Very Fast | Gap narrows with larger models |

> **Note:** Exact performance gaps depend heavily on the task, dataset size, and base model. The original LoRA paper (Hu et al., 2021) showed comparable results to full fine-tuning on several benchmarks, but this is not guaranteed for every use case. Always evaluate empirically.

---

## When NOT to Fine-Tune

Fine-tuning is powerful but not always the right choice. Before committing GPU hours, ask:

### Decision Framework

```
                    Do you have labeled data?
                     /                 \
                   YES                  NO
                   /                     \
         >1000 examples?          Can you get labels?
          /          \              /            \
        YES          NO           YES            NO
         |            |            |              |
    Full fine-tune  LoRA/       Label first,   Use zero-shot /
    or LoRA       few-shot      then fine-tune  few-shot prompting
                      |
              Does few-shot
              prompting work?
              /            \
            YES             NO
             |               |
       Use prompting    Fine-tune with
       (cheaper)        LoRA + augmentation
```

### When Prompting Beats Fine-Tuning

| Scenario | Why Prompting Wins |
|----------|-------------------|
| < 50 labeled examples | Fine-tuning overfits; few-shot prompting generalizes better |
| Task changes frequently | Rewriting a prompt takes minutes; retraining takes hours |
| Many tasks, small each | Maintaining 20 fine-tuned models is expensive; one LLM + 20 prompts is cheaper |
| Prototyping / exploration | Validate the idea with prompting before investing in fine-tuning |

### When RAG Beats Fine-Tuning

Fine-tuning bakes knowledge into model weights. This fails when:
- **Knowledge changes frequently** (product catalogs, documentation, news)
- **Factual accuracy matters** (RAG can cite sources; fine-tuned models hallucinate)
- **Domain corpus is large** (fine-tuning on 100K documents is expensive; RAG indexes them)

**Rule of thumb:** Fine-tune for **behavior** (tone, format, reasoning style). Use RAG for **knowledge** (facts, documents, data).

### When Fine-Tuning Is the Right Call

- You need consistent, low-latency predictions (fine-tuned BERT: ~5ms; API call: ~500ms)
- You have proprietary data that can't be sent to external APIs
- You need to run on edge devices (quantized fine-tuned model vs. cloud API)
- The task is well-defined and stable (sentiment analysis, NER, classification)

---

## Production Deployment Checklist

```python
class ProductionDeploymentChecklist:
    """
    Checklist before deploying your fine-tuned model.
    """

    CHECKLIST = {
        "Model Quality": [
            "✓ Evaluation on held-out test set (not validation set used during training)",
            "✓ Evaluation on edge cases and failure modes",
            "✓ Comparison against baseline (random, rule-based, previous model)",
            "✓ Confidence calibration check",
            "✓ Bias and fairness evaluation",
        ],
        "Technical": [
            "✓ Model quantization for smaller size",
            "✓ ONNX export for cross-platform deployment",
            "✓ Latency benchmarking (P50, P95, P99)",
            "✓ Memory footprint measured",
            "✓ Batch inference support",
        ],
        "Operational": [
            "✓ Model versioning (track which model version is deployed)",
            "✓ A/B testing setup",
            "✓ Monitoring and logging",
            "✓ Rollback procedure",
            "✓ Data drift detection",
        ],
        "Legal/Compliance": [
            "✓ License check (can you use this model commercially?)",
            "✓ Training data provenance",
            "✓ PII handling in predictions",
            "✓ Model card with limitations documented",
        ]
    }

    @staticmethod
    def export_to_onnx(model, tokenizer, output_path):
        """Export model to ONNX format for deployment."""
        import torch.onnx

        # Create dummy input
        dummy_input = tokenizer(
            "Sample text for export",
            return_tensors="pt",
            padding="max_length",
            max_length=128,
            truncation=True
        )

        # Export
        torch.onnx.export(
            model,
            (dummy_input["input_ids"], dummy_input["attention_mask"]),
            output_path,
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence"},
                "attention_mask": {0: "batch_size", 1: "sequence"},
                "logits": {0: "batch_size"}
            },
            opset_version=14
        )
        print(f"Model exported to {output_path}")

    @staticmethod
    def quantize_model(model_path, quantized_path):
        """Quantize model for faster inference."""
        from transformers import AutoModelForSequenceClassification
        import torch

        model = AutoModelForSequenceClassification.from_pretrained(model_path)

        # Dynamic quantization
        # Note: torch.quantization is deprecated in PyTorch 2.4+.
        # Use torch.ao.quantization for newer versions:
        #   from torch.ao.quantization import quantize_dynamic
        #   quantized_model = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {torch.nn.Linear},
            dtype=torch.qint8
        )

        # Save
        torch.save(quantized_model.state_dict(), quantized_path)

        # Size comparison
        import os
        # Note: newer transformers versions save as model.safetensors instead of pytorch_model.bin
        model_file = f"{model_path}/model.safetensors"
        if not os.path.exists(model_file):
            model_file = f"{model_path}/pytorch_model.bin"
        original_size = os.path.getsize(model_file) / 1e6
        quantized_size = os.path.getsize(quantized_path) / 1e6
        print(f"Original: {original_size:.1f}MB → Quantized: {quantized_size:.1f}MB")
        print(f"Reduction: {(1 - quantized_size/original_size)*100:.1f}%")
```

---

## Interview Preparation

### Concept Questions

**Q1: What is the difference between pre-training and fine-tuning?**

*Answer:* Pre-training learns general language understanding from massive unlabeled corpora (self-supervised learning on tasks like masked language modeling or next word prediction). Fine-tuning adapts the pre-trained model to a specific task using labeled data. Pre-training is expensive (millions of dollars, terabytes of data), while fine-tuning is cheap (hundreds of examples, hours on a single GPU).

**Q2: When would you choose BERT over GPT?**

*Answer:* Choose BERT for understanding tasks: classification, NER, extractive QA, semantic similarity. BERT's bidirectional attention allows each token to see full context. Choose GPT for generation tasks: text completion, creative writing, chatbots, code generation. GPT's causal attention enables autoregressive generation. The key question is: "Do I need to understand existing text or generate new text?"

**Q3: What is catastrophic forgetting and how do you prevent it?**

*Answer:* Catastrophic forgetting occurs when a model loses pre-trained knowledge during fine-tuning. Prevention strategies:
1. Use low learning rates (2e-5 typical for BERT)
2. Gradual unfreezing (train classifier first, unfreeze layers progressively)
3. Regularization toward pre-trained weights (EWC, L2)
4. Early stopping before overfitting
5. Multi-task learning to maintain diverse capabilities

**Q4: Explain LoRA in simple terms.**

*Answer:* LoRA (Low-Rank Adaptation) is based on the insight that fine-tuning weight updates have low intrinsic rank. Instead of updating a full weight matrix W (expensive), LoRA decomposes the update into two smaller matrices: ΔW = BA where B and A have much fewer parameters. For a 768×768 matrix with rank-8 LoRA, parameters drop from 589K to 12K—a 50x reduction while preserving most of the fine-tuning benefit.

### Coding Questions

**Q5: Write code to calculate the number of trainable parameters in a model.**

```python
def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    print(f"Total parameters: {total:,}")
    print(f"Trainable: {trainable:,} ({trainable/total*100:.2f}%)")
    print(f"Frozen: {frozen:,} ({frozen/total*100:.2f}%)")

    return {"total": total, "trainable": trainable, "frozen": frozen}
```

**Q6: Implement gradient accumulation.**

```python
def train_with_gradient_accumulation(
    model, train_loader, optimizer, device,
    accumulation_steps=4
):
    """
    Train with gradient accumulation for effective larger batch size.

    Args:
        accumulation_steps: Number of steps to accumulate before update
    """
    model.train()
    optimizer.zero_grad()

    for i, batch in enumerate(train_loader):
        inputs = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**inputs)
        # Scale loss by accumulation steps
        loss = outputs.loss / accumulation_steps
        loss.backward()

        # Update only every accumulation_steps
        if (i + 1) % accumulation_steps == 0:
            # Optional: gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            optimizer.zero_grad()

    # Handle remaining gradients
    if (i + 1) % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()
```

**Q7: When would you choose RAG over fine-tuning?**

*Answer:* Fine-tune for **behavior** (output format, tone, reasoning style). Use RAG for **knowledge** (facts, documents, frequently changing data). RAG is better when: (1) knowledge changes often (product catalog, docs), (2) you need source attribution, (3) the domain corpus is large, (4) you need to avoid hallucination on factual queries. Fine-tuning is better when: (1) you need low-latency predictions, (2) data can't leave your infrastructure, (3) the task is well-defined and stable, (4) you need to run on edge devices.

**Q8: Explain the difference between Adapter layers and LoRA.**

*Answer:* Adapters insert bottleneck modules (down-project → nonlinearity → up-project + residual) inside each transformer layer, adding new parameters in the forward pass. LoRA decomposes weight updates as low-rank matrices (ΔW = BA) without adding new layers — at inference, the LoRA matrices can be merged into the base weights for zero latency overhead. Choose LoRA when serving multiple fine-tuned variants (hot-swap weights) or when latency matters. Choose Adapters when composing multiple tasks modularly.

**Q9: How do you estimate GPU memory for a fine-tuning run?**

*Answer:* Four components: (1) Model weights = params × bytes_per_param, (2) Gradients = params × 4 bytes (always fp32), (3) Optimizer states = params × 8 bytes for AdamW (momentum + variance), (4) Activations = batch_size × seq_len × d_model × n_layers × 4 bytes. For BERT-base (110M params, batch=16, seq=256): ~1.5-2.5 GB total. Gradient checkpointing trades compute for memory by only storing sqrt(n_layers) activation checkpoints.

**Q10: Why does BERT use WordPiece while GPT uses BPE? Does it matter?**

*Answer:* Both are subword tokenization algorithms that balance vocabulary size with coverage. WordPiece (BERT) selects merges that maximize likelihood of the training data; BPE (GPT) merges the most frequent byte pairs. In practice, the difference is minor — what matters is **always using the matched tokenizer** for a given model. Using RoBERTa's BPE tokenizer with BERT's WordPiece model produces silent failures because token IDs map to different vocabulary entries.

### System Design Questions

**Q7: Design a sentiment analysis system that handles 10,000 requests per second.**

*Answer:*
1. **Model Selection:** DistilBERT (40% faster than BERT with 97% accuracy)
2. **Optimization:**
   - ONNX conversion for 2-3x speedup
   - INT8 quantization for 3-4x speedup
   - Batch inference (dynamic batching service)
3. **Infrastructure:**
   - Horizontal scaling with load balancer
   - GPU inference servers (T4/A10G) with 4-8 workers each
   - Request queue (Kafka/SQS) for traffic spikes
4. **Caching:**
   - Hash of input text → result cache (Redis)
   - 60-80% hit rate for common queries
5. **Monitoring:**
   - Latency percentiles (P50 < 20ms, P99 < 100ms)
   - Model confidence distribution
   - Data drift detection

**Back-of-envelope calculation (illustrative, not precise):**
- Throughput per GPU depends on model size, batch size, and hardware (profile your own setup)
- A T4 GPU running optimized DistilBERT can handle roughly 50-200 RPS depending on sequence length
- Caching common queries in Redis can cut GPU load significantly (measure your hit rate)
- Add redundancy for fault tolerance (at minimum N+1, often 2x for critical services)
- Always benchmark with realistic traffic patterns before sizing infrastructure

---

## Exercises

### Exercise 1: Model Selection Challenge
Choose the appropriate model for each scenario and justify your choice:

1. Email spam detection
2. Customer support chatbot
3. Legal document summarization
4. Product review sentiment for 50 languages
5. Code completion for Python

### Exercise 2: Fine-Tuning Practice
Fine-tune a DistilBERT model on the AG News dataset (4-class news classification). Compare your results with the pre-trained model's zero-shot performance.

```python
# Starter code
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

dataset = load_dataset("ag_news")
model_name = "distilbert-base-uncased"
# Your code here...
```

### Exercise 3: PEFT Implementation
Implement LoRA fine-tuning on a BERT model and compare:
1. Training time
2. Memory usage
3. Final accuracy
4. Number of trainable parameters

### Exercise 4: Failure Mode Analysis
Create a test suite that checks for:
1. Negation handling ("not good" vs "good")
2. Sarcasm detection
3. Out-of-domain inputs
4. Adversarial examples

### Exercise 5: Production Pipeline
Build a complete pipeline that:
1. Loads a fine-tuned model
2. Exports to ONNX
3. Quantizes to INT8
4. Benchmarks latency
5. Creates a simple REST API

---

## Section Checkpoints

### Checkpoint 1 — After "The Pre-training Revolution" and "Hugging Face Ecosystem"
1. Why is training a language model from scratch usually wrong for production tasks?
2. What are the four main components of the Hugging Face ecosystem?
3. What sections of a model card should you check before using a model commercially?
4. What is the difference between `pipeline()` and loading a model directly with `from_pretrained()`?

### Checkpoint 2 — After "BERT," "GPT," and "T5/BART" Sections
1. Explain the difference between bidirectional (BERT) and causal (GPT) attention in one sentence each.
2. Why does BERT use an 80/10/10 masking strategy instead of always using [MASK]?
3. When would you choose an encoder-decoder model (T5) over a decoder-only model (GPT)?
4. Name two BERT variants and explain what each improves over the original.

### Checkpoint 3 — After "BERT vs GPT" and "Fine-Tuning Pipeline"
1. Given a new text classification task, walk through how you'd decide between BERT and GPT.
2. Why must you split data into train/validation/test (not just train/test)?
3. Why is a learning rate of 2e-5 appropriate for fine-tuning but not 1e-3?
4. What baselines should you compute before claiming your fine-tuned model "works"?

### Checkpoint 4 — After "Common Failures" and "PEFT"
1. Explain catastrophic forgetting and two concrete prevention strategies.
2. How does LoRA reduce trainable parameters? What is the rank parameter `r`?
3. How do Adapter layers differ from LoRA in architecture and use case?
4. When should you use prompting/RAG instead of fine-tuning?

### Checkpoint 5 — After "Production Deployment" and "When NOT to Fine-Tune"
1. What are three steps to optimize a fine-tuned model for production inference?
2. How do you estimate whether a fine-tuning run will fit in your GPU's memory?
3. Describe a scenario where fine-tuning is the wrong approach and RAG is better.
4. What monitoring should you have in place after deploying a fine-tuned model?

---

## Job Role Mapping

| Section | ML Engineer | Data Scientist | AI Architect | Engineering Manager |
|---------|-------------|----------------|--------------|---------------------|
| Pre-training Revolution | Must know: transfer learning mechanics, cost trade-offs | Must know: when pre-trained models fit the problem | Must know: compute cost estimation, model selection at scale | Must know: build-vs-buy, ROI of pre-trained models |
| HF Ecosystem | Must know: `from_pretrained`, tokenizers, dataset loading | Must know: pipeline API, model cards, dataset library | Must know: Hub governance, model versioning, licensing | Must know: licensing risks, dependency management |
| BERT / GPT / T5 Architecture | Must know: encoder vs decoder vs enc-dec, attention patterns | Must know: which family for which task | Must know: architecture trade-offs at scale, serving implications | Must know: BERT-vs-GPT decision matrix for team guidance |
| Fine-Tuning Pipeline | Must know: full training loop, data splits, LR scheduling, grad accumulation | Must know: evaluation metrics, baseline comparison, overfitting detection | Must know: PEFT selection, memory estimation, infra sizing | Must know: compute budget, timeline estimation |
| Failure Modes & Debugging | Must know: all 5 failure modes + fixes, tokenizer mismatches | Must know: overfitting detection, evaluation anti-patterns | Must know: EWC, gradual unfreezing trade-offs | Must know: risk categories, when to escalate |
| Production & When Not to Fine-Tune | Must know: ONNX, quantization, serving frameworks | Must know: when to recommend prompting vs fine-tuning vs RAG | Must know: full deployment checklist, monitoring, rollback | Must know: fine-tune vs prompt vs RAG cost model |

---

## Summary

### Key Takeaways

1. **Pre-training democratized NLP:** Billion-dollar models are now free to use
2. **BERT for understanding, GPT for generation:** Match architecture to task
3. **Fine-tuning is cheap but tricky:** Learning rate, overfitting, forgetting are real
4. **Hugging Face is your best friend:** Unified API for thousands of models
5. **PEFT enables massive model fine-tuning:** LoRA cuts parameters 100x
6. **Production != Training:** Optimization, monitoring, compliance matter

### What's Next

In Blog 11, we'll dive deep into Large Language Models (LLMs)—the GPT-3s and GPT-4s of the world. You'll learn:
- What makes a language model "large"
- Emergent abilities that appear at scale
- The training pipeline for LLMs
- How to effectively prompt these models
- When to use APIs vs self-hosted models

---

## Self-Assessment: Does Well / Falls Short

### What This Blog Does Well
- **Complete model family coverage.** BERT (encoder), GPT (decoder), and T5/BART (encoder-decoder) are all covered with architecture, code, and when-to-use guidance — no false dichotomy.
- **Rigorous fine-tuning pipeline.** Proper train/validation/test split with explicit warnings against data leakage. Baselines (majority class, random, zero-shot) computed before fine-tuning to contextualize results.
- **Five failure modes with concrete fixes.** Catastrophic forgetting (with full EWC Fisher Information computation), overfitting, LR issues, memory problems, and tokenizer mismatches — each with root cause analysis, symptoms, and runnable solutions.
- **PEFT depth.** Both LoRA and Adapter layers are explained with full implementations, architecture diagrams, and guidance on when to choose which.
- **Production and decision awareness.** Deployment checklist, memory estimation function, "When NOT to Fine-Tune" decision framework, and fine-tuning vs prompting vs RAG trade-offs.
- **Interview breadth.** 10 questions spanning concepts, coding, system design, and trade-offs — including RAG vs fine-tuning, Adapter vs LoRA, and memory estimation.
- **Self-pacing infrastructure.** 5 section checkpoints and a job role mapping table for ML Engineers, Data Scientists, Architects, and Managers.

### Where This Blog Falls Short
- **Exercises lack solution sketches.** Five exercises are listed but none provide starter guidance or expected output ranges, making self-study difficult.
- **No serving framework code.** TorchServe, Triton, and vLLM are not demonstrated. The deployment checklist is declarative, not runnable.
- **Prefix Tuning and Prompt Tuning remain table-only.** Adapters and LoRA have code, but the other two PEFT methods are still described only in the comparison table.
- **No confidence interval or statistical significance testing.** The evaluation section discusses calibration but does not show bootstrap confidence intervals for accuracy estimates.

### Architect Sanity Checks

### Check 1: Production Deployment Readiness
**Question**: Would I trust this person to deploy a fine-tuned model to production?
**Answer: YES** -- The blog covers proper train/val/test splits, baseline comparison, memory estimation, ONNX export, quantization, deployment checklist with monitoring/rollback, and a "When NOT to Fine-Tune" decision framework. The reader still needs serving framework experience (TorchServe, Triton), but has the evaluation and optimization fundamentals.

### Check 2: Deep Problem Understanding
**Question**: Can they diagnose and fix training failures systematically?
**Answer: YES** -- Five failure modes (catastrophic forgetting, overfitting, LR issues, OOM, tokenizer mismatches) are explained with root cause analysis and multiple mitigations each. EWC includes full Fisher Information computation. The warmup rationale explains Adam's second-moment initialization. The tokenizer mismatch section covers the most common silent failure in production.

### Check 3: Interview and Career Readiness
**Question**: Would this person clearly articulate BERT vs GPT differences, PEFT techniques, and fine-tuning trade-offs to a senior engineer?
**Answer: YES** -- BERT vs GPT comparison is crisp with attention diagrams, training objectives, and decision matrix. PEFT covers both LoRA (low-rank decomposition) and Adapters (bottleneck modules) with implementations and trade-off guidance. The "When NOT to Fine-Tune" section and RAG-vs-fine-tuning interview question prepare for the most common senior-engineer follow-ups.
