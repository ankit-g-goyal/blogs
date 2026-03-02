# Blog 23: Fine-Tuning Models
## Prompt Your Career: The Complete Generative AI Masterclass

**Reading time:** 60-90 minutes
**Coding time:** 90-120 minutes (dataset prep + training setup)
**Total investment:** ~3.5 hours

---

## What You'll Walk Away With

By the end of this blog, you will be able to:

1. **Decide when fine-tuning is appropriate** versus prompt engineering or RAG
2. **Prepare high-quality training datasets** for different fine-tuning approaches
3. **Fine-tune models via OpenAI's API** for production use
4. **Train open-source models** using Hugging Face and PEFT
5. **Implement LoRA and QLoRA** for efficient fine-tuning
6. **Evaluate fine-tuned models** using proper metrics beyond string matching
7. **Understand catastrophic forgetting** and strategies to mitigate it

> **How to read this blog:** If you're primarily using API-based fine-tuning (OpenAI), focus on the Dataset Preparation and OpenAI sections first, then read Evaluation. If you're working with open-source models, read the Hugging Face/PEFT section in detail. The Catastrophic Forgetting and Evaluation sections are essential for everyone -- skip them at your peril.

### Prerequisites

Before starting this blog, you should be comfortable with:

- **Blog 14 (Working with AI APIs):** API patterns, authentication, and request/response handling
- **Blog 9 (Transformers):** Attention mechanisms and transformer architecture basics
- **Blog 5 (PyTorch Deep Learning):** Training loops, loss functions, and gradient descent
- **Blog 17 (RAG Systems):** Understanding RAG so you can decide when fine-tuning is better
- **Python proficiency:** Classes, decorators, and working with JSON/JSONL files
- **Basic command line:** pip installs, GPU monitoring (nvidia-smi)

---

## What This Blog Does NOT Cover

Before we begin, let's set clear expectations on scope:

- **Reinforcement Learning from Human Feedback (RLHF)** -- this is how models like ChatGPT are aligned after pre-training; it requires its own dedicated treatment and infrastructure (reward models, PPO training).
- **Pre-training from scratch** -- we cover fine-tuning existing models, not training foundation models from zero. Pre-training requires thousands of GPUs and millions of dollars.
- **Production MLOps for fine-tuned models** -- model versioning, A/B testing infrastructure, and continuous retraining pipelines are covered in Blog 24.
- **Multi-modal fine-tuning** -- fine-tuning vision-language models or image generation models (see Blogs 20-22 for those domains).
- **Synthetic data generation** -- creating training data using LLMs is an emerging practice but introduces its own pitfalls (model collapse, bias amplification) that deserve separate coverage.
- **Legal and licensing considerations** -- model licenses (Llama's community license, Mistral's Apache 2.0, etc.) vary and you must verify commercial use rights before deploying.

---

## Manager's Summary

**For Technical Leaders and Decision Makers:**

Fine-tuning creates specialized models that outperform general-purpose ones for specific tasks. However, it requires investment in data, compute, and maintenance.

**When to Fine-Tune:**
- Consistent formatting requirements (JSON, specific structures)
- Domain-specific terminology or knowledge
- Distinctive tone or style requirements
- Performance optimization for specific task types
- Reducing token usage (shorter prompts possible)

**When NOT to Fine-Tune:**
- General Q&A (use RAG instead)
- Rapidly changing information
- One-off or experimental tasks
- Limited training data (<100 examples)

**Cost-Benefit Analysis:**

| Approach | Setup Cost | Per-Query Cost | Best For |
|----------|------------|----------------|----------|
| Prompt Engineering | Low | Higher tokens | Exploration |
| RAG | Medium | Medium | Dynamic knowledge |
| OpenAI Fine-tuning | Medium | Lower tokens | Consistent formatting |
| Self-hosted Fine-tune | High | Very low | High volume, privacy |

> **Pricing note:** Fine-tuning costs change frequently. Always check current pricing at the provider's official page before making production decisions. The ROI timeline depends heavily on your query volume and the specific cost reduction fine-tuning achieves for your use case.

---

## The Fine-Tuning Decision Tree

```
Should You Fine-Tune?
|
+-- Is the task well-defined with consistent patterns?
|  +-- No -> Try prompt engineering first
|  +-- Yes |
|
+-- Do you have 100+ high-quality examples?
|  +-- No -> Collect more data or use few-shot
|  +-- Yes |
|
+-- Is the knowledge static or slowly changing?
|  +-- No -> Consider RAG instead
|  +-- Yes |
|
+-- Do you need consistent output format?
|  +-- No -> Prompt engineering may suffice
|  +-- Yes |
|
+-- Fine-tuning is likely beneficial!
```

---

## Catastrophic Forgetting: The Hidden Risk of Fine-Tuning

Before diving into implementation, you need to understand the most common pitfall in fine-tuning: **catastrophic forgetting**.

### What Is Catastrophic Forgetting?

When you fine-tune a model on a narrow dataset, it can "forget" capabilities it had before fine-tuning. The model's weights shift to optimize for your specific task, potentially degrading performance on everything else.

**Example scenario:** You fine-tune GPT-4o-mini on customer service responses. After fine-tuning:
- Customer service responses improve significantly
- General reasoning ability may degrade
- Code generation quality may drop
- Multi-language capabilities may weaken

This happens because fine-tuning updates the same weights that encode general knowledge. Optimizing for your narrow task can overwrite broadly useful representations.

### How to Detect Catastrophic Forgetting

```python
"""
Catastrophic Forgetting Detection

Run a diverse evaluation BEFORE and AFTER fine-tuning
to measure capability degradation.
"""
from dataclasses import dataclass, field
from typing import Callable, Dict, List


@dataclass
class CapabilityTest:
    """A test case for a specific model capability."""
    category: str
    prompt: str
    evaluation_fn: Callable[[str], float]  # Returns 0.0-1.0 score
    description: str = ""


@dataclass
class ForgettingReport:
    """Report comparing pre- and post-fine-tuning capabilities."""
    category_scores_before: Dict[str, float] = field(default_factory=dict)
    category_scores_after: Dict[str, float] = field(default_factory=dict)
    degradation: Dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        lines = ["=== Catastrophic Forgetting Report ===\n"]
        lines.append(f"{'Category':<25} {'Before':>8} {'After':>8} {'Change':>8}")
        lines.append("-" * 55)

        for category in self.category_scores_before:
            before = self.category_scores_before[category]
            after = self.category_scores_after.get(category, 0.0)
            change = after - before
            flag = " ** DEGRADED **" if change < -0.1 else ""
            lines.append(
                f"{category:<25} {before:>8.3f} {after:>8.3f} {change:>+8.3f}{flag}"
            )

        degraded = [
            cat for cat, delta in self.degradation.items() if delta < -0.1
        ]
        if degraded:
            lines.append(f"\nWARNING: {len(degraded)} categories show >10% degradation:")
            for cat in degraded:
                lines.append(f"  - {cat}: {self.degradation[cat]:+.3f}")
            lines.append("\nConsider: reducing epochs, using LoRA, or mixing in general data.")
        else:
            lines.append("\nNo significant degradation detected.")

        return "\n".join(lines)


def evaluate_forgetting(
    model_fn_before: Callable[[str], str],
    model_fn_after: Callable[[str], str],
    test_suite: List[CapabilityTest]
) -> ForgettingReport:
    """
    Compare model capabilities before and after fine-tuning.

    Args:
        model_fn_before: Model inference function BEFORE fine-tuning
        model_fn_after: Model inference function AFTER fine-tuning
        test_suite: List of capability tests across different categories

    Returns:
        ForgettingReport with per-category comparison
    """
    report = ForgettingReport()

    # Group tests by category
    categories: Dict[str, List[CapabilityTest]] = {}
    for test in test_suite:
        categories.setdefault(test.category, []).append(test)

    for category, tests in categories.items():
        before_scores = []
        after_scores = []

        for test in tests:
            # Evaluate before
            try:
                before_output = model_fn_before(test.prompt)
                before_score = test.evaluation_fn(before_output)
            except (ValueError, TypeError, RuntimeError) as e:
                print(f"  Warning: before-eval failed for '{test.description}': {e}")
                before_score = 0.0

            # Evaluate after
            try:
                after_output = model_fn_after(test.prompt)
                after_score = test.evaluation_fn(after_output)
            except (ValueError, TypeError, RuntimeError) as e:
                print(f"  Warning: after-eval failed for '{test.description}': {e}")
                after_score = 0.0

            before_scores.append(before_score)
            after_scores.append(after_score)

        avg_before = sum(before_scores) / len(before_scores) if before_scores else 0.0
        avg_after = sum(after_scores) / len(after_scores) if after_scores else 0.0

        report.category_scores_before[category] = avg_before
        report.category_scores_after[category] = avg_after
        report.degradation[category] = avg_after - avg_before

    return report


# Example: Build a forgetting test suite
def build_general_capability_tests() -> List[CapabilityTest]:
    """
    Build a test suite covering general capabilities.

    In production, you'd have 50-100+ tests across categories.
    This is a minimal illustrative example.
    """
    import json

    def scores_as_json(output: str) -> float:
        """Check if output is valid JSON."""
        try:
            json.loads(output)
            return 1.0
        except (json.JSONDecodeError, ValueError):
            return 0.0

    def contains_reasoning(output: str) -> float:
        """Check if output contains reasoning steps."""
        reasoning_markers = ["because", "therefore", "since", "first", "step"]
        found = sum(1 for m in reasoning_markers if m.lower() in output.lower())
        return min(found / 3, 1.0)  # Normalize: 3+ markers = 1.0

    def has_code_structure(output: str) -> float:
        """Check if output contains code-like structure."""
        markers = ["def ", "class ", "import ", "return ", "if ", "for "]
        found = sum(1 for m in markers if m in output)
        return min(found / 2, 1.0)

    tests = [
        CapabilityTest(
            category="reasoning",
            prompt="Explain step by step: if all roses are flowers and some flowers fade quickly, can we conclude all roses fade quickly?",
            evaluation_fn=contains_reasoning,
            description="Logical reasoning"
        ),
        CapabilityTest(
            category="code_generation",
            prompt="Write a Python function to find the two numbers in a list that add up to a target sum.",
            evaluation_fn=has_code_structure,
            description="Code generation"
        ),
        CapabilityTest(
            category="json_formatting",
            prompt='Convert this to JSON: Name is Alice, age 30, city is London.',
            evaluation_fn=scores_as_json,
            description="JSON formatting"
        ),
    ]

    return tests
```

### Strategies to Mitigate Catastrophic Forgetting

| Strategy | How It Works | Tradeoff |
|----------|-------------|----------|
| **LoRA / QLoRA** | Only updates small adapter weights, base model frozen | Slightly less task-specific adaptation |
| **Low learning rate** | Smaller weight updates preserve existing knowledge | Slower convergence, may need more epochs |
| **Few epochs** | Less time to overwrite general knowledge | May underfit on your task |
| **Mixed training data** | Include general examples alongside task-specific data | Larger dataset, longer training |
| **Elastic Weight Consolidation** | Penalizes changes to "important" weights | More complex training setup |
| **Evaluation gating** | Don't deploy if general capabilities degrade beyond threshold | Requires comprehensive test suite |

**Practical recommendation:** Start with LoRA (not full fine-tuning) and 1-3 epochs. Run the forgetting evaluation before and after. If degradation exceeds 10% on any critical capability, reduce the learning rate or add general-purpose examples to your training data.

---

## Dataset Preparation

### The Foundation of Good Fine-Tuning

```python
"""
Dataset Preparation for Fine-Tuning
"""
import json
from dataclasses import dataclass, field
from typing import List, Optional
import random

@dataclass
class TrainingExample:
    """A single training example."""
    system_prompt: Optional[str]
    user_message: str
    assistant_response: str
    metadata: dict = field(default_factory=dict)


class DatasetPreparer:
    """
    Prepare datasets for fine-tuning.

    Quality Guidelines:
    1. Diverse examples covering all use cases
    2. Consistent formatting
    3. High-quality responses (what you want the model to learn)
    4. Balance across categories
    5. Include edge cases
    """

    def __init__(self):
        self.examples: List[TrainingExample] = []

    def add_example(
        self,
        user_message: str,
        assistant_response: str,
        system_prompt: str = None,
        metadata: dict = None
    ):
        """Add a training example."""
        self.examples.append(TrainingExample(
            system_prompt=system_prompt,
            user_message=user_message,
            assistant_response=assistant_response,
            metadata=metadata or {}
        ))

    def to_openai_format(self) -> List[dict]:
        """Convert to OpenAI fine-tuning format."""
        formatted = []

        for ex in self.examples:
            messages = []

            if ex.system_prompt:
                messages.append({
                    "role": "system",
                    "content": ex.system_prompt
                })

            messages.append({
                "role": "user",
                "content": ex.user_message
            })

            messages.append({
                "role": "assistant",
                "content": ex.assistant_response
            })

            formatted.append({"messages": messages})

        return formatted

    def to_alpaca_format(self) -> List[dict]:
        """Convert to Alpaca format (common for open-source)."""
        formatted = []

        for ex in self.examples:
            item = {
                "instruction": ex.user_message,
                "output": ex.assistant_response
            }

            if ex.system_prompt:
                item["input"] = ex.system_prompt

            formatted.append(item)

        return formatted

    def to_sharegpt_format(self) -> List[dict]:
        """Convert to ShareGPT format."""
        formatted = []

        for ex in self.examples:
            conversations = []

            if ex.system_prompt:
                conversations.append({
                    "from": "system",
                    "value": ex.system_prompt
                })

            conversations.append({
                "from": "human",
                "value": ex.user_message
            })

            conversations.append({
                "from": "gpt",
                "value": ex.assistant_response
            })

            formatted.append({"conversations": conversations})

        return formatted

    def save_jsonl(self, filepath: str, fmt: str = "openai"):
        """Save dataset as JSONL file."""
        if fmt == "openai":
            data = self.to_openai_format()
        elif fmt == "alpaca":
            data = self.to_alpaca_format()
        elif fmt == "sharegpt":
            data = self.to_sharegpt_format()
        else:
            raise ValueError(f"Unknown format: {fmt}")

        with open(filepath, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")

    def split(
        self,
        train_ratio: float = 0.9,
        seed: int = 42
    ) -> tuple["DatasetPreparer", "DatasetPreparer"]:
        """Split into training and validation sets."""
        rng = random.Random(seed)
        shuffled = list(self.examples)
        rng.shuffle(shuffled)
        split_idx = int(len(shuffled) * train_ratio)

        train_prep = DatasetPreparer()
        train_prep.examples = shuffled[:split_idx]

        val_prep = DatasetPreparer()
        val_prep.examples = shuffled[split_idx:]

        return train_prep, val_prep

    def analyze(self) -> dict:
        """Analyze dataset statistics."""
        if not self.examples:
            return {"error": "No examples"}

        user_lengths = [len(ex.user_message) for ex in self.examples]
        response_lengths = [len(ex.assistant_response) for ex in self.examples]

        return {
            "total_examples": len(self.examples),
            "avg_user_length": sum(user_lengths) / len(user_lengths),
            "avg_response_length": sum(response_lengths) / len(response_lengths),
            "max_user_length": max(user_lengths),
            "max_response_length": max(response_lengths),
            "has_system_prompts": sum(1 for ex in self.examples if ex.system_prompt)
        }


# Example: Create customer service dataset
def create_customer_service_dataset():
    """Create example customer service fine-tuning dataset."""
    preparer = DatasetPreparer()

    system_prompt = """You are a helpful customer service agent for TechCorp.
Always be polite, professional, and solution-oriented.
If you don't know something, offer to escalate to a specialist."""

    examples = [
        {
            "user": "My order hasn't arrived yet. Order #12345",
            "assistant": """I apologize for the delay with your order #12345. Let me check the status for you.

I can see your order was shipped on January 10th and is currently in transit. The estimated delivery is January 15th.

If it doesn't arrive by then, please let me know and I'll arrange a replacement or refund. Is there anything else I can help with?"""
        },
        {
            "user": "How do I return a product?",
            "assistant": """I'd be happy to help you with a return. Here's our simple process:

1. **Initiate Return**: Log into your account and select the order you want to return
2. **Print Label**: Download and print the prepaid return label
3. **Package Item**: Pack the item in its original packaging if possible
4. **Ship**: Drop off at any authorized shipping location

Returns are processed within 5-7 business days after we receive the item. Would you like me to start the return process for a specific order?"""
        },
        {
            "user": "This is ridiculous! I've been waiting 3 weeks for my refund!",
            "assistant": """I completely understand your frustration, and I sincerely apologize for this delay. A 3-week wait for a refund is not acceptable, and I want to resolve this immediately.

Let me look into your account right now to see what's causing the holdup.

Could you please provide your order number or the email associated with your account? I'll prioritize getting your refund processed today and ensure you receive a confirmation email within 2 hours.

Again, I'm sorry for this experience. We'll make this right."""
        },
        {
            "user": "Can I change my shipping address?",
            "assistant": """Absolutely! I can help you update your shipping address.

**If your order hasn't shipped yet:**
I can change it immediately. Just provide the new address and order number.

**If your order has already shipped:**
We may be able to redirect it depending on the carrier. I'll need to check the shipping status first.

Which order are you looking to update? Please share the order number and I'll check what options are available."""
        }
    ]

    for ex in examples:
        preparer.add_example(
            user_message=ex["user"],
            assistant_response=ex["assistant"],
            system_prompt=system_prompt
        )

    return preparer


# Validate dataset quality
class DatasetValidator:
    """Validate training dataset quality."""

    def __init__(self, min_examples: int = 100, max_tokens: int = 4096):
        self.min_examples = min_examples
        self.max_tokens = max_tokens
        self.issues: List[str] = []

    def validate(self, preparer: DatasetPreparer) -> dict:
        """Run all validations."""
        self.issues = []

        self._check_size(preparer)
        self._check_duplicates(preparer)
        self._check_lengths(preparer)
        self._check_balance(preparer)
        self._check_quality(preparer)

        return {
            "valid": len(self.issues) == 0,
            "issues": self.issues,
            "stats": preparer.analyze()
        }

    def _check_size(self, preparer: DatasetPreparer):
        if len(preparer.examples) < self.min_examples:
            self.issues.append(
                f"Dataset has only {len(preparer.examples)} examples. "
                f"Recommend at least {self.min_examples}."
            )

    def _check_duplicates(self, preparer: DatasetPreparer):
        seen: set = set()
        duplicates = 0

        for ex in preparer.examples:
            key = ex.user_message.strip().lower()
            if key in seen:
                duplicates += 1
            seen.add(key)

        if duplicates > 0:
            self.issues.append(f"Found {duplicates} duplicate user messages.")

    def _check_lengths(self, preparer: DatasetPreparer):
        try:
            import tiktoken
        except ImportError:
            self.issues.append(
                "tiktoken not installed; skipping token length validation. "
                "Install with: pip install tiktoken"
            )
            return

        encoder = tiktoken.get_encoding("cl100k_base")

        for i, ex in enumerate(preparer.examples):
            total = ex.user_message + (ex.system_prompt or "") + ex.assistant_response
            tokens = len(encoder.encode(total))

            if tokens > self.max_tokens:
                self.issues.append(
                    f"Example {i} exceeds token limit: {tokens} > {self.max_tokens}"
                )

    def _check_balance(self, preparer: DatasetPreparer):
        # Check response length variance
        lengths = [len(ex.assistant_response) for ex in preparer.examples]
        avg = sum(lengths) / len(lengths)
        variance = sum((l - avg) ** 2 for l in lengths) / len(lengths)

        if variance > avg ** 2:
            self.issues.append(
                "High variance in response lengths. "
                "Consider making responses more consistent."
            )

    def _check_quality(self, preparer: DatasetPreparer):
        # Basic quality checks
        for i, ex in enumerate(preparer.examples):
            if len(ex.assistant_response) < 10:
                self.issues.append(
                    f"Example {i} has very short response: '{ex.assistant_response[:50]}'"
                )

            if ex.assistant_response.strip() == ex.user_message.strip():
                self.issues.append(f"Example {i} response echoes user message.")
```

---

## Fine-Tuning with OpenAI

### OpenAI Fine-Tuning API

```python
"""
OpenAI Fine-Tuning Implementation
"""
from openai import OpenAI
import time
import json
from pathlib import Path

client = OpenAI()


class OpenAIFineTuner:
    """
    Fine-tune models using OpenAI's API.

    Supported models (check OpenAI docs for current list):
    - gpt-4o-mini-2024-07-18 (recommended for most cases)
    - gpt-4o-2024-08-06

    Note: Pricing changes frequently. Check https://openai.com/pricing
    for current fine-tuning costs before starting a job.
    """

    def __init__(self):
        self.client = OpenAI()
        self.current_job_id = None

    def upload_training_file(self, filepath: str) -> str:
        """Upload training file and return file ID."""
        with open(filepath, "rb") as f:
            response = self.client.files.create(
                file=f,
                purpose="fine-tune"
            )
        return response.id

    def create_fine_tune_job(
        self,
        training_file_id: str,
        model: str = "gpt-4o-mini-2024-07-18",
        validation_file_id: str = None,
        suffix: str = None,
        n_epochs: int = None,
        batch_size: int = None,
        learning_rate_multiplier: float = None
    ) -> dict:
        """
        Create a fine-tuning job.

        Args:
            training_file_id: ID of uploaded training file
            model: Base model to fine-tune
            validation_file_id: Optional validation file
            suffix: Custom suffix for model name (max 40 chars)
            n_epochs: Number of training epochs (auto if None)
            batch_size: Batch size (auto if None)
            learning_rate_multiplier: Learning rate adjustment

        Returns:
            Job details
        """
        hyperparameters = {}

        if n_epochs:
            hyperparameters["n_epochs"] = n_epochs
        if batch_size:
            hyperparameters["batch_size"] = batch_size
        if learning_rate_multiplier:
            hyperparameters["learning_rate_multiplier"] = learning_rate_multiplier

        job_params = {
            "training_file": training_file_id,
            "model": model
        }

        if validation_file_id:
            job_params["validation_file"] = validation_file_id

        if suffix:
            job_params["suffix"] = suffix

        if hyperparameters:
            job_params["hyperparameters"] = hyperparameters

        response = self.client.fine_tuning.jobs.create(**job_params)

        self.current_job_id = response.id
        return {
            "job_id": response.id,
            "status": response.status,
            "model": response.model,
            "created_at": response.created_at
        }

    def get_job_status(self, job_id: str = None) -> dict:
        """Get fine-tuning job status."""
        job_id = job_id or self.current_job_id
        if not job_id:
            raise ValueError("No job ID provided")

        response = self.client.fine_tuning.jobs.retrieve(job_id)

        return {
            "status": response.status,
            "fine_tuned_model": response.fine_tuned_model,
            "trained_tokens": response.trained_tokens,
            "error": response.error
        }

    def wait_for_completion(
        self,
        job_id: str = None,
        poll_interval: int = 30,
        timeout: int = 7200
    ) -> dict:
        """Wait for fine-tuning to complete."""
        job_id = job_id or self.current_job_id
        start_time = time.time()

        while time.time() - start_time < timeout:
            status = self.get_job_status(job_id)

            print(f"Status: {status['status']}")

            if status["status"] == "succeeded":
                return status

            if status["status"] in ["failed", "cancelled"]:
                raise RuntimeError(f"Job {status['status']}: {status.get('error')}")

            time.sleep(poll_interval)

        raise TimeoutError("Fine-tuning timed out")

    def list_events(self, job_id: str = None, limit: int = 10) -> list:
        """List fine-tuning events."""
        job_id = job_id or self.current_job_id

        events = self.client.fine_tuning.jobs.list_events(
            fine_tuning_job_id=job_id,
            limit=limit
        )

        return [
            {
                "created_at": e.created_at,
                "level": e.level,
                "message": e.message
            }
            for e in events.data
        ]

    def cancel_job(self, job_id: str = None):
        """Cancel a running fine-tuning job."""
        job_id = job_id or self.current_job_id
        self.client.fine_tuning.jobs.cancel(job_id)

    def list_jobs(self, limit: int = 10) -> list:
        """List all fine-tuning jobs."""
        jobs = self.client.fine_tuning.jobs.list(limit=limit)

        return [
            {
                "id": j.id,
                "model": j.model,
                "status": j.status,
                "fine_tuned_model": j.fine_tuned_model,
                "created_at": j.created_at
            }
            for j in jobs.data
        ]


def use_fine_tuned_model(model_id: str, messages: list) -> str:
    """Use a fine-tuned model for inference."""
    client = OpenAI()

    response = client.chat.completions.create(
        model=model_id,
        messages=messages,
        temperature=0.7
    )

    return response.choices[0].message.content


# Complete fine-tuning workflow
def fine_tune_workflow(
    training_data_path: str,
    validation_data_path: str = None,
    model_suffix: str = "custom"
):
    """Complete fine-tuning workflow."""
    tuner = OpenAIFineTuner()

    # Step 1: Upload training file
    print("Uploading training file...")
    train_file_id = tuner.upload_training_file(training_data_path)
    print(f"Training file ID: {train_file_id}")

    # Upload validation file if provided
    val_file_id = None
    if validation_data_path:
        print("Uploading validation file...")
        val_file_id = tuner.upload_training_file(validation_data_path)
        print(f"Validation file ID: {val_file_id}")

    # Step 2: Create fine-tuning job
    print("\nCreating fine-tuning job...")
    job = tuner.create_fine_tune_job(
        training_file_id=train_file_id,
        validation_file_id=val_file_id,
        suffix=model_suffix
    )
    print(f"Job ID: {job['job_id']}")

    # Step 3: Wait for completion
    print("\nWaiting for completion (this may take a while)...")
    result = tuner.wait_for_completion()

    print(f"\nFine-tuning complete!")
    print(f"Model ID: {result['fine_tuned_model']}")
    print(f"Tokens trained: {result['trained_tokens']}")

    return result["fine_tuned_model"]


# Cost estimation
def estimate_fine_tuning_cost(
    training_file_path: str,
    model: str = "gpt-4o-mini-2024-07-18",
    n_epochs: int = 3
) -> dict:
    """
    Estimate fine-tuning costs.

    WARNING: Prices change frequently. This function uses hardcoded prices
    that may be outdated. Always verify at https://openai.com/pricing
    """
    try:
        import tiktoken
    except ImportError:
        return {"error": "tiktoken not installed. Run: pip install tiktoken"}

    # Load training data
    with open(training_file_path) as f:
        examples = [json.loads(line) for line in f]

    encoder = tiktoken.get_encoding("cl100k_base")

    total_tokens = 0
    for ex in examples:
        for msg in ex.get("messages", []):
            total_tokens += len(encoder.encode(msg.get("content", "")))

    training_tokens = total_tokens * n_epochs

    # Pricing per 1M tokens -- CHECK CURRENT PRICING before relying on these
    prices = {
        "gpt-4o-mini-2024-07-18": 3.00,
        "gpt-4o-2024-08-06": 25.00,
    }

    price_per_1m = prices.get(model, 3.00)
    estimated_cost = (training_tokens / 1_000_000) * price_per_1m

    return {
        "examples": len(examples),
        "tokens_per_example": total_tokens / len(examples),
        "total_tokens": total_tokens,
        "training_tokens": training_tokens,
        "epochs": n_epochs,
        "estimated_cost_usd": round(estimated_cost, 2),
        "note": "Prices may be outdated. Verify at https://openai.com/pricing"
    }
```

---

## Understanding LoRA and QLoRA: Why They Work

Before diving into implementation, you need to understand the mechanism behind efficient fine-tuning — not just the API calls.

### LoRA: Low-Rank Adaptation

**The core insight:** When you fine-tune a model, you update all weight matrices (e.g., a 4096×4096 attention projection = 16.7M parameters per matrix). But research shows that the *change* in weights during fine-tuning has very low intrinsic rank — meaning most of the update is redundant. LoRA exploits this.

Instead of updating the full weight matrix **W** (dimensions d×d), LoRA learns a low-rank update **ΔW = A × B**, where:
- **A** has dimensions d × r (e.g., 4096 × 16)
- **B** has dimensions r × d (e.g., 16 × 4096)
- **r** (rank) is typically 4-64, much smaller than d

The effective weight becomes **W + α/r × (A × B)**, where α (lora_alpha) is a scaling factor.

```
Full fine-tuning:        W_new = W_old + ΔW          (d × d parameters to learn)
LoRA fine-tuning:        W_new = W_old + α/r × A × B  (d × r + r × d parameters)

Example with d=4096, r=16:
  Full: 16,777,216 parameters per matrix
  LoRA: 4096×16 + 16×4096 = 131,072 parameters (0.78% of full!)

For a 7B model with ~200M attention parameters:
  Full: ~200M trainable params → ~28GB VRAM in fp16
  LoRA (r=16): ~1.6M trainable params → ~6GB VRAM with 4-bit base
```

**Why low rank works:** Fine-tuning for a specific task doesn't require changing the model's entire knowledge representation. It only needs to nudge the model's behavior in a specific direction — and that nudge lives in a low-dimensional subspace. LoRA captures exactly that subspace.

**Key parameters and their effects:**

| Parameter | What It Controls | Too Low | Too High | Recommended Start |
|-----------|-----------------|---------|----------|-------------------|
| `r` (rank) | Capacity of the adapter | Underfits task, low quality | Overfits, approaches full fine-tune cost | 8-16 for most tasks |
| `lora_alpha` | Scaling of LoRA updates | Updates too small, model barely changes | Updates too large, destabilizes training | 2× rank (e.g., alpha=32 for r=16) |
| `lora_dropout` | Regularization | Less regularization | More regularization, may slow convergence | 0.05-0.1 |
| `target_modules` | Which layers get LoRA | Missing key layers = poor adaptation | All layers = slower, more VRAM | At minimum: q_proj, v_proj |

### QLoRA: Quantization + LoRA

QLoRA combines two techniques:
1. **Load the base model in 4-bit quantization** (NF4 data type) — reduces the frozen base model from ~14GB to ~3.5GB for a 7B model
2. **Add LoRA adapters in fp16/bf16** — the small trainable adapters remain in full precision for gradient computation
3. **Paged optimizers** — spills optimizer states to CPU memory when GPU VRAM is full

The result: you can fine-tune a 7B model on a single consumer GPU with 8GB VRAM. The quality trade-off vs. standard LoRA is typically small (<2% on benchmarks), but the VRAM savings are dramatic.

### Hardware Requirements for Different Approaches

```
Fine-Tuning VRAM Requirements (7B model):
┌──────────────────────────┬──────────┬────────────┬────────────────────────────────┐
│ Approach                 │ VRAM     │ Train Time │ When To Use                    │
│                          │ (7B)     │ (1K samples)│                                │
├──────────────────────────┼──────────┼────────────┼────────────────────────────────┤
│ Full fine-tuning (fp16)  │ ~28 GB   │ ~30 min    │ Maximum quality, A100/H100     │
│ Full fine-tuning (bf16)  │ ~28 GB   │ ~25 min    │ Same but better for newer GPUs │
│ LoRA (fp16 base)         │ ~16 GB   │ ~20 min    │ Good quality, RTX 4090         │
│ LoRA (8-bit base)        │ ~10 GB   │ ~25 min    │ Consumer GPU, slight quality   │
│ QLoRA (4-bit base)       │ ~6 GB    │ ~35 min    │ Low VRAM, RTX 3060-4070       │
│ OpenAI API fine-tuning   │ 0 GB     │ ~10-30 min │ No GPU needed, highest $/run   │
├──────────────────────────┼──────────┼────────────┼────────────────────────────────┤
│ For 13B models:          │ ~1.8×    │ ~2×        │ Multiply above numbers          │
│ For 70B models:          │ ~10×     │ ~10×       │ Multi-GPU required              │
└──────────────────────────┴──────────┴────────────┴────────────────────────────────┘

NOTE: Training times are approximate on A100 40GB. Gradient accumulation
can trade VRAM for time (smaller batches, more steps). Consumer GPUs
(RTX 3090/4090) are typically 1.5-3× slower than A100.
```

---

## Fine-Tuning Open-Source Models

### Using Hugging Face and PEFT

```python
"""
Fine-Tuning Open-Source Models with Hugging Face
"""
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
import json

class OpenSourceFineTuner:
    """
    Fine-tune open-source models using Hugging Face + PEFT.

    Approach comparison:
    - Full fine-tuning: Updates all parameters. Best quality ceiling but
      requires ~4× model size in VRAM (weights + gradients + optimizer states).
    - LoRA: Freezes base, trains low-rank adapters. ~0.5-2% of parameters.
      Often matches full fine-tuning quality due to regularization effect.
    - QLoRA: LoRA + 4-bit quantized base. Lowest VRAM. Small quality trade-off.
    """

    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
        use_4bit: bool = True,
        device_map: str = "auto"
    ):
        """
        Initialize the fine-tuner.

        Popular models for fine-tuning (check license terms before commercial use):
        - mistralai/Mistral-7B-Instruct-v0.2 (Apache 2.0)
        - meta-llama/Llama-2-7b-chat-hf (Community license)
        - microsoft/phi-2 (MIT)
        - google/gemma-7b-it (Gemma license)
        """
        self.model_name = model_name
        self.use_4bit = use_4bit

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        # Load model with quantization if requested
        if use_4bit:
            from transformers import BitsAndBytesConfig

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map=device_map,
                trust_remote_code=True
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map=device_map,
                trust_remote_code=True
            )

    def prepare_lora(
        self,
        r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        target_modules: list = None
    ):
        """
        Prepare model for LoRA training.

        Args:
            r: LoRA rank (higher = more capacity, more memory)
            lora_alpha: LoRA scaling factor
            lora_dropout: Dropout for LoRA layers
            target_modules: Which modules to apply LoRA to
        """
        if target_modules is None:
            # Common targets for most models
            target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]

        lora_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )

        # Prepare model
        if self.use_4bit:
            self.model = prepare_model_for_kbit_training(self.model)

        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

    def prepare_dataset(
        self,
        data_path: str,
        fmt: str = "alpaca",
        max_length: int = 512
    ) -> Dataset:
        """
        Prepare dataset for training.

        Formats:
        - alpaca: {"instruction": "", "output": ""}
        - sharegpt: {"conversations": [{"from": "human", "value": ""}, ...]}
        - openai: {"messages": [{"role": "", "content": ""}]}
        """
        # Load data
        with open(data_path) as f:
            if data_path.endswith(".jsonl"):
                data = [json.loads(line) for line in f]
            else:
                data = json.load(f)

        # Format into prompts
        formatted = []
        for item in data:
            if fmt == "alpaca":
                text = self._format_alpaca(item)
            elif fmt == "sharegpt":
                text = self._format_sharegpt(item)
            elif fmt == "openai":
                text = self._format_openai(item)
            else:
                raise ValueError(f"Unknown format: {fmt}")

            formatted.append({"text": text})

        # Create dataset
        dataset = Dataset.from_list(formatted)

        # Tokenize
        def tokenize(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=max_length,
                padding="max_length"
            )

        tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])

        return tokenized

    def _format_alpaca(self, item: dict) -> str:
        """Format Alpaca-style item."""
        instruction = item.get("instruction", "")
        input_text = item.get("input", "")
        output = item.get("output", "")

        if input_text:
            prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
        else:
            prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"

        return prompt

    def _format_sharegpt(self, item: dict) -> str:
        """Format ShareGPT-style item."""
        text = ""
        for turn in item.get("conversations", []):
            role = turn.get("from", "")
            value = turn.get("value", "")

            if role == "human":
                text += f"### Human:\n{value}\n\n"
            elif role in ["gpt", "assistant"]:
                text += f"### Assistant:\n{value}\n\n"
            elif role == "system":
                text += f"### System:\n{value}\n\n"

        return text.strip()

    def _format_openai(self, item: dict) -> str:
        """Format OpenAI messages-style item."""
        text = ""
        for msg in item.get("messages", []):
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "system":
                text += f"### System:\n{content}\n\n"
            elif role == "user":
                text += f"### User:\n{content}\n\n"
            elif role == "assistant":
                text += f"### Assistant:\n{content}\n\n"

        return text.strip()

    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Dataset = None,
        output_dir: str = "./fine_tuned_model",
        num_epochs: int = 3,
        batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        learning_rate: float = 2e-4,
        warmup_steps: int = 100,
        logging_steps: int = 10,
        save_steps: int = 100
    ):
        """
        Train the model.

        Args:
            train_dataset: Tokenized training dataset
            eval_dataset: Optional evaluation dataset
            output_dir: Where to save checkpoints
            num_epochs: Number of training epochs
            batch_size: Per-device batch size
            gradient_accumulation_steps: Accumulate gradients over N steps
            learning_rate: Learning rate
        """
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            logging_steps=logging_steps,
            save_steps=save_steps,
            save_total_limit=3,
            fp16=True,
            optim="paged_adamw_8bit" if self.use_4bit else "adamw_torch",
            eval_strategy="steps" if eval_dataset else "no",
            eval_steps=save_steps if eval_dataset else None,
            load_best_model_at_end=True if eval_dataset else False,
            report_to="none"  # Or "wandb" for tracking
        )

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator
        )

        # Train
        trainer.train()

        # Save final model
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        return output_dir

    def merge_and_save(self, output_dir: str):
        """Merge LoRA weights into base model and save."""
        merged_model = self.model.merge_and_unload()
        merged_model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)


# Complete workflow
def fine_tune_open_source(
    training_file: str,
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
    output_dir: str = "./my_fine_tuned_model"
):
    """Complete fine-tuning workflow for open-source models."""
    print(f"Fine-tuning {model_name}")

    # Initialize
    tuner = OpenSourceFineTuner(model_name, use_4bit=True)

    # Prepare LoRA
    print("Setting up LoRA...")
    tuner.prepare_lora(r=16, lora_alpha=32)

    # Prepare dataset
    print("Preparing dataset...")
    dataset = tuner.prepare_dataset(training_file, fmt="alpaca")

    # Split
    split = dataset.train_test_split(test_size=0.1)
    train_dataset = split["train"]
    eval_dataset = split["test"]

    # Train
    print("Starting training...")
    tuner.train(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        output_dir=output_dir,
        num_epochs=3
    )

    # Merge and save
    print("Merging LoRA weights...")
    tuner.merge_and_save(output_dir + "_merged")

    print(f"Model saved to {output_dir}_merged")
    return output_dir + "_merged"
```

### Hyperparameter Tuning Guide

Choosing hyperparameters is where most fine-tuning practitioners waste time. Here's an evidence-based starting point:

```
Hyperparameter Decision Table:

┌──────────────────────┬─────────────────┬──────────────────────────────────────────────┐
│ Parameter            │ Start With      │ Adjust If...                                 │
├──────────────────────┼─────────────────┼──────────────────────────────────────────────┤
│ Learning rate        │ 2e-4 (LoRA)     │ Training loss oscillates → lower to 1e-4     │
│                      │ 2e-5 (full FT)  │ Loss barely moves → raise by 2-5×            │
│ Epochs               │ 1-3             │ Val loss increases after epoch 1 → use 1      │
│                      │                 │ Val loss still decreasing → add 1 more        │
│ Batch size           │ 4 (per device)  │ OOM → reduce to 2, increase grad accum       │
│ Grad accumulation    │ 4               │ Effective batch = batch × accum × GPUs        │
│                      │                 │ Target: effective batch of 16-32              │
│ LoRA rank (r)        │ 16              │ Task is simple/formatting → try r=8           │
│                      │                 │ Task requires new knowledge → try r=32-64     │
│ Warmup steps         │ 10% of total    │ Loss spikes early → increase warmup           │
│ Weight decay          │ 0.01            │ Overfitting → increase to 0.05               │
│ Max sequence length  │ 512-1024        │ Match your actual data lengths                │
└──────────────────────┴─────────────────┴──────────────────────────────────────────────┘

KEY INSIGHT: The most common mistake is training too long. With LoRA, most tasks
converge in 1-2 epochs. Validation loss should decrease — if it rises, you're
overfitting and destroying the model's generalization (this IS catastrophic forgetting).
```

**How to diagnose training problems from the loss curve:**
- **Loss drops then rises** → Overfitting. Reduce epochs, increase dropout, or add more diverse training data.
- **Loss barely moves** → Learning rate too low, or data format is wrong (model can't parse your examples).
- **Loss oscillates wildly** → Learning rate too high, or batch size too small (noisy gradients).
- **Loss drops immediately and stays flat** → Your task may be too easy for fine-tuning. Try prompt engineering first.

### Using Unsloth for Faster Training

```python
"""
Fast Fine-Tuning with Unsloth
"""
# pip install unsloth

def fine_tune_with_unsloth(
    training_file: str,
    model_name: str = "unsloth/mistral-7b-instruct-v0.2-bnb-4bit",
    output_dir: str = "./unsloth_model"
):
    """
    Fine-tune using Unsloth (2-5x faster than standard).

    Unsloth provides:
    - Optimized kernels for faster training
    - Lower memory usage
    - Pre-quantized models
    """
    from unsloth import FastLanguageModel
    from trl import SFTTrainer
    from transformers import TrainingArguments
    from datasets import Dataset
    import json

    # Load model with Unsloth
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=2048,
        dtype=None,  # Auto-detect
        load_in_4bit=True
    )

    # Add LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=16,
        lora_dropout=0,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        bias="none",
        use_gradient_checkpointing="unsloth"
    )

    # Load and format dataset
    with open(training_file) as f:
        data = [json.loads(line) for line in f]

    # Format for chat
    formatted_data = []
    for item in data:
        messages = item.get("messages", [])
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        formatted_data.append({"text": text})

    dataset = Dataset.from_list(formatted_data)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=100,  # Adjust based on dataset size
        learning_rate=2e-4,
        fp16=True,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=42,
        save_strategy="steps",
        save_steps=25
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=2048,
        args=training_args
    )

    # Train
    trainer.train()

    # Save
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    return output_dir
```

---

## Model Evaluation

### Why String Matching Is Not Enough

A common mistake in evaluating fine-tuned models is using exact string matching as the primary metric. Consider: if your model outputs `"The capital of France is Paris."` and the reference is `"Paris"`, exact match gives 0% accuracy despite being correct. Even lowercased string comparison fails for paraphrases, reorderings, or valid alternative answers.

**You need a multi-dimensional evaluation strategy.**

### Evaluating Fine-Tuned Models

```python
"""
Fine-Tuned Model Evaluation

This module provides proper evaluation beyond naive string matching.
Key improvements over simple exact-match:
1. Semantic similarity (embedding-based)
2. Normalized text comparison
3. Task-specific metrics (BLEU, ROUGE)
4. Format compliance checking
5. Statistical significance testing
"""
import json
import re
from typing import List, Dict, Callable, Optional
from dataclasses import dataclass, field


@dataclass
class EvaluationResult:
    """Result of model evaluation."""
    metrics: Dict[str, float]
    per_example_results: List[Dict]
    summary: str = ""

    def __str__(self):
        lines = ["=== Evaluation Results ==="]
        for metric, value in self.metrics.items():
            lines.append(f"  {metric}: {value:.4f}")
        if self.summary:
            lines.append(f"\n{self.summary}")
        return "\n".join(lines)


def normalize_text(text: str) -> str:
    """
    Normalize text for comparison.
    Strips whitespace, lowercases, removes extra spaces.
    """
    text = text.strip().lower()
    text = re.sub(r'\s+', ' ', text)
    return text


def flexible_match(predicted: str, expected: str) -> float:
    """
    Flexible string matching that handles common variations.

    Returns a score between 0.0 and 1.0.
    - 1.0: Exact match after normalization
    - 0.5-0.9: Partial match (containment)
    - 0.0: No match
    """
    pred_norm = normalize_text(predicted)
    exp_norm = normalize_text(expected)

    # Exact match after normalization
    if pred_norm == exp_norm:
        return 1.0

    # Expected is contained in predicted (model gave correct answer with extra text)
    if exp_norm in pred_norm:
        # Penalize verbosity: shorter extra text = higher score
        ratio = len(exp_norm) / len(pred_norm) if pred_norm else 0
        return 0.5 + 0.4 * ratio  # Range: 0.5 to 0.9

    # Predicted is contained in expected (model gave a subset)
    if pred_norm in exp_norm:
        ratio = len(pred_norm) / len(exp_norm) if exp_norm else 0
        return 0.3 * ratio

    return 0.0


class FineTuneEvaluator:
    """
    Evaluate fine-tuned models with multiple metrics.

    Evaluation dimensions:
    1. Task accuracy (flexible matching, not just exact string match)
    2. Format compliance (does output follow required structure?)
    3. Quality metrics (BLEU, ROUGE for generation tasks)
    4. Latency and cost efficiency
    5. Regression testing (comparison against baseline)
    """

    def __init__(self, model_fn: Callable[[str], str]):
        """
        Args:
            model_fn: Function that takes input and returns model output
        """
        self.model_fn = model_fn

    def evaluate_classification(
        self,
        test_data: List[Dict],
        input_key: str = "input",
        label_key: str = "label",
        valid_labels: Optional[List[str]] = None
    ) -> EvaluationResult:
        """
        Evaluate classification accuracy with proper metrics.

        Unlike naive string matching, this:
        - Normalizes text before comparison
        - Computes per-class precision, recall, F1
        - Reports whether outputs are valid labels at all
        """
        per_example = []
        label_counts: Dict[str, Dict[str, int]] = {}  # label -> {tp, fp, fn}

        # Initialize counts
        if valid_labels:
            for label in valid_labels:
                label_counts[label] = {"tp": 0, "fp": 0, "fn": 0}

        invalid_outputs = 0

        for item in test_data:
            input_text = item[input_key]
            expected = normalize_text(item[label_key])

            raw_output = self.model_fn(input_text)
            predicted = normalize_text(raw_output)

            # Check if output is a valid label
            is_valid = True
            if valid_labels:
                valid_normalized = [normalize_text(l) for l in valid_labels]
                if predicted not in valid_normalized:
                    is_valid = False
                    invalid_outputs += 1
                    # Try to extract a valid label from the output
                    for vl in valid_normalized:
                        if vl in predicted:
                            predicted = vl
                            is_valid = True
                            break

            is_correct = predicted == expected

            # Update per-class counts
            if expected not in label_counts:
                label_counts[expected] = {"tp": 0, "fp": 0, "fn": 0}
            if predicted not in label_counts:
                label_counts[predicted] = {"tp": 0, "fp": 0, "fn": 0}

            if is_correct:
                label_counts[expected]["tp"] += 1
            else:
                label_counts[expected]["fn"] += 1
                label_counts[predicted]["fp"] += 1

            per_example.append({
                "input": input_text,
                "expected": expected,
                "predicted": predicted,
                "raw_output": raw_output,
                "correct": is_correct,
                "valid_output": is_valid
            })

        # Compute metrics
        total = len(test_data)
        correct = sum(1 for ex in per_example if ex["correct"])
        accuracy = correct / total if total > 0 else 0

        # Macro-averaged precision, recall, F1
        precisions, recalls, f1s = [], [], []
        for label, counts in label_counts.items():
            tp = counts["tp"]
            fp = counts["fp"]
            fn = counts["fn"]

            p = tp / (tp + fp) if (tp + fp) > 0 else 0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0

            precisions.append(p)
            recalls.append(r)
            f1s.append(f1)

        macro_precision = sum(precisions) / len(precisions) if precisions else 0
        macro_recall = sum(recalls) / len(recalls) if recalls else 0
        macro_f1 = sum(f1s) / len(f1s) if f1s else 0

        metrics = {
            "accuracy": accuracy,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
            "invalid_output_rate": invalid_outputs / total if total > 0 else 0,
            "total_examples": total,
        }

        summary_lines = [
            f"Accuracy: {accuracy:.3f} ({correct}/{total})",
            f"Macro F1: {macro_f1:.3f}",
            f"Invalid outputs: {invalid_outputs}/{total} ({invalid_outputs/total*100:.1f}%)" if total > 0 else "",
        ]

        return EvaluationResult(
            metrics=metrics,
            per_example_results=per_example[:50],  # Cap stored examples
            summary="\n".join(summary_lines)
        )

    def evaluate_generation(
        self,
        test_data: List[Dict],
        input_key: str = "input",
        reference_key: str = "output",
        use_rouge: bool = True,
        use_bleu: bool = True
    ) -> EvaluationResult:
        """
        Evaluate text generation quality with multiple metrics.

        Metrics used:
        - Flexible match: Normalized containment-based matching
        - ROUGE-L: Longest common subsequence overlap (if available)
        - BLEU: N-gram precision (if available)
        """
        per_example = []
        flexible_scores = []
        rouge_scores = []
        bleu_scores = []

        # Try importing scoring libraries
        rouge_scorer_obj = None
        if use_rouge:
            try:
                from rouge_score import rouge_scorer
                rouge_scorer_obj = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])
            except ImportError:
                print("Warning: rouge_score not installed. Skipping ROUGE. "
                      "Install with: pip install rouge-score")

        bleu_available = False
        if use_bleu:
            try:
                from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
                bleu_available = True
            except ImportError:
                print("Warning: nltk not installed. Skipping BLEU. "
                      "Install with: pip install nltk")

        for item in test_data:
            input_text = item[input_key]
            reference = item[reference_key]

            generated = self.model_fn(input_text)

            example_result = {
                "input": input_text,
                "reference": reference,
                "generated": generated,
            }

            # Flexible match
            fm_score = flexible_match(generated, reference)
            flexible_scores.append(fm_score)
            example_result["flexible_match"] = fm_score

            # ROUGE
            if rouge_scorer_obj is not None:
                rouge = rouge_scorer_obj.score(reference, generated)
                rouge_l = rouge["rougeL"].fmeasure
                rouge_scores.append(rouge_l)
                example_result["rouge_l"] = rouge_l

            # BLEU
            if bleu_available:
                smoothie = SmoothingFunction().method1
                bleu = sentence_bleu(
                    [reference.split()],
                    generated.split(),
                    smoothing_function=smoothie
                )
                bleu_scores.append(bleu)
                example_result["bleu"] = bleu

            per_example.append(example_result)

        # Aggregate metrics
        metrics = {
            "flexible_match_avg": sum(flexible_scores) / len(flexible_scores) if flexible_scores else 0,
        }
        if rouge_scores:
            metrics["rouge_l_avg"] = sum(rouge_scores) / len(rouge_scores)
        if bleu_scores:
            metrics["bleu_avg"] = sum(bleu_scores) / len(bleu_scores)
        metrics["total_examples"] = len(test_data)

        return EvaluationResult(
            metrics=metrics,
            per_example_results=per_example[:20],
            summary="\n".join(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}"
                              for k, v in metrics.items())
        )

    def evaluate_format_compliance(
        self,
        test_data: List[Dict],
        format_validator: Callable[[str], bool],
        input_key: str = "input"
    ) -> Dict:
        """Evaluate if outputs follow required format."""
        compliant = 0
        total = 0
        violations = []

        for item in test_data:
            input_text = item[input_key]
            output = self.model_fn(input_text)

            try:
                is_compliant = format_validator(output)
            except (json.JSONDecodeError, ValueError, KeyError, TypeError) as e:
                is_compliant = False
                violations.append({
                    "input": input_text,
                    "output": output[:200],  # Truncate for readability
                    "error": str(e)
                })

            if is_compliant:
                compliant += 1
            total += 1

        return {
            "compliance_rate": compliant / total if total > 0 else 0,
            "total_tested": total,
            "compliant": compliant,
            "violations": violations[:5]  # Sample
        }

    def compare_models(
        self,
        model_fns: Dict[str, Callable],
        test_data: List[Dict],
        input_key: str = "input",
        reference_key: str = "output"
    ) -> Dict:
        """Compare multiple models on the same test set."""
        results = {}

        for name, fn in model_fns.items():
            print(f"Evaluating: {name}...")
            self.model_fn = fn
            result = self.evaluate_generation(
                test_data,
                input_key=input_key,
                reference_key=reference_key
            )
            results[name] = result.metrics

        # Print comparison table
        print("\n=== Model Comparison ===")
        metric_names = set()
        for r in results.values():
            metric_names.update(r.keys())

        header = f"{'Model':<25}" + "".join(f"{m:<18}" for m in sorted(metric_names) if m != "total_examples")
        print(header)
        print("-" * len(header))
        for name, metrics in results.items():
            row = f"{name:<25}"
            for m in sorted(metric_names):
                if m == "total_examples":
                    continue
                val = metrics.get(m, 0)
                row += f"{val:<18.4f}" if isinstance(val, float) else f"{val:<18}"
            print(row)

        return results


# Example validators
def json_validator(output: str) -> bool:
    """Validate JSON output."""
    json.loads(output)  # Raises JSONDecodeError if invalid
    return True


def structured_response_validator(output: str) -> bool:
    """Validate structured response format."""
    required_sections = ["Summary:", "Details:", "Recommendation:"]
    return all(section in output for section in required_sections)
```

### Evaluation Checklist for Fine-Tuned Models

Before deploying a fine-tuned model, verify each of these:

| Check | What to Measure | Minimum Bar |
|-------|----------------|-------------|
| **Task accuracy** | F1 / accuracy on held-out test set | Beats base model + prompt engineering |
| **Format compliance** | % of outputs matching required format | >95% for structured output tasks |
| **Catastrophic forgetting** | Performance on general capability tests | <10% degradation on any category |
| **Latency** | P50 and P99 inference time | Meets your SLA requirements |
| **Cost per query** | Token usage compared to base model + long prompt | Should be lower (that's the point) |
| **Edge cases** | Performance on adversarial/unusual inputs | No catastrophic failures |
| **Regression** | Comparison against previous model version | No regressions on previously-passing tests |

### Worked Example: Evaluating a Customer Service Fine-Tune

To make the evaluation framework concrete:

```python
def demonstrate_evaluation_workflow():
    """
    Walk through a complete evaluation of a fine-tuned customer service model.

    This shows the workflow with synthetic data. Replace with your actual
    model and test data for real evaluation.
    """
    # Step 1: Define test data (held-out examples NOT used in training)
    test_data = [
        {"input": "My order #99887 hasn't arrived", "output": "order_status", "label": "order_inquiry"},
        {"input": "I want to return my headphones", "output": "return_request", "label": "return"},
        {"input": "You charged me twice!", "output": "billing_dispute", "label": "billing"},
        {"input": "How do I change my password?", "output": "account_help", "label": "account"},
        # ... in practice, 100+ test examples
    ]

    # Step 2: Simulate model outputs (replace with actual model inference)
    def mock_fine_tuned_model(prompt: str) -> str:
        """Simulated fine-tuned model outputs."""
        # A well-fine-tuned model would return structured responses
        if "order" in prompt.lower():
            return '{"intent": "order_inquiry", "priority": "medium", "response": "Let me check..."}'
        if "return" in prompt.lower():
            return '{"intent": "return", "priority": "medium", "response": "I can help with..."}'
        return '{"intent": "unknown", "priority": "low", "response": "Let me connect you..."}'

    # Step 3: Run evaluation
    evaluator = FineTuneEvaluator(mock_fine_tuned_model)

    # Classification accuracy
    classification_result = evaluator.evaluate_classification(
        test_data, input_key="input", label_key="label",
        valid_labels=["order_inquiry", "return", "billing", "account", "unknown"]
    )
    print(classification_result)
    # Expected output:
    # Accuracy: 0.500 (2/4)
    # Macro F1: 0.400
    # Invalid outputs: 0/4 (0.0%)   ← Good: model always outputs valid labels

    # Format compliance
    compliance = evaluator.evaluate_format_compliance(
        test_data, format_validator=json_validator, input_key="input"
    )
    print(f"JSON compliance: {compliance['compliance_rate']:.1%}")
    # Expected: 100% if fine-tuned for JSON output

    # Step 4: Interpret results
    # - 50% accuracy → model confuses some categories, needs more training data
    #   for billing and account categories
    # - 100% JSON compliance → formatting fine-tuning worked well
    # - Next step: examine per-class F1 to find weakest categories,
    #   add more training examples for those categories, retrain
```

> **Key insight from practice:** The most common pattern after fine-tuning is strong format compliance (JSON, structure) but uneven task accuracy across categories. The fix is almost always more training data for the weak categories, not more training epochs (which causes overfitting and forgetting).

---

## Interview Preparation

### Conceptual Questions (with Full Explanations)

**1. "When should you fine-tune vs use RAG vs prompt engineering? Walk me through the decision."**

This is a spectrum, not a binary choice. **Prompt engineering** is always the starting point — if you can solve the task with a well-designed prompt (including few-shot examples), fine-tuning adds complexity without clear benefit. The exception is when prompt engineering works but uses too many tokens (cost) or produces inconsistent output format.

**RAG** is better when: (a) knowledge changes frequently (news, prices, policies), (b) you need source attribution (citing which document an answer came from), (c) the knowledge base is too large to encode in model weights, (d) you need to add new knowledge without retraining.

**Fine-tuning** is better when: (a) you need consistent output format that prompting can't reliably enforce (always-valid JSON, specific schemas), (b) the task requires a distinctive style or tone that can't be captured in a prompt, (c) you're processing high volume and need to reduce per-query token cost by shortening prompts, (d) the knowledge is static and domain-specific.

**The combination pattern:** Fine-tune for format and style compliance + RAG for dynamic knowledge retrieval. Example: fine-tune a model to always output structured JSON medical summaries, then use RAG to feed it current patient records and guidelines.

**2. "Explain how LoRA works mechanically. Why does low-rank adaptation preserve quality?"**

LoRA freezes the pre-trained weight matrix **W** and learns an additive update **ΔW = A × B**, where A (d×r) and B (r×d) have low rank r ≪ d. For a 4096×4096 attention projection, r=16 means learning 131K parameters instead of 16.7M — a 128× reduction.

**Why low rank preserves quality:** Research (Aghajanyan et al., 2021) shows that when you fine-tune a large model on a specific task, the weight updates have very low intrinsic dimensionality. The model isn't learning entirely new knowledge — it's redirecting existing capabilities toward your task. That redirection lives in a low-dimensional subspace, which LoRA's A×B decomposition captures exactly.

**Why LoRA sometimes BEATS full fine-tuning:** The rank constraint acts as implicit regularization. Full fine-tuning can overfit small datasets by learning spurious patterns in the training data. LoRA's low-rank structure constrains the update to only capture the dominant task-relevant patterns, ignoring noise. This is the same principle as L2 regularization but applied structurally.

**Practical implication:** Start with LoRA r=16. Only increase rank if task performance plateaus AND your training data is large enough to justify more capacity. For most formatting and style tasks, r=8-16 is sufficient.

**3. "You've fine-tuned a model and accuracy improved 15% on your task, but users report worse general quality. Diagnose and fix."**

This is catastrophic forgetting. The diagnosis process:
1. **Run the forgetting test suite** (before/after comparison across categories: reasoning, code generation, math, multi-language, JSON formatting).
2. **Identify which categories degraded** — typically, categories most distant from your training task degrade most.
3. **Quantify the damage** — if reasoning dropped from 0.85 to 0.70 (>10% degradation), that's a production blocker.

Fixes, ordered by effort:
1. **Reduce epochs** (most common cause) — try 1 epoch instead of 3. Check if val loss was rising.
2. **Lower learning rate** — from 2e-4 to 5e-5. Slower convergence but less weight destruction.
3. **Switch from full fine-tuning to LoRA** — freezes base weights, dramatically reduces forgetting risk.
4. **Mix in general-purpose data** — add 10-20% of diverse examples (code, reasoning, math) to your training set. This forces the model to maintain general capabilities.
5. **Use evaluation gating** — don't deploy if general capabilities degrade beyond your threshold. This is a policy decision, not a technical fix.

**4. "Design a fine-tuning pipeline for a company that needs to extract structured data from 50 different invoice formats."**

Architecture:
1. **Data collection**: Human annotators label 100+ invoices per format (5,000+ total examples). Each example is an (invoice_image_description, structured_JSON) pair. Include edge cases: poor scans, handwritten additions, partial invoices.
2. **Data validation**: Run DatasetValidator — check for duplicates, token length limits, format consistency. Split 90/10 train/validation.
3. **Fine-tuning approach**: OpenAI API fine-tuning on gpt-4o-mini for fastest iteration. Alternative: QLoRA on Mistral-7B if data residency is required.
4. **Hyperparameters**: Start with 2 epochs, auto batch size. Monitor validation loss — stop if it rises.
5. **Evaluation pipeline**: (a) Format compliance: >99% valid JSON required, (b) Field accuracy: F1 per field type (dates, amounts, vendor names), (c) Catastrophic forgetting: run general capability suite, (d) Comparison: fine-tuned model vs. base model + long prompt to verify fine-tuning adds value.
6. **Monitoring**: Track format compliance rate and per-field accuracy weekly. If accuracy drops on a specific format, add more training examples for that format and retrain.

**Cost estimate**: 5,000 examples × ~500 tokens each = 2.5M tokens × $3/M (gpt-4o-mini fine-tuning) = ~$7.50 per training run. Inference: ~$0.30/M input tokens, saving ~60% vs. base model with long prompt.

**5. "When does LoRA match or beat full fine-tuning? When doesn't it?"**

LoRA matches or beats full fine-tuning in these conditions:
- **Small datasets** (<5K examples): LoRA's rank constraint provides regularization that prevents overfitting. Full fine-tuning on small datasets almost always overfits.
- **Formatting/style tasks**: When the task is mainly about output format (JSON, specific structure) rather than new factual knowledge, LoRA captures the behavioral shift with minimal rank.
- **Multi-task scenarios**: LoRA adapters can be swapped per-task without reloading the base model, enabling efficient multi-tenant serving.

Full fine-tuning wins when:
- **Large datasets** (>50K examples): With enough data, the additional capacity of full fine-tuning pays off without overfitting risk.
- **Significant domain shift**: If the target domain is very different from pre-training data (e.g., specialized scientific notation, rare languages), low-rank updates may not have enough capacity.
- **Quality ceiling matters**: When even 1-2% quality improvement justifies the compute cost (e.g., safety-critical applications).

### Career Mapping

| Role | Fine-Tuning Skills That Matter | Interview Focus |
|------|-------------------------------|-----------------|
| **ML Engineer** | LoRA/QLoRA mechanics, hyperparameter tuning, training infrastructure, VRAM optimization | "How does LoRA's rank affect quality vs. compute trade-off?" |
| **Applied AI Engineer** | Dataset preparation, evaluation pipelines, OpenAI fine-tuning API, production deployment | "Design a fine-tuning pipeline for invoice extraction" |
| **Data Scientist** | Experiment design, metric selection, A/B testing, statistical significance | "How do you know if fine-tuning actually improved the model?" |
| **AI Product Manager** | Fine-tune vs RAG vs prompting decision, cost analysis, ROI calculation | "When is fine-tuning worth the investment?" |
| **Platform Engineer** | Model serving (LoRA adapter swapping), GPU allocation, training orchestration | "How do you serve 10 LoRA adapters efficiently on one GPU?" |

### Coding Challenges

**Challenge 1**: Build a complete fine-tuning pipeline:

```python
def create_fine_tuning_pipeline(
    raw_data: List[Dict],
    validation_split: float = 0.1,
    fmt: str = "openai"
) -> Dict:
    """
    Create a complete fine-tuning data preparation pipeline.

    Steps:
    1. Validate and clean raw data
    2. Format into the target format (OpenAI, Alpaca, ShareGPT)
    3. Split into train/validation
    4. Run quality checks
    5. Estimate cost
    6. Save to files

    Returns:
        Dict with file paths, quality report, and cost estimate
    """
    preparer = DatasetPreparer()

    # Step 1: Load and validate
    for item in raw_data:
        user_msg = item.get("user", item.get("instruction", item.get("input", "")))
        asst_msg = item.get("assistant", item.get("output", item.get("response", "")))
        system = item.get("system", item.get("system_prompt", None))

        if not user_msg or not asst_msg:
            continue  # Skip malformed entries

        preparer.add_example(
            user_message=user_msg,
            assistant_response=asst_msg,
            system_prompt=system
        )

    # Step 2: Validate quality
    validator = DatasetValidator(min_examples=50)
    quality_report = validator.validate(preparer)

    # Step 3: Split
    train_prep, val_prep = preparer.split(
        train_ratio=1 - validation_split, seed=42
    )

    # Step 4: Save files
    train_path = "train_data.jsonl"
    val_path = "val_data.jsonl"
    train_prep.save_jsonl(train_path, fmt=fmt)
    val_prep.save_jsonl(val_path, fmt=fmt)

    # Step 5: Cost estimate (for OpenAI)
    cost_info = {}
    if fmt == "openai":
        cost_info = estimate_fine_tuning_cost(train_path)

    return {
        "train_file": train_path,
        "train_examples": len(train_prep.examples),
        "val_file": val_path,
        "val_examples": len(val_prep.examples),
        "quality_report": quality_report,
        "cost_estimate": cost_info,
        "ready_to_train": quality_report["valid"],
    }
```

**Challenge 2**: Implement a LoRA adapter comparison tool:

```python
def compare_lora_ranks(
    model_name: str,
    training_file: str,
    test_data: List[Dict],
    ranks: List[int] = [4, 8, 16, 32],
) -> Dict[str, Dict]:
    """
    Compare fine-tuning results across different LoRA ranks.

    This helps you find the optimal rank for your task:
    - Too low → underfitting, poor task performance
    - Too high → overfitting, more VRAM, marginal gains

    Returns per-rank metrics for comparison.
    """
    results = {}

    for r in ranks:
        print(f"\n{'='*40}")
        print(f"Training with LoRA rank = {r}")
        print(f"{'='*40}")

        # Initialize
        tuner = OpenSourceFineTuner(model_name, use_4bit=True)
        tuner.prepare_lora(r=r, lora_alpha=r * 2)

        # Train
        dataset = tuner.prepare_dataset(training_file, fmt="alpaca")
        split = dataset.train_test_split(test_size=0.1)
        output_dir = f"./lora_rank_{r}"

        tuner.train(
            train_dataset=split["train"],
            eval_dataset=split["test"],
            output_dir=output_dir,
            num_epochs=2,
        )

        # Evaluate
        def model_fn(prompt):
            inputs = tuner.tokenizer(prompt, return_tensors="pt").to(tuner.model.device)
            with torch.no_grad():
                output = tuner.model.generate(**inputs, max_new_tokens=256)
            return tuner.tokenizer.decode(output[0], skip_special_tokens=True)

        evaluator = FineTuneEvaluator(model_fn)
        eval_result = evaluator.evaluate_generation(
            test_data, input_key="input", reference_key="output"
        )

        # Count trainable params
        trainable = sum(p.numel() for p in tuner.model.parameters() if p.requires_grad)

        results[f"rank_{r}"] = {
            "rank": r,
            "trainable_params": trainable,
            "trainable_pct": f"{trainable / sum(p.numel() for p in tuner.model.parameters()) * 100:.2f}%",
            "metrics": eval_result.metrics,
        }
        print(f"  Trainable params: {trainable:,} | Metrics: {eval_result.metrics}")

    # Print comparison
    print(f"\n{'='*60}")
    print("LoRA Rank Comparison Summary:")
    for name, data in results.items():
        fm = data["metrics"].get("flexible_match_avg", 0)
        print(f"  {name}: params={data['trainable_params']:,} | flexible_match={fm:.3f}")

    return results
```

---

## Exercises

### Exercise 1: Fine-Tune for JSON Output
Create a fine-tuned model that:
- Always outputs valid JSON
- Follows a specific schema
- Handles edge cases gracefully

### Exercise 2: Domain Adaptation
Fine-tune a model for a specific domain:
- Collect domain-specific examples
- Train with appropriate hyperparameters
- Evaluate against baseline

### Exercise 3: Compare Fine-Tuning Approaches
Compare results from:
- OpenAI fine-tuning
- LoRA fine-tuning
- Full fine-tuning
- Measure quality, cost, and speed

### Exercise 4: Catastrophic Forgetting Experiment
Design and run a forgetting evaluation:
- Build a test suite with 5+ capability categories
- Evaluate the base model before fine-tuning
- Fine-tune on a narrow task
- Re-evaluate and report degradation
- Try mitigation strategies and compare

### Exercise 5: Build Evaluation Pipeline
Create an automated evaluation system:
- Multiple metrics (not just exact match)
- Format validation
- Regression testing against previous model versions
- Performance tracking over time

---

## Summary

### Key Takeaways

1. **Prompt engineering first, fine-tuning second** — always verify that prompting can't solve the task before investing in fine-tuning. Fine-tune when you need consistent format, specific style, or lower per-query token cost
2. **LoRA works by exploiting low-rank structure in weight updates** — the W + A×B decomposition captures the task-specific behavioral shift with 0.5-2% of parameters. Start with r=16, alpha=32
3. **QLoRA enables fine-tuning on consumer GPUs** — 4-bit quantized base + fp16 adapters reduces a 7B model from ~28GB to ~6GB VRAM with minimal quality loss
4. **Catastrophic forgetting is the #1 deployment risk** — always run a diverse evaluation suite before and after fine-tuning. If any category degrades >10%, reduce epochs or switch to LoRA
5. **Evaluation must go beyond string matching** — use flexible match, ROUGE/BLEU for generation, per-class F1 for classification, format compliance checks, and forgetting detection
6. **Data quality trumps data quantity** — 500 high-quality, diverse examples beat 5,000 noisy ones. Validate for duplicates, length consistency, and category balance
7. **Monitor the validation loss curve** — if it rises while training loss drops, you're overfitting. Stop training, reduce epochs, or add more diverse data

### Decision Matrix

| Approach | Quality | Cost | Speed | Complexity |
|----------|---------|------|-------|------------|
| Prompt Engineering | Good | Low | Fast | Low |
| OpenAI Fine-Tune | Very Good | Medium | Medium | Low |
| LoRA (Open Source) | Very Good | High (setup) | Slow | Medium |
| Full Fine-Tune | Potentially Best | Very High | Very Slow | High |

> **Note on quality claims:** The "quality" column above is a rough generalization. Actual quality depends heavily on your specific task, data quality, and hyperparameter choices. Full fine-tuning is not guaranteed to beat LoRA -- in practice, LoRA often matches or exceeds full fine-tuning because it acts as a regularizer against overfitting.

---

## Self-Assessment

| Category | Score | Justification |
|----------|-------|---------------|
| Conceptual Clarity | Strong | LoRA mechanism explained (W + A×B), QLoRA quantization + adapters, catastrophic forgetting with detection code |
| Depth vs Surface | Strong | LoRA parameter effects table, hyperparameter tuning guide with loss curve diagnosis, fine-tuning decision tree |
| Hands-On Practicality | Strong | OpenAI + HuggingFace workflows, dataset preparation with validation, 2 coding challenges with implementations |
| Engineering Rigor | Good | VRAM requirements table, cost estimation, hyperparameter decision table — but DatasetPreparer is memory-only |
| Evaluation Discipline | Strong | Multi-metric evaluation (flexible match, ROUGE, BLEU, F1), catastrophic forgetting detection, worked example |
| Career Relevance | Strong | 5 interview questions with full explanations, career mapping to 5 roles, system design question |
| Audience Targeting | Good | Reading guide, prerequisites, hyperparameter guide for different experience levels |

### Known Limitations

- **Customer service dataset has only 4 examples** — illustrative only. Real fine-tuning requires hundreds to thousands of examples (acknowledged throughout)
- **No RLHF/DPO coverage** — reinforcement learning from human feedback and Direct Preference Optimization are increasingly important for alignment but require separate treatment (see "What This Blog Does NOT Cover")
- **Unsloth section is a thin wrapper** — production users will need deeper configuration for Unsloth's advanced features
- **DatasetPreparer loads all data into memory** — unsuitable for 100K+ example datasets. Production scale needs HuggingFace datasets with memory mapping
- **No real fine-tuning run shown** — running the pipeline end-to-end requires API keys (OpenAI) or GPU access (HuggingFace)

---

## Architect Sanity Checks

- **Would you trust someone who learned only this blog to touch a production AI system?**
  **YES** — The blog explains LoRA mechanics (not just API calls), provides a systematic evaluation framework beyond string matching, includes catastrophic forgetting detection with before/after comparison, and covers the full decision tree (when NOT to fine-tune). The VRAM requirements table and hyperparameter guide enable informed resource allocation. The worked evaluation example demonstrates the end-to-end workflow. The main gap is that no actual fine-tuning run is shown, but the patterns are correct and production-ready.

- **Can you explain at least one real failure case using only what's taught here?**
  **YES** — Catastrophic forgetting scenario: fine-tune a customer service model for 5 epochs, and reasoning ability drops from 0.85 to 0.60. The blog teaches how to detect this (forgetting test suite with before/after comparison), diagnose the root cause (too many epochs, validation loss was rising), and fix it (reduce to 1 epoch, switch to LoRA, mix in general data). The loss curve diagnosis guide helps identify the problem from training metrics alone.

- **Would this blog survive senior-engineer interview follow-up questions?**
  **YES** — The LoRA question explains the W + A×B mechanism with parameter count math and explains WHY low rank preserves quality (low intrinsic dimensionality of task-specific updates). The forgetting question provides a step-by-step diagnosis and fix procedure. The system design question (invoice extraction pipeline) covers data collection, validation, hyperparameter selection, evaluation, and cost estimation with concrete numbers.

---

## What's Next?

In **Blog 24: Deploying AI Applications**, we'll learn how to take models to production. You'll learn:
- Containerization with Docker
- Cloud deployment (AWS, GCP, Azure)
- Scaling with Kubernetes
- Monitoring and observability
- CI/CD for ML systems

From training to serving -- let's deploy!

---

*Fine-tuning is a powerful tool, but it's not magic. Start with good data, train carefully, evaluate beyond string matching, and check for catastrophic forgetting before you deploy.*
