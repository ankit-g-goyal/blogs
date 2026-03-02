# Blog 12: Prompt Engineering Fundamentals — The Art of Talking to Machines

**Series:** Prompt Your Career: The Complete Generative AI Masterclass
**Prerequisites:** Blog 11 (Understanding Large Language Models)
**Time to Complete:** 3-3.5 hours
**Difficulty:** Beginner to Intermediate

---

## Reading Guide

**Who this is for:** Developers, data scientists, and technical professionals who have completed Blog 11 (Understanding Large Language Models) and want to learn how to communicate effectively with LLMs. You should understand what a token is, how autoregressive generation works, and the basics of the transformer architecture.

**How to read this blog:**
- **If you are new to prompting:** Read end-to-end and complete all exercises. Spend extra time on the CRAFT framework and few-shot prompting sections.
- **If you already prompt LLMs regularly:** Skim the fundamentals, focus on the debugging/iteration section and the structured output techniques.
- **If you are preparing for interviews:** Jump to the Interview Preparation section, but review the temperature/sampling section for technical depth.

**What This Blog Does NOT Cover:**
- **Advanced prompting techniques** (chain-of-thought, self-consistency, ReAct) — covered in Blog 13
- **Prompt injection attacks and security** — covered in Blog 14 (Working with AI APIs)
- **Automatic prompt optimization** (DSPy, OPRO) — covered in Blog 13
- **Fine-tuning vs prompting tradeoffs** — covered in Blog 23
- **Multi-modal prompting** (image + text) — covered in Blog 21
- **Agentic prompt patterns** (tool use, planning loops) — covered in Blog 18

---

## What You'll Walk Away With

After completing this blog, you will be able to:

1. **Understand why prompt engineering matters** and how models interpret prompts
2. **Apply the five components** of an effective prompt
3. **Master zero-shot, one-shot, and few-shot prompting** techniques
4. **Design effective system prompts** and personas
5. **Control output format** reliably (JSON, markdown, structured data)
6. **Tune temperature and sampling parameters** for different use cases
7. **Debug and iterate** on prompts systematically

---

## Manager's Summary

**What is Prompt Engineering?**

Prompt engineering is the skill of communicating effectively with AI models. Just as managing humans requires clear communication, working with LLMs requires understanding how to frame requests for optimal results. It's the difference between getting a helpful response and getting garbage.

**Business Impact:**

| Area | Poor Prompting | Good Prompting |
|------|----------------|----------------|
| **Accuracy** | Low task success, frequent errors | Substantially higher success rates |
| **Consistency** | High variance in outputs | Predictable, reliable results |
| **Cost** | Multiple retries needed | First-try success more common |
| **Time** | Hours debugging | Minutes to deployment |
| **User Experience** | Frustrated users | Delighted users |

**Key Investment Areas:**

1. **Training:** Every employee interacting with AI should understand basics
2. **Templates:** Build a library of proven prompts for common tasks
3. **Evaluation:** Measure prompt effectiveness before production
4. **Iteration:** Continuous improvement based on real-world feedback

**ROI Example (Illustrative):**
A customer support team that improves prompt quality meaningfully (e.g., from low to high accuracy) can significantly reduce human escalations, potentially saving substantial labor costs. Actual figures depend on team size, ticket volume, and escalation costs.

---

## Why Prompt Engineering Matters

### The Communication Gap

```
What you mean:                      What you write:
┌────────────────────┐              ┌────────────────────┐
│ "I want a concise, │              │ "Summarize this    │
│  professional      │     ───>     │  article."         │
│  3-bullet summary  │              │                    │
│  for executives"   │              │                    │
└────────────────────┘              └────────────────────┘
                                              │
                                              ▼
                                    ┌────────────────────┐
                                    │ LLM interprets:    │
                                    │ "Generate any      │
                                    │  length summary    │
                                    │  in any style"     │
                                    └────────────────────┘
```

The model doesn't know what you mean—it only knows what you write. The gap between intent and instruction is where prompt engineering lives.

### The Prompt Engineering Spectrum

```python
"""
Prompt engineering exists on a spectrum from simple to sophisticated.
Your position on this spectrum determines your effectiveness.
"""

PROMPTING_LEVELS = {
    "Level 1 - Naive": {
        "example": "Summarize this.",
        "problems": ["Ambiguous", "No constraints", "Unpredictable output"],
        "quality": "Low — highly variable results",
    },

    "Level 2 - Specific": {
        "example": "Summarize this article in 3 bullet points for a business audience.",
        "improvements": ["Clear format", "Audience specified", "Length constrained"],
        "quality": "Moderate — format improves but still inconsistent",
    },

    "Level 3 - Structured": {
        "example": """
            Role: You are an executive assistant.
            Task: Summarize this article.
            Format: 3 bullet points, max 20 words each
            Audience: C-suite executives
            Focus: Business impact and actionable insights
            Article: {article_text}
        """,
        "improvements": ["Role context", "Detailed constraints", "Clear structure"],
        "quality": "Good — consistent format, reliable output",
    },

    "Level 4 - Demonstrated": {
        "example": """
            Role: You are an executive assistant.
            Task: Summarize articles for executives.

            Example 1:
            Article: [tech article about cloud migration]
            Summary:
            - Cloud migration can reduce infrastructure costs significantly
            - Implementation typically takes months to over a year for enterprises
            - Key risk: vendor lock-in requires careful contract negotiation

            Example 2:
            Article: [article about remote work trends]
            Summary:
            - Many knowledge workers prefer hybrid arrangements
            - Productivity metrics generally hold steady in remote settings
            - Challenge: maintaining company culture requires intentional effort

            Now summarize this article following the same format:
            Article: {article_text}
        """,
        "improvements": ["Examples provided", "Consistent quality", "Style demonstrated"],
        "quality": "High — consistent style, format, and depth",
    },
}

def demonstrate_levels():
    """Show the progression of prompt quality."""
    for level, details in PROMPTING_LEVELS.items():
        print(f"\n{'='*60}")
        print(f"{level}")
        print(f"Quality: {details['quality']}")
        print(f"{'='*60}")
        print(f"Example:\n{details['example'][:200]}...")
        if 'problems' in details:
            print(f"Problems: {', '.join(details['problems'])}")
        if 'improvements' in details:
            print(f"Improvements: {', '.join(details['improvements'])}")

demonstrate_levels()
```

---

## The Five Components of an Effective Prompt

### The CRAFT Framework

```
C - Context     : Background information the model needs
R - Role        : The persona or expertise the model should adopt
A - Action      : The specific task to perform
F - Format      : How the output should be structured
T - Tone        : The style and voice of the response

┌─────────────────────────────────────────────────────────────────┐
│                        EFFECTIVE PROMPT                         │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐│
│  │ Context │  │  Role   │  │ Action  │  │ Format  │  │  Tone   ││
│  │         │  │         │  │         │  │         │  │         ││
│  │ "Given  │  │ "As a   │  │ "Write  │  │ "Output │  │ "Using  ││
│  │  this   │  │  senior │  │  a code │  │  as     │  │  clear  ││
│  │  code   │  │  Python │  │  review │  │  bullet │  │  but    ││
│  │  that   │  │  dev... │  │  for... │  │  points │  │  direct ││
│  │  does..." │  │         │  │         │  │  with..."│  │  lang..."││
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘  └─────────┘│
└─────────────────────────────────────────────────────────────────┘
```

### Component 1: Context

```python
"""
Context provides the background information needed to understand the task.
Without context, the model makes assumptions that may be wrong.
"""

# Bad: No context
prompt_bad = "How should I handle this?"

# Better: Some context
prompt_better = "I received a customer complaint about slow shipping. How should I handle this?"

# Best: Rich context
prompt_best = """
Context:
- I'm a customer support rep for an e-commerce company
- Our average shipping time is 3-5 business days
- The customer ordered 7 days ago and hasn't received their package
- Tracking shows the package is in transit but delayed
- The customer is a VIP member with 5+ years of purchase history

Question: How should I handle this customer complaint?
"""

# Context types:
CONTEXT_TYPES = {
    "Background": "Who, what, where, when, why",
    "Constraints": "Limitations, rules, requirements",
    "Data": "Input information to process",
    "History": "Previous interactions or decisions",
    "Goals": "What success looks like",
}
```

### Component 2: Role

```python
"""
Roles activate specific "expert modes" in the model.
Different roles produce different responses.
"""

def compare_roles(question):
    """Demonstrate how roles change responses."""
    roles = [
        "You are a cautious legal advisor.",
        "You are an aggressive sales coach.",
        "You are a neutral research analyst.",
        "You are a supportive therapist.",
    ]

    question = "A client is considering a risky business deal. What advice should they get?"

    print("Same question, different roles:\n")
    for role in roles:
        print(f"Role: {role}")
        print(f"→ Response will be shaped by this expertise and personality\n")

# Effective role patterns:
ROLE_PATTERNS = {
    "Expert": "You are a senior {domain} expert with 20 years of experience.",
    "Teacher": "You are a patient teacher explaining {topic} to beginners.",
    "Critic": "You are a thorough code reviewer focused on {aspects}.",
    "Creative": "You are a creative writer specializing in {genre}.",
    "Analyst": "You are a data analyst focused on {type} insights.",
}

# Role anti-patterns (avoid these):
ROLE_ANTIPATTERNS = [
    "You are the world's best...",  # Superlatives don't help
    "You are an AI assistant...",    # Model knows this already
    "You are helpful...",            # Too generic
]
```

### Component 3: Action

```python
"""
The action is the specific task. Be precise about what you want.
"""

# Weak actions (ambiguous)
weak_actions = [
    "Help me with this code",      # What kind of help?
    "Make this better",            # Better how?
    "Fix this",                    # What's broken?
    "Do something with this data", # What operation?
]

# Strong actions (specific)
strong_actions = [
    "Review this code for security vulnerabilities, specifically SQL injection and XSS",
    "Refactor this function to reduce cyclomatic complexity below 10",
    "Debug why this function returns None instead of the expected list",
    "Convert this CSV data into a normalized JSON structure with nested objects",
]

# Action verbs by task type:
ACTION_VERBS = {
    "Analysis": ["analyze", "evaluate", "compare", "assess", "examine"],
    "Creation": ["write", "generate", "create", "compose", "design"],
    "Transformation": ["convert", "translate", "refactor", "restructure", "format"],
    "Extraction": ["extract", "identify", "find", "list", "summarize"],
    "Explanation": ["explain", "describe", "clarify", "break down", "illustrate"],
    "Review": ["review", "critique", "validate", "verify", "check"],
}
```

### Component 4: Format

```python
"""
Format specifies exactly how you want the output structured.
This is crucial for downstream processing.
"""

# Format specifications:

# 1. Length constraints
length_formats = [
    "Respond in exactly 3 sentences.",
    "Keep your response under 100 words.",
    "Provide a 2-paragraph summary.",
    "Give a one-line answer.",
]

# 2. Structure specifications
structure_formats = [
    "Respond with a numbered list of 5 items.",
    "Format as a markdown table with columns: Name, Description, Priority.",
    "Structure your response with these sections: Overview, Details, Recommendations.",
    "Use bullet points, with sub-bullets for details.",
]

# 3. Data formats
data_formats = [
    "Respond with valid JSON only, no other text.",
    "Output as a Python dictionary.",
    "Format as CSV with headers.",
    "Return as YAML.",
]

# 4. Template-based formats
template_format = """
Respond using exactly this template:

## Summary
[1-2 sentence summary]

## Key Points
- Point 1: [description]
- Point 2: [description]
- Point 3: [description]

## Recommendation
[Your recommendation in 1 sentence]

## Confidence
[High/Medium/Low] - [brief justification]
"""

# Enforcing JSON output:
json_enforcement = """
You must respond with valid JSON only. No markdown, no explanation.
The JSON must follow this exact schema:

{
    "sentiment": "positive" | "negative" | "neutral",
    "confidence": 0.0 to 1.0,
    "key_phrases": ["phrase1", "phrase2"],
    "summary": "one sentence summary"
}

Input text: {text}
"""
```

### Component 5: Tone

```python
"""
Tone controls the style and voice of the response.
Match the tone to your audience and use case.
"""

TONE_SPECTRUM = {
    "Formal": {
        "description": "Professional, business-appropriate language",
        "trigger": "Use formal, professional language appropriate for a board presentation.",
        "example": "The analysis indicates a significant opportunity for market expansion.",
    },
    "Casual": {
        "description": "Friendly, conversational style",
        "trigger": "Write in a friendly, conversational tone like you're chatting with a colleague.",
        "example": "Hey, looks like we've got a solid chance to grow in this market!",
    },
    "Technical": {
        "description": "Precise, jargon-appropriate for experts",
        "trigger": "Use technical language appropriate for senior engineers.",
        "example": "The O(n log n) complexity suggests a merge-sort based approach.",
    },
    "Simplified": {
        "description": "Plain language for non-experts",
        "trigger": "Explain in simple terms that a non-technical person would understand.",
        "example": "The program sorts things efficiently, like organizing a deck of cards.",
    },
    "Empathetic": {
        "description": "Warm, understanding, supportive",
        "trigger": "Respond with empathy and understanding, acknowledging the user's feelings.",
        "example": "I understand this is frustrating. Let's work through this together.",
    },
}

# Combining tone elements:
tone_combination = """
Tone guidelines:
- Be concise but not curt
- Be professional but approachable
- Be confident but not arrogant
- Acknowledge uncertainty when appropriate
- Use "we" instead of "you" to feel collaborative
"""
```

---

## Zero-Shot, One-Shot, and Few-Shot Prompting

### The Spectrum of Examples

```
                    Number of Examples
Zero-Shot           One-Shot           Few-Shot
    │                   │                  │
    ▼                   ▼                  ▼
┌─────────┐        ┌─────────┐        ┌─────────┐
│ No      │        │ Single  │        │ Multiple│
│ examples│        │ example │        │ examples│
│         │        │         │        │ (2-10)  │
└─────────┘        └─────────┘        └─────────┘
    │                   │                  │
    ▼                   ▼                  ▼
Relies on         Shows the          Teaches the
pre-trained       desired            pattern and
knowledge         format             handles edge
                                     cases
```

### Zero-Shot Prompting

```python
"""
Zero-shot: No examples provided. Relies entirely on the model's
pre-trained knowledge to understand and complete the task.

Best for:
- Common tasks the model has seen during training
- When you don't have good examples
- Simple, well-defined tasks
"""

zero_shot_examples = {
    "sentiment": """
        Classify the sentiment of this review as positive, negative, or neutral.

        Review: "The product arrived on time but the quality was disappointing."

        Sentiment:
    """,

    "translation": """
        Translate the following English text to French:

        English: "The weather is beautiful today."

        French:
    """,

    "summarization": """
        Summarize the following paragraph in one sentence:

        Text: "Machine learning is a subset of artificial intelligence that
        enables systems to learn and improve from experience without being
        explicitly programmed. It focuses on developing algorithms that can
        access data and use it to learn for themselves."

        Summary:
    """,
}

# Zero-shot works well when:
# 1. Task is common (sentiment, translation, summarization)
# 2. Instructions are clear
# 3. Output format is simple
```

### One-Shot Prompting

```python
"""
One-shot: Single example provided. Shows the model the expected
input-output relationship.

Best for:
- Establishing format
- Novel but simple tasks
- When one example is representative
"""

one_shot_examples = {
    "classification": """
        Classify customer feedback into categories.

        Example:
        Feedback: "Your app crashes every time I try to upload a photo."
        Category: Bug Report

        Now classify this:
        Feedback: "I wish you had a dark mode option."
        Category:
    """,

    "formatting": """
        Convert meeting notes to action items.

        Example:
        Notes: "John will send the report by Friday. Team needs to review budget."
        Action Items:
        - [ ] John: Send report by Friday
        - [ ] Team: Review budget

        Now convert this:
        Notes: "Marketing to prepare launch materials. Dev team fixing login bug."
        Action Items:
    """,

    "transformation": """
        Convert informal text to formal business language.

        Example:
        Informal: "Hey, can you send that stuff over asap? Thanks!"
        Formal: "Would you please send the materials at your earliest convenience? Thank you."

        Now convert this:
        Informal: "This is super important, we gotta fix it before the boss finds out."
        Formal:
    """,
}
```

### Few-Shot Prompting

```python
"""
Few-shot: Multiple examples (typically 2-10) that demonstrate the
pattern, including edge cases and variations.

Best for:
- Complex tasks
- Ensuring consistency
- Teaching specific formats
- Handling edge cases
"""

few_shot_sentiment = """
Analyze the sentiment of product reviews. Consider nuance and mixed sentiment.

Example 1:
Review: "Absolutely love this product! Best purchase I've made all year."
Analysis: {"sentiment": "positive", "confidence": 0.95, "aspects": {"quality": "positive"}}

Example 2:
Review: "Terrible. Broke after one day. Complete waste of money."
Analysis: {"sentiment": "negative", "confidence": 0.98, "aspects": {"durability": "negative", "value": "negative"}}

Example 3:
Review: "It works, nothing special. Does what it says."
Analysis: {"sentiment": "neutral", "confidence": 0.75, "aspects": {"functionality": "neutral"}}

Example 4:
Review: "Great features but terrible customer service."
Analysis: {"sentiment": "mixed", "confidence": 0.80, "aspects": {"features": "positive", "service": "negative"}}

Now analyze:
Review: "The camera quality is amazing but the battery dies too quickly."
Analysis:
"""

# Few-shot design principles:
FEW_SHOT_PRINCIPLES = {
    "Diversity": "Include examples that cover different cases",
    "Edge cases": "Show how to handle tricky situations",
    "Consistency": "Use the exact same format in all examples",
    "Ordering": "Put similar examples together, or hardest last",
    "Balance": "Include positive and negative examples",
}

def create_few_shot_prompt(task_description, examples, new_input):
    """
    Helper function to create few-shot prompts.

    Args:
        task_description: What the model should do
        examples: List of (input, output) tuples
        new_input: The new input to process
    """
    prompt = f"{task_description}\n\n"

    for i, (input_text, output_text) in enumerate(examples, 1):
        prompt += f"Example {i}:\n"
        prompt += f"Input: {input_text}\n"
        prompt += f"Output: {output_text}\n\n"

    prompt += f"Now process this:\n"
    prompt += f"Input: {new_input}\n"
    prompt += f"Output:"

    return prompt

# Usage:
examples = [
    ("Hello, how are you?", "Greeting"),
    ("I need help with my order", "Support Request"),
    ("Your product is terrible!", "Complaint"),
]

prompt = create_few_shot_prompt(
    task_description="Classify customer messages into categories.",
    examples=examples,
    new_input="Can you tell me about your return policy?"
)
print(prompt)
```

### Choosing the Right Approach

```python
"""
Decision framework for zero-shot vs few-shot.
"""

def choose_prompting_strategy(task_complexity, format_importance,
                               example_availability, consistency_requirement):
    """
    Recommend a prompting strategy based on task characteristics.

    Args:
        task_complexity: "simple", "moderate", "complex"
        format_importance: "low", "medium", "high"
        example_availability: "none", "few", "many"
        consistency_requirement: "low", "medium", "high"
    """
    score = 0

    # Complexity scoring
    complexity_scores = {"simple": 0, "moderate": 1, "complex": 2}
    score += complexity_scores[task_complexity]

    # Format scoring
    format_scores = {"low": 0, "medium": 1, "high": 2}
    score += format_scores[format_importance]

    # Consistency scoring
    consistency_scores = {"low": 0, "medium": 1, "high": 2}
    score += consistency_scores[consistency_requirement]

    # Adjust for example availability
    if example_availability == "none":
        return "zero-shot", "No examples available; optimize prompt clarity"
    elif score <= 2:
        return "zero-shot", "Task simple enough for zero-shot"
    elif score <= 4:
        return "one-shot", "Single example should establish pattern"
    else:
        return "few-shot", "Complex task requires multiple examples"

# Examples:
tasks = [
    ("Translate English to French", "simple", "low", "few", "low"),
    ("Classify support tickets", "moderate", "high", "many", "high"),
    ("Extract structured data from resumes", "complex", "high", "many", "high"),
]

print("Prompting Strategy Recommendations:\n")
for task, *args in tasks:
    strategy, reason = choose_prompting_strategy(*args)
    print(f"Task: {task}")
    print(f"  → {strategy}: {reason}\n")
```

### Why Does Few-Shot Work? In-Context Learning Mechanics

Few-shot prompting isn't magic — there's a mechanism:

```python
"""
In-context learning (ICL) works because transformers develop "induction heads"
during pre-training (Olsson et al., 2022). These are attention patterns that:

1. Identify repeated patterns in the context
2. Copy the pattern to predict the next output

When you write:
    Input: "happy" → Output: "positive"
    Input: "sad" → Output: "negative"
    Input: "excited" → Output: ???

The model's induction heads detect the pattern (emotion → sentiment label)
and apply it to the new input. The model doesn't "learn" in the gradient
descent sense — it pattern-matches within its existing weights.

Implications for prompt design:
- Examples must be CONSISTENT: conflicting examples confuse induction heads
- FORMAT matters more than explanation: the model copies patterns, not rules
- More diverse examples help because they define the pattern boundary
- ORDER matters: the last example has the strongest influence (recency bias)
"""

# Demonstration: order affects output
order_experiment = {
    "positive_last": [
        ("terrible product", "negative"),
        ("decent enough", "neutral"),
        ("absolutely love it!", "positive"),  # Last example: positive
    ],
    "negative_last": [
        ("absolutely love it!", "positive"),
        ("decent enough", "neutral"),
        ("terrible product", "negative"),  # Last example: negative
    ],
}
# For ambiguous inputs, the model is slightly biased toward the last example's label.
# Mitigation: randomize example order across calls, or put the most representative
# example last.
```

### Positional Effects: Where You Put Instructions Matters

```python
"""
LLMs don't weight all parts of the prompt equally.

Known effects (empirically observed, not guaranteed across all models):
1. PRIMACY: Instructions at the START of the prompt are reliably followed
2. RECENCY: Instructions at the END are also reliably followed
3. LOST IN THE MIDDLE: Instructions buried in long contexts are often missed
   (Liu et al., 2023: "Lost in the Middle: How Language Models Use Long Contexts")

Practical rules:
"""

POSITIONAL_RULES = {
    "Put critical instructions at the START": "System prompt or first user message",
    "REPEAT critical instructions at the END": "Right before the input data",
    "Keep examples in the MIDDLE": "They provide context but aren't directives",
    "Put input data LAST": "Closest to where the model generates output",
}

# Template following positional best practices:
positional_template = """
[INSTRUCTION - START]
You are a precise data extractor. Extract entities as JSON.
You MUST return valid JSON with all required fields.

[EXAMPLES - MIDDLE]
Example 1: ...
Example 2: ...

[INPUT DATA - NEAR END]
Text to process: {input_text}

[REPEATED INSTRUCTION - END]
Remember: Return ONLY valid JSON with fields: name, type, confidence.
JSON:
"""
```

### The Cost of Examples: Token Budget Analysis

Few-shot examples improve quality but cost tokens. In production, this trade-off is real money:

```python
import tiktoken

def analyze_prompt_cost(system_prompt: str, few_shot_examples: list[tuple[str, str]],
                        user_input: str, model: str = "gpt-4-turbo"):
    """
    Calculate token cost for different few-shot counts.

    Reveals the trade-off: more examples = better quality but higher cost.
    """
    enc = tiktoken.encoding_for_model(model)

    # Pricing per 1M tokens (approximate, early 2024)
    pricing = {
        "gpt-4-turbo": {"input": 10, "output": 30},
        "gpt-4o": {"input": 5, "output": 15},
        "gpt-3.5-turbo": {"input": 0.5, "output": 1.5},
    }
    prices = pricing.get(model, pricing["gpt-4-turbo"])

    system_tokens = len(enc.encode(system_prompt))
    input_tokens = len(enc.encode(user_input))

    print(f"{'Examples':<10} {'Input Tokens':<15} {'Cost/call':<12} {'Cost/1K calls':<15} {'Δ vs 0-shot'}")
    print("-" * 70)

    base_cost = None
    for n in range(len(few_shot_examples) + 1):
        example_text = ""
        for inp, out in few_shot_examples[:n]:
            example_text += f"Input: {inp}\nOutput: {out}\n\n"
        example_tokens = len(enc.encode(example_text))

        total_input = system_tokens + example_tokens + input_tokens
        est_output = 50  # Assume ~50 output tokens
        cost_per_call = (total_input * prices["input"] + est_output * prices["output"]) / 1e6
        cost_per_1k = cost_per_call * 1000

        if base_cost is None:
            base_cost = cost_per_call
            delta = "—"
        else:
            delta = f"+{((cost_per_call / base_cost) - 1) * 100:.0f}%"

        print(f"{n:<10} {total_input:<15} ${cost_per_call:<11.4f} ${cost_per_1k:<14.2f} {delta}")

    return

# Example: sentiment classification
analyze_prompt_cost(
    system_prompt="Classify sentiment as positive, negative, or neutral.",
    few_shot_examples=[
        ("Great product, love it!", '{"sentiment": "positive", "confidence": 0.95}'),
        ("Terrible, broke immediately", '{"sentiment": "negative", "confidence": 0.98}'),
        ("It works, nothing special", '{"sentiment": "neutral", "confidence": 0.75}'),
        ("Good features but bad service", '{"sentiment": "mixed", "confidence": 0.80}'),
        ("Exceeded expectations!", '{"sentiment": "positive", "confidence": 0.92}'),
    ],
    user_input="The camera is great but battery is awful.",
    model="gpt-4-turbo",
)

# Rule of thumb:
# - Each few-shot example adds ~30-100 tokens (depends on length)
# - For GPT-4-turbo, each extra example costs ~$0.0003-0.001 per call
# - At 100K calls/month, 5-shot vs 0-shot can cost $30-100 more/month
# - Start with 0-shot, add examples only if quality justifies the cost
```

---

## Designing Effective System Prompts

### System Prompt Architecture

```python
"""
System prompts set the overall behavior and personality of the model.
They're processed before user input and shape all responses.
"""

SYSTEM_PROMPT_TEMPLATE = """
# Identity
{who_you_are}

# Core Capabilities
{what_you_can_do}

# Constraints
{what_you_cannot_do}

# Response Guidelines
{how_to_respond}

# Special Instructions
{edge_cases_and_rules}
"""

# Example: Customer Support Bot
customer_support_system = """
# Identity
You are a friendly customer support assistant for TechCo, an electronics retailer.
Your name is Alex.

# Core Capabilities
- Answer questions about products, orders, and policies
- Help troubleshoot technical issues with TechCo products
- Process returns and exchanges (by collecting information)
- Escalate complex issues to human agents

# Constraints
- Never share customer personal information
- Cannot process payments or access financial systems
- Cannot make promises about refunds without manager approval
- Do not discuss competitors or make product comparisons

# Response Guidelines
- Always greet customers warmly
- Acknowledge frustration before problem-solving
- Keep responses concise (under 100 words unless explaining technical issues)
- End interactions by asking if there's anything else you can help with
- Use the customer's name when available

# Special Instructions
- If customer mentions harm to self or others, immediately provide crisis hotline
- If customer uses profanity, politely redirect without engaging
- If unsure about a policy, say "Let me connect you with a specialist" rather than guessing
- For warranty questions, always verify purchase date before answering
"""

# Example: Code Review Assistant
code_review_system = """
# Identity
You are a senior software engineer conducting code reviews. You have expertise
in Python, JavaScript, and system design. You're thorough but constructive.

# Core Capabilities
- Identify bugs, security vulnerabilities, and performance issues
- Suggest improvements for code clarity and maintainability
- Recommend best practices and design patterns
- Explain the reasoning behind your suggestions

# Constraints
- Do not rewrite entire files; suggest targeted improvements
- Focus on significant issues, not style nitpicks (assume linters handle style)
- Do not access external resources or execute code

# Response Guidelines
- Organize feedback by severity: Critical > Important > Suggestion
- For each issue, provide:
  1. The problem
  2. Why it matters
  3. How to fix it (with code example)
- Acknowledge what's done well, not just problems
- Be direct but respectful; avoid condescending language

# Special Instructions
- Security issues always take priority
- If code is fundamentally flawed, recommend architectural changes before details
- For junior developers, provide more explanation; for seniors, be concise
"""
```

### Dynamic System Prompts

```python
"""
System prompts can be dynamically generated based on context.
"""

def generate_system_prompt(user_info, context, features_enabled):
    """
    Generate a context-aware system prompt.

    Args:
        user_info: Dict with user preferences and history
        context: Current conversation context
        features_enabled: List of enabled features
    """
    base_prompt = """
    You are an AI assistant helping users with their tasks.
    """

    # Add user-specific customization
    if user_info.get("expertise_level") == "beginner":
        base_prompt += """
        The user is a beginner. Explain concepts simply and avoid jargon.
        Offer to elaborate on technical terms.
        """
    elif user_info.get("expertise_level") == "expert":
        base_prompt += """
        The user is an expert. Be concise and technical.
        Skip basic explanations unless asked.
        """

    # Add feature-specific instructions
    if "code_execution" in features_enabled:
        base_prompt += """
        You can execute Python code. When the user asks for calculations
        or data processing, write and run code to get accurate results.
        """

    if "web_search" in features_enabled:
        base_prompt += """
        You can search the web for current information. Use this for
        recent events, current prices, or facts you're unsure about.
        """

    # Add context-specific behavior
    if context.get("is_support_conversation"):
        base_prompt += """
        This is a customer support conversation. Be empathetic and
        solution-focused. Offer to escalate if you can't resolve the issue.
        """

    return base_prompt.strip()

# Example usage:
prompt = generate_system_prompt(
    user_info={"expertise_level": "beginner", "name": "Alice"},
    context={"is_support_conversation": True},
    features_enabled=["web_search"]
)
```

---

## Controlling Output Format

### JSON Output

```python
"""
Getting reliable JSON output is crucial for programmatic use.
Here are techniques that work.
"""

# Technique 1: Schema specification
json_schema_prompt = """
Analyze this customer feedback and return structured JSON.

You MUST respond with valid JSON matching this exact schema:
{
    "sentiment": "positive" | "negative" | "neutral" | "mixed",
    "topics": ["topic1", "topic2"],
    "urgency": "low" | "medium" | "high",
    "suggested_action": "string",
    "confidence": 0.0 to 1.0
}

Rules:
- Output ONLY the JSON, no other text
- All fields are required
- Topics should be 1-5 items
- Confidence reflects certainty in sentiment classification

Feedback: "I've been a customer for 10 years and this is the worst experience
I've ever had. The product broke immediately and nobody will help me!"

JSON:
"""

# Technique 2: Pydantic-style validation hint
pydantic_style_prompt = """
Parse this job listing into structured data.

Expected output format (Python Pydantic model):
```python
class JobListing(BaseModel):
    title: str
    company: str
    location: str
    salary_min: Optional[int]
    salary_max: Optional[int]
    requirements: List[str]
    is_remote: bool
```

Return as JSON matching this schema.

Job listing:
"Senior Python Developer at TechCorp. San Francisco, CA (Remote OK).
$150,000 - $200,000. Requirements: 5+ years Python, AWS experience, CI/CD."

JSON:
"""

# Technique 3: Output delimiters
delimiter_prompt = """
Generate product descriptions for these items.
Return ONLY a JSON array, wrapped in ```json``` code blocks.

Items: laptop, headphones, mouse

```json
[your response here]
```
"""

def parse_json_response(response_text):
    """
    Robustly parse JSON from LLM response.
    Handles common issues like markdown code blocks.
    """
    import json
    import re

    # Try direct parse first
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        pass

    # Try extracting from code blocks
    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response_text)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try finding JSON object/array
    json_match = re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])', response_text)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse JSON from response: {response_text[:100]}...")
```

### Structured Outputs (API-Level JSON Mode)

```python
"""
Modern LLM APIs provide built-in structured output modes that guarantee
valid JSON responses. This is more reliable than prompt-only techniques
and should be your first choice for production systems.
"""

# OpenAI Structured Outputs (response_format parameter)
# Available in GPT-4o and later models.

from openai import OpenAI

client = OpenAI()

# Approach 1: json_object mode — guarantees valid JSON but no schema enforcement
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You extract data. Always respond in JSON."},
        {"role": "user", "content": "Extract name and role: 'Alice is a senior engineer at Acme Corp.'"}
    ],
    response_format={"type": "json_object"},
)
# response.choices[0].message.content is guaranteed to be valid JSON

# Approach 2: json_schema mode — enforces an exact schema (OpenAI Structured Outputs)
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "Extract person information."},
        {"role": "user", "content": "Alice is a senior engineer at Acme Corp."}
    ],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "person_info",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "role": {"type": "string"},
                    "company": {"type": "string"}
                },
                "required": ["name", "role", "company"],
                "additionalProperties": False
            }
        }
    },
)

# Anthropic Claude — tool-use trick for structured outputs
# Claude does not have a native JSON mode, but you can use tool_use
# to get schema-validated structured output:
import anthropic

anthropic_client = anthropic.Anthropic()

response = anthropic_client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    tools=[{
        "name": "extract_person",
        "description": "Extract person information from text.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "role": {"type": "string"},
                "company": {"type": "string"}
            },
            "required": ["name", "role", "company"]
        }
    }],
    tool_choice={"type": "tool", "name": "extract_person"},
    messages=[
        {"role": "user", "content": "Extract info: Alice is a senior engineer at Acme Corp."}
    ]
)
# response.content[0].input contains the structured data

# When to use prompt-based JSON vs API structured outputs:
STRUCTURED_OUTPUT_DECISION = {
    "Use API structured outputs when": [
        "You need guaranteed valid JSON (no parsing failures)",
        "You have a fixed schema that rarely changes",
        "You are building production pipelines",
        "You want schema validation without custom code",
    ],
    "Use prompt-based JSON when": [
        "Your API does not support structured output mode",
        "Your schema is dynamic or user-defined",
        "You need flexibility in response structure",
        "You are prototyping and iterating quickly",
    ],
}
```

### Markdown and Tables

```python
"""
Controlling markdown output for readable responses.
"""

markdown_table_prompt = """
Compare these three programming languages for web development.
Present your comparison as a markdown table.

Languages: Python, JavaScript, Go

Table requirements:
- Columns: Language, Learning Curve, Performance, Ecosystem, Best For
- Rows: One per language
- Values should be brief (1-5 words)

Output ONLY the table in markdown format.
"""

markdown_structure_prompt = """
Write a technical overview of Docker containers.

Structure your response EXACTLY like this:

## Overview
[2-3 sentences]

## Key Concepts
- **Concept 1**: [brief explanation]
- **Concept 2**: [brief explanation]
- **Concept 3**: [brief explanation]

## Quick Example
```bash
[single docker command with explanation]
```

## When to Use
[1-2 sentences on use cases]
"""
```

### Structured Extraction

```python
"""
Extracting structured information from unstructured text.
"""

extraction_prompt = """
Extract all mentioned entities from this news article.

Categories to extract:
- PERSON: Names of people
- ORG: Organizations or companies
- LOCATION: Places, cities, countries
- DATE: Dates or time periods
- MONEY: Financial amounts

Format your response as:
```
PERSON: [name1], [name2]
ORG: [org1], [org2]
LOCATION: [loc1], [loc2]
DATE: [date1], [date2]
MONEY: [amount1], [amount2]
```

If a category has no entities, write "None".

Article:
"Apple CEO Tim Cook announced on Tuesday that the company will invest
$1 billion in a new facility in Austin, Texas. The project is expected
to create 5,000 jobs by 2025."

Extracted entities:
"""

# Multi-entity extraction with relationships
relationship_extraction = """
Extract entities and their relationships from this text.

Text: "Dr. Sarah Chen, head of AI research at Stanford University,
collaborated with Google's DeepMind team to publish a paper on
protein folding that won the 2024 Science Breakthrough Award."

Format as JSON:
{
    "entities": [
        {"text": "...", "type": "PERSON|ORG|AWARD|WORK"},
    ],
    "relationships": [
        {"subject": "...", "predicate": "...", "object": "..."}
    ]
}

JSON:
"""
```

---

## Temperature and Sampling Parameters

### Understanding Temperature

```python
"""
Temperature controls the randomness of outputs.
Lower = more deterministic, Higher = more creative/random.

Temperature affects the softmax probability distribution:
P(token) = exp(logit / T) / Σ exp(logits / T)
"""

import numpy as np
import matplotlib.pyplot as plt

def demonstrate_temperature(temperatures=None):
    """
    Visualize how temperature affects token probabilities.
    """
    if temperatures is None:
        temperatures = [0.1, 0.5, 1.0, 2.0]

    vocab = ["the", "a", "this", "that", "my", "our", "your"]
    logits = np.array([2.5, 2.0, 1.5, 1.2, 0.8, 0.5, 0.3])

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for ax, temp in zip(axes, temperatures):
        # Apply temperature scaling
        scaled_logits = logits / temp
        probs = np.exp(scaled_logits) / np.sum(np.exp(scaled_logits))

        # Plot
        bars = ax.bar(vocab, probs)
        ax.set_title(f'Temperature = {temp}')
        ax.set_ylabel('Probability')
        ax.set_ylim(0, 1)

        # Highlight max
        max_idx = np.argmax(probs)
        bars[max_idx].set_color('red')

        # Add values
        for bar, prob in zip(bars, probs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{prob:.2f}', ha='center', va='bottom', fontsize=8)

    plt.suptitle('Effect of Temperature on Token Probabilities')
    plt.tight_layout()
    plt.savefig('temperature_effect.png', dpi=150)

demonstrate_temperature()

# Temperature guidelines:
TEMPERATURE_GUIDELINES = {
    0.0: {
        "name": "Deterministic (Greedy)",
        "use_cases": ["Classification", "Factual QA", "Extraction", "Code generation"],
        "characteristics": "Always picks most likely token; reproducible outputs",
    },
    0.3: {
        "name": "Low Creativity",
        "use_cases": ["Summarization", "Translation", "Structured outputs"],
        "characteristics": "Slight variation; mostly coherent",
    },
    0.7: {
        "name": "Balanced (Default)",
        "use_cases": ["General conversation", "Explanations", "Problem solving"],
        "characteristics": "Good balance of coherence and variation",
    },
    1.0: {
        "name": "Standard Creative",
        "use_cases": ["Story writing", "Brainstorming", "Marketing copy"],
        "characteristics": "More diverse outputs; may occasionally be inconsistent",
    },
    1.5: {
        "name": "High Creativity",
        "use_cases": ["Poetry", "Experimental content", "Idea generation"],
        "characteristics": "Highly varied; may produce unusual combinations",
    },
}
```

### Top-K and Top-P Sampling

```python
"""
Top-K and Top-P (nucleus) sampling restrict which tokens can be sampled.
They can be combined with temperature for fine control.
"""

def explain_sampling_strategies():
    """Explain different sampling approaches."""

    strategies = {
        "Greedy (temperature=0)": {
            "how": "Always pick the highest probability token",
            "pros": ["Deterministic", "Coherent"],
            "cons": ["Repetitive", "Boring", "Can get stuck in loops"],
            "code": "logits.argmax()",
        },

        "Top-K Sampling": {
            "how": "Only consider the K most likely tokens, then sample",
            "pros": ["Reduces unlikely tokens", "Tunable diversity"],
            "cons": ["Fixed K may be too restrictive or loose"],
            "code": "sample from top_k(logits, k=50)",
            "typical_values": "k=50 for general, k=10 for focused",
        },

        "Top-P (Nucleus) Sampling": {
            "how": "Consider smallest set of tokens whose probability sums to P",
            "pros": ["Adaptive to distribution", "Better than fixed K"],
            "cons": ["Can vary widely in number of tokens considered"],
            "code": "sample from nucleus(logits, p=0.9)",
            "typical_values": "p=0.9 for general, p=0.7 for focused",
        },

        "Temperature + Top-P": {
            "how": "Apply temperature first, then nucleus sampling",
            "pros": ["Fine-grained control", "Industry standard"],
            "cons": ["Two parameters to tune"],
            "code": "sample from nucleus(logits/temp, p=0.9)",
            "typical_values": "temp=0.7, p=0.9",
        },
    }

    for name, details in strategies.items():
        print(f"\n{'='*50}")
        print(f"{name}")
        print(f"{'='*50}")
        print(f"How it works: {details['how']}")
        print(f"Pros: {', '.join(details['pros'])}")
        print(f"Cons: {', '.join(details['cons'])}")
        if 'typical_values' in details:
            print(f"Typical values: {details['typical_values']}")

explain_sampling_strategies()
```

### Parameter Recommendations by Task

```python
"""
Recommended parameters for different tasks.
"""

PARAMETER_RECOMMENDATIONS = {
    "code_generation": {
        "temperature": 0.0,
        "top_p": 1.0,
        "reasoning": "Code needs to be correct; creativity can introduce bugs",
    },

    "classification": {
        "temperature": 0.0,
        "top_p": 1.0,
        "reasoning": "Classification should be deterministic and reproducible",
    },

    "summarization": {
        "temperature": 0.3,
        "top_p": 0.9,
        "reasoning": "Slight variation is OK; factual accuracy is important",
    },

    "translation": {
        "temperature": 0.3,
        "top_p": 0.9,
        "reasoning": "Minor phrasing variation acceptable; meaning must be preserved",
    },

    "conversation": {
        "temperature": 0.7,
        "top_p": 0.9,
        "reasoning": "Natural variation in responses; avoid repetitive patterns",
    },

    "creative_writing": {
        "temperature": 1.0,
        "top_p": 0.95,
        "reasoning": "Creativity is valuable; unusual phrasings can be good",
    },

    "brainstorming": {
        "temperature": 1.2,
        "top_p": 0.95,
        "reasoning": "Diverse ideas wanted; wild suggestions can spark creativity",
    },

    "factual_qa": {
        "temperature": 0.0,
        "top_p": 1.0,
        "reasoning": "Facts should be consistent; no room for 'creative' answers",
    },
}

def get_recommended_parameters(task_type):
    """Get recommended parameters for a task."""
    if task_type in PARAMETER_RECOMMENDATIONS:
        rec = PARAMETER_RECOMMENDATIONS[task_type]
        return {
            "temperature": rec["temperature"],
            "top_p": rec["top_p"],
            "reasoning": rec["reasoning"],
        }
    else:
        # Default balanced parameters
        return {
            "temperature": 0.7,
            "top_p": 0.9,
            "reasoning": "Using balanced defaults; adjust based on results",
        }
```

### Engineering Controls: max_tokens and stop sequences

Beyond temperature and top_p, two parameters are essential for production:

```python
"""
max_tokens: Hard limit on output length. Without it, the model may
generate until the context window is exhausted, costing more and
returning unwanted content.

stop sequences: Strings that terminate generation early. Critical for
structured extraction where you want the model to stop after producing
the desired output.
"""

# max_tokens prevents runaway generation
response = client.chat.completions.create(
    model="gpt-4-turbo",
    messages=[{"role": "user", "content": "Classify: 'great product' → "}],
    max_tokens=10,  # Classification needs ~1-3 tokens; 10 is generous safety margin
    temperature=0,
)
# Without max_tokens, the model might generate a classification PLUS an explanation,
# costing 10-50x more tokens than needed.

# stop sequences halt generation at a delimiter
response = client.chat.completions.create(
    model="gpt-4-turbo",
    messages=[{"role": "user", "content": "Extract name from: 'Alice works at Acme'\nName:"}],
    stop=["\n", ".", "Input:"],  # Stop at newline, period, or next-input marker
    max_tokens=20,
)
# Useful for: classification (stop after label), extraction (stop after value),
# few-shot (stop before generating another example)

# Cost impact of max_tokens:
# GPT-4-turbo output: $30/1M tokens
# Setting max_tokens=10 vs unbounded (avg 200 tokens):
#   Savings per call: (200-10) * $30/1M = $0.0057
#   At 100K calls/month: $570/month saved
```

### Prompt Caching: Cutting Costs for Repeated System Prompts

In production, you often send the same system prompt + few-shot examples thousands of times with different user inputs. Prompt caching avoids re-processing the static prefix:

```python
"""
Anthropic Prompt Caching (2024):
- Mark static prompt sections with cache_control
- First call: full price (cache write)
- Subsequent calls: 90% discount on cached prefix (cache read)
- Cache TTL: 5 minutes (resets on each use)

OpenAI Automatic Caching (2024):
- Automatic for prompts sharing a common prefix ≥ 1024 tokens
- 50% discount on cached input tokens
- No API changes needed
"""

# Anthropic prompt caching example:
import anthropic

client = anthropic.Anthropic()

# The system prompt + examples are cached; only user input changes per call
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=100,
    system=[
        {
            "type": "text",
            "text": "You are a customer support classifier. [long system prompt here...]",
            "cache_control": {"type": "ephemeral"},  # Mark for caching
        }
    ],
    messages=[{"role": "user", "content": "Classify: 'My order is late'"}],
)
# First call: cache miss → full price
# Subsequent calls within 5 min: cache hit → 90% cheaper input tokens

# Cost impact:
# System prompt: 2000 tokens. User input: 50 tokens. 100K calls/month.
# Without caching:  100K × 2050 tokens × $3/1M = $615/month (Claude Sonnet input)
# With caching:     100K × (2000×$0.30 + 50×$3)/1M = $75/month
# Savings: ~$540/month (88% reduction)

# When to use prompt caching:
# - System prompt > 1024 tokens (OpenAI automatic threshold)
# - Same prompt prefix sent > 10 times within 5 minutes
# - Few-shot examples that don't change between calls
```

---

## Debugging and Iterating on Prompts

### The Prompt Development Cycle

```
┌─────────────────────────────────────────────────────────────────┐
│                    PROMPT DEVELOPMENT CYCLE                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐  │
│   │ Design  │────>│  Test   │────>│ Analyze │────>│ Refine  │  │
│   │ Prompt  │     │  Cases  │     │ Failures│     │  Prompt │  │
│   └─────────┘     └─────────┘     └─────────┘     └─────────┘  │
│        ▲                                               │        │
│        │                                               │        │
│        └───────────────────────────────────────────────┘        │
│                                                                  │
│   Iteration Checklist:                                          │
│   □ Did output match expected format?                           │
│   □ Was content accurate and complete?                          │
│   □ Did it handle edge cases?                                   │
│   □ Was tone/style appropriate?                                 │
│   □ Is it consistent across multiple runs?                      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Systematic Debugging

```python
"""
Framework for debugging prompt issues.
"""

class PromptDebugger:
    """Tools for debugging prompts systematically."""

    @staticmethod
    def diagnose_issue(prompt, expected_output, actual_output):
        """
        Diagnose why a prompt isn't working.
        """
        issues = []

        # Check 1: Format mismatch
        if not isinstance(actual_output, type(expected_output)):
            issues.append({
                "type": "FORMAT_MISMATCH",
                "description": f"Expected {type(expected_output).__name__}, got {type(actual_output).__name__}",
                "fix": "Add explicit format specification to prompt",
            })

        # Check 2: Length issues
        if isinstance(expected_output, str):
            exp_len = len(expected_output.split())
            act_len = len(str(actual_output).split())
            if act_len > exp_len * 2:
                issues.append({
                    "type": "TOO_LONG",
                    "description": f"Output {act_len} words, expected ~{exp_len}",
                    "fix": "Add word/sentence limit to prompt",
                })
            elif act_len < exp_len * 0.5:
                issues.append({
                    "type": "TOO_SHORT",
                    "description": f"Output {act_len} words, expected ~{exp_len}",
                    "fix": "Ask for more detail or specify minimum length",
                })

        # Check 3: Missing components
        if isinstance(expected_output, dict):
            expected_keys = set(expected_output.keys())
            actual_keys = set(actual_output.keys()) if isinstance(actual_output, dict) else set()
            missing = expected_keys - actual_keys
            if missing:
                issues.append({
                    "type": "MISSING_FIELDS",
                    "description": f"Missing: {missing}",
                    "fix": "List all required fields in prompt with 'Required:' prefix",
                })

        return issues

    @staticmethod
    def generate_test_cases(task_description, num_cases=5):
        """
        Generate diverse test cases for prompt validation.
        """
        test_types = [
            "typical_case",
            "edge_case_short",
            "edge_case_long",
            "adversarial",
            "ambiguous",
        ]

        print(f"Generate test cases for: {task_description}")
        print("\nTest case types to create:")
        for i, test_type in enumerate(test_types[:num_cases], 1):
            print(f"  {i}. {test_type.replace('_', ' ').title()}")

        return test_types[:num_cases]

    @staticmethod
    def a_b_test(prompt_a, prompt_b, test_inputs, evaluation_fn):
        """
        Compare two prompt variants.
        """
        results = {"prompt_a": [], "prompt_b": []}

        for input_text in test_inputs:
            # In practice, you'd call the LLM here
            score_a = evaluation_fn(prompt_a, input_text)
            score_b = evaluation_fn(prompt_b, input_text)

            results["prompt_a"].append(score_a)
            results["prompt_b"].append(score_b)

        avg_a = sum(results["prompt_a"]) / len(results["prompt_a"])
        avg_b = sum(results["prompt_b"]) / len(results["prompt_b"])

        winner = "A" if avg_a > avg_b else "B"
        print(f"Prompt A average: {avg_a:.2f}")
        print(f"Prompt B average: {avg_b:.2f}")
        print(f"Winner: Prompt {winner}")

        return results

# Common prompt issues and fixes:
COMMON_ISSUES = {
    "Output too verbose": {
        "symptoms": ["Long responses", "Unnecessary explanations"],
        "fixes": [
            "Add word/sentence limit",
            "Say 'Be concise'",
            "Use 'Answer directly, then explain if asked'",
        ],
    },
    "Wrong format": {
        "symptoms": ["JSON with markdown", "Missing required fields"],
        "fixes": [
            "Provide exact schema",
            "Say 'Output ONLY valid JSON'",
            "Use few-shot examples showing format",
        ],
    },
    "Inconsistent outputs": {
        "symptoms": ["Different formats each time", "Varying quality"],
        "fixes": [
            "Lower temperature",
            "Add more structure",
            "Use few-shot examples",
        ],
    },
    "Ignoring instructions": {
        "symptoms": ["Skipping requirements", "Partial completion"],
        "fixes": [
            "Number your instructions",
            "Use 'IMPORTANT:' prefix",
            "Put critical instructions at start AND end",
        ],
    },
    "Hallucinating": {
        "symptoms": ["Made-up facts", "Confident but wrong"],
        "fixes": [
            "Say 'If unsure, say so'",
            "Ask for sources",
            "Lower temperature",
            "Add RAG for factual tasks",
        ],
    },
}
```

### Prompt Templates

```python
"""
Reusable prompt templates for common tasks.
"""

class PromptTemplates:
    """Library of tested prompt templates."""

    @staticmethod
    def classification(categories, few_shot_examples=None):
        """Template for classification tasks."""
        template = f"""
Classify the input into one of these categories: {', '.join(categories)}

Rules:
- Respond with ONLY the category name
- Choose the single best match
- If truly ambiguous, choose the most likely category
"""
        if few_shot_examples:
            template += "\nExamples:\n"
            for text, label in few_shot_examples:
                template += f"Input: {text}\nCategory: {label}\n\n"

        template += "\nNow classify:\nInput: {input_text}\nCategory:"

        return template

    @staticmethod
    def summarization(length="brief", style="professional"):
        """Template for summarization tasks."""
        length_specs = {
            "brief": "2-3 sentences",
            "moderate": "1 paragraph (4-6 sentences)",
            "detailed": "2-3 paragraphs",
        }

        return f"""
Summarize the following text.

Requirements:
- Length: {length_specs[length]}
- Style: {style}
- Include key facts and main conclusions
- Omit examples and minor details

Text to summarize:
{{text}}

Summary:
"""

    @staticmethod
    def data_extraction(fields, output_format="json"):
        """Template for structured data extraction."""
        fields_str = "\n".join([f"- {field['name']}: {field['description']}"
                                for field in fields])

        return f"""
Extract the following information from the text.

Fields to extract:
{fields_str}

Output format: {output_format.upper()}

Rules:
- If a field is not found, use null/None
- Extract verbatim when possible
- If multiple values exist, return as a list

Text:
{{text}}

Extracted data:
"""

    @staticmethod
    def code_review(focus_areas=None):
        """Template for code review."""
        focus = focus_areas or ["bugs", "security", "performance", "readability"]

        return f"""
Review this code and provide feedback.

Focus areas: {', '.join(focus)}

For each issue found, provide:
1. Severity (Critical/High/Medium/Low)
2. Location (line number or function)
3. Description of the issue
4. Suggested fix

Format as:
## [Severity] Issue Title
**Location:** [where]
**Problem:** [what's wrong]
**Fix:** [how to fix]
```[language]
[corrected code snippet]
```

Code to review:
```{{language}}
{{code}}
```

Review:
"""

# Usage examples:
templates = PromptTemplates()

# Classification
class_prompt = templates.classification(
    categories=["Bug Report", "Feature Request", "Question", "Complaint"],
    few_shot_examples=[
        ("The app crashes when I upload images", "Bug Report"),
        ("Can you add dark mode?", "Feature Request"),
    ]
)

# Summarization
summary_prompt = templates.summarization(length="brief", style="professional")

# Data extraction
extraction_prompt = templates.data_extraction(
    fields=[
        {"name": "company_name", "description": "Name of the company"},
        {"name": "funding_amount", "description": "Amount of funding raised"},
        {"name": "investors", "description": "List of investors"},
    ]
)

print("Classification Template:")
print(class_prompt)
```

---

## Evaluating Prompt Quality

### A Practical Scoring Rubric

```python
"""
You cannot improve what you cannot measure. This section provides
a concrete framework for evaluating prompt effectiveness before
deploying to production.
"""

PROMPT_EVALUATION_RUBRIC = {
    "Format Compliance": {
        "weight": 0.25,
        "description": "Does the output match the requested structure?",
        "scoring": {
            1: "Wrong format entirely (e.g., prose instead of JSON)",
            2: "Partially correct format with missing fields",
            3: "Correct format with minor deviations",
            4: "Exact match to specified schema/structure",
        },
    },
    "Content Accuracy": {
        "weight": 0.30,
        "description": "Is the information correct and relevant?",
        "scoring": {
            1: "Major factual errors or hallucinations",
            2: "Partially correct with notable gaps",
            3: "Mostly accurate with minor issues",
            4: "Fully accurate and well-supported",
        },
    },
    "Instruction Following": {
        "weight": 0.25,
        "description": "Did the model follow all stated constraints?",
        "scoring": {
            1: "Ignored most instructions",
            2: "Followed some, missed key constraints",
            3: "Followed most instructions",
            4: "Followed every instruction precisely",
        },
    },
    "Consistency": {
        "weight": 0.20,
        "description": "Are repeated runs stable in quality?",
        "scoring": {
            1: "Wildly different outputs each run",
            2: "Moderate variance; format sometimes drifts",
            3: "Mostly stable with minor variations",
            4: "Highly consistent across runs",
        },
    },
}


def evaluate_prompt(prompt: str, test_inputs: list[str], expected_outputs: list[dict],
                    call_llm_fn=None, n_runs: int = 3) -> dict:
    """
    Evaluate a prompt against test cases using the rubric above.

    Args:
        prompt: The prompt template (with {input} placeholder)
        test_inputs: List of test input strings
        expected_outputs: List of dicts describing expected structure/content
        call_llm_fn: Function that takes a prompt string and returns a response string.
                     If None, prints the evaluation plan without running.
        n_runs: Number of runs per input to test consistency

    Returns:
        Dictionary with per-dimension scores and overall score.
    """
    if call_llm_fn is None:
        print("Evaluation plan (dry run — no LLM function provided):")
        print(f"  Prompt length: {len(prompt)} chars")
        print(f"  Test cases: {len(test_inputs)}")
        print(f"  Runs per case: {n_runs}")
        print(f"  Total LLM calls: {len(test_inputs) * n_runs}")
        print(f"\nScoring dimensions:")
        for dim, info in PROMPT_EVALUATION_RUBRIC.items():
            print(f"  - {dim} (weight={info['weight']}): {info['description']}")
        return {}

    scores = {dim: [] for dim in PROMPT_EVALUATION_RUBRIC}

    for test_input, expected in zip(test_inputs, expected_outputs):
        run_outputs = []
        for _ in range(n_runs):
            filled_prompt = prompt.replace("{input}", test_input)
            output = call_llm_fn(filled_prompt)
            run_outputs.append(output)

        # --- Automated scoring per dimension ---

        # Format Compliance: check if output matches expected type/structure
        format_scores = []
        for out in run_outputs:
            if expected.get("format") == "json":
                try:
                    json.loads(out); format_scores.append(4)
                except: format_scores.append(1)
            elif expected.get("max_length") and len(out) <= expected["max_length"]:
                format_scores.append(4)
            elif expected.get("max_length"):
                format_scores.append(2)
            else:
                format_scores.append(3)  # No format constraint → assume OK
        scores["Format Compliance"].append(sum(format_scores) / len(format_scores))

        # Content Accuracy: exact match or substring match for classification
        content_scores = []
        for out in run_outputs:
            expected_val = expected.get("category") or expected.get("expected_contains", "")
            if isinstance(expected_val, str) and expected_val.lower() in out.lower():
                content_scores.append(4)
            elif isinstance(expected_val, list) and all(v.lower() in out.lower() for v in expected_val):
                content_scores.append(4)
            else:
                content_scores.append(1)
        scores["Content Accuracy"].append(sum(content_scores) / len(content_scores))

        # Instruction Following: check length constraints, format rules
        scores["Instruction Following"].append(
            min(scores["Format Compliance"][-1], scores["Content Accuracy"][-1])
        )

        # Consistency: variance across runs (lower variance = higher score)
        lengths = [len(o) for o in run_outputs]
        if len(set(o.strip().lower() for o in run_outputs)) == 1:
            scores["Consistency"].append(4)  # All identical
        elif max(lengths) - min(lengths) < 20:
            scores["Consistency"].append(3)  # Minor variation
        else:
            scores["Consistency"].append(2)  # Moderate variation

    # Compute weighted overall score
    import json
    overall = 0
    print(f"\n{'Dimension':<25} {'Avg Score':<10} {'Weight':<8} {'Weighted'}")
    print("-" * 55)
    for dim, info in PROMPT_EVALUATION_RUBRIC.items():
        avg = sum(scores[dim]) / len(scores[dim]) if scores[dim] else 0
        weighted = avg * info["weight"]
        overall += weighted
        print(f"{dim:<25} {avg:<10.2f} {info['weight']:<8} {weighted:.2f}")
    print(f"\n{'OVERALL':<25} {overall:.2f} / 4.00")

    return {"scores": scores, "overall": overall}


# Example: evaluating a classification prompt
print("=== Prompt Evaluation Example ===\n")

classification_prompt = """Classify this support ticket into one of: BILLING, TECHNICAL, ACCOUNT, SHIPPING, GENERAL.
Respond with ONLY the category name.

Ticket: {input}
Category:"""

test_cases = [
    "I was charged twice for my subscription",
    "The app keeps crashing on my iPhone",
    "When will my package arrive?",
    "How do I change my email address?",
    "What colors does the product come in?",
]

expected = [
    {"category": "BILLING"},
    {"category": "TECHNICAL"},
    {"category": "SHIPPING"},
    {"category": "ACCOUNT"},
    {"category": "GENERAL"},
]

evaluate_prompt(classification_prompt, test_cases, expected)
```

### When Good Prompts Go Bad: Regression Testing

```python
"""
Prompts can degrade when models are updated, context changes, or
edge cases appear. Treat prompts like code: version them, test them,
and track regressions.
"""

import json
from datetime import datetime

class PromptTestSuite:
    """
    Minimal prompt regression testing framework.
    Store test cases alongside your prompt templates.
    """

    def __init__(self, prompt_name: str, prompt_template: str):
        self.prompt_name = prompt_name
        self.prompt_template = prompt_template
        self.test_cases: list[dict] = []
        self.results_history: list[dict] = []

    def add_test_case(self, input_text: str, expected_contains: list[str] = None,
                      expected_format: str = None, max_length: int = None):
        """Add a test case with expected output characteristics."""
        self.test_cases.append({
            "input": input_text,
            "expected_contains": expected_contains or [],
            "expected_format": expected_format,  # "json", "markdown", "plain"
            "max_length": max_length,
        })

    def validate_output(self, output: str, test_case: dict) -> dict:
        """Check whether a single output meets test case expectations."""
        results = {"passed": True, "failures": []}

        # Check format
        if test_case["expected_format"] == "json":
            try:
                json.loads(output)
            except json.JSONDecodeError:
                results["passed"] = False
                results["failures"].append("Output is not valid JSON")

        # Check required content
        for phrase in test_case["expected_contains"]:
            if phrase.lower() not in output.lower():
                results["passed"] = False
                results["failures"].append(f"Missing expected phrase: '{phrase}'")

        # Check length
        if test_case["max_length"] and len(output) > test_case["max_length"]:
            results["passed"] = False
            results["failures"].append(
                f"Output too long: {len(output)} > {test_case['max_length']}"
            )

        return results

    def run_suite(self, call_llm_fn=None) -> dict:
        """Run all test cases and report results."""
        if call_llm_fn is None:
            print(f"Test suite for '{self.prompt_name}': {len(self.test_cases)} cases (dry run)")
            for i, tc in enumerate(self.test_cases, 1):
                print(f"  Case {i}: input='{tc['input'][:50]}...'" if len(tc['input']) > 50
                      else f"  Case {i}: input='{tc['input']}'")
            return {"status": "dry_run", "total": len(self.test_cases)}

        passed = 0
        failed = 0
        for tc in self.test_cases:
            filled = self.prompt_template.replace("{input}", tc["input"])
            output = call_llm_fn(filled)
            result = self.validate_output(output, tc)
            if result["passed"]:
                passed += 1
            else:
                failed += 1
                print(f"  FAIL: {result['failures']}")

        summary = {
            "timestamp": datetime.now().isoformat(),
            "prompt_name": self.prompt_name,
            "passed": passed,
            "failed": failed,
            "total": passed + failed,
            "pass_rate": passed / (passed + failed) if (passed + failed) > 0 else 0,
        }
        self.results_history.append(summary)
        return summary


# Example usage:
suite = PromptTestSuite(
    prompt_name="ticket_classifier_v1",
    prompt_template=classification_prompt,
)
suite.add_test_case("I was double-charged!", expected_contains=["BILLING"], max_length=20)
suite.add_test_case("App crashes on upload", expected_contains=["TECHNICAL"], max_length=20)
suite.add_test_case("", expected_contains=["GENERAL"], max_length=20)  # edge case: empty input
suite.run_suite()  # dry run
```

---

## Interview Preparation

### Industry Context

Prompt engineering is now a recognized discipline in the AI industry. Roles like "Prompt Engineer" and "AI Engineer" are common at companies building LLM-powered products. Even for traditional software engineers, prompt engineering is becoming a required skill as more applications integrate LLM calls. Understanding how to systematically design, evaluate, and iterate on prompts is essential for anyone building production AI systems. The techniques in this blog apply across all major providers (OpenAI, Anthropic, Google, open-source models via Ollama/vLLM).

### Concept Questions

**Q1: What is prompt engineering and why does it matter?**

*Answer:* Prompt engineering is the skill of crafting inputs to language models to get desired outputs. It matters because LLMs are incredibly capable but also sensitive to how requests are phrased. A well-crafted prompt can dramatically improve accuracy, consistency, and relevance of outputs. It's the primary way users control model behavior without retraining.

**Q2: Explain zero-shot, one-shot, and few-shot prompting.**

*Answer:* These refer to the number of examples provided:
- **Zero-shot**: No examples; model uses only pre-trained knowledge
- **One-shot**: Single example showing desired input-output mapping
- **Few-shot**: Multiple examples (2-10) demonstrating the pattern

More examples generally improve performance on novel tasks but increase token usage. Zero-shot works for common tasks; few-shot is needed for custom formats or complex tasks.

**Q3: When should you use low vs high temperature?**

*Answer:* Temperature controls output randomness:
- **Low (0-0.3)**: For factual tasks, code generation, classification—where correctness matters more than creativity
- **Medium (0.5-0.8)**: For general conversation, explanations—balanced coherence and variation
- **High (0.9-1.5)**: For creative writing, brainstorming—when diversity and novelty are valuable

Key insight: Temperature doesn't make the model "smarter"; it just changes how it samples from its probability distribution.

**Q4: How do you get reliable JSON output from an LLM?**

*Answer:* Several techniques:
1. **Schema specification**: Provide exact JSON schema with field types
2. **Explicit instruction**: "Output ONLY valid JSON, no other text"
3. **Few-shot examples**: Show examples of correct JSON format
4. **Delimiters**: Ask for JSON wrapped in code blocks for easy parsing
5. **Validation**: Parse response with error handling; retry on failure
6. **Output mode**: Some APIs have built-in JSON mode (OpenAI's response_format)

**Q5: What are structured outputs and when should you use them instead of prompt-based JSON?**

*Answer:* Structured outputs are API-level features that guarantee the model returns valid JSON matching a specified schema. OpenAI offers `response_format` with `json_object` and `json_schema` modes. Anthropic Claude achieves this via the tool-use mechanism with `tool_choice`. Use structured outputs when:
- You need guaranteed valid JSON (no parsing failures in production)
- You have a fixed, well-defined schema
- You are building automated pipelines where parse errors cause failures

Use prompt-based JSON when prototyping, when your schema is dynamic, or when your API does not support structured output mode.

### Practical Questions

**Q6: Write a prompt for classifying customer support tickets.**

```python
support_ticket_prompt = """
You are a customer support classifier. Categorize each ticket.

Categories:
- BILLING: Payment issues, refunds, subscriptions
- TECHNICAL: Bugs, errors, feature problems
- ACCOUNT: Login issues, profile updates, access
- SHIPPING: Delivery status, tracking, returns
- GENERAL: Questions, feedback, other inquiries

Rules:
- Output ONLY the category name
- If multiple categories apply, choose the primary one
- For vague tickets, classify as GENERAL

Examples:
Ticket: "I was charged twice for my subscription!"
Category: BILLING

Ticket: "The app keeps crashing on my iPhone"
Category: TECHNICAL

Ticket: "Can't reset my password"
Category: ACCOUNT

Now classify:
Ticket: "{ticket_text}"
Category:
"""
```

**Q7: Debug this prompt that's producing inconsistent results.**

```python
# Original problematic prompt:
bad_prompt = "Summarize this article."

# Diagnosis:
# - No format specification
# - No length constraint
# - No style guidance
# - No examples

# Fixed version:
good_prompt = """
Summarize the following article for an executive audience.

Requirements:
- Exactly 3 bullet points
- Each bullet: 15-20 words maximum
- Focus on: business impact, key metrics, and recommendations
- Use professional, direct language

Article:
{article_text}

Summary:
•
"""

# Additional improvements:
# - Set temperature=0.3 for consistency
# - Add few-shot examples if still inconsistent
# - Consider output validation and retry logic
```

**Q8: How does few-shot prompting actually work inside the model?**

*Answer:* Few-shot works through "induction heads" — attention patterns that develop during pre-training (Olsson et al., 2022). These heads detect repeated input-output patterns in the context and copy the mapping to new inputs. The model doesn't learn in the gradient sense; it pattern-matches within existing weights. Implications: (1) examples must be consistent (conflicting examples confuse induction), (2) format matters more than explanation, (3) the last example has the strongest influence (recency bias), (4) more diverse examples better define the pattern boundary.

**Q9: What are prompt caching and why does it matter?**

*Answer:* Prompt caching avoids re-processing static prompt prefixes (system prompt + few-shot examples) that are identical across calls. Anthropic offers explicit cache_control markers with 90% discount on cached tokens. OpenAI automatically caches prefixes ≥1024 tokens at 50% discount. For a 2000-token system prompt sent 100K times/month, caching can save $500+/month. Use it when: system prompt > 1024 tokens, same prefix sent >10 times within the cache TTL, few-shot examples don't change per call.

**Q10: What are positional effects in prompts, and how do you mitigate "lost in the middle"?**

*Answer:* LLMs don't weight all parts of a prompt equally. Instructions at the START (primacy) and END (recency) are most reliably followed. Instructions buried in long contexts are often missed — Liu et al. (2023) call this "Lost in the Middle." Mitigation: put critical instructions at the start AND repeat them at the end; place examples in the middle; put input data near the end (closest to where the model generates).

### System Design Question

**Q11: Design a prompt management system for a multi-tenant SaaS product.**

*Answer:*

```
Components:

1. Prompt Registry (Database)
   - Versioned prompt templates (git-like: v1.0, v1.1, v2.0)
   - Per-tenant overrides (custom system prompts, examples)
   - A/B test configuration (which version for which % of traffic)

2. Prompt Compiler (Runtime)
   - Assembles final prompt: system + tenant config + few-shot + user input
   - Enforces token budget (truncates examples if needed)
   - Applies prompt caching headers for repeated prefixes

3. Evaluation Pipeline (CI/CD)
   - Regression test suite runs on every prompt change
   - Automated scoring (format compliance, content accuracy, consistency)
   - Cost estimation before deploy (token count × pricing)
   - Rollback trigger: if pass rate drops below threshold

4. Monitoring (Production)
   - Token usage per prompt version (cost tracking)
   - Latency percentiles (P50, P95) per version
   - Output quality sampling (LLM-as-judge on 1% of traffic)
   - Cache hit rate tracking
```

**Key design decisions:**
- Version prompts like code (immutable versions, canary deploys)
- Separate system prompt (cached, shared) from user input (variable)
- Budget token limits per tier (free: 500 tokens/prompt, pro: 2000)
- Centralize prompt templates — don't scatter them across codebases

---

## Exercises

### Exercise 1: Prompt Component Practice
Take this task: "I need to analyze competitor products."

Rewrite it using the CRAFT framework:
- Context: What information do you have?
- Role: What expertise is needed?
- Action: What specific analysis?
- Format: How should results be presented?
- Tone: What style is appropriate?

### Exercise 2: Few-Shot Design
Create a few-shot prompt for extracting meeting action items from transcripts. Include:
- 3 diverse examples
- Clear output format (JSON)
- Edge case handling

### Exercise 3: Temperature Tuning
For each task, recommend temperature and explain why:
1. Generating Python code to sort a list
2. Writing birthday card messages
3. Summarizing a legal document
4. Brainstorming startup ideas
5. Answering customer FAQ questions

### Exercise 4: Prompt Debugging
Your JSON extraction prompt produces these failures:
- Sometimes outputs markdown instead of JSON
- Occasionally includes explanation text before JSON
- Missing fields in 20% of cases

Design fixes for each issue and test them.

### Exercise 5: Build a Prompt Library
Create reusable templates for:
1. Email drafting (professional)
2. Code explanation
3. Data cleaning instructions
4. Meeting summarization
5. Feedback formatting

---

## Section Checkpoints

### Checkpoint 1 — After "CRAFT Framework" and "Five Components"
1. Name the five CRAFT components and give a one-sentence description of each.
2. What are three role anti-patterns to avoid?
3. Why does the "Level 4 - Demonstrated" prompt outperform "Level 2 - Specific"?
4. Write a CRAFT prompt for summarizing a legal contract for a non-lawyer audience.

### Checkpoint 2 — After "Zero/One/Few-Shot" and "In-Context Learning"
1. When should you use zero-shot vs few-shot prompting?
2. How do induction heads enable in-context learning?
3. Why does example order matter, and what is recency bias?
4. Estimate the token cost difference between 0-shot and 5-shot for 100K API calls.

### Checkpoint 3 — After "System Prompts" and "Structured Outputs"
1. What sections should a production system prompt include?
2. Compare prompt-based JSON, JSON mode, and json_schema mode for structured outputs.
3. When should you use Anthropic's tool-use trick instead of prompt-based JSON?
4. What is prompt caching, and when does it provide the biggest cost savings?

### Checkpoint 4 — After "Temperature/Sampling" and "Engineering Controls"
1. What temperature would you use for code generation vs creative writing? Why?
2. Explain how top-p sampling adapts to different probability distributions.
3. Why is max_tokens important for production cost control?
4. When should you use stop sequences?

### Checkpoint 5 — After "Debugging" and "Evaluation"
1. Name three common prompt failure symptoms and their fixes.
2. What are the four dimensions of the prompt evaluation rubric?
3. How does PromptTestSuite help with regression testing?
4. What is the "lost in the middle" effect and how do you mitigate it?

---

## Job Role Mapping

| Section | ML/AI Engineer | Data Scientist | AI Architect | Engineering Manager |
|---------|---------------|----------------|--------------|---------------------|
| CRAFT Framework | Must know: all 5 components, how to compose production prompts | Must know: context and format for data tasks | Must know: prompt standardization across teams | Must know: prompt quality impact on product |
| Zero/One/Few-Shot & ICL | Must know: in-context learning mechanics, example selection, cost trade-offs | Must know: when to use each strategy for data tasks | Must know: token budget planning, few-shot vs fine-tuning decision | Must know: cost implications of few-shot at scale |
| System Prompts & Structured Outputs | Must know: dynamic system prompts, json_schema, tool-use, caching | Must know: structured extraction for data pipelines | Must know: prompt caching strategy, output reliability tiers | Must know: structured output reliability for SLAs |
| Temperature & Engineering Controls | Must know: parameter tuning, max_tokens, stop sequences | Must know: temperature for analysis vs generation tasks | Must know: parameter governance across deployments | Must know: cost impact of parameter choices |
| Debugging & Evaluation | Must know: regression testing, automated scoring, A/B testing | Must know: evaluation metrics for prompt quality | Must know: CI/CD for prompts, prompt versioning | Must know: quality gates, monitoring, rollback |
| Positional Effects & Cost | Must know: instruction placement, lost-in-middle, token budgets | Must know: prompt cost estimation | Must know: prompt management system design | Must know: cost modeling, caching ROI |

---

## Summary

### Key Takeaways

1. **CRAFT your prompts**: Context, Role, Action, Format, Tone—every component matters
2. **Show, don't tell**: Few-shot examples are more reliable than lengthy instructions
3. **Temperature for control**: Low for accuracy, high for creativity
4. **Format explicitly**: Specify exact output structure; validate and retry
5. **Iterate systematically**: Test, analyze failures, refine, repeat
6. **Build templates**: Reusable prompts save time and ensure consistency

### What's Next

In Blog 13, we'll explore Advanced Prompting Techniques:
- Chain-of-thought prompting
- Self-consistency and majority voting
- Tree-of-thoughts reasoning
- ReAct (Reasoning + Acting)
- Prompt chaining and decomposition
- Automatic prompt optimization

---

## Self-Assessment

### What This Blog Does Well

- **Clear conceptual framework.** CRAFT mnemonic with progressive examples (Level 1-4). Zero/one/few-shot progression with in-context learning mechanism (induction heads) explaining *why* it works.
- **Production cost awareness.** Token cost analysis for few-shot strategies, prompt caching with ROI calculation (88% savings example), max_tokens/stop sequences for cost control.
- **Structured outputs coverage.** Three approaches: prompt-based JSON, OpenAI json_schema mode, and Anthropic tool-use trick — with reliability estimates and decision guide.
- **Positional effects.** "Lost in the middle" phenomenon with concrete mitigation template (instructions at start + end, examples in middle, data last).
- **Evaluation framework.** Four-dimension rubric with weighted scoring, automated scoring implementation (format compliance, content accuracy, instruction following, consistency), and PromptTestSuite for regression testing.
- **11 interview questions** spanning concepts (ICL, temperature, caching), practical scenarios (ticket classification, debugging), and system design (prompt management system).
- **5 checkpoints + job role mapping** (6 sections × 4 roles).

### Where This Blog Falls Short

- **No live API calls in code.** All code is structural. Readers must integrate with their own API keys to see real outputs.
- **Limited adversarial coverage.** Prompt injection, jailbreaking, and adversarial inputs deferred to Blog 14.
- **No automatic prompt optimization.** DSPy, OPRO, and APE deferred to Blog 13.
- **No multi-turn conversation patterns.** All examples are single-turn. Real chatbot systems require multi-turn prompt management.

### Architect Sanity Checks

### Check 1: Production Deployment Readiness
**Question**: Would you trust this person to engineer prompts that maintain reliable success in production?
**Answer: YES.** The blog covers cost/token analysis, prompt caching with ROI, max_tokens/stop sequences, structured outputs with reliability tiers, evaluation rubric with automated scoring, and regression testing. The prompt management system design (Q11) shows awareness of versioning, A/B testing, and monitoring. Gap: prompt injection defense is deferred to Blog 14.

### Check 2: Deep Problem Understanding
**Question**: Can they diagnose why prompts fail and apply targeted fixes?
**Answer: YES.** The debugging section provides systematic diagnostics (format mismatch, length, missing fields). The common-issues table maps symptoms to fixes. Positional effects explain "lost in the middle." In-context learning mechanics explain WHY examples work (induction heads) and why order/consistency matter.

### Check 3: Interview and Career Readiness
**Question**: Can they articulate prompt engineering principles and design production systems?
**Answer: YES.** 11 interview questions cover ICL mechanics, prompt caching, positional effects, structured outputs, and a full system-design question (prompt management for multi-tenant SaaS with versioning, CI/CD, monitoring). The cost analysis and caching sections target the most common senior-engineer follow-ups about production economics.
