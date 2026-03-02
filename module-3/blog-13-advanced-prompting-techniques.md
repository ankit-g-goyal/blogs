# Blog 13: Advanced Prompting Techniques — Unlocking Hidden Capabilities

**Series:** Prompt Your Career: The Complete Generative AI Masterclass
**Prerequisites:** Blog 12 (Prompt Engineering Fundamentals)
**Time to Complete:** 3.5-4 hours
**Difficulty:** Intermediate to Advanced

---

## What You'll Walk Away With

After completing this blog, you will be able to:

1. **Implement Chain-of-Thought (CoT)** prompting for complex reasoning tasks
2. **Apply self-consistency** and majority voting for improved accuracy
3. **Use Tree-of-Thoughts (ToT)** for exploring multiple reasoning paths
4. **Build ReAct agents** that combine reasoning with action
5. **Decompose complex tasks** with prompt chaining
6. **Understand meta-prompting** and automatic prompt optimization
7. **Evaluate which technique to use** for different problem types

---

## Manager's Summary

**Why Advanced Prompting Matters:**

Standard prompting often struggles on complex reasoning tasks. Advanced techniques can substantially close that gap — the difference between "interesting experiment" and "production-ready system." The gains below are approximate ranges drawn from published research (Wei et al. 2022 for CoT, Wang et al. 2023 for Self-Consistency, Yao et al. 2023 for ToT and ReAct). Your results will vary by model, task, and prompt quality.

**Technique ROI Matrix (approximate, from published benchmarks):**

| Technique | Typical Accuracy Gain | Cost Increase | Best For |
|-----------|----------------------|---------------|----------|
| **Chain-of-Thought** | +10-25% on reasoning tasks (Wei et al. 2022) | 1.5-2x tokens | Math, logic, multi-step |
| **Self-Consistency** | +5-10% over single CoT (Wang et al. 2023) | 3-5x tokens | Critical decisions |
| **Tree-of-Thoughts** | +10-20% on search/planning (Yao et al. 2023) | 5-10x tokens | Creative problem-solving |
| **ReAct** | Significant on knowledge tasks (Yao et al. 2023) | Variable | Tasks needing external data |
| **Prompt Chaining** | Task-dependent | 1.5-3x tokens | Complex workflows |

**When to Invest in Advanced Prompting:**

| Scenario | Technique Priority | Investment Level |
|----------|-------------------|------------------|
| Math/logic tasks | CoT → Self-Consistency | Medium |
| High-stakes decisions | Self-Consistency → Human review | High |
| Creative exploration | ToT → CoT | High |
| Research with tools | ReAct → Prompt Chaining | Medium |
| Multi-step workflows | Prompt Chaining → CoT | Medium |

**Risk Considerations:**
- Higher token costs (3-10x baseline)
- Increased latency (multiple LLM calls)
- Complexity in production systems
- Debugging difficulty increases

> **How to read this blog:** If you already use basic prompting (few-shot, zero-shot, system prompts from Blog 12), start anywhere — each technique section stands alone. If you're new to prompt engineering, read the sections in order: CoT first, then Self-Consistency, then ToT, then ReAct, then Prompt Chaining. The Manager's Summary gives a quick overview if you need to decide which techniques are relevant to your use case. Code examples are pseudocode-style (they define classes and prompt templates but require an actual LLM client to run).

---

## What This Blog Does NOT Cover

Before we begin, let's set clear expectations on scope:

- **LangChain / LlamaIndex frameworks** — framework-level prompt orchestration is covered in Blog 19 (LangChain Deep Dive). This blog teaches the underlying techniques those frameworks implement.
- **Fine-tuning vs prompting tradeoffs** — when to abandon prompting and fine-tune a model is covered in Blog 23.
- **RAG (Retrieval-Augmented Generation)** — combining prompting with document retrieval is covered in Blog 17. This blog focuses on prompting techniques in isolation.
- **Prompt injection and security** — adversarial attacks on prompts and defenses are important production concerns but are outside this blog's scope.
- **Multi-modal prompting** — techniques for vision-language models are covered in Blog 21.
- **Benchmark-specific evaluation** — we discuss when each technique helps conceptually, but rigorous benchmark comparisons (GSM8K, MMLU, HotpotQA) are referenced via papers rather than reproduced here.

---

## Chain-of-Thought Prompting

### The Breakthrough Discovery

In 2022, Wei et al. ("Chain-of-Thought Prompting Elicits Reasoning in Large Language Models") showed that including intermediate reasoning steps in prompts dramatically improved performance on complex reasoning benchmarks. Separately, Kojima et al. (2022) found that simply adding "Let's think step by step" (zero-shot CoT) also yields significant gains. This family of techniques became known as Chain-of-Thought (CoT) prompting.

```
Standard Prompting:
Q: Roger has 5 tennis balls. He buys 2 cans with 3 balls each. How many total?
A: 11

Chain-of-Thought Prompting:
Q: Roger has 5 tennis balls. He buys 2 cans with 3 balls each. How many total?
A: Let me think step by step.
   Roger starts with 5 balls.
   He buys 2 cans × 3 balls = 6 balls.
   Total = 5 + 6 = 11 balls.
   The answer is 11.
```

### Why CoT Works: The Computational Argument

Chain-of-Thought isn't just a "trick" — it addresses a fundamental limitation of autoregressive transformers. A transformer with L layers and d-dimensional hidden states can only compose a fixed number of sequential computations per forward pass. For problems requiring more sequential reasoning steps than the model's effective depth, a single forward pass *cannot* produce the correct answer — the model lacks sufficient "compute."

CoT solves this by **offloading intermediate state into the generated tokens themselves.** Each reasoning step becomes part of the context for subsequent generation, effectively giving the model external working memory. This is why "Let's think step by step" helps: it forces the model to produce intermediate tokens that encode partial results, which then feed into the next forward pass.

**Formal intuition (Feng et al., 2023):** A constant-depth transformer cannot solve problems requiring O(n) sequential steps (like multi-digit addition) in a single forward pass. But with CoT, each generated token adds one step of computation, enabling the model to solve problems requiring O(n) steps by generating O(n) intermediate tokens.

```
Without CoT:                         With CoT:
┌─────────────────┐                 ┌─────────────────┐
│     Problem     │                 │     Problem     │
│                 │                 │                 │
│  "5 + 2×3 = ?"  │                 │  "5 + 2×3 = ?"  │
└────────┬────────┘                 └────────┬────────┘
         │                                   │
         │ Direct                            │ Step-by-step
         │ mapping                           │ reasoning
         ▼                                   ▼
┌─────────────────┐                 ┌─────────────────┐
│  Answer: "11"   │                 │  Step 1: 5      │
│  (or "7"?)      │                 │  Step 2: 2×3=6  │
│                 │                 │  Step 3: 5+6=11 │
└─────────────────┘                 └────────┬────────┘
                                             ▼
                                    ┌─────────────────┐
                                    │  Answer: 11     │
                                    │  (verified!)    │
                                    └─────────────────┘

CoT forces the model to:
1. Decompose the problem
2. Show intermediate work
3. Catch errors in reasoning
4. Build toward final answer
```

### CoT Implementation

```python
"""
Chain-of-Thought implementation patterns.
"""

# Pattern 1: Zero-shot CoT (simplest)
zero_shot_cot = """
{question}

Let's think step by step.
"""

# Pattern 2: Few-shot CoT (more reliable)
few_shot_cot = """
Q: A store had 42 apples. They sold 12 in the morning and 15 in the afternoon.
How many apples are left?

A: Let me solve this step by step.
1. Starting apples: 42
2. Sold in morning: 12
3. Remaining after morning: 42 - 12 = 30
4. Sold in afternoon: 15
5. Remaining after afternoon: 30 - 15 = 15
The answer is 15 apples.

Q: A train travels at 60 mph for 2 hours, then 80 mph for 1.5 hours.
What is the total distance traveled?

A: Let me solve this step by step.
1. First leg: 60 mph × 2 hours = 120 miles
2. Second leg: 80 mph × 1.5 hours = 120 miles
3. Total distance: 120 + 120 = 240 miles
The answer is 240 miles.

Q: {question}

A: Let me solve this step by step.
"""

# Pattern 3: Structured CoT with explicit format
structured_cot = """
Solve this problem using the following format:

Problem: {problem}

## Given Information
[List the key facts and numbers]

## What We Need to Find
[State the goal clearly]

## Solution Steps
1. [First step with calculation]
2. [Second step with calculation]
...

## Verification
[Check the answer makes sense]

## Final Answer
[State the answer clearly]
"""

def create_cot_prompt(problem, style="few_shot", examples=None):
    """
    Create a Chain-of-Thought prompt.

    Args:
        problem: The problem to solve
        style: "zero_shot", "few_shot", or "structured"
        examples: List of (question, cot_answer) tuples for few-shot
    """
    if style == "zero_shot":
        return f"{problem}\n\nLet's think step by step."

    elif style == "few_shot":
        prompt = ""
        if examples:
            for q, a in examples:
                prompt += f"Q: {q}\n\nA: {a}\n\n"
        prompt += f"Q: {problem}\n\nA: Let's think step by step.\n"
        return prompt

    elif style == "structured":
        return structured_cot.format(problem=problem)

    else:
        raise ValueError(f"Unknown style: {style}")


# Example usage:
problem = """
A farmer has 3 fields. The first field produces 250 bushels per acre and is 40 acres.
The second field produces 300 bushels per acre and is 35 acres.
The third field produces 275 bushels per acre and is 50 acres.
What is the total production across all fields?
"""

# Different CoT approaches:
print("Zero-shot CoT:")
print(create_cot_prompt(problem, style="zero_shot"))

print("\n" + "="*50 + "\n")

print("Structured CoT:")
print(create_cot_prompt(problem, style="structured"))
```

### When CoT Helps (and When It Doesn't)

```python
"""
CoT effectiveness varies dramatically by task type.
"""

# Note: Improvement ranges are approximate, drawn from Wei et al. 2022
# (GSM8K, SVAMP, CommonsenseQA benchmarks with PaLM 540B).
# Your results will vary by model size, task, and prompt design.
COT_EFFECTIVENESS = {
    "Math word problems": {
        "improvement": "High (Wei et al. report ~+20% on GSM8K with PaLM 540B)",
        "why": "Forces decomposition of multi-step calculations",
        "example": "Compute total cost with discounts and tax",
    },
    "Logical reasoning": {
        "improvement": "High (significant gains on commonsense/symbolic reasoning benchmarks)",
        "why": "Makes deduction chains explicit",
        "example": "If A implies B, and B implies C, what about A→C?",
    },
    "Multi-hop QA": {
        "improvement": "Moderate (helps track information across multiple facts)",
        "why": "Tracks information across multiple facts",
        "example": "Who is the spouse of the CEO of the company that made iPhone?",
    },
    "Code debugging": {
        "improvement": "Moderate (anecdotal; forces line-by-line analysis)",
        "why": "Forces line-by-line analysis",
        "example": "Find the bug in this function",
    },
    "Simple factual QA": {
        "improvement": "Low (minimal — no reasoning needed, just retrieval)",
        "why": "No reasoning needed, just retrieval",
        "example": "What is the capital of France?",
    },
    "Sentiment analysis": {
        "improvement": "Low (minimal — intuitive task, reasoning adds little)",
        "why": "Intuitive task, reasoning doesn't help",
        "example": "Is this review positive or negative?",
    },
    "Translation": {
        "improvement": "Minimal to negative (can hurt fluency by over-analyzing)",
        "why": "Can hurt fluency by over-analyzing",
        "example": "Translate this sentence to French",
    },
}

def should_use_cot(task_type, accuracy_requirement):
    """
    Recommend whether to use Chain-of-Thought.

    Args:
        task_type: Type of task
        accuracy_requirement: "low", "medium", "high"
    """
    effectiveness = COT_EFFECTIVENESS.get(task_type, {})
    improvement = effectiveness.get("improvement", "Unknown")

    if "High" in improvement:
        return True, "CoT significantly improves this task type"
    elif "Moderate" in improvement and accuracy_requirement in ["medium", "high"]:
        return True, "CoT helps; worth the extra tokens for your accuracy needs"
    elif "Low" in improvement or "negative" in improvement:
        return False, "CoT provides minimal benefit; simpler prompts recommended"
    else:
        return "Maybe", "Test both approaches on your specific data"
```

### CoT Failure Modes

```python
"""
CoT is not universally beneficial. Knowing when it fails is as important as knowing when it helps.
"""

COT_FAILURE_MODES = {
    "Faithful but wrong reasoning": {
        "description": "Model produces plausible-looking steps that contain a subtle error, "
                       "then confidently reaches a wrong answer",
        "example": "Q: 23 × 17. CoT: 23 × 10 = 230, 23 × 7 = 151 [error: should be 161], "
                   "Total = 381 [wrong]. The step-by-step format makes the wrong answer LOOK correct.",
        "mitigation": "Pair CoT with Self-Consistency (multiple paths catch arithmetic errors) "
                      "or external verification (calculator tool via ReAct)",
    },
    "Verbose nonsense on easy tasks": {
        "description": "Model over-reasons on simple tasks, adding unnecessary steps that "
                       "can introduce errors where none would exist with direct answering",
        "example": "Q: What is the capital of France? CoT: 'Let me think... France is in Europe... "
                   "European capitals include London, Berlin, Paris... considering historical context...' "
                   "→ slower, more tokens, no accuracy gain",
        "mitigation": "Use CoT only for tasks scoring 'High' or 'Moderate' in COT_EFFECTIVENESS. "
                      "For simple retrieval tasks, direct prompting is better.",
    },
    "Reasoning chain collapse on very long problems": {
        "description": "For problems requiring 10+ reasoning steps, models lose track of earlier "
                       "steps — a form of 'lost in the middle' applied to self-generated context",
        "mitigation": "Break into sub-problems via Prompt Chaining, or use Structured CoT "
                      "with explicit 'Given', 'Step N', 'Verify' sections to anchor each step",
    },
    "Sycophantic reasoning": {
        "description": "If the prompt contains a hint (even wrong), the model's CoT will "
                       "rationalize the hinted answer rather than reason independently",
        "mitigation": "Never include answer hints in CoT prompts. Use neutral framing.",
    },
}
```

---

## Self-Consistency: Wisdom of the Crowd

### The Core Idea

Self-consistency samples multiple reasoning paths and takes the majority vote. It's based on the insight that correct answers are more likely to be reached via diverse reasoning paths.

```
                         Problem
                            │
            ┌───────────────┼───────────────┐
            │               │               │
            ▼               ▼               ▼
       Path 1 (T=0.7)  Path 2 (T=0.7)  Path 3 (T=0.7)
            │               │               │
            ▼               ▼               ▼
        ┌───────┐       ┌───────┐       ┌───────┐
        │Step 1 │       │Step 1'│       │Step 1"│
        │Step 2 │       │Step 2'│       │Step 2"│
        │Step 3 │       │Step 3'│       │Step 3"│
        └───┬───┘       └───┬───┘       └───┬───┘
            │               │               │
            ▼               ▼               ▼
        Answer: 42      Answer: 42      Answer: 37
            │               │               │
            └───────────────┼───────────────┘
                            │
                            ▼
                    Majority Vote
                            │
                            ▼
                    Final: 42 (2/3)
```

### Implementation

```python
"""
Self-consistency implementation.
"""

from collections import Counter
import re

class SelfConsistency:
    """
    Self-consistency with majority voting.
    """

    def __init__(self, llm_client, num_samples=5, temperature=0.7):
        """
        Initialize self-consistency.

        Args:
            llm_client: Function that takes prompt and returns response
            num_samples: Number of reasoning paths to sample
            temperature: Temperature for diverse sampling
        """
        self.llm = llm_client
        self.num_samples = num_samples
        self.temperature = temperature

    def solve(self, prompt, answer_extractor=None):
        """
        Solve a problem using self-consistency.

        Args:
            prompt: The problem prompt (should encourage CoT)
            answer_extractor: Function to extract final answer from response
        """
        responses = []
        answers = []

        # Sample multiple reasoning paths
        for i in range(self.num_samples):
            response = self.llm(
                prompt,
                temperature=self.temperature
            )
            responses.append(response)

            # Extract answer
            if answer_extractor:
                answer = answer_extractor(response)
            else:
                answer = self._default_extract(response)

            answers.append(answer)

        # Majority vote
        counter = Counter(answers)
        most_common = counter.most_common(1)[0]
        final_answer = most_common[0]
        confidence = most_common[1] / len(answers)

        return {
            "answer": final_answer,
            "confidence": confidence,
            "all_answers": answers,
            "distribution": dict(counter),
            "responses": responses
        }

    def _default_extract(self, response):
        """Extract answer from response (default: last number)."""
        numbers = re.findall(r'-?\d+\.?\d*', response)
        return numbers[-1] if numbers else response.strip().split()[-1]


def extract_multiple_choice(response):
    """Extract letter answer from multiple choice response."""
    # Look for patterns like "The answer is A" or "(A)" or "A."
    patterns = [
        r'answer is ([A-D])',
        r'answer: ([A-D])',
        r'\(([A-D])\)',
        r'^([A-D])\.',
    ]
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return None


# Example usage (pseudocode):
"""
sc = SelfConsistency(my_llm_client, num_samples=5)

result = sc.solve(
    prompt='''
    Q: A bat and ball cost $1.10. The bat costs $1 more than the ball.
    How much does the ball cost?

    Think step by step before giving your final answer.
    ''',
    answer_extractor=lambda r: re.search(r'\$?([\d.]+)', r.split('answer')[-1]).group(1)
)

print(f"Answer: ${result['answer']}")
print(f"Confidence: {result['confidence']:.0%}")
print(f"Distribution: {result['distribution']}")
"""
```

### Self-Consistency Configurations

```python
"""
Different self-consistency configurations for different needs.
"""

CONFIGURATIONS = {
    "quick": {
        "num_samples": 3,
        "temperature": 0.5,
        "use_case": "Low-stakes, fast results needed",
        "cost_multiplier": "3x",
    },
    "standard": {
        "num_samples": 5,
        "temperature": 0.7,
        "use_case": "General purpose, good balance",
        "cost_multiplier": "5x",
    },
    "thorough": {
        "num_samples": 10,
        "temperature": 0.8,
        "use_case": "High-stakes, accuracy critical",
        "cost_multiplier": "10x",
    },
    "diverse": {
        "num_samples": 7,
        "temperature": 1.0,
        "use_case": "Complex problems, need diverse paths",
        "cost_multiplier": "7x",
    },
}

def weighted_voting(answers, confidences):
    """
    Weighted voting based on per-response confidence.

    Args:
        answers: List of answers
        confidences: List of confidence scores for each answer
    """
    weighted_counts = {}

    for answer, conf in zip(answers, confidences):
        if answer not in weighted_counts:
            weighted_counts[answer] = 0
        weighted_counts[answer] += conf

    return max(weighted_counts, key=weighted_counts.get)


def free_text_voting(responses: list[str], llm_client) -> dict:
    """
    Self-consistency for free-text outputs where exact string matching fails.

    For numeric/classification tasks, majority vote works directly.
    For open-ended text (summaries, explanations), use semantic clustering:
    1. Generate N responses
    2. Use an LLM to cluster them into semantic groups
    3. Pick the largest cluster's representative

    This costs one extra LLM call but handles free-text robustly.
    """
    cluster_prompt = f"""Below are {len(responses)} responses to the same question.
Group them by semantic meaning (responses saying essentially the same thing = same group).
Return JSON: {{"groups": [{{"members": [0, 2, 4], "summary": "..."}}, ...]}}

Responses:
"""
    for i, r in enumerate(responses):
        cluster_prompt += f"\n[{i}] {r[:200]}..."  # Truncate for token efficiency

    clustering = llm_client(cluster_prompt)
    # Parse and return largest group's representative
    import json
    groups = json.loads(clustering)["groups"]
    largest = max(groups, key=lambda g: len(g["members"]))
    representative_idx = largest["members"][0]
    return {
        "answer": responses[representative_idx],
        "cluster_size": len(largest["members"]),
        "confidence": len(largest["members"]) / len(responses),
        "num_clusters": len(groups),
    }
```

### Self-Consistency Failure Modes

```python
SC_FAILURE_MODES = {
    "Systematic bias convergence": {
        "description": "All N samples converge on the same WRONG answer because the model "
                       "has a systematic bias (e.g., always rounds down, always picks option A)",
        "mitigation": "Increase temperature to force diversity. If all 5 samples agree on "
                      "the same answer, that's NOT necessarily high confidence — check against "
                      "a ground truth sample.",
    },
    "Diverse but all wrong": {
        "description": "Samples produce different answers but none is correct. Majority vote "
                       "picks the most popular wrong answer.",
        "mitigation": "Self-consistency only helps when the correct answer has a higher "
                      "probability than any individual wrong answer. For very hard problems, "
                      "combine with external verification (ReAct with calculator).",
    },
    "High cost, marginal gain": {
        "description": "5-10x token cost for 2-3% accuracy improvement on tasks where the "
                       "model already achieves 90%+.",
        "mitigation": "Use compare_techniques framework (below) to measure ROI before deploying.",
    },
}
```

---

## Tree-of-Thoughts: Exploring Multiple Paths

### The Concept

Tree-of-Thoughts (ToT) extends CoT by explicitly exploring multiple reasoning branches and evaluating them. It's like playing chess—considering multiple moves before choosing.

```
                            Problem
                               │
                               ▼
                    ┌────────────────────┐
                    │  Initial Thoughts  │
                    └────────────────────┘
                               │
            ┌──────────────────┼──────────────────┐
            │                  │                  │
            ▼                  ▼                  ▼
       ┌─────────┐        ┌─────────┐        ┌─────────┐
       │Thought A│        │Thought B│        │Thought C│
       │(eval: 3)│        │(eval: 7)│        │(eval: 5)│
       └────┬────┘        └────┬────┘        └────┬────┘
            │                  │                  │
            │             ┌────┴────┐             │
         Pruned           │         │          Pruned
                          ▼         ▼
                     ┌─────────┐ ┌─────────┐
                     │ B1 (8)  │ │ B2 (6)  │
                     └────┬────┘ └────┬────┘
                          │           │
                          ▼        Pruned
                     ┌─────────┐
                     │ Solution│
                     │  Found! │
                     └─────────┘
```

### ToT Implementation

```python
"""
Tree-of-Thoughts implementation.
"""

class TreeOfThoughts:
    """
    Tree-of-Thoughts reasoning framework.
    """

    def __init__(self, llm_client, max_depth=3, branch_factor=3, pruning_threshold=5):
        """
        Initialize ToT.

        Args:
            llm_client: Function for LLM calls
            max_depth: Maximum reasoning depth
            branch_factor: Number of thoughts to generate at each step
            pruning_threshold: Minimum score to continue exploring (1-10)
        """
        self.llm = llm_client
        self.max_depth = max_depth
        self.branch_factor = branch_factor
        self.pruning_threshold = pruning_threshold

    def generate_thoughts(self, problem, current_state):
        """Generate possible next thoughts."""
        prompt = f"""
Problem: {problem}

Current reasoning state:
{current_state if current_state else "Starting fresh"}

Generate {self.branch_factor} distinct next steps or approaches to continue solving this problem.
Each approach should be different and explore a unique angle.

Format:
THOUGHT 1: [description]
THOUGHT 2: [description]
THOUGHT 3: [description]
"""
        response = self.llm(prompt, temperature=0.8)
        thoughts = self._parse_thoughts(response)
        return thoughts

    def evaluate_thought(self, problem, thought_path):
        """Evaluate how promising a thought path is."""
        prompt = f"""
Problem: {problem}

Reasoning path so far:
{thought_path}

Evaluate this reasoning path on a scale of 1-10:
- 10: Directly leads to solution
- 7-9: Very promising, clear progress
- 4-6: Potentially useful, needs more work
- 1-3: Unlikely to lead to solution

Score: [number]
Reasoning: [brief explanation]
"""
        response = self.llm(prompt, temperature=0.3)
        score = self._extract_score(response)
        return score

    def solve(self, problem):
        """
        Solve problem using Tree-of-Thoughts.
        """
        # Initialize
        root = {"state": "", "score": 10, "children": [], "depth": 0}
        best_solution = None
        best_score = 0

        # BFS/DFS exploration
        queue = [root]

        while queue:
            node = queue.pop(0)  # BFS (use pop() for DFS)

            if node["depth"] >= self.max_depth:
                # Check if this is a solution
                if node["score"] > best_score:
                    best_score = node["score"]
                    best_solution = node["state"]
                continue

            # Generate children
            thoughts = self.generate_thoughts(problem, node["state"])

            for thought in thoughts:
                new_state = f"{node['state']}\n→ {thought}" if node["state"] else thought
                score = self.evaluate_thought(problem, new_state)

                child = {
                    "state": new_state,
                    "score": score,
                    "children": [],
                    "depth": node["depth"] + 1
                }

                node["children"].append(child)

                # Prune low-scoring branches
                if score >= self.pruning_threshold:
                    queue.append(child)

        return {
            "solution": best_solution,
            "score": best_score,
            "tree": root
        }

    def _parse_thoughts(self, response):
        """Parse generated thoughts from response."""
        thoughts = []
        for line in response.split('\n'):
            if line.strip().startswith('THOUGHT'):
                thought = line.split(':', 1)[1].strip() if ':' in line else line
                thoughts.append(thought)
        return thoughts[:self.branch_factor]

    def _extract_score(self, response):
        """Extract numerical score from evaluation response."""
        import re
        match = re.search(r'Score:\s*(\d+)', response)
        return int(match.group(1)) if match else 5


# Simplified ToT prompt for single LLM call
TOT_SINGLE_PROMPT = """
Problem: {problem}

Explore this problem using Tree-of-Thoughts reasoning.

For each step:
1. Generate 2-3 possible approaches
2. Briefly evaluate each (promising/neutral/unpromising)
3. Select the most promising and continue
4. Repeat until you reach a solution

Show your exploration:

STEP 1 - Initial approaches:
- Approach A: [description] → [evaluation]
- Approach B: [description] → [evaluation]
- Approach C: [description] → [evaluation]
→ Selecting: [choice]

STEP 2 - Continuing from [choice]:
- Approach A': [description] → [evaluation]
- Approach B': [description] → [evaluation]
→ Selecting: [choice]

[Continue until solution]

FINAL ANSWER: [solution]
"""
```

### When to Use ToT

```python
"""
ToT is expensive (5-10x cost). Use it wisely.
"""

TOT_USE_CASES = {
    "Creative problem solving": {
        "recommended": True,
        "example": "Design a product that solves X problem",
        "why": "Benefits from exploring diverse approaches",
    },
    "Game playing / puzzles": {
        "recommended": True,
        "example": "Solve this Sudoku / Game of 24",
        "why": "Search space benefits from systematic exploration",
    },
    "Planning complex tasks": {
        "recommended": True,
        "example": "Plan a marketing campaign",
        "why": "Multiple valid strategies worth comparing",
    },
    "Mathematical proofs": {
        "recommended": True,
        "example": "Prove this theorem",
        "why": "Different proof strategies exist",
    },
    "Simple calculations": {
        "recommended": False,
        "example": "What is 15% of 240?",
        "why": "Single path suffices; ToT is overkill",
    },
    "Factual questions": {
        "recommended": False,
        "example": "What year was Python created?",
        "why": "No reasoning paths to explore",
    },
}
```

---

## ReAct: Reasoning + Acting

### The ReAct Framework

ReAct combines reasoning (thinking) with acting (using tools). The model alternates between thinking about what to do and taking actions.

```
┌────────────────────────────────────────────────────────────────┐
│                        ReAct Loop                              │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│   Question: "What is the population of the capital of Japan?"  │
│                                                                │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │ Thought 1: I need to find the capital of Japan first.   │  │
│   │ Action 1: search("capital of Japan")                    │  │
│   │ Observation 1: The capital of Japan is Tokyo.           │  │
│   └─────────────────────────────────────────────────────────┘  │
│                           │                                    │
│                           ▼                                    │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │ Thought 2: Now I need to find Tokyo's population.       │  │
│   │ Action 2: search("population of Tokyo")                 │  │
│   │ Observation 2: Tokyo has ~14 million in city proper.    │  │
│   └─────────────────────────────────────────────────────────┘  │
│                           │                                    │
│                           ▼                                    │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │ Thought 3: I now have the answer.                       │  │
│   │ Action 3: finish("14 million")                          │  │
│   └─────────────────────────────────────────────────────────┘  │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### ReAct Implementation

```python
"""
ReAct implementation with tool use.
"""

import re
from typing import Dict, Callable, List

class ReActAgent:
    """
    ReAct agent that combines reasoning with action.
    """

    def __init__(self, llm_client, tools: Dict[str, Callable], max_steps=10):
        """
        Initialize ReAct agent.

        Args:
            llm_client: Function for LLM calls
            tools: Dictionary mapping tool names to functions
            max_steps: Maximum reasoning steps
        """
        self.llm = llm_client
        self.tools = tools
        self.max_steps = max_steps

    def _build_tool_descriptions(self):
        """Build tool descriptions for the prompt."""
        descriptions = []
        for name, func in self.tools.items():
            doc = func.__doc__ or "No description available"
            descriptions.append(f"- {name}: {doc.strip()}")
        return "\n".join(descriptions)

    def run(self, question: str) -> dict:
        """
        Run the ReAct loop to answer a question.

        Returns:
            Dictionary with answer and reasoning trace
        """
        tool_desc = self._build_tool_descriptions()

        system_prompt = f"""
You are a helpful assistant that answers questions by reasoning and using tools.

Available tools:
{tool_desc}

Always use this format:

Thought: [your reasoning about what to do next]
Action: [tool_name]("argument")
Observation: [will be filled in with tool result]

Repeat Thought/Action/Observation until you can answer.
When ready to give final answer:

Thought: I now have enough information to answer.
Action: finish("your final answer")

Important:
- Always think before acting
- Use tools to get facts, don't make them up
- If a tool fails, try a different approach
"""

        # Initialize conversation
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Question: {question}"}
        ]

        trace = []

        for step in range(self.max_steps):
            # Get LLM response
            response = self.llm(messages)

            # Parse thought and action
            thought, action, action_arg = self._parse_response(response)

            trace.append({
                "step": step + 1,
                "thought": thought,
                "action": action,
                "action_arg": action_arg
            })

            # Check for finish
            if action == "finish":
                return {
                    "answer": action_arg,
                    "trace": trace,
                    "steps": step + 1
                }

            # Execute action
            if action in self.tools:
                try:
                    observation = self.tools[action](action_arg)
                except Exception as e:
                    observation = f"Error: {str(e)}"
            else:
                observation = f"Error: Unknown tool '{action}'"

            trace[-1]["observation"] = observation

            # Add to conversation
            messages.append({"role": "assistant", "content": response})
            messages.append({
                "role": "user",
                "content": f"Observation: {observation}"
            })

        return {
            "answer": None,
            "trace": trace,
            "error": "Max steps reached without answer"
        }

    def _parse_response(self, response: str):
        """Parse thought and action from response."""
        thought = ""
        action = ""
        action_arg = ""

        # Extract thought
        thought_match = re.search(r'Thought:\s*(.+?)(?=\nAction:|$)', response, re.DOTALL)
        if thought_match:
            thought = thought_match.group(1).strip()

        # Extract action
        action_match = re.search(r'Action:\s*(\w+)\("([^"]*)"\)', response)
        if action_match:
            action = action_match.group(1)
            action_arg = action_match.group(2)

        return thought, action, action_arg


# Example tools
def search(query: str) -> str:
    """Search for information on the web."""
    # In production, this would call a real search API
    fake_results = {
        "capital of Japan": "Tokyo is the capital of Japan.",
        "population of Tokyo": "Tokyo has approximately 14 million people.",
        "Python creator": "Python was created by Guido van Rossum.",
    }
    for key, value in fake_results.items():
        if key.lower() in query.lower():
            return value
    return f"No results found for: {query}"

def calculate(expression: str) -> str:
    """Evaluate a mathematical expression safely."""
    import ast
    try:
        # ast.literal_eval safely evaluates literals; for arithmetic,
        # we compile and evaluate with no builtins to prevent code injection.
        # Only allow basic math operations.
        allowed_names = {"__builtins__": {}}
        tree = compile(expression, "<string>", "eval")
        result = eval(tree, allowed_names)
        return str(result)
    except (SyntaxError, TypeError, ValueError, NameError, ZeroDivisionError) as e:
        return f"Calculation error: {e}"

def lookup(entity: str) -> str:
    """Look up information about an entity in a knowledge base."""
    kb = {
        "Python": "Python is a programming language created in 1991.",
        "Tokyo": "Tokyo is the capital of Japan with 14 million people.",
    }
    return kb.get(entity, f"No information found for: {entity}")


# Example usage:
"""
agent = ReActAgent(
    llm_client=my_llm,
    tools={
        "search": search,
        "calculate": calculate,
        "lookup": lookup
    }
)

result = agent.run("What is the square root of the population of Tokyo?")
print(f"Answer: {result['answer']}")
for step in result['trace']:
    print(f"Step {step['step']}:")
    print(f"  Thought: {step['thought']}")
    print(f"  Action: {step['action']}({step['action_arg']})")
    print(f"  Observation: {step.get('observation', 'N/A')}")
"""
```

### ReAct Failure Modes and Production Hardening

```python
"""
ReAct agents can fail in ways that are expensive and hard to debug.
"""

REACT_FAILURE_MODES = {
    "Infinite loops": {
        "description": "Agent repeats the same thought-action cycle without making progress. "
                       "Common when the tool returns unhelpful results and the model can't adapt.",
        "mitigation": "Track action history; if the same action+arg appears twice, force a "
                      "different approach or escalate. The max_steps parameter is a blunt safeguard.",
    },
    "Tool hallucination": {
        "description": "Model invents tool names that don't exist, or passes malformed arguments. "
                       "The 'Action: search(\"...\")' format is fragile.",
        "mitigation": "Validate action names against the tool registry. Use structured output "
                      "(function calling) instead of string parsing for production ReAct agents.",
    },
    "Premature finish": {
        "description": "Agent calls finish() with a partial or wrong answer before gathering "
                       "sufficient information.",
        "mitigation": "Add a verification step: before finish(), prompt the model to check "
                      "whether the answer fully addresses the original question.",
    },
    "Context window exhaustion": {
        "description": "Each thought-action-observation cycle adds to the context. After 5-10 "
                       "steps, the context can exceed limits, especially with long observations.",
        "mitigation": "Summarize earlier steps periodically. Truncate long observations. "
                      "Set max_observation_tokens per tool call.",
    },
}
```

### ReAct vs Standard Prompting

```python
"""
Compare ReAct to standard prompting on knowledge-intensive tasks.
"""

REACT_COMPARISON = {
    "Without ReAct (Standard)": {
        "query": "What company acquired the creator of Java?",
        "response": "Sun Microsystems created Java... Oracle acquired Sun...",
        "issues": [
            "May hallucinate if knowledge is outdated",
            "Can't verify current information",
            "Single-pass reasoning"
        ]
    },
    "With ReAct": {
        "query": "What company acquired the creator of Java?",
        "trace": [
            "Thought: First find who created Java",
            "Action: search('Java creator')",
            "Obs: James Gosling at Sun Microsystems",
            "Thought: Now find who acquired Sun",
            "Action: search('Sun Microsystems acquired')",
            "Obs: Oracle acquired Sun in 2010",
            "Thought: I have the answer",
            "Action: finish('Oracle')"
        ],
        "advantages": [
            "Can look up current facts",
            "Transparent reasoning",
            "Can recover from errors"
        ]
    }
}
```

---

## Prompt Chaining: Breaking Down Complex Tasks

### The Concept

Prompt chaining decomposes complex tasks into simpler subtasks, passing outputs between steps. It's like an assembly line for AI tasks.

```
Complex Task: "Analyze this codebase and suggest improvements"

┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Step 1    │────>│   Step 2    │────>│   Step 3    │
│  List Files │     │   Analyze   │     │  Summarize  │
│             │     │   Quality   │     │  & Suggest  │
└─────────────┘     └─────────────┘     └─────────────┘
      │                   │                   │
      ▼                   ▼                   ▼
 file_list.json     analysis.json      report.md
```

### Implementation

```python
"""
Prompt chaining for complex workflows.
"""

from typing import List, Dict, Any, Callable
import json

class PromptChain:
    """
    Chain multiple prompts together into a workflow.
    """

    def __init__(self, llm_client):
        self.llm = llm_client
        self.steps: List[Dict] = []
        self.results: Dict[str, Any] = {}

    def add_step(self, name: str, prompt_template: str,
                 output_parser: Callable = None,
                 input_mapper: Callable = None):
        """
        Add a step to the chain.

        Args:
            name: Unique name for this step
            prompt_template: Template with {placeholders}
            output_parser: Function to parse LLM output
            input_mapper: Function to transform previous results into inputs
        """
        self.steps.append({
            "name": name,
            "template": prompt_template,
            "parser": output_parser or (lambda x: x),
            "mapper": input_mapper or (lambda results: results)
        })
        return self  # Enable chaining

    def run(self, initial_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the chain.

        Args:
            initial_input: Initial inputs to the chain
        """
        self.results = initial_input.copy()

        for step in self.steps:
            print(f"Executing step: {step['name']}")

            # Map results to inputs
            inputs = step["mapper"](self.results)

            # Format prompt
            try:
                prompt = step["template"].format(**inputs)
            except KeyError as e:
                raise ValueError(f"Missing input for step '{step['name']}': {e}")

            # Call LLM
            response = self.llm(prompt)

            # Parse output
            parsed = step["parser"](response)

            # Store result
            self.results[step["name"]] = parsed

        return self.results


# Example: Document Analysis Chain
def create_document_analysis_chain(llm):
    """
    Create a chain for comprehensive document analysis.
    """
    chain = PromptChain(llm)

    # Step 1: Extract key information
    chain.add_step(
        name="extraction",
        prompt_template="""
Extract the key information from this document:

Document:
{document}

Extract:
1. Main topic (1 sentence)
2. Key entities (people, organizations, places)
3. Key facts (5-10 bullet points)
4. Sentiment (positive/negative/neutral)

Format as JSON.
""",
        output_parser=json.loads
    )

    # Step 2: Generate questions
    chain.add_step(
        name="questions",
        prompt_template="""
Based on this extracted information:
{extraction}

Generate 5 thought-provoking questions that this document raises or leaves unanswered.
Format as a JSON list of strings.
""",
        output_parser=json.loads,
        input_mapper=lambda r: {"extraction": json.dumps(r["extraction"])}
    )

    # Step 3: Generate summary
    chain.add_step(
        name="summary",
        prompt_template="""
Original document:
{document}

Key information:
{extraction}

Write a concise executive summary (2-3 paragraphs) that captures:
- Main thesis/topic
- Supporting evidence
- Implications
""",
        input_mapper=lambda r: {
            "document": r["document"],
            "extraction": json.dumps(r["extraction"])
        }
    )

    return chain


# Example: Code Review Chain
def create_code_review_chain(llm):
    """
    Create a chain for thorough code review.
    """
    chain = PromptChain(llm)

    # Step 1: Understand the code
    chain.add_step(
        name="understanding",
        prompt_template="""
Analyze this code and explain what it does:

```{language}
{code}
```

Provide:
1. Overall purpose
2. Main functions/classes and their roles
3. Data flow
4. External dependencies
""",
    )

    # Step 2: Find issues
    chain.add_step(
        name="issues",
        prompt_template="""
Code understanding:
{understanding}

Original code:
{code}

Identify all issues:
1. Bugs or logical errors
2. Security vulnerabilities
3. Performance problems
4. Code smell / maintainability issues

For each issue, provide: location, severity, description.
Format as JSON list.
""",
        output_parser=json.loads,
        input_mapper=lambda r: {
            "understanding": r["understanding"],
            "code": r["code"]
        }
    )

    # Step 3: Suggest improvements
    chain.add_step(
        name="improvements",
        prompt_template="""
Based on these identified issues:
{issues}

And this code:
{code}

Provide specific code improvements with explanations.
For each improvement, show before/after code snippets.
""",
        input_mapper=lambda r: {
            "issues": json.dumps(r["issues"]),
            "code": r["code"]
        }
    )

    return chain


# Usage:
"""
chain = create_document_analysis_chain(my_llm)

results = chain.run({
    "document": "Your long document text here..."
})

print("Extracted info:", results["extraction"])
print("Generated questions:", results["questions"])
print("Summary:", results["summary"])
"""
```

### Error Propagation and Validation Gates

Chains have a critical risk: **if step N produces garbage, steps N+1 through end amplify it.** In production, you must add validation gates between steps:

```python
class ValidatedPromptChain(PromptChain):
    """
    Prompt chain with validation gates between steps.
    If a step fails validation, retry or halt — don't propagate garbage downstream.
    """

    def add_step(self, name: str, prompt_template: str,
                 output_parser: Callable = None,
                 input_mapper: Callable = None,
                 validator: Callable = None,
                 max_retries: int = 2):
        """
        Add a step with optional validation.

        Args:
            validator: Function(parsed_output) -> bool. If False, step is retried.
            max_retries: How many times to retry before raising.
        """
        self.steps.append({
            "name": name,
            "template": prompt_template,
            "parser": output_parser or (lambda x: x),
            "mapper": input_mapper or (lambda results: results),
            "validator": validator,
            "max_retries": max_retries,
        })
        return self

    def run(self, initial_input: Dict[str, Any]) -> Dict[str, Any]:
        self.results = initial_input.copy()

        for step in self.steps:
            inputs = step["mapper"](self.results)

            for attempt in range(step["max_retries"] + 1):
                prompt = step["template"].format(**inputs)
                response = self.llm(prompt)
                parsed = step["parser"](response)

                # Validate if validator provided
                if step["validator"] is None or step["validator"](parsed):
                    self.results[step["name"]] = parsed
                    break
                else:
                    if attempt == step["max_retries"]:
                        raise ValueError(
                            f"Step '{step['name']}' failed validation after "
                            f"{step['max_retries'] + 1} attempts. Last output: {str(parsed)[:200]}"
                        )
                    print(f"  Step '{step['name']}' failed validation (attempt {attempt + 1}), retrying...")

        return self.results


# Example: validated chain for document analysis
def is_valid_json_extraction(output):
    """Validate that extraction step produced required fields."""
    if not isinstance(output, dict):
        return False
    required = {"main_topic", "key_entities", "key_facts", "sentiment"}
    return required.issubset(output.keys())

# Usage:
# chain.add_step("extraction", template, output_parser=json.loads,
#                validator=is_valid_json_extraction, max_retries=2)
```

### Parallel vs Sequential Chaining

```python
"""
Sometimes steps can run in parallel.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor

class ParallelChain:
    """
    Chain with parallel execution support.
    """

    def __init__(self, llm_client, max_workers=3):
        self.llm = llm_client
        self.max_workers = max_workers
        self.steps = {}
        self.dependencies = {}

    def add_step(self, name: str, prompt_template: str,
                 depends_on: List[str] = None, **kwargs):
        """Add step with optional dependencies."""
        self.steps[name] = {
            "template": prompt_template,
            **kwargs
        }
        self.dependencies[name] = depends_on or []

    def run(self, initial_input: Dict) -> Dict:
        """Run chain with parallel execution where possible."""
        results = initial_input.copy()
        completed = set()

        while len(completed) < len(self.steps):
            # Find steps that can run (all dependencies met)
            ready = [
                name for name, deps in self.dependencies.items()
                if name not in completed and all(d in completed for d in deps)
            ]

            if not ready:
                raise ValueError("Circular dependency detected!")

            # Run ready steps in parallel
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(self._run_step, name, results): name
                    for name in ready
                }

                for future in futures:
                    name = futures[future]
                    results[name] = future.result()
                    completed.add(name)

        return results

    def _run_step(self, name: str, results: Dict) -> Any:
        """Run a single step."""
        step = self.steps[name]
        prompt = step["template"].format(**results)
        response = self.llm(prompt)
        parser = step.get("parser", lambda x: x)
        return parser(response)


# Parallel chain example:
"""
            ┌──────────────┐
            │    Input     │
            └──────┬───────┘
                   │
      ┌────────────┼────────────┐
      │            │            │
      ▼            ▼            ▼
┌──────────┐ ┌──────────┐ ┌──────────┐
│ Sentiment│ │ Entities │ │ Topics   │  (parallel)
└─────┬────┘ └─────┬────┘ └─────┬────┘
      │            │            │
      └────────────┼────────────┘
                   │
                   ▼
            ┌──────────────┐
            │   Combine    │  (sequential - depends on all)
            └──────────────┘
"""
```

---

## Meta-Prompting and Automatic Optimization

### Meta-Prompting: Prompts that Write Prompts

The idea of using LLMs to optimize prompts has evolved into a research subfield. Key established approaches:

| Approach | Paper | Mechanism | When to Use |
|----------|-------|-----------|-------------|
| **APE** (Automatic Prompt Engineer) | Zhou et al. 2023 | LLM generates prompt candidates, evaluates on held-out set | Simple tasks, small eval set |
| **OPRO** (Optimization by PROmpting) | Yang et al. 2023 | LLM iteratively improves prompts using past performance as context | Iterative refinement with clear metrics |
| **DSPy** | Khattab et al. 2023 | Compiles declarative programs into optimized prompts/fine-tuning | Multi-step pipelines, systematic optimization |

**DSPy is the most production-relevant.** Instead of manually crafting prompts, you define *signatures* (input/output specifications) and DSPy compiles them into optimized prompts or fine-tuning data using your evaluation metrics. This is the direction the field is moving — prompts as compiled artifacts, not hand-crafted strings.

The `PromptOptimizer` below demonstrates the core loop (generate → evaluate → improve) that these frameworks automate at scale:

```python
"""
Using LLMs to generate and improve prompts.
This is a simplified version of the loop that APE, OPRO, and DSPy automate.
"""

META_PROMPT_GENERATOR = """
You are an expert prompt engineer. Your task is to create optimal prompts.

Given this task description:
{task_description}

And these examples of desired inputs/outputs:
{examples}

Generate a prompt that will reliably produce these outputs for similar inputs.

Your prompt should include:
1. Clear task description
2. Specific format requirements
3. Relevant examples (if helpful)
4. Edge case handling
5. Output constraints

Generated prompt:
"""

META_PROMPT_IMPROVER = """
You are an expert prompt engineer analyzing prompt performance.

Original prompt:
{prompt}

This prompt produced these results:
- Success rate: {success_rate}%
- Common failures: {failures}
- Example bad output: {bad_example}

Analyze why this prompt is failing and provide an improved version.

Analysis:
1. Why is it failing?
2. What's ambiguous or missing?
3. How can we fix it?

Improved prompt:
"""

class PromptOptimizer:
    """
    Automatically optimize prompts based on feedback.
    """

    def __init__(self, llm_client):
        self.llm = llm_client
        self.prompt_history = []
        self.performance_history = []

    def generate_initial_prompt(self, task_description: str,
                                 examples: List[tuple]) -> str:
        """Generate an initial prompt from task description."""
        examples_text = "\n".join([
            f"Input: {inp}\nOutput: {out}"
            for inp, out in examples
        ])

        meta_prompt = META_PROMPT_GENERATOR.format(
            task_description=task_description,
            examples=examples_text
        )

        prompt = self.llm(meta_prompt)
        self.prompt_history.append(prompt)
        return prompt

    def evaluate_prompt(self, prompt: str, test_cases: List[tuple],
                        evaluator: Callable) -> Dict:
        """
        Evaluate prompt performance on test cases.

        Args:
            prompt: The prompt to evaluate
            test_cases: List of (input, expected_output) tuples
            evaluator: Function(output, expected) -> bool
        """
        results = []

        for input_text, expected in test_cases:
            full_prompt = prompt.format(input=input_text)
            output = self.llm(full_prompt)
            is_correct = evaluator(output, expected)
            results.append({
                "input": input_text,
                "expected": expected,
                "output": output,
                "correct": is_correct
            })

        success_rate = sum(r["correct"] for r in results) / len(results) * 100
        failures = [r for r in results if not r["correct"]]

        evaluation = {
            "success_rate": success_rate,
            "failures": failures,
            "total": len(results)
        }

        self.performance_history.append(evaluation)
        return evaluation

    def improve_prompt(self, prompt: str, evaluation: Dict) -> str:
        """Generate improved prompt based on evaluation."""
        failures_text = "\n".join([
            f"Input: {f['input']}\nExpected: {f['expected']}\nGot: {f['output']}"
            for f in evaluation["failures"][:3]
        ])

        bad_example = evaluation["failures"][0]["output"] if evaluation["failures"] else "N/A"

        meta_prompt = META_PROMPT_IMPROVER.format(
            prompt=prompt,
            success_rate=evaluation["success_rate"],
            failures=failures_text,
            bad_example=bad_example
        )

        improved = self.llm(meta_prompt)
        self.prompt_history.append(improved)
        return improved

    def optimize(self, task_description: str, examples: List[tuple],
                 test_cases: List[tuple], evaluator: Callable,
                 target_accuracy: float = 0.9, max_iterations: int = 5) -> str:
        """
        Iteratively optimize a prompt.

        Returns the best prompt found.
        """
        # Generate initial prompt
        prompt = self.generate_initial_prompt(task_description, examples)

        for i in range(max_iterations):
            # Evaluate
            evaluation = self.evaluate_prompt(prompt, test_cases, evaluator)
            print(f"Iteration {i+1}: {evaluation['success_rate']:.1f}% accuracy")

            # Check if target reached
            if evaluation["success_rate"] >= target_accuracy * 100:
                print("Target accuracy reached!")
                return prompt

            # Improve
            prompt = self.improve_prompt(prompt, evaluation)

        # Return best prompt from history
        best_idx = max(range(len(self.performance_history)),
                      key=lambda i: self.performance_history[i]["success_rate"])
        return self.prompt_history[best_idx]


# Usage example:
"""
optimizer = PromptOptimizer(my_llm)

# Define task
task = "Classify customer support messages into categories"
examples = [
    ("My order hasn't arrived", "Shipping"),
    ("How do I reset my password?", "Account"),
    ("I was charged twice", "Billing"),
]
test_cases = [
    ("Where is my package?", "Shipping"),
    ("Can't log in to my account", "Account"),
    ...
]

def evaluator(output, expected):
    return expected.lower() in output.lower()

best_prompt = optimizer.optimize(
    task_description=task,
    examples=examples,
    test_cases=test_cases,
    evaluator=evaluator,
    target_accuracy=0.9
)
"""
```

---

## Production Economics: Cost and Latency Analysis

Before choosing a technique, understand the production cost and latency implications:

```python
"""
Concrete cost and latency estimates for advanced prompting techniques.

Assumptions:
- GPT-4o: $5/1M input, $15/1M output tokens (early 2024 pricing)
- Average prompt: 500 input tokens, 200 output tokens per base call
- Base cost per call: (500 × $5 + 200 × $15) / 1M = $0.0055
- Average latency per call: ~2 seconds (GPT-4o, P50)
"""

def estimate_technique_cost(
    base_input_tokens: int = 500,
    base_output_tokens: int = 200,
    monthly_calls: int = 100_000,
    model_pricing: dict = None,
):
    """
    Estimate monthly cost and latency for each technique.
    """
    if model_pricing is None:
        model_pricing = {"input_per_1M": 5.0, "output_per_1M": 15.0}

    base_cost = (
        base_input_tokens * model_pricing["input_per_1M"]
        + base_output_tokens * model_pricing["output_per_1M"]
    ) / 1e6

    latency_per_call_s = 2.0  # P50 for GPT-4o

    techniques = {
        "Standard prompting": {
            "calls_per_query": 1,
            "input_multiplier": 1.0,
            "output_multiplier": 1.0,
            "sequential_calls": 1,
        },
        "Zero-shot CoT": {
            "calls_per_query": 1,
            "input_multiplier": 1.05,  # ~5% more input (added instruction)
            "output_multiplier": 2.5,  # 2-3x more output (reasoning steps)
            "sequential_calls": 1,
        },
        "Few-shot CoT (3 examples)": {
            "calls_per_query": 1,
            "input_multiplier": 2.0,  # Examples double input
            "output_multiplier": 2.5,
            "sequential_calls": 1,
        },
        "Self-Consistency (k=5)": {
            "calls_per_query": 5,
            "input_multiplier": 1.05,
            "output_multiplier": 2.5,
            "sequential_calls": 1,  # Parallel calls possible
        },
        "Tree-of-Thoughts (d=3, b=3)": {
            # Worst case: 3 + 9 + 27 = 39 LLM calls; with pruning ~12-15
            "calls_per_query": 15,  # ~15 with pruning
            "input_multiplier": 1.5,  # Growing context
            "output_multiplier": 1.5,
            "sequential_calls": 9,  # 3 depths × 3 evaluations
        },
        "ReAct (avg 4 steps)": {
            "calls_per_query": 4,
            "input_multiplier": 1.2,  # Growing context per step
            "output_multiplier": 0.5,  # Short thought + action
            "sequential_calls": 4,
        },
        "Prompt Chain (3 steps)": {
            "calls_per_query": 3,
            "input_multiplier": 1.0,
            "output_multiplier": 1.0,
            "sequential_calls": 3,
        },
    }

    print(f"{'Technique':<30} {'Cost/call':<12} {'Monthly':<12} {'P50 Latency':<12} {'vs Base'}")
    print("-" * 80)

    for name, config in techniques.items():
        cost = (
            config["calls_per_query"]
            * (
                base_input_tokens * config["input_multiplier"] * model_pricing["input_per_1M"]
                + base_output_tokens * config["output_multiplier"] * model_pricing["output_per_1M"]
            )
            / 1e6
        )
        monthly = cost * monthly_calls
        latency = config["sequential_calls"] * latency_per_call_s
        cost_ratio = cost / base_cost

        print(
            f"{name:<30} ${cost:<10.4f} ${monthly:<10.0f} {latency:<10.1f}s {cost_ratio:<.1f}x"
        )

    return techniques


# Run the analysis
print("=== Advanced Prompting: Cost & Latency at 100K calls/month (GPT-4o) ===\n")
estimate_technique_cost()

# Key takeaways:
# - Self-Consistency (k=5) costs ~5x but CAN be parallelized (same latency as CoT)
# - ToT is the most expensive: ~15x cost AND ~9x latency (sequential evaluation)
# - ReAct latency scales with step count; budget max_steps carefully
# - Prompt Chaining: 3x cost, 3x latency — each step is sequential
```

---

## Emerging Techniques (2023-2024)

The field moves fast. These newer approaches build on the techniques above:

```python
"""
Emerging techniques that extend CoT, ToT, and ReAct.
These are not yet as well-established but represent active research directions.
"""

EMERGING_TECHNIQUES = {
    "Reflexion (Shinn et al., 2023)": {
        "idea": "Agent that reflects on its own failures and retries with self-generated feedback",
        "mechanism": "Run task → evaluate result → generate verbal reflection on what went wrong "
                     "→ retry with reflection in context. Repeat until success or max attempts.",
        "vs_baseline": "Outperforms ReAct on AlfWorld and HotpotQA by learning from mistakes "
                       "within a single episode (no weight updates).",
        "cost": "2-5x per attempt (each retry adds reflection + re-execution)",
        "use_when": "Tasks with binary success/failure signals (code generation, QA with ground truth)",
    },
    "LATS (Language Agent Tree Search, Zhou et al., 2023)": {
        "idea": "Combines ToT's tree search with ReAct's tool use and environmental feedback",
        "mechanism": "Monte Carlo Tree Search (MCTS) over action sequences. Each node is a "
                     "(state, action) pair. Use LLM for both expansion and value estimation.",
        "vs_baseline": "Outperforms ReAct + Reflexion on complex decision-making tasks",
        "cost": "10-50x (MCTS requires many simulations)",
        "use_when": "Complex sequential decision-making where backtracking helps (code debugging, "
                    "web navigation)",
    },
    "Skeleton-of-Thought (Ning et al., 2023)": {
        "idea": "Generate answer skeleton first, then fill in each point in parallel",
        "mechanism": "Step 1: LLM outputs a skeleton (bullet points / section headers). "
                     "Step 2: Each skeleton point is expanded in parallel. "
                     "Result: faster generation with similar quality.",
        "vs_baseline": "Up to 2x speed-up on long-form generation with minimal quality loss",
        "cost": "Similar total tokens, but lower wall-clock time due to parallelism",
        "use_when": "Long-form content generation where latency matters (articles, reports, explanations)",
    },
}

# How to decide: technique evolution tree
TECHNIQUE_EVOLUTION = """
Standard Prompting
    └── CoT (add reasoning steps)
         ├── Self-Consistency (multiple CoT + vote)
         ├── ToT (structured tree search over thoughts)
         │    └── LATS (ToT + ReAct + MCTS)
         └── Reflexion (CoT + self-critique + retry)

ReAct (reasoning + tools)
    └── Reflexion (ReAct + self-reflection)
         └── LATS (ReAct + tree search)

Skeleton-of-Thought (orthogonal: parallel generation for speed)

DSPy/OPRO/APE (orthogonal: automatic prompt optimization)
"""
print(TECHNIQUE_EVOLUTION)
```

---

## Technique Selection Guide

### Decision Framework

```python
"""
Framework for selecting the right advanced prompting technique.
"""

TECHNIQUE_SELECTOR = {
    "task_characteristics": {
        "requires_reasoning": {
            True: ["CoT", "ToT"],
            False: ["Standard", "Few-shot"]
        },
        "requires_external_data": {
            True: ["ReAct"],
            False: ["CoT", "Self-Consistency"]
        },
        "has_multiple_valid_paths": {
            True: ["ToT", "Self-Consistency"],
            False: ["CoT"]
        },
        "is_multi_step_workflow": {
            True: ["Prompt Chaining"],
            False: ["Single prompt"]
        },
        "high_stakes_decision": {
            True: ["Self-Consistency"],
            False: ["Single-pass"]
        }
    }
}

def select_technique(task_description: str,
                     requires_reasoning: bool = False,
                     requires_external_data: bool = False,
                     has_multiple_valid_paths: bool = False,
                     is_multi_step: bool = False,
                     is_high_stakes: bool = False,
                     budget: str = "medium") -> dict:
    """
    Recommend prompting technique based on task characteristics.

    Args:
        task_description: Description of the task
        requires_reasoning: Does the task require multi-step reasoning?
        requires_external_data: Does it need external information?
        has_multiple_valid_paths: Are there multiple valid solution approaches?
        is_multi_step: Is it a multi-step workflow?
        is_high_stakes: Is accuracy critical?
        budget: "low", "medium", "high" token budget

    Returns:
        Dictionary with recommended technique and reasoning
    """
    recommendations = []

    # Check characteristics
    if requires_external_data:
        recommendations.append({
            "technique": "ReAct",
            "reason": "Task requires external data/tools",
            "cost": "High (variable)",
        })

    if is_multi_step:
        recommendations.append({
            "technique": "Prompt Chaining",
            "reason": "Task has multiple distinct steps",
            "cost": "Medium (N prompts)",
        })

    if requires_reasoning:
        if budget == "low":
            recommendations.append({
                "technique": "Zero-shot CoT",
                "reason": "Reasoning needed, low budget",
                "cost": "Low (1.5x)",
            })
        elif has_multiple_valid_paths and budget == "high":
            recommendations.append({
                "technique": "Tree-of-Thoughts",
                "reason": "Multiple solution paths worth exploring",
                "cost": "High (5-10x)",
            })
        else:
            recommendations.append({
                "technique": "Few-shot CoT",
                "reason": "Reasoning needed, moderate budget",
                "cost": "Medium (2x)",
            })

    if is_high_stakes and budget != "low":
        recommendations.append({
            "technique": "Self-Consistency",
            "reason": "High stakes, need confidence",
            "cost": "Medium-High (3-5x)",
        })

    # Default fallback
    if not recommendations:
        recommendations.append({
            "technique": "Standard Prompting",
            "reason": "Simple task, no special needs",
            "cost": "Low (1x)",
        })

    # Sort by cost if budget constrained
    if budget == "low":
        recommendations.sort(key=lambda x: {"Low": 0, "Medium": 1, "High": 2}.get(x["cost"].split()[0], 1))

    return {
        "task": task_description,
        "recommendations": recommendations,
        "primary": recommendations[0],
    }

# Example usage:
print("\n" + "="*60)
print("Technique Selection Examples")
print("="*60)

examples = [
    ("Calculate shipping cost with discounts", True, False, False, False, False, "low"),
    ("Research competitor pricing", True, True, False, False, True, "high"),
    ("Analyze legal contract", True, False, True, True, True, "high"),
    ("Translate text to French", False, False, False, False, False, "low"),
]

for args in examples:
    result = select_technique(*args)
    print(f"\nTask: {result['task']}")
    print(f"Primary recommendation: {result['primary']['technique']}")
    print(f"Reason: {result['primary']['reason']}")
```

---

## Measuring Technique Effectiveness: An Empirical Comparison Framework

The accuracy gains quoted throughout this blog come from published papers tested on specific benchmarks. Before adopting a technique in production, you should measure its impact on **your** task. Here is a lightweight comparison framework:

```python
"""
Framework for empirically comparing prompting techniques.
Run each technique on a shared evaluation set and compare accuracy vs cost.
"""

import time
from typing import List, Tuple, Callable, Dict

def compare_techniques(
    llm_client: Callable,
    test_cases: List[Tuple[str, str]],
    evaluator: Callable,
    techniques: Dict[str, Callable],
) -> Dict[str, Dict]:
    """
    Compare multiple prompting techniques on the same test set.

    Args:
        llm_client: Function for LLM calls
        test_cases: List of (input, expected_output) tuples (use 20+ for reliability)
        evaluator: Function(output, expected) -> bool
        techniques: Dict mapping technique name to prompt-building function

    Returns:
        Dict with accuracy, avg latency, and token cost estimate per technique
    """
    results = {}

    for name, build_prompt in techniques.items():
        correct = 0
        total_latency = 0.0
        total_tokens = 0

        for input_text, expected in test_cases:
            prompt = build_prompt(input_text)

            start = time.time()
            output = llm_client(prompt)
            elapsed = time.time() - start

            total_latency += elapsed
            # Rough token estimate: 1 token ~ 4 chars
            total_tokens += len(prompt) // 4 + len(output) // 4

            if evaluator(output, expected):
                correct += 1

        n = len(test_cases)
        results[name] = {
            "accuracy": correct / n,
            "avg_latency_s": total_latency / n,
            "est_total_tokens": total_tokens,
            "correct": correct,
            "total": n,
        }

    return results


def print_comparison(results: Dict[str, Dict]):
    """Pretty-print technique comparison results."""
    print(f"{'Technique':<25} {'Accuracy':>10} {'Avg Latency':>12} {'Est Tokens':>12}")
    print("-" * 62)
    for name, data in sorted(results.items(), key=lambda x: -x[1]["accuracy"]):
        print(
            f"{name:<25} {data['accuracy']:>9.1%} {data['avg_latency_s']:>10.2f}s "
            f"{data['est_total_tokens']:>10,}"
        )


# Example usage (pseudocode — requires a real LLM client):
"""
techniques = {
    "Standard":       lambda q: q,
    "Zero-shot CoT":  lambda q: f"{q}\n\nLet's think step by step.",
    "Few-shot CoT":   lambda q: create_cot_prompt(q, style="few_shot"),
    "Self-Consistency (k=5)": lambda q: ...,  # wrap SelfConsistency.solve
}

test_cases = [
    ("Roger has 5 balls. He buys 2 cans of 3. Total?", "11"),
    # ... 19 more test cases for statistical reliability
]

results = compare_techniques(my_llm, test_cases, evaluator=lambda o, e: e in o, techniques=techniques)
print_comparison(results)
"""
```

**Interpreting results:**
- 20+ test cases gives a rough signal; 100+ gives statistical confidence (use binomial confidence intervals).
- If Self-Consistency gains only 2% over CoT on your task but costs 5x more, stick with CoT.
- Always measure on **your domain** — published benchmark gains may not transfer.

---

## Interview Preparation

### Concept Questions

**Q1: Explain Chain-of-Thought prompting and when to use it.**

*Answer:* Chain-of-Thought (CoT) prompting asks the model to show its reasoning steps before giving a final answer. Instead of directly answering "What's 15% of 240?" the model reasons: "15% means 15/100 = 0.15. Multiply: 240 × 0.15 = 36." Use CoT for math problems, logical reasoning, multi-step tasks—anywhere showing work helps. Don't use it for simple factual questions or tasks like translation where reasoning doesn't help.

**Q2: What is self-consistency and how does it improve accuracy?**

*Answer:* Self-consistency samples multiple reasoning paths (using temperature > 0) and takes the majority vote. The insight is that correct answers are more likely to be reached via diverse reasoning paths. If 5 runs give answers [42, 42, 37, 42, 45], the majority vote (42, with 60% agreement) is more reliable than a single run. It costs 3-5x more tokens but can improve accuracy by 5-10% on complex tasks.

**Q3: Describe the ReAct framework.**

*Answer:* ReAct (Reasoning + Acting) interleaves thinking with tool use. The model alternates: Thought → Action → Observation → Thought → ... until it has enough information to answer. For "What's the population of Japan's capital?", ReAct would: think "need capital", search "capital of Japan", observe "Tokyo", think "need population", search "Tokyo population", observe "14 million", think "have answer", finish "14 million". It prevents hallucination by grounding reasoning in real lookups.

**Q4: When would you choose Tree-of-Thoughts over Chain-of-Thought?**

*Answer:* Use ToT when: (1) Multiple valid approaches exist and exploring them helps, (2) The problem benefits from backtracking (like puzzles), (3) Creative solutions are valuable, (4) Budget allows 5-10x cost. ToT explicitly generates multiple thought branches and evaluates them, pruning unpromising paths. Use CoT when there's a clear solution path and ToT's overhead isn't justified. ToT for "design a product" (creative), CoT for "calculate tax" (straightforward math).

### Coding Question

**Q5: Implement a basic Chain-of-Thought prompt generator.**

```python
def create_cot_prompt(task: str, examples: list = None, style: str = "detailed") -> str:
    """
    Create a Chain-of-Thought prompt.

    Args:
        task: The problem or task
        examples: Optional list of (question, step_by_step_answer) tuples
        style: "simple" (just "Let's think step by step") or "detailed" (structured)
    """
    prompt = ""

    if examples:
        for q, a in examples:
            prompt += f"Question: {q}\n\n"
            prompt += f"Answer: {a}\n\n"
            prompt += "---\n\n"

    prompt += f"Question: {task}\n\n"

    if style == "simple":
        prompt += "Answer: Let's think step by step.\n"
    else:
        prompt += """Answer: Let me solve this step by step.

Step 1: [Identify what we know]
Step 2: [Identify what we need to find]
Step 3: [Apply relevant method/formula]
Step 4: [Calculate/reason]
Step 5: [Verify the answer makes sense]

Final Answer: """

    return prompt

# Test
problem = "A train travels 120 miles in 2 hours, then 150 miles in 3 hours. What's the average speed?"
print(create_cot_prompt(problem, style="detailed"))
```

**Q6: When does Self-Consistency fail, and how do you handle free-text voting?**

*Answer:* Self-Consistency fails when the model has a *systematic* bias — all N samples converge on the same wrong answer. Increasing temperature helps but doesn't eliminate systematic errors. For free-text outputs (summaries, explanations) where exact string matching fails, use semantic clustering: generate N responses, use an LLM to group them by meaning, and pick the representative from the largest cluster. This costs one extra LLM call but handles text robustly.

**Q7: What are the failure modes of ReAct agents in production?**

*Answer:* Four main failure modes: (1) Infinite loops — agent repeats the same action; mitigate by tracking action history and forcing alternatives. (2) Tool hallucination — agent invents nonexistent tools; mitigate with structured output / function calling instead of string parsing. (3) Premature finish — agent answers before gathering enough info; add a verification step. (4) Context exhaustion — long traces exceed the window; summarize earlier steps periodically. Production ReAct needs all four mitigations plus cost budgeting (max_steps × cost_per_call).

**Q8: Compare DSPy to manual prompt engineering. When should you use each?**

*Answer:* Manual prompt engineering works best for: simple tasks, rapid prototyping, < 5 prompt templates to maintain. DSPy is better when: you have clear evaluation metrics, maintain many prompts, need systematic optimization, or want to compile prompts from declarative specifications. DSPy treats prompts as compiled artifacts — you define input/output signatures and optimization metrics, and DSPy finds the best prompt (or decides to fine-tune) automatically. Use manual prompting to start, then graduate to DSPy as complexity grows.

### System Design Question

**Q9: Design a multi-technique reasoning pipeline for an AI-powered financial research assistant.**

*Answer:*

```
Requirements: Users ask complex financial questions (e.g., "Should I invest in NVIDIA
given current AI trends and their Q3 earnings?"). System must use real-time data,
reason over multiple factors, and produce a structured report.

Architecture:

1. Query Router (Standard prompting, T=0)
   - Classify query type: factual / analytical / opinion
   - Determine which tools and techniques are needed
   - Estimated latency: 1s

2. Data Gathering (ReAct, max_steps=8)
   - Tools: stock_price(), earnings_report(), news_search(), sec_filing()
   - Agent gathers all relevant data
   - Validation: check that ≥3 data sources were consulted
   - Estimated latency: 8-16s (4-8 sequential tool calls)

3. Multi-Factor Analysis (Parallel Chain)
   - Branch A: Financial analysis (CoT on earnings data)
   - Branch B: Market sentiment (few-shot classification on news)
   - Branch C: Competitive landscape (CoT on industry data)
   - All three run in parallel
   - Estimated latency: 2-4s (parallel)

4. Synthesis (Self-Consistency, k=3)
   - Combine all analyses into a recommendation
   - 3 independent synthesis passes → majority vote on recommendation direction
   - Estimated latency: 2-4s (parallelizable)

5. Report Generation (Structured CoT)
   - Generate structured report with: summary, data sources, analysis,
     risks, recommendation, confidence level
   - Validation: check all required sections present
   - Estimated latency: 3-5s

Cost estimate (GPT-4o, per query):
- Router: $0.005
- ReAct (8 calls): $0.044
- Parallel analysis (3 calls): $0.017
- Self-Consistency (3 calls): $0.017
- Report: $0.010
- Total: ~$0.09/query → $9,000/month at 100K queries

Latency budget: 15-25s total (acceptable for research assistant)

Key design decisions:
- ReAct for data gathering (needs real-time tools)
- Parallel chain for analysis (independent factors)
- Self-Consistency for synthesis (high-stakes recommendation)
- Validation gates between stages (prevent garbage propagation)
- Cost budget: reject queries that would exceed $0.50 per query
```

---

## Exercises

### Exercise 1: CoT Implementation
Implement Chain-of-Thought prompting for word problems. Test on:
- Multi-step arithmetic
- Rate/distance/time problems
- Percentage calculations

Compare accuracy with and without CoT.

### Exercise 2: Self-Consistency System
Build a self-consistency wrapper that:
- Takes any prompt
- Runs it N times with temperature
- Parses answers
- Returns majority vote with confidence

### Exercise 3: ReAct Agent
Create a ReAct agent with these tools:
- `search(query)` - web search
- `calculate(expr)` - math calculator
- `lookup(entity)` - knowledge base lookup

Test on questions requiring multiple tool uses.

### Exercise 4: Prompt Chain
Design a prompt chain for document analysis:
1. Extract key entities
2. Identify relationships
3. Generate summary
4. Create FAQ

Implement with proper input/output mapping.

### Exercise 5: Technique Comparison
For a complex task of your choice:
- Implement standard prompting, CoT, and self-consistency
- Measure accuracy on 20+ test cases
- Calculate token costs
- Create cost-accuracy comparison

---

## Section Checkpoints

### Checkpoint 1 — After "Chain-of-Thought" (including failure modes)
1. Explain WHY CoT works in terms of autoregressive model computation limits.
2. When does CoT hurt performance? Give two examples.
3. What is "sycophantic reasoning" and how do you mitigate it in CoT prompts?
4. Compare zero-shot CoT vs few-shot CoT: when would you choose each?

### Checkpoint 2 — After "Self-Consistency" (including free-text voting)
1. Why does majority voting improve accuracy over single-pass CoT?
2. When does Self-Consistency converge on a wrong answer? What's the mitigation?
3. How do you apply Self-Consistency to free-text (non-numeric) outputs?
4. Estimate the cost of Self-Consistency (k=5) vs standard prompting at 100K calls/month.

### Checkpoint 3 — After "Tree-of-Thoughts" and "ReAct"
1. When should you use ToT instead of CoT? Give a concrete example.
2. What is the latency implication of ToT with depth=3 and branch_factor=3?
3. Name three ReAct failure modes and their mitigations.
4. How does ReAct prevent hallucination compared to standard prompting?

### Checkpoint 4 — After "Prompt Chaining" and "Error Propagation"
1. What is error propagation in chains and why is it dangerous?
2. How do validation gates prevent garbage propagation?
3. When can chain steps run in parallel vs when must they be sequential?
4. Design a 3-step chain for a task of your choice with validation gates.

### Checkpoint 5 — After "Meta-Prompting" and "Cost/Latency Analysis"
1. How does DSPy differ from the manual PromptOptimizer shown in this blog?
2. What is Reflexion and when should you use it instead of Self-Consistency?
3. Which technique has the highest latency? Why?
4. At what monthly volume does ToT's cost become prohibitive for a startup?

---

## Job Role Mapping

| Section | ML/AI Engineer | Data Scientist | AI Architect | Engineering Manager |
|---------|---------------|----------------|--------------|---------------------|
| CoT & Working Memory | Must know: computational argument for WHY CoT works, few-shot vs zero-shot CoT, failure modes | Must know: when CoT improves data analysis tasks | Must know: CoT as a building block for reasoning pipelines | Must know: CoT cost/quality tradeoff |
| Self-Consistency | Must know: implementation, free-text voting, failure modes, cost multiplier | Must know: when SC is worth the cost for analytical accuracy | Must know: parallelization strategy for SC (latency = 1 call despite k calls) | Must know: cost implications (5x), when to approve SC spend |
| ToT & ReAct | Must know: ToT BFS/DFS, ReAct loop, stuck-loop detection, tool validation | Must know: when ReAct improves research workflows | Must know: ReAct agent design, context window management, LATS for complex tasks | Must know: latency budgets, ReAct failure modes, monitoring |
| Prompt Chaining | Must know: chain design, error propagation, validation gates, parallel vs sequential | Must know: chaining for multi-step data processing | Must know: DAG-based execution, retry policies, validation architecture | Must know: pipeline reliability, error budgets |
| Meta-Prompting & Optimization | Must know: DSPy, OPRO, APE — when to use systematic optimization vs manual | Must know: evaluation-driven prompt improvement | Must know: prompt CI/CD, DSPy integration, optimization pipeline design | Must know: when to invest in automatic optimization vs manual tuning |
| Cost & Latency | Must know: cost estimation per technique, latency analysis, budget constraints | Must know: cost modeling for technique selection | Must know: cost budgeting per query, latency SLAs, technique selection governance | Must know: monthly cost projections, ROI of advanced techniques |

---

## Summary

### Key Takeaways

1. **CoT unlocks reasoning:** "Let's think step by step" can substantially boost accuracy on reasoning-heavy tasks (gains vary by model and task; always measure)
2. **Self-consistency for confidence:** Multiple samples + majority vote reduces random errors
3. **ToT for exploration:** Explicitly explore multiple paths for creative/complex problems
4. **ReAct for grounding:** Combine reasoning with tools to avoid hallucination
5. **Chaining for complexity:** Break complex tasks into manageable subtasks
6. **Meta-prompting:** Use LLMs to write and improve prompts

### What's Next

In Blog 14, we'll explore Working with AI APIs:
- OpenAI, Anthropic, Google API deep dive
- Authentication and rate limiting
- Streaming responses
- Error handling and retries
- Cost optimization strategies

---

## Self-Assessment Rubric

Rate yourself honestly after completing this blog:

| Criteria | Excellent (9-10) | Good (7-8) | Needs Work (5-6) |
|----------|-----------------|------------|-------------------|
| **CoT & Self-Consistency** | Can implement both, explain tradeoffs, and choose between them | Understands concepts and can write CoT prompts | Confuses techniques or cannot implement |
| **ToT & ReAct** | Can build a ToT explorer or ReAct agent with tools | Understands the loop structure and when to use | Cannot distinguish from basic CoT |
| **Prompt Chaining** | Can design multi-step chains with proper data flow | Understands sequential decomposition | Cannot map outputs to inputs between steps |
| **Technique Selection** | Can recommend and justify technique for novel problems | Can follow the selection framework | Applies techniques randomly |
| **Cost-Accuracy Tradeoffs** | Can estimate token cost multipliers and justify spend | Knows techniques vary in cost | Ignores cost considerations |

### What This Blog Does Well
- **Five techniques with full implementation + failure modes.** CoT, Self-Consistency, ToT, ReAct, and Prompt Chaining each have class-based implementations AND failure mode analysis explaining when/how they break.
- **Deeper CoT mechanism.** Explains WHY CoT works (intermediate tokens as working memory for autoregressive computation, Feng et al. 2023), not just that it works.
- **Production economics.** Concrete cost and latency analysis with dollar amounts at 100K calls/month for each technique. Readers can directly compare cost-accuracy tradeoffs.
- **Error propagation and validation gates.** ValidatedPromptChain shows how to prevent garbage propagation in chains — a critical production pattern missing from most tutorials.
- **Emerging techniques.** Reflexion, LATS, Skeleton-of-Thought, plus a technique evolution tree showing how they relate to core techniques.
- **DSPy, OPRO, APE context.** Meta-prompting section connects to the established research landscape rather than reinventing the wheel.
- **Free-text Self-Consistency voting.** Semantic clustering approach for non-numeric answers.
- **9 interview questions** including a full system design question (multi-technique financial research pipeline with per-component cost estimates).
- **5 checkpoints + job role mapping** (6 sections × 4 roles).

### Where This Blog Falls Short
- **All code requires an LLM client.** No code is runnable without API access. Readers must integrate their own client.
- **No prompt injection discussion.** ReAct agents that call tools with user-controlled input are vulnerable to injection. Deferred to Blog 14 but a forward-reference would help.
- **ToT implementation is simplified.** Synchronous BFS; production needs async execution and cost cap.
- **No monitoring patterns.** How to track technique performance over time (accuracy drift, cost per query, latency percentiles) is not covered.

### Architect Sanity Checks

### Check 1: Production Deployment Readiness
**Question**: Would you trust this person to select and implement the right prompting technique for each production use case?
**Answer: YES.** The blog covers technique selection with cost/latency analysis, failure modes for every technique, validation gates in chains, ReAct stuck-loop mitigation, free-text voting for Self-Consistency, and a concrete cost estimation function. The system design answer (Q9) demonstrates production thinking with per-component cost budgets and latency SLAs. Gap: monitoring and prompt injection are deferred.

### Check 2: Deep Problem Understanding
**Question**: Can they diagnose when techniques fail and select appropriate alternatives?
**Answer: YES.** Every technique has explicit failure mode analysis: CoT (sycophantic reasoning, verbose nonsense, chain collapse), Self-Consistency (systematic bias convergence, diverse-but-wrong), ReAct (infinite loops, tool hallucination, premature finish, context exhaustion). The compare_techniques framework enables empirical measurement before production deployment.

### Check 3: Interview and Career Readiness
**Question**: Can they explain advanced prompting mechanics, implement complex prompting systems, and make principled technique selections?
**Answer: YES.** 9 interview questions cover computational argument for CoT, Self-Consistency failure modes, ReAct production hardening, DSPy vs manual prompting, and a full system design (multi-technique financial research pipeline). The technique evolution tree and emerging techniques section prepares for "what's next?" follow-ups.
