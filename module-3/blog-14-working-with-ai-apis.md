# Blog 14: Working with AI APIs — From First Call to Production

**Series:** Prompt Your Career: The Complete Generative AI Masterclass
**Prerequisites:** Blog 13 (Advanced Prompting Techniques)
**Time to Complete:** 3.5-4 hours
**Difficulty:** Intermediate

---

> **How to read this blog:** Start with the Manager's Summary if you want the business context. Then work through the API Fundamentals and each provider section (OpenAI, Anthropic, Google) sequentially — they build on shared concepts but each has important differences. If you're already comfortable with one provider, skip to the others. The Error Handling, Rate Limiting, and Cost Optimization sections are essential for anyone building production systems — don't skip them even if you've made API calls before. The Production-Ready Wrapper ties everything together.
>
> **Dependencies:** `pip install openai anthropic google-generativeai tiktoken`
>
> **What you need before starting:** An API key for at least one provider (OpenAI, Anthropic, or Google). You can follow the code without keys, but hands-on practice requires at least one active account.

---

## What You'll Walk Away With

After completing this blog, you will be able to:

1. **Set up and authenticate** with OpenAI, Anthropic, and Google APIs
2. **Make API calls** for chat completions, embeddings, and more
3. **Handle streaming responses** for real-time user experiences
4. **Implement robust error handling** with retries and fallbacks
5. **Manage rate limits** without hitting API restrictions
6. **Optimize costs** through caching, batching, and model selection
7. **Build production-ready wrappers** for AI API integration

---

## What This Blog Does NOT Cover

Before we begin, let's set clear expectations on scope:

- **Prompt engineering techniques** — crafting effective prompts is covered in Blog 13. This blog assumes you know what to say and focuses on how to say it reliably via APIs.
- **LangChain and orchestration frameworks** — Blog 19 covers LangChain in depth. Here we work directly with provider SDKs.
- **Embeddings and vector databases in depth** — Blog 16 covers these. We show the embedding API call but not storage/retrieval pipelines.
- **Function calling and AI agents** — Blog 18 covers these. We mention tool definitions in the API anatomy but don't build agents here.
- **Fine-tuning** — Blog 23 covers model customization. This blog uses pre-trained models as-is.
- **Deployment infrastructure** — Blog 24 covers containerization, scaling, and CI/CD. This blog builds the wrapper code but doesn't deploy it.

---

## Manager's Summary

**Why API Mastery Matters:**

AI APIs are the foundation of every production GenAI application. Poor API implementation leads to outages, runaway costs, and frustrated users. Good implementation is invisible — it just works.

**Cost Reality Check:**

> **Pricing changes frequently.** The table below reflects approximate pricing as of early 2025. Always verify current pricing at the provider's official pricing page before making production decisions: [OpenAI Pricing](https://openai.com/pricing), [Anthropic Pricing](https://www.anthropic.com/pricing), [Google AI Pricing](https://ai.google.dev/pricing).

| Provider | Model | Approximate Cost per 1M tokens | Monthly Estimate @ 10M tokens |
|----------|-------|-------------------------------|-------------------------------|
| OpenAI | GPT-4o | ~$2.50 in / ~$10 out | ~$60 (blended) |
| OpenAI | GPT-4o-mini | ~$0.15 in / ~$0.60 out | ~$4 |
| Anthropic | Claude 3.5 Sonnet | ~$3 in / ~$15 out | ~$90 |
| Anthropic | Claude 3.5 Haiku | ~$0.25 in / ~$1.25 out | ~$8 |
| Google | Gemini 1.5 Pro | ~$3.50 in / ~$10.50 out | ~$70 |

*Prices are approximate and subject to change. Check provider pricing pages for current rates.*

**Build vs Buy Considerations:**

| Aspect | Direct API | Middleware (LangChain) | Full Platform |
|--------|------------|------------------------|---------------|
| **Control** | Maximum | High | Limited |
| **Setup Time** | Hours | Days | Minutes |
| **Customization** | Full | High | Limited |
| **Cost** | API only | API + compute | Premium |
| **Best For** | Simple apps | Complex apps | POC/MVPs |

**Risk Mitigation:**
- Always have a fallback model
- Implement request caching
- Monitor costs in real-time
- Set hard spending limits

---

## API Fundamentals

### Understanding the API Landscape

```
                    AI API Ecosystem

+------------------------------------------------------------------+
|                      YOUR APPLICATION                              |
+------------------------------------------------------------------+
|                                                                    |
|   +------------------------------------------------------------+  |
|   |                   API WRAPPER LAYER                         |  |
|   |  (Authentication, Retries, Caching, Rate Limiting)         |  |
|   +------------------------------------------------------------+  |
|                              |                                     |
|         +--------------------+--------------------+               |
|         |                    |                    |               |
|         v                    v                    v               |
|   +----------+        +----------+        +----------+           |
|   |  OpenAI  |        |Anthropic |        |  Google  |           |
|   |   API    |        |   API    |        |   API    |           |
|   +----------+        +----------+        +----------+           |
|                                                                    |
+------------------------------------------------------------------+
                              |
                              v
                    +-----------------+
                    |   LLM Servers   |
                    |   (GPUs, TPUs)  |
                    +-----------------+
```

### API Components

```python
"""
Understanding what makes up an AI API request/response.
"""

# A typical chat completion request has these components:
REQUEST_ANATOMY = {
    "authentication": {
        "type": "API Key in header",
        "format": "Authorization: Bearer sk-xxx...",
        "security": "Never commit to git, use environment variables",
    },

    "endpoint": {
        "example": "https://api.openai.com/v1/chat/completions",
        "method": "POST",
        "content_type": "application/json",
    },

    "request_body": {
        "model": "Which model to use (gpt-4o, claude-3-5-sonnet, etc.)",
        "messages": "The conversation history",
        "temperature": "Randomness (0.0 to 2.0)",
        "max_tokens": "Maximum response length",
        "stream": "Whether to stream the response",
        "tools": "Function calling / tool definitions",
    },

    "response": {
        "id": "Unique response identifier",
        "choices": "Array of generated completions",
        "usage": "Token counts (prompt, completion, total)",
        "created": "Timestamp",
    },
}
```

---

## OpenAI API Deep Dive

### Setup and Authentication

```python
"""
Complete OpenAI API setup and basic usage.
"""

# Installation: pip install openai

import os
from openai import OpenAI, AuthenticationError, APIConnectionError

# Method 1: Environment variable (recommended)
# Set: export OPENAI_API_KEY="sk-..."
client = OpenAI()  # Automatically reads OPENAI_API_KEY

# Method 2: Explicit (useful for multiple keys)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Method 3: Configuration file (~/.openai/config.json)
# {
#   "api_key": "sk-...",
#   "organization": "org-..."
# }

# Verify connection
def verify_openai_connection():
    """Test API connectivity."""
    try:
        response = client.models.list()
        print(f"Connected to OpenAI. Available models: {len(response.data)}")
        return True
    except AuthenticationError as e:
        print(f"Authentication failed — check your API key: {e}")
        return False
    except APIConnectionError as e:
        print(f"Connection failed — check your network: {e}")
        return False

verify_openai_connection()
```

### Chat Completions

```python
"""
OpenAI Chat Completions API - the core of GPT interactions.
"""

from openai import OpenAI, APIError, RateLimitError

client = OpenAI()

def basic_chat_completion(messages, model="gpt-4o"):
    """
    Basic chat completion call.

    Args:
        messages: List of message dicts with 'role' and 'content'
        model: Model to use
    """
    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )

    return response.choices[0].message.content


def advanced_chat_completion(messages, **kwargs):
    """
    Chat completion with all major parameters.
    """
    # Default parameters
    params = {
        "model": "gpt-4o",
        "messages": messages,
        "temperature": 0.7,      # 0.0 (deterministic) to 2.0 (random)
        "max_tokens": 1000,      # Max response length
        "top_p": 1.0,           # Nucleus sampling (usually leave at 1.0)
        "frequency_penalty": 0,  # Reduce repetition (-2.0 to 2.0)
        "presence_penalty": 0,   # Encourage new topics (-2.0 to 2.0)
        "stop": None,           # Stop sequences
        "n": 1,                 # Number of completions to generate
        "seed": None,           # For reproducibility (beta)
    }

    # Override with provided kwargs
    params.update(kwargs)

    response = client.chat.completions.create(**params)

    return {
        "content": response.choices[0].message.content,
        "finish_reason": response.choices[0].finish_reason,
        "usage": {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        },
        "model": response.model,
        "id": response.id,
    }


# Example usage
messages = [
    {"role": "system", "content": "You are a helpful coding assistant."},
    {"role": "user", "content": "Explain Python decorators in 2 sentences."},
]

result = advanced_chat_completion(
    messages,
    temperature=0.3,  # Lower for more focused response
    max_tokens=150,   # Limit response length
)

print(f"Response: {result['content']}")
print(f"Tokens used: {result['usage']['total_tokens']}")
```

### Message Roles and Conversation History

```python
"""
Understanding and managing conversation history.
"""

# Message roles in OpenAI API:
MESSAGE_ROLES = {
    "system": {
        "purpose": "Set assistant behavior and personality",
        "when": "First message, sets context for entire conversation",
        "example": "You are a helpful assistant that responds in JSON.",
    },
    "user": {
        "purpose": "Human messages / inputs",
        "when": "Every user turn in conversation",
        "example": "What's the weather like?",
    },
    "assistant": {
        "purpose": "AI responses (for context in multi-turn)",
        "when": "Include previous AI responses for continuity",
        "example": "The weather today is sunny and 72F.",
    },
    "tool": {
        "purpose": "Results from function/tool calls",
        "when": "After executing a function the AI requested",
        "example": '{"temperature": 72, "condition": "sunny"}',
    },
}


class ConversationManager:
    """
    Manage conversation history with token limits.
    """

    def __init__(self, system_prompt, max_history_tokens=3000):
        self.system_prompt = system_prompt
        self.max_history_tokens = max_history_tokens
        self.messages = [{"role": "system", "content": system_prompt}]

    def add_user_message(self, content):
        """Add a user message."""
        self.messages.append({"role": "user", "content": content})
        self._trim_history()

    def add_assistant_message(self, content):
        """Add an assistant message."""
        self.messages.append({"role": "assistant", "content": content})
        self._trim_history()

    def _trim_history(self):
        """Remove old messages if exceeding token limit."""
        # Estimate tokens (rough: 4 chars ~ 1 token)
        while self._estimate_tokens() > self.max_history_tokens:
            # Keep system message, remove oldest user/assistant pair
            if len(self.messages) > 3:
                self.messages.pop(1)  # Remove oldest non-system message
            else:
                break

    def _estimate_tokens(self):
        """Rough token estimation."""
        total_chars = sum(len(m["content"]) for m in self.messages)
        return total_chars // 4

    def get_messages(self):
        """Get current message list for API call."""
        return self.messages.copy()

    def clear(self):
        """Clear conversation, keeping system prompt."""
        self.messages = [{"role": "system", "content": self.system_prompt}]


# Usage
conversation = ConversationManager(
    system_prompt="You are a helpful Python tutor.",
    max_history_tokens=2000
)

conversation.add_user_message("What are list comprehensions?")
# ... get response and add it
conversation.add_assistant_message("List comprehensions are...")

conversation.add_user_message("Show me an example")
# Messages now include full context
```

### Streaming Responses

```python
"""
Streaming for real-time response display.
"""

from openai import OpenAI, APIError, APIConnectionError

client = OpenAI()

def stream_chat_completion(messages, model="gpt-4o"):
    """
    Stream response token by token.
    Essential for good UX in chat applications.
    """
    try:
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
        )
    except APIConnectionError as e:
        print(f"Connection error during stream setup: {e}")
        return ""

    full_response = ""

    for chunk in stream:
        # Check if there's content in this chunk
        if chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            print(content, end="", flush=True)  # Print as received
            full_response += content

    print()  # Newline at end
    return full_response


async def async_stream_chat(messages, model="gpt-4o"):
    """
    Async streaming for concurrent applications.

    Note: This is an async generator. Use 'async for' to consume it.
    Each yielded value is a text chunk. The caller is responsible
    for assembling the full response if needed.
    """
    from openai import AsyncOpenAI

    async_client = AsyncOpenAI()

    stream = await async_client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
    )

    async for chunk in stream:
        if chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            yield content


# Web framework example (FastAPI)
"""
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    async def generate():
        async for chunk in async_stream_chat(request.messages):
            yield f"data: {chunk}\\n\\n"  # SSE format
        yield "data: [DONE]\\n\\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )
"""
```

### Streaming Failure Modes and Recovery

```python
"""
Streaming introduces failure modes that don't exist with regular requests.
The connection can drop mid-response, leaving you with a partial result.
"""

STREAMING_FAILURE_MODES = {
    "Connection drop mid-stream": {
        "cause": "Network interruption, server timeout, or proxy disconnect",
        "symptom": "Stream stops yielding chunks; no finish_reason received",
        "mitigation": "Track accumulated content; if no chunk for >30s, treat as failure. "
                      "Option A: retry entire request. Option B: retry with accumulated "
                      "content as context ('Continue from: ...')",
    },
    "Incomplete JSON in structured output": {
        "cause": "Stream terminates before closing braces",
        "symptom": "json.loads() fails on accumulated response",
        "mitigation": "Don't parse JSON until finish_reason='stop'. If stream fails, "
                      "retry with non-streaming fallback for structured outputs.",
    },
    "Rate limit during stream": {
        "cause": "Rate limit hit after stream starts (rare but possible with some proxies)",
        "symptom": "Stream yields an error chunk instead of content",
        "mitigation": "Check each chunk for error indicators before appending to content.",
    },
}


def resilient_stream(client, messages, model="gpt-4o", max_retries=2):
    """
    Streaming with failure detection and retry.

    Detects: connection drops, incomplete responses, timeouts.
    Recovery: retries the full request (partial continuation is unreliable).
    """
    for attempt in range(max_retries + 1):
        full_response = ""
        finish_reason = None
        last_chunk_time = time.time()

        try:
            stream = client.chat.completions.create(
                model=model, messages=messages, stream=True, timeout=60,
            )

            for chunk in stream:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    last_chunk_time = time.time()
                    yield content  # Yield to caller for real-time display

                if chunk.choices[0].finish_reason:
                    finish_reason = chunk.choices[0].finish_reason

            # Verify we got a complete response
            if finish_reason == "stop":
                return  # Success — full response delivered
            elif finish_reason == "length":
                print(f"Warning: response truncated by max_tokens")
                return  # Truncated but valid
            else:
                raise ConnectionError(f"Stream ended without finish_reason (got: {finish_reason})")

        except (ConnectionError, TimeoutError, Exception) as e:
            if attempt < max_retries:
                print(f"Stream failed (attempt {attempt+1}): {e}. Retrying...")
                import time; time.sleep(2 ** attempt)
            else:
                raise RuntimeError(
                    f"Streaming failed after {max_retries+1} attempts. "
                    f"Partial response ({len(full_response)} chars): {full_response[:100]}..."
                )
```

### OpenAI Embeddings

```python
"""
Generate embeddings for semantic search, clustering, etc.
"""

from openai import OpenAI, APIError

client = OpenAI()

def get_embedding(text, model="text-embedding-3-small"):
    """
    Get embedding vector for text.

    Models:
    - text-embedding-3-small: Cheaper, 1536 dimensions, good for most uses
    - text-embedding-3-large: More accurate, 3072 dimensions
    - text-embedding-ada-002: Legacy model
    """
    response = client.embeddings.create(
        input=text,
        model=model,
    )

    return response.data[0].embedding


def get_embeddings_batch(texts, model="text-embedding-3-small"):
    """
    Batch embedding for efficiency.
    Can process up to 2048 texts in one call.
    """
    response = client.embeddings.create(
        input=texts,
        model=model,
    )

    # Return embeddings in same order as input
    return [item.embedding for item in response.data]


def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors."""
    import numpy as np

    vec1 = np.array(vec1)
    vec2 = np.array(vec2)

    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


# Example: Semantic similarity
texts = [
    "The cat sat on the mat",
    "A feline rested on the rug",
    "Python is a programming language",
]

embeddings = get_embeddings_batch(texts)

print("Similarity scores:")
print(f"Text 0 vs 1 (similar): {cosine_similarity(embeddings[0], embeddings[1]):.3f}")
print(f"Text 0 vs 2 (different): {cosine_similarity(embeddings[0], embeddings[2]):.3f}")
```

---

## Anthropic (Claude) API

### Setup and Basic Usage

```python
"""
Anthropic Claude API setup and usage.
"""

# Installation: pip install anthropic

import os
from anthropic import Anthropic, AuthenticationError, APIConnectionError, RateLimitError

# Initialize client
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

def claude_chat(messages, system_prompt=None, model="claude-3-5-sonnet-20241022"):
    """
    Basic Claude chat completion.

    Note: Anthropic uses a different message format than OpenAI.
    System prompt is a separate parameter, not a message.
    """
    response = client.messages.create(
        model=model,
        max_tokens=1024,
        system=system_prompt,  # System prompt is separate!
        messages=messages,
    )

    return response.content[0].text


def claude_advanced(messages, **kwargs):
    """
    Claude with all major parameters.
    """
    params = {
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 1024,
        "messages": messages,
        "temperature": 0.7,    # 0.0 to 1.0
        "top_p": 1.0,
        "top_k": None,         # Alternative to top_p
        "stop_sequences": [],  # Stop generation triggers
        "system": None,        # System prompt
    }

    params.update(kwargs)

    # Remove None values
    params = {k: v for k, v in params.items() if v is not None}

    response = client.messages.create(**params)

    return {
        "content": response.content[0].text,
        "stop_reason": response.stop_reason,
        "usage": {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        },
        "model": response.model,
        "id": response.id,
    }


# Claude model options (approximate pricing as of early 2025 — verify at anthropic.com/pricing):
CLAUDE_MODELS = {
    "claude-3-5-sonnet-20241022": {
        "description": "Best balance of performance and cost",
        "context": "200K tokens",
        "cost": "~$3/$15 per 1M tokens (approximate)",
    },
    "claude-3-5-haiku-20241022": {
        "description": "Fastest, most affordable",
        "context": "200K tokens",
        "cost": "~$0.25/$1.25 per 1M tokens (approximate)",
    },
    "claude-3-opus-20240229": {
        "description": "Most capable, highest quality",
        "context": "200K tokens",
        "cost": "~$15/$75 per 1M tokens (approximate)",
    },
}

# Example usage
messages = [
    {"role": "user", "content": "Explain quantum computing in one paragraph."},
]

result = claude_advanced(
    messages,
    system="You are a physics professor explaining to undergraduates.",
    model="claude-3-5-sonnet-20241022",
    temperature=0.5,
)

print(f"Response: {result['content']}")
```

### Claude Streaming

```python
"""
Streaming responses with Claude.
"""

from anthropic import Anthropic, APIConnectionError

client = Anthropic()

def stream_claude(messages, system_prompt=None, model="claude-3-5-sonnet-20241022"):
    """Stream Claude responses."""
    try:
        with client.messages.stream(
            model=model,
            max_tokens=1024,
            system=system_prompt,
            messages=messages,
        ) as stream:
            full_response = ""
            for text in stream.text_stream:
                print(text, end="", flush=True)
                full_response += text
    except APIConnectionError as e:
        print(f"Connection error during streaming: {e}")
        return ""

    print()
    return full_response


# Async streaming
async def async_stream_claude(messages, system_prompt=None):
    """
    Async streaming for Claude.

    Note: This is an async generator. The caller is responsible for
    assembling the full response from yielded chunks.
    """
    from anthropic import AsyncAnthropic

    async_client = AsyncAnthropic()

    async with async_client.messages.stream(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        system=system_prompt,
        messages=messages,
    ) as stream:
        async for text in stream.text_stream:
            yield text
```

---

## Google Gemini API

### Setup and Usage

```python
"""
Google Gemini API setup and usage.
"""

# Installation: pip install google-generativeai

import os
import google.generativeai as genai

# Configure API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def gemini_chat(messages, model_name="gemini-1.5-pro"):
    """
    Basic Gemini chat.

    Note: Gemini uses a different conversation format.
    """
    model = genai.GenerativeModel(model_name)

    # Convert messages to Gemini format
    gemini_messages = []
    for msg in messages:
        role = "user" if msg["role"] == "user" else "model"
        gemini_messages.append({
            "role": role,
            "parts": [msg["content"]]
        })

    chat = model.start_chat(history=gemini_messages[:-1])
    response = chat.send_message(gemini_messages[-1]["parts"][0])

    return response.text


def gemini_generate(prompt, model_name="gemini-1.5-pro", **kwargs):
    """
    Direct generation (non-chat) with Gemini.
    """
    generation_config = genai.GenerationConfig(
        temperature=kwargs.get("temperature", 0.7),
        max_output_tokens=kwargs.get("max_tokens", 1024),
        top_p=kwargs.get("top_p", 1.0),
        top_k=kwargs.get("top_k", 40),
    )

    model = genai.GenerativeModel(
        model_name=model_name,
        generation_config=generation_config,
    )

    response = model.generate_content(prompt)

    return {
        "content": response.text,
        "finish_reason": response.candidates[0].finish_reason.name,
        "safety_ratings": [
            {"category": r.category.name, "probability": r.probability.name}
            for r in response.candidates[0].safety_ratings
        ],
    }


# Gemini models:
GEMINI_MODELS = {
    "gemini-1.5-pro": {
        "context": "1M tokens",
        "description": "Best for complex tasks",
    },
    "gemini-1.5-flash": {
        "context": "1M tokens",
        "description": "Faster, more efficient",
    },
}

# Example usage
prompt = "Write a haiku about artificial intelligence."
result = gemini_generate(prompt, temperature=0.8)
print(result["content"])
```

---

## Error Handling and Retries

### Robust Error Handling

```python
"""
Production-grade error handling for AI APIs.
"""

import time
import random
from typing import Callable, Any
from functools import wraps

# Import provider-specific exceptions for precise error handling
from openai import (
    APIError as OpenAIAPIError,
    RateLimitError as OpenAIRateLimitError,
    AuthenticationError as OpenAIAuthError,
    BadRequestError as OpenAIBadRequestError,
    APIConnectionError as OpenAIConnectionError,
    APITimeoutError as OpenAITimeoutError,
    InternalServerError as OpenAIServerError,
)
from anthropic import (
    APIError as AnthropicAPIError,
    RateLimitError as AnthropicRateLimitError,
    AuthenticationError as AnthropicAuthError,
    BadRequestError as AnthropicBadRequestError,
    APIConnectionError as AnthropicConnectionError,
    APITimeoutError as AnthropicTimeoutError,
    InternalServerError as AnthropicServerError,
)


# Custom error hierarchy for our wrapper
class APIWrapperError(Exception):
    """Base class for API wrapper errors."""
    pass

class RateLimitExceeded(APIWrapperError):
    """Rate limit exceeded."""
    pass

class AuthenticationFailed(APIWrapperError):
    """Invalid API key."""
    pass

class InvalidRequestError(APIWrapperError):
    """Invalid request parameters."""
    pass

class ServiceUnavailableError(APIWrapperError):
    """API service is down."""
    pass


def classify_error(error):
    """
    Classify error and determine if retryable.

    Returns: (error_class, retryable, suggested_wait_seconds)
    """
    # Rate limit errors — retryable
    if isinstance(error, (OpenAIRateLimitError, AnthropicRateLimitError)):
        return RateLimitExceeded, True, 60

    # Authentication errors — not retryable
    if isinstance(error, (OpenAIAuthError, AnthropicAuthError)):
        return AuthenticationFailed, False, 0

    # Bad request — not retryable (fix the request)
    if isinstance(error, (OpenAIBadRequestError, AnthropicBadRequestError)):
        return InvalidRequestError, False, 0

    # Server errors — retryable
    if isinstance(error, (OpenAIServerError, AnthropicServerError)):
        return ServiceUnavailableError, True, 30

    # Connection/timeout errors — retryable
    if isinstance(error, (
        OpenAIConnectionError, OpenAITimeoutError,
        AnthropicConnectionError, AnthropicTimeoutError
    )):
        return APIWrapperError, True, 5

    # Unknown API errors — try retry
    if isinstance(error, (OpenAIAPIError, AnthropicAPIError)):
        return APIWrapperError, True, 10

    # Completely unknown — not retryable (don't hide bugs)
    return APIWrapperError, False, 0


def retry_with_exponential_backoff(
    max_retries: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
):
    """
    Decorator for retrying with exponential backoff.

    WHY exponential backoff with jitter?
    Without it, when an API goes down and recovers, all waiting clients retry
    simultaneously — the "thundering herd" problem. This causes the API to
    immediately overload again, creating a retry storm cycle.

    Exponential delay spreads retries over time. Jitter randomizes the exact
    timing so clients don't synchronize. Together, they give the API time to
    recover gracefully.

    delay = min(base_delay * (exponential_base ** attempt), max_delay) * random(0.5, 1.5)

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Base for exponential calculation
        jitter: Whether to add random jitter (always True in production)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except (
                    OpenAIRateLimitError, OpenAIServerError,
                    OpenAIConnectionError, OpenAITimeoutError,
                    AnthropicRateLimitError, AnthropicServerError,
                    AnthropicConnectionError, AnthropicTimeoutError,
                ) as e:
                    # Retryable errors
                    error_class, retryable, suggested_wait = classify_error(e)
                    last_exception = e

                    if attempt == max_retries:
                        raise error_class(str(e)) from e

                    # Calculate delay
                    delay = min(
                        base_delay * (exponential_base ** attempt),
                        max_delay
                    )

                    # Use suggested wait if higher
                    delay = max(delay, suggested_wait)

                    # Add jitter
                    if jitter:
                        delay = delay * (0.5 + random.random())

                    print(f"Attempt {attempt + 1} failed: {e}")
                    print(f"Retrying in {delay:.2f} seconds...")
                    time.sleep(delay)

                except (
                    OpenAIAuthError, OpenAIBadRequestError,
                    AnthropicAuthError, AnthropicBadRequestError,
                ) as e:
                    # Non-retryable errors — raise immediately
                    error_class, _, _ = classify_error(e)
                    raise error_class(str(e)) from e

            raise last_exception

        return wrapper
    return decorator


# Apply to API calls
@retry_with_exponential_backoff(max_retries=3)
def reliable_chat_completion(client, messages, **kwargs):
    """Chat completion with automatic retry."""
    return client.chat.completions.create(
        messages=messages,
        **kwargs
    )


# More sophisticated: Circuit breaker pattern
class CircuitBreaker:
    """
    Circuit breaker to prevent cascading failures.

    States:
    - CLOSED: Normal operation
    - OPEN: Failing, reject all requests
    - HALF_OPEN: Testing if service recovered
    """

    def __init__(self, failure_threshold=5, recovery_timeout=60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failures = 0
        self.last_failure_time = 0
        self.state = "CLOSED"

    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
            else:
                raise ServiceUnavailableError("Circuit breaker is OPEN")

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except (
            OpenAIServerError, OpenAIConnectionError, OpenAITimeoutError,
            AnthropicServerError, AnthropicConnectionError, AnthropicTimeoutError,
            ServiceUnavailableError,
        ) as e:
            self._on_failure()
            raise

    def _on_success(self):
        """Handle successful call."""
        self.failures = 0
        self.state = "CLOSED"

    def _on_failure(self):
        """Handle failed call."""
        self.failures += 1
        self.last_failure_time = time.time()

        if self.failures >= self.failure_threshold:
            self.state = "OPEN"
            print(f"Circuit breaker OPEN after {self.failures} failures")


# Usage
circuit_breaker = CircuitBreaker(failure_threshold=3)

def safe_api_call(messages):
    return circuit_breaker.call(
        reliable_chat_completion,
        client,
        messages,
        model="gpt-4o"
    )
```

### Idempotency: The Hidden Retry Risk

```python
"""
Retrying API calls is safe for read-only operations (chat completions, embeddings).
But if your LLM call triggers side effects (function calling, tool use, sending emails),
retrying can cause DUPLICATE ACTIONS.

Example: LLM calls send_email() tool → request times out → you retry →
LLM calls send_email() again → customer gets two emails.
"""

IDEMPOTENCY_RULES = {
    "Safe to retry (idempotent)": [
        "Chat completions (text generation)",
        "Embeddings",
        "Classification / extraction",
        "Any read-only operation",
    ],
    "DANGEROUS to retry (non-idempotent)": [
        "Function calling that triggers side effects (send email, create order, delete record)",
        "Tool use with external APIs (Slack messages, database writes)",
        "Any operation that changes state",
    ],
    "Mitigation strategies": [
        "Use idempotency keys: pass a unique request ID; if the server already processed it, "
        "return the cached result instead of re-executing",
        "Separate generation from execution: let the LLM PLAN actions, but execute them in a "
        "separate step with deduplication (check if action already completed before executing)",
        "Log all tool calls with unique IDs before execution; on retry, check the log first",
    ],
}
```

---

## API Security Beyond Environment Variables

```python
"""
"Use env vars" is the minimum. Production API security requires more.
"""

API_SECURITY_PRACTICES = {
    "Key management": {
        "Rotation": "Rotate API keys every 90 days. Both OpenAI and Anthropic support "
                    "multiple active keys — create the new key, deploy it, then revoke the old one.",
        "Least privilege": "Use separate keys for dev/staging/prod. If your provider supports "
                          "scoped keys (e.g., read-only, specific models), use them.",
        "Never log keys": "Ensure your logging framework redacts API keys from request headers. "
                         "Grep your logs for 'sk-' or 'Bearer' patterns periodically.",
    },
    "Network security": {
        "Proxy pattern": "Route API calls through a proxy server that adds authentication. "
                        "Client apps never see the API key — they authenticate with the proxy.",
        "IP allowlisting": "Some providers allow restricting API access to specific IP ranges. "
                          "Use this for server-to-server calls.",
        "TLS verification": "Always verify TLS certificates. Never set verify=False in production.",
    },
    "Cost protection": {
        "Hard spending limits": "Set monthly limits in your provider dashboard (OpenAI and "
                               "Anthropic both support this). Set limits at 80% of budget as warning, "
                               "100% as hard stop.",
        "Per-user rate limits": "If your app exposes LLM calls to users, rate-limit per user "
                               "to prevent a single user from exhausting your budget.",
        "Anomaly detection": "Alert on sudden token usage spikes (>3x normal). Could indicate "
                            "prompt injection, abuse, or a bug generating infinite loops.",
    },
}
```

---

## Rate Limiting

### Understanding Rate Limits

```python
"""
Rate limiting strategies for AI APIs.

Note: Rate limits vary by account tier and change over time.
The values below are illustrative examples — check your provider
dashboard for your actual limits.
"""

import time

# Illustrative rate limits (actual limits vary by tier and change frequently):
RATE_LIMITS_EXAMPLE = {
    "openai": {
        "gpt-4o": {
            "requests_per_minute": "varies by tier (e.g., 500-10000)",
            "tokens_per_minute": "varies by tier (e.g., 30000-800000)",
        },
    },
    "anthropic": {
        "claude-3-5-sonnet": {
            "requests_per_minute": "varies by tier",
            "tokens_per_minute": "varies by tier",
        },
    },
}


class RateLimiter:
    """
    Token bucket rate limiter.
    """

    def __init__(self, requests_per_minute=60, tokens_per_minute=40000):
        self.requests_per_minute = requests_per_minute
        self.tokens_per_minute = tokens_per_minute

        # Token buckets
        self.request_tokens = requests_per_minute
        self.token_tokens = tokens_per_minute

        self.last_update = time.time()

    def _refill_buckets(self):
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed_minutes = (now - self.last_update) / 60

        # Refill
        self.request_tokens = min(
            self.requests_per_minute,
            self.request_tokens + (elapsed_minutes * self.requests_per_minute)
        )
        self.token_tokens = min(
            self.tokens_per_minute,
            self.token_tokens + (elapsed_minutes * self.tokens_per_minute)
        )

        self.last_update = now

    def acquire(self, estimated_tokens=1000):
        """
        Acquire permission to make a request.
        Blocks if rate limit would be exceeded.
        """
        self._refill_buckets()

        # Check if we have capacity
        while self.request_tokens < 1 or self.token_tokens < estimated_tokens:
            # Calculate wait time
            request_wait = (1 - self.request_tokens) / (self.requests_per_minute / 60)
            token_wait = (estimated_tokens - self.token_tokens) / (self.tokens_per_minute / 60)
            wait_time = max(request_wait, token_wait, 0.1)

            print(f"Rate limited. Waiting {wait_time:.2f}s...")
            time.sleep(wait_time)
            self._refill_buckets()

        # Consume tokens
        self.request_tokens -= 1
        self.token_tokens -= estimated_tokens

    def report_actual_usage(self, estimated_tokens, actual_tokens):
        """
        Adjust token bucket for actual vs estimated usage.

        If we overestimated, give back excess tokens to the bucket.
        If we underestimated, consume the difference.
        This improves accuracy of rate limiting over time.
        """
        difference = estimated_tokens - actual_tokens
        self.token_tokens = min(
            self.tokens_per_minute,
            self.token_tokens + difference
        )


# Usage with rate limiter
rate_limiter = RateLimiter(requests_per_minute=60, tokens_per_minute=40000)

def rate_limited_completion(messages, estimated_tokens=1000):
    """Make API call with rate limiting."""
    rate_limiter.acquire(estimated_tokens)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
    )

    # Report actual usage for better estimates
    actual_tokens = response.usage.total_tokens
    rate_limiter.report_actual_usage(estimated_tokens, actual_tokens)

    return response
```

### Async Rate Limiting

```python
"""
Async rate limiting for concurrent applications.
"""

import asyncio
import time
from asyncio import Semaphore

class AsyncRateLimiter:
    """
    Async-compatible rate limiter using a lock to prevent
    race conditions on shared state.
    """

    def __init__(self, max_concurrent=10, requests_per_second=5):
        self.semaphore = Semaphore(max_concurrent)
        self.requests_per_second = requests_per_second
        self.request_times = []
        self._lock = asyncio.Lock()  # Protect shared state

    async def acquire(self):
        """Acquire rate limit slot."""
        await self.semaphore.acquire()
        try:
            async with self._lock:
                # Remove old timestamps
                now = time.time()
                self.request_times = [
                    t for t in self.request_times
                    if now - t < 1.0
                ]

                # Wait if at limit
                while len(self.request_times) >= self.requests_per_second:
                    await asyncio.sleep(0.1)
                    now = time.time()
                    self.request_times = [
                        t for t in self.request_times
                        if now - t < 1.0
                    ]

                self.request_times.append(now)
        finally:
            self.semaphore.release()

    async def __aenter__(self):
        await self.acquire()
        return self

    async def __aexit__(self, *args):
        pass


# Async batch processing with rate limiting
async def process_batch(items, process_func, rate_limiter):
    """
    Process items concurrently with rate limiting.

    Note: Tasks are created after acquiring rate limit to avoid
    launching all tasks simultaneously before rate limiting kicks in.
    """
    results = []
    for item in items:
        async with rate_limiter:
            result = await process_func(item)
            results.append(result)
    return results


# Example usage:
"""
from openai import AsyncOpenAI

async def analyze_document(doc):
    async_client = AsyncOpenAI()
    try:
        response = await async_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": f"Summarize: {doc}"}]
        )
        return response.choices[0].message.content
    except (OpenAIRateLimitError, OpenAIServerError) as e:
        return f"Error processing document: {e}"

rate_limiter = AsyncRateLimiter(max_concurrent=10, requests_per_second=5)
documents = ["doc1", "doc2", "doc3", ...]

results = await process_batch(documents, analyze_document, rate_limiter)
"""
```

---

## Cost Optimization

### Token Counting and Cost Estimation

```python
"""
Accurate token counting and cost estimation.

Note on tokenizers:
- OpenAI models: Use tiktoken (official OpenAI tokenizer library).
- Anthropic models: Use Anthropic's own tokenizer via their SDK
  (client.count_tokens() or the anthropic tokenizer). Do NOT use
  tiktoken for Claude — it uses a different tokenizer.
- Google models: The Gemini SDK includes count_tokens() methods.

For cross-provider estimation, use each provider's native tools.
"""

import tiktoken


def count_tokens_openai(text, model="gpt-4o"):
    """
    Count tokens for OpenAI models using tiktoken.
    Only accurate for OpenAI models.
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    return len(encoding.encode(text))


def count_tokens_anthropic(text, client=None):
    """
    Count tokens for Anthropic models.

    The Anthropic SDK provides a count_tokens method.
    Falls back to rough estimation if client is not available.
    """
    if client is not None:
        try:
            return client.count_tokens(text)
        except AttributeError:
            pass

    # Rough estimation fallback (not accurate for production use)
    # Anthropic's tokenizer typically produces slightly different counts than tiktoken
    return len(text) // 4


def count_message_tokens(messages, model="gpt-4o"):
    """
    Count tokens in a list of messages (OpenAI format).
    Includes overhead for message formatting.
    """
    encoding = tiktoken.encoding_for_model(model)

    # Token overhead per message (varies by model)
    tokens_per_message = 3  # <|start|>role<|end|>
    tokens_per_name = 1

    total_tokens = 0

    for message in messages:
        total_tokens += tokens_per_message
        total_tokens += len(encoding.encode(message["content"]))
        total_tokens += len(encoding.encode(message["role"]))

        if "name" in message:
            total_tokens += len(encoding.encode(message["name"]))
            total_tokens += tokens_per_name

    total_tokens += 3  # <|start|>assistant<|end|>

    return total_tokens


def estimate_cost(input_tokens, output_tokens, model="gpt-4o"):
    """
    Estimate API cost.

    IMPORTANT: These prices are approximate as of early 2025 and change
    frequently. Always check the provider's pricing page for current rates.
    """
    # Approximate prices per 1M tokens (as of early 2025 -- verify before use)
    PRICING = {
        "gpt-4o": {"input": 2.50, "output": 10.0},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4-turbo": {"input": 10.0, "output": 30.0},
        "claude-3-5-sonnet": {"input": 3.0, "output": 15.0},
        "claude-3-5-haiku": {"input": 0.25, "output": 1.25},
        "claude-3-opus": {"input": 15.0, "output": 75.0},
    }

    prices = PRICING.get(model, PRICING["gpt-4o"])

    input_cost = (input_tokens / 1_000_000) * prices["input"]
    output_cost = (output_tokens / 1_000_000) * prices["output"]

    return {
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": input_cost + output_cost,
        "model": model,
    }


# Cost tracker
class CostTracker:
    """Track API costs over time."""

    def __init__(self):
        self.usage_log = []
        self.total_cost = 0

    def log_usage(self, input_tokens, output_tokens, model):
        """Log API usage."""
        cost = estimate_cost(input_tokens, output_tokens, model)

        self.usage_log.append({
            "timestamp": time.time(),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "model": model,
            "cost": cost["total_cost"],
        })

        self.total_cost += cost["total_cost"]

        return cost

    def get_summary(self, period="day"):
        """Get usage summary."""
        import time
        now = time.time()
        cutoffs = {
            "hour": 3600,
            "day": 86400,
            "week": 604800,
        }

        cutoff = now - cutoffs.get(period, cutoffs["day"])
        recent = [u for u in self.usage_log if u["timestamp"] > cutoff]

        return {
            "period": period,
            "total_requests": len(recent),
            "total_input_tokens": sum(u["input_tokens"] for u in recent),
            "total_output_tokens": sum(u["output_tokens"] for u in recent),
            "total_cost": sum(u["cost"] for u in recent),
            "by_model": self._group_by_model(recent),
        }

    def _group_by_model(self, usage_list):
        """Group usage by model."""
        by_model = {}
        for u in usage_list:
            model = u["model"]
            if model not in by_model:
                by_model[model] = {"requests": 0, "cost": 0}
            by_model[model]["requests"] += 1
            by_model[model]["cost"] += u["cost"]
        return by_model


# Usage
tracker = CostTracker()
```

### Cost Optimization Strategies

```python
"""
Strategies to reduce API costs.
"""

import hashlib
import json

class CostOptimizer:
    """Collection of cost optimization techniques."""

    def __init__(self, cache_enabled=True):
        self.cache = {} if cache_enabled else None

    # Strategy 1: Response caching with TTL
    def cached_completion(self, messages, model, cache_ttl=3600, **kwargs):
        """
        Cache identical requests with TTL (time-to-live).

        IMPORTANT: Only cache when temperature=0 (deterministic).
        Caching non-deterministic responses returns stale/wrong answers.

        Args:
            cache_ttl: Cache expiry in seconds (default 1 hour)
        """
        if self.cache is None:
            return self._make_request(messages, model, **kwargs)

        # Only cache deterministic responses
        temperature = kwargs.get("temperature", 0.7)
        if temperature > 0:
            return self._make_request(messages, model, **kwargs)

        # Create cache key from request
        cache_key = hashlib.md5(
            json.dumps({
                "messages": messages,
                "model": model,
                "temperature": temperature,
            }, sort_keys=True).encode()
        ).hexdigest()

        # Check cache with TTL
        if cache_key in self.cache:
            entry = self.cache[cache_key]
            if time.time() - entry["timestamp"] < cache_ttl:
                print("Cache hit!")
                return entry["response"]
            else:
                del self.cache[cache_key]  # Expired

        # Evict oldest if cache too large (simple LRU approximation)
        if len(self.cache) > 1000:
            oldest_key = min(self.cache, key=lambda k: self.cache[k]["timestamp"])
            del self.cache[oldest_key]

        response = self._make_request(messages, model, **kwargs)
        self.cache[cache_key] = {"response": response, "timestamp": time.time()}
        return response

    def _make_request(self, messages, model, **kwargs):
        """Make actual API request."""
        return client.chat.completions.create(
            model=model,
            messages=messages,
            **kwargs
        )

    # Strategy 2: Model tiering
    @staticmethod
    def select_model(task_complexity, accuracy_requirement):
        """
        Select cheapest model that meets requirements.
        """
        tiers = {
            ("low", "low"): "gpt-4o-mini",
            ("low", "medium"): "gpt-4o-mini",
            ("low", "high"): "gpt-4o",
            ("medium", "low"): "gpt-4o-mini",
            ("medium", "medium"): "gpt-4o",
            ("medium", "high"): "gpt-4o",
            ("high", "low"): "gpt-4o",
            ("high", "medium"): "gpt-4o",
            ("high", "high"): "gpt-4o",
        }

        return tiers.get((task_complexity, accuracy_requirement), "gpt-4o")

    # Strategy 3: Prompt compression
    @staticmethod
    def compress_prompt(prompt, max_tokens=1000):
        """
        Compress prompt to reduce token count.
        Simple approach: truncate with summarization request.
        """
        current_tokens = count_tokens_openai(prompt)

        if current_tokens <= max_tokens:
            return prompt

        # Request compression via LLM (meta!)
        compression_prompt = f"""
        Compress this text to under {max_tokens} tokens while preserving key information:

        {prompt}

        Compressed version:
        """

        # In practice, use a cheaper model for compression
        return compression_prompt  # Would call API here

    # Strategy 4: Batching
    @staticmethod
    def batch_similar_requests(requests, max_batch_size=10):
        """
        Batch similar requests into single calls where possible.
        """
        # Group by task type
        batches = {}
        for req in requests:
            task_type = req.get("task_type", "default")
            if task_type not in batches:
                batches[task_type] = []
            batches[task_type].append(req)

        # Process batches
        results = []
        for task_type, batch in batches.items():
            # Create batched prompt
            if len(batch) == 1:
                results.extend(batch)
            else:
                # Combine into single prompt — one API call instead of N
                combined_prompt = f"Process these {len(batch)} items. Return results as a numbered list:\n\n"
                for i, req in enumerate(batch, 1):
                    combined_prompt += f"{i}. {req['content']}\n"

                # Single API call for entire batch
                response = client.chat.completions.create(
                    model="gpt-4o-mini",  # Use cheaper model for batch processing
                    messages=[{"role": "user", "content": combined_prompt}],
                    temperature=0,
                )

                # Parse numbered responses back to individual results
                batch_text = response.choices[0].message.content
                for i, req in enumerate(batch):
                    req["result"] = batch_text  # In production, parse numbered items
                    results.append(req)

        return results


# Cost optimization config
OPTIMIZATION_CONFIG = {
    "cache_ttl": 3600,           # 1 hour cache
    "compression_threshold": 2000,# Compress prompts over 2000 tokens
    "batch_size": 10,            # Batch up to 10 similar requests
    "fallback_model": "gpt-4o-mini",  # Use cheaper model if main fails
}
```

---

## Structured Logging for API Calls

```python
"""
Production API wrappers need structured logging (JSON format) for debugging,
observability, and integration with log aggregation tools (Datadog, Splunk, ELK).

Plain text logs like 'API call failed' are useless at scale.
Structured logs let you query: "show me all GPT-4o calls that took >5s yesterday."
"""

import logging
import json
import time

class StructuredLogger:
    """JSON-structured logger for API calls."""

    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def log_request(self, provider, model, input_tokens_est, request_id):
        """Log outgoing API request."""
        self.logger.info(json.dumps({
            "event": "api_request",
            "provider": provider,
            "model": model,
            "input_tokens_est": input_tokens_est,
            "request_id": request_id,
            "timestamp": time.time(),
        }))

    def log_response(self, provider, model, latency_ms, input_tokens, output_tokens,
                     cost, cached, used_fallback, request_id):
        """Log API response with full metrics."""
        self.logger.info(json.dumps({
            "event": "api_response",
            "provider": provider,
            "model": model,
            "latency_ms": latency_ms,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": cost,
            "cached": cached,
            "used_fallback": used_fallback,
            "request_id": request_id,
            "timestamp": time.time(),
        }))

    def log_error(self, provider, model, error_type, retryable, attempt, request_id):
        """Log API error."""
        self.logger.error(json.dumps({
            "event": "api_error",
            "provider": provider,
            "model": model,
            "error_type": error_type,
            "retryable": retryable,
            "attempt": attempt,
            "request_id": request_id,
            "timestamp": time.time(),
        }))
```

---

## Production-Ready API Wrapper

### Complete Wrapper Implementation

```python
"""
Production-ready AI API wrapper with all features:
- Multi-provider with fallback
- Retry with exponential backoff
- Circuit breaker
- Rate limiting
- Caching with TTL
- Cost tracking and limits
- Structured logging
- Metrics collection
"""

import os
import time
import json
import hashlib
import logging
import uuid
from typing import Optional, Dict, List, Any
from dataclasses import dataclass
from functools import wraps

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class APIConfig:
    """Configuration for API wrapper."""
    primary_provider: str = "openai"
    fallback_provider: str = "anthropic"
    max_retries: int = 3
    timeout: int = 30
    cache_enabled: bool = True
    rate_limit_rpm: int = 60
    cost_limit_daily: float = 100.0


class AIAPIWrapper:
    """
    Production-ready wrapper for multiple AI APIs.
    """

    def __init__(self, config: APIConfig = None):
        self.config = config or APIConfig()

        # Initialize clients
        self.clients = {}
        self._init_clients()

        # Initialize components
        self.cache = {} if self.config.cache_enabled else None
        self.rate_limiter = RateLimiter(self.config.rate_limit_rpm)
        self.cost_tracker = CostTracker()
        self.circuit_breaker = CircuitBreaker()
        self.metrics = APIMetrics()
        self.structured_logger = StructuredLogger("ai_api")

    def _init_clients(self):
        """Initialize API clients."""
        if os.getenv("OPENAI_API_KEY"):
            from openai import OpenAI
            self.clients["openai"] = OpenAI()

        if os.getenv("ANTHROPIC_API_KEY"):
            from anthropic import Anthropic
            self.clients["anthropic"] = Anthropic()

        if os.getenv("GOOGLE_API_KEY"):
            import google.generativeai as genai
            genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
            self.clients["google"] = genai

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        stream: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Send chat completion request with full production features.
        """
        request_id = str(uuid.uuid4())[:8]
        start_time = time.time()
        cached = False
        used_fallback = False

        # Check cost limits
        if self.cost_tracker.total_cost > self.config.cost_limit_daily:
            raise APIWrapperError(f"Daily cost limit exceeded: ${self.config.cost_limit_daily}")

        # Determine provider and model
        provider = self.config.primary_provider
        if model is None:
            model = self._default_model(provider)

        # Try cache (only for temperature=0)
        cache_key = self._get_cache_key(messages, model, temperature)
        if self.cache and temperature == 0 and cache_key in self.cache:
            entry = self.cache[cache_key]
            if time.time() - entry.get("timestamp", 0) < 3600:
                cached = True
                self.metrics.record_call(0, True, True, 0)
                return entry["response"]

        # Rate limiting
        estimated_tokens = self._estimate_tokens(messages)
        self.rate_limiter.acquire(estimated_tokens)

        # Log request
        self.structured_logger.log_request(provider, model, estimated_tokens, request_id)

        # Make request with circuit breaker
        try:
            result = self.circuit_breaker.call(
                self._make_request,
                provider, messages, model, temperature, max_tokens, stream, **kwargs
            )
        except ServiceUnavailableError:
            # Fallback to secondary provider
            used_fallback = True
            logger.warning(f"Primary provider {provider} unavailable, falling back")
            provider = self.config.fallback_provider
            model = self._default_model(provider)
            result = self._make_request(
                provider, messages, model, temperature, max_tokens, stream, **kwargs
            )

        # Calculate metrics
        latency_ms = (time.time() - start_time) * 1000
        cost = 0

        # Track costs
        if "usage" in result:
            cost_info = self.cost_tracker.log_usage(
                result["usage"]["input_tokens"],
                result["usage"]["output_tokens"],
                model
            )
            cost = cost_info["total_cost"]

        # Record metrics
        self.metrics.record_call(latency_ms, True, cached, cost, used_fallback)

        # Log response
        self.structured_logger.log_response(
            provider, model, latency_ms,
            result.get("usage", {}).get("input_tokens", 0),
            result.get("usage", {}).get("output_tokens", 0),
            cost, cached, used_fallback, request_id
        )

        # Cache result (only temperature=0)
        if self.cache and temperature == 0:
            self.cache[cache_key] = {"response": result, "timestamp": time.time()}

        return result

    def _make_request(
        self,
        provider: str,
        messages: List[Dict],
        model: str,
        temperature: float,
        max_tokens: int,
        stream: bool,
        **kwargs
    ) -> Dict[str, Any]:
        """Make request to specific provider."""

        if provider == "openai":
            return self._openai_request(messages, model, temperature, max_tokens, stream, **kwargs)
        elif provider == "anthropic":
            return self._anthropic_request(messages, model, temperature, max_tokens, stream, **kwargs)
        elif provider == "google":
            return self._google_request(messages, model, temperature, max_tokens, **kwargs)
        else:
            raise ValueError(f"Unknown provider: {provider}")

    @retry_with_exponential_backoff(max_retries=3)
    def _openai_request(self, messages, model, temperature, max_tokens, stream, **kwargs):
        """OpenAI-specific request."""
        client = self.clients["openai"]

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
            **kwargs
        )

        if stream:
            return {"stream": response}

        return {
            "content": response.choices[0].message.content,
            "finish_reason": response.choices[0].finish_reason,
            "usage": {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
            },
            "model": model,
            "provider": "openai",
        }

    @retry_with_exponential_backoff(max_retries=3)
    def _anthropic_request(self, messages, model, temperature, max_tokens, stream, **kwargs):
        """Anthropic-specific request."""
        client = self.clients["anthropic"]

        # Extract system prompt
        system_prompt = None
        filtered_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            else:
                filtered_messages.append(msg)

        response = client.messages.create(
            model=model,
            messages=filtered_messages,
            system=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        return {
            "content": response.content[0].text,
            "finish_reason": response.stop_reason,
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
            "model": model,
            "provider": "anthropic",
        }

    def _google_request(self, messages, model, temperature, max_tokens, **kwargs):
        """Google Gemini request."""
        genai = self.clients["google"]
        model_instance = genai.GenerativeModel(model)

        # Convert messages to Gemini format
        prompt = "\n".join([
            f"{msg['role'].upper()}: {msg['content']}"
            for msg in messages
        ])

        response = model_instance.generate_content(prompt)

        # Extract token counts (Gemini SDK provides usage_metadata)
        input_tokens = getattr(response.usage_metadata, 'prompt_token_count', 0) if hasattr(response, 'usage_metadata') else 0
        output_tokens = getattr(response.usage_metadata, 'candidates_token_count', 0) if hasattr(response, 'usage_metadata') else 0

        return {
            "content": response.text,
            "finish_reason": "stop",
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            },
            "model": model,
            "provider": "google",
        }

    def _default_model(self, provider: str) -> str:
        """Get default model for provider."""
        defaults = {
            "openai": "gpt-4o",
            "anthropic": "claude-3-5-sonnet-20241022",
            "google": "gemini-1.5-pro",
        }
        return defaults.get(provider, "gpt-4o")

    def _get_cache_key(self, messages, model, temperature) -> str:
        """Generate cache key."""
        data = json.dumps({
            "messages": messages,
            "model": model,
            "temperature": temperature,
        }, sort_keys=True)
        return hashlib.md5(data.encode()).hexdigest()

    def _estimate_tokens(self, messages) -> int:
        """Estimate token count."""
        return sum(len(m["content"]) // 4 for m in messages) + 100

    def get_cost_summary(self) -> Dict:
        """Get cost tracking summary."""
        return self.cost_tracker.get_summary()


# Usage example
"""
api = AIAPIWrapper(APIConfig(
    primary_provider="openai",
    fallback_provider="anthropic",
    cache_enabled=True,
    cost_limit_daily=50.0,
))

response = api.chat(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is machine learning?"},
    ],
    temperature=0.7,
)

print(response["content"])
print(f"Cost summary: {api.get_cost_summary()}")
"""
```

---

## Measuring API Integration Quality

### What to Measure in Production

When you deploy an API integration, you need to track these metrics to know whether it's actually working:

| Metric | What It Tells You | Target |
|--------|-------------------|--------|
| **Latency (p50, p95, p99)** | How fast responses arrive | p95 < 3s for chat |
| **Error rate** | How often calls fail | < 1% after retries |
| **Cache hit rate** | How effective your cache is | > 30% for repeated workloads |
| **Cost per request** | Average spend | Depends on model and use case |
| **Fallback trigger rate** | How often primary fails | < 5% |
| **Token efficiency** | Actual vs estimated tokens | Within 20% of estimate |

```python
"""
Simple metrics collector for API integration monitoring.
"""

import time
from collections import defaultdict


class APIMetrics:
    """Track API integration health metrics."""

    def __init__(self):
        self.latencies = []
        self.errors = 0
        self.successes = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.fallback_triggers = 0
        self.costs = []

    def record_call(self, latency_ms, success, cached, cost, used_fallback=False):
        """Record a single API call's metrics."""
        self.latencies.append(latency_ms)
        if success:
            self.successes += 1
        else:
            self.errors += 1
        if cached:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
        if used_fallback:
            self.fallback_triggers += 1
        self.costs.append(cost)

    def summary(self):
        """Get metrics summary."""
        total = self.successes + self.errors
        if total == 0:
            return {"status": "no calls recorded"}

        sorted_latencies = sorted(self.latencies)
        p50_idx = int(len(sorted_latencies) * 0.5)
        p95_idx = int(len(sorted_latencies) * 0.95)

        cache_total = self.cache_hits + self.cache_misses

        return {
            "total_calls": total,
            "error_rate": f"{(self.errors / total) * 100:.1f}%",
            "p50_latency_ms": sorted_latencies[p50_idx] if sorted_latencies else 0,
            "p95_latency_ms": sorted_latencies[p95_idx] if sorted_latencies else 0,
            "cache_hit_rate": f"{(self.cache_hits / cache_total) * 100:.1f}%" if cache_total else "N/A",
            "fallback_rate": f"{(self.fallback_triggers / total) * 100:.1f}%",
            "total_cost": f"${sum(self.costs):.4f}",
            "avg_cost_per_call": f"${sum(self.costs) / total:.6f}" if total else "$0",
        }
```

---

## Interview Preparation

### Concept Questions

**Q1: How do you handle rate limits in production?**

*Answer:* I implement a token bucket rate limiter that tracks both requests per minute and tokens per minute. Before each API call, I check available capacity and wait if needed. I also implement exponential backoff for 429 errors, starting with the `Retry-After` header if provided. For high-throughput applications, I use a distributed rate limiter (Redis-based) to coordinate across multiple servers.

**Q2: Explain the differences between OpenAI and Anthropic APIs.**

*Answer:* Key differences: (1) System prompts -- OpenAI uses a system message in the messages array, Anthropic uses a separate `system` parameter. (2) Model naming -- OpenAI uses versions like `gpt-4o`, Anthropic uses dates like `claude-3-5-sonnet-20241022`. (3) Streaming -- similar but different object structures. (4) Token counting -- OpenAI returns prompt_tokens/completion_tokens, Anthropic uses input_tokens/output_tokens. (5) Tokenizers -- OpenAI uses tiktoken, Anthropic uses its own tokenizer. (6) Context limits -- Claude 3.5 has 200K tokens, GPT-4o has 128K.

**Q3: How would you implement a fallback system for AI APIs?**

*Answer:* I use a circuit breaker pattern with multiple providers. Primary calls go to the preferred provider (e.g., OpenAI). If failures exceed a threshold (e.g., 3 in 1 minute), the circuit opens and routes to the fallback provider (e.g., Anthropic). After a recovery timeout, it tests the primary again. I also map models across providers (GPT-4o to Claude 3.5 Sonnet) to ensure comparable quality during fallback.

**Q4: What strategies do you use to reduce API costs?**

*Answer:* Five main strategies: (1) Caching -- identical requests return cached responses (cache hit rates depend heavily on your workload; repeated queries benefit most). (2) Model tiering -- use GPT-4o-mini for simple tasks, GPT-4o only when needed. (3) Prompt compression -- reduce token count by being concise. (4) Batching -- combine similar requests where possible. (5) Monitoring -- track costs in real-time and set alerts at 80% of budget.

### Coding Question

**Q5: Implement a simple retry mechanism with exponential backoff.**

```python
import time
import random
from functools import wraps
from openai import RateLimitError, InternalServerError, APIConnectionError

def retry_with_backoff(max_retries=3, base_delay=1.0, max_delay=30.0):
    """Decorator for retrying functions with exponential backoff."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except (RateLimitError, InternalServerError, APIConnectionError) as e:
                    last_exception = e

                    if attempt == max_retries - 1:
                        raise

                    # Calculate delay with jitter
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    delay = delay * (0.5 + random.random())

                    print(f"Attempt {attempt + 1} failed: {e}")
                    print(f"Retrying in {delay:.2f}s...")
                    time.sleep(delay)

            raise last_exception

        return wrapper
    return decorator

@retry_with_backoff(max_retries=3)
def call_api(prompt):
    """Example API call that might fail."""
    from openai import OpenAI
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content
```

**Q6: What happens when a streaming response fails mid-stream? How do you handle it?**

*Answer:* The connection drops and you have a partial response with no `finish_reason`. Three options: (1) Retry the entire request — simplest, but the user sees the response restart. (2) Retry with a "continue from" prompt including the partial content — works for text but unreliable for structured output. (3) Fall back to non-streaming for the retry — guarantees a complete response. Always track `finish_reason`: `stop` = complete, `length` = truncated by max_tokens, `None` = failure. For structured output (JSON), never parse until you confirm `finish_reason='stop'`.

**Q7: How do you handle idempotency when retrying API calls that trigger side effects?**

*Answer:* Chat completions are safe to retry (idempotent — they just generate text). But if the LLM uses function calling to trigger side effects (send email, create order), retrying can cause duplicates. Mitigation: separate generation from execution — let the LLM plan actions, then execute in a separate step with deduplication. Use idempotency keys (unique request IDs) so the execution layer can detect retries. Log all tool calls with unique IDs before execution; on retry, check the log first.

**Q8: What structured logging fields would you include for API call observability?**

*Answer:* Every API call should log: (1) request_id (correlation), (2) provider and model, (3) input/output token counts, (4) latency_ms, (5) cost_usd, (6) cached (bool), (7) used_fallback (bool), (8) error_type if applicable, (9) timestamp. JSON format for machine parsing. This enables queries like "show all GPT-4o calls >5s latency in the last hour" and powers dashboards for cost, latency percentiles, error rates, and cache hit rates.

### System Design Question

**Q9: Design a multi-tenant API gateway that serves 10K requests/second across multiple LLM providers.**

*Answer:*

```
Architecture:

1. Load Balancer (nginx / AWS ALB)
   - Routes requests to gateway instances
   - Health checks on /healthz endpoint

2. API Gateway (stateless, horizontally scalable)
   - Authentication: per-tenant API keys with rate limit tiers
   - Rate limiting: Redis-based sliding window (per-tenant + global)
   - Request routing: model → provider mapping with tenant overrides
   - Caching: Redis cache with TTL (only for temperature=0)

3. Provider Pool
   - OpenAI, Anthropic, Google with circuit breakers per provider
   - Connection pooling (reuse HTTP connections)
   - Retry with exponential backoff + jitter per request
   - Fallback chain: primary → secondary → tertiary provider

4. Observability
   - Structured JSON logs → log aggregator (Datadog / ELK)
   - Metrics: latency percentiles, error rate, cache hit rate, cost per tenant
   - Alerts: error rate >1%, p95 latency >5s, daily cost >threshold
   - Distributed tracing (OpenTelemetry) for request lifecycle

5. Cost Management
   - Per-tenant token budgets (daily/monthly limits in Redis)
   - Per-request cost estimation before execution (reject if over budget)
   - Async cost reconciliation (actual vs estimated) via usage logs

Scaling math:
- 10K req/s × avg 500ms latency = 5000 concurrent connections
- With async I/O: ~50 gateway instances (100 concurrent per instance)
- Redis: single instance handles 100K ops/s (sufficient for rate limits + cache)
- Cache: 30% hit rate → 3K req/s avoided → saves ~$1.5K/day at GPT-4o pricing

Key design decisions:
- Stateless gateway (all state in Redis) for horizontal scaling
- Separate rate limiting from retry logic (rate limit = admission control,
  retry = error recovery)
- Cache ONLY temperature=0 requests (non-deterministic results are uncacheable)
- Provider circuit breakers are per-instance (fast detection) + shared state
  in Redis (cluster-wide awareness)
```

---

## Exercises

### Exercise 1: Multi-Provider Wrapper
Build a wrapper that:
- Supports OpenAI and Anthropic
- Auto-selects provider based on availability
- Normalizes response format across providers

### Exercise 2: Cost Dashboard
Create a cost tracking dashboard that:
- Logs all API calls with costs
- Shows daily/weekly/monthly spending
- Alerts when approaching budget limits
- Breaks down costs by model and task type

### Exercise 3: Caching System
Implement a caching layer with:
- TTL-based expiration
- LRU eviction when cache is full
- Semantic similarity matching (not just exact match)

### Exercise 4: Rate Limit Simulator
Build a tool that:
- Simulates API rate limits
- Tests your rate limiting implementation
- Measures throughput under various conditions

### Exercise 5: Fallback System
Implement a complete fallback system:
- Primary: OpenAI GPT-4o
- Secondary: Anthropic Claude
- Tertiary: Local Llama (if available)
- Test failure scenarios

---

## Section Checkpoints

### Checkpoint 1 — After "API Fundamentals" and provider sections (OpenAI, Anthropic, Google)
1. Name three differences between OpenAI and Anthropic API message formats.
2. What is the purpose of the `finish_reason` field in a chat completion response?
3. Why should you NOT use tiktoken for counting Anthropic tokens?
4. Write code that streams a response from OpenAI and detects incomplete streams.

### Checkpoint 2 — After "Streaming Failure Modes" and "Error Handling"
1. What is the thundering herd problem and how does jitter mitigate it?
2. Which API errors are retryable vs terminal? Give two examples of each.
3. How does the circuit breaker pattern prevent cascading failures?
4. What are three streaming failure modes and their mitigations?

### Checkpoint 3 — After "Idempotency" and "Security"
1. When is retrying an API call dangerous? Give a concrete example.
2. Name three API security practices beyond environment variables.
3. What is the proxy pattern for API key management?
4. How do you detect anomalous API usage that might indicate abuse?

### Checkpoint 4 — After "Rate Limiting" and "Cost Optimization"
1. How does a token bucket rate limiter work?
2. Why should you only cache responses from temperature=0 calls?
3. Calculate the monthly cost of 100K GPT-4o calls (500 input, 200 output tokens each).
4. What is model tiering and when should you use it?

### Checkpoint 5 — After "Production Wrapper" and "Metrics"
1. What six metrics should you track for API integration health?
2. How do structured logs differ from plain text logs? Why does it matter at scale?
3. What does the AIAPIWrapper do when the primary provider's circuit breaker opens?
4. Design a cache invalidation strategy for an LLM API cache.

---

## Job Role Mapping

| Section | ML/AI Engineer | Data Scientist | AI Architect | Engineering Manager |
|---------|---------------|----------------|--------------|---------------------|
| API Fundamentals & Providers | Must know: all three SDKs, message format differences, tokenizer distinctions | Must know: basic API calls, embedding API for data tasks | Must know: provider comparison, context limits, pricing models | Must know: provider selection criteria, pricing implications |
| Streaming & Failure Modes | Must know: streaming implementation, failure detection, resilient_stream pattern | Must know: when streaming matters for user-facing apps | Must know: SSE architecture, streaming failure recovery | Must know: UX impact of streaming, failure mode risks |
| Error Handling & Retries | Must know: retry decorator, circuit breaker, error classification, idempotency | Must know: basic retry for batch processing | Must know: retry architecture, thundering herd, circuit breaker design | Must know: error budgets, SLA implications of retry strategies |
| Security | Must know: key rotation, proxy pattern, never-log-keys, per-user limits | Must know: env var basics, key security | Must know: full security architecture, proxy, IP allowlisting, anomaly detection | Must know: key management policy, compliance requirements |
| Rate Limiting & Cost | Must know: token bucket, async rate limiting, cost estimation, caching with TTL | Must know: rate limits for batch jobs, cost estimation | Must know: distributed rate limiting, Redis-based coordination, cost budgeting | Must know: cost forecasting, budget alerts, per-tenant limits |
| Production Wrapper & Metrics | Must know: multi-provider wrapper, structured logging, metrics integration | Must know: using the wrapper for data pipelines | Must know: gateway architecture, observability stack, scaling strategy | Must know: SLAs, monitoring dashboards, alerting thresholds |

---

## Summary

### Key Takeaways

1. **Authentication matters:** Use environment variables, never commit keys
2. **Always handle errors:** Retries with specific exception types, circuit breakers, fallbacks
3. **Rate limit proactively:** Don't wait for 429s
4. **Track costs religiously:** Monitor and set limits
5. **Cache aggressively:** Especially for repeated or deterministic queries
6. **Use streaming:** Essential for good UX in chat apps
7. **Use the right tokenizer:** tiktoken for OpenAI, Anthropic's SDK for Claude

### What's Next

In Blog 15, we'll build a Complete Chatbot:
- End-to-end chat application architecture
- Conversation management and memory
- User interface integration
- Deployment considerations

---

## Self-Assessment Rubric

Rate yourself honestly after completing this blog:

| Criteria | Excellent (9-10) | Good (7-8) | Needs Work (5-6) |
|----------|------------------|------------|-------------------|
| **API Fundamentals** | Can set up and authenticate with all three providers, explain differences | Can use one provider fluently | Struggles with authentication |
| **Error Handling** | Can implement retry logic with specific exceptions and circuit breakers | Understands retry concept, uses basic try/except | No error handling strategy |
| **Rate Limiting** | Can implement token bucket limiter, understands async concurrency issues | Knows rate limits exist, uses basic sleep | Ignores rate limits |
| **Cost Optimization** | Can implement caching, model tiering, and cost tracking | Uses caching for repeated calls | No cost awareness |
| **Production Readiness** | Can build multi-provider wrapper with fallback, monitoring, and metrics | Has a working wrapper for one provider | Code works but is not production-ready |

### What This Blog Does Well
- Complete working code for all three major providers (OpenAI, Anthropic, Google)
- Production patterns: retry logic with thundering herd explanation, circuit breakers, token bucket rate limiting, TTL-based caching with LRU eviction
- Streaming failure modes with resilient_stream() reconnection logic
- Idempotency rules separating safe-to-retry from dangerous-to-retry operations
- API security practices: key rotation, proxy pattern, anomaly detection, least-privilege keys
- Structured logging (JSON) with request/response/error methods and full observability fields
- APIMetrics fully wired into production wrapper with latency, cost, cache hit, and fallback tracking
- System design interview question (multi-tenant API gateway for 10K req/s) plus 8 other interview Qs
- Section checkpoints and job role mapping for 4 roles across 6 sections

### Where This Blog Falls Short
- The async rate limiter is simplified — production systems need distributed rate limiting (e.g., Redis-based) for multi-server deployments
- Cost estimation uses hardcoded prices that go stale quickly — a production system should fetch current pricing or use provider billing APIs
- The caching implementation is in-memory with TTL — production needs persistent caching (Redis, database) for multi-server consistency
- No load testing or benchmarking examples — the metrics collector is shown but never tested under realistic conditions
- Token counting across providers uses rough estimation for non-OpenAI models rather than precise provider-specific tokenizers
- No distributed tracing (OpenTelemetry) integration — structured logging covers single-service observability but not cross-service correlation

---

### Architect Sanity Checks

### Check 1: Production API Architecture Readiness
**Question**: Would you trust this person to build a production API integration handling failures, costs, rate limits, and reliability?
**Answer: YES** — The blog covers exponential backoff with jitter (including thundering herd explanation), circuit breaker pattern, token bucket rate limiting with actual-vs-estimated adjustment, TTL-based caching with LRU eviction and temperature-awareness, cost tracking, multi-provider fallback, structured JSON logging with full observability fields, and APIMetrics wired into the production wrapper. Idempotency rules distinguish safe-to-retry from dangerous-to-retry operations. API security covers key rotation, proxy patterns, and anomaly detection. Remaining gaps (distributed rate limiting, persistent caching, hardcoded prices) are explicitly called out as known limitations, which is itself a sign of production maturity.

### Check 2: Deep Failure Mode Understanding
**Question**: Can they diagnose API failures and implement appropriate recovery mechanisms?
**Answer: YES** — The error handling section covers specific exception types for both OpenAI and Anthropic (rate limits, auth failures, server errors, timeouts, connection issues), classifies each as retryable vs terminal, and implements appropriate recovery: exponential backoff for transient errors, immediate failure for auth/bad request errors, and circuit breaker for cascading failures. Streaming failure modes are covered explicitly (connection drops, incomplete JSON, rate limits mid-stream) with a resilient_stream() implementation that detects missing finish_reason and retries. Idempotency rules prevent dangerous retries of state-changing operations. The thundering herd problem is explained as the motivation for jitter.

### Check 3: Interview and Career Readiness
**Question**: Can they design API systems under constraints, implement production patterns, and solve real operational problems?
**Answer: YES** — Interview section includes 9 questions covering rate limiting, provider differences, fallback systems, cost optimization, streaming failures, idempotency, structured logging, and a full system design question (multi-tenant API gateway for 10K req/s with load balancer, Redis caching/rate-limiting, circuit breakers, and observability). Section checkpoints map to 4 job roles (ML/AI Engineer, Data Scientist, AI Architect, Engineering Manager) with role-specific focus areas. The coding question demonstrates retry logic with proper exception handling.

---

*Questions? Found an error? Comments are open. Technical corrections get priority.*
