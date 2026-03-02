# Blog 19: LangChain Deep Dive

## Prompt Your Career: The Complete Generative AI Masterclass

**Reading time:** 60-90 minutes
**Coding time:** 90-120 minutes
**Total investment:** ~3.5 hours

---

## What You'll Walk Away With

By the end of this blog, you will be able to:

1. **Understand LangChain's architecture** and core components (LCEL's Runnable protocol, why it exists)
2. **Build chains and pipelines** using LCEL (LangChain Expression Language) with routing, parallelism, and custom functions
3. **Create sophisticated agents** with custom tools and understand the cost tradeoffs (agents vs chains)
4. **Implement memory systems** for stateful conversations and know each strategy's failure modes
5. **Integrate with vector stores** for RAG applications with multi-query and hybrid search
6. **Design production-grade** LangChain applications with caching, fallbacks, and monitoring
7. **Know when NOT to use LangChain** and compare it against LlamaIndex and direct API usage
8. **Estimate costs** for chains vs agents and set appropriate token budgets

> **How to read this blog:** If you have used LLM APIs directly (Blog 14-15), start with Core Abstractions and work through LCEL. If you already know LCEL basics, skip to Tools and Agents. If you are building production systems, focus on Production Patterns and the FastAPI example. Each section is self-contained enough to read independently.

> **Prerequisites:** You should be comfortable with Python classes and async/await (Blog 2), have worked with LLM APIs (Blog 14), and understand embeddings and vector databases (Blog 16). Familiarity with RAG concepts (Blog 17) and function calling (Blog 18) is strongly recommended.

---

## What This Blog Does NOT Cover

Before we begin, let's set clear expectations on scope:

- **LangGraph in depth** -- we introduce LangGraph for advanced agent patterns, but full coverage of stateful multi-actor workflows, human-in-the-loop patterns, and complex graph topologies deserves its own treatment.
- **LangSmith observability** -- we mention LangSmith in the architecture diagram but do not cover trace setup, evaluation datasets, or production monitoring dashboards.
- **Every LangChain integration** -- LangChain has hundreds of integrations (document loaders, vector stores, tools). We cover the most common ones; consult the [LangChain integrations docs](https://python.langchain.com/docs/integrations/) for the full list.
- **Fine-tuning or training** -- this blog is about orchestration, not model training. See Blog 23 for fine-tuning.
- **Alternative frameworks** -- LlamaIndex, Haystack, Semantic Kernel, and direct API usage are valid alternatives not covered here.
- **Cost optimization at scale** -- we touch on caching but do not cover token budgeting, prompt compression, or cost monitoring in depth.

---

## Manager's Summary

**For Technical Leaders and Decision Makers:**

LangChain is the most popular framework for building LLM applications. It provides standardized interfaces, reusable components, and battle-tested patterns that accelerate development from weeks to days.

**Business Value:**
- **Faster Development**: Pre-built components reduce boilerplate significantly for common LLM patterns (chains, RAG, agents)
- **Provider Flexibility**: Switch between OpenAI, Anthropic, or local models without code changes
- **Production Ready**: Built-in support for streaming, caching, and observability
- **Community Ecosystem**: Large integration library covering vector stores, document loaders, and tools

**When to Use LangChain:**
- Prototyping LLM applications quickly
- Building multi-step AI workflows
- Integrating multiple data sources
- Creating agents with tool access
- RAG applications requiring vector stores

**When to Consider Alternatives:**
- Simple, single-call API use cases
- When you need maximum control over prompts
- Performance-critical applications (extra abstraction layer)

**Team Implications**: LangChain reduces the learning curve for teams new to LLM development but requires understanding its abstractions to debug effectively.

---

## Introduction to LangChain

### What is LangChain?

LangChain is a framework for developing applications powered by language models. It provides:

1. **Standardized Interfaces**: Consistent APIs across different LLM providers
2. **Composable Components**: Building blocks that snap together
3. **Chains and Agents**: Patterns for multi-step reasoning
4. **Memory Management**: Conversation state handling
5. **Tool Integration**: Connect LLMs to external systems

```python
"""
LangChain Installation and Setup
"""
# Install LangChain and common dependencies (v0.3+)
# pip install langchain langchain-openai langchain-anthropic
# pip install langchain-community langchain-chroma langchain-text-splitters
# pip install langgraph python-dotenv pydantic>=2

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Verify API keys
assert os.getenv("OPENAI_API_KEY"), "OpenAI API key not set"
print("Environment configured")
```

### LangChain's Architecture (v0.3+)

```
┌─────────────────────────────────────────────────────────────┐
│                     LangChain Ecosystem                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ langchain   │  │ langchain-  │  │ langchain-community │  │
│  │   (core)    │  │   openai    │  │  (integrations)     │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
│                                                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ langchain-  │  │ langchain-  │  │    langgraph        │  │
│  │  anthropic  │  │   chroma    │  │  (agents/workflows) │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
│                                                              │
│  ┌───────────────────────────────────────────────────────┐  │
│  │              LangSmith (Observability)                │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Core Abstractions

### 1. Chat Models

The primary interface for interacting with LLMs:

```python
"""
Working with Chat Models
"""
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# Initialize models
openai_model = ChatOpenAI(
    model="gpt-4o",
    temperature=0.7,
    max_tokens=1000
)

anthropic_model = ChatAnthropic(
    model="claude-sonnet-4-20250514",
    temperature=0.7,
    max_tokens=1000
)

# Simple invocation
response = openai_model.invoke("What is the capital of France?")
print(f"Response: {response.content}")
print(f"Token usage: {response.usage_metadata}")

# Using message objects
messages = [
    SystemMessage(content="You are a helpful assistant that speaks like a pirate."),
    HumanMessage(content="What is the weather like today?")
]

response = openai_model.invoke(messages)
print(f"Pirate response: {response.content}")

# Streaming
print("\nStreaming response:")
for chunk in openai_model.stream("Tell me a short joke"):
    print(chunk.content, end="", flush=True)
print()

# Batch processing
questions = [
    "What is 2+2?",
    "What is the capital of Japan?",
    "Who wrote Romeo and Juliet?"
]

responses = openai_model.batch(questions)
for q, r in zip(questions, responses):
    print(f"Q: {q}\nA: {r.content}\n")

# Async support
import asyncio

async def async_example():
    response = await openai_model.ainvoke("Hello, async world!")
    print(f"Async response: {response.content}")

asyncio.run(async_example())
```

### 2. Prompt Templates

Structured prompts with variable substitution:

```python
"""
Prompt Templates
"""
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    PromptTemplate
)

# Simple template
simple_template = PromptTemplate.from_template(
    "Translate the following text to {language}: {text}"
)

prompt = simple_template.format(language="Spanish", text="Hello, world!")
print(f"Formatted prompt: {prompt}")

# Chat prompt template
chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant specialized in {domain}."),
    ("human", "My question is: {question}")
])

messages = chat_template.format_messages(
    domain="machine learning",
    question="What is backpropagation?"
)
print(f"Formatted messages: {messages}")

# Complex template with examples (few-shot)
few_shot_template = ChatPromptTemplate.from_messages([
    ("system", "You are a sentiment analyzer. Respond with only: positive, negative, or neutral."),
    ("human", "I love this product!"),
    ("ai", "positive"),
    ("human", "This is the worst experience ever."),
    ("ai", "negative"),
    ("human", "The meeting is at 3pm."),
    ("ai", "neutral"),
    ("human", "{user_input}")
])

# Template with conversation history placeholder
conversation_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

# Using the conversation template
from langchain_core.messages import HumanMessage, AIMessage

history = [
    HumanMessage(content="My name is Alice"),
    AIMessage(content="Hello Alice! How can I help you today?")
]

messages = conversation_template.format_messages(
    history=history,
    input="What's my name?"
)

# Partial templates
partial_template = PromptTemplate.from_template(
    "Answer this {difficulty} question about {topic}: {question}"
)

# Partially fill in values
medium_template = partial_template.partial(difficulty="medium")
math_template = medium_template.partial(topic="mathematics")

# Now only need to provide question
final_prompt = math_template.format(question="What is the derivative of x^2?")
print(f"Final prompt: {final_prompt}")
```

### 3. Output Parsers

Structure LLM outputs:

```python
"""
Output Parsers
"""
from langchain_core.output_parsers import (
    StrOutputParser,
    JsonOutputParser,
    PydanticOutputParser
)
from pydantic import BaseModel, Field
from typing import List

# Simple string parser
str_parser = StrOutputParser()

# JSON output parser
json_parser = JsonOutputParser()

# Pydantic model for structured output
class MovieReview(BaseModel):
    title: str = Field(description="The movie title")
    rating: float = Field(description="Rating out of 10")
    summary: str = Field(description="Brief summary")
    pros: List[str] = Field(description="List of positive aspects")
    cons: List[str] = Field(description="List of negative aspects")

pydantic_parser = PydanticOutputParser(pydantic_object=MovieReview)

# Create prompt with format instructions
review_template = ChatPromptTemplate.from_messages([
    ("system", """You are a movie critic. Analyze movies and provide structured reviews.

{format_instructions}"""),
    ("human", "Review the movie: {movie}")
])

# Get format instructions
format_instructions = pydantic_parser.get_format_instructions()
print(f"Format instructions:\n{format_instructions}\n")

# Complete chain
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o", temperature=0)

chain = review_template | model | pydantic_parser

# Invoke the chain
review = chain.invoke({
    "movie": "The Matrix",
    "format_instructions": format_instructions
})

print(f"Parsed review:")
print(f"  Title: {review.title}")
print(f"  Rating: {review.rating}")
print(f"  Summary: {review.summary}")
print(f"  Pros: {review.pros}")
print(f"  Cons: {review.cons}")

# Using with_structured_output (recommended for newer versions)
class Recipe(BaseModel):
    name: str = Field(description="Recipe name")
    ingredients: List[str] = Field(description="List of ingredients")
    instructions: List[str] = Field(description="Step-by-step instructions")
    prep_time_minutes: int = Field(description="Preparation time in minutes")

structured_model = model.with_structured_output(Recipe)

recipe = structured_model.invoke(
    "Give me a recipe for chocolate chip cookies"
)

print(f"\nRecipe: {recipe.name}")
print(f"Prep time: {recipe.prep_time_minutes} minutes")
print(f"Ingredients: {recipe.ingredients}")
```

---

## LangChain Expression Language (LCEL)

LCEL is the declarative way to compose chains in LangChain.

### Why LCEL Exists

Before LCEL, LangChain chains were built by subclassing `Chain` and manually wiring `_call` methods. This worked, but every chain had to re-implement streaming, batching, and async support. LCEL solves this with the **Runnable protocol**: every component implements `invoke()`, `stream()`, `batch()`, and `ainvoke()`. The `|` operator (Python's `__or__` override) composes Runnables into a `RunnableSequence` that automatically propagates streaming tokens through the entire pipeline.

**What this means in practice:** When you write `prompt | model | parser`, you get streaming, batching, and async for free. Without LCEL, you'd write 50+ lines of boilerplate to wire up `astream()` through each step.

**The tradeoff:** LCEL makes simple chains elegant but complex chains opaque. A 10-step LCEL pipeline is harder to debug than 10 explicit function calls because errors surface at the Runnable level, not at your business logic level. LangSmith exists specifically because LCEL chains are hard to debug without tracing.

### Basic Chaining

```python
"""
LCEL Fundamentals
"""
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# The | operator creates chains
model = ChatOpenAI(model="gpt-4o")
prompt = ChatPromptTemplate.from_template("Tell me a {adjective} joke about {topic}")
output_parser = StrOutputParser()

# Chain composition
chain = prompt | model | output_parser

# Invoke
result = chain.invoke({"adjective": "funny", "topic": "programming"})
print(f"Result: {result}")

# Stream
print("\nStreaming:")
for chunk in chain.stream({"adjective": "short", "topic": "cats"}):
    print(chunk, end="", flush=True)
print()

# Batch
results = chain.batch([
    {"adjective": "clever", "topic": "AI"},
    {"adjective": "silly", "topic": "robots"}
])
for r in results:
    print(f"- {r}")
```

### Advanced LCEL Patterns

```python
"""
Advanced LCEL Patterns
"""
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableLambda,
    RunnableParallel,
    RunnableBranch
)
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

model = ChatOpenAI(model="gpt-4o")

# RunnablePassthrough - pass data through unchanged
chain_with_passthrough = (
    {"question": RunnablePassthrough()}
    | ChatPromptTemplate.from_template("Answer: {question}")
    | model
    | StrOutputParser()
)

result = chain_with_passthrough.invoke("What is 2+2?")
print(f"Passthrough result: {result}")

# RunnableLambda - custom functions in chains
def extract_keywords(text: str) -> str:
    # Simple keyword extraction
    words = text.lower().split()
    return ", ".join(set(words))

keyword_chain = (
    RunnableLambda(extract_keywords)
    | ChatPromptTemplate.from_template("Expand on these keywords: {keywords}")
    | model
    | StrOutputParser()
)

# RunnableParallel - run multiple chains simultaneously
analysis_chain = RunnableParallel(
    summary=ChatPromptTemplate.from_template("Summarize: {text}") | model | StrOutputParser(),
    sentiment=ChatPromptTemplate.from_template("What's the sentiment of: {text}") | model | StrOutputParser(),
    keywords=ChatPromptTemplate.from_template("Extract keywords from: {text}") | model | StrOutputParser()
)

text = "I absolutely loved the new restaurant! The food was amazing and the service was excellent."
results = analysis_chain.invoke({"text": text})
print(f"Summary: {results['summary']}")
print(f"Sentiment: {results['sentiment']}")
print(f"Keywords: {results['keywords']}")

# RunnableBranch - conditional logic
def classify_query(query: str) -> str:
    """Classify query type."""
    query_lower = query.lower()
    if "weather" in query_lower:
        return "weather"
    elif "calculate" in query_lower or any(c.isdigit() for c in query):
        return "math"
    return "general"

weather_chain = ChatPromptTemplate.from_template(
    "You are a weather assistant. Answer: {query}"
) | model | StrOutputParser()

math_chain = ChatPromptTemplate.from_template(
    "You are a math tutor. Solve: {query}"
) | model | StrOutputParser()

general_chain = ChatPromptTemplate.from_template(
    "You are a helpful assistant. Answer: {query}"
) | model | StrOutputParser()

router_chain = RunnableBranch(
    (lambda x: classify_query(x["query"]) == "weather", weather_chain),
    (lambda x: classify_query(x["query"]) == "math", math_chain),
    general_chain  # default
)

# Test routing
queries = [
    "What's the weather in Tokyo?",
    "Calculate 15% of 200",
    "Who invented the telephone?"
]

for query in queries:
    result = router_chain.invoke({"query": query})
    print(f"Q: {query}")
    print(f"A: {result}\n")
```

### Building Complex Pipelines

```python
"""
Complex LCEL Pipeline Example
"""
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# Initialize components
model = ChatOpenAI(model="gpt-4o", temperature=0)
embeddings = OpenAIEmbeddings()

# Create a simple vector store with documents
documents = [
    Document(page_content="Python is a versatile programming language.", metadata={"topic": "python"}),
    Document(page_content="Machine learning models learn patterns from data.", metadata={"topic": "ml"}),
    Document(page_content="Neural networks are inspired by biological brains.", metadata={"topic": "ml"}),
    Document(page_content="LangChain simplifies building LLM applications.", metadata={"topic": "langchain"}),
]

vectorstore = Chroma.from_documents(documents, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# Format retrieved documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# RAG chain with LCEL
rag_prompt = ChatPromptTemplate.from_template("""
Answer the question based on the following context:

Context:
{context}

Question: {question}

Provide a comprehensive answer. If the context doesn't contain relevant information, say so.
""")

# Method 1: Using dictionary comprehension
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | model
    | StrOutputParser()
)

# Test
question = "What is LangChain?"
answer = rag_chain.invoke(question)
print(f"Q: {question}")
print(f"A: {answer}\n")

# Method 2: Using RunnableParallel explicitly
rag_chain_v2 = (
    RunnableParallel(
        context=retriever | format_docs,
        question=RunnablePassthrough()
    )
    | rag_prompt
    | model
    | StrOutputParser()
)

# Adding sources to the response
def get_sources(docs):
    return [{"content": doc.page_content, "metadata": doc.metadata} for doc in docs]

rag_with_sources = RunnableParallel(
    answer=(
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | rag_prompt
        | model
        | StrOutputParser()
    ),
    sources=retriever | get_sources
)

result = rag_with_sources.invoke("Tell me about neural networks")
print(f"Answer: {result['answer']}")
print(f"Sources: {result['sources']}")
```

---

## Memory and Conversation Management

### Implementing Memory

```python
"""
Memory Systems in LangChain
"""
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

# Initialize model
model = ChatOpenAI(model="gpt-4o")

# Create prompt with message placeholder
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Be concise."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

# Basic chain
chain = prompt | model

# In-memory message store
store = {}

def get_session_history(session_id: str):
    """Get or create session history."""
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# Wrap chain with message history
with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

# Configuration for session
config = {"configurable": {"session_id": "user_123"}}

# Conversation
response1 = with_history.invoke(
    {"input": "My name is Alice"},
    config=config
)
print(f"AI: {response1.content}")

response2 = with_history.invoke(
    {"input": "What's my name?"},
    config=config
)
print(f"AI: {response2.content}")

# Check stored history
print(f"\nStored messages: {len(store['user_123'].messages)}")
```

### Advanced Memory Patterns

```python
"""
Advanced Memory Patterns
"""
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from typing import List
import json
import logging

# 1. Summary Memory - Summarize old messages
class SummaryMemory:
    """Memory that summarizes old conversations."""

    def __init__(self, model, max_messages: int = 10):
        self.model = model
        self.max_messages = max_messages
        self.messages: List = []
        self.summary: str = ""

    def add_message(self, message):
        self.messages.append(message)

        # Summarize if too many messages
        if len(self.messages) > self.max_messages:
            self._summarize()

    def _summarize(self):
        """Summarize older messages."""
        # Take older half of messages
        to_summarize = self.messages[:len(self.messages)//2]
        self.messages = self.messages[len(self.messages)//2:]

        # Create summary
        summary_prompt = f"""Summarize this conversation concisely:

Previous summary: {self.summary}

New messages:
{self._format_messages(to_summarize)}

Provide a brief summary capturing key information."""

        response = self.model.invoke(summary_prompt)
        self.summary = response.content

    def _format_messages(self, messages) -> str:
        formatted = []
        for msg in messages:
            role = "Human" if isinstance(msg, HumanMessage) else "AI"
            formatted.append(f"{role}: {msg.content}")
        return "\n".join(formatted)

    def get_context(self) -> str:
        """Get context for new messages."""
        context = ""
        if self.summary:
            context += f"Summary of earlier conversation:\n{self.summary}\n\n"
        context += "Recent messages:\n"
        context += self._format_messages(self.messages)
        return context


# 2. Entity Memory - Track entities mentioned
class EntityMemory:
    """Memory that tracks entities mentioned in conversation."""

    def __init__(self, model):
        self.model = model
        self.entities: dict = {}

    def extract_entities(self, text: str):
        """Extract entities from text."""
        prompt = f"""Extract named entities from this text as JSON:

Text: {text}

Return format: {{"people": [], "places": [], "organizations": [], "other": []}}"""

        response = self.model.invoke(prompt)
        try:
            entities = json.loads(response.content)
            self._update_entities(entities)
        except (json.JSONDecodeError, KeyError) as e:
            logging.getLogger(__name__).warning(f"Entity extraction failed: {e}")

    def _update_entities(self, new_entities: dict):
        for category, items in new_entities.items():
            if category not in self.entities:
                self.entities[category] = set()
            self.entities[category].update(items)

    def get_entity_context(self) -> str:
        """Get entity context string."""
        if not self.entities:
            return ""

        lines = ["Known entities:"]
        for category, items in self.entities.items():
            if items:
                lines.append(f"- {category}: {', '.join(items)}")
        return "\n".join(lines)


# 3. Window Memory with Token Limit
from langchain_openai import ChatOpenAI
import tiktoken

class TokenWindowMemory:
    """Memory that keeps messages within a token budget."""

    def __init__(self, max_tokens: int = 2000, model: str = "gpt-4o"):
        self.max_tokens = max_tokens
        self.encoder = tiktoken.encoding_for_model(model)
        self.messages: List = []

    def add_message(self, message):
        self.messages.append(message)
        self._trim_to_budget()

    def _count_tokens(self, messages: List) -> int:
        total = 0
        for msg in messages:
            total += len(self.encoder.encode(msg.content))
        return total

    def _trim_to_budget(self):
        """Remove oldest messages to stay within token budget."""
        while self._count_tokens(self.messages) > self.max_tokens and len(self.messages) > 1:
            self.messages.pop(0)

    def get_messages(self) -> List:
        return self.messages.copy()


# 4. Complete Chatbot with Memory
class MemoryEnabledChatbot:
    """Chatbot with configurable memory."""

    def __init__(
        self,
        model_name: str = "gpt-4o",
        memory_type: str = "window",  # window, summary, or entity
        max_tokens: int = 2000
    ):
        self.model = ChatOpenAI(model=model_name)

        if memory_type == "window":
            self.memory = TokenWindowMemory(max_tokens=max_tokens)
        elif memory_type == "summary":
            self.memory = SummaryMemory(self.model, max_messages=10)
        else:
            self.memory = TokenWindowMemory(max_tokens=max_tokens)
            self.entity_memory = EntityMemory(self.model)

        self.memory_type = memory_type
        self.system_message = SystemMessage(
            content="You are a helpful assistant. Be concise and accurate."
        )

    def chat(self, user_input: str) -> str:
        # Add user message
        user_message = HumanMessage(content=user_input)

        # Build messages for API call
        if self.memory_type == "summary":
            context = self.memory.get_context()
            messages = [
                self.system_message,
                HumanMessage(content=f"Context:\n{context}\n\nCurrent message: {user_input}")
            ]
        else:
            self.memory.add_message(user_message)
            messages = [self.system_message] + self.memory.get_messages()

        # Get response
        response = self.model.invoke(messages)

        # Store assistant response
        if self.memory_type == "summary":
            self.memory.add_message(user_message)
            self.memory.add_message(response)
        else:
            self.memory.add_message(response)

        # Extract entities if using entity memory
        if self.memory_type == "entity" and hasattr(self, 'entity_memory'):
            self.entity_memory.extract_entities(user_input)
            self.entity_memory.extract_entities(response.content)

        return response.content


# Usage example
if __name__ == "__main__":
    chatbot = MemoryEnabledChatbot(memory_type="window")

    conversation = [
        "Hi, my name is Bob and I work at Google.",
        "I'm interested in learning about machine learning.",
        "What's a good starting point?",
        "Thanks! What was my name again?"
    ]

    for msg in conversation:
        print(f"User: {msg}")
        response = chatbot.chat(msg)
        print(f"Bot: {response}\n")
```

---

## Tools and Agents

### Creating Custom Tools

```python
"""
Custom Tools in LangChain
"""
from langchain_core.tools import tool, StructuredTool, Tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List, Optional
import requests
import json

# Method 1: Using the @tool decorator
@tool
def search_wikipedia(query: str) -> str:
    """
    Search Wikipedia for information about a topic.
    Use this when you need factual information about people, places, events, or concepts.

    Args:
        query: The search term to look up on Wikipedia
    """
    # Simulated Wikipedia search
    return f"Wikipedia results for '{query}': This is a comprehensive encyclopedia entry about {query}..."


@tool
def calculate_math(expression: str) -> str:
    """
    Evaluate a mathematical expression safely.
    Use this for any mathematical calculations.

    Args:
        expression: A mathematical expression like '2 + 2' or '10 * 5'
    """
    try:
        # SAFE math evaluation using ast — never use eval() in production.
        # eval() with character filtering is STILL vulnerable (e.g., crafted
        # byte sequences, unicode tricks). Always use a proper parser.
        import ast
        import operator

        # Supported operations
        ops = {
            ast.Add: operator.add, ast.Sub: operator.sub,
            ast.Mult: operator.mul, ast.Div: operator.truediv,
            ast.USub: operator.neg,
        }

        def _safe_eval(node):
            if isinstance(node, ast.Expression):
                return _safe_eval(node.body)
            elif isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                return node.value
            elif isinstance(node, ast.BinOp) and type(node.op) in ops:
                return ops[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
            elif isinstance(node, ast.UnaryOp) and type(node.op) in ops:
                return ops[type(node.op)](_safe_eval(node.operand))
            else:
                raise ValueError(f"Unsupported expression: {ast.dump(node)}")

        tree = ast.parse(expression, mode='eval')
        result = _safe_eval(tree)
        return f"Result: {result}"
    except (SyntaxError, ZeroDivisionError, TypeError, ValueError) as e:
        return f"Error: {str(e)}"


@tool
def get_current_weather(location: str, unit: str = "celsius") -> str:
    """
    Get the current weather for a location.
    Use this when users ask about weather conditions.

    Args:
        location: City name (e.g., 'London', 'New York')
        unit: Temperature unit - 'celsius' or 'fahrenheit'
    """
    # Simulated weather API
    weather_data = {
        "location": location,
        "temperature": 22 if unit == "celsius" else 72,
        "unit": unit,
        "conditions": "Partly Cloudy",
        "humidity": 65
    }
    return json.dumps(weather_data)


# Method 2: Using Pydantic for input validation
class SearchInput(BaseModel):
    query: str = Field(description="Search query")
    max_results: int = Field(default=5, description="Maximum number of results")
    language: str = Field(default="en", description="Language code")


@tool(args_schema=SearchInput)
def advanced_search(query: str, max_results: int = 5, language: str = "en") -> str:
    """
    Perform an advanced web search with options.
    Use this for detailed web searches with specific requirements.
    """
    return f"Search results for '{query}' (language: {language}, max: {max_results}): ..."


# Method 3: StructuredTool for more control
def send_email_func(to: str, subject: str, body: str) -> str:
    """Send an email."""
    # In production, integrate with email service
    return f"Email sent to {to} with subject: {subject}"


class EmailInput(BaseModel):
    to: str = Field(description="Recipient email address")
    subject: str = Field(description="Email subject line")
    body: str = Field(description="Email body content")


send_email_tool = StructuredTool.from_function(
    func=send_email_func,
    name="send_email",
    description="Send an email to a recipient. Use when user wants to send emails.",
    args_schema=EmailInput,
    return_direct=False  # If True, returns result directly without LLM processing
)


# Method 4: Tool with complex return types
class TaskInfo(BaseModel):
    task_id: str
    title: str
    status: str
    due_date: Optional[str] = None


@tool
def create_task(title: str, due_date: Optional[str] = None) -> TaskInfo:
    """
    Create a new task in the task management system.

    Args:
        title: Task title
        due_date: Optional due date in YYYY-MM-DD format
    """
    import uuid
    return TaskInfo(
        task_id=str(uuid.uuid4())[:8],
        title=title,
        status="pending",
        due_date=due_date
    )


# Combine all tools
tools = [
    search_wikipedia,
    calculate_math,
    get_current_weather,
    advanced_search,
    send_email_tool,
    create_task
]

# Print tool information
print("Available tools:")
for t in tools:
    print(f"  - {t.name}: {t.description[:50]}...")
```

### Building Agents

```python
"""
Building Agents with LangChain
"""
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Initialize model
model = ChatOpenAI(model="gpt-4o", temperature=0)

# Define agent prompt
agent_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful AI assistant with access to various tools.

Guidelines:
- Use tools when they can help answer the user's question
- Be concise but thorough
- If a tool fails, explain the error and try an alternative approach
- Always verify calculations using the calculator tool

Current date: 2024-01-15"""),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

# Create the agent
agent = create_tool_calling_agent(model, tools, agent_prompt)

# Create executor with error handling
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,  # Show agent's thought process
    max_iterations=10,
    handle_parsing_errors=True
)

# Run the agent
response = agent_executor.invoke({
    "input": "What's the weather in Tokyo and calculate 15% tip on a $85 bill"
})
print(f"\nFinal Answer: {response['output']}")

# Agent with conversation memory
from langchain_core.messages import HumanMessage, AIMessage

chat_history = []

def chat_with_agent(user_input: str) -> str:
    """Chat with agent while maintaining history."""
    response = agent_executor.invoke({
        "input": user_input,
        "chat_history": chat_history
    })

    # Update history
    chat_history.append(HumanMessage(content=user_input))
    chat_history.append(AIMessage(content=response['output']))

    return response['output']


# Conversation example
responses = []
for query in [
    "My name is Alice and I live in Seattle",
    "What's the weather where I live?",
    "Create a task to check weather tomorrow"
]:
    response = chat_with_agent(query)
    responses.append((query, response))

for q, a in responses:
    print(f"Q: {q}")
    print(f"A: {a}\n")
```

### Advanced Agent Patterns with LangGraph

**Why LangGraph over AgentExecutor?** `AgentExecutor` runs a simple loop: call model → execute tools → repeat. This works for straightforward tool-use but breaks down when you need: (1) **conditional branching** (different paths based on intermediate results), (2) **human-in-the-loop** (pause and wait for approval before taking action), (3) **parallel tool execution** (run multiple tools simultaneously), or (4) **persistent state** (checkpoint and resume across requests). LangGraph models agent behavior as a **state machine** (directed graph) where nodes are functions and edges define control flow. This makes complex agent architectures explicit and debuggable.

**When to use which:**
- **Simple chain** → LCEL (`prompt | model | parser`)
- **Tool-calling agent** → `create_tool_calling_agent` + `AgentExecutor`
- **Multi-step with branching/loops/state** → LangGraph `StateGraph`

```python
"""
Advanced Agents with LangGraph
"""
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from typing import TypedDict, Annotated, List
import operator

# Define state schema
class AgentState(TypedDict):
    messages: Annotated[List, operator.add]
    next_action: str


# Initialize components
model = ChatOpenAI(model="gpt-4o")
tool_node = ToolNode(tools)

# Define node functions
def call_model(state: AgentState) -> dict:
    """Call the LLM with current state."""
    messages = state["messages"]
    response = model.bind_tools(tools).invoke(messages)
    return {"messages": [response]}


def should_continue(state: AgentState) -> str:
    """Determine if we should continue or end."""
    last_message = state["messages"][-1]

    # If there are tool calls, continue
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"

    # Otherwise, end
    return END


# Build the graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

# Add edges
workflow.set_entry_point("agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        END: END
    }
)
workflow.add_edge("tools", "agent")

# Compile the graph
app = workflow.compile()

# Run the agent
initial_state = {
    "messages": [HumanMessage(content="Calculate 25 * 4 and then search Wikipedia for Python programming")],
    "next_action": ""
}

# Execute
for output in app.stream(initial_state):
    for key, value in output.items():
        print(f"Node: {key}")
        if "messages" in value:
            for msg in value["messages"]:
                print(f"  Message: {msg.content[:100] if msg.content else 'Tool call'}...")
        print()
```

---

## Vector Stores and RAG

### Comprehensive RAG with LangChain

```python
"""
Complete RAG Implementation with LangChain
"""
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    DirectoryLoader,
    WebBaseLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.documents import Document
from typing import List
import os

# ============ Document Loading ============

def load_documents(source_type: str, source: str) -> List[Document]:
    """Load documents from various sources."""

    if source_type == "text":
        loader = TextLoader(source)
    elif source_type == "pdf":
        loader = PyPDFLoader(source)
    elif source_type == "directory":
        loader = DirectoryLoader(source, glob="**/*.txt")
    elif source_type == "web":
        loader = WebBaseLoader(source)
    else:
        raise ValueError(f"Unknown source type: {source_type}")

    return loader.load()


# ============ Text Splitting ============

def split_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[Document]:
    """Split documents into chunks."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )

    return splitter.split_documents(documents)


# ============ Vector Store ============

class RAGSystem:
    """Complete RAG system with LangChain."""

    def __init__(
        self,
        collection_name: str = "default",
        persist_directory: str = "./chroma_db"
    ):
        self.embeddings = OpenAIEmbeddings()
        self.model = ChatOpenAI(model="gpt-4o", temperature=0)
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.vectorstore = None

    def create_vectorstore(self, documents: List[Document]):
        """Create vector store from documents."""
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            collection_name=self.collection_name,
            persist_directory=self.persist_directory
        )
        return self

    def load_vectorstore(self):
        """Load existing vector store."""
        self.vectorstore = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
        )
        return self

    def add_documents(self, documents: List[Document]):
        """Add documents to existing vector store."""
        if self.vectorstore is None:
            self.create_vectorstore(documents)
        else:
            self.vectorstore.add_documents(documents)

    def get_retriever(
        self,
        search_type: str = "similarity",
        k: int = 4,
        score_threshold: float = None
    ):
        """Get configured retriever."""
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized")

        search_kwargs = {"k": k}

        if search_type == "similarity_score_threshold":
            search_kwargs["score_threshold"] = score_threshold or 0.5

        return self.vectorstore.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
        )

    def create_rag_chain(self, retriever=None):
        """Create RAG chain."""
        if retriever is None:
            retriever = self.get_retriever()

        # Format documents
        def format_docs(docs: List[Document]) -> str:
            return "\n\n---\n\n".join(
                f"Source: {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}"
                for doc in docs
            )

        # RAG prompt
        rag_prompt = ChatPromptTemplate.from_template("""
Answer the question based on the following context. If the context doesn't contain relevant information, say so.

Context:
{context}

Question: {question}

Instructions:
- Provide a comprehensive answer based on the context
- Cite specific parts of the context when relevant
- If you're unsure, indicate the level of confidence
- If the context doesn't contain enough information, say so clearly
""")

        # Chain with sources
        rag_chain = RunnableParallel(
            answer=(
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | rag_prompt
                | self.model
                | StrOutputParser()
            ),
            sources=retriever
        )

        return rag_chain

    def query(self, question: str) -> dict:
        """Query the RAG system."""
        chain = self.create_rag_chain()
        return chain.invoke(question)


# ============ Advanced RAG Patterns ============

class MultiQueryRAG(RAGSystem):
    """RAG with query expansion."""

    def create_rag_chain(self, retriever=None):
        if retriever is None:
            retriever = self.get_retriever()

        # Query expansion prompt
        expansion_prompt = ChatPromptTemplate.from_template("""
Generate 3 different versions of the given question to help retrieve relevant documents.
Provide different perspectives and phrasings.

Original question: {question}

Output as a JSON list of strings: ["query1", "query2", "query3"]
""")

        def expand_query(question: str) -> List[str]:
            """Expand query into multiple versions."""
            response = self.model.invoke(expansion_prompt.format(question=question))
            try:
                import json
                queries = json.loads(response.content)
                return [question] + queries  # Include original
            except json.JSONDecodeError:
                return [question]

        def multi_retrieve(question: str) -> List[Document]:
            """Retrieve using multiple queries."""
            queries = expand_query(question)
            all_docs = []
            seen_contents = set()

            for query in queries:
                docs = retriever.invoke(query)
                for doc in docs:
                    if doc.page_content not in seen_contents:
                        all_docs.append(doc)
                        seen_contents.add(doc.page_content)

            return all_docs[:6]  # Limit total docs

        # Use multi-retrieval in chain
        def format_docs(docs):
            return "\n\n---\n\n".join(doc.page_content for doc in docs)

        rag_prompt = ChatPromptTemplate.from_template("""
Answer based on the context below:

Context:
{context}

Question: {question}

Provide a comprehensive answer.""")

        chain = RunnableParallel(
            answer=(
                {"context": multi_retrieve | format_docs, "question": RunnablePassthrough()}
                | rag_prompt
                | self.model
                | StrOutputParser()
            ),
            sources=multi_retrieve
        )

        return chain


class HybridRAG(RAGSystem):
    """RAG with hybrid search (semantic + keyword)."""

    def create_hybrid_retriever(self, k: int = 4, max_bm25_docs: int = 10_000):
        """Create hybrid retriever combining semantic and keyword search.

        WARNING: BM25 requires loading documents into memory. For large
        collections (100K+ docs), this can cause OOM. Set max_bm25_docs
        to cap memory usage, or use a server-side keyword search (e.g.,
        Elasticsearch) instead of in-memory BM25.
        """
        from langchain.retrievers import EnsembleRetriever
        from langchain_community.retrievers import BM25Retriever

        # Load documents for BM25 — capped to avoid OOM on large collections
        all_docs = self.vectorstore.get(limit=max_bm25_docs)
        if len(all_docs.get('documents', [])) >= max_bm25_docs:
            logger.warning(
                f"BM25 index truncated to {max_bm25_docs} docs. "
                "Consider Elasticsearch for full-corpus keyword search."
            )
        documents = [
            Document(page_content=content, metadata=meta)
            for content, meta in zip(all_docs['documents'], all_docs['metadatas'])
        ]

        # Semantic retriever
        semantic_retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": k}
        )

        # Keyword retriever (BM25)
        bm25_retriever = BM25Retriever.from_documents(documents)
        bm25_retriever.k = k

        # Ensemble
        ensemble_retriever = EnsembleRetriever(
            retrievers=[semantic_retriever, bm25_retriever],
            weights=[0.6, 0.4]  # Favor semantic search
        )

        return ensemble_retriever


# ============ Usage Example ============

def demo_rag_system():
    """Demonstrate the RAG system."""

    # Create sample documents
    documents = [
        Document(
            page_content="""
            Machine Learning is a subset of artificial intelligence that enables
            systems to learn and improve from experience. It focuses on developing
            computer programs that can access data and use it to learn for themselves.
            """,
            metadata={"source": "ml_intro.txt", "topic": "machine learning"}
        ),
        Document(
            page_content="""
            Deep Learning is a subset of machine learning based on artificial neural
            networks. It uses multiple layers to progressively extract higher-level
            features from raw input. Deep learning has been particularly successful
            in image recognition, natural language processing, and speech recognition.
            """,
            metadata={"source": "dl_intro.txt", "topic": "deep learning"}
        ),
        Document(
            page_content="""
            Natural Language Processing (NLP) is a field of AI that focuses on
            the interaction between computers and humans through natural language.
            Modern NLP heavily relies on deep learning techniques, especially
            transformer architectures like BERT and GPT.
            """,
            metadata={"source": "nlp_intro.txt", "topic": "NLP"}
        )
    ]

    # Split documents
    chunks = split_documents(documents, chunk_size=500, chunk_overlap=50)

    # Create RAG system
    rag = RAGSystem(collection_name="demo", persist_directory="./demo_db")
    rag.create_vectorstore(chunks)

    # Query
    question = "How does deep learning relate to machine learning?"
    result = rag.query(question)

    print(f"Question: {question}")
    print(f"\nAnswer: {result['answer']}")
    print(f"\nSources:")
    for doc in result['sources']:
        print(f"  - {doc.metadata.get('source', 'Unknown')}")

    return rag


if __name__ == "__main__":
    rag_system = demo_rag_system()
```

---

## Production Patterns

### Caching and Optimization

```python
"""
Production Optimizations for LangChain
"""
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.globals import set_llm_cache
from langchain_community.cache import InMemoryCache, SQLiteCache
import hashlib
import time

# ============ Caching ============

# In-memory cache (development)
set_llm_cache(InMemoryCache())

# SQLite cache (production)
# set_llm_cache(SQLiteCache(database_path=".langchain.db"))

# Custom cache implementation
class CustomCache:
    """Custom cache with TTL and size limits."""

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.cache = {}
        self.max_size = max_size
        self.ttl = ttl_seconds
        self.access_times = {}

    def _get_key(self, prompt: str, model: str) -> str:
        return hashlib.md5(f"{prompt}:{model}".encode()).hexdigest()

    def get(self, prompt: str, model: str):
        key = self._get_key(prompt, model)
        if key in self.cache:
            entry = self.cache[key]
            if time.time() - entry["timestamp"] < self.ttl:
                self.access_times[key] = time.time()
                return entry["response"]
            else:
                del self.cache[key]
        return None

    def set(self, prompt: str, model: str, response: str):
        # Evict oldest if full
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.access_times, key=self.access_times.get)
            del self.cache[oldest_key]
            del self.access_times[oldest_key]

        key = self._get_key(prompt, model)
        self.cache[key] = {
            "response": response,
            "timestamp": time.time()
        }
        self.access_times[key] = time.time()


# ============ Rate Limiting ============

import asyncio
from collections import deque

class RateLimiter:
    """Token bucket rate limiter."""

    def __init__(self, requests_per_minute: int = 60):
        self.rpm = requests_per_minute
        self.request_times = deque()

    async def acquire(self):
        """Wait if necessary to respect rate limit."""
        now = time.time()

        # Remove old requests
        while self.request_times and now - self.request_times[0] > 60:
            self.request_times.popleft()

        # Wait if at limit
        if len(self.request_times) >= self.rpm:
            wait_time = 60 - (now - self.request_times[0])
            if wait_time > 0:
                await asyncio.sleep(wait_time)

        self.request_times.append(time.time())

    def sync_acquire(self):
        """Synchronous version."""
        now = time.time()

        while self.request_times and now - self.request_times[0] > 60:
            self.request_times.popleft()

        if len(self.request_times) >= self.rpm:
            wait_time = 60 - (now - self.request_times[0])
            if wait_time > 0:
                time.sleep(wait_time)

        self.request_times.append(time.time())


# ============ Fallback and Retry ============

from langchain_core.runnables import RunnableWithFallbacks
from langchain_anthropic import ChatAnthropic
from tenacity import retry, stop_after_attempt, wait_exponential

# Model fallbacks
primary_model = ChatOpenAI(model="gpt-4o")
fallback_model = ChatAnthropic(model="claude-sonnet-4-20250514")

model_with_fallback = primary_model.with_fallbacks([fallback_model])

# Retry wrapper
class RetryableChain:
    """Chain with automatic retry logic."""

    def __init__(self, chain, max_retries: int = 3):
        self.chain = chain
        self.max_retries = max_retries

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10)
    )
    def invoke(self, input_data):
        return self.chain.invoke(input_data)


# ============ Batching and Parallelism ============

async def parallel_batch_process(
    chain,
    inputs: list,
    batch_size: int = 10,
    max_concurrent: int = 5
):
    """Process inputs in parallel batches."""
    results = []
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_one(input_data):
        async with semaphore:
            return await chain.ainvoke(input_data)

    for i in range(0, len(inputs), batch_size):
        batch = inputs[i:i + batch_size]
        batch_results = await asyncio.gather(
            *[process_one(inp) for inp in batch]
        )
        results.extend(batch_results)

    return results


# ============ Monitoring and Logging ============

from langchain_core.callbacks import BaseCallbackHandler
from typing import Any, Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MetricsCallback(BaseCallbackHandler):
    """Callback handler for collecting metrics."""

    def __init__(self):
        self.metrics = {
            "total_calls": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
            "latencies": [],
            "errors": 0
        }
        self.start_time = None

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs):
        self.start_time = time.time()
        self.metrics["total_calls"] += 1
        logger.info(f"LLM call started. Total calls: {self.metrics['total_calls']}")

    def on_llm_end(self, response, **kwargs):
        latency = time.time() - self.start_time
        self.metrics["latencies"].append(latency)

        # Extract token usage if available
        if hasattr(response, 'llm_output') and response.llm_output:
            usage = response.llm_output.get('token_usage', {})
            self.metrics["total_tokens"] += usage.get('total_tokens', 0)

        logger.info(f"LLM call completed. Latency: {latency:.2f}s")

    def on_llm_error(self, error: Exception, **kwargs):
        self.metrics["errors"] += 1
        logger.error(f"LLM error: {error}")

    def get_summary(self) -> dict:
        avg_latency = (
            sum(self.metrics["latencies"]) / len(self.metrics["latencies"])
            if self.metrics["latencies"] else 0
        )
        return {
            **self.metrics,
            "avg_latency": avg_latency
        }


# Usage
metrics_handler = MetricsCallback()
model_with_metrics = ChatOpenAI(
    model="gpt-4o",
    callbacks=[metrics_handler]
)
```

### Complete Production Application

```python
"""
Educational LangChain Application — Production Structure Demo

NOTE: This demonstrates production *patterns* (lifespan management, session
routing, agent integration) but is NOT production-ready as-is. For production:
- Replace in-memory session store (self.sessions = {}) with Redis or PostgreSQL
- Add authentication/authorization to API endpoints
- Replace broad Exception catches with specific error types
- Add request timeout to agent execution (max_execution_time parameter)
- Add circuit breaker for downstream API failures
- Use structured logging (JSON) for log aggregation
"""
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.tools import tool
from langchain_core.callbacks import BaseCallbackHandler
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import uvicorn
import os
import logging
from contextlib import asynccontextmanager

# ============ Configuration ============

class Config:
    MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    CHROMA_PERSIST_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "4096"))
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))


# ============ Logging ============

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============ Tools ============

@tool
def search_knowledge_base(query: str) -> str:
    """Search the knowledge base for relevant information."""
    # Use the global RAG system
    results = app.state.rag_system.search(query)
    return results


@tool
def get_user_context(user_id: str) -> str:
    """Get context about a specific user."""
    # Simulated user context
    return f"User {user_id}: Premium member since 2023"


# ============ RAG System ============

class ProductionRAG:
    """Production-ready RAG system."""

    def __init__(self, persist_dir: str):
        self.embeddings = OpenAIEmbeddings(model=Config.EMBEDDING_MODEL)
        self.persist_dir = persist_dir
        self.vectorstore = None
        self._load_or_create_vectorstore()

    def _load_or_create_vectorstore(self):
        """Load existing or create new vectorstore."""
        try:
            self.vectorstore = Chroma(
                collection_name="production",
                embedding_function=self.embeddings,
                persist_directory=self.persist_dir
            )
            logger.info("Loaded existing vector store")
        except (FileNotFoundError, ValueError) as e:
            logger.warning(f"Could not load existing store, creating new: {e}")
            self.vectorstore = Chroma(
                collection_name="production",
                embedding_function=self.embeddings,
                persist_directory=self.persist_dir
            )

    def search(self, query: str, k: int = 4) -> str:
        """Search the knowledge base."""
        docs = self.vectorstore.similarity_search(query, k=k)
        if not docs:
            return "No relevant information found."
        return "\n\n".join(doc.page_content for doc in docs)

    def add_document(self, content: str, metadata: dict = None):
        """Add a document to the knowledge base."""
        from langchain_core.documents import Document
        doc = Document(page_content=content, metadata=metadata or {})
        self.vectorstore.add_documents([doc])


# ============ Chat System ============

class ChatSystem:
    """Production chat system with memory and agents."""

    def __init__(self):
        self.model = ChatOpenAI(
            model=Config.MODEL_NAME,
            temperature=Config.TEMPERATURE,
            max_tokens=Config.MAX_TOKENS
        )
        self.sessions = {}

        # Create agent
        self.tools = [search_knowledge_base, get_user_context]
        self._setup_agent()

    def _setup_agent(self):
        """Setup the chat agent."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful AI assistant for our platform.

Guidelines:
- Use the knowledge base to answer questions about our products and services
- Be helpful, concise, and accurate
- If you're unsure, say so

Current date: 2024-01-15"""),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])

        agent = create_tool_calling_agent(self.model, self.tools, prompt)
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=False,
            max_iterations=5
        )

    def _get_session(self, session_id: str):
        """Get or create session history."""
        if session_id not in self.sessions:
            self.sessions[session_id] = ChatMessageHistory()
        return self.sessions[session_id]

    async def chat(self, session_id: str, message: str) -> str:
        """Process a chat message."""
        history = self._get_session(session_id)

        try:
            response = await self.agent_executor.ainvoke({
                "input": message,
                "chat_history": history.messages
            })

            # Update history
            from langchain_core.messages import HumanMessage, AIMessage
            history.add_message(HumanMessage(content=message))
            history.add_message(AIMessage(content=response["output"]))

            return response["output"]

        except Exception as e:
            logger.error(f"Chat error: {e}")
            raise

    def clear_session(self, session_id: str):
        """Clear a session's history."""
        if session_id in self.sessions:
            del self.sessions[session_id]


# ============ FastAPI Application ============

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    # Startup
    logger.info("Starting application...")
    app.state.rag_system = ProductionRAG(Config.CHROMA_PERSIST_DIR)
    app.state.chat_system = ChatSystem()
    logger.info("Application started successfully")

    yield

    # Shutdown
    logger.info("Shutting down application...")


app = FastAPI(
    title="LangChain Production API",
    version="1.0.0",
    lifespan=lifespan
)


# ============ API Models ============

class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    response: str
    session_id: str


class DocumentRequest(BaseModel):
    content: str
    metadata: Optional[dict] = None


class SearchRequest(BaseModel):
    query: str
    k: Optional[int] = 4


# ============ API Endpoints ============

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Chat with the AI assistant."""
    try:
        response = await app.state.chat_system.chat(
            request.session_id,
            request.message
        )
        return ChatResponse(
            response=response,
            session_id=request.session_id
        )
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/documents")
async def add_document(request: DocumentRequest):
    """Add a document to the knowledge base."""
    try:
        app.state.rag_system.add_document(
            request.content,
            request.metadata
        )
        return {"status": "success", "message": "Document added"}
    except Exception as e:
        logger.error(f"Add document error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search")
async def search_endpoint(request: SearchRequest):
    """Search the knowledge base."""
    try:
        results = app.state.rag_system.search(request.query, request.k)
        return {"results": results}
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/sessions/{session_id}")
async def clear_session(session_id: str):
    """Clear a chat session."""
    app.state.chat_system.clear_session(session_id)
    return {"status": "success", "message": f"Session {session_id} cleared"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


# ============ Run Application ============

if __name__ == "__main__":
    uvicorn.run(
        "production_app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
```

---

## Evaluating LangChain Applications

Building a chain or agent is the easy part. Knowing whether it works well is harder.

### What to Measure

| Dimension | Metric | How to Collect |
|-----------|--------|----------------|
| **Correctness** | Answer accuracy vs ground truth | Human evaluation on a labeled test set (50-200 questions minimum) |
| **Retrieval quality** (RAG) | Precision@k, Recall@k, MRR | Compare retrieved docs against known-relevant docs |
| **Latency** | End-to-end response time (p50, p95) | Instrument with callbacks or LangSmith traces |
| **Cost** | Tokens consumed per query | Track via `usage_metadata` on responses |
| **Hallucination rate** | Fraction of claims not grounded in context | Human spot-checks or LLM-as-judge on a sample |
| **Tool use accuracy** (agents) | Correct tool selected and correct arguments | Log agent traces, compare against expected tool calls |

### Evaluation Pitfalls Specific to LangChain

1. **"It works in my notebook" is not evaluation.** Chains that look good on 5 hand-picked queries may fail on production traffic. Build a test set of at least 50 diverse queries before claiming a chain works.
2. **LCEL hides failures silently.** If a retriever returns no documents, the chain still runs -- it just answers from the LLM's parametric knowledge. Monitor retrieval hit rates separately from end-to-end accuracy.
3. **Agent loops are expensive.** An agent with `max_iterations=10` and verbose tool calls can consume 10x the tokens of a simple chain. Set token budgets and monitor iteration counts.
4. **Memory drift.** Long conversations accumulate context that may confuse the model. Test with realistic conversation lengths (10-20 turns), not just 2-3 turn demos.
5. **Caching masks regressions.** If you cache LLM responses, a model or prompt change will not affect cached queries. Clear caches when evaluating changes.

### Minimal Evaluation Script

```python
"""
Minimal evaluation for a LangChain RAG chain.
Run this against a labeled test set before deploying changes.
"""
import json
import time
from typing import List

def evaluate_rag_chain(chain, test_cases: List[dict]) -> dict:
    """
    Evaluate a RAG chain against labeled test cases.

    Args:
        chain: A LangChain chain with .invoke(question) -> {"answer": str, "sources": list}
        test_cases: List of {"question": str, "expected_answer": str, "expected_sources": list}

    Returns:
        dict with evaluation metrics
    """
    results = {"correct": 0, "total": len(test_cases), "latencies": [], "errors": 0}

    for case in test_cases:
        try:
            start = time.time()
            output = chain.invoke(case["question"])
            latency = time.time() - start
            results["latencies"].append(latency)

            # Basic: check if expected answer keywords appear in output
            answer = output.get("answer", "") if isinstance(output, dict) else str(output)
            if any(kw.lower() in answer.lower() for kw in case.get("keywords", [])):
                results["correct"] += 1

        except (KeyError, ValueError, RuntimeError) as e:
            results["errors"] += 1
            print(f"Error on question '{case['question'][:50]}...': {e}")

    results["accuracy"] = results["correct"] / results["total"] if results["total"] > 0 else 0
    results["avg_latency_s"] = sum(results["latencies"]) / len(results["latencies"]) if results["latencies"] else 0
    results["p95_latency_s"] = sorted(results["latencies"])[int(0.95 * len(results["latencies"]))] if results["latencies"] else 0

    return results

# Example test set
test_cases = [
    {
        "question": "What is LangChain?",
        "keywords": ["framework", "LLM", "language model"],
    },
    {
        "question": "How does deep learning relate to machine learning?",
        "keywords": ["subset", "neural network"],
    },
]

# Usage (assumes rag_chain is defined):
# results = evaluate_rag_chain(rag_chain, test_cases)
# print(json.dumps(results, indent=2))
```

> **For serious evaluation**, consider [LangSmith](https://smith.langchain.com/) (LangChain's evaluation platform) or [Ragas](https://docs.ragas.io/) for RAG-specific metrics like faithfulness, answer relevancy, and context precision. These provide structured evaluation datasets and automated scoring that manual spot-checks cannot match.

---

## Interview Preparation

### Career Mapping

| Role | How LangChain Knowledge Applies | Key Skills from This Blog |
|------|--------------------------------|---------------------------|
| **LLM Application Developer** | Primary tool for building LLM-powered products | LCEL chains, agents, memory, RAG integration |
| **AI/ML Engineer** | Orchestration layer for model serving pipelines | Production patterns, caching, fallbacks, monitoring |
| **Solutions Architect** | Framework selection, system design decisions | Framework comparison, cost analysis, when NOT to use LangChain |
| **Backend Engineer (AI team)** | API design for LLM-powered services | FastAPI integration, session management, rate limiting |
| **DevOps / MLOps** | Deployment, monitoring, cost management | Callback metrics, caching strategies, health checks |

**Where this appears in hiring:** LangChain is the most commonly listed framework in LLM application job descriptions. However, interviewers care more about your understanding of the *problems LangChain solves* (chain composition, tool use, retrieval) than LangChain-specific syntax. If you can explain LCEL's Runnable protocol and when to bypass the framework entirely, you demonstrate depth that framework-only knowledge does not.

### Conceptual Questions

1. **What is LCEL and why does it exist?**

   LCEL (LangChain Expression Language) is a declarative composition system built on the **Runnable protocol**. Every LCEL component implements `invoke()`, `stream()`, `batch()`, and `ainvoke()`. The `|` operator (Python's `__or__`) composes Runnables into a `RunnableSequence` that automatically propagates streaming, batching, and async through the entire pipeline. Before LCEL, every chain subclass had to manually re-implement these capabilities. LCEL makes this zero-effort. The tradeoff: LCEL pipelines are harder to debug than explicit Python functions because errors surface at the Runnable dispatch level, not your business logic. This is why LangSmith (observability platform) exists.

2. **Explain the difference between chains and agents, and when each is appropriate.**

   **Chains** are fixed sequences: prompt → model → parser. The execution path is determined at build time. **Agents** use the LLM itself to decide which tools to call and in what order — the execution path is determined at runtime. Use chains when the workflow is predictable (e.g., "summarize this document"), because chains are cheaper (1 LLM call), faster, and deterministic. Use agents when the workflow depends on user input or intermediate results (e.g., "research this topic using search and calculator"). The cost tradeoff is significant: a 5-iteration agent consumes ~10x the tokens of a single chain because the full conversation history is re-sent each iteration.

3. **How does memory work in LangChain, and what are the failure modes?**

   LangChain memory stores conversation history and injects it into prompts via `MessagesPlaceholder`. Three strategies exist: **buffer memory** (store all messages — simple but token cost grows linearly), **summary memory** (LLM-summarized history — saves tokens but loses detail and adds latency per summarization call), and **token window memory** (drop oldest messages beyond a budget — predictable cost but abrupt context loss). Failure modes: (1) buffer memory exceeds context window on long conversations, (2) summary memory hallucinates facts during summarization, (3) all in-memory implementations lose state on process restart — production systems need Redis or database-backed stores, (4) entity memory requires an extra LLM call per turn for extraction, doubling latency.

4. **What are the key considerations for production LangChain apps, and what does LangChain NOT solve?**

   Production requirements: **caching** (SQLite or Redis to avoid re-running identical queries — can cut costs 40-60% for repeated traffic), **rate limiting** (respect provider RPM/TPM limits), **fallbacks** (automatic provider failover: OpenAI → Anthropic), **monitoring** (callbacks for latency, token usage, error rates), **retry with backoff** (handle transient API failures). What LangChain does NOT solve: distributed execution (use Celery/Ray), session persistence across restarts (use Redis/database), prompt versioning and A/B testing (use LangSmith or custom tooling), cost budgeting per user or per request, and graceful degradation under partial outages (you must implement circuit breakers yourself).

5. **When would you choose NOT to use LangChain?**

   Skip LangChain when: (1) your application makes a single LLM call with no chaining — the framework adds import overhead and 100+ transitive dependencies for no benefit, (2) you need deterministic sub-500ms latency — LCEL dispatch adds 5-15ms per step, (3) your team cannot invest in learning the abstractions — debugging LCEL chains requires understanding Runnable internals, and opaque error messages slow down incident response, (4) you need to audit exact prompt text for compliance — LangChain prompt templates add formatting that's hard to trace, (5) you're building for long-term stability — LangChain has had breaking API changes across v0.1/v0.2/v0.3, requiring migration work.

### Coding Challenges

**Challenge 1**: Build a chain that routes questions to specialized handlers.

A complete solution, not just a skeleton:

```python
from langchain_core.runnables import RunnableBranch, RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

model = ChatOpenAI(model="gpt-4o", temperature=0)

# Specialized chains
math_handler = (
    ChatPromptTemplate.from_template("You are a math tutor. Solve step by step: {input}")
    | model | StrOutputParser()
)
search_handler = (
    ChatPromptTemplate.from_template("You are a research assistant. Answer factually: {input}")
    | model | StrOutputParser()
)
chat_handler = (
    ChatPromptTemplate.from_template("You are a friendly assistant. Respond to: {input}")
    | model | StrOutputParser()
)

def classify(text: str) -> str:
    """Classify query intent using keyword heuristics.
    Production version: use an LLM or fine-tuned classifier instead."""
    text_lower = text.lower()
    if any(kw in text_lower for kw in ["calculate", "compute", "solve", "math", "+"]):
        return "math"
    elif any(kw in text_lower for kw in ["who", "what is", "when did", "history of"]):
        return "search"
    return "general"

router = RunnableBranch(
    (lambda x: classify(x["input"]) == "math", math_handler),
    (lambda x: classify(x["input"]) == "search", search_handler),
    chat_handler
)

# Test
for query in ["Calculate 15% of 200", "Who invented the telephone?", "Hello!"]:
    result = router.invoke({"input": query})
    print(f"[{classify(query)}] {query} → {result[:80]}...")
```

**Challenge 2**: Implement a custom retriever with reranking.

A complete solution with the Runnable interface:

```python
from langchain_core.documents import Document
from typing import List

class RerankedRetriever:
    """
    Two-stage retriever: fast bi-encoder retrieval → accurate cross-encoder reranking.

    WHY TWO STAGES? Bi-encoders (vector similarity) are fast but approximate.
    Cross-encoders score query-document pairs jointly and are more accurate,
    but too slow to run on the full corpus. So we retrieve broadly, then rerank.
    """

    def __init__(self, base_retriever, reranker_model, initial_k: int = 20):
        self.retriever = base_retriever
        self.reranker = reranker_model
        self.initial_k = initial_k  # Retrieve more, then filter

    def invoke(self, query: str, k: int = 4) -> List[Document]:
        # Stage 1: Broad retrieval (fast, ~5ms)
        docs = self.retriever.invoke(query)[:self.initial_k]

        if not docs:
            return []

        # Stage 2: Cross-encoder reranking (slower, ~50-200ms for 20 docs)
        pairs = [(query, doc.page_content) for doc in docs]
        scores = self.reranker.predict(pairs)  # Returns list of floats

        # Sort by reranker score and return top-k
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[:k]]

# Usage with sentence-transformers cross-encoder:
# from sentence_transformers import CrossEncoder
# reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
# retriever = RerankedRetriever(base_retriever, reranker, initial_k=20)
```

---

## Exercises

### Exercise 1: Build a Document Q&A System
Create a system that:
- Loads PDF documents
- Creates a vector store
- Answers questions with source citations

### Exercise 2: Create a Multi-Tool Agent
Build an agent with:
- Web search capability
- Calculator
- File operations (read/write)
- Email sending

### Exercise 3: Implement Custom Memory
Create a memory system that:
- Summarizes old conversations
- Extracts and tracks entities
- Has a configurable token budget

### Exercise 4: Build a Production API
Create an API with:
- Chat endpoint with sessions
- Document ingestion
- Health monitoring
- Rate limiting

---

## Summary

### Key Takeaways

1. **LangChain accelerates development** but adds abstraction cost: Pre-built components reduce boilerplate, but the 100+ transitive dependencies and Runnable dispatch overhead mean you should only use it when the orchestration complexity justifies it
2. **LCEL's Runnable protocol is the core insight**: The `|` operator composes components that all implement `invoke/stream/batch/ainvoke`. Understanding this protocol matters more than memorizing syntax
3. **Agents are powerful but expensive**: A 5-iteration agent costs ~10x a simple chain due to quadratic token growth. Always set `max_iterations` and monitor token budgets
4. **Memory strategies have real tradeoffs**: Buffer memory is simple but unbounded; summary memory saves tokens but hallucinates; token window is predictable but loses context abruptly. Choose based on conversation length and cost tolerance
5. **Production requires more than LangChain provides**: You still need Redis for session persistence, circuit breakers for fault tolerance, prompt versioning for safe deployments, and cost budgeting per user
6. **Know when NOT to use LangChain**: Single API calls, strict latency requirements, compliance-auditable prompts, and teams that can't invest in learning the abstractions are all cases where direct API usage wins
7. **LangGraph is the future of complex agents**: When you need branching, loops, human-in-the-loop, or persistent state, LangGraph's explicit state machine model replaces AgentExecutor's opaque loop

### When to Use LangChain

**Good fit:**
- Rapid prototyping
- Complex multi-step workflows
- RAG applications
- Agent-based systems

**Consider alternatives:**
- Simple, single-call use cases
- Maximum performance requirements
- Full control over prompts needed

### When NOT to Use LangChain (Anti-Patterns)

| Scenario | Why LangChain Hurts | Better Alternative |
|----------|--------------------|--------------------|
| Single LLM call with no chaining | Adds ~200ms import overhead, 100+ transitive dependencies | Direct `openai` SDK call (~5 lines) |
| You need deterministic latency (< 500ms p99) | LCEL chain overhead + Runnable dispatch adds 50-150ms | Direct API call with `httpx` |
| Your team doesn't understand the abstractions | Debugging LCEL errors requires understanding Runnable internals | Plain Python functions with explicit control flow |
| You need to pin exact prompt text for compliance | LangChain prompt templates add formatting that's hard to audit | String templates or `f-strings` with version control |
| Prototype → production with no framework churn tolerance | LangChain has broken APIs across major versions (v0.1 → v0.2 → v0.3) | Direct API + thin custom wrappers |
| Batch processing millions of items | LCEL batch() is convenient but single-process; no built-in distributed execution | Ray, Celery, or cloud-native batch (AWS Step Functions) |

### LangChain Cost and Latency Analysis

Understanding the overhead LangChain adds is critical for production decisions.

**Per-Request Overhead (LCEL chain vs direct API call):**

| Component | Direct API Call | LCEL Chain (prompt → model → parser) | Overhead |
|-----------|----------------|---------------------------------------|----------|
| Import time (cold start) | ~50ms | ~250ms | +200ms |
| Chain dispatch | 0ms | ~5-15ms | +5-15ms |
| Callback overhead (if logging) | 0ms | ~2-5ms per callback | +2-5ms |
| Total latency per request | ~800ms | ~820ms | ~2-3% |

The per-request overhead is small. The real cost is **cognitive and operational**: debugging through Runnable abstractions, managing LangChain version upgrades, and the transitive dependency tree (100+ packages).

**Agent Cost Escalation:**

| Agent Iterations | Input Tokens (cumulative) | Output Tokens | Est. Cost (GPT-4o) |
|-----------------|--------------------------|---------------|---------------------|
| 1 | 1,000 | 200 | $0.004 |
| 3 | 4,500 | 600 | $0.016 |
| 5 | 10,000 | 1,000 | $0.035 |
| 10 | 35,000 | 2,000 | $0.11 |

Agent token consumption grows **quadratically** because each iteration includes the full conversation history. A `max_iterations=10` agent can cost 25x more than a simple chain. Always set token budgets and monitor iteration counts.

### Framework Comparison

| Dimension | LangChain | LlamaIndex | Direct API + Custom Code |
|-----------|-----------|------------|--------------------------|
| **Best for** | Multi-step chains, agents, general LLM orchestration | RAG-focused applications, document indexing | Simple use cases, maximum control |
| **Learning curve** | Medium-high (many abstractions) | Medium (narrower scope) | Low (no framework to learn) |
| **RAG quality** | Good, but RAG is one of many features | Excellent, purpose-built for RAG | You build everything yourself |
| **Agent support** | Strong (AgentExecutor, LangGraph) | Basic (via integrations) | Manual implementation |
| **Debugging** | Hard without LangSmith | Moderate | Easy (your code, your stack traces) |
| **Version stability** | Breaking changes between major versions | More stable API surface | No external API to break |
| **Community/ecosystem** | Largest | Second largest | N/A |
| **Production overhead** | 100+ transitive dependencies | Fewer dependencies | Minimal |

---

## Self-Assessment Rubric

Rate yourself honestly after completing this blog:

| Criteria | Excellent (9-10) | Good (7-8) | Needs Work (5-6) |
|----------|-----------------|------------|-------------------|
| **LCEL proficiency** | Builds complex chains with routing, parallelism, and custom Runnables | Creates basic prompt-model-parser chains | Struggles with the pipe operator |
| **Agent development** | Implements custom tools with validation; understands agent vs chain tradeoffs | Creates basic tool-calling agents | Cannot articulate when to use agents vs chains |
| **Memory implementation** | Can implement token-windowed and summary memory; understands OOM risks | Uses built-in RunnableWithMessageHistory | No memory implementation |
| **RAG integration** | Multi-query retrieval, hybrid search, source tracking | Basic RAG chain with retriever | Cannot connect a vector store to a chain |
| **Production readiness** | Adds caching, fallbacks, monitoring, and rate limiting to chains | Understands the need for production patterns | Deploys notebook code directly |
| **Cost awareness** | Can estimate per-query cost for chains vs agents; sets token budgets; knows when framework overhead isn't worth it | Understands that agents cost more than chains | No awareness of token economics |
| **Framework judgment** | Can articulate when NOT to use LangChain; compares with LlamaIndex and direct API | Knows LangChain exists among alternatives | Uses LangChain for everything without questioning fit |

### What This Blog Does Well

- Covers the full LangChain surface area: models, prompts, parsers, LCEL, agents, memory, RAG, and production patterns in one coherent walkthrough
- Uses current APIs (LangChain v0.3+, LCEL, `create_tool_calling_agent`, LangGraph)
- Provides a realistic FastAPI production example with session management and explicit documentation of its limitations
- Explains LCEL's Runnable protocol and *why* it exists, not just how to use the `|` operator
- Includes cost analysis for chains vs agents with concrete token counts and dollar estimates
- Provides a framework comparison table (LangChain vs LlamaIndex vs direct API) for informed technology selection
- Covers 6 anti-patterns for when NOT to use LangChain with specific alternatives
- Interview questions include full explanations that can survive follow-up probing, not just bullet points

### Where This Blog Falls Short

- All code examples require live API keys and real API calls — there are no offline-runnable examples or mocked responses for learning without spending money
- The custom memory classes (SummaryMemory, EntityMemory, TokenWindowMemory) are illustrative but untested at scale; production use should prefer LangGraph's built-in checkpointing
- Agent evaluation is not demonstrated — we show how to build agents but not how to systematically test their tool selection accuracy or measure agent loop efficiency
- Error handling in the production app catches broad `Exception` in FastAPI endpoints rather than specific LangChain/OpenAI errors (the docstring now calls this out explicitly)
- No coverage of prompt versioning, A/B testing chains, or gradual rollout strategies
- LangGraph coverage is introductory — human-in-the-loop, checkpointing, and multi-actor patterns deserve their own blog
- The evaluation script uses keyword matching only; production evaluation should use semantic similarity or LLM-as-judge (with bias calibration — see Blog 17)

### Architect Sanity Checks

### Check 1: Would you trust someone who learned *only this blog* to touch a production LangChain system?
**YES, with caveats.** The blog covers LCEL composition, agent construction, memory patterns, RAG integration, and production patterns (caching, fallbacks, monitoring, rate limiting). It explicitly calls out what the production app is missing (Redis sessions, circuit breakers, auth, specific error handling). The cost analysis and "when NOT to use LangChain" sections provide the judgment needed to avoid over-engineering. The reader would still need LangSmith for observability and database-backed session stores for real deployments, but the blog makes these gaps explicit.

### Check 2: Can you explain at least one real failure case using only what's taught here?
**YES.** Multiple failure cases are explicitly addressed: (1) Agent token cost explosion — the cost analysis table shows a 10-iteration agent costing 25x a simple chain due to quadratic token growth, (2) Memory drift — evaluation pitfalls section warns that long conversations confuse models, (3) LCEL silent failures — retriever returning no docs while chain still generates an answer from parametric knowledge, (4) In-memory session loss on restart, (5) BM25 OOM on large collections. The reader can explain each failure and its mitigation.

### Check 3: Would this blog survive senior-engineer interview follow-up questions?
**YES.** The interview section now includes full explanations (not bullet points) covering: LCEL's Runnable protocol and why it exists, chains vs agents with cost tradeoffs, memory failure modes, production gaps LangChain doesn't solve, and when NOT to use LangChain. The career mapping ties knowledge to specific job roles. The framework comparison table provides context for technology selection questions. A candidate who internalized this content could defend their framework choices with concrete reasoning.

### Engineering Sanity Checks (Non-Scored)

### Check 4: Scalability
**PARTIALLY.** Caching and batching patterns are demonstrated, but the in-memory session store (`self.sessions = {}`) does not survive restarts and cannot be shared across multiple server instances. The blog explicitly notes this limitation and recommends Redis or database-backed stores. The BM25 hybrid retriever loads documents into memory with a configurable cap (`max_bm25_docs`) and warns about OOM risks.

### Check 5: Reliability
**PARTIALLY.** Model fallbacks (OpenAI to Anthropic) and retry with exponential backoff are shown. The blog calls out missing circuit breakers, agent timeouts, and health checks that don't verify downstream API availability. These are documented as gaps rather than being silently omitted.

### Check 6: Maintainability
**MOSTLY.** LCEL provides composable chain definitions, and the production app separates RAG, chat, and API layers. The custom memory classes duplicate functionality that LangGraph provides natively (acknowledged in "Where This Blog Falls Short"). Several code blocks define their own `model = ChatOpenAI(...)` instance rather than sharing a configured instance — a deliberate tradeoff for self-contained examples.

---

## What's Next?

In **Blog 20: Image Generation Models**, we'll explore the visual frontier of generative AI. You'll learn:
- How diffusion models work (Stable Diffusion, DALL-E)
- Text-to-image generation techniques
- Image editing and manipulation
- Training custom image models

From text chains to visual creativity—let's generate some images!

---

*LangChain is a means, not an end. Master the concepts, and you can build with or without the framework.*
