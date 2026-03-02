# Blog 18: Function Calling and AI Agents

## Prompt Your Career: The Complete Generative AI Masterclass

**Reading time:** 60-90 minutes
**Coding time:** 90-120 minutes
**Total investment:** ~3.5 hours

**Prerequisites:**
- Blogs 14-15 (Working with AI APIs, Building a Chatbot) -- you need working knowledge of API calls to LLM providers
- Blog 17 (Building RAG Systems) -- understanding of how LLMs interact with external data
- Comfortable with Python classes, decorators, and async patterns
- An OpenAI or Anthropic API key with billing enabled (agent loops consume more tokens than single calls)

---

## What You'll Walk Away With

By the end of this blog, you will be able to:

1. **Explain how function calling works internally** — from schema injection to trained decision-making to structured output
2. **Implement function calling** with OpenAI, Anthropic, and Google APIs (and understand their differences)
3. **Design tool schemas** that LLMs can reliably invoke, and know why schema quality directly affects tool selection accuracy
4. **Choose between agent architectures** — ReAct vs. Plan-and-Execute with concrete tradeoffs (cost, latency, reliability)
5. **Know when NOT to use agents** — single-call tasks, high-reliability requirements, tight latency budgets
6. **Build autonomous agents** with error recovery, fallback strategies, and safety guardrails
7. **Estimate and control agent costs** — per-iteration cost analysis, quadratic token growth, budget enforcement
8. **Evaluate agent performance** with task success rate, failure categorization, and latency tracking

> **How to read this blog:** Start with "From Chat to Action" and "Understanding Function Calling" to build the conceptual foundation. Then work through the provider-specific implementations (pick the one matching your API key). The agent architecture sections (ReAct, Plan-and-Execute) build on each other, so read them in order. The Safety section is essential reading for anyone planning to deploy agents. If you are short on time, skip the Multi-Agent Systems section on your first pass.

---

## What This Blog Does NOT Cover

Before we begin, let's set clear expectations on scope:

- **LangChain and framework-specific patterns** -- Blog 19 covers LangChain in depth; this blog builds agents from scratch to understand the fundamentals.
- **Fine-tuning models for tool use** -- Blog 23 covers fine-tuning; here we use pre-trained models with function calling capabilities.
- **Production deployment and scaling** -- Blog 24 covers deployment pipelines; this blog focuses on agent logic, not infrastructure.
- **Embedding-based retrieval for agent context** -- Blog 16-17 cover embeddings and RAG; we assume you know how to retrieve context.
- **Formal verification of agent behavior** -- This is an active research area; we cover practical safety guardrails, not formal proofs of agent correctness.
- **Autonomous agents operating without human oversight** -- All patterns here include human-in-the-loop checkpoints; fully autonomous agents are outside scope.

---

## Manager's Summary

**For Technical Leaders and Decision Makers:**

Function calling transforms LLMs from text generators into action-takers. Instead of just answering questions, AI systems can now search databases, call APIs, execute code, and interact with external systems—all while maintaining natural conversation.

**Business Impact:**
- **Automation**: Automate complex workflows that previously required human intervention
- **Integration**: Connect AI to existing business systems (CRM, ERP, databases)
- **Reliability**: Structured outputs reduce parsing errors and improve system reliability
- **Scalability**: Build AI assistants that handle thousands of concurrent tasks

**Key Decisions:**
- **Tool Design**: Well-designed tool schemas dramatically improve success rates
- **Safety Boundaries**: Define clear limits on what agents can and cannot do
- **Human-in-the-Loop**: Determine when human approval is required
- **Cost Management**: Agent loops can multiply API costs—implement guardrails

**ROI Consideration**: Well-defined, repetitive workflows (e.g., data lookup, form filling, routing) see the highest returns from agent automation. However, agent systems require careful testing and monitoring to prevent unintended actions, and ROI varies widely depending on task complexity and error tolerance.

---

## From Chat to Action: The Evolution

Traditional chatbots had a fundamental limitation—they could only respond with text. Need to check inventory? The bot tells you how to check, but can't actually look it up. Need to book a meeting? It provides instructions, not a booked meeting.

Function calling changes everything:

```
Traditional Flow:
User: "What's the weather in Tokyo?"
Bot: "You can check weather.com or use a weather app to find Tokyo's current weather."

Function Calling Flow:
User: "What's the weather in Tokyo?"
Bot: [Calls get_weather(location="Tokyo")]
     "It's currently 22°C and sunny in Tokyo with 65% humidity."
```

---

## Understanding Function Calling

### The Core Concept

Function calling (also called "tool use") allows LLMs to:
1. Recognize when a user request requires external action
2. Generate structured parameters for that action
3. Receive and incorporate results into their response

```python
"""
Function Calling Architecture
"""
from dataclasses import dataclass
from typing import Any, Callable
import json

@dataclass
class Tool:
    """Represents a callable tool for the LLM."""
    name: str
    description: str
    parameters: dict  # JSON Schema
    function: Callable

    def to_openai_format(self) -> dict:
        """Convert to OpenAI function format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }

    def to_anthropic_format(self) -> dict:
        """Convert to Anthropic tool format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters
        }

    def execute(self, **kwargs) -> Any:
        """Execute the tool with given parameters."""
        return self.function(**kwargs)


class ToolRegistry:
    """Registry for managing available tools."""

    def __init__(self):
        self.tools: dict[str, Tool] = {}

    def register(self, tool: Tool):
        """Register a new tool."""
        self.tools[tool.name] = tool
        return tool

    def get(self, name: str) -> Tool | None:
        """Get tool by name."""
        return self.tools.get(name)

    def list_tools(self) -> list[Tool]:
        """List all registered tools."""
        return list(self.tools.values())

    def to_openai_format(self) -> list[dict]:
        """Get all tools in OpenAI format."""
        return [tool.to_openai_format() for tool in self.tools.values()]

    def to_anthropic_format(self) -> list[dict]:
        """Get all tools in Anthropic format."""
        return [tool.to_anthropic_format() for tool in self.tools.values()]
```

### How Function Calling Works Under the Hood

When you pass `tools=[...]` to an API call, the model doesn't execute code. Here's what actually happens:

1. **Schema Injection.** The provider serializes your tool schemas into the system prompt (or a special token region). The model sees the tool names, descriptions, and parameter schemas as part of its context.

2. **Trained Decision-Making.** During fine-tuning, the model was trained on examples of conversations where tool calls were appropriate. It learned to emit a structured output (JSON with tool name and arguments) when the user's request matches a tool's description. This is why **tool descriptions matter so much** — the model's decision is based on semantic matching between the user query and the tool description.

3. **Structured Output Generation.** When the model decides to call a tool, it generates a JSON object conforming to the parameter schema. The provider's API layer parses this and returns it as a `tool_call` object (OpenAI) or `tool_use` content block (Anthropic) — not as free text.

4. **Your Code Executes.** The actual function execution happens in your code, not in the model or API. You parse the tool call, run the function, and send the result back.

5. **Result Integration.** The model receives the tool result as a new message and generates a natural language response incorporating it.

**Why this matters:** Understanding this mechanism explains common failure modes:
- **Vague tool descriptions → model picks wrong tool** (decision is based on description matching)
- **Schema mismatch → model generates invalid parameters** (model follows schema but can still hallucinate values)
- **Too many tools → model gets confused** (all schemas compete for attention in the context window; keep to 10-20 tools max for reliable selection)

**Native function calling vs. ReAct text parsing:**

| Aspect | Native Function Calling (API-level) | ReAct (Text Parsing) |
|--------|-------------------------------------|---------------------|
| **Reliability** | Higher — structured output enforced by API | Lower — depends on regex parsing, model may deviate from format |
| **Provider lock-in** | Yes — each provider has different format | No — works with any LLM that generates text |
| **Parallel tool calls** | Supported (OpenAI) | Not naturally supported |
| **Reasoning transparency** | Opaque — model decides internally | Explicit — "Thought" step visible in output |
| **Best for** | Production systems with single provider | Prototyping, multi-provider, research |

### Designing Effective Tool Schemas

The quality of your tool schemas directly impacts how reliably the LLM uses them:

```python
"""
Tool Schema Design Best Practices
"""

# Bad: Vague description, unclear parameters
bad_weather_tool = {
    "name": "weather",
    "description": "Gets weather",
    "parameters": {
        "type": "object",
        "properties": {
            "loc": {"type": "string"}
        }
    }
}

# Good: Clear description, well-defined parameters with constraints
good_weather_tool = {
    "name": "get_current_weather",
    "description": "Get the current weather conditions for a specific location. Returns temperature, humidity, conditions, and wind speed. Use this when users ask about current weather, temperature, or if they should bring an umbrella.",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "City and country (e.g., 'Tokyo, Japan' or 'New York, USA')"
            },
            "units": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "description": "Temperature unit preference",
                "default": "celsius"
            }
        },
        "required": ["location"]
    }
}

# Comprehensive tool definitions
def create_tool_schemas() -> list[dict]:
    """Create well-designed tool schemas."""

    return [
        {
            "name": "search_database",
            "description": """Search the product database for items matching criteria.
            Use this when users ask about:
            - Finding products by name, category, or features
            - Checking product availability
            - Comparing product specifications
            Returns up to 10 matching products with details.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (product name, keywords, or description)"
                    },
                    "category": {
                        "type": "string",
                        "enum": ["electronics", "clothing", "home", "sports", "all"],
                        "description": "Product category filter",
                        "default": "all"
                    },
                    "price_max": {
                        "type": "number",
                        "description": "Maximum price filter in USD"
                    },
                    "in_stock_only": {
                        "type": "boolean",
                        "description": "Only return items currently in stock",
                        "default": True
                    },
                    "sort_by": {
                        "type": "string",
                        "enum": ["relevance", "price_low", "price_high", "rating"],
                        "default": "relevance"
                    }
                },
                "required": ["query"]
            }
        },
        {
            "name": "create_calendar_event",
            "description": """Create a new calendar event. Use this when users want to:
            - Schedule meetings or appointments
            - Set reminders for specific times
            - Block time on their calendar
            Returns the created event details with confirmation.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Event title/name"
                    },
                    "start_time": {
                        "type": "string",
                        "description": "Start time in ISO 8601 format (e.g., '2024-01-15T14:00:00')"
                    },
                    "end_time": {
                        "type": "string",
                        "description": "End time in ISO 8601 format"
                    },
                    "description": {
                        "type": "string",
                        "description": "Optional event description or notes"
                    },
                    "attendees": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of attendee email addresses"
                    },
                    "location": {
                        "type": "string",
                        "description": "Physical location or video call link"
                    },
                    "reminder_minutes": {
                        "type": "integer",
                        "description": "Send reminder this many minutes before",
                        "default": 15
                    }
                },
                "required": ["title", "start_time", "end_time"]
            }
        },
        {
            "name": "execute_sql_query",
            "description": """Execute a read-only SQL query against the analytics database.
            IMPORTANT: Only SELECT queries are allowed. No INSERT, UPDATE, DELETE.
            Use for: sales reports, user analytics, inventory checks.
            Returns query results as JSON.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "SQL SELECT query to execute"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum rows to return",
                        "default": 100,
                        "maximum": 1000
                    }
                },
                "required": ["query"]
            }
        }
    ]
```

---

## Function Calling with Different Providers

### OpenAI Function Calling

```python
"""
OpenAI Function Calling Implementation
"""
from openai import OpenAI
import json
from typing import Any

client = OpenAI()

# Define tools
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Get the current stock price for a given ticker symbol",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol (e.g., 'AAPL', 'GOOGL')"
                    }
                },
                "required": ["ticker"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_mortgage",
            "description": "Calculate monthly mortgage payment",
            "parameters": {
                "type": "object",
                "properties": {
                    "principal": {
                        "type": "number",
                        "description": "Loan amount in dollars"
                    },
                    "annual_rate": {
                        "type": "number",
                        "description": "Annual interest rate as percentage (e.g., 6.5)"
                    },
                    "years": {
                        "type": "integer",
                        "description": "Loan term in years"
                    }
                },
                "required": ["principal", "annual_rate", "years"]
            }
        }
    }
]

# Tool implementations
def get_stock_price(ticker: str) -> dict:
    """Simulate stock price lookup."""
    # In production, call a real stock API
    prices = {
        "AAPL": 178.50,
        "GOOGL": 141.25,
        "MSFT": 378.90,
        "AMZN": 178.35
    }
    price = prices.get(ticker.upper())
    if price:
        return {"ticker": ticker.upper(), "price": price, "currency": "USD"}
    return {"error": f"Unknown ticker: {ticker}"}

def calculate_mortgage(principal: float, annual_rate: float, years: int) -> dict:
    """Calculate mortgage payment."""
    monthly_rate = annual_rate / 100 / 12
    num_payments = years * 12

    if monthly_rate == 0:
        monthly_payment = principal / num_payments
    else:
        monthly_payment = principal * (
            monthly_rate * (1 + monthly_rate) ** num_payments
        ) / ((1 + monthly_rate) ** num_payments - 1)

    total_payment = monthly_payment * num_payments
    total_interest = total_payment - principal

    return {
        "monthly_payment": round(monthly_payment, 2),
        "total_payment": round(total_payment, 2),
        "total_interest": round(total_interest, 2)
    }

# Map function names to implementations
TOOL_FUNCTIONS = {
    "get_stock_price": get_stock_price,
    "calculate_mortgage": calculate_mortgage
}


def execute_tool(name: str, arguments: dict) -> Any:
    """Execute a tool by name with given arguments."""
    if name not in TOOL_FUNCTIONS:
        return {"error": f"Unknown tool: {name}"}
    return TOOL_FUNCTIONS[name](**arguments)


def chat_with_tools(user_message: str, messages: list = None) -> str:
    """
    Complete chat interaction with function calling.
    Handles the full loop of tool calls and responses.
    """
    if messages is None:
        messages = []

    messages.append({"role": "user", "content": user_message})

    # Initial API call
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=tools,
        tool_choice="auto"  # Let model decide when to use tools
    )

    assistant_message = response.choices[0].message

    # Check if model wants to call tools
    while assistant_message.tool_calls:
        # Add assistant's message with tool calls
        messages.append(assistant_message)

        # Execute each tool call
        for tool_call in assistant_message.tool_calls:
            function_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)

            print(f"Calling tool: {function_name}({arguments})")

            # Execute the tool
            result = execute_tool(function_name, arguments)

            # Add tool result to messages
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result)
            })

        # Get next response (may include more tool calls)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )

        assistant_message = response.choices[0].message

    # Final response (no more tool calls)
    final_response = assistant_message.content
    messages.append({"role": "assistant", "content": final_response})

    return final_response


# Example usage
if __name__ == "__main__":
    # Single tool call
    response = chat_with_tools("What's Apple's current stock price?")
    print(f"Response: {response}\n")

    # Multiple tool calls in one request
    response = chat_with_tools(
        "Compare the stock prices of Apple and Google, "
        "and also calculate the monthly payment for a $500,000 mortgage at 6.5% for 30 years"
    )
    print(f"Response: {response}")
```

### Anthropic Tool Use

```python
"""
Anthropic Tool Use Implementation
"""
import anthropic
import json
from typing import Any

client = anthropic.Anthropic()

# Define tools in Anthropic format
tools = [
    {
        "name": "search_web",
        "description": "Search the web for current information. Use this for recent events, news, or any information that might have changed since the knowledge cutoff.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query"
                },
                "num_results": {
                    "type": "integer",
                    "description": "Number of results to return",
                    "default": 5
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "send_email",
        "description": "Send an email to a recipient. Use this when users want to compose and send emails.",
        "input_schema": {
            "type": "object",
            "properties": {
                "to": {
                    "type": "string",
                    "description": "Recipient email address"
                },
                "subject": {
                    "type": "string",
                    "description": "Email subject line"
                },
                "body": {
                    "type": "string",
                    "description": "Email body content"
                },
                "cc": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "CC recipients"
                }
            },
            "required": ["to", "subject", "body"]
        }
    }
]

# Tool implementations
def search_web(query: str, num_results: int = 5) -> dict:
    """Simulate web search."""
    # In production, use a real search API
    return {
        "query": query,
        "results": [
            {
                "title": f"Result {i+1} for '{query}'",
                "url": f"https://example.com/{i}",
                "snippet": f"This is a sample search result about {query}..."
            }
            for i in range(num_results)
        ]
    }

def send_email(to: str, subject: str, body: str, cc: list = None) -> dict:
    """Simulate sending email."""
    # In production, use email service
    return {
        "status": "sent",
        "to": to,
        "subject": subject,
        "message_id": "msg_12345",
        "timestamp": "2024-01-15T10:30:00Z"
    }

TOOL_FUNCTIONS = {
    "search_web": search_web,
    "send_email": send_email
}


def chat_with_tools_anthropic(user_message: str, messages: list = None) -> str:
    """
    Anthropic chat with tool use.
    Handles the tool use loop with proper stop reason checking.
    """
    if messages is None:
        messages = []

    messages.append({"role": "user", "content": user_message})

    while True:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            tools=tools,
            messages=messages
        )

        # Check stop reason
        if response.stop_reason == "end_turn":
            # Model finished without tool calls
            return response.content[0].text

        elif response.stop_reason == "tool_use":
            # Process tool calls
            assistant_content = response.content
            messages.append({"role": "assistant", "content": assistant_content})

            tool_results = []

            for block in assistant_content:
                if block.type == "tool_use":
                    tool_name = block.name
                    tool_input = block.input
                    tool_use_id = block.id

                    print(f"Calling tool: {tool_name}({tool_input})")

                    # Execute tool
                    if tool_name in TOOL_FUNCTIONS:
                        result = TOOL_FUNCTIONS[tool_name](**tool_input)
                    else:
                        result = {"error": f"Unknown tool: {tool_name}"}

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": json.dumps(result)
                    })

            # Add tool results
            messages.append({"role": "user", "content": tool_results})

        else:
            # Unexpected stop reason
            return f"Unexpected stop: {response.stop_reason}"


# Example with streaming
def stream_with_tools_anthropic(user_message: str) -> str:
    """Stream responses while handling tool use."""
    messages = [{"role": "user", "content": user_message}]
    full_response = ""

    while True:
        collected_content = []
        current_tool_use = None

        with client.messages.stream(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            tools=tools,
            messages=messages
        ) as stream:
            for event in stream:
                if hasattr(event, 'type'):
                    if event.type == 'content_block_start':
                        if event.content_block.type == 'text':
                            pass  # Text block starting
                        elif event.content_block.type == 'tool_use':
                            current_tool_use = {
                                'id': event.content_block.id,
                                'name': event.content_block.name,
                                'input': ''
                            }

                    elif event.type == 'content_block_delta':
                        if hasattr(event.delta, 'text'):
                            print(event.delta.text, end='', flush=True)
                            full_response += event.delta.text
                        elif hasattr(event.delta, 'partial_json'):
                            if current_tool_use:
                                current_tool_use['input'] += event.delta.partial_json

                    elif event.type == 'content_block_stop':
                        if current_tool_use:
                            collected_content.append(current_tool_use)
                            current_tool_use = None

        # Check if we need to execute tools
        final_message = stream.get_final_message()

        if final_message.stop_reason == "end_turn":
            return full_response

        elif final_message.stop_reason == "tool_use":
            # Add assistant message
            messages.append({
                "role": "assistant",
                "content": final_message.content
            })

            # Execute tools and add results
            tool_results = []
            for block in final_message.content:
                if block.type == "tool_use":
                    if block.name in TOOL_FUNCTIONS:
                        result = TOOL_FUNCTIONS[block.name](**block.input)
                    else:
                        result = {"error": f"Unknown tool: {block.name}"}

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result)
                    })

            messages.append({"role": "user", "content": tool_results})
```

### Google Gemini Function Calling

```python
"""
Google Gemini Function Calling
"""
import google.generativeai as genai
from google.protobuf.struct_pb2 import Struct
import json

genai.configure(api_key="your-api-key")

# Define function declarations
get_weather_func = genai.protos.FunctionDeclaration(
    name="get_weather",
    description="Get current weather for a location",
    parameters=genai.protos.Schema(
        type=genai.protos.Type.OBJECT,
        properties={
            "location": genai.protos.Schema(
                type=genai.protos.Type.STRING,
                description="City name"
            ),
            "unit": genai.protos.Schema(
                type=genai.protos.Type.STRING,
                enum=["celsius", "fahrenheit"]
            )
        },
        required=["location"]
    )
)

book_flight_func = genai.protos.FunctionDeclaration(
    name="book_flight",
    description="Book a flight between two cities",
    parameters=genai.protos.Schema(
        type=genai.protos.Type.OBJECT,
        properties={
            "origin": genai.protos.Schema(
                type=genai.protos.Type.STRING,
                description="Departure city"
            ),
            "destination": genai.protos.Schema(
                type=genai.protos.Type.STRING,
                description="Arrival city"
            ),
            "date": genai.protos.Schema(
                type=genai.protos.Type.STRING,
                description="Travel date (YYYY-MM-DD)"
            ),
            "passengers": genai.protos.Schema(
                type=genai.protos.Type.INTEGER,
                description="Number of passengers"
            )
        },
        required=["origin", "destination", "date"]
    )
)

# Create tool with functions
tools = genai.protos.Tool(
    function_declarations=[get_weather_func, book_flight_func]
)

# Tool implementations
def get_weather(location: str, unit: str = "celsius") -> dict:
    return {
        "location": location,
        "temperature": 22 if unit == "celsius" else 72,
        "unit": unit,
        "conditions": "Sunny"
    }

def book_flight(origin: str, destination: str, date: str, passengers: int = 1) -> dict:
    return {
        "confirmation": "FL12345",
        "origin": origin,
        "destination": destination,
        "date": date,
        "passengers": passengers,
        "status": "confirmed"
    }

FUNCTIONS = {
    "get_weather": get_weather,
    "book_flight": book_flight
}


def chat_with_gemini_tools(prompt: str):
    """Use Gemini with function calling."""
    model = genai.GenerativeModel(
        model_name="gemini-1.5-pro",
        tools=[tools]
    )

    chat = model.start_chat()
    response = chat.send_message(prompt)

    # Check for function calls
    while response.candidates[0].content.parts:
        function_calls = [
            part.function_call
            for part in response.candidates[0].content.parts
            if part.function_call.name
        ]

        if not function_calls:
            # No function calls, return text
            return response.text

        # Execute functions
        function_responses = []
        for fc in function_calls:
            func_name = fc.name
            func_args = dict(fc.args)

            print(f"Calling: {func_name}({func_args})")

            result = FUNCTIONS[func_name](**func_args)

            function_responses.append(
                genai.protos.Part(
                    function_response=genai.protos.FunctionResponse(
                        name=func_name,
                        response={"result": result}
                    )
                )
            )

        # Send function results back
        response = chat.send_message(function_responses)

    return response.text
```

---

## Building AI Agents

### What is an Agent?

An agent is an AI system that can:
1. **Perceive** its environment (through tools and their outputs)
2. **Reason** about what to do next (via LLM inference)
3. **Act** by executing tools
4. **Observe** results and incorporate them into the next reasoning step

Note: agents do not "learn" in the ML sense — they do not update weights or permanently improve. They accumulate observations within a session and use conversation history as working memory. Each new session starts fresh unless you implement explicit persistent memory.

```python
"""
Core Agent Architecture
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable
from enum import Enum
import json

class AgentState(Enum):
    """Agent execution states."""
    IDLE = "idle"
    THINKING = "thinking"
    ACTING = "acting"
    OBSERVING = "observing"
    FINISHED = "finished"
    ERROR = "error"


@dataclass
class AgentAction:
    """Represents an action the agent wants to take."""
    tool_name: str
    tool_input: dict
    reasoning: str = ""


@dataclass
class AgentObservation:
    """Result of an action."""
    action: AgentAction
    result: Any
    success: bool
    error: str = ""


@dataclass
class AgentStep:
    """A single step in agent execution."""
    thought: str
    action: AgentAction | None
    observation: AgentObservation | None


@dataclass
class AgentMemory:
    """Agent's working memory."""
    steps: list[AgentStep] = field(default_factory=list)
    context: dict = field(default_factory=dict)

    def add_step(self, step: AgentStep):
        self.steps.append(step)

    def get_history(self) -> str:
        """Format history for the LLM."""
        history = []
        for i, step in enumerate(self.steps):
            history.append(f"Step {i+1}:")
            history.append(f"Thought: {step.thought}")
            if step.action:
                history.append(f"Action: {step.action.tool_name}({step.action.tool_input})")
            if step.observation:
                history.append(f"Observation: {step.observation.result}")
        return "\n".join(history)


class BaseAgent(ABC):
    """Base class for all agents."""

    def __init__(
        self,
        tools: list[Tool],
        max_iterations: int = 10,
        verbose: bool = True
    ):
        self.tools = {tool.name: tool for tool in tools}
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.memory = AgentMemory()
        self.state = AgentState.IDLE

    @abstractmethod
    def think(self, objective: str) -> tuple[str, AgentAction | None]:
        """
        Decide what to do next.
        Returns (thought, action) where action is None if done.
        """
        pass

    def act(self, action: AgentAction) -> AgentObservation:
        """Execute an action."""
        tool = self.tools.get(action.tool_name)

        if not tool:
            return AgentObservation(
                action=action,
                result=None,
                success=False,
                error=f"Unknown tool: {action.tool_name}"
            )

        try:
            result = tool.execute(**action.tool_input)
            return AgentObservation(
                action=action,
                result=result,
                success=True
            )
        except Exception as e:
            return AgentObservation(
                action=action,
                result=None,
                success=False,
                error=str(e)
            )

    def run(self, objective: str) -> str:
        """
        Execute the agent loop until completion or max iterations.
        """
        self.state = AgentState.THINKING
        self.memory = AgentMemory()  # Reset memory

        for iteration in range(self.max_iterations):
            if self.verbose:
                print(f"\n--- Iteration {iteration + 1} ---")

            # Think
            self.state = AgentState.THINKING
            thought, action = self.think(objective)

            if self.verbose:
                print(f"Thought: {thought}")

            # Check if done
            if action is None:
                self.state = AgentState.FINISHED
                if self.verbose:
                    print("Agent finished!")
                return thought

            if self.verbose:
                print(f"Action: {action.tool_name}({action.tool_input})")

            # Act
            self.state = AgentState.ACTING
            observation = self.act(action)

            # Observe
            self.state = AgentState.OBSERVING
            if self.verbose:
                print(f"Observation: {observation.result}")

            # Record step
            step = AgentStep(
                thought=thought,
                action=action,
                observation=observation
            )
            self.memory.add_step(step)

        self.state = AgentState.ERROR
        return "Max iterations reached without completing objective"
```

### ReAct Agent Implementation

ReAct (Reasoning + Acting) is a powerful pattern where the agent explicitly reasons before each action:

```python
"""
ReAct Agent Implementation
"""
from openai import OpenAI
import json
import re

class ReActAgent(BaseAgent):
    """
    ReAct agent that interleaves reasoning and acting.

    Pattern:
    1. Thought: Reason about current state
    2. Action: Choose and execute a tool
    3. Observation: Process result
    4. Repeat until done
    """

    def __init__(
        self,
        tools: list[Tool],
        model: str = "gpt-4o",
        **kwargs
    ):
        super().__init__(tools, **kwargs)
        self.client = OpenAI()
        self.model = model

    def _build_system_prompt(self) -> str:
        """Build the system prompt with tool descriptions."""
        tool_descriptions = "\n".join([
            f"- {name}: {tool.description}"
            for name, tool in self.tools.items()
        ])

        return f"""You are a helpful AI assistant that can use tools to accomplish tasks.

Available tools:
{tool_descriptions}

You operate in a loop of Thought, Action, and Observation.

IMPORTANT RULES:
1. Always start with a Thought explaining your reasoning
2. Use exactly ONE action per turn
3. Wait for the Observation before continuing
4. When you have enough information to answer, use Action: finish with your final answer

Response format:
Thought: [Your reasoning about what to do next]
Action: [tool_name]
Action Input: [JSON object with parameters]

OR when done:
Thought: [Your final reasoning]
Action: finish
Action Input: {{"answer": "Your final answer here"}}

Always be precise and complete in your answers."""

    def _build_user_prompt(self, objective: str) -> str:
        """Build the user prompt with objective and history."""
        history = self.memory.get_history()

        if history:
            return f"""Objective: {objective}

Previous steps:
{history}

Continue from where you left off."""
        else:
            return f"""Objective: {objective}

Begin working on this objective."""

    def _parse_response(self, response: str) -> tuple[str, AgentAction | None]:
        """Parse LLM response into thought and action."""
        # Extract thought
        thought_match = re.search(r'Thought:\s*(.+?)(?=Action:|$)', response, re.DOTALL)
        thought = thought_match.group(1).strip() if thought_match else ""

        # Extract action
        action_match = re.search(r'Action:\s*(\w+)', response)
        if not action_match:
            return thought, None

        action_name = action_match.group(1).strip()

        # Check if finished
        if action_name.lower() == "finish":
            input_match = re.search(r'Action Input:\s*({.+})', response, re.DOTALL)
            if input_match:
                try:
                    data = json.loads(input_match.group(1))
                    return data.get("answer", thought), None
                except json.JSONDecodeError:
                    return thought, None
            return thought, None

        # Extract action input
        input_match = re.search(r'Action Input:\s*({.+})', response, re.DOTALL)
        if input_match:
            try:
                action_input = json.loads(input_match.group(1))
            except json.JSONDecodeError:
                action_input = {}
        else:
            action_input = {}

        return thought, AgentAction(
            tool_name=action_name,
            tool_input=action_input,
            reasoning=thought
        )

    def think(self, objective: str) -> tuple[str, AgentAction | None]:
        """Generate next thought and action."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self._build_system_prompt()},
                {"role": "user", "content": self._build_user_prompt(objective)}
            ],
            temperature=0
        )

        content = response.choices[0].message.content
        return self._parse_response(content)


# Example usage
def create_research_agent() -> ReActAgent:
    """Create an agent for research tasks."""

    # Define tools
    def web_search(query: str, num_results: int = 3) -> list[dict]:
        """Simulate web search."""
        return [
            {"title": f"Result {i+1}", "snippet": f"Information about {query}..."}
            for i in range(num_results)
        ]

    def wikipedia_lookup(topic: str) -> dict:
        """Simulate Wikipedia lookup."""
        return {
            "title": topic,
            "summary": f"Wikipedia article about {topic}...",
            "url": f"https://en.wikipedia.org/wiki/{topic.replace(' ', '_')}"
        }

    def calculator(expression: str) -> float:
        """Evaluate mathematical expression safely using AST parsing."""
        import ast
        import operator

        # Safe operator mapping -- no eval() needed
        ops = {
            ast.Add: operator.add, ast.Sub: operator.sub,
            ast.Mult: operator.mul, ast.Div: operator.truediv,
            ast.Pow: operator.pow, ast.USub: operator.neg,
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
            raise ValueError(f"Unsupported expression element: {type(node).__name__}")

        try:
            tree = ast.parse(expression, mode='eval')
            return _safe_eval(tree)
        except (SyntaxError, ValueError) as e:
            raise ValueError(f"Invalid expression: {e}")

    tools = [
        Tool(
            name="web_search",
            description="Search the web for information. Use for current events, facts, or general knowledge.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "num_results": {"type": "integer", "default": 3}
                },
                "required": ["query"]
            },
            function=web_search
        ),
        Tool(
            name="wikipedia",
            description="Look up a topic on Wikipedia for detailed background information.",
            parameters={
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "Topic to look up"}
                },
                "required": ["topic"]
            },
            function=wikipedia_lookup
        ),
        Tool(
            name="calculator",
            description="Calculate mathematical expressions. Use for any math operations.",
            parameters={
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Math expression to evaluate"}
                },
                "required": ["expression"]
            },
            function=calculator
        )
    ]

    return ReActAgent(tools=tools, verbose=True)


# Run the agent
if __name__ == "__main__":
    agent = create_research_agent()
    result = agent.run(
        "What is the population of Tokyo? "
        "Calculate what percentage of Japan's population lives in Tokyo."
    )
    print(f"\nFinal Answer: {result}")
```

### Choosing Between Agent Architectures: ReAct vs. Plan-and-Execute

Before implementing Plan-and-Execute, understand when each architecture wins:

| Dimension | ReAct | Plan-and-Execute |
|-----------|-------|-----------------|
| **Best for** | Exploratory tasks where next step depends on previous result | Well-defined tasks with predictable steps |
| **Latency** | Lower per-step (no planning overhead) | Higher initial latency (planning step), but can be faster overall for complex tasks |
| **Cost** | Lower for simple tasks (2-5 steps) | Higher fixed cost (planning LLM call), but more token-efficient for complex tasks because executor prompts are focused |
| **Reliability** | Can get stuck in loops; each step risks going off-track | Plan provides guardrails; replanning corrects course |
| **Transparency** | Each thought is visible | Plan is visible upfront; individual step reasoning may be less detailed |
| **Error recovery** | Must reason about error in next thought | Can replan the remaining steps after failure |
| **Failure mode** | Circular reasoning, repeated actions | Bad initial plan leads to wasted steps; replanning adds cost |

**Rule of thumb:** Use ReAct for tasks with fewer than 5 steps or where the path is unpredictable. Use Plan-and-Execute for tasks with 5+ steps, clear subtask decomposition, or when you need auditability (the plan serves as a contract for what the agent will do).

**When NOT to use agents at all:**
- **Single tool call suffices.** If the user asks "what's the weather?" and you have a weather tool, just call it. An agent loop adds latency and cost for no benefit.
- **High-reliability requirements.** Each agent step has ~90-95% success probability. Over 10 steps, that compounds to 35-60% end-to-end. If you need >99% reliability, use deterministic workflows with hardcoded tool sequences.
- **Latency budget under 2 seconds.** A single agent iteration takes 1-3 seconds (LLM inference + tool execution). Multi-step agents routinely take 10-30 seconds.
- **Cost sensitivity.** A 5-step agent loop costs 5-10x a single LLM call. At scale, this adds up fast.

### Plan-and-Execute Agent

This architecture separates planning from execution:

```python
"""
Plan-and-Execute Agent
Separates high-level planning from step-by-step execution.
"""
from openai import OpenAI
from dataclasses import dataclass
import json

@dataclass
class Plan:
    """A plan consisting of multiple steps."""
    objective: str
    steps: list[str]
    current_step: int = 0
    completed_steps: list[dict] = None

    def __post_init__(self):
        if self.completed_steps is None:
            self.completed_steps = []

    @property
    def is_complete(self) -> bool:
        return self.current_step >= len(self.steps)

    @property
    def current_task(self) -> str | None:
        if self.is_complete:
            return None
        return self.steps[self.current_step]

    def complete_current(self, result: str):
        self.completed_steps.append({
            "step": self.steps[self.current_step],
            "result": result
        })
        self.current_step += 1


class PlanAndExecuteAgent:
    """
    Two-phase agent:
    1. Planner: Creates a high-level plan
    2. Executor: Executes each step of the plan
    """

    def __init__(
        self,
        tools: list[Tool],
        planner_model: str = "gpt-4o",
        executor_model: str = "gpt-4o"
    ):
        self.tools = {tool.name: tool for tool in tools}
        self.client = OpenAI()
        self.planner_model = planner_model
        self.executor_model = executor_model

    def create_plan(self, objective: str) -> Plan:
        """Create a plan for achieving the objective."""
        tool_list = "\n".join([
            f"- {name}: {tool.description}"
            for name, tool in self.tools.items()
        ])

        response = self.client.chat.completions.create(
            model=self.planner_model,
            messages=[
                {
                    "role": "system",
                    "content": f"""You are a planning assistant. Create a step-by-step plan to achieve objectives.

Available tools:
{tool_list}

Output your plan as a JSON array of steps. Each step should be a clear, actionable task.
Keep plans concise (3-7 steps typically).
Consider dependencies between steps.

Example output:
["Search for information about X", "Calculate Y based on search results", "Summarize findings"]"""
                },
                {
                    "role": "user",
                    "content": f"Create a plan for: {objective}"
                }
            ],
            response_format={"type": "json_object"},
            temperature=0
        )

        content = json.loads(response.choices[0].message.content)
        steps = content.get("steps", content.get("plan", []))

        return Plan(objective=objective, steps=steps)

    def execute_step(self, plan: Plan) -> str:
        """Execute the current step of the plan."""
        tool_list = "\n".join([
            f"- {name}: {tool.description}"
            for name, tool in self.tools.items()
        ])

        # Build context from completed steps
        context = ""
        if plan.completed_steps:
            context = "Previous steps and results:\n"
            for completed in plan.completed_steps:
                context += f"- {completed['step']}: {completed['result']}\n"

        response = self.client.chat.completions.create(
            model=self.executor_model,
            messages=[
                {
                    "role": "system",
                    "content": f"""You are an execution assistant. Complete the given task using available tools.

Available tools:
{tool_list}

Respond with a JSON object:
{{"tool": "tool_name", "input": {{...tool parameters...}}}}

Or if no tool is needed:
{{"result": "your direct answer"}}"""
                },
                {
                    "role": "user",
                    "content": f"""Overall objective: {plan.objective}

{context}

Current task: {plan.current_task}

Execute this task."""
                }
            ],
            response_format={"type": "json_object"},
            temperature=0
        )

        action = json.loads(response.choices[0].message.content)

        if "result" in action:
            return action["result"]

        tool_name = action.get("tool")
        tool_input = action.get("input", {})

        if tool_name in self.tools:
            result = self.tools[tool_name].execute(**tool_input)
            return json.dumps(result)

        return f"Unknown tool: {tool_name}"

    def maybe_replan(self, plan: Plan) -> Plan:
        """Optionally replan based on execution results."""
        if plan.is_complete:
            return plan

        # Check if we should replan
        response = self.client.chat.completions.create(
            model=self.planner_model,
            messages=[
                {
                    "role": "system",
                    "content": """Analyze if the current plan needs adjustment based on results so far.

Respond with JSON:
{"replan": false} - if the current plan is still good
{"replan": true, "new_steps": ["step1", "step2", ...]} - if plan needs changes"""
                },
                {
                    "role": "user",
                    "content": f"""Objective: {plan.objective}

Completed steps:
{json.dumps(plan.completed_steps, indent=2)}

Remaining steps:
{json.dumps(plan.steps[plan.current_step:], indent=2)}

Should we adjust the plan?"""
                }
            ],
            response_format={"type": "json_object"},
            temperature=0
        )

        decision = json.loads(response.choices[0].message.content)

        if decision.get("replan"):
            new_steps = plan.steps[:plan.current_step] + decision.get("new_steps", [])
            return Plan(
                objective=plan.objective,
                steps=new_steps,
                current_step=plan.current_step,
                completed_steps=plan.completed_steps
            )

        return plan

    def run(self, objective: str, verbose: bool = True) -> str:
        """Execute the full plan-and-execute loop."""
        # Phase 1: Planning
        if verbose:
            print("=== Planning Phase ===")

        plan = self.create_plan(objective)

        if verbose:
            print(f"Created plan with {len(plan.steps)} steps:")
            for i, step in enumerate(plan.steps):
                print(f"  {i+1}. {step}")

        # Phase 2: Execution
        if verbose:
            print("\n=== Execution Phase ===")

        while not plan.is_complete:
            if verbose:
                print(f"\nExecuting step {plan.current_step + 1}: {plan.current_task}")

            result = self.execute_step(plan)

            if verbose:
                print(f"Result: {result[:200]}...")

            plan.complete_current(result)

            # Optional: replan after each step
            plan = self.maybe_replan(plan)

        # Generate final answer
        response = self.client.chat.completions.create(
            model=self.planner_model,
            messages=[
                {
                    "role": "system",
                    "content": "Synthesize the results of the completed plan into a final answer."
                },
                {
                    "role": "user",
                    "content": f"""Objective: {plan.objective}

Completed steps and results:
{json.dumps(plan.completed_steps, indent=2)}

Provide a comprehensive final answer."""
                }
            ]
        )

        return response.choices[0].message.content
```

---

## Advanced Agent Patterns

### Multi-Agent Systems

```python
"""
Multi-Agent System with Specialized Agents
"""
from typing import Protocol
from dataclasses import dataclass
import json

class AgentProtocol(Protocol):
    """Protocol for agent communication."""

    def process(self, message: str, context: dict) -> str:
        """Process a message and return response."""
        ...


@dataclass
class AgentMessage:
    """Message between agents."""
    sender: str
    recipient: str
    content: str
    message_type: str = "request"  # request, response, broadcast


class AgentOrchestrator:
    """
    Coordinates multiple specialized agents.
    Routes tasks to appropriate agents and synthesizes results.
    """

    def __init__(self, model: str = "gpt-4o"):
        self.agents: dict[str, AgentProtocol] = {}
        self.client = OpenAI()
        self.model = model
        self.message_history: list[AgentMessage] = []

    def register_agent(self, name: str, agent: AgentProtocol, description: str):
        """Register a specialized agent."""
        self.agents[name] = {
            "agent": agent,
            "description": description
        }

    def route_task(self, task: str) -> str:
        """Determine which agent(s) should handle a task."""
        agent_descriptions = "\n".join([
            f"- {name}: {info['description']}"
            for name, info in self.agents.items()
        ])

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": f"""You are a task router. Determine which agent(s) should handle the task.

Available agents:
{agent_descriptions}

Respond with JSON:
{{"agents": ["agent1", "agent2"], "subtasks": {{"agent1": "specific task", "agent2": "specific task"}}}}

Use multiple agents if the task requires different specializations."""
                },
                {"role": "user", "content": f"Route this task: {task}"}
            ],
            response_format={"type": "json_object"}
        )

        return json.loads(response.choices[0].message.content)

    def execute_task(self, task: str, context: dict = None) -> str:
        """Execute a task using the multi-agent system."""
        if context is None:
            context = {}

        # Route the task
        routing = self.route_task(task)
        agents_to_use = routing.get("agents", [])
        subtasks = routing.get("subtasks", {})

        # Execute with each agent
        results = {}
        for agent_name in agents_to_use:
            if agent_name not in self.agents:
                continue

            subtask = subtasks.get(agent_name, task)
            agent = self.agents[agent_name]["agent"]

            # Create message
            message = AgentMessage(
                sender="orchestrator",
                recipient=agent_name,
                content=subtask
            )
            self.message_history.append(message)

            # Get response
            result = agent.process(subtask, {**context, "results_so_far": results})
            results[agent_name] = result

            # Record response
            response_message = AgentMessage(
                sender=agent_name,
                recipient="orchestrator",
                content=result,
                message_type="response"
            )
            self.message_history.append(response_message)

        # Synthesize results
        return self.synthesize_results(task, results)

    def synthesize_results(self, original_task: str, results: dict) -> str:
        """Combine results from multiple agents."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": """Synthesize the results from multiple agents into a coherent response.
Combine insights, resolve any conflicts, and provide a unified answer."""
                },
                {
                    "role": "user",
                    "content": f"""Original task: {original_task}

Results from agents:
{json.dumps(results, indent=2)}

Provide a synthesized response."""
                }
            ]
        )

        return response.choices[0].message.content


# Specialized agents
class ResearchAgent:
    """Agent specialized for research and information gathering."""

    def __init__(self):
        self.client = OpenAI()

    def process(self, message: str, context: dict) -> str:
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are a research specialist. Gather and analyze information thoroughly."
                },
                {"role": "user", "content": message}
            ]
        )
        return response.choices[0].message.content


class AnalysisAgent:
    """Agent specialized for data analysis and interpretation."""

    def __init__(self):
        self.client = OpenAI()

    def process(self, message: str, context: dict) -> str:
        # Include previous results if available
        context_str = ""
        if "results_so_far" in context:
            context_str = f"Context from other agents: {json.dumps(context['results_so_far'])}"

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": f"You are an analysis specialist. Analyze data and provide insights. {context_str}"
                },
                {"role": "user", "content": message}
            ]
        )
        return response.choices[0].message.content


class WritingAgent:
    """Agent specialized for content creation."""

    def __init__(self):
        self.client = OpenAI()

    def process(self, message: str, context: dict) -> str:
        context_str = ""
        if "results_so_far" in context:
            context_str = f"Use this research: {json.dumps(context['results_so_far'])}"

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": f"You are a writing specialist. Create clear, engaging content. {context_str}"
                },
                {"role": "user", "content": message}
            ]
        )
        return response.choices[0].message.content


# Usage example
def demo_multi_agent():
    orchestrator = AgentOrchestrator()

    orchestrator.register_agent(
        "researcher",
        ResearchAgent(),
        "Gathers information, finds facts, and does research"
    )
    orchestrator.register_agent(
        "analyst",
        AnalysisAgent(),
        "Analyzes data, identifies patterns, and provides insights"
    )
    orchestrator.register_agent(
        "writer",
        WritingAgent(),
        "Creates written content, reports, and summaries"
    )

    result = orchestrator.execute_task(
        "Research the current state of electric vehicle adoption, "
        "analyze the trends, and write a brief report"
    )

    return result
```

### Tool-Using Agent with Error Recovery

```python
"""
Robust Agent with Error Handling and Recovery
"""
from tenacity import retry, stop_after_attempt, wait_exponential
from typing import Callable
import traceback
import json

class RobustAgent:
    """
    Agent with comprehensive error handling, retries, and recovery strategies.
    """

    def __init__(
        self,
        tools: list[Tool],
        model: str = "gpt-4o",
        max_retries: int = 3,
        max_iterations: int = 15
    ):
        self.tools = {tool.name: tool for tool in tools}
        self.client = OpenAI()
        self.model = model
        self.max_retries = max_retries
        self.max_iterations = max_iterations
        self.error_history: list[dict] = []

    def _handle_tool_error(
        self,
        tool_name: str,
        error: Exception,
        attempt: int
    ) -> dict:
        """Handle and log tool errors."""
        error_info = {
            "tool": tool_name,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
            "attempt": attempt
        }
        self.error_history.append(error_info)
        return error_info

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10)
    )
    def _call_llm(self, messages: list) -> str:
        """Call LLM with retry logic."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0
        )
        return response.choices[0].message.content

    def _execute_tool_with_recovery(
        self,
        tool_name: str,
        tool_input: dict
    ) -> tuple[bool, Any]:
        """Execute tool with retry and recovery logic."""
        tool = self.tools.get(tool_name)

        if not tool:
            return False, f"Tool '{tool_name}' not found"

        last_error = None

        for attempt in range(1, self.max_retries + 1):
            try:
                result = tool.execute(**tool_input)
                return True, result

            except Exception as e:
                last_error = e
                error_info = self._handle_tool_error(tool_name, e, attempt)

                if attempt < self.max_retries:
                    # Try to fix the input
                    fixed_input = self._attempt_fix_input(
                        tool_name,
                        tool_input,
                        error_info
                    )
                    if fixed_input:
                        tool_input = fixed_input
                        continue

        return False, f"Tool failed after {self.max_retries} attempts: {last_error}"

    def _attempt_fix_input(
        self,
        tool_name: str,
        original_input: dict,
        error_info: dict
    ) -> dict | None:
        """Ask LLM to fix problematic input."""
        tool = self.tools[tool_name]

        try:
            response = self._call_llm([
                {
                    "role": "system",
                    "content": f"""You are debugging a tool call. The tool failed with an error.

Tool: {tool_name}
Tool schema: {json.dumps(tool.parameters)}
Error: {error_info['error_type']}: {error_info['error_message']}

Analyze the error and provide a corrected input JSON that might work.
Respond with ONLY valid JSON, no explanation."""
                },
                {
                    "role": "user",
                    "content": f"Original input that failed: {json.dumps(original_input)}"
                }
            ])

            return json.loads(response)

        except Exception:
            return None

    def _get_fallback_action(
        self,
        objective: str,
        failed_action: str,
        error: str
    ) -> tuple[str, dict] | None:
        """Get alternative action when primary fails."""
        tool_descriptions = "\n".join([
            f"- {name}: {tool.description}"
            for name, tool in self.tools.items()
        ])

        try:
            response = self._call_llm([
                {
                    "role": "system",
                    "content": f"""A tool action failed. Suggest an alternative approach.

Available tools:
{tool_descriptions}

Respond with JSON:
{{"alternative_tool": "tool_name", "input": {{...}}, "reasoning": "why this might work"}}

Or if no alternative:
{{"give_up": true, "explanation": "why we cannot continue"}}"""
                },
                {
                    "role": "user",
                    "content": f"""Objective: {objective}
Failed action: {failed_action}
Error: {error}

What's an alternative approach?"""
                }
            ])

            result = json.loads(response)

            if result.get("give_up"):
                return None

            return result.get("alternative_tool"), result.get("input", {})

        except Exception:
            return None

    def run(self, objective: str, verbose: bool = True) -> str:
        """Run the agent with full error handling."""
        messages = [
            {
                "role": "system",
                "content": self._build_system_prompt()
            }
        ]

        iteration = 0
        consecutive_errors = 0
        max_consecutive_errors = 3

        while iteration < self.max_iterations:
            iteration += 1

            if verbose:
                print(f"\n--- Iteration {iteration} ---")

            # Get next action
            messages.append({
                "role": "user",
                "content": f"Objective: {objective}\n\nWhat's your next step?"
            })

            try:
                response = self._call_llm(messages)
            except Exception as e:
                if verbose:
                    print(f"LLM call failed: {e}")
                consecutive_errors += 1
                if consecutive_errors >= max_consecutive_errors:
                    return f"Failed due to repeated LLM errors: {e}"
                continue

            messages.append({"role": "assistant", "content": response})

            # Parse response
            thought, action = self._parse_response(response)

            if verbose:
                print(f"Thought: {thought}")

            # Check if done
            if action is None:
                return thought

            tool_name, tool_input = action

            if verbose:
                print(f"Action: {tool_name}({tool_input})")

            # Execute with error handling
            success, result = self._execute_tool_with_recovery(tool_name, tool_input)

            if success:
                consecutive_errors = 0
                observation = f"Success: {json.dumps(result)}"
            else:
                if verbose:
                    print(f"Tool failed: {result}")

                # Try fallback
                fallback = self._get_fallback_action(objective, f"{tool_name}({tool_input})", str(result))

                if fallback:
                    alt_tool, alt_input = fallback
                    if verbose:
                        print(f"Trying fallback: {alt_tool}({alt_input})")

                    success, result = self._execute_tool_with_recovery(alt_tool, alt_input)

                    if success:
                        observation = f"Fallback succeeded: {json.dumps(result)}"
                    else:
                        observation = f"Both primary and fallback failed. Error: {result}"
                        consecutive_errors += 1
                else:
                    observation = f"Error (no fallback available): {result}"
                    consecutive_errors += 1

            if verbose:
                print(f"Observation: {observation[:200]}...")

            messages.append({
                "role": "user",
                "content": f"Observation: {observation}"
            })

            if consecutive_errors >= max_consecutive_errors:
                return f"Stopping due to {consecutive_errors} consecutive errors"

        return "Max iterations reached"

    def _build_system_prompt(self) -> str:
        tool_descriptions = "\n".join([
            f"- {name}: {tool.description}"
            for name, tool in self.tools.items()
        ])

        return f"""You are a helpful AI assistant with access to tools.

Available tools:
{tool_descriptions}

Response format:
Thought: [Your reasoning]
Action: [tool_name]
Action Input: [JSON parameters]

When finished:
Thought: [Final reasoning]
Action: finish
Action Input: {{"answer": "your answer"}}

If a tool fails, analyze the error and try a different approach."""

    def _parse_response(self, response: str) -> tuple[str, tuple[str, dict] | None]:
        """Parse LLM response."""
        import re

        thought_match = re.search(r'Thought:\s*(.+?)(?=Action:|$)', response, re.DOTALL)
        thought = thought_match.group(1).strip() if thought_match else ""

        action_match = re.search(r'Action:\s*(\w+)', response)
        if not action_match:
            return thought, None

        action_name = action_match.group(1).strip()

        if action_name.lower() == "finish":
            input_match = re.search(r'Action Input:\s*({.+})', response, re.DOTALL)
            if input_match:
                try:
                    data = json.loads(input_match.group(1))
                    return data.get("answer", thought), None
                except json.JSONDecodeError:
                    pass
            return thought, None

        input_match = re.search(r'Action Input:\s*({.+})', response, re.DOTALL)
        if input_match:
            try:
                action_input = json.loads(input_match.group(1))
            except json.JSONDecodeError:
                action_input = {}
        else:
            action_input = {}

        return thought, (action_name, action_input)
```

---

## Building a Complete Agent Application

Let's build a full-featured personal assistant agent:

```python
"""
Complete Personal Assistant Agent
A production-ready agent with multiple capabilities.
"""
import os
from datetime import datetime, timedelta
from openai import OpenAI
from dataclasses import dataclass, field
from typing import Any, Callable
import json
import re

# ============ Tool Definitions ============

@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    function: Callable
    requires_confirmation: bool = False  # Safety flag

    def to_openai_format(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }


# ============ Tool Implementations ============

class ToolImplementations:
    """Implementations of all available tools."""

    @staticmethod
    def get_current_time(timezone: str = "UTC") -> dict:
        """Get current date and time."""
        from zoneinfo import ZoneInfo
        try:
            tz = ZoneInfo(timezone)
            now = datetime.now(tz)
            return {
                "datetime": now.isoformat(),
                "date": now.strftime("%Y-%m-%d"),
                "time": now.strftime("%H:%M:%S"),
                "timezone": timezone,
                "day_of_week": now.strftime("%A")
            }
        except Exception as e:
            return {"error": str(e)}

    @staticmethod
    def search_web(query: str, num_results: int = 5) -> dict:
        """Search the web (simulated)."""
        # In production, integrate with a real search API
        return {
            "query": query,
            "results": [
                {
                    "title": f"Result {i+1} for '{query}'",
                    "url": f"https://example.com/result{i}",
                    "snippet": f"This is relevant information about {query}..."
                }
                for i in range(num_results)
            ]
        }

    @staticmethod
    def send_email(
        to: str,
        subject: str,
        body: str,
        cc: list = None
    ) -> dict:
        """Send an email (simulated)."""
        # In production, integrate with email service
        return {
            "status": "sent",
            "message_id": f"msg_{datetime.now().timestamp()}",
            "to": to,
            "subject": subject,
            "timestamp": datetime.now().isoformat()
        }

    @staticmethod
    def create_reminder(
        title: str,
        due_datetime: str,
        description: str = "",
        priority: str = "normal"
    ) -> dict:
        """Create a reminder."""
        return {
            "id": f"reminder_{datetime.now().timestamp()}",
            "title": title,
            "due": due_datetime,
            "description": description,
            "priority": priority,
            "status": "created"
        }

    @staticmethod
    def calculate(expression: str) -> dict:
        """Safely evaluate math expressions."""
        import ast
        import operator

        # Safe operators
        operators = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Pow: operator.pow,
            ast.USub: operator.neg,
        }

        def eval_expr(node):
            if isinstance(node, ast.Num):
                return node.n
            elif isinstance(node, ast.BinOp):
                return operators[type(node.op)](
                    eval_expr(node.left),
                    eval_expr(node.right)
                )
            elif isinstance(node, ast.UnaryOp):
                return operators[type(node.op)](eval_expr(node.operand))
            else:
                raise ValueError(f"Unsupported operation: {type(node)}")

        try:
            tree = ast.parse(expression, mode='eval')
            result = eval_expr(tree.body)
            return {"expression": expression, "result": result}
        except Exception as e:
            return {"error": str(e)}

    @staticmethod
    def get_weather(location: str, units: str = "celsius") -> dict:
        """Get weather information (simulated)."""
        # In production, call real weather API
        return {
            "location": location,
            "temperature": 22 if units == "celsius" else 72,
            "units": units,
            "conditions": "Partly Cloudy",
            "humidity": 65,
            "wind_speed": 12,
            "forecast": "Clear skies expected later"
        }

    @staticmethod
    def manage_tasks(
        action: str,
        task_id: str = None,
        title: str = None,
        description: str = None,
        due_date: str = None,
        priority: str = "normal"
    ) -> dict:
        """Manage tasks (create, update, delete, list)."""
        # In production, integrate with task management system
        if action == "create":
            return {
                "id": f"task_{datetime.now().timestamp()}",
                "title": title,
                "description": description,
                "due_date": due_date,
                "priority": priority,
                "status": "pending",
                "created": datetime.now().isoformat()
            }
        elif action == "list":
            return {
                "tasks": [
                    {"id": "task_1", "title": "Review document", "status": "pending"},
                    {"id": "task_2", "title": "Call client", "status": "completed"},
                ]
            }
        elif action == "update":
            return {"id": task_id, "status": "updated"}
        elif action == "delete":
            return {"id": task_id, "status": "deleted"}
        return {"error": "Unknown action"}


# ============ Tool Registry ============

def create_tool_registry() -> list[Tool]:
    """Create all available tools."""
    return [
        Tool(
            name="get_current_time",
            description="Get the current date and time. Useful for scheduling and time-based queries.",
            parameters={
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": "IANA timezone (e.g., 'America/New_York', 'Europe/London')",
                        "default": "UTC"
                    }
                }
            },
            function=ToolImplementations.get_current_time
        ),
        Tool(
            name="search_web",
            description="Search the web for information. Use for current events, facts, or any query requiring up-to-date information.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of results (1-10)",
                        "default": 5
                    }
                },
                "required": ["query"]
            },
            function=ToolImplementations.search_web
        ),
        Tool(
            name="send_email",
            description="Send an email to a recipient. Use when the user wants to compose and send emails.",
            parameters={
                "type": "object",
                "properties": {
                    "to": {
                        "type": "string",
                        "description": "Recipient email address"
                    },
                    "subject": {
                        "type": "string",
                        "description": "Email subject"
                    },
                    "body": {
                        "type": "string",
                        "description": "Email body content"
                    },
                    "cc": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "CC recipients"
                    }
                },
                "required": ["to", "subject", "body"]
            },
            function=ToolImplementations.send_email,
            requires_confirmation=True  # Email needs user confirmation
        ),
        Tool(
            name="create_reminder",
            description="Create a reminder for a specific date/time.",
            parameters={
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Reminder title"
                    },
                    "due_datetime": {
                        "type": "string",
                        "description": "When to remind (ISO 8601 format)"
                    },
                    "description": {
                        "type": "string",
                        "description": "Additional details"
                    },
                    "priority": {
                        "type": "string",
                        "enum": ["low", "normal", "high"],
                        "default": "normal"
                    }
                },
                "required": ["title", "due_datetime"]
            },
            function=ToolImplementations.create_reminder
        ),
        Tool(
            name="calculate",
            description="Perform mathematical calculations. Use for any math operations.",
            parameters={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression (e.g., '2 + 2', '10 * 5', '2 ** 8')"
                    }
                },
                "required": ["expression"]
            },
            function=ToolImplementations.calculate
        ),
        Tool(
            name="get_weather",
            description="Get current weather for a location.",
            parameters={
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name (e.g., 'London', 'New York')"
                    },
                    "units": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "default": "celsius"
                    }
                },
                "required": ["location"]
            },
            function=ToolImplementations.get_weather
        ),
        Tool(
            name="manage_tasks",
            description="Create, update, delete, or list tasks.",
            parameters={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["create", "update", "delete", "list"],
                        "description": "Action to perform"
                    },
                    "task_id": {
                        "type": "string",
                        "description": "Task ID (for update/delete)"
                    },
                    "title": {
                        "type": "string",
                        "description": "Task title (for create)"
                    },
                    "description": {
                        "type": "string",
                        "description": "Task description"
                    },
                    "due_date": {
                        "type": "string",
                        "description": "Due date (ISO 8601)"
                    },
                    "priority": {
                        "type": "string",
                        "enum": ["low", "normal", "high"],
                        "default": "normal"
                    }
                },
                "required": ["action"]
            },
            function=ToolImplementations.manage_tasks
        )
    ]


# ============ Personal Assistant Agent ============

class PersonalAssistant:
    """
    A complete personal assistant agent with:
    - Multiple tool capabilities
    - Conversation memory
    - User confirmation for sensitive actions
    - Error handling
    """

    def __init__(
        self,
        user_name: str = "User",
        model: str = "gpt-4o",
        max_iterations: int = 10
    ):
        self.user_name = user_name
        self.client = OpenAI()
        self.model = model
        self.max_iterations = max_iterations

        # Initialize tools
        self.tools = {tool.name: tool for tool in create_tool_registry()}

        # Conversation history
        self.messages: list[dict] = []

        # Pending confirmations
        self.pending_confirmation: dict | None = None

    def _get_system_prompt(self) -> str:
        """Generate the system prompt."""
        return f"""You are a helpful personal assistant for {self.user_name}.

Your capabilities include:
- Searching the web for information
- Sending emails (requires user confirmation)
- Creating reminders and managing tasks
- Getting weather information
- Performing calculations
- Telling time in different timezones

Guidelines:
1. Be proactive and helpful
2. Ask clarifying questions when needed
3. For sensitive actions (sending emails, deleting tasks), summarize what you're about to do
4. Provide concise, actionable responses
5. When using tools, explain what you're doing

Current date/time: {datetime.now().isoformat()}"""

    def _build_openai_tools(self) -> list[dict]:
        """Convert tools to OpenAI format."""
        return [tool.to_openai_format() for tool in self.tools.values()]

    def _execute_tool(self, name: str, arguments: dict) -> tuple[bool, Any]:
        """Execute a tool, handling confirmation if needed."""
        tool = self.tools.get(name)

        if not tool:
            return False, f"Unknown tool: {name}"

        # Check if confirmation needed
        if tool.requires_confirmation:
            self.pending_confirmation = {
                "tool": name,
                "arguments": arguments,
                "description": self._describe_action(name, arguments)
            }
            return False, f"CONFIRMATION_NEEDED: {self.pending_confirmation['description']}"

        try:
            result = tool.function(**arguments)
            return True, result
        except Exception as e:
            return False, f"Error: {str(e)}"

    def _describe_action(self, tool_name: str, arguments: dict) -> str:
        """Generate human-readable description of an action."""
        if tool_name == "send_email":
            return f"Send email to {arguments.get('to')} with subject: '{arguments.get('subject')}'"
        elif tool_name == "manage_tasks" and arguments.get("action") == "delete":
            return f"Delete task: {arguments.get('task_id')}"
        return f"Execute {tool_name} with {arguments}"

    def confirm_pending_action(self, confirmed: bool) -> str:
        """Process user confirmation for pending action."""
        if not self.pending_confirmation:
            return "No action pending confirmation."

        if confirmed:
            tool = self.tools[self.pending_confirmation["tool"]]
            try:
                result = tool.function(**self.pending_confirmation["arguments"])
                response = f"✓ Action completed: {json.dumps(result)}"
            except Exception as e:
                response = f"✗ Action failed: {str(e)}"
        else:
            response = "Action cancelled by user."

        self.pending_confirmation = None
        return response

    def chat(self, user_input: str) -> str:
        """Process a user message and return response."""
        # Handle confirmation responses
        if self.pending_confirmation:
            if user_input.lower() in ["yes", "y", "confirm", "ok", "proceed"]:
                return self.confirm_pending_action(True)
            elif user_input.lower() in ["no", "n", "cancel", "stop"]:
                return self.confirm_pending_action(False)

        # Add user message
        if not self.messages:
            self.messages.append({
                "role": "system",
                "content": self._get_system_prompt()
            })

        self.messages.append({"role": "user", "content": user_input})

        # Get response
        for iteration in range(self.max_iterations):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                tools=self._build_openai_tools(),
                tool_choice="auto"
            )

            assistant_message = response.choices[0].message

            # Check for tool calls
            if not assistant_message.tool_calls:
                # No tool calls, return response
                self.messages.append({
                    "role": "assistant",
                    "content": assistant_message.content
                })
                return assistant_message.content

            # Process tool calls
            self.messages.append(assistant_message)

            for tool_call in assistant_message.tool_calls:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)

                success, result = self._execute_tool(func_name, func_args)

                # Check if confirmation needed
                if not success and str(result).startswith("CONFIRMATION_NEEDED"):
                    # Return confirmation request
                    return f"⚠️ I need your confirmation:\n{self.pending_confirmation['description']}\n\nReply 'yes' to proceed or 'no' to cancel."

                # Add tool result
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result) if success else str(result)
                })

        return "I apologize, but I couldn't complete that request. Could you try rephrasing?"

    def reset_conversation(self):
        """Clear conversation history."""
        self.messages = []
        self.pending_confirmation = None


# ============ CLI Interface ============

def run_assistant():
    """Run the personal assistant in CLI mode."""
    print("=" * 50)
    print("Personal Assistant Agent")
    print("=" * 50)

    name = input("What's your name? ").strip() or "User"
    assistant = PersonalAssistant(user_name=name)

    print(f"\nHello {name}! I'm your personal assistant.")
    print("I can help with web searches, emails, reminders, weather, calculations, and more.")
    print("Type 'quit' to exit, 'reset' to clear conversation.\n")

    while True:
        try:
            user_input = input(f"{name}: ").strip()

            if not user_input:
                continue

            if user_input.lower() == "quit":
                print("Goodbye!")
                break

            if user_input.lower() == "reset":
                assistant.reset_conversation()
                print("Conversation reset.\n")
                continue

            response = assistant.chat(user_input)
            print(f"\nAssistant: {response}\n")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    run_assistant()
```

---

## Safety and Guardrails

### Cost Guardrails for Agents

Agent loops can rapidly accumulate costs when agents iterate multiple times or call expensive models repeatedly. Implement the following guardrails to manage expenses:

**Per-Loop Cost Tracking:**
Monitor token usage and API costs for each agent iteration:
```python
class CostTracker:
    def __init__(self, model: str):
        self.model = model
        self.loop_costs = []
        self.total_tokens = 0

    def track_iteration(self, tokens_used: int, cost: float):
        self.loop_costs.append({
            "tokens": tokens_used,
            "cost": cost,
            "timestamp": datetime.now()
        })
        self.total_tokens += tokens_used

    def get_loop_cost(self) -> float:
        return sum(loop["cost"] for loop in self.loop_costs)
```

**Maximum Iteration Limits:**
Enforce hard limits on agent loop iterations to prevent runaway execution:
- Typical limit: 10-15 iterations per task
- Critical operations: 3-5 iterations maximum
- Long-running tasks: 100+ iterations with monitoring

**Budget Enforcement:**
Set and enforce strict budget caps:
```python
class BudgetEnforcer:
    def __init__(self, max_budget: float):
        self.max_budget = max_budget
        self.spent = 0.0

    def can_proceed(self, estimated_cost: float) -> bool:
        return self.spent + estimated_cost <= self.max_budget

    def charge(self, cost: float):
        self.spent += cost
        if self.spent >= self.max_budget * 0.8:
            self.alert_budget_warning()
```

**Cost Alerts and Circuit Breakers:**
Implement alerts at key thresholds and automatic circuit breakers:
- 50% of budget: Informational log
- 80% of budget: Alert and reduce exploration
- 95% of budget: Enter strict mode (no optional operations)
- 100% of budget: Circuit breaker—stop agent immediately

---

### Agent Evaluation Framework

Measure agent effectiveness beyond just "did it work?" Use a comprehensive framework:

**Task Success Rate Measurement:**
- Count successful completions vs. failures
- Track partial successes separately (found some but not all results)
- Measure success within acceptable cost and latency bounds

**Token Efficiency Metrics:**
```python
efficiency_score = task_value / tokens_used
cost_per_success = total_cost / successful_tasks
```
- Better agents achieve same results with fewer tokens
- Compare against baseline single-call performance

**Latency Per Step Tracking:**
Monitor time at each step to identify bottlenecks:
- API call latency: How long for function execution?
- Reasoning latency: How long for LLM to decide next step?
- Total task latency: E2E time from start to finish

**Failure Mode Categorization:**
Track *why* agents fail to improve systematically:
- Tool selection errors: Agent picked wrong tool
- Parameter errors: Correct tool, wrong arguments
- Logic errors: Tool executed but produced unexpected result
- Context exhaustion: Ran out of context window
- Budget exhaustion: Hit cost limits mid-task
- Timeout: Exceeded time budget

**Practical Measurement Code:**

```python
"""
Agent Evaluation Harness
Run a suite of test tasks and measure agent performance.
"""
from dataclasses import dataclass, field
from typing import Any
import time
import json

@dataclass
class TaskResult:
    """Result of a single agent task evaluation."""
    task_id: str
    task_description: str
    success: bool
    iterations_used: int
    total_tokens: int
    latency_seconds: float
    tools_called: list[str] = field(default_factory=list)
    error: str = ""
    failure_category: str = ""  # "tool_selection", "parameter", "logic", "timeout", "budget"

@dataclass
class EvaluationReport:
    """Aggregated evaluation results."""
    results: list[TaskResult]

    @property
    def success_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.success) / len(self.results)

    @property
    def avg_iterations(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.iterations_used for r in self.results) / len(self.results)

    @property
    def avg_latency(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.latency_seconds for r in self.results) / len(self.results)

    @property
    def failure_breakdown(self) -> dict[str, int]:
        failures = [r for r in self.results if not r.success]
        breakdown: dict[str, int] = {}
        for f in failures:
            cat = f.failure_category or "unknown"
            breakdown[cat] = breakdown.get(cat, 0) + 1
        return breakdown

    def summary(self) -> str:
        lines = [
            f"Tasks run: {len(self.results)}",
            f"Success rate: {self.success_rate:.1%}",
            f"Avg iterations: {self.avg_iterations:.1f}",
            f"Avg latency: {self.avg_latency:.1f}s",
            f"Failure breakdown: {self.failure_breakdown}",
        ]
        return "\n".join(lines)


def evaluate_agent(agent, test_tasks: list[dict]) -> EvaluationReport:
    """
    Run an agent against a suite of test tasks.

    Each task dict should have:
      - "id": str
      - "description": str (the objective to pass to agent.run())
      - "expected_tools": list[str] (tools that should be called)
      - "validator": callable(result) -> bool (checks if result is correct)
    """
    results = []
    for task in test_tasks:
        start = time.time()
        try:
            result = agent.run(task["description"])
            elapsed = time.time() - start

            success = task.get("validator", lambda _: True)(result)

            # Extract iteration count from agent memory if available
            memory = getattr(agent, "memory", None)
            iterations = len(memory.steps) if memory and hasattr(memory, "steps") else 0

            # Extract tools called from agent memory
            tools_used = []
            if memory and hasattr(memory, "steps"):
                for step in memory.steps:
                    if step.action:
                        tools_used.append(step.action.tool_name)

            results.append(TaskResult(
                task_id=task["id"],
                task_description=task["description"],
                success=success,
                iterations_used=iterations,
                total_tokens=0,  # In production: sum from API response.usage fields
                latency_seconds=elapsed,
                tools_called=tools_used,
            ))
        except Exception as e:
            elapsed = time.time() - start
            results.append(TaskResult(
                task_id=task["id"],
                task_description=task["description"],
                success=False,
                iterations_used=0,
                total_tokens=0,
                latency_seconds=elapsed,
                error=str(e),
                failure_category="timeout" if "timeout" in str(e).lower() else "logic",
            ))

    return EvaluationReport(results=results)
```

> **Why evaluation matters for agents:** Unlike single-call LLM applications, agents make multiple decisions in sequence. A 90% per-step accuracy compounds to just 35% over 10 steps (0.9^10). Measuring end-to-end task success rate -- not per-step accuracy -- is the metric that matters.

---

### When Agents Fail

**Infinite Loops:**
Agents can get stuck repeatedly calling the same tool or cycling between states.
- *Cause*: Poor tool design, ambiguous success criteria, lack of progress detection
- *Solution*: Implement state tracking and break after detecting repeated states; add explicit "done" tool; limit iterations

**Tool Call Failures:**
Tools may fail to execute, return errors, or provide unexpected responses.
- *Cause*: External service down, network issues, invalid parameters, permission denied
- *Solution*: Implement retry logic with exponential backoff; provide detailed error messages; add fallback tools

**Budget Runaway Scenarios:**
Costs spiral out of control during agent execution.
- *Cause*: Inefficient prompts, excessive reasoning loops, expensive model selection
- *Solution*: Set iteration caps; use cheaper models for reasoning; implement cost-aware routing

**Context Window Overflow:**
Agent conversation history grows so large it exceeds token limits.
- *Cause*: Long reasoning chains, many tool results, verbose logging
- *Solution*: Implement rolling window memory; summarize tool results; use hierarchical agents (delegate to sub-agents)

---

### Agent Cost and Latency Analysis

Agent loops multiply the cost of a single LLM call. You must understand this before building.

**Cost Per Agent Iteration (Approximate, GPT-4o, Jan 2025):**

| Component | Tokens (Typical) | Cost | Notes |
|-----------|-----------------|------|-------|
| System prompt + tool schemas | 500-2000 input | $0.004-0.015 | Fixed cost every iteration |
| Conversation history | Grows ~500/iteration | Cumulative | Iteration N costs N × history |
| Model reasoning output | 100-300 output | $0.001-0.004 | Thought + action |
| Tool execution | 0 (LLM tokens) | Varies | External API costs apply |
| **Per-iteration total** | **~1000-3000** | **$0.005-0.02** | |

**Total cost for a multi-step agent task:**

| Steps | Cumulative Input Tokens | Approximate Cost (GPT-4o) | Cost (GPT-4o-mini) |
|-------|------------------------|--------------------------|---------------------|
| 1 | ~1,500 | $0.01 | $0.001 |
| 3 | ~6,000 | $0.05 | $0.005 |
| 5 | ~12,000 | $0.10 | $0.01 |
| 10 | ~30,000 | $0.25 | $0.025 |

> **Key insight:** Input token cost grows quadratically with agent steps because each iteration includes all previous history. A 10-step agent doesn't cost 10x a single call — it costs ~25x due to cumulative context. This is why context window management and history summarization matter for agents even more than for simple chat.

**Latency Per Iteration:**

| Component | Typical Latency | Notes |
|-----------|----------------|-------|
| LLM inference (reasoning) | 500-2000ms | Scales with context length |
| Tool execution | 50-5000ms | Depends on external API |
| JSON parsing/validation | <10ms | Negligible |
| **Total per iteration** | **~600-7000ms** | |

A 5-step agent typically takes **5-20 seconds** end-to-end. A 10-step agent takes **15-60 seconds**. Users will not wait this long without progress indicators.

**Cost Optimization Strategies for Agents:**
1. **Use cheaper models for reasoning.** GPT-4o-mini or Claude Haiku for the think/act loop; reserve GPT-4o for final synthesis.
2. **Summarize history.** After 5 steps, compress previous steps into a summary to reduce input tokens.
3. **Limit tool schemas.** Only include relevant tools per step, not all 20 tools every time.
4. **Cache tool results.** If the same tool is called with the same inputs, return cached results.
5. **Set strict iteration limits.** Most tasks should complete in 3-7 steps. If you hit 10+, something is wrong.

---

### Implementing Agent Safety

```python
"""
Agent Safety Framework
"""
from dataclasses import dataclass
from typing import Callable
from enum import Enum
import re

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SafetyRule:
    """A safety rule for agent actions."""
    name: str
    description: str
    check: Callable[[str, dict], bool]  # Returns True if safe
    risk_level: RiskLevel
    action_on_violation: str  # "block", "warn", "confirm"


class AgentSafetyGuard:
    """
    Safety framework for AI agents.
    Prevents harmful or unintended actions.
    """

    def __init__(self):
        self.rules: list[SafetyRule] = []
        self.violation_log: list[dict] = []
        self.blocked_patterns: list[re.Pattern] = []

        # Initialize default rules
        self._init_default_rules()

    def _init_default_rules(self):
        """Initialize default safety rules."""

        # Block destructive operations
        self.add_rule(SafetyRule(
            name="no_mass_delete",
            description="Prevent mass deletion operations",
            check=lambda tool, args: not (
                "delete" in tool.lower() and
                args.get("all", False) or args.get("recursive", False)
            ),
            risk_level=RiskLevel.CRITICAL,
            action_on_violation="block"
        ))

        # Prevent credential access
        self.add_rule(SafetyRule(
            name="no_credential_access",
            description="Block access to credentials or secrets",
            check=lambda tool, args: not any(
                key in str(args).lower()
                for key in ["password", "secret", "token", "api_key", "credential"]
            ),
            risk_level=RiskLevel.CRITICAL,
            action_on_violation="block"
        ))

        # Confirm financial transactions
        self.add_rule(SafetyRule(
            name="confirm_financial",
            description="Require confirmation for financial actions",
            check=lambda tool, args: tool not in ["transfer_money", "make_payment", "purchase"],
            risk_level=RiskLevel.HIGH,
            action_on_violation="confirm"
        ))

        # Warn on external communication
        self.add_rule(SafetyRule(
            name="warn_external_comm",
            description="Warn when sending external communications",
            check=lambda tool, args: tool not in ["send_email", "send_message", "post_social"],
            risk_level=RiskLevel.MEDIUM,
            action_on_violation="warn"
        ))

        # Block SQL injection patterns
        self.add_blocked_pattern(r";\s*DROP\s+TABLE", "SQL injection attempt")
        self.add_blocked_pattern(r";\s*DELETE\s+FROM", "SQL injection attempt")
        self.add_blocked_pattern(r"UNION\s+SELECT", "SQL injection attempt")

    def add_rule(self, rule: SafetyRule):
        """Add a safety rule."""
        self.rules.append(rule)

    def add_blocked_pattern(self, pattern: str, reason: str):
        """Add a blocked pattern."""
        self.blocked_patterns.append({
            "pattern": re.compile(pattern, re.IGNORECASE),
            "reason": reason
        })

    def check_action(self, tool_name: str, arguments: dict) -> dict:
        """
        Check if an action is safe to execute.

        Returns:
            dict with keys:
            - safe: bool
            - action: "allow" | "block" | "warn" | "confirm"
            - violations: list of violated rules
            - message: explanation
        """
        violations = []
        highest_action = "allow"
        action_priority = {"allow": 0, "warn": 1, "confirm": 2, "block": 3}

        # Check against rules
        for rule in self.rules:
            try:
                if not rule.check(tool_name, arguments):
                    violations.append({
                        "rule": rule.name,
                        "description": rule.description,
                        "risk_level": rule.risk_level.value,
                        "action": rule.action_on_violation
                    })

                    if action_priority[rule.action_on_violation] > action_priority[highest_action]:
                        highest_action = rule.action_on_violation
            except Exception as e:
                # Rule check failed - be conservative
                violations.append({
                    "rule": rule.name,
                    "description": f"Rule check failed: {e}",
                    "risk_level": RiskLevel.MEDIUM.value,
                    "action": "warn"
                })

        # Check for blocked patterns in arguments
        args_str = str(arguments)
        for blocked in self.blocked_patterns:
            if blocked["pattern"].search(args_str):
                violations.append({
                    "rule": "blocked_pattern",
                    "description": blocked["reason"],
                    "risk_level": RiskLevel.CRITICAL.value,
                    "action": "block"
                })
                highest_action = "block"

        # Log violations
        if violations:
            self.violation_log.append({
                "tool": tool_name,
                "arguments": arguments,
                "violations": violations,
                "action_taken": highest_action
            })

        return {
            "safe": highest_action in ["allow", "warn"],
            "action": highest_action,
            "violations": violations,
            "message": self._format_message(violations, highest_action)
        }

    def _format_message(self, violations: list, action: str) -> str:
        """Format a user-friendly message about violations."""
        if not violations:
            return "Action approved."

        if action == "block":
            return f"⛔ Action blocked: {violations[0]['description']}"
        elif action == "confirm":
            return f"⚠️ Confirmation required: {violations[0]['description']}"
        elif action == "warn":
            return f"⚡ Warning: {violations[0]['description']}"

        return "Action allowed with caveats."


class SafeAgent:
    """Agent wrapper with safety guardrails."""

    def __init__(self, agent, safety_guard: AgentSafetyGuard = None):
        self.agent = agent
        self.safety = safety_guard or AgentSafetyGuard()
        self.confirmation_callback: Callable[[str], bool] = lambda msg: False

    def set_confirmation_callback(self, callback: Callable[[str], bool]):
        """Set callback for user confirmation."""
        self.confirmation_callback = callback

    def execute_tool(self, tool_name: str, arguments: dict) -> tuple[bool, Any]:
        """Execute tool with safety checks."""
        # Check safety
        safety_result = self.safety.check_action(tool_name, arguments)

        if safety_result["action"] == "block":
            return False, safety_result["message"]

        if safety_result["action"] == "confirm":
            confirmed = self.confirmation_callback(safety_result["message"])
            if not confirmed:
                return False, "Action cancelled by user."

        if safety_result["action"] == "warn":
            print(f"Warning: {safety_result['message']}")

        # Execute the tool
        return self.agent.execute_tool(tool_name, arguments)
```

---

## Interview Preparation

### Career Mapping

Agent and function calling knowledge maps directly to these industry roles:

| Role | What They Need | What This Blog Gives Them |
|------|---------------|--------------------------|
| **AI/ML Engineer** | Build and optimize agent pipelines | ReAct, Plan-and-Execute, error recovery, evaluation framework |
| **AI Platform Engineer** | Scalable agent infrastructure | Cost analysis, safety framework, iteration limits, budget enforcement |
| **Solutions Architect** | Design agent systems for clients | Architecture tradeoffs (ReAct vs. Plan-Execute), when-not-to-use-agents analysis |
| **Full-Stack Engineer** | Integrate agents into applications | Multi-provider function calling, CLI interface, confirmation flows |
| **DevOps/SRE** | Monitor and maintain agent systems | Cost tracking, failure categorization, circuit breakers |

> **Framework context:** This blog builds agents from scratch to teach the underlying mechanics. In production, teams commonly use **LangChain** (Blog 19), **LangGraph**, **CrewAI**, or **AutoGen** for agent orchestration. Understanding the internals here is essential for three reasons: (1) debugging framework-level issues requires knowing the underlying loop, (2) frameworks impose architectural opinions that may not fit your use case, and (3) for simple agent tasks, the overhead of a framework often exceeds the cost of writing the loop yourself.

### Conceptual Questions

1. **What is function calling and how does it differ from text generation?**
   - Function calling enables LLMs to invoke structured tools by generating JSON conforming to a schema, rather than free-form text.
   - The model doesn't execute code — it generates structured output that your code parses and executes.
   - The model decides when to call tools based on semantic matching between the user query and tool descriptions in the prompt.
   - Native function calling (API-level) is more reliable but provider-locked; ReAct-style text parsing is more portable but fragile.

2. **When would you use ReAct vs. Plan-and-Execute, and when would you use neither?**
   - **ReAct**: Exploratory tasks with <5 steps where the path depends on intermediate results. Lower overhead per step.
   - **Plan-and-Execute**: Well-defined tasks with 5+ steps, need for auditability, or when replanning is valuable. Higher initial cost but more focused execution.
   - **Neither (single function call)**: When a task maps to one tool invocation. Adding an agent loop to "get the weather" wastes latency and money.
   - **Neither (deterministic workflow)**: When you need >99% reliability and the tool sequence is known in advance. Agents introduce stochastic failure at each step.

3. **How do you design effective tool schemas?**
   - Clear, specific descriptions that tell the model *when* to use the tool, not just what it does
   - Well-defined parameter types with enums for constrained values
   - Include usage examples in descriptions (improves model tool selection accuracy)
   - Keep total tools under 10-20 for reliable selection; use dynamic tool loading for large registries

4. **What are the most common agent failure modes and how do you mitigate them?**
   - **Infinite loops**: Detect repeated states, enforce iteration limits, add explicit "done" tool
   - **Tool selection errors**: Improve tool descriptions, reduce number of available tools, log and analyze selection patterns
   - **Budget runaway**: Track cumulative tokens per session, implement circuit breakers at 80%/95%/100% of budget
   - **Context overflow**: Summarize history after N steps, truncate tool results, use hierarchical agents
   - **Compounding failure**: Each step has ~90-95% success rate; 10 steps → 35-60% E2E success. Reduce step count, add error recovery.

5. **How much does an agent loop cost compared to a single LLM call?**
   - A single GPT-4o call with 2K tokens costs ~$0.01. A 5-step agent costs ~$0.10 (not 5x, but ~10x, because input tokens accumulate quadratically with conversation history). A 10-step agent costs ~$0.25. At 10K tasks/day, that's $2,500/day. Cost optimization: use cheaper models for reasoning steps, summarize history, cache tool results, set strict iteration limits.

### Coding Challenges

**Challenge 1**: Implement a tool that requires multi-step validation:

```python
def create_validated_tool():
    """
    Create a tool that validates inputs before execution.
    The tool should:
    1. Validate email format
    2. Check recipient against allowed list
    3. Require confirmation for external recipients
    """
    import re

    INTERNAL_DOMAINS = ["company.com", "internal.org"]

    def send_secure_email(
        to: str,
        subject: str,
        body: str,
        confirmed: bool = False
    ) -> dict:
        # Validate email format
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, to):
            return {"error": "Invalid email format"}

        # Check if external
        domain = to.split("@")[1]
        is_external = domain not in INTERNAL_DOMAINS

        if is_external and not confirmed:
            return {
                "status": "confirmation_required",
                "message": f"External recipient detected: {to}",
                "action": "Please confirm sending to external recipient"
            }

        # Send email
        return {
            "status": "sent",
            "to": to,
            "external": is_external,
            "message_id": "msg_12345"
        }

    return Tool(
        name="send_secure_email",
        description="Send email with security validation",
        parameters={
            "type": "object",
            "properties": {
                "to": {"type": "string"},
                "subject": {"type": "string"},
                "body": {"type": "string"},
                "confirmed": {"type": "boolean", "default": False}
            },
            "required": ["to", "subject", "body"]
        },
        function=send_secure_email
    )
```

**Challenge 2**: Build an agent that can recover from failures:

```python
class RecoveryAgent:
    """Agent that implements multiple recovery strategies."""

    def execute_with_recovery(
        self,
        tool_name: str,
        arguments: dict,
        max_attempts: int = 3
    ) -> dict:
        strategies = [
            self._retry_same,
            self._retry_with_fixed_args,
            self._use_alternative_tool
        ]

        last_error = None

        for attempt, strategy in enumerate(strategies[:max_attempts]):
            try:
                result = strategy(tool_name, arguments, last_error)
                if result.get("success"):
                    return result
                last_error = result.get("error")
            except Exception as e:
                last_error = str(e)

        return {"success": False, "error": last_error}
```

---

## Exercises

### Exercise 1: Build a Customer Service Agent
Create an agent that can:
- Look up customer information
- Check order status
- Process returns
- Escalate to human when needed

### Exercise 2: Implement Tool Chaining
Build a system where tools can call other tools:
```python
# Example: search_and_summarize calls search, then summarize
```

### Exercise 3: Add Memory to an Agent
Implement persistent memory that:
- Stores conversation history
- Remembers user preferences
- Maintains context across sessions

### Exercise 4: Build a Safety Monitor
Create a monitoring system that:
- Tracks all agent actions
- Detects anomalies
- Alerts on suspicious patterns
- Generates audit reports

---

## Summary

### Key Takeaways

1. **Function calling is schema-driven**: The model decides to call tools based on semantic matching between user queries and tool descriptions — not by executing code. Schema quality directly determines tool selection accuracy.
2. **Agents don't learn — they observe**: Agents accumulate observations within a session but don't update weights. Each session starts fresh without explicit persistent memory.
3. **Choose your architecture deliberately**: ReAct for exploratory <5-step tasks; Plan-and-Execute for structured 5+ step tasks; single function calls when one tool suffices; deterministic workflows when you need >99% reliability.
4. **Agent costs grow quadratically**: Each iteration includes all previous history as input tokens. A 10-step agent costs ~25x a single call, not 10x. Budget enforcement and history summarization are not optional.
5. **Compounding failure is the agent-specific risk**: 90% per-step accuracy → 35% over 10 steps. Minimize step count, add error recovery, and measure end-to-end task success rate — not per-step accuracy.
6. **Safety is the first thing to build, not the last**: Implement guardrails (confirmation flows, budget limits, blocked patterns) before deploying any agent to users.

### Production Readiness Checklist

- [ ] All tools have comprehensive error handling
- [ ] Sensitive actions require user confirmation
- [ ] Agent actions are logged for audit
- [ ] Rate limiting prevents cost overruns
- [ ] Fallback strategies exist for tool failures
- [ ] Input validation prevents injection attacks
- [ ] Human escalation path is defined

---

## Self-Assessment Rubric

Rate yourself honestly after completing this blog:

| Criteria | Excellent (9-10) | Good (7-8) | Needs Work (5-6) |
|----------|------------------|-----------|------------------|
| Function calling implementation | Multi-provider support, understand internal mechanism (schema injection, trained decision-making), proper error handling | Single provider working end-to-end with error handling | Can define tools but struggle with the response loop |
| Tool schema design | Clear descriptions with constraints, enums, usage hints; understand why descriptions affect model tool selection | Correct types and descriptions, mostly complete | Vague descriptions, missing type info |
| Agent architecture | Understand ReAct vs. Plan-Execute tradeoffs (cost, latency, reliability), can implement both, know when NOT to use agents | One pattern implemented cleanly | Basic loop without clear reasoning structure |
| Cost awareness | Can estimate per-iteration cost, understand quadratic token growth, implement budget enforcement | Aware that agents cost more than single calls, set iteration limits | No cost awareness |
| Error handling & recovery | Retry logic, fallback strategies, input correction, failure categorization | Retry with basic error messages | Crashes on tool failures |
| Safety & guardrails | Confirmation flows, budget limits, input validation, violation logging, pattern blocking | Basic safety checks and iteration limits | No safety measures |

### What This Blog Does Well
- Multi-provider function calling coverage (OpenAI, Anthropic, Google) with working code for each
- Explains how function calling works internally (schema injection, trained decision-making, structured output generation)
- Multiple agent architectures (ReAct, Plan-and-Execute, Multi-Agent) explained from scratch with explicit tradeoff analysis
- Clear guidance on when NOT to use agents (single-call tasks, high-reliability, latency-constrained, cost-sensitive)
- Comprehensive safety framework with rule-based guards, pattern blocking, and confirmation flows
- Agent evaluation framework with failure mode categorization and the critical compounding accuracy insight
- Per-iteration cost and latency analysis with quadratic token growth explanation
- Career mapping to five specific industry roles with framework contextualization

### Where This Blog Falls Short
- All tool implementations are simulated -- no real API integrations (weather, email, search) are demonstrated end-to-end
- The multi-agent system is conceptually sound but untested at scale; real orchestration involves distributed state, timeouts, and partial failure handling not covered here
- No async/streaming agent patterns -- production agents need non-blocking tool execution
- Cost tracking code is illustrative only; real cost tracking requires parsing provider-specific token usage from API responses
- No discussion of prompt caching or memory compression for long agent sessions

### Architect Sanity Checks

**Check 1: Can They Build a Working Agent System?**
- **YES**: The blog provides complete ReAct and Plan-and-Execute implementations with error recovery and safety guardrails. The reader understands the mechanics of function calling (schema injection, trained decision-making), can implement agents with multiple providers, and knows how to add confirmation flows for sensitive actions. Tool implementations are simulated, but the integration pattern is correct and transferable to real APIs.

**Check 2: Can They Diagnose Agent Failures and Control Costs?**
- **YES**: The failure mode categorization (tool selection, parameter, logic, context exhaustion, budget, timeout) provides a diagnostic framework. The compounding accuracy analysis (0.9^10 = 0.35) gives the reader the mental model for why agents fail at scale. The cost analysis quantifies per-iteration and cumulative costs with the quadratic growth insight. Budget enforcement with circuit breakers is covered. The reader can explain why a 10-step agent costs 25x a single call, not 10x.

**Check 3: Can They Make Architecture Decisions?**
- **YES**: The blog provides concrete tradeoffs between ReAct and Plan-and-Execute (cost, latency, reliability, best use cases), between native function calling and ReAct text parsing (reliability vs. portability), and between agents and simpler alternatives (single calls, deterministic workflows). The reader can justify their architecture choice to a senior engineer and explain when agents are overkill.

---

## What's Next?

In **Blog 19: LangChain Deep Dive**, we'll explore the most popular framework for building LLM applications. You'll learn:
- LangChain's core abstractions (Chains, Agents, Tools)
- Building complex workflows with LCEL
- Integration with vector stores and memory
- Production deployment patterns

The agent patterns you learned today form the foundation—LangChain provides the scaffolding to build faster.

---

*Practice building agents with different tool combinations. Start simple, add complexity gradually, and always implement safety first.*
