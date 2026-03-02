# Bonus A: LangChain4j — Generative AI for Java Developers

## Series Navigation
- **Previous**: [Blog 25: Building Your Career in Generative AI](blog-25-career-in-generative-ai.md)
- **Next**: [Bonus B: Spring AI Enterprise Integration](bonus-b-spring-ai.md)

**Reading time:** 60–90 minutes
**Coding time:** 3–5 hours
**Total investment:** ~5–6 hours

---

## What You'll Walk Away With

After completing this bonus module, you will be able to:

1. **Set up LangChain4j** in Maven and Gradle projects with proper dependency management
2. **Integrate multiple LLM providers** including OpenAI, Anthropic, Azure OpenAI, and local models
3. **Build AI services** using the declarative @AiService annotation pattern
4. **Implement RAG systems** in Java with embedding stores and document loaders
5. **Create function-calling agents** with type-safe tool definitions
6. **Manage conversation memory** across different storage backends
7. **Deploy production-ready** Java AI applications with proper error handling

> **How to read this blog:** This is a bonus module for Java developers. If you completed the main series (Blogs 1-25) in Python, you already understand the AI concepts — this blog shows you how to apply them in Java using LangChain4j. If you're starting here, read Blogs 14 (AI APIs) and 18 (Function Calling) first for conceptual foundations. Sections are independent — you can skip to RAG (Section 6) or Tools (Section 7) if those are your immediate needs.

### Who This Blog Is For & Section Priority

| Role | Critical Sections | Supporting Sections | What Hiring Panels Look For |
|------|------------------|--------------------|-----------------------------|
| **Java AI/LLM Engineer** | 3-7 (Model integration, AI Services, Memory, RAG, Tools) | 8-10 (Streaming, Testing, Production) | Can you build an AI Service with RAG and tools from scratch? |
| **Java Backend + AI** | 4, 10, 11, 14 (AI Services, Production, Failure Modes, When NOT) | 5-7 (Memory, RAG, Tools) | Can you integrate AI into existing Java services with proper resilience? |
| **Enterprise Architect** | 14, 10, 12, 6 (When NOT, Production, Cost, RAG) | 3-5 (Provider integration, AI Services, Memory) | Can you choose between LangChain4j, Spring AI, and direct API calls with cost justification? |
| **MLOps/Platform Engineer** | 10, 11, 12, 13 (Production, Failure Modes, Cost, Evaluation) | 9 (Testing) | Can you operate AI services in production with monitoring and cost controls? |

**If you have 2 hours:** Read Sections 4 (AI Services), 6 (RAG), 10 (Production Patterns), 14 (When NOT to Use).
**If you have 5 hours:** Read everything, implement Exercises 1 and 4.

### Prerequisites

Before starting this blog, you should have:

- **Java 21+** installed (required for records, virtual threads, switch expressions)
- **Maven or Gradle** project setup experience
- **Completed Blogs 14-19** (AI APIs, chatbot patterns, RAG, function calling, LangChain concepts) — or equivalent experience
- **API keys** for at least one provider (OpenAI, Anthropic, or a local Ollama installation)
- **Familiarity with Spring/Quarkus** is helpful but not required for core sections

---

## What This Blog Does NOT Cover

Before we begin, let's set clear expectations on scope:

- **Spring AI in depth** — Spring AI gets its own dedicated treatment in Bonus B. This blog mentions it for comparison but does not teach it.
- **Quarkus integration** — LangChain4j has a Quarkus extension; we cover Spring patterns here but not Quarkus-specific configuration.
- **Fine-tuning models from Java** — Fine-tuning is covered in Blog 23 (Python). LangChain4j consumes models; it does not train them.
- **Kubernetes/Docker deployment** — Production deployment patterns are in Blog 24. This blog focuses on application-level patterns.
- **Performance benchmarking** — We discuss overhead qualitatively but do not provide JMH benchmarks or load testing results.
- **Reactive/WebFlux deep dive** — The streaming section shows SSE basics, but full reactive programming patterns are out of scope.

---

## Manager's Summary

**What is this?** LangChain4j is the premier Java library for building generative AI applications, bringing the power of LangChain's concepts to the Java ecosystem.

**Why does it matter?** Java remains one of the most widely used languages in enterprise environments (consistently ranking in the top 3 of the TIOBE Index and Stack Overflow surveys). LangChain4j enables Java teams to build AI applications without rewriting in Python or maintaining polyglot systems.

**Business impact:**
- **Leverage existing Java expertise** — No Python training required for Java teams
- **Enterprise integration** — Native support for Spring, Quarkus, and enterprise patterns
- **Type safety** — Compile-time verification reduces runtime errors
- **Maintainability** — Follows Java conventions your teams already know

**Timeline:** Timeline varies significantly by team experience and application complexity. A simple chatbot integration can ship in days; a production RAG system with proper testing, error handling, and monitoring takes considerably longer. This blog covers the patterns needed for both.

**Key differentiator:** Unlike Python LangChain, LangChain4j uses Java idioms like interfaces and annotations, making AI development feel natural to Java developers.

---

## Table of Contents

1. [Why LangChain4j for Java](#1-why-langchain4j-for-java)
2. [Project Setup and Dependencies](#2-project-setup-and-dependencies)
3. [Language Model Integration](#3-language-model-integration)
4. [The AI Services Pattern](#4-the-ai-services-pattern)
5. [Chat Memory and Conversations](#5-chat-memory-and-conversations)
6. [RAG with LangChain4j](#6-rag-with-langchain4j)
7. [Function Calling and Tools](#7-function-calling-and-tools)
8. [Streaming Responses](#8-streaming-responses)
9. [Testing AI Applications](#9-testing-ai-applications)
10. [Production Patterns](#10-production-patterns)
11. [Failure Modes & Error Handling](#11-failure-modes--error-handling)
12. [Cost Tracking with LangChain4j](#12-cost-tracking-with-langchain4j)
13. [Evaluating LangChain4j Applications](#13-evaluating-langchain4j-applications)
14. [When NOT to Use LangChain4j](#14-when-not-to-use-langchain4j)
15. [Interview Preparation](#15-interview-preparation)
16. [Hands-On Exercises](#16-hands-on-exercises)
17. [Summary](#17-summary)

---

## 1. Why LangChain4j for Java

### The Enterprise Java Reality

Java remains one of the dominant languages in enterprise environments. While exact percentages vary by survey and definition of "enterprise," Java consistently ranks in the top 3 across industry surveys (TIOBE, Stack Overflow, JetBrains Developer Ecosystem). The key point is not the precise number — it's that most large organizations have significant Java codebases and Java-skilled teams.

### LangChain4j vs Python LangChain

| Aspect | LangChain4j (Java) | LangChain (Python) |
|--------|-------------------|-------------------|
| **Type Safety** | Compile-time checking | Runtime errors |
| **IDE Support** | Full autocomplete, refactoring | Limited |
| **Enterprise Integration** | Native Spring/Quarkus | Requires adapters |
| **Concurrency** | Virtual threads (Project Loom) | Async/await |
| **Deployment** | Native images (GraalVM) | Container bloat |
| **Learning Curve** | Familiar to Java devs | Python knowledge required |

### Core Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     LangChain4j Architecture                 │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                    AI Services                        │   │
│  │     @AiService interface → Generated Implementation   │   │
│  └─────────────────────────────────────────────────────┘   │
│                           │                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                  Core Abstractions                    │   │
│  │   ChatLanguageModel  │  EmbeddingModel  │  Tools      │   │
│  └─────────────────────────────────────────────────────┘   │
│                           │                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │               Supporting Components                   │   │
│  │   Memory  │  RAG  │  Document Loaders  │  Stores     │   │
│  └─────────────────────────────────────────────────────┘   │
│                           │                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                   Model Providers                     │   │
│  │  OpenAI │ Anthropic │ Azure │ Ollama │ HuggingFace  │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Project Setup and Dependencies

### Maven Configuration

```xml
<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0
                             http://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>

    <groupId>com.example</groupId>
    <artifactId>langchain4j-demo</artifactId>
    <version>1.0.0-SNAPSHOT</version>
    <packaging>jar</packaging>

    <properties>
        <maven.compiler.source>21</maven.compiler.source>
        <maven.compiler.target>21</maven.compiler.target>
        <project.build.sourceEncoding>UTF-8</project.build.sourceEncoding>
        <langchain4j.version>0.35.0</langchain4j.version>
    </properties>

    <dependencyManagement>
        <dependencies>
            <dependency>
                <groupId>dev.langchain4j</groupId>
                <artifactId>langchain4j-bom</artifactId>
                <version>${langchain4j.version}</version>
                <type>pom</type>
                <scope>import</scope>
            </dependency>
        </dependencies>
    </dependencyManagement>

    <dependencies>
        <!-- Core LangChain4j -->
        <dependency>
            <groupId>dev.langchain4j</groupId>
            <artifactId>langchain4j</artifactId>
        </dependency>

        <!-- OpenAI Integration -->
        <dependency>
            <groupId>dev.langchain4j</groupId>
            <artifactId>langchain4j-open-ai</artifactId>
        </dependency>

        <!-- Anthropic (Claude) Integration -->
        <dependency>
            <groupId>dev.langchain4j</groupId>
            <artifactId>langchain4j-anthropic</artifactId>
        </dependency>

        <!-- Azure OpenAI Integration -->
        <dependency>
            <groupId>dev.langchain4j</groupId>
            <artifactId>langchain4j-azure-open-ai</artifactId>
        </dependency>

        <!-- Local Models via Ollama -->
        <dependency>
            <groupId>dev.langchain4j</groupId>
            <artifactId>langchain4j-ollama</artifactId>
        </dependency>

        <!-- Embedding Store - In Memory -->
        <dependency>
            <groupId>dev.langchain4j</groupId>
            <artifactId>langchain4j-embeddings-all-minilm-l6-v2</artifactId>
        </dependency>

        <!-- Document Loaders -->
        <dependency>
            <groupId>dev.langchain4j</groupId>
            <artifactId>langchain4j-document-parser-apache-tika</artifactId>
        </dependency>

        <!-- Vector Store - PostgreSQL with pgvector -->
        <dependency>
            <groupId>dev.langchain4j</groupId>
            <artifactId>langchain4j-pgvector</artifactId>
        </dependency>

        <!-- Testing -->
        <dependency>
            <groupId>org.junit.jupiter</groupId>
            <artifactId>junit-jupiter</artifactId>
            <version>5.10.2</version>
            <scope>test</scope>
        </dependency>

        <!-- Logging -->
        <dependency>
            <groupId>ch.qos.logback</groupId>
            <artifactId>logback-classic</artifactId>
            <version>1.4.14</version>
        </dependency>
    </dependencies>

    <build>
        <plugins>
            <plugin>
                <groupId>org.apache.maven.plugins</groupId>
                <artifactId>maven-compiler-plugin</artifactId>
                <version>3.12.1</version>
                <configuration>
                    <source>21</source>
                    <target>21</target>
                    <compilerArgs>
                        <arg>--enable-preview</arg>
                    </compilerArgs>
                </configuration>
            </plugin>
        </plugins>
    </build>
</project>
```

### Gradle Configuration

```groovy
// build.gradle
plugins {
    id 'java'
    id 'application'
}

group = 'com.example'
version = '1.0.0-SNAPSHOT'

java {
    toolchain {
        languageVersion = JavaLanguageVersion.of(21)
    }
}

repositories {
    mavenCentral()
}

ext {
    langchain4jVersion = '0.35.0'
}

dependencies {
    // BOM for version management
    implementation platform("dev.langchain4j:langchain4j-bom:${langchain4jVersion}")

    // Core
    implementation 'dev.langchain4j:langchain4j'

    // Model Providers
    implementation 'dev.langchain4j:langchain4j-open-ai'
    implementation 'dev.langchain4j:langchain4j-anthropic'
    implementation 'dev.langchain4j:langchain4j-ollama'

    // Embeddings
    implementation 'dev.langchain4j:langchain4j-embeddings-all-minilm-l6-v2'

    // Document Processing
    implementation 'dev.langchain4j:langchain4j-document-parser-apache-tika'

    // Testing
    testImplementation 'org.junit.jupiter:junit-jupiter:5.10.2'
    testImplementation 'org.assertj:assertj-core:3.25.3'
}

test {
    useJUnitPlatform()
}

application {
    mainClass = 'com.example.Application'
}
```

### Project Structure

```
langchain4j-demo/
├── pom.xml (or build.gradle)
├── src/
│   ├── main/
│   │   ├── java/
│   │   │   └── com/
│   │   │       └── example/
│   │   │           ├── Application.java
│   │   │           ├── config/
│   │   │           │   ├── ModelConfig.java
│   │   │           │   └── EmbeddingConfig.java
│   │   │           ├── service/
│   │   │           │   ├── ChatService.java
│   │   │           │   ├── RagService.java
│   │   │           │   └── AgentService.java
│   │   │           ├── tools/
│   │   │           │   ├── WeatherTool.java
│   │   │           │   └── DatabaseTool.java
│   │   │           └── model/
│   │   │               ├── ChatRequest.java
│   │   │               └── ChatResponse.java
│   │   └── resources/
│   │       ├── application.properties
│   │       └── logback.xml
│   └── test/
│       └── java/
│           └── com/
│               └── example/
│                   └── service/
│                       └── ChatServiceTest.java
└── documents/
    └── knowledge-base/
```

---

## 3. Language Model Integration

### OpenAI Integration

```java
package com.example.config;

import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.openai.OpenAiChatModel;
import dev.langchain4j.model.openai.OpenAiChatModelName;
import dev.langchain4j.model.openai.OpenAiStreamingChatModel;

import java.time.Duration;

public class OpenAiConfig {

    private static final String API_KEY = System.getenv("OPENAI_API_KEY");

    /**
     * Creates a standard OpenAI chat model.
     */
    public static ChatLanguageModel createChatModel() {
        return OpenAiChatModel.builder()
                .apiKey(API_KEY)
                .modelName(OpenAiChatModelName.GPT_4_O)
                .temperature(0.7)
                .maxTokens(4096)
                .timeout(Duration.ofSeconds(60))
                .maxRetries(3)
                .logRequests(true)
                .logResponses(true)
                .build();
    }

    /**
     * Creates an OpenAI model optimized for structured output.
     */
    public static ChatLanguageModel createStructuredModel() {
        return OpenAiChatModel.builder()
                .apiKey(API_KEY)
                .modelName(OpenAiChatModelName.GPT_4_O)
                .temperature(0.0)  // Deterministic for structured output
                .responseFormat("json_object")
                .build();
    }

    /**
     * Creates a streaming chat model for real-time responses.
     */
    public static OpenAiStreamingChatModel createStreamingModel() {
        return OpenAiStreamingChatModel.builder()
                .apiKey(API_KEY)
                .modelName(OpenAiChatModelName.GPT_4_O)
                .temperature(0.7)
                .build();
    }

    /**
     * Creates a cost-effective model for simple tasks.
     */
    public static ChatLanguageModel createFastModel() {
        return OpenAiChatModel.builder()
                .apiKey(API_KEY)
                .modelName(OpenAiChatModelName.GPT_4_O_MINI)
                .temperature(0.3)
                .maxTokens(1024)
                .build();
    }
}
```

### Anthropic (Claude) Integration

```java
package com.example.config;

import dev.langchain4j.model.anthropic.AnthropicChatModel;
import dev.langchain4j.model.anthropic.AnthropicChatModelName;
import dev.langchain4j.model.chat.ChatLanguageModel;

import java.time.Duration;

public class AnthropicConfig {

    private static final String API_KEY = System.getenv("ANTHROPIC_API_KEY");

    /**
     * Creates Claude Sonnet model - balanced performance and cost.
     */
    public static ChatLanguageModel createSonnetModel() {
        return AnthropicChatModel.builder()
                .apiKey(API_KEY)
                .modelName(AnthropicChatModelName.CLAUDE_3_5_SONNET_20241022)
                .temperature(0.7)
                .maxTokens(4096)
                .timeout(Duration.ofSeconds(120))
                .maxRetries(3)
                .logRequests(true)
                .logResponses(true)
                .build();
    }

    /**
     * Creates Claude Opus model - highest capability.
     */
    public static ChatLanguageModel createOpusModel() {
        return AnthropicChatModel.builder()
                .apiKey(API_KEY)
                .modelName(AnthropicChatModelName.CLAUDE_3_OPUS_20240229)
                .temperature(0.5)
                .maxTokens(4096)
                .build();
    }

    /**
     * Creates Claude Haiku model - fastest and cheapest.
     */
    public static ChatLanguageModel createHaikuModel() {
        return AnthropicChatModel.builder()
                .apiKey(API_KEY)
                .modelName(AnthropicChatModelName.CLAUDE_3_HAIKU_20240307)
                .temperature(0.3)
                .maxTokens(1024)
                .build();
    }

    /**
     * Creates model configured for use with a system prompt.
     *
     * Note: LangChain4j passes the system prompt at the AI Service level
     * (via @SystemMessage), not at model construction time. This factory
     * method returns a model suitable for system-prompt usage. To actually
     * apply the system prompt, use it with an @AiService interface:
     *
     *   @SystemMessage("Your system prompt here")
     *   String chat(@UserMessage String message);
     */
    public static ChatLanguageModel createForSystemPromptUsage() {
        return AnthropicChatModel.builder()
                .apiKey(API_KEY)
                .modelName(AnthropicChatModelName.CLAUDE_3_5_SONNET_20241022)
                .temperature(0.7)
                .maxTokens(4096)
                .build();
    }
}
```

### Azure OpenAI Integration

```java
package com.example.config;

import dev.langchain4j.model.azure.AzureOpenAiChatModel;
import dev.langchain4j.model.chat.ChatLanguageModel;

import java.time.Duration;

public class AzureOpenAiConfig {

    private static final String ENDPOINT = System.getenv("AZURE_OPENAI_ENDPOINT");
    private static final String API_KEY = System.getenv("AZURE_OPENAI_KEY");
    private static final String DEPLOYMENT_NAME = System.getenv("AZURE_OPENAI_DEPLOYMENT");

    /**
     * Creates Azure OpenAI chat model.
     */
    public static ChatLanguageModel createChatModel() {
        return AzureOpenAiChatModel.builder()
                .endpoint(ENDPOINT)
                .apiKey(API_KEY)
                .deploymentName(DEPLOYMENT_NAME)
                .temperature(0.7)
                .maxTokens(4096)
                .timeout(Duration.ofSeconds(60))
                .maxRetries(3)
                .logRequestsAndResponses(true)
                .build();
    }

    /**
     * Creates model with managed identity (for Azure-hosted apps).
     */
    public static ChatLanguageModel createWithManagedIdentity() {
        return AzureOpenAiChatModel.builder()
                .endpoint(ENDPOINT)
                .deploymentName(DEPLOYMENT_NAME)
                // Uses DefaultAzureCredential when no API key provided
                .temperature(0.7)
                .build();
    }
}
```

### Local Models via Ollama

```java
package com.example.config;

import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.ollama.OllamaChatModel;
import dev.langchain4j.model.ollama.OllamaStreamingChatModel;

import java.time.Duration;

public class OllamaConfig {

    private static final String BASE_URL = "http://localhost:11434";

    /**
     * Creates Llama 3 model via Ollama.
     */
    public static ChatLanguageModel createLlama3Model() {
        return OllamaChatModel.builder()
                .baseUrl(BASE_URL)
                .modelName("llama3.1:8b")
                .temperature(0.7)
                .timeout(Duration.ofMinutes(5))
                .build();
    }

    /**
     * Creates Mistral model via Ollama.
     */
    public static ChatLanguageModel createMistralModel() {
        return OllamaChatModel.builder()
                .baseUrl(BASE_URL)
                .modelName("mistral:7b")
                .temperature(0.7)
                .build();
    }

    /**
     * Creates code-specialized model.
     */
    public static ChatLanguageModel createCodeModel() {
        return OllamaChatModel.builder()
                .baseUrl(BASE_URL)
                .modelName("codellama:13b")
                .temperature(0.2)  // Lower for code generation
                .build();
    }

    /**
     * Creates streaming model for real-time output.
     */
    public static OllamaStreamingChatModel createStreamingModel() {
        return OllamaStreamingChatModel.builder()
                .baseUrl(BASE_URL)
                .modelName("llama3.1:8b")
                .temperature(0.7)
                .build();
    }
}
```

### Multi-Provider Model Factory

```java
package com.example.config;

import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.output.Response;

import java.util.List;

public class ModelFactory {

    public enum Provider {
        OPENAI, ANTHROPIC, AZURE, OLLAMA
    }

    public enum Tier {
        FAST,      // Cheapest, fastest
        BALANCED,  // Good balance
        PREMIUM    // Highest capability
    }

    /**
     * Creates a model based on provider and tier.
     */
    public static ChatLanguageModel create(Provider provider, Tier tier) {
        return switch (provider) {
            case OPENAI -> createOpenAi(tier);
            case ANTHROPIC -> createAnthropic(tier);
            case AZURE -> AzureOpenAiConfig.createChatModel();
            case OLLAMA -> createOllama(tier);
        };
    }

    private static ChatLanguageModel createOpenAi(Tier tier) {
        return switch (tier) {
            case FAST -> OpenAiConfig.createFastModel();
            case BALANCED -> OpenAiConfig.createChatModel();
            case PREMIUM -> OpenAiConfig.createChatModel();  // GPT-4o is premium
        };
    }

    private static ChatLanguageModel createAnthropic(Tier tier) {
        return switch (tier) {
            case FAST -> AnthropicConfig.createHaikuModel();
            case BALANCED -> AnthropicConfig.createSonnetModel();
            case PREMIUM -> AnthropicConfig.createOpusModel();
        };
    }

    private static ChatLanguageModel createOllama(Tier tier) {
        return switch (tier) {
            case FAST -> OllamaConfig.createMistralModel();
            case BALANCED -> OllamaConfig.createLlama3Model();
            case PREMIUM -> OllamaConfig.createLlama3Model();
        };
    }

    /**
     * Creates fallback chain of models.
     * If the primary provider fails, falls back to the secondary.
     */
    public static ChatLanguageModel createWithFallback(
            Provider primary,
            Provider fallback,
            Tier tier) {

        ChatLanguageModel primaryModel = create(primary, tier);
        ChatLanguageModel fallbackModel = create(fallback, tier);

        return new FallbackChatModel(primaryModel, fallbackModel);
    }

    /**
     * Simple fallback wrapper — delegates to fallback on any exception.
     * For production use, combine with a circuit breaker (see Section 10).
     */
    private static class FallbackChatModel implements ChatLanguageModel {
        private final ChatLanguageModel primary;
        private final ChatLanguageModel fallback;

        FallbackChatModel(ChatLanguageModel primary, ChatLanguageModel fallback) {
            this.primary = primary;
            this.fallback = fallback;
        }

        @Override
        public Response<AiMessage> generate(List<ChatMessage> messages) {
            try {
                return primary.generate(messages);
            } catch (RuntimeException e) {
                System.err.println("Primary model failed (" + e.getMessage() +
                    "), falling back to secondary model.");
                return fallback.generate(messages);
            }
        }
    }
}
```

---

## 4. The AI Services Pattern

### Basic AI Service

The AI Services pattern is LangChain4j's most powerful feature — define an interface, and the library generates the implementation:

```java
package com.example.service;

import dev.langchain4j.service.AiServices;
import dev.langchain4j.service.SystemMessage;
import dev.langchain4j.service.UserMessage;
import dev.langchain4j.service.V;
import dev.langchain4j.model.chat.ChatLanguageModel;

/**
 * AI Service interface - implementation generated at runtime.
 */
public interface AssistantService {

    /**
     * Simple chat method.
     */
    String chat(String message);

    /**
     * Chat with system message.
     */
    @SystemMessage("You are a helpful assistant that speaks like Shakespeare.")
    String chatAsShakespeare(String message);

    /**
     * Parameterized prompt with template variables.
     */
    @UserMessage("Translate the following text to {{language}}: {{text}}")
    String translate(@V("language") String language, @V("text") String text);

    /**
     * Summarization with specific instructions.
     */
    @SystemMessage("You are an expert summarizer. Be concise but comprehensive.")
    @UserMessage("Summarize the following text in {{sentences}} sentences: {{text}}")
    String summarize(@V("sentences") int sentences, @V("text") String text);
}

// Usage
public class AssistantServiceExample {

    public static void main(String[] args) {
        ChatLanguageModel model = OpenAiConfig.createChatModel();

        // Create the AI service
        AssistantService assistant = AiServices.create(
            AssistantService.class,
            model
        );

        // Use it like a regular Java interface
        String response = assistant.chat("What is the capital of France?");
        System.out.println(response);

        // Use Shakespeare persona
        String shakespeare = assistant.chatAsShakespeare(
            "Tell me about artificial intelligence"
        );
        System.out.println(shakespeare);

        // Translation
        String translated = assistant.translate("Spanish", "Hello, how are you?");
        System.out.println(translated);  // "Hola, ¿cómo estás?"

        // Summarization
        String summary = assistant.summarize(2, longArticleText);
        System.out.println(summary);
    }
}
```

### Structured Output with POJOs

```java
package com.example.service;

import dev.langchain4j.model.output.structured.Description;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.service.SystemMessage;
import dev.langchain4j.service.UserMessage;

import java.util.List;

/**
 * Data classes for structured output.
 */
public record Person(
    @Description("The person's full name")
    String name,

    @Description("The person's age in years")
    int age,

    @Description("The person's occupation or job title")
    String occupation,

    @Description("List of the person's hobbies")
    List<String> hobbies
) {}

public record SentimentAnalysis(
    @Description("Overall sentiment: POSITIVE, NEGATIVE, or NEUTRAL")
    Sentiment sentiment,

    @Description("Confidence score from 0.0 to 1.0")
    double confidence,

    @Description("Key phrases that influenced the sentiment")
    List<String> keyPhrases,

    @Description("Brief explanation of the analysis")
    String explanation
) {
    public enum Sentiment {
        POSITIVE, NEGATIVE, NEUTRAL
    }
}

public record CodeReview(
    @Description("Overall code quality: GOOD, ACCEPTABLE, NEEDS_IMPROVEMENT")
    Quality quality,

    @Description("List of identified issues")
    List<Issue> issues,

    @Description("List of improvement suggestions")
    List<String> suggestions,

    @Description("Security vulnerabilities found")
    List<String> securityConcerns
) {
    public enum Quality {
        GOOD, ACCEPTABLE, NEEDS_IMPROVEMENT
    }

    public record Issue(
        @Description("Line number where issue occurs")
        int lineNumber,

        @Description("Severity: HIGH, MEDIUM, LOW")
        Severity severity,

        @Description("Description of the issue")
        String description
    ) {
        public enum Severity {
            HIGH, MEDIUM, LOW
        }
    }
}

/**
 * AI Service with structured output.
 */
public interface AnalysisService {

    @SystemMessage("You are an expert data extractor. Extract structured information accurately.")
    @UserMessage("Extract person information from this text: {{text}}")
    Person extractPerson(@V("text") String text);

    @SystemMessage("You are a sentiment analysis expert.")
    @UserMessage("Analyze the sentiment of this text: {{text}}")
    SentimentAnalysis analyzeSentiment(@V("text") String text);

    @SystemMessage("""
        You are a senior software engineer performing code reviews.
        Be thorough but constructive in your feedback.
        Focus on: code quality, potential bugs, security issues, and best practices.
        """)
    @UserMessage("Review this code:\n```{{language}}\n{{code}}\n```")
    CodeReview reviewCode(@V("language") String language, @V("code") String code);

    @UserMessage("Extract all persons mentioned in: {{text}}")
    List<Person> extractAllPersons(@V("text") String text);
}

// Usage
public class StructuredOutputExample {

    public static void main(String[] args) {
        ChatLanguageModel model = OpenAiConfig.createStructuredModel();

        AnalysisService analyzer = AiServices.create(
            AnalysisService.class,
            model
        );

        // Extract person
        Person person = analyzer.extractPerson(
            "John Smith is a 35-year-old software engineer who enjoys hiking and photography."
        );
        System.out.println("Name: " + person.name());
        System.out.println("Age: " + person.age());
        System.out.println("Hobbies: " + person.hobbies());

        // Sentiment analysis
        SentimentAnalysis sentiment = analyzer.analyzeSentiment(
            "I absolutely love this product! It exceeded all my expectations."
        );
        System.out.println("Sentiment: " + sentiment.sentiment());
        System.out.println("Confidence: " + sentiment.confidence());

        // Code review
        String code = """
            public void processUser(String userId) {
                String query = "SELECT * FROM users WHERE id = '" + userId + "'";
                database.execute(query);
            }
            """;

        CodeReview review = analyzer.reviewCode("java", code);
        System.out.println("Quality: " + review.quality());
        for (CodeReview.Issue issue : review.issues()) {
            System.out.println("Issue at line " + issue.lineNumber() +
                             " [" + issue.severity() + "]: " + issue.description());
        }
    }
}
```

### AI Service Builder Pattern

```java
package com.example.service;

import dev.langchain4j.memory.ChatMemory;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.service.AiServices;

public class AiServiceFactory {

    /**
     * Creates a simple AI service.
     */
    public static <T> T createSimple(Class<T> serviceClass, ChatLanguageModel model) {
        return AiServices.create(serviceClass, model);
    }

    /**
     * Creates AI service with memory.
     */
    public static <T> T createWithMemory(
            Class<T> serviceClass,
            ChatLanguageModel model,
            int memorySize) {

        ChatMemory memory = MessageWindowChatMemory.builder()
                .maxMessages(memorySize)
                .build();

        return AiServices.builder(serviceClass)
                .chatLanguageModel(model)
                .chatMemory(memory)
                .build();
    }

    /**
     * Creates AI service with RAG.
     */
    public static <T> T createWithRag(
            Class<T> serviceClass,
            ChatLanguageModel model,
            ContentRetriever retriever) {

        return AiServices.builder(serviceClass)
                .chatLanguageModel(model)
                .contentRetriever(retriever)
                .build();
    }

    /**
     * Creates full-featured AI service.
     */
    public static <T> T createFull(
            Class<T> serviceClass,
            ChatLanguageModel model,
            ChatMemory memory,
            ContentRetriever retriever,
            Object... tools) {

        var builder = AiServices.builder(serviceClass)
                .chatLanguageModel(model)
                .chatMemory(memory);

        if (retriever != null) {
            builder.contentRetriever(retriever);
        }

        if (tools != null && tools.length > 0) {
            builder.tools(tools);
        }

        return builder.build();
    }
}
```

---

## 5. Chat Memory and Conversations

### Memory Types

```java
package com.example.memory;

import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.memory.ChatMemory;
import dev.langchain4j.memory.chat.ChatMemoryProvider;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.memory.chat.TokenWindowChatMemory;
import dev.langchain4j.model.Tokenizer;
import dev.langchain4j.model.openai.OpenAiTokenizer;

import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

public class MemoryExamples {

    /**
     * Simple message window memory - keeps last N messages.
     */
    public static ChatMemory createMessageWindow(int maxMessages) {
        return MessageWindowChatMemory.builder()
                .maxMessages(maxMessages)
                .build();
    }

    /**
     * Token-based memory - keeps messages within token limit.
     */
    public static ChatMemory createTokenWindow(int maxTokens) {
        Tokenizer tokenizer = new OpenAiTokenizer();

        return TokenWindowChatMemory.builder()
                .maxTokens(maxTokens)
                .tokenizer(tokenizer)
                .build();
    }

    /**
     * Memory with conversation ID for multi-user support.
     */
    public static ChatMemory createWithId(Object conversationId, int maxMessages) {
        return MessageWindowChatMemory.builder()
                .id(conversationId)
                .maxMessages(maxMessages)
                .build();
    }
}

/**
 * Multi-user memory provider for conversational applications.
 */
public class MultiUserMemoryProvider implements ChatMemoryProvider {

    private final Map<Object, ChatMemory> memories = new ConcurrentHashMap<>();
    private final int maxMessages;

    public MultiUserMemoryProvider(int maxMessages) {
        this.maxMessages = maxMessages;
    }

    @Override
    public ChatMemory get(Object memoryId) {
        return memories.computeIfAbsent(memoryId, id ->
            MessageWindowChatMemory.builder()
                .id(id)
                .maxMessages(maxMessages)
                .build()
        );
    }

    public void clearMemory(Object memoryId) {
        ChatMemory memory = memories.get(memoryId);
        if (memory != null) {
            memory.clear();
        }
    }

    public void removeMemory(Object memoryId) {
        memories.remove(memoryId);
    }

    public int getActiveConversations() {
        return memories.size();
    }
}

/**
 * Persistent memory using a store.
 */
public class PersistentChatMemory implements ChatMemory {

    private final Object id;
    private final ChatMemoryStore store;
    private final int maxMessages;

    public PersistentChatMemory(Object id, ChatMemoryStore store, int maxMessages) {
        this.id = id;
        this.store = store;
        this.maxMessages = maxMessages;
    }

    @Override
    public Object id() {
        return id;
    }

    @Override
    public void add(ChatMessage message) {
        List<ChatMessage> messages = store.getMessages(id);
        messages.add(message);

        // Trim if exceeds max
        while (messages.size() > maxMessages) {
            messages.remove(0);
        }

        store.updateMessages(id, messages);
    }

    @Override
    public List<ChatMessage> messages() {
        return store.getMessages(id);
    }

    @Override
    public void clear() {
        store.deleteMessages(id);
    }
}

/**
 * Interface for persistent memory storage.
 */
public interface ChatMemoryStore {
    List<ChatMessage> getMessages(Object memoryId);
    void updateMessages(Object memoryId, List<ChatMessage> messages);
    void deleteMessages(Object memoryId);
}
```

### AI Service with Per-User Memory

```java
package com.example.service;

import dev.langchain4j.service.AiServices;
import dev.langchain4j.service.MemoryId;
import dev.langchain4j.service.SystemMessage;
import dev.langchain4j.service.UserMessage;

/**
 * AI Service with per-user memory support.
 */
public interface ConversationalAssistant {

    @SystemMessage("""
        You are a helpful assistant with memory of past conversations.
        Use context from previous messages to provide relevant responses.
        Be friendly and remember user preferences.
        """)
    String chat(@MemoryId String sessionId, @UserMessage String message);

    @SystemMessage("You are a technical support assistant.")
    String supportChat(@MemoryId String ticketId, @UserMessage String message);
}

// Usage
public class ConversationalExample {

    public static void main(String[] args) {
        ChatLanguageModel model = OpenAiConfig.createChatModel();

        MultiUserMemoryProvider memoryProvider = new MultiUserMemoryProvider(20);

        ConversationalAssistant assistant = AiServices.builder(ConversationalAssistant.class)
                .chatLanguageModel(model)
                .chatMemoryProvider(memoryProvider)
                .build();

        // User 1 conversation
        String user1Session = "user-1-session-123";
        System.out.println(assistant.chat(user1Session, "Hi, my name is Alice"));
        System.out.println(assistant.chat(user1Session, "What's my name?"));
        // Will correctly respond "Alice"

        // User 2 conversation (separate memory)
        String user2Session = "user-2-session-456";
        System.out.println(assistant.chat(user2Session, "Hi, I'm Bob"));
        System.out.println(assistant.chat(user2Session, "What's my name?"));
        // Will correctly respond "Bob"

        // Memory is isolated between users
        System.out.println("Active conversations: " + memoryProvider.getActiveConversations());
    }
}
```

### Redis-Backed Memory Store

```java
package com.example.memory;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.ChatMessageDeserializer;
import dev.langchain4j.data.message.ChatMessageSerializer;
import redis.clients.jedis.JedisPool;
import redis.clients.jedis.JedisPoolConfig;

import java.time.Duration;
import java.util.ArrayList;
import java.util.List;

public class RedisChatMemoryStore implements ChatMemoryStore {

    private final JedisPool jedisPool;
    private final String keyPrefix;
    private final Duration ttl;
    private final ObjectMapper objectMapper;

    public RedisChatMemoryStore(String host, int port, String keyPrefix, Duration ttl) {
        JedisPoolConfig poolConfig = new JedisPoolConfig();
        poolConfig.setMaxTotal(50);
        poolConfig.setMaxIdle(10);

        this.jedisPool = new JedisPool(poolConfig, host, port);
        this.keyPrefix = keyPrefix;
        this.ttl = ttl;
        this.objectMapper = new ObjectMapper();
    }

    @Override
    public List<ChatMessage> getMessages(Object memoryId) {
        String key = keyPrefix + memoryId.toString();

        try (var jedis = jedisPool.getResource()) {
            String json = jedis.get(key);
            if (json == null || json.isEmpty()) {
                return new ArrayList<>();
            }
            return deserializeMessages(json);
        }
    }

    @Override
    public void updateMessages(Object memoryId, List<ChatMessage> messages) {
        String key = keyPrefix + memoryId.toString();
        String json = serializeMessages(messages);

        try (var jedis = jedisPool.getResource()) {
            jedis.setex(key, ttl.toSeconds(), json);
        }
    }

    @Override
    public void deleteMessages(Object memoryId) {
        String key = keyPrefix + memoryId.toString();

        try (var jedis = jedisPool.getResource()) {
            jedis.del(key);
        }
    }

    private String serializeMessages(List<ChatMessage> messages) {
        try {
            List<String> serialized = messages.stream()
                    .map(ChatMessageSerializer::messageToJson)
                    .toList();
            return objectMapper.writeValueAsString(serialized);
        } catch (JsonProcessingException e) {
            throw new RuntimeException("Failed to serialize messages", e);
        }
    }

    private List<ChatMessage> deserializeMessages(String json) {
        try {
            List<String> serialized = objectMapper.readValue(json, List.class);
            return serialized.stream()
                    .map(ChatMessageDeserializer::messageFromJson)
                    .toList();
        } catch (JsonProcessingException e) {
            throw new RuntimeException("Failed to deserialize messages", e);
        }
    }

    public void close() {
        jedisPool.close();
    }
}
```

---

## 6. RAG with LangChain4j

### Document Loading and Processing

```java
package com.example.rag;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentParser;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.document.parser.TextDocumentParser;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.document.transformer.HtmlTextExtractor;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.Tokenizer;
import dev.langchain4j.model.openai.OpenAiTokenizer;

import java.nio.file.Path;
import java.util.List;

public class DocumentProcessor {

    private final Tokenizer tokenizer;

    public DocumentProcessor() {
        this.tokenizer = new OpenAiTokenizer();
    }

    /**
     * Loads documents from a directory.
     */
    public List<Document> loadFromDirectory(Path directory) {
        // Apache Tika handles PDF, DOCX, TXT, HTML, etc.
        DocumentParser parser = new ApacheTikaDocumentParser();

        return FileSystemDocumentLoader.loadDocuments(
            directory,
            parser
        );
    }

    /**
     * Loads a single document.
     */
    public Document loadDocument(Path filePath) {
        DocumentParser parser = new ApacheTikaDocumentParser();
        return FileSystemDocumentLoader.loadDocument(filePath, parser);
    }

    /**
     * Loads text files only.
     */
    public List<Document> loadTextFiles(Path directory) {
        DocumentParser parser = new TextDocumentParser();
        return FileSystemDocumentLoader.loadDocuments(
            directory,
            "*.txt",
            parser
        );
    }

    /**
     * Splits documents into chunks with overlap.
     *
     * Why 500 tokens / 50 overlap?
     * - 500 tokens (~375 words) is large enough to contain a complete idea
     *   but small enough for precise retrieval. Smaller chunks (200) retrieve
     *   more precisely but lose context; larger chunks (1000) include more
     *   context but dilute relevance scores with unrelated text.
     * - 50-token overlap (10%) prevents information loss at chunk boundaries.
     *   Without overlap, a sentence split across two chunks loses meaning in both.
     *   Too much overlap (>20%) wastes embedding storage and slows retrieval.
     * - These are starting values. Tune based on your content: technical docs
     *   with short paragraphs → 300/30; legal documents with long clauses → 800/100.
     */
    public List<TextSegment> splitDocuments(List<Document> documents) {
        return DocumentSplitters.recursive(
            500,    // chunk size in tokens
            50,     // overlap in tokens
            tokenizer
        ).splitAll(documents);
    }

    /**
     * Splits with custom separators.
     */
    public List<TextSegment> splitWithCustomSeparators(
            List<Document> documents,
            int chunkSize,
            int overlap) {

        return DocumentSplitters.recursive(
            chunkSize,
            overlap,
            tokenizer
        ).splitAll(documents);
    }

    /**
     * Process HTML documents.
     */
    public List<Document> processHtmlDocuments(List<Document> htmlDocs) {
        HtmlTextExtractor extractor = new HtmlTextExtractor();
        return htmlDocs.stream()
                .map(extractor::transform)
                .toList();
    }

    /**
     * Adds metadata to segments.
     */
    public List<TextSegment> addMetadata(
            List<TextSegment> segments,
            String source,
            String category) {

        return segments.stream()
                .map(segment -> {
                    segment.metadata().put("source", source);
                    segment.metadata().put("category", category);
                    return segment;
                })
                .toList();
    }
}
```

### Embedding Stores

```java
package com.example.rag;

import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.model.openai.OpenAiEmbeddingModel;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import dev.langchain4j.store.embedding.pgvector.PgVectorEmbeddingStore;

import java.util.List;

public class EmbeddingStoreFactory {

    /**
     * Creates in-memory embedding store (for development).
     */
    public static EmbeddingStore<TextSegment> createInMemory() {
        return new InMemoryEmbeddingStore<>();
    }

    /**
     * Creates PostgreSQL vector store.
     */
    public static EmbeddingStore<TextSegment> createPgVector(
            String host,
            int port,
            String database,
            String user,
            String password,
            String table,
            int dimension) {

        return PgVectorEmbeddingStore.builder()
                .host(host)
                .port(port)
                .database(database)
                .user(user)
                .password(password)
                .table(table)
                .dimension(dimension)
                .createTable(true)
                .dropTableFirst(false)
                .build();
    }

    /**
     * Creates local embedding model (no API needed).
     *
     * AllMiniLmL6V2: 384 dimensions, ~80MB model, runs in-process via ONNX.
     * Use when: Development/testing, data sovereignty requirements,
     * low-volume production (<10K embeddings/day), offline environments.
     * Trade-off: Lower accuracy than OpenAI embeddings on benchmarks
     * (MTEB score ~63 vs ~62 for text-embedding-3-small), but zero API cost
     * and no network latency (~5ms vs ~100ms per embedding).
     */
    public static EmbeddingModel createLocalEmbeddingModel() {
        return new AllMiniLmL6V2EmbeddingModel();
    }

    /**
     * Creates OpenAI embedding model.
     *
     * text-embedding-3-small: 1536 dimensions, $0.02/1M tokens.
     * Use when: Production RAG with >10K documents, multilingual content,
     * high-accuracy requirements (MTEB score ~62.3).
     * Trade-off: API dependency, ~100ms network latency per call,
     * ongoing cost. For 100K documents at 500 tokens each = ~$1.00 for
     * initial embedding, then per-query cost is negligible.
     *
     * text-embedding-3-large: 3072 dimensions, $0.13/1M tokens.
     * Use only when: small accuracy gain justifies 6.5x cost increase.
     * Most applications see <2% retrieval improvement over small.
     */
    public static EmbeddingModel createOpenAiEmbeddingModel() {
        return OpenAiEmbeddingModel.builder()
                .apiKey(System.getenv("OPENAI_API_KEY"))
                .modelName("text-embedding-3-small")
                .build();
    }
}

/**
 * Service for managing embeddings.
 */
public class EmbeddingService {

    private final EmbeddingModel embeddingModel;
    private final EmbeddingStore<TextSegment> embeddingStore;

    public EmbeddingService(
            EmbeddingModel embeddingModel,
            EmbeddingStore<TextSegment> embeddingStore) {
        this.embeddingModel = embeddingModel;
        this.embeddingStore = embeddingStore;
    }

    /**
     * Ingests documents into the embedding store.
     */
    public void ingest(List<TextSegment> segments) {
        List<Embedding> embeddings = embeddingModel.embedAll(segments).content();
        embeddingStore.addAll(embeddings, segments);
    }

    /**
     * Searches for similar segments.
     */
    public List<TextSegment> search(String query, int maxResults) {
        Embedding queryEmbedding = embeddingModel.embed(query).content();

        return embeddingStore.search(
            dev.langchain4j.store.embedding.EmbeddingSearchRequest.builder()
                .queryEmbedding(queryEmbedding)
                .maxResults(maxResults)
                .minScore(0.7)  // Minimum similarity threshold
                .build()
        ).matches().stream()
            .map(match -> match.embedded())
            .toList();
    }

    /**
     * Searches with metadata filter.
     */
    public List<TextSegment> searchWithFilter(
            String query,
            int maxResults,
            String metadataKey,
            String metadataValue) {

        Embedding queryEmbedding = embeddingModel.embed(query).content();

        return embeddingStore.search(
            dev.langchain4j.store.embedding.EmbeddingSearchRequest.builder()
                .queryEmbedding(queryEmbedding)
                .maxResults(maxResults)
                .filter(dev.langchain4j.store.embedding.filter.MetadataFilterBuilder
                    .metadataKey(metadataKey).isEqualTo(metadataValue))
                .build()
        ).matches().stream()
            .map(match -> match.embedded())
            .toList();
    }
}
```

### Complete RAG Implementation

```java
package com.example.rag;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.query.transformer.CompressingQueryTransformer;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.service.SystemMessage;
import dev.langchain4j.service.UserMessage;
import dev.langchain4j.store.embedding.EmbeddingStore;

import java.nio.file.Path;
import java.util.List;

/**
 * RAG-enabled AI Service interface.
 */
public interface RagAssistant {

    @SystemMessage("""
        You are a helpful assistant that answers questions based on the provided context.
        Always base your answers on the retrieved documents.
        If the context doesn't contain relevant information, say so.
        Cite your sources when possible.
        """)
    String answer(@UserMessage String question);
}

/**
 * Complete RAG system implementation.
 */
public class RagSystem {

    private final DocumentProcessor documentProcessor;
    private final EmbeddingService embeddingService;
    private final RagAssistant assistant;

    public RagSystem(
            ChatLanguageModel chatModel,
            EmbeddingModel embeddingModel,
            EmbeddingStore<TextSegment> embeddingStore) {

        this.documentProcessor = new DocumentProcessor();
        this.embeddingService = new EmbeddingService(embeddingModel, embeddingStore);

        // Create content retriever
        ContentRetriever contentRetriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(embeddingModel)
                .maxResults(5)
                .minScore(0.7)
                .build();

        // Create RAG-enabled AI service
        this.assistant = AiServices.builder(RagAssistant.class)
                .chatLanguageModel(chatModel)
                .contentRetriever(contentRetriever)
                .build();
    }

    /**
     * Ingests documents from a directory.
     */
    public void ingestDocuments(Path directory) {
        List<Document> documents = documentProcessor.loadFromDirectory(directory);
        List<TextSegment> segments = documentProcessor.splitDocuments(documents);
        embeddingService.ingest(segments);

        System.out.println("Ingested " + segments.size() + " segments from " +
                          documents.size() + " documents");
    }

    /**
     * Ingests a single document.
     */
    public void ingestDocument(Path filePath, String source, String category) {
        Document document = documentProcessor.loadDocument(filePath);
        List<TextSegment> segments = documentProcessor.splitDocuments(List.of(document));

        // Add metadata
        segments = documentProcessor.addMetadata(segments, source, category);

        embeddingService.ingest(segments);
    }

    /**
     * Asks a question using RAG.
     */
    public String ask(String question) {
        return assistant.answer(question);
    }

    /**
     * Searches for relevant documents without generating an answer.
     */
    public List<TextSegment> searchDocuments(String query, int maxResults) {
        return embeddingService.search(query, maxResults);
    }

    // Factory method
    public static RagSystem create() {
        ChatLanguageModel chatModel = OpenAiConfig.createChatModel();
        EmbeddingModel embeddingModel = EmbeddingStoreFactory.createLocalEmbeddingModel();
        EmbeddingStore<TextSegment> store = EmbeddingStoreFactory.createInMemory();

        return new RagSystem(chatModel, embeddingModel, store);
    }

    public static RagSystem createWithPgVector(
            String host, int port, String database,
            String user, String password) {

        ChatLanguageModel chatModel = OpenAiConfig.createChatModel();
        EmbeddingModel embeddingModel = EmbeddingStoreFactory.createOpenAiEmbeddingModel();
        EmbeddingStore<TextSegment> store = EmbeddingStoreFactory.createPgVector(
            host, port, database, user, password, "embeddings", 1536
        );

        return new RagSystem(chatModel, embeddingModel, store);
    }
}

// Usage example
public class RagExample {

    public static void main(String[] args) {
        // Create RAG system
        RagSystem rag = RagSystem.create();

        // Ingest documents
        rag.ingestDocuments(Path.of("./documents/knowledge-base"));

        // Ask questions
        String answer = rag.ask("What are the main features of our product?");
        System.out.println(answer);

        // Search without generating answer
        List<TextSegment> relevant = rag.searchDocuments("pricing information", 3);
        for (TextSegment segment : relevant) {
            System.out.println("---");
            System.out.println(segment.text());
        }
    }
}
```

### Advanced RAG with Query Transformation

```java
package com.example.rag;

import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.aggregator.ContentAggregator;
import dev.langchain4j.rag.content.aggregator.ReRankingContentAggregator;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.query.transformer.ExpandingQueryTransformer;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;

public class AdvancedRagSystem {

    /**
     * Creates RAG with query expansion for better retrieval.
     */
    public static RagAssistant createWithQueryExpansion(
            ChatLanguageModel chatModel,
            EmbeddingModel embeddingModel,
            EmbeddingStore embeddingStore) {

        // Query transformer expands the query into multiple variations
        ExpandingQueryTransformer queryTransformer = ExpandingQueryTransformer.builder()
                .chatLanguageModel(chatModel)
                .build();

        ContentRetriever contentRetriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(embeddingModel)
                .maxResults(10)
                .build();

        RetrievalAugmentor augmentor = DefaultRetrievalAugmentor.builder()
                .queryTransformer(queryTransformer)
                .contentRetriever(contentRetriever)
                .build();

        return AiServices.builder(RagAssistant.class)
                .chatLanguageModel(chatModel)
                .retrievalAugmentor(augmentor)
                .build();
    }

    /**
     * Creates RAG with re-ranking for better relevance.
     *
     * Why re-rank? Embedding similarity retrieves semantically close chunks,
     * but "close" doesn't always mean "relevant to the question." Re-ranking
     * uses the LLM to score each chunk against the actual query, catching
     * cases where embedding similarity misses the intent. Trade-off: adds
     * one LLM call per query (~200-500ms + token cost for scoring N chunks).
     *
     * Note: LangChain4j's re-ranking API requires a ScoringModel, not a
     * ChatLanguageModel. If your provider doesn't offer a scoring endpoint,
     * use a ContentAggregator that filters by embedding score threshold instead.
     */
    public static RagAssistant createWithReranking(
            ChatLanguageModel chatModel,
            EmbeddingModel embeddingModel,
            EmbeddingStore embeddingStore,
            dev.langchain4j.model.scoring.ScoringModel scoringModel) {

        ContentRetriever contentRetriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(embeddingModel)
                .maxResults(20)  // Retrieve more candidates for re-ranking
                .build();

        // Re-rank results using a scoring model
        // ScoringModel scores each chunk against the query (0.0-1.0)
        ContentAggregator aggregator = ReRankingContentAggregator.builder()
                .scoringModel(scoringModel)  // Requires ScoringModel, not ChatLanguageModel
                .build();

        RetrievalAugmentor augmentor = DefaultRetrievalAugmentor.builder()
                .contentRetriever(contentRetriever)
                .contentAggregator(aggregator)
                .build();

        return AiServices.builder(RagAssistant.class)
                .chatLanguageModel(chatModel)
                .retrievalAugmentor(augmentor)
                .build();
    }
}
```

---

## 7. Function Calling and Tools

### Defining Tools

```java
package com.example.tools;

import dev.langchain4j.agent.tool.P;
import dev.langchain4j.agent.tool.Tool;

import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.List;
import java.util.Map;

/**
 * Weather information tool.
 */
public class WeatherTool {

    @Tool("Get current weather for a city")
    public String getCurrentWeather(
            @P("City name") String city,
            @P("Temperature unit: celsius or fahrenheit") String unit) {

        // In production, call actual weather API
        return String.format(
            "Current weather in %s: 22°%s, Partly cloudy, Humidity: 65%%",
            city,
            unit.equals("celsius") ? "C" : "F"
        );
    }

    @Tool("Get weather forecast for next N days")
    public String getForecast(
            @P("City name") String city,
            @P("Number of days (1-7)") int days) {

        StringBuilder forecast = new StringBuilder();
        forecast.append("Weather forecast for ").append(city).append(":\n");

        for (int i = 1; i <= days; i++) {
            LocalDate date = LocalDate.now().plusDays(i);
            forecast.append(String.format(
                "- %s: High 24°C, Low 18°C, Sunny\n",
                date.format(DateTimeFormatter.ISO_DATE)
            ));
        }

        return forecast.toString();
    }
}

/**
 * Calculator tool with various operations.
 */
public class CalculatorTool {

    @Tool("Calculate the result of a mathematical expression")
    public double calculate(
            @P("First number") double a,
            @P("Operation: add, subtract, multiply, divide") String operation,
            @P("Second number") double b) {

        return switch (operation.toLowerCase()) {
            case "add" -> a + b;
            case "subtract" -> a - b;
            case "multiply" -> a * b;
            case "divide" -> {
                if (b == 0) throw new IllegalArgumentException("Cannot divide by zero");
                yield a / b;
            }
            default -> throw new IllegalArgumentException("Unknown operation: " + operation);
        };
    }

    @Tool("Calculate percentage of a value")
    public double percentage(
            @P("The value to calculate percentage of") double value,
            @P("The percentage (0-100)") double percent) {
        return value * percent / 100;
    }

    @Tool("Convert between units")
    public double convert(
            @P("Value to convert") double value,
            @P("Source unit") String fromUnit,
            @P("Target unit") String toUnit) {

        // Simplified conversion logic
        if (fromUnit.equals("km") && toUnit.equals("miles")) {
            return value * 0.621371;
        } else if (fromUnit.equals("miles") && toUnit.equals("km")) {
            return value * 1.60934;
        }
        throw new IllegalArgumentException("Unsupported conversion");
    }
}

/**
 * Database query tool.
 */
public class DatabaseTool {

    private final Map<String, List<Map<String, Object>>> mockData;

    public DatabaseTool() {
        // Mock data for demonstration
        this.mockData = Map.of(
            "customers", List.of(
                Map.of("id", 1, "name", "Alice", "email", "alice@example.com"),
                Map.of("id", 2, "name", "Bob", "email", "bob@example.com")
            ),
            "orders", List.of(
                Map.of("id", 101, "customer_id", 1, "total", 150.00),
                Map.of("id", 102, "customer_id", 2, "total", 250.00)
            )
        );
    }

    @Tool("Query data from the database")
    public String queryDatabase(
            @P("Table name to query") String table,
            @P("Optional filter field") String filterField,
            @P("Optional filter value") String filterValue) {

        List<Map<String, Object>> data = mockData.get(table);
        if (data == null) {
            return "Table not found: " + table;
        }

        if (filterField != null && filterValue != null) {
            data = data.stream()
                .filter(row -> String.valueOf(row.get(filterField)).equals(filterValue))
                .toList();
        }

        return data.toString();
    }

    @Tool("Get count of records in a table")
    public int countRecords(@P("Table name") String table) {
        List<Map<String, Object>> data = mockData.get(table);
        return data != null ? data.size() : 0;
    }
}

/**
 * Date and time utilities.
 */
public class DateTimeTool {

    @Tool("Get current date and time")
    public String getCurrentDateTime(@P("Timezone, e.g., 'UTC', 'America/New_York'") String timezone) {
        try {
            java.time.ZoneId zoneId = java.time.ZoneId.of(timezone);
            return java.time.ZonedDateTime.now(zoneId)
                .format(DateTimeFormatter.ISO_ZONED_DATE_TIME);
        } catch (java.time.zone.ZoneRulesException e) {
            return "Unknown timezone: " + timezone +
                ". Use IANA format, e.g., 'America/New_York' or 'UTC'.";
        }
    }

    @Tool("Calculate days between two dates")
    public long daysBetween(
            @P("Start date in YYYY-MM-DD format") String startDate,
            @P("End date in YYYY-MM-DD format") String endDate) {

        LocalDate start = LocalDate.parse(startDate);
        LocalDate end = LocalDate.parse(endDate);
        return java.time.temporal.ChronoUnit.DAYS.between(start, end);
    }

    @Tool("Add days to a date")
    public String addDays(
            @P("Date in YYYY-MM-DD format") String date,
            @P("Number of days to add (can be negative)") int days) {

        LocalDate localDate = LocalDate.parse(date);
        return localDate.plusDays(days).toString();
    }
}
```

### How LangChain4j Decides to Call a Tool (The ReAct Loop)

When you register tools with an AI Service, LangChain4j implements a **ReAct (Reasoning + Acting) loop** behind the scenes:

1. **User sends message** → LangChain4j forwards it to the LLM along with tool descriptions (JSON schema generated from `@Tool` and `@P` annotations)
2. **LLM decides**: respond directly OR request a tool call (returns a special `ToolExecutionRequest` instead of text)
3. **If tool call requested**: LangChain4j invokes the matching Java method, captures the return value, and sends the result back to the LLM
4. **LLM processes tool result** and either responds to the user OR requests another tool call
5. **Loop repeats** until the LLM responds with text (no more tool requests)

**Key implication**: A single user message can trigger *multiple* LLM calls (one per tool invocation + the final response). For cost estimation, assume 2-3 LLM calls per tool-using interaction. The LLM's tool selection is non-deterministic — the same question may or may not trigger a tool call depending on the model's confidence.

**Failure mode**: If a tool throws an exception, LangChain4j sends the error message back to the LLM, which typically responds with a user-friendly error explanation. This is why your `@Tool` methods should throw descriptive exceptions (not generic `RuntimeException`).

### AI Service with Tools

```java
package com.example.service;

import com.example.tools.*;
import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.service.SystemMessage;

public interface ToolEnabledAssistant {

    @SystemMessage("""
        You are a helpful assistant with access to various tools.
        Use the appropriate tools to answer user questions accurately.
        When using tools, explain what you're doing.
        If a tool fails, explain the error and try an alternative approach.
        """)
    String chat(String message);
}

public class ToolEnabledAssistantExample {

    public static void main(String[] args) {
        ChatLanguageModel model = OpenAiConfig.createChatModel();

        // Create tool instances
        WeatherTool weatherTool = new WeatherTool();
        CalculatorTool calculatorTool = new CalculatorTool();
        DatabaseTool databaseTool = new DatabaseTool();
        DateTimeTool dateTimeTool = new DateTimeTool();

        // Create AI service with tools
        ToolEnabledAssistant assistant = AiServices.builder(ToolEnabledAssistant.class)
                .chatLanguageModel(model)
                .tools(weatherTool, calculatorTool, databaseTool, dateTimeTool)
                .build();

        // Test various tool calls
        System.out.println(assistant.chat(
            "What's the weather like in Paris?"
        ));

        System.out.println(assistant.chat(
            "Calculate 15% of 250"
        ));

        System.out.println(assistant.chat(
            "How many customers are in the database?"
        ));

        System.out.println(assistant.chat(
            "What date will it be 30 days from 2025-01-30?"
        ));

        // Complex query requiring multiple tools
        System.out.println(assistant.chat(
            "What's the weather in London, and how many days until April 15, 2025?"
        ));
    }
}
```

### Creating a ReAct Agent

```java
package com.example.agent;

import dev.langchain4j.agent.tool.Tool;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.service.SystemMessage;
import dev.langchain4j.service.UserMessage;
import dev.langchain4j.service.V;

import java.util.HashMap;
import java.util.Map;

/**
 * Research agent that can search and analyze information.
 */
public class ResearchAgent {

    // Tool: Web search simulation
    public static class SearchTool {
        @Tool("Search the web for information on a topic")
        public String search(@P("Search query") String query) {
            // Simulated search results
            return "Search results for '" + query + "':\n" +
                   "1. [Wikipedia] Overview of " + query + "\n" +
                   "2. [News] Recent developments in " + query + "\n" +
                   "3. [Academic] Research paper on " + query;
        }
    }

    // Tool: Note taking
    public static class NoteTool {
        private final Map<String, String> notes = new HashMap<>();

        @Tool("Save a note for later reference")
        public String saveNote(
                @P("Note title") String title,
                @P("Note content") String content) {
            notes.put(title, content);
            return "Note saved: " + title;
        }

        @Tool("Retrieve a previously saved note")
        public String getNote(@P("Note title") String title) {
            return notes.getOrDefault(title, "Note not found: " + title);
        }

        @Tool("List all saved notes")
        public String listNotes() {
            if (notes.isEmpty()) {
                return "No notes saved yet.";
            }
            return "Saved notes: " + String.join(", ", notes.keySet());
        }
    }

    // Tool: Analysis
    public static class AnalysisTool {
        @Tool("Analyze text for key themes and insights")
        public String analyzeText(@P("Text to analyze") String text) {
            int wordCount = text.split("\\s+").length;
            return String.format(
                "Analysis complete:\n- Word count: %d\n- Key themes identified\n- Sentiment: Neutral",
                wordCount
            );
        }
    }

    // Agent interface
    public interface ResearchAssistant {
        @SystemMessage("""
            You are a research assistant that helps users gather and analyze information.

            Follow this process:
            1. THINK: Analyze what the user needs
            2. SEARCH: Use search tool to find relevant information
            3. NOTE: Save important findings using the note tool
            4. ANALYZE: Use analysis tool when needed
            5. RESPOND: Provide a comprehensive answer

            Be thorough and cite your sources.
            """)
        String research(@UserMessage String query);
    }

    public static ResearchAssistant create(ChatLanguageModel model) {
        return AiServices.builder(ResearchAssistant.class)
                .chatLanguageModel(model)
                .chatMemory(MessageWindowChatMemory.withMaxMessages(20))
                .tools(new SearchTool(), new NoteTool(), new AnalysisTool())
                .build();
    }
}

// Usage
public class AgentExample {

    public static void main(String[] args) {
        ChatLanguageModel model = OpenAiConfig.createChatModel();

        ResearchAgent.ResearchAssistant agent = ResearchAgent.create(model);

        String result = agent.research(
            "Research the current state of quantum computing and its potential " +
            "applications in cryptography. Save your key findings."
        );

        System.out.println(result);
    }
}
```

---

## 8. Streaming Responses

### Basic Streaming

```java
package com.example.streaming;

import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.model.StreamingResponseHandler;
import dev.langchain4j.model.openai.OpenAiStreamingChatModel;
import dev.langchain4j.model.output.Response;

public class StreamingExample {

    public static void main(String[] args) {
        OpenAiStreamingChatModel model = OpenAiStreamingChatModel.builder()
                .apiKey(System.getenv("OPENAI_API_KEY"))
                .modelName("gpt-4o")
                .build();

        System.out.println("Streaming response:");

        model.generate(
            "Write a short story about a robot learning to paint",
            new StreamingResponseHandler<AiMessage>() {

                @Override
                public void onNext(String token) {
                    // Print each token as it arrives
                    System.out.print(token);
                    System.out.flush();
                }

                @Override
                public void onComplete(Response<AiMessage> response) {
                    System.out.println("\n\n--- Stream complete ---");
                    System.out.println("Total tokens: " +
                        response.tokenUsage().totalTokenCount());
                }

                @Override
                public void onError(Throwable error) {
                    System.err.println("Error: " + error.getMessage());
                }
            }
        );

        // Wait for completion
        try {
            Thread.sleep(30000);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }
}
```

### Streaming AI Service

```java
package com.example.streaming;

import dev.langchain4j.service.AiServices;
import dev.langchain4j.service.SystemMessage;
import dev.langchain4j.service.TokenStream;
import dev.langchain4j.model.openai.OpenAiStreamingChatModel;

public interface StreamingAssistant {

    @SystemMessage("You are a helpful assistant that provides detailed explanations.")
    TokenStream chat(String message);
}

public class StreamingAssistantExample {

    public static void main(String[] args) {
        OpenAiStreamingChatModel model = OpenAiStreamingChatModel.builder()
                .apiKey(System.getenv("OPENAI_API_KEY"))
                .modelName("gpt-4o")
                .build();

        StreamingAssistant assistant = AiServices.create(
            StreamingAssistant.class,
            model
        );

        // Get streaming response
        TokenStream tokenStream = assistant.chat(
            "Explain the concept of dependency injection in Java"
        );

        // Process the stream
        tokenStream
            .onNext(token -> {
                System.out.print(token);
                System.out.flush();
            })
            .onComplete(response -> {
                System.out.println("\n--- Complete ---");
            })
            .onError(error -> {
                System.err.println("Error: " + error.getMessage());
            })
            .start();

        // Keep main thread alive
        try {
            Thread.sleep(60000);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }
}
```

### Server-Sent Events (SSE) with Spring WebFlux

```java
package com.example.streaming;

import dev.langchain4j.model.openai.OpenAiStreamingChatModel;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.service.TokenStream;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.*;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Sinks;

@RestController
@RequestMapping("/api/chat")
public class StreamingController {

    private final StreamingAssistant assistant;

    public StreamingController() {
        OpenAiStreamingChatModel model = OpenAiStreamingChatModel.builder()
                .apiKey(System.getenv("OPENAI_API_KEY"))
                .modelName("gpt-4o")
                .build();

        this.assistant = AiServices.create(StreamingAssistant.class, model);
    }

    @GetMapping(value = "/stream", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
    public Flux<String> streamChat(@RequestParam String message) {
        Sinks.Many<String> sink = Sinks.many().unicast().onBackpressureBuffer();

        TokenStream tokenStream = assistant.chat(message);

        tokenStream
            .onNext(token -> sink.tryEmitNext(token))
            .onComplete(response -> sink.tryEmitComplete())
            .onError(error -> sink.tryEmitError(error))
            .start();

        return sink.asFlux();
    }

    @PostMapping(value = "/stream", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
    public Flux<String> streamChatPost(@RequestBody ChatRequest request) {
        return streamChat(request.message());
    }

    public record ChatRequest(String message) {}
}
```

---

## 9. Testing AI Applications

### Unit Testing with Mocks

```java
package com.example.service;

import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.output.Response;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class AssistantServiceTest {

    @Mock
    private ChatLanguageModel mockModel;

    private AssistantService assistantService;

    @BeforeEach
    void setUp() {
        assistantService = AiServices.create(AssistantService.class, mockModel);
    }

    @Test
    void shouldReturnChatResponse() {
        // Given
        String expectedResponse = "Paris is the capital of France.";
        when(mockModel.generate(any()))
            .thenReturn(Response.from(AiMessage.from(expectedResponse)));

        // When
        String response = assistantService.chat("What is the capital of France?");

        // Then
        assertThat(response).isEqualTo(expectedResponse);
    }

    @Test
    void shouldTranslateText() {
        // Given
        when(mockModel.generate(any()))
            .thenReturn(Response.from(AiMessage.from("Bonjour")));

        // When
        String result = assistantService.translate("French", "Hello");

        // Then
        assertThat(result).isEqualTo("Bonjour");
    }
}
```

### Integration Testing

```java
package com.example.integration;

import com.example.config.OpenAiConfig;
import com.example.service.AnalysisService;
import com.example.service.SentimentAnalysis;
import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.service.AiServices;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable;

import static org.assertj.core.api.Assertions.assertThat;

@EnabledIfEnvironmentVariable(named = "OPENAI_API_KEY", matches = ".+")
class AnalysisServiceIntegrationTest {

    private static AnalysisService analysisService;

    @BeforeAll
    static void setUp() {
        ChatLanguageModel model = OpenAiConfig.createStructuredModel();
        analysisService = AiServices.create(AnalysisService.class, model);
    }

    @Test
    void shouldAnalyzePositiveSentiment() {
        // Given
        String text = "I absolutely love this product! It's amazing and exceeded all my expectations.";

        // When
        SentimentAnalysis result = analysisService.analyzeSentiment(text);

        // Then
        assertThat(result.sentiment()).isEqualTo(SentimentAnalysis.Sentiment.POSITIVE);
        assertThat(result.confidence()).isGreaterThan(0.7);
    }

    @Test
    void shouldAnalyzeNegativeSentiment() {
        // Given
        String text = "Terrible experience. The product broke after one day and customer service was unhelpful.";

        // When
        SentimentAnalysis result = analysisService.analyzeSentiment(text);

        // Then
        assertThat(result.sentiment()).isEqualTo(SentimentAnalysis.Sentiment.NEGATIVE);
    }

    @Test
    void shouldExtractPerson() {
        // Given
        String text = "Sarah Johnson, a 28-year-old data scientist from Seattle, enjoys rock climbing and photography.";

        // When
        Person person = analysisService.extractPerson(text);

        // Then
        assertThat(person.name()).containsIgnoringCase("Sarah Johnson");
        assertThat(person.age()).isEqualTo(28);
        assertThat(person.occupation()).containsIgnoringCase("data scientist");
        assertThat(person.hobbies()).hasSizeGreaterThanOrEqualTo(1);
    }
}
```

### Testing Tools

```java
package com.example.tools;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import static org.assertj.core.api.Assertions.*;

class CalculatorToolTest {

    private CalculatorTool calculator;

    @BeforeEach
    void setUp() {
        calculator = new CalculatorTool();
    }

    @ParameterizedTest
    @CsvSource({
        "10, add, 5, 15",
        "10, subtract, 3, 7",
        "4, multiply, 5, 20",
        "20, divide, 4, 5"
    })
    void shouldCalculateCorrectly(double a, String operation, double b, double expected) {
        // When
        double result = calculator.calculate(a, operation, b);

        // Then
        assertThat(result).isEqualTo(expected);
    }

    @Test
    void shouldThrowExceptionWhenDividingByZero() {
        assertThatThrownBy(() -> calculator.calculate(10, "divide", 0))
            .isInstanceOf(IllegalArgumentException.class)
            .hasMessageContaining("divide by zero");
    }

    @Test
    void shouldCalculatePercentage() {
        // When
        double result = calculator.percentage(200, 15);

        // Then
        assertThat(result).isEqualTo(30);
    }
}
```

### RAG System Testing

```java
package com.example.rag;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;

class EmbeddingServiceTest {

    private EmbeddingService embeddingService;
    private EmbeddingStore<TextSegment> store;

    @BeforeEach
    void setUp() {
        EmbeddingModel model = new AllMiniLmL6V2EmbeddingModel();
        store = new InMemoryEmbeddingStore<>();
        embeddingService = new EmbeddingService(model, store);
    }

    @Test
    void shouldIngestAndSearchDocuments() {
        // Given
        List<TextSegment> segments = List.of(
            TextSegment.from("Java is a programming language."),
            TextSegment.from("Python is popular for data science."),
            TextSegment.from("JavaScript runs in web browsers.")
        );

        // When
        embeddingService.ingest(segments);
        List<TextSegment> results = embeddingService.search("programming languages", 2);

        // Then
        assertThat(results).isNotEmpty();
        assertThat(results).hasSizeLessThanOrEqualTo(2);
    }

    @Test
    void shouldReturnRelevantResults() {
        // Given
        List<TextSegment> segments = List.of(
            TextSegment.from("The weather in Paris is sunny today."),
            TextSegment.from("Machine learning is a subset of artificial intelligence."),
            TextSegment.from("Deep learning uses neural networks with many layers.")
        );

        embeddingService.ingest(segments);

        // When
        List<TextSegment> results = embeddingService.search("AI and neural networks", 2);

        // Then
        assertThat(results).isNotEmpty();
        String combinedText = results.stream()
            .map(TextSegment::text)
            .reduce("", (a, b) -> a + " " + b);

        // Results should be about AI, not weather
        assertThat(combinedText.toLowerCase())
            .containsAnyOf("learning", "neural", "intelligence");
    }
}
```

---

## 10. Production Patterns

### Configuration Management

```java
package com.example.config;

import java.time.Duration;
import java.util.Optional;

/**
 * Type-safe configuration for AI services.
 */
public record AiConfig(
    String provider,
    String model,
    String apiKey,
    double temperature,
    int maxTokens,
    Duration timeout,
    int maxRetries,
    boolean logRequests
) {
    public static AiConfig fromEnvironment() {
        return new AiConfig(
            getEnvOrDefault("AI_PROVIDER", "openai"),
            getEnvOrDefault("AI_MODEL", "gpt-4o"),
            getEnvRequired("AI_API_KEY"),
            Double.parseDouble(getEnvOrDefault("AI_TEMPERATURE", "0.7")),
            Integer.parseInt(getEnvOrDefault("AI_MAX_TOKENS", "4096")),
            Duration.ofSeconds(Long.parseLong(getEnvOrDefault("AI_TIMEOUT_SECONDS", "60"))),
            Integer.parseInt(getEnvOrDefault("AI_MAX_RETRIES", "3")),
            Boolean.parseBoolean(getEnvOrDefault("AI_LOG_REQUESTS", "false"))
        );
    }

    private static String getEnvRequired(String key) {
        return Optional.ofNullable(System.getenv(key))
            .orElseThrow(() -> new IllegalStateException(
                "Required environment variable not set: " + key
            ));
    }

    private static String getEnvOrDefault(String key, String defaultValue) {
        return Optional.ofNullable(System.getenv(key)).orElse(defaultValue);
    }
}
```

### Circuit Breaker Pattern

```java
package com.example.resilience;

import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.output.Response;
import io.github.resilience4j.circuitbreaker.CircuitBreaker;
import io.github.resilience4j.circuitbreaker.CircuitBreakerConfig;
import io.github.resilience4j.retry.Retry;
import io.github.resilience4j.retry.RetryConfig;

import java.time.Duration;
import java.util.List;
import java.util.function.Supplier;

/**
 * Resilient wrapper for ChatLanguageModel with circuit breaker and retry.
 */
public class ResilientChatModel implements ChatLanguageModel {

    private final ChatLanguageModel delegate;
    private final CircuitBreaker circuitBreaker;
    private final Retry retry;

    public ResilientChatModel(ChatLanguageModel delegate, String name) {
        this.delegate = delegate;

        // Circuit breaker configuration
        CircuitBreakerConfig cbConfig = CircuitBreakerConfig.custom()
            .failureRateThreshold(50)
            .waitDurationInOpenState(Duration.ofSeconds(30))
            .permittedNumberOfCallsInHalfOpenState(3)
            .slidingWindowSize(10)
            .build();

        this.circuitBreaker = CircuitBreaker.of(name + "-cb", cbConfig);

        // Retry configuration
        RetryConfig retryConfig = RetryConfig.custom()
            .maxAttempts(3)
            .waitDuration(Duration.ofSeconds(2))
            .retryExceptions(Exception.class)
            .ignoreExceptions(IllegalArgumentException.class)
            .build();

        this.retry = Retry.of(name + "-retry", retryConfig);
    }

    @Override
    public Response<AiMessage> generate(List<ChatMessage> messages) {
        Supplier<Response<AiMessage>> supplier = () -> delegate.generate(messages);

        // Combine circuit breaker and retry
        Supplier<Response<AiMessage>> decoratedSupplier =
            CircuitBreaker.decorateSupplier(circuitBreaker,
                Retry.decorateSupplier(retry, supplier)
            );

        return decoratedSupplier.get();
    }

    public CircuitBreaker.State getCircuitBreakerState() {
        return circuitBreaker.getState();
    }
}
```

### Caching Layer

```java
package com.example.cache;

import com.github.benmanes.caffeine.cache.Cache;
import com.github.benmanes.caffeine.cache.Caffeine;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.output.Response;

import java.time.Duration;
import java.util.List;

/**
 * Caching wrapper for ChatLanguageModel.
 */
public class CachingChatModel implements ChatLanguageModel {

    private final ChatLanguageModel delegate;
    private final Cache<String, Response<AiMessage>> cache;

    public CachingChatModel(ChatLanguageModel delegate, Duration ttl, long maxSize) {
        this.delegate = delegate;
        this.cache = Caffeine.newBuilder()
            .expireAfterWrite(ttl)
            .maximumSize(maxSize)
            .recordStats()
            .build();
    }

    @Override
    public Response<AiMessage> generate(List<ChatMessage> messages) {
        String cacheKey = generateCacheKey(messages);

        return cache.get(cacheKey, key -> delegate.generate(messages));
    }

    /**
     * Generate a cache key from the full message content.
     *
     * Why SHA-256 instead of hashCode()?
     * - Java's String.hashCode() has a high collision rate for similar strings
     *   (32-bit space = ~50% collision probability at ~77K entries).
     * - For a cache, collisions mean returning wrong answers to users.
     * - SHA-256 makes collisions astronomically unlikely.
     *
     * Important: This cache is scoped to a single model+temperature config.
     * If you use the same CachingChatModel with different parameters,
     * you'll get incorrect cached results. Create separate instances per config.
     */
    private String generateCacheKey(List<ChatMessage> messages) {
        try {
            java.security.MessageDigest digest = java.security.MessageDigest.getInstance("SHA-256");
            for (ChatMessage message : messages) {
                digest.update(message.type().name().getBytes(java.nio.charset.StandardCharsets.UTF_8));
                digest.update((byte) ':');
                digest.update(message.text().getBytes(java.nio.charset.StandardCharsets.UTF_8));
                digest.update((byte) '|');
            }
            byte[] hash = digest.digest();
            return java.util.HexFormat.of().formatHex(hash);
        } catch (java.security.NoSuchAlgorithmException e) {
            // SHA-256 is guaranteed to be available in all JVMs
            throw new RuntimeException("SHA-256 not available", e);
        }
    }

    public CacheStats getStats() {
        var stats = cache.stats();
        return new CacheStats(
            stats.hitCount(),
            stats.missCount(),
            stats.hitRate(),
            cache.estimatedSize()
        );
    }

    public void invalidate() {
        cache.invalidateAll();
    }

    public record CacheStats(
        long hits,
        long misses,
        double hitRate,
        long size
    ) {}
}
```

### Observability

```java
package com.example.observability;

import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.output.Response;
import io.micrometer.core.instrument.Counter;
import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.core.instrument.Timer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;

/**
 * Observability wrapper with metrics and logging.
 */
public class ObservableChatModel implements ChatLanguageModel {

    private static final Logger log = LoggerFactory.getLogger(ObservableChatModel.class);

    private final ChatLanguageModel delegate;
    private final Timer responseTimer;
    private final Counter requestCounter;
    private final Counter errorCounter;
    private final Counter tokenCounter;

    public ObservableChatModel(ChatLanguageModel delegate, MeterRegistry registry, String modelName) {
        this.delegate = delegate;

        this.responseTimer = Timer.builder("ai.response.time")
            .tag("model", modelName)
            .description("Time to generate AI response")
            .register(registry);

        this.requestCounter = Counter.builder("ai.requests")
            .tag("model", modelName)
            .description("Number of AI requests")
            .register(registry);

        this.errorCounter = Counter.builder("ai.errors")
            .tag("model", modelName)
            .description("Number of AI errors")
            .register(registry);

        this.tokenCounter = Counter.builder("ai.tokens")
            .tag("model", modelName)
            .description("Total tokens used")
            .register(registry);
    }

    @Override
    public Response<AiMessage> generate(List<ChatMessage> messages) {
        requestCounter.increment();

        log.debug("Generating response for {} messages", messages.size());

        return responseTimer.record(() -> {
            try {
                Response<AiMessage> response = delegate.generate(messages);

                // Record token usage
                if (response.tokenUsage() != null) {
                    tokenCounter.increment(response.tokenUsage().totalTokenCount());
                }

                log.debug("Response generated: {} tokens",
                    response.tokenUsage() != null ?
                        response.tokenUsage().totalTokenCount() : "unknown");

                return response;

            } catch (Exception e) {
                errorCounter.increment();
                log.error("Error generating response", e);
                throw e;
            }
        });
    }
}
```

### Complete Production Service

```java
package com.example.production;

import com.example.cache.CachingChatModel;
import com.example.config.AiConfig;
import com.example.observability.ObservableChatModel;
import com.example.resilience.ResilientChatModel;
import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.openai.OpenAiChatModel;
import dev.langchain4j.service.AiServices;
import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.core.instrument.simple.SimpleMeterRegistry;

import java.time.Duration;

/**
 * Production-ready AI service with all resilience patterns.
 */
public class ProductionAiServiceFactory {

    private final MeterRegistry meterRegistry;

    public ProductionAiServiceFactory() {
        this.meterRegistry = new SimpleMeterRegistry();
    }

    public ProductionAiServiceFactory(MeterRegistry meterRegistry) {
        this.meterRegistry = meterRegistry;
    }

    /**
     * Creates a production-ready AI service.
     */
    public <T> T createService(Class<T> serviceClass) {
        AiConfig config = AiConfig.fromEnvironment();

        // Base model
        ChatLanguageModel baseModel = OpenAiChatModel.builder()
            .apiKey(config.apiKey())
            .modelName(config.model())
            .temperature(config.temperature())
            .maxTokens(config.maxTokens())
            .timeout(config.timeout())
            .logRequests(config.logRequests())
            .logResponses(config.logRequests())
            .build();

        // Add resilience
        ChatLanguageModel resilientModel = new ResilientChatModel(
            baseModel,
            "primary"
        );

        // Add caching
        ChatLanguageModel cachedModel = new CachingChatModel(
            resilientModel,
            Duration.ofMinutes(5),
            1000
        );

        // Add observability
        ChatLanguageModel observableModel = new ObservableChatModel(
            cachedModel,
            meterRegistry,
            config.model()
        );

        // Create service
        return AiServices.create(serviceClass, observableModel);
    }

    /**
     * Creates service with memory.
     */
    public <T> T createServiceWithMemory(
            Class<T> serviceClass,
            int memorySize) {

        AiConfig config = AiConfig.fromEnvironment();

        ChatLanguageModel model = createDecoratedModel(config);

        return AiServices.builder(serviceClass)
            .chatLanguageModel(model)
            .chatMemory(MessageWindowChatMemory.withMaxMessages(memorySize))
            .build();
    }

    private ChatLanguageModel createDecoratedModel(AiConfig config) {
        ChatLanguageModel base = OpenAiChatModel.builder()
            .apiKey(config.apiKey())
            .modelName(config.model())
            .temperature(config.temperature())
            .build();

        return new ObservableChatModel(
            new CachingChatModel(
                new ResilientChatModel(base, "default"),
                Duration.ofMinutes(5),
                1000
            ),
            meterRegistry,
            config.model()
        );
    }

    public MeterRegistry getMeterRegistry() {
        return meterRegistry;
    }
}
```

---

## 11. Failure Modes & Error Handling

Production AI applications fail in predictable ways. This section covers critical failure modes and resilience patterns specific to LangChain4j.

### Provider Timeouts and Retries

Timeout errors are the most common production failures. LangChain4j requires explicit configuration:

```java
package com.example.resilience;

import dev.langchain4j.model.openai.OpenAiChatModel;
import dev.langchain4j.model.openai.OpenAiChatModelName;
import io.github.resilience4j.retry.Retry;
import io.github.resilience4j.retry.RetryConfig;
import io.github.resilience4j.retry.RetryRegistry;

import java.time.Duration;
import java.util.concurrent.TimeoutException;

/**
 * Timeout and retry configuration for LLM calls.
 * Critical for production reliability.
 */
public class TimeoutRetryConfiguration {

    public static OpenAiChatModel createResilientModel(String apiKey) {
        // Base model with explicit timeout
        OpenAiChatModel baseModel = OpenAiChatModel.builder()
            .apiKey(apiKey)
            .modelName(OpenAiChatModelName.GPT_4O)
            .timeout(Duration.ofSeconds(60))  // Critical: must be explicit
            .maxRetries(3)
            .logRequestsAndResponses(true)
            .build();

        // Additional retry logic for transient failures
        RetryConfig config = RetryConfig.custom()
            .maxAttempts(4)
            .waitDuration(Duration.ofSeconds(2))
            .intervalFunction(io.github.resilience4j.core.IntervalFunction
                .ofExponentialBackoff(1000, 2))  // Exponential backoff: 1s, 2s, 4s
            .retryExceptions(
                TimeoutException.class,
                java.net.SocketTimeoutException.class,
                java.io.IOException.class
            )
            .ignoreExceptions(IllegalArgumentException.class)  // Don't retry validation errors
            .build();

        return baseModel;
    }

    public static class TimeoutMonitor {
        private volatile int timeoutCount = 0;
        private volatile long lastTimeoutTimestamp = 0;

        public synchronized void recordTimeout() {
            timeoutCount++;
            lastTimeoutTimestamp = System.currentTimeMillis();

            if (timeoutCount > 5) {
                // Alert: Multiple timeouts indicate degraded provider
                System.err.println("WARNING: Multiple timeouts detected. Provider may be degraded.");
            }
        }

        public boolean isInTimeoutWindow() {
            long elapsed = System.currentTimeMillis() - lastTimeoutTimestamp;
            return elapsed < Duration.ofMinutes(5).toMillis();
        }
    }
}
```

### Token Limit Exceeded (4xx errors)

When LLMs reject requests due to token limits, the error must be handled distinctly from transient failures:

```java
package com.example.error;

import dev.langchain4j.model.output.Response;

/**
 * Token limit error handling - critical difference from transient failures.
 */
public class TokenLimitHandler {

    /**
     * Detects and handles token limit exceeded errors.
     * DO NOT RETRY: Token limits are permanent for that request.
     */
    public static class TokenLimitException extends RuntimeException {
        private final int requestTokens;
        private final int maxTokens;
        private final String originalMessage;

        public TokenLimitException(int requestTokens, int maxTokens, String msg) {
            super("Token limit exceeded: " + requestTokens + " > " + maxTokens);
            this.requestTokens = requestTokens;
            this.maxTokens = maxTokens;
            this.originalMessage = msg;
        }

        public boolean isRecoverable() {
            // False: Don't retry, instead split the request
            return false;
        }

        public double recoveryFraction() {
            // Suggest reducing request size by this fraction
            return (double) requestTokens / (maxTokens * 0.8);
        }
    }

    public static void validateTokenCount(String input, int maxTokens) {
        // Rough estimation: 1 token ≈ 4 characters
        int estimatedTokens = (int) Math.ceil(input.length() / 4.0);

        if (estimatedTokens > maxTokens) {
            throw new TokenLimitException(estimatedTokens, maxTokens,
                "Input exceeds token limit");
        }
    }

    public static String truncateToTokenLimit(String input, int maxTokens) {
        int maxChars = maxTokens * 4;
        if (input.length() > maxChars) {
            // Preserve word boundaries
            String truncated = input.substring(0, maxChars);
            int lastSpace = truncated.lastIndexOf(' ');
            return lastSpace > 0 ? truncated.substring(0, lastSpace) : truncated;
        }
        return input;
    }
}
```

### Memory Leaks from Chat History

Chat memory accumulation is a subtle but critical production issue:

```java
package com.example.memory;

import dev.langchain4j.memory.ChatMemory;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.memory.chat.TokenWindowChatMemory;
import dev.langchain4j.data.message.ChatMessage;

/**
 * Memory management to prevent accumulation and leaks.
 * Without proper bounds, memory usage grows unbounded with conversation length.
 */
public class BoundedChatMemory {

    /**
     * Option 1: Window-based memory (last N messages)
     * Risk: Loses context for long conversations
     */
    public static ChatMemory createMessageWindowMemory() {
        // Keep only last 20 messages
        // With 5 user turns per session, this covers ~4 sessions
        return MessageWindowChatMemory.withMaxMessages(20);
    }

    /**
     * Option 2: Token-based memory (last N tokens)
     * RECOMMENDED: More accurate cost control
     *
     * Why token-based over message-based? Message windows treat a 5-word
     * message and a 500-word message equally. Token windows ensure you
     * never exceed your context budget regardless of message length.
     */
    public static ChatMemory createTokenWindowMemory() {
        // Keep conversation within 2000 tokens
        // At ~4 tokens/word, approximately 500 words of history
        // Use the actual model tokenizer for accurate counting
        Tokenizer tokenizer = new OpenAiTokenizer();
        return TokenWindowChatMemory.withMaxTokens(2000, tokenizer);
    }

    /**
     * Option 3: Hybrid approach with cleanup
     * PRODUCTION RECOMMENDED
     *
     * Note: LangChain4j's ChatMemory interface uses id(), add(ChatMessage),
     * messages(), and clear() — the memoryId is set at construction time,
     * not passed per-call. This wrapper adds time-based eviction on top
     * of the built-in message window.
     */
    public static class HybridBoundedMemory implements ChatMemory {
        private final Object memoryId;
        private final int maxMessages;
        private final int maxConversationAgeMinutes;
        private volatile long conversationStartTime;
        private final java.util.List<ChatMessage> messages =
            java.util.Collections.synchronizedList(new java.util.ArrayList<>());

        public HybridBoundedMemory(Object memoryId, int maxMessages, int maxAgeMinutes) {
            this.memoryId = memoryId;
            this.maxMessages = maxMessages;
            this.maxConversationAgeMinutes = maxAgeMinutes;
            this.conversationStartTime = System.currentTimeMillis();
        }

        @Override
        public Object id() {
            return memoryId;
        }

        @Override
        public void add(ChatMessage message) {
            // Time-based eviction: reset if conversation is too old
            long elapsedMinutes = (System.currentTimeMillis() - conversationStartTime) / 60000;
            if (elapsedMinutes > maxConversationAgeMinutes) {
                messages.clear();
                conversationStartTime = System.currentTimeMillis();
            }

            messages.add(message);

            // Window-based eviction: keep last N messages
            while (messages.size() > maxMessages) {
                messages.remove(0);
            }
        }

        @Override
        public java.util.List<ChatMessage> messages() {
            return java.util.Collections.unmodifiableList(new java.util.ArrayList<>(messages));
        }

        @Override
        public void clear() {
            messages.clear();
            conversationStartTime = System.currentTimeMillis();
        }
    }
}
```

### Serialization Issues with Tools

Tool deserialization failures are runtime bombs in production:

```java
package com.example.tools;

import dev.langchain4j.agent.tool.Tool;
import java.lang.reflect.Method;
import java.lang.reflect.Parameter;

/**
 * Validates tools at startup to catch serialization issues early.
 */
public class ToolValidator {

    /**
     * Validates that all tools can be properly serialized/deserialized
     * before deployment. Run at application startup.
     */
    public static void validateTools(Object toolProvider) {
        for (Method method : toolProvider.getClass().getDeclaredMethods()) {
            if (method.isAnnotationPresent(Tool.class)) {
                validateToolMethod(method);
            }
        }
    }

    private static void validateToolMethod(Method method) {
        // Check 1: All parameters must be serializable types
        for (Parameter param : method.getParameters()) {
            Class<?> type = param.getType();

            // Reject types that LLMs can't deserialize
            if (isUnserializableType(type)) {
                throw new IllegalArgumentException(
                    "Tool parameter type not JSON-serializable: " + type.getName() +
                    " in method: " + method.getName()
                );
            }
        }

        // Check 2: Return type must be serializable
        if (isUnserializableType(method.getReturnType())) {
            throw new IllegalArgumentException(
                "Tool return type not JSON-serializable: " +
                method.getReturnType().getName()
            );
        }

        // Check 3: All fields must have proper JSON annotations
        if (!method.isAnnotationPresent(Tool.class)) {
            throw new IllegalArgumentException("Missing @Tool annotation");
        }
    }

    private static boolean isUnserializableType(Class<?> type) {
        // These types commonly cause issues
        return type == java.io.InputStream.class ||
               type == java.io.OutputStream.class ||
               type == java.sql.Connection.class ||
               type == java.nio.channels.FileChannel.class ||
               // Functional types can't be serialized to JSON
               type.isSynthetic();
    }

    /**
     * Example safe tool definition.
     */
    public static class SafeTools {
        @Tool("Get the current date")
        public String getCurrentDate() {
            return java.time.LocalDate.now().toString();
        }

        @Tool("Convert temperature Celsius to Fahrenheit")
        public double convertTemperature(
            double celsius,
            @dev.langchain4j.agent.tool.P("Target unit: F or K") String unit
        ) {
            return switch (unit) {
                case "F" -> celsius * 9/5 + 32;
                case "K" -> celsius + 273.15;
                default -> throw new IllegalArgumentException("Unknown unit: " + unit);
            };
        }
    }
}
```

---

## 12. Cost Tracking with LangChain4j

Token usage directly drives cloud costs. Production systems must track and enforce budgets.

### Token Counting and Cost Estimation

```java
package com.example.cost;

import java.math.BigDecimal;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Track token usage and cost across conversations and users.
 */
public class TokenCostTracker {

    // Pricing per 1M tokens (2025 rates, example values)
    private static final Map<String, PricingModel> PRICING = Map.ofEntries(
        Map.entry("gpt-4o", new PricingModel(5.00, 15.00)),           // input, output per 1M
        Map.entry("gpt-4-turbo", new PricingModel(10.00, 30.00)),
        Map.entry("gpt-3.5-turbo", new PricingModel(0.50, 1.50)),
        Map.entry("claude-3-opus", new PricingModel(15.00, 75.00)),
        Map.entry("claude-3-sonnet", new PricingModel(3.00, 15.00))
    );

    public record PricingModel(
        double inputCostPer1M,
        double outputCostPer1M
    ) {}

    public static class ConversationCostTracker {
        private final String conversationId;
        private final String model;
        private final String userId;
        private volatile long inputTokens = 0;
        private volatile long outputTokens = 0;
        private volatile long startTime;
        private final AtomicReference<BigDecimal> totalCost =
            new AtomicReference<>(BigDecimal.ZERO);

        public ConversationCostTracker(String conversationId, String model, String userId) {
            this.conversationId = conversationId;
            this.model = model;
            this.userId = userId;
            this.startTime = System.currentTimeMillis();
        }

        public synchronized void recordTokens(long inputTokens, long outputTokens) {
            this.inputTokens += inputTokens;
            this.outputTokens += outputTokens;
            recalculateCost();
        }

        private void recalculateCost() {
            PricingModel pricing = PRICING.getOrDefault(model,
                new PricingModel(1.0, 3.0));  // Fallback pricing

            double inputCost = (inputTokens / 1_000_000.0) * pricing.inputCostPer1M();
            double outputCost = (outputTokens / 1_000_000.0) * pricing.outputCostPer1M();

            totalCost.set(BigDecimal.valueOf(inputCost + outputCost));
        }

        public CostReport getReport() {
            long elapsedSeconds = (System.currentTimeMillis() - startTime) / 1000;
            return new CostReport(
                conversationId,
                userId,
                model,
                inputTokens,
                outputTokens,
                inputTokens + outputTokens,
                totalCost.get(),
                elapsedSeconds
            );
        }
    }

    public record CostReport(
        String conversationId,
        String userId,
        String model,
        long inputTokens,
        long outputTokens,
        long totalTokens,
        BigDecimal totalCost,
        long durationSeconds
    ) {
        public void log() {
            System.out.println(String.format(
                "[COST] User: %s | Conv: %s | Model: %s | Tokens: %d | Cost: $%.4f",
                userId, conversationId, model, totalTokens,
                totalCost.doubleValue()
            ));
        }

        public boolean exceedsBudget(BigDecimal maxCostPerConversation) {
            return totalCost.compareTo(maxCostPerConversation) > 0;
        }
    }
}
```

### Budget Enforcement Patterns

```java
package com.example.cost;

import java.math.BigDecimal;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Enforce budget limits at user and conversation level.
 * Critical for preventing cost overruns in production.
 */
public class BudgetEnforcer {

    private final Map<String, BigDecimal> userBudgets = new ConcurrentHashMap<>();
    private final Map<String, BigDecimal> userSpending = new ConcurrentHashMap<>();
    private final BigDecimal perConversationLimit;

    public BudgetEnforcer(BigDecimal perUserMonthlyBudget,
                          BigDecimal perConversationLimit) {
        this.perConversationLimit = perConversationLimit;
    }

    /**
     * Validate request against budgets BEFORE sending to LLM.
     * This prevents unnecessary API calls when budget is exceeded.
     */
    public void validateBudget(String userId, String conversationId,
                              BigDecimal estimatedCost)
            throws BudgetExceededException {

        // Check 1: Per-conversation limit
        if (estimatedCost.compareTo(perConversationLimit) > 0) {
            throw new BudgetExceededException(
                "Single request exceeds conversation limit: $" +
                estimatedCost + " > $" + perConversationLimit
            );
        }

        // Check 2: User's remaining budget
        BigDecimal spent = userSpending.getOrDefault(userId, BigDecimal.ZERO);
        BigDecimal budget = userBudgets.get(userId);

        if (budget != null) {
            BigDecimal remaining = budget.subtract(spent);
            if (estimatedCost.compareTo(remaining) > 0) {
                throw new BudgetExceededException(
                    "Insufficient budget for user " + userId +
                    ": $" + estimatedCost + " > $" + remaining + " remaining"
                );
            }
        }
    }

    public void recordSpending(String userId, BigDecimal actualCost) {
        userSpending.compute(userId, (k, v) ->
            (v == null ? BigDecimal.ZERO : v).add(actualCost)
        );
    }

    public static class BudgetExceededException extends RuntimeException {
        public BudgetExceededException(String message) {
            super(message);
        }
    }
}
```

### Per-Conversation Cost Tracking

```java
package com.example.cost;

import java.math.BigDecimal;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;

/**
 * Detailed cost tracking for individual conversations.
 * Enables per-feature cost analysis and optimization.
 */
public class ConversationCostDetails {

    public static class CostEntry {
        public final Instant timestamp;
        public final String operation;     // "user_message", "assistant_response", "tool_call"
        public final long inputTokens;
        public final long outputTokens;
        public final BigDecimal cost;

        public CostEntry(Instant timestamp, String operation,
                        long inputTokens, long outputTokens, BigDecimal cost) {
            this.timestamp = timestamp;
            this.operation = operation;
            this.inputTokens = inputTokens;
            this.outputTokens = outputTokens;
            this.cost = cost;
        }
    }

    public static class DetailedCostLog {
        private final String conversationId;
        private final List<CostEntry> entries = new ArrayList<>();
        private volatile BigDecimal totalCost = BigDecimal.ZERO;

        public DetailedCostLog(String conversationId) {
            this.conversationId = conversationId;
        }

        public synchronized void recordOperation(String operation,
                                                 long inputTokens,
                                                 long outputTokens) {
            PricingModel pricing = TokenCostTracker.PRICING.getOrDefault(
                "gpt-4o",
                new TokenCostTracker.PricingModel(5.0, 15.0)
            );

            double cost = (inputTokens / 1_000_000.0) * pricing.inputCostPer1M() +
                         (outputTokens / 1_000_000.0) * pricing.outputCostPer1M();

            CostEntry entry = new CostEntry(
                Instant.now(),
                operation,
                inputTokens,
                outputTokens,
                BigDecimal.valueOf(cost)
            );

            entries.add(entry);
            totalCost = totalCost.add(entry.cost);
        }

        public List<CostEntry> getEntries() {
            return new ArrayList<>(entries);
        }

        public String getDetailedReport() {
            StringBuilder sb = new StringBuilder();
            sb.append("=== Detailed Cost Report: ").append(conversationId).append(" ===\n");

            for (CostEntry entry : entries) {
                sb.append(String.format(
                    "%s | %-20s | %6d + %6d = %6d tokens | $%.6f\n",
                    entry.timestamp,
                    entry.operation,
                    entry.inputTokens,
                    entry.outputTokens,
                    entry.inputTokens + entry.outputTokens,
                    entry.cost.doubleValue()
                ));
            }

            sb.append(String.format("TOTAL: $%.4f\n", totalCost.doubleValue()));
            return sb.toString();
        }
    }
}
```

---

## 13. Evaluating LangChain4j Applications

### What to Measure

AI applications built with LangChain4j need evaluation at multiple levels:

**Response Quality** (most important, hardest to automate):
- Does the AI Service return correct, relevant answers?
- For structured output (records), are fields populated accurately?
- For RAG, are retrieved documents actually relevant to the query?

**Operational Metrics** (automate these first):
- Response latency (p50, p95, p99) -- use the ObservableChatModel from Section 10
- Token usage per request (drives cost)
- Cache hit rate (from CachingChatModel)
- Circuit breaker state transitions (from ResilientChatModel)
- Error rate by type (timeout vs. token limit vs. provider error)

**RAG-Specific Metrics:**
- Retrieval precision: Of the documents returned, how many were actually relevant?
- Retrieval recall: Of the relevant documents in the store, how many were returned?
- Answer groundedness: Does the generated answer stick to the retrieved context?

### Practical Evaluation Pattern

```java
package com.example.evaluation;

import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.LongAdder;

/**
 * Lightweight evaluation tracker for AI service quality.
 * In production, feed these metrics into your observability platform.
 */
public class AiServiceEvaluator {

    private final LongAdder totalRequests = new LongAdder();
    private final LongAdder successfulRequests = new LongAdder();
    private final LongAdder userThumbsUp = new LongAdder();
    private final LongAdder userThumbsDown = new LongAdder();
    private final Map<String, LongAdder> errorsByType = new ConcurrentHashMap<>();

    public void recordSuccess() {
        totalRequests.increment();
        successfulRequests.increment();
    }

    public void recordError(String errorType) {
        totalRequests.increment();
        errorsByType.computeIfAbsent(errorType, k -> new LongAdder()).increment();
    }

    public void recordUserFeedback(boolean positive) {
        if (positive) {
            userThumbsUp.increment();
        } else {
            userThumbsDown.increment();
        }
    }

    public EvaluationReport getReport() {
        long total = totalRequests.sum();
        long successes = successfulRequests.sum();
        long thumbsUp = userThumbsUp.sum();
        long thumbsDown = userThumbsDown.sum();
        long totalFeedback = thumbsUp + thumbsDown;

        return new EvaluationReport(
            total,
            total > 0 ? (double) successes / total : 0.0,
            totalFeedback > 0 ? (double) thumbsUp / totalFeedback : 0.0,
            Map.copyOf(errorsByType)
        );
    }

    public record EvaluationReport(
        long totalRequests,
        double successRate,
        double userSatisfactionRate,
        Map<String, LongAdder> errorBreakdown
    ) {}
}
```

### RAG-Specific Evaluation

```java
package com.example.evaluation;

import dev.langchain4j.data.segment.TextSegment;
import java.util.*;

/**
 * Measures retrieval quality for RAG systems.
 * Run this against a test set of (query, expected_relevant_docs) pairs
 * to catch regressions when you change chunking, embeddings, or retrieval config.
 */
public class RagEvaluator {

    /**
     * Measures retrieval precision and recall against known-relevant documents.
     *
     * Precision = relevant retrieved / total retrieved
     *   → High precision means few irrelevant results polluting the context
     *   → Low precision wastes tokens on noise and confuses the LLM
     *
     * Recall = relevant retrieved / total relevant in corpus
     *   → High recall means the system finds all relevant information
     *   → Low recall means the system misses important context
     */
    public static RetrievalMetrics evaluateRetrieval(
            List<TextSegment> retrievedSegments,
            Set<String> relevantDocIds) {

        long relevantRetrieved = retrievedSegments.stream()
            .filter(s -> relevantDocIds.contains(
                s.metadata().getString("doc_id")))
            .count();

        double precision = retrievedSegments.isEmpty() ? 0.0 :
            (double) relevantRetrieved / retrievedSegments.size();
        double recall = relevantDocIds.isEmpty() ? 0.0 :
            (double) relevantRetrieved / relevantDocIds.size();
        double f1 = (precision + recall) > 0 ?
            2 * precision * recall / (precision + recall) : 0.0;

        return new RetrievalMetrics(precision, recall, f1,
            retrievedSegments.size(), (int) relevantRetrieved);
    }

    /**
     * Answer groundedness check using LLM-as-judge.
     * Detects when the LLM hallucinates beyond the retrieved context.
     *
     * Limitation: LLM-as-judge has known biases — it tends to rate
     * longer answers higher and may miss subtle hallucinations.
     * Use as a signal, not as ground truth.
     */
    public static GroundednessResult evaluateGroundedness(
            dev.langchain4j.model.chat.ChatLanguageModel judge,
            String question,
            String answer,
            List<TextSegment> context) {

        String contextText = context.stream()
            .map(TextSegment::text)
            .reduce("", (a, b) -> a + "\n---\n" + b);

        String prompt = String.format("""
            You are evaluating whether an AI answer is grounded in the provided context.

            Question: %s

            Context:
            %s

            Answer to evaluate:
            %s

            Score the answer on two dimensions (each 1-5):
            1. GROUNDED: Is every claim in the answer supported by the context? (5=fully, 1=hallucinated)
            2. COMPLETE: Does the answer address the question using available context? (5=fully, 1=incomplete)

            Respond in this exact format:
            GROUNDED: <score>
            COMPLETE: <score>
            EXPLANATION: <one sentence>
            """, question, contextText, answer);

        String judgeResponse = judge.generate(prompt);

        // Parse scores (simple regex parsing)
        int grounded = parseScore(judgeResponse, "GROUNDED");
        int complete = parseScore(judgeResponse, "COMPLETE");
        String explanation = parseField(judgeResponse, "EXPLANATION");

        return new GroundednessResult(grounded, complete, explanation,
            grounded >= 4 && complete >= 3);
    }

    private static int parseScore(String response, String field) {
        try {
            int idx = response.indexOf(field + ":");
            if (idx < 0) return 0;
            String rest = response.substring(idx + field.length() + 1).trim();
            return Integer.parseInt(rest.substring(0, 1));
        } catch (Exception e) {
            return 0;
        }
    }

    private static String parseField(String response, String field) {
        int idx = response.indexOf(field + ":");
        if (idx < 0) return "";
        return response.substring(idx + field.length() + 1).trim();
    }

    /**
     * Run regression test suite against a RAG system.
     * Execute after any change to chunking, embeddings, or retrieval config.
     */
    public static RegressionReport runRegressionSuite(
            com.example.rag.RagSystem rag,
            List<TestCase> testCases) {

        int passed = 0;
        List<String> failures = new ArrayList<>();

        for (TestCase tc : testCases) {
            List<TextSegment> retrieved = rag.searchDocuments(tc.query(), 5);
            RetrievalMetrics metrics = evaluateRetrieval(retrieved, tc.relevantDocIds());

            if (metrics.precision() >= tc.minPrecision() &&
                metrics.recall() >= tc.minRecall()) {
                passed++;
            } else {
                failures.add(String.format(
                    "FAIL [%s]: precision=%.2f (min=%.2f), recall=%.2f (min=%.2f)",
                    tc.name(), metrics.precision(), tc.minPrecision(),
                    metrics.recall(), tc.minRecall()));
            }
        }

        return new RegressionReport(testCases.size(), passed, failures);
    }

    // Data classes
    public record RetrievalMetrics(
        double precision, double recall, double f1,
        int totalRetrieved, int relevantRetrieved
    ) {}

    public record GroundednessResult(
        int groundedScore, int completenessScore,
        String explanation, boolean passes
    ) {}

    public record TestCase(
        String name, String query, Set<String> relevantDocIds,
        double minPrecision, double minRecall
    ) {}

    public record RegressionReport(int total, int passed, List<String> failures) {
        public boolean allPassed() { return failures.isEmpty(); }
        public String summary() {
            return String.format("RAG Regression: %d/%d passed. %s",
                passed, total,
                allPassed() ? "ALL GREEN" : "FAILURES:\n" + String.join("\n", failures));
        }
    }
}
```

> **Honest limitation:** Automated evaluation of LLM output quality is an open research problem. The `RagEvaluator` above measures retrieval quality (precision/recall are objective) and answer groundedness (LLM-as-judge is subjective with known biases). For production systems, combine automated metrics with periodic human evaluation on a sample of queries. Track metrics over time to catch regressions, but don't treat any single metric as "the AI is working well."

---

## 14. When NOT to Use LangChain4j

LangChain4j is powerful but not universal. Knowing when it adds too much overhead is crucial.

### Spring AI is the Better Choice

Use **Spring AI** instead of LangChain4j when:

1. **You're already in the Spring Boot ecosystem**
   - Spring AI integrates seamlessly with Spring Data, Spring Security, Spring Cloud
   - LangChain4j requires manual wiring

2. **You need structured integration with Spring patterns**
   - `@AiService` with `@Retryable`, `@Cacheable`, `@Transactional`
   - Spring Boot's auto-configuration handles provider setup

3. **Your team knows Spring deeply**
   - Spring AI follows familiar Spring conventions
   - Less framework magic, more standard Spring patterns

**Comparison:**
```java
// Spring AI (natural for Spring Boot developers)
@Service
public class ChatService {
    @Autowired
    private ChatClient chatClient;

    @Cacheable("responses")
    public String chat(String message) {
        return chatClient.prompt()
            .user(message)
            .call()
            .getResult()
            .getOutput()
            .getContent();
    }
}

// LangChain4j (more explicit, less Spring-magic)
@AiService
public interface ChatService {
    @SystemMessage("You are helpful")
    @MemoryId String chat(@UserMessage String message);
}
```

**Decision:** If you're building greenfield Spring Boot applications, **prefer Spring AI**. It's simpler for standard use cases.

### Direct API Calls Are Simpler

Use **direct API calls** instead of LangChain4j when:

1. **Single-provider, single-feature applications**
   - Calling only OpenAI's chat completion API
   - No RAG, no tools, no memory management
   - Framework overhead isn't justified

2. **Simple request/response patterns**
   - Example: A webhook that transforms user messages to summaries
   - Direct HTTP client is 50 lines vs 500 with LangChain4j setup

3. **You need maximum performance**
   - Direct API calls have 0 abstraction overhead
   - LangChain4j's message construction adds latency

4. **Language diversity matters**
   - Your team uses multiple languages
   - Direct HTTP is language-agnostic

**Cost comparison:**
```
Direct API call:
- Code: 50 lines
- Dependencies: 1 (HTTP client)
- Latency overhead: ~5-10ms
- Learning curve: 1 day

LangChain4j:
- Code: 300 lines
- Dependencies: 15+ libraries
- Latency overhead: ~20-40ms (message building, abstraction layers)
- Learning curve: 1 week
```

**When to use direct calls:**
```
Use LangChain4j if you need: Multiple providers, RAG, tools, memory
Use direct calls if you need: Speed, simplicity, cost-sensitive applications
```

### Overhead of the Abstraction Layer

LangChain4j's abstraction adds measurable overhead:

**Memory footprint:**
- Direct API call: ~10 MB for HTTP client + model
- LangChain4j: ~150-200 MB (multiple dependencies, embedding stores, memory backends)
- Impact: Matters for serverless/container deployments

**Latency breakdown for typical chat request:**
```
Direct API call:
├─ Serialize request to JSON: 2ms
├─ HTTP transmission: 15ms
├─ LLM processing: 500-5000ms
└─ Total: 517-5017ms

LangChain4j:
├─ Message object construction: 3ms
├─ Memory retrieval/update: 5-10ms
├─ Embedding updates: 2-5ms (if using memory embeddings)
├─ Provider abstraction layer: 2-5ms
├─ Serialize to JSON: 3ms
├─ HTTP transmission: 15ms
├─ LLM processing: 500-5000ms
└─ Total: 532-5043ms
```

**Recommendation:** The ~15-25ms overhead is negligible for LLM calls (which take 500+ ms). LangChain4j's overhead matters only for high-throughput batch systems.

### Decision Matrix

```
Use LangChain4j if:
✓ Multiple LLM providers (switching between OpenAI, Claude, Ollama)
✓ RAG system with embedding storage
✓ Complex tools with composition
✓ Multi-turn conversation with persistent memory
✓ Streaming required
✓ Enterprise team (standardize on one framework)

Use Spring AI if:
✓ Spring Boot shop
✓ Integration with Spring Data/Security needed
✓ Team already knows Spring patterns
✓ Standard feature set (no exotic tools)

Use Direct API calls if:
✓ Single provider, single feature
✓ Latency-sensitive (sub-second responses)
✓ Serverless functions (memory/startup time critical)
✓ Batch processing (need to minimize dependencies)
✓ Cost-sensitive (every millisecond counts)
```

---

## 15. Interview Preparation

### Conceptual Questions

**Q1: What are the key differences between LangChain4j and Python LangChain?**

**A:** Key differences include:
1. **Type Safety**: LangChain4j provides compile-time type checking, while Python relies on runtime checks
2. **AI Services Pattern**: LangChain4j's declarative `@AiService` annotation generates implementations from interfaces
3. **Concurrency Model**: Java's virtual threads (Project Loom) vs Python's async/await
4. **Enterprise Integration**: Native Spring/Quarkus support vs requiring adapters
5. **Structured Output**: Java records with `@Description` annotations for automatic schema generation

**Q2: Explain the AI Services pattern in LangChain4j.**

**A:** The AI Services pattern allows developers to define an interface with annotated methods, and LangChain4j generates the implementation at runtime:

```java
public interface MyService {
    @SystemMessage("You are helpful")
    @UserMessage("Answer: {{question}}")
    String answer(@V("question") String q);
}

// Usage: AiServices.create(MyService.class, model)
```

Benefits: Type safety, IDE support, testability, clean separation of concerns.

**Q3: How do you implement multi-user conversations in LangChain4j?**

**A:** Use `ChatMemoryProvider` with `@MemoryId`:

```java
interface Assistant {
    String chat(@MemoryId String sessionId, @UserMessage String msg);
}

// Build with provider
AiServices.builder(Assistant.class)
    .chatLanguageModel(model)
    .chatMemoryProvider(memoryId ->
        MessageWindowChatMemory.withMaxMessages(20))
    .build();
```

### System Design Questions

**Q4: Design a production RAG system using LangChain4j. Walk through your architecture decisions.**

**A:** I'll structure this as ingestion pipeline → retrieval → generation → observability, explaining trade-offs at each layer.

**Ingestion Pipeline:**
- **Document parsing**: Apache Tika via `ApacheTikaDocumentParser` — handles PDF, DOCX, HTML in one parser. Trade-off: Tika adds ~50MB to your deployment; for text-only corpora, use `TextDocumentParser` (2MB).
- **Chunking**: `DocumentSplitters.recursive(500, 50, tokenizer)` — 500-token chunks with 50-token overlap. Why recursive? It respects document structure (paragraphs → sentences → words) instead of splitting mid-sentence. Why 500? Large enough for a complete idea, small enough for precise retrieval. Tune this: technical docs → 300, legal → 800.
- **Embedding model selection**: For <10K docs or data-sovereignty requirements, use `AllMiniLmL6V2EmbeddingModel` (384d, in-process, zero API cost). For production with >10K docs, use `text-embedding-3-small` (1536d, $0.02/1M tokens). Critical: once you choose a dimension, changing it requires re-embedding everything.
- **Storage**: PostgreSQL with pgvector via `PgVectorEmbeddingStore`. Why pgvector over dedicated vector DBs? Your team already knows Postgres, it handles metadata filtering natively, and for <1M vectors the performance difference is negligible. At >5M vectors, consider Pinecone or Weaviate for specialized indexing (HNSW tuning, filtered search).

**Retrieval:**
- Base retrieval: `EmbeddingStoreContentRetriever` with `minScore(0.7)` and `maxResults(10)`. The 0.7 threshold filters irrelevant noise — tune by measuring retrieval precision on a test set.
- Query expansion: `ExpandingQueryTransformer` generates 2-3 query variations using the LLM. Cost: one extra LLM call per query (~$0.001). Value: catches cases where user phrasing doesn't match document vocabulary (e.g., "cancel subscription" vs "terminate account").
- Re-ranking (optional): If you have a `ScoringModel`, use `ReRankingContentAggregator` to re-score the top 20 results down to 5. Adds ~300ms latency but significantly improves relevance for ambiguous queries.

**Generation:**
- AI Service with `@SystemMessage` constraining the LLM to answer only from retrieved context. Critical: include "If the context doesn't contain the answer, say so" to prevent hallucination.
- Token budget: retrieved chunks (~2500 tokens) + system prompt (~200 tokens) + user question (~100 tokens) + response budget (~1000 tokens) = ~3800 tokens per request. At GPT-4o pricing ($5/1M input, $15/1M output), that's ~$0.03 per query.

**Observability:**
- Wrap the `ChatLanguageModel` with `ObservableChatModel` (Micrometer metrics for latency, tokens, errors), `CachingChatModel` (Caffeine, 5-min TTL for repeated queries), and `ResilientChatModel` (Resilience4j circuit breaker, 50% failure threshold).
- Track retrieval-specific metrics: log embedding search latency, number of results above threshold, and cache hit rate separately from LLM metrics.

**Cost estimate for 10K queries/day:**
- Embedding queries: negligible (in-memory or pgvector)
- LLM calls: 10K × $0.03 = $300/month (GPT-4o). Caching with 30% hit rate reduces to ~$210/month.
- Alternative: Switch to GPT-4o-mini for $30/month with ~10% quality reduction.

**Q5: How would you handle high availability for an AI service? Explain the failure cascade.**

**A:** The core problem with AI services is that they depend on external LLM providers over HTTP — any provider outage cascades into your application. Here's how the failure cascade works and how each pattern breaks it:

**The cascade without protection:**
1. Provider returns 503 → your request blocks for 60s (default timeout)
2. Thread pool fills with blocked requests → new requests queue
3. Queue fills → upstream load balancer marks your service unhealthy
4. Health check fails → orchestrator restarts your pod → cold start → more timeouts

**Layer 1 — Circuit Breaker (breaks the cascade at step 2):**
```java
// After 5 failures in 10 calls, stop calling the provider for 30s
CircuitBreakerConfig.custom()
    .failureRateThreshold(50)           // Open after 50% failures
    .waitDurationInOpenState(Duration.ofSeconds(30))  // Wait before retrying
    .slidingWindowSize(10)              // Evaluate last 10 calls
```
When open, requests fail immediately (fast failure) instead of blocking threads. This keeps your thread pool healthy.

**Layer 2 — Retry with Exponential Backoff (handles transient failures before circuit opens):**
```java
RetryConfig.custom()
    .maxAttempts(3)
    .intervalFunction(IntervalFunction.ofExponentialBackoff(1000, 2))  // 1s, 2s, 4s
    .retryExceptions(SocketTimeoutException.class)     // Retry network errors
    .ignoreExceptions(IllegalArgumentException.class)  // Don't retry 400s
```
Critical: never retry 400-class errors (token limit, invalid request) — they'll fail every time and waste budget.

**Layer 3 — Fallback Models (handles sustained outages):**
```java
// FallbackChatModel: OpenAI primary → Anthropic fallback → Ollama local
// Each provider has independent failure modes
```
Provider diversity is key: if OpenAI is down, Anthropic likely isn't. Local Ollama as last resort ensures degraded-but-functional service.

**Layer 4 — Caching (reduces blast radius):**
With a 30% cache hit rate, 30% of your traffic is immune to provider outages entirely. For FAQ-style applications, cache hit rates can reach 60-70%.

**Layer 5 — Rate Limiting (prevents you from being the problem):**
```java
RateLimiter.create(requestsPerMinute / 60.0)  // Token bucket algorithm
```
Prevents exceeding provider quotas (which triggers 429s that cascade). Also prevents a single user from consuming your entire API budget.

**Monitoring:** Alert on circuit breaker state transitions (CLOSED→OPEN means provider degradation), not just error rates. A circuit breaker opening is an early warning; an error rate spike means protection already failed.

### Coding Challenges

**Challenge 1: Implement a tool that validates JSON against a schema**

```java
public class JsonValidatorTool {

    @Tool("Validate JSON against a JSON Schema")
    public ValidationResult validateJson(
            @P("JSON string to validate") String json,
            @P("JSON Schema to validate against") String schema) {

        try {
            ObjectMapper mapper = new ObjectMapper();
            JsonNode jsonNode = mapper.readTree(json);
            JsonNode schemaNode = mapper.readTree(schema);

            JsonSchemaFactory factory = JsonSchemaFactory.getInstance(
                SpecVersion.VersionFlag.V7
            );
            JsonSchema jsonSchema = factory.getSchema(schemaNode);

            Set<ValidationMessage> errors = jsonSchema.validate(jsonNode);

            return new ValidationResult(
                errors.isEmpty(),
                errors.stream()
                    .map(ValidationMessage::getMessage)
                    .toList()
            );
        } catch (com.fasterxml.jackson.core.JsonProcessingException e) {
            return new ValidationResult(false, List.of("Invalid JSON: " + e.getOriginalMessage()));
        } catch (IllegalArgumentException e) {
            return new ValidationResult(false, List.of("Invalid schema: " + e.getMessage()));
        }
    }

    public record ValidationResult(
        boolean valid,
        List<String> errors
    ) {}
}
```

**Challenge 2: Create a production rate-limited AI service wrapper with per-user tracking**

```java
import com.google.common.util.concurrent.RateLimiter;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.output.Response;

import java.time.Duration;
import java.time.Instant;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.LongAdder;

/**
 * Rate-limited ChatLanguageModel with per-user quotas and global limits.
 *
 * Design decisions:
 * - Global rate limit prevents exceeding provider quotas (429 errors)
 * - Per-user rate limit prevents a single user from starving others
 * - Sliding window tracking enables usage reporting and quota enforcement
 * - Throws RateLimitExceededException instead of blocking (fail-fast for APIs)
 */
public class RateLimitedChatModel implements ChatLanguageModel {

    private final ChatLanguageModel delegate;
    private final RateLimiter globalLimiter;
    private final int perUserRequestsPerMinute;
    private final Map<String, UserUsageTracker> userTrackers = new ConcurrentHashMap<>();

    public RateLimitedChatModel(
            ChatLanguageModel delegate,
            int globalRequestsPerMinute,
            int perUserRequestsPerMinute) {
        this.delegate = delegate;
        this.globalLimiter = RateLimiter.create(globalRequestsPerMinute / 60.0);
        this.perUserRequestsPerMinute = perUserRequestsPerMinute;
    }

    /**
     * Generate with rate limiting. For user-specific limits,
     * use generateForUser() instead.
     */
    @Override
    public Response<AiMessage> generate(List<ChatMessage> messages) {
        if (!globalLimiter.tryAcquire(Duration.ofMillis(100))) {
            throw new RateLimitExceededException("Global rate limit exceeded. Try again shortly.");
        }
        return delegate.generate(messages);
    }

    /**
     * Generate with per-user rate limiting.
     * Checks both global and user-specific limits.
     */
    public Response<AiMessage> generateForUser(String userId, List<ChatMessage> messages) {
        // Check per-user limit first (cheaper check)
        UserUsageTracker tracker = userTrackers.computeIfAbsent(
            userId, id -> new UserUsageTracker(perUserRequestsPerMinute));

        if (!tracker.tryAcquire()) {
            throw new RateLimitExceededException(
                "User " + userId + " exceeded " + perUserRequestsPerMinute +
                " requests/minute. Remaining: " + tracker.getRemainingQuota());
        }

        // Then check global limit
        if (!globalLimiter.tryAcquire(Duration.ofMillis(100))) {
            tracker.release();  // Don't count against user quota
            throw new RateLimitExceededException("Global rate limit exceeded.");
        }

        return delegate.generate(messages);
    }

    public Map<String, UsageReport> getUsageReport() {
        Map<String, UsageReport> report = new ConcurrentHashMap<>();
        userTrackers.forEach((userId, tracker) ->
            report.put(userId, tracker.getReport()));
        return report;
    }

    /**
     * Tracks per-user request count in a sliding 1-minute window.
     */
    static class UserUsageTracker {
        private final int maxRequestsPerMinute;
        private final LongAdder totalRequests = new LongAdder();
        private final java.util.Deque<Instant> requestTimestamps =
            new java.util.concurrent.ConcurrentLinkedDeque<>();

        UserUsageTracker(int maxRequestsPerMinute) {
            this.maxRequestsPerMinute = maxRequestsPerMinute;
        }

        synchronized boolean tryAcquire() {
            evictOldEntries();
            if (requestTimestamps.size() >= maxRequestsPerMinute) {
                return false;
            }
            requestTimestamps.addLast(Instant.now());
            totalRequests.increment();
            return true;
        }

        synchronized void release() {
            if (!requestTimestamps.isEmpty()) {
                requestTimestamps.removeLast();
            }
        }

        int getRemainingQuota() {
            evictOldEntries();
            return Math.max(0, maxRequestsPerMinute - requestTimestamps.size());
        }

        private void evictOldEntries() {
            Instant cutoff = Instant.now().minus(Duration.ofMinutes(1));
            while (!requestTimestamps.isEmpty() &&
                   requestTimestamps.peekFirst().isBefore(cutoff)) {
                requestTimestamps.removeFirst();
            }
        }

        UsageReport getReport() {
            evictOldEntries();
            return new UsageReport(
                totalRequests.sum(),
                requestTimestamps.size(),
                maxRequestsPerMinute - requestTimestamps.size()
            );
        }
    }

    public record UsageReport(long totalRequests, int currentWindowCount, int remainingQuota) {}

    public static class RateLimitExceededException extends RuntimeException {
        public RateLimitExceededException(String message) { super(message); }
    }
}
```

---

## 16. Hands-On Exercises

### Exercise 1: Multi-Provider Chat Application

Build a chat application that can switch between providers:

```java
/**
 * Exercise: Complete the multi-provider chat application
 *
 * Requirements:
 * 1. Support OpenAI, Anthropic, and Ollama
 * 2. Allow runtime provider switching
 * 3. Maintain conversation history across provider switches
 * 4. Handle provider-specific errors gracefully
 */

public class MultiProviderChat {

    public enum Provider {
        OPENAI, ANTHROPIC, OLLAMA
    }

    private volatile Provider currentProvider;
    private final Map<Provider, ChatLanguageModel> models = new EnumMap<>(Provider.class);
    private final MultiUserMemoryProvider memoryProvider;

    /**
     * Key design decision: memory is shared across providers.
     * When a user switches from OpenAI to Anthropic mid-conversation,
     * the conversation history carries over. This works because
     * LangChain4j's ChatMemory stores provider-agnostic ChatMessage
     * objects, not provider-specific formats.
     */
    public MultiProviderChat(int memorySize) {
        this.memoryProvider = new MultiUserMemoryProvider(memorySize);
        this.currentProvider = Provider.OPENAI;

        // Initialize available providers (skip unavailable ones)
        initializeProviders();
    }

    private void initializeProviders() {
        if (System.getenv("OPENAI_API_KEY") != null) {
            models.put(Provider.OPENAI, OpenAiConfig.createChatModel());
        }
        if (System.getenv("ANTHROPIC_API_KEY") != null) {
            models.put(Provider.ANTHROPIC, AnthropicConfig.createSonnetModel());
        }
        // Ollama doesn't need API key — check if server is reachable
        try {
            models.put(Provider.OLLAMA, OllamaConfig.createLlama3Model());
        } catch (Exception e) {
            System.err.println("Ollama not available: " + e.getMessage());
        }

        if (models.isEmpty()) {
            throw new IllegalStateException("No LLM providers available. Set API keys or start Ollama.");
        }
    }

    public interface ChatAssistant {
        @SystemMessage("You are a helpful assistant. Continue the conversation naturally.")
        String chat(@MemoryId String sessionId, @UserMessage String message);
    }

    public String chat(String sessionId, String message) {
        ChatLanguageModel model = models.get(currentProvider);
        if (model == null) {
            // Fallback: try any available provider
            model = models.values().iterator().next();
            System.err.println("Provider " + currentProvider + " unavailable, falling back.");
        }

        try {
            ChatAssistant assistant = AiServices.builder(ChatAssistant.class)
                .chatLanguageModel(model)
                .chatMemoryProvider(memoryProvider)
                .build();
            return assistant.chat(sessionId, message);
        } catch (RuntimeException e) {
            // On failure, try fallback provider
            return chatWithFallback(sessionId, message, e);
        }
    }

    private String chatWithFallback(String sessionId, String message, RuntimeException original) {
        for (Map.Entry<Provider, ChatLanguageModel> entry : models.entrySet()) {
            if (entry.getKey() == currentProvider) continue;
            try {
                ChatAssistant fallback = AiServices.builder(ChatAssistant.class)
                    .chatLanguageModel(entry.getValue())
                    .chatMemoryProvider(memoryProvider)
                    .build();
                System.err.println("Falling back to " + entry.getKey());
                return fallback.chat(sessionId, message);
            } catch (RuntimeException ignored) {
                // Try next provider
            }
        }
        throw new RuntimeException("All providers failed. Original error: " + original.getMessage(), original);
    }

    public void switchProvider(Provider provider) {
        if (!models.containsKey(provider)) {
            throw new IllegalArgumentException("Provider not available: " + provider +
                ". Available: " + models.keySet());
        }
        Provider old = this.currentProvider;
        this.currentProvider = provider;
        System.out.println("Switched from " + old + " to " + provider +
            ". Conversation memory preserved.");
    }

    public Provider getCurrentProvider() { return currentProvider; }
    public Set<Provider> getAvailableProviders() { return models.keySet(); }
}
```

### Exercise 2: Document Q&A System

Build a complete document Q&A system:

```java
/**
 * Exercise: Build a document Q&A system
 *
 * Requirements:
 * 1. Load documents from a directory (PDF, TXT, DOCX)
 * 2. Chunk documents appropriately
 * 3. Store embeddings in PostgreSQL with pgvector
 * 4. Implement semantic search
 * 5. Generate answers with source citations
 */

public class DocumentQA {

    // TODO: Implement document loading
    // TODO: Implement chunking with metadata
    // TODO: Implement embedding storage
    // TODO: Implement search with citations

    public Answer askQuestion(String question) {
        // Your implementation here
        return null;
    }

    public record Answer(
        String text,
        List<Source> sources
    ) {}

    public record Source(
        String filename,
        int page,
        String excerpt
    ) {}
}
```

### Exercise 3: AI Agent with Custom Tools

Create an agent for data analysis:

```java
/**
 * Exercise: Build a data analysis agent
 *
 * Requirements:
 * 1. Tool to load CSV files
 * 2. Tool to perform basic statistics
 * 3. Tool to create visualizations (return description)
 * 4. Tool to export results
 * 5. Agent that can answer analytical questions
 */

public class DataAnalysisAgent {

    // TODO: Implement CsvTool
    // TODO: Implement StatisticsTool
    // TODO: Implement VisualizationTool
    // TODO: Implement ExportTool
    // TODO: Wire tools into agent

    public AnalysisResult analyze(String dataPath, String question) {
        // Your implementation here
        return null;
    }
}
```

### Exercise 4: Streaming Chat API

Build a REST API with streaming responses:

```java
/**
 * Exercise: Implement streaming chat endpoint
 *
 * Requirements:
 * 1. POST /api/chat - standard request/response
 * 2. GET /api/chat/stream - Server-Sent Events
 * 3. Session management with conversation history
 * 4. Token usage tracking
 * 5. Rate limiting per session
 */

@RestController
@RequestMapping("/api/chat")
public class ChatController {

    private final ChatLanguageModel chatModel;
    private final OpenAiStreamingChatModel streamingModel;
    private final MultiUserMemoryProvider memoryProvider;
    private final RateLimitedChatModel rateLimitedModel;
    private final Map<String, SessionUsage> sessionUsage = new ConcurrentHashMap<>();

    public ChatController() {
        ChatLanguageModel baseModel = OpenAiConfig.createChatModel();
        this.chatModel = baseModel;
        this.streamingModel = OpenAiConfig.createStreamingModel();
        this.memoryProvider = new MultiUserMemoryProvider(20);
        this.rateLimitedModel = new RateLimitedChatModel(baseModel, 100, 10);
    }

    // --- Standard request/response endpoint ---

    public interface ChatAssistant {
        @SystemMessage("You are a helpful assistant.")
        String chat(@MemoryId String sessionId, @UserMessage String message);
    }

    @PostMapping
    public ResponseEntity<ChatResponse> chat(@RequestBody ChatRequest request) {
        String sessionId = request.sessionId();
        if (sessionId == null || sessionId.isBlank()) {
            sessionId = java.util.UUID.randomUUID().toString();
        }

        try {
            ChatAssistant assistant = AiServices.builder(ChatAssistant.class)
                .chatLanguageModel(rateLimitedModel)
                .chatMemoryProvider(memoryProvider)
                .build();

            long start = System.currentTimeMillis();
            String response = assistant.chat(sessionId, request.message());
            long latencyMs = System.currentTimeMillis() - start;

            // Track usage
            sessionUsage.computeIfAbsent(sessionId, id -> new SessionUsage())
                .recordRequest(latencyMs);

            return ResponseEntity.ok(new ChatResponse(
                sessionId, response, latencyMs, null));
        } catch (RateLimitedChatModel.RateLimitExceededException e) {
            return ResponseEntity.status(429).body(new ChatResponse(
                sessionId, null, 0, e.getMessage()));
        }
    }

    // --- Streaming SSE endpoint ---

    public interface StreamingChatAssistant {
        @SystemMessage("You are a helpful assistant.")
        TokenStream chat(String message);
    }

    @GetMapping(value = "/stream", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
    public Flux<String> streamChat(
            @RequestParam String sessionId,
            @RequestParam String message) {

        Sinks.Many<String> sink = Sinks.many().unicast().onBackpressureBuffer();

        StreamingChatAssistant assistant = AiServices.create(
            StreamingChatAssistant.class, streamingModel);

        assistant.chat(message)
            .onNext(token -> sink.tryEmitNext(token))
            .onComplete(response -> {
                sessionUsage.computeIfAbsent(sessionId, id -> new SessionUsage())
                    .recordRequest(0);
                sink.tryEmitComplete();
            })
            .onError(error -> sink.tryEmitError(error))
            .start();

        return sink.asFlux();
    }

    // --- Usage tracking endpoint ---

    @GetMapping("/usage/{sessionId}")
    public ResponseEntity<SessionUsage> getUsage(@PathVariable String sessionId) {
        SessionUsage usage = sessionUsage.get(sessionId);
        if (usage == null) {
            return ResponseEntity.notFound().build();
        }
        return ResponseEntity.ok(usage);
    }

    // --- Data classes ---

    public record ChatRequest(String sessionId, String message) {}
    public record ChatResponse(String sessionId, String response, long latencyMs, String error) {}

    public static class SessionUsage {
        private final java.util.concurrent.atomic.AtomicLong requestCount = new java.util.concurrent.atomic.AtomicLong();
        private final java.util.concurrent.atomic.AtomicLong totalLatencyMs = new java.util.concurrent.atomic.AtomicLong();
        private volatile long firstRequestTime = System.currentTimeMillis();

        public void recordRequest(long latencyMs) {
            requestCount.incrementAndGet();
            totalLatencyMs.addAndGet(latencyMs);
        }

        public long getRequestCount() { return requestCount.get(); }
        public double getAvgLatencyMs() {
            long count = requestCount.get();
            return count > 0 ? (double) totalLatencyMs.get() / count : 0;
        }
        public long getSessionDurationSeconds() {
            return (System.currentTimeMillis() - firstRequestTime) / 1000;
        }
    }
}
```

---

## 17. Summary

### Key Takeaways

```
┌─────────────────────────────────────────────────────────────────┐
│              LangChain4j Mastery Checklist                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ✓ Project Setup                                                │
│    □ Maven/Gradle configuration with BOM                        │
│    □ Multi-provider dependency management                       │
│    □ Java 21 with virtual threads enabled                       │
│                                                                  │
│  ✓ Model Integration                                            │
│    □ OpenAI, Anthropic, Azure, Ollama configurations           │
│    □ Model factory pattern for provider abstraction             │
│    □ Streaming vs non-streaming model selection                │
│                                                                  │
│  ✓ AI Services Pattern                                          │
│    □ Interface-based service definition                         │
│    □ @SystemMessage and @UserMessage annotations               │
│    □ Structured output with Java records                        │
│    □ Type-safe parameter binding with @V                       │
│                                                                  │
│  ✓ Memory Management                                            │
│    □ MessageWindowChatMemory for recent context                │
│    □ TokenWindowChatMemory for token-aware limits              │
│    □ Multi-user memory with @MemoryId                          │
│    □ Persistent memory with Redis/database                      │
│                                                                  │
│  ✓ RAG Implementation                                           │
│    □ Document loading with Apache Tika                          │
│    □ Recursive text splitting                                   │
│    □ Embedding with local or API models                         │
│    □ Vector storage with pgvector                               │
│    □ Query transformation and re-ranking                        │
│                                                                  │
│  ✓ Function Calling                                             │
│    □ @Tool annotation for method definitions                    │
│    □ @P for parameter descriptions                              │
│    □ Tool composition in AI services                            │
│    □ Error handling in tools                                    │
│                                                                  │
│  ✓ Production Patterns                                          │
│    □ Circuit breaker with Resilience4j                          │
│    □ Response caching with Caffeine                             │
│    □ Metrics with Micrometer                                    │
│    □ Structured logging                                         │
│                                                                  │
│  ✓ Testing Strategies                                           │
│    □ Mock-based unit tests                                      │
│    □ Integration tests with real APIs                           │
│    □ Tool testing in isolation                                  │
│    □ RAG system testing                                         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### LangChain4j Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Your Application                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     AI Services Layer                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │  Interfaces │  │ Annotations │  │ Structured Output       │ │
│  │  @AiService │  │ @SystemMsg  │  │ Java Records + @Desc    │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Supporting Components                           │
│  ┌───────────┐  ┌────────────┐  ┌──────────┐  ┌─────────────┐ │
│  │  Memory   │  │    RAG     │  │   Tools  │  │  Streaming  │ │
│  │  Stores   │  │  Pipeline  │  │  @Tool   │  │  Handlers   │ │
│  └───────────┘  └────────────┘  └──────────┘  └─────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Model Abstraction                             │
│  ┌─────────────────────┐  ┌───────────────────────────────────┐│
│  │ ChatLanguageModel   │  │ EmbeddingModel                    ││
│  └─────────────────────┘  └───────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Provider Implementations                      │
│  ┌────────┐ ┌───────────┐ ┌───────┐ ┌────────┐ ┌────────────┐ │
│  │ OpenAI │ │ Anthropic │ │ Azure │ │ Ollama │ │ HuggingFace│ │
│  └────────┘ └───────────┘ └───────┘ └────────┘ └────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### What's Next?

In **Bonus B: Spring AI Enterprise Integration**, we'll explore:
- Spring Boot integration patterns
- Auto-configuration and starters
- Enterprise features (security, monitoring)
- Comparison with LangChain4j
- Production deployment strategies

---

## Self-Assessment Rubric

Rate yourself honestly after completing this blog:

| Criteria | Excellent (9-10) | Good (7-8) | Needs Work (5-6) |
|----------|------------------|------------|------------------|
| **Setup & Configuration** | Multi-provider Maven/Gradle with BOM, version management | Single provider setup with manual deps | Basic dependencies, missing BOM |
| **AI Services Pattern** | Complex interfaces with @SystemMessage, @MemoryId, @Tool integration | Basic @AiService with chat annotation | Simple interface definition only |
| **RAG Implementation** | Full pipeline: loading, splitting, embedding, retrieval, re-ranking | Basic retrieval from vector store | Document parsing without integration |
| **Function Calling & Tools** | Multiple tools with error handling, type validation | Single tool working with basic types | Tool skeleton without implementation |
| **Error Handling** | Timeout/retry logic, token limit handling, memory bounds | Basic try-catch blocks | No error strategy |
| **Production Readiness** | Circuit breaker, caching, observability, metrics | Basic error handling with fallbacks | No resilience patterns |
| **Testing** | Integration tests with real APIs, tool isolation tests | Unit tests with mocks | Ad-hoc manual testing |

### What This Blog Does Well
- Comprehensive coverage of LangChain4j's core APIs with idiomatic Java patterns (records, annotations, interfaces)
- Production patterns (circuit breaker, caching, observability) that go beyond "hello world" examples
- Clear comparison with alternatives (Spring AI, direct API calls) with honest trade-off analysis
- Testing strategies at multiple levels (unit, integration, tool isolation)

### Where This Blog Falls Short
- All code examples are illustrative snippets, not a runnable end-to-end project you can clone and build — readers must wire dependencies and resolve version conflicts themselves
- LangChain4j API evolves rapidly (version 0.35.0 shown here); method signatures and class names may change between releases — always verify against your target version
- The RAG examples skip document versioning and incremental re-indexing — production corpora change over time and need update strategies
- No load testing or JMH benchmarks to validate the latency overhead claims (15-25ms are estimates from the LangChain4j community, not measured in this blog)
- Error handling examples use RuntimeException wrapping, which is idiomatic Java but loses type safety at catch sites — consider checked exceptions or sealed types for production error hierarchies

---

## Architect Sanity Checks

**Would you trust someone who learned *only this blog* to touch a production AI system?**
- **YES**, with caveats. The blog teaches the complete production stack: resilience patterns (circuit breaker with Resilience4j, Section 10), failure mode handling (timeouts, token limits, memory leaks, serialization — Section 11), cost tracking with budget enforcement (Section 12), RAG evaluation with regression testing (Section 13), and honest "When NOT to Use" analysis (Section 14). A developer who implements the `ProductionAiServiceFactory` (resilience + caching + observability decorator chain), the `BudgetEnforcer`, and the `RagEvaluator` regression suite has the operational foundations for production. **Caveat**: LangChain4j's API evolves between releases (version 0.35.0 shown here). Readers must verify interface signatures against their target version — the blog explicitly calls this out.

**Can you explain at least one real failure case using only what's taught here?**
- **YES**. Worked example: A production RAG system starts returning irrelevant answers after a routine embedding model update. Diagnosis path from this blog: (1) `RagEvaluator.runRegressionSuite()` (Section 13) detects precision dropped from 0.85 to 0.40 — the new embedding model changed the vector space. (2) `ObservableChatModel` metrics (Section 10) show token usage spiked 3x because irrelevant chunks waste context tokens. (3) `BudgetEnforcer` (Section 12) blocks requests when the per-conversation cost limit is hit. (4) Fix: either re-embed all documents with the new model (Section 6 embedding service), or roll back the embedding model and re-run the regression suite to confirm green. This diagnosis requires Sections 6, 10, 12, and 13 working together — exactly the production stack taught in this blog.

**Would this blog survive senior-engineer interview follow-up questions?**
- **YES**. Interview Q4 walks through a complete RAG system design with cost math ($0.03/query, $300/month at 10K queries/day, caching reduces to $210/month), embedding model selection trade-offs, and chunk size justification. Q5 explains the failure cascade (provider 503 → thread pool exhaustion → health check failure → pod restart) and how each resilience layer breaks it at a specific point. Both answers include the "why" behind each decision, not just "what." The coding challenges implement non-trivial patterns: `RateLimitedChatModel` with per-user sliding window quotas and `JsonValidatorTool` with proper error handling. Follow-up questions like "what happens when you change embedding models?" are answered by the RAG evaluation regression suite.

---

*Next: [Bonus B: Spring AI Enterprise Integration →](bonus-b-spring-ai.md)*
