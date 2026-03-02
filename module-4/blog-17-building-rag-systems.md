# Blog 17: Building RAG Systems -- Giving LLMs Your Private Data

**Series:** Prompt Your Career: The Complete Generative AI Masterclass
**Prerequisites:** Blog 16 (Embeddings and Vector Databases), Blog 14 (Working with AI APIs)
**Reading time:** 60-90 minutes
**Coding time:** 90-120 minutes
**Total investment:** ~3.5 hours
**Difficulty:** Intermediate to Advanced

---

## What You'll Walk Away With

After completing this blog, you will be able to:

1. **Explain RAG architecture** — the mechanism (parametric vs. non-parametric memory), not just the benefits
2. **Decide when RAG is appropriate** — and when it's the wrong approach
3. **Design chunking strategies** for different document types with explicit tradeoffs
4. **Build a complete RAG pipeline** from ingestion to response with source attribution
5. **Implement retrieval optimization** — reranking (with cross-encoder vs. bi-encoder understanding), hybrid search (BM25 + dense), and query transformation
6. **Handle context window management** with token-aware chunk selection
7. **Evaluate RAG system quality** with retrieval metrics, faithfulness scoring, and LLM-judge calibration
8. **Estimate and optimize costs** — per-query and per-indexing cost breakdowns
9. **Debug common RAG failures** using structured failure mode analysis

> **How to read this blog:** If you're new to RAG, read the architecture overview and chunking sections first, then work through the complete RAG pipeline code. If you already understand retrieval concepts from Blog 16, jump directly to the pipeline implementation and focus on the advanced techniques (reranking, query transformation, context management). The evaluation and debugging sections are essential for anyone planning to deploy RAG in production -- don't skip them.

---

## What This Blog Does NOT Cover

Before we begin, let's set clear expectations on scope:

- **Fine-tuning vs. RAG tradeoffs in depth** -- we compare them briefly in the Manager's Summary, but a full treatment of when to fine-tune is in Blog 23.
- **Production deployment and scaling** -- containerization, load balancing, and infrastructure for RAG at scale are covered in Blog 24.
- **Agent architectures that use RAG** -- function calling and agentic RAG patterns are in Blog 18.
- **Advanced embedding model selection and training** -- we use off-the-shelf embeddings here; custom embedding fine-tuning is beyond this blog's scope.
- **Multi-modal RAG** -- retrieving and reasoning over images, tables, and mixed media requires techniques not covered here.
- **Security and access control** -- document-level permissions, PII filtering, and data governance for RAG systems are critical production concerns not addressed here.

---

## Manager's Summary

**Why RAG Matters:**

LLMs have two fundamental limitations: (1) Knowledge cutoff—they don't know recent information, and (2) No access to private data—they can't reference your company's documents. RAG solves both by retrieving relevant context before generating responses.

**Business Value of RAG:**

| Use Case | Without RAG | With RAG | Value |
|----------|-------------|----------|-------|
| **Customer Support** | Generic answers | Company-specific answers | More accurate resolution |
| **Internal Q&A** | "I don't know" | Answers from docs | Hours saved/employee/week |
| **Legal/Compliance** | Can't reference policies | Cites specific clauses | Reduced liability |
| **Research** | Limited to training data | Access to latest papers | Faster insights |

**RAG vs Fine-Tuning:**

| Aspect | RAG | Fine-Tuning |
|--------|-----|-------------|
| **Data freshness** | Real-time updates | Requires retraining |
| **Setup time** | Hours | Days-Weeks |
| **Cost** | Per-query (retrieval) | One-time (+ periodic) |
| **Interpretability** | Shows sources | Black box |
| **Best for** | Factual Q&A, search | Style/format changes |

**Implementation Complexity:**

| Component | Effort | Criticality |
|-----------|--------|-------------|
| Document ingestion | Medium | High |
| Chunking strategy | Medium | Critical |
| Vector database | Low | High |
| Retrieval tuning | High | Critical |
| Response generation | Low | Medium |
| Evaluation | Medium | High |

---

## RAG Architecture Overview

### The RAG Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           RAG ARCHITECTURE                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  OFFLINE (Indexing)                                                         │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────────────────┐   │
│  │ Documents│───>│ Chunking │───>│ Embedding│───>│   Vector Database    │   │
│  │ (PDF,    │    │          │    │          │    │                      │   │
│  │  HTML,   │    │ Split    │    │ Convert  │    │  Store embeddings    │   │
│  │  TXT...) │    │ into     │    │ to       │    │  + metadata          │   │
│  │          │    │ chunks   │    │ vectors  │    │                      │   │
│  └──────────┘    └──────────┘    └──────────┘    └──────────────────────┘   │
│                                                              │               │
│                                                              │               │
│  ONLINE (Querying)                                           │               │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐               │               │
│  │ User     │───>│ Query    │───>│ Retrieve │<──────────────┘               │
│  │ Question │    │ Embedding│    │ Top-K    │                               │
│  └──────────┘    └──────────┘    └──────────┘                               │
│                                         │                                    │
│                                         │ Retrieved Chunks                   │
│                                         ▼                                    │
│                               ┌──────────────────┐    ┌──────────────────┐  │
│                               │ Prompt           │───>│      LLM         │  │
│                               │ Construction     │    │                  │  │
│                               │                  │    │  Generate answer │  │
│                               │ Question +       │    │  grounded in     │  │
│                               │ Context Chunks   │    │  retrieved docs  │  │
│                               └──────────────────┘    └────────┬─────────┘  │
│                                                                 │            │
│                                                                 ▼            │
│                                                        ┌──────────────────┐  │
│                                                        │ Response with    │  │
│                                                        │ citations        │  │
│                                                        └──────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Why RAG Works (The Mechanism, Not Just the Benefits)

LLMs are **parametric** models: everything they know is encoded in their weights during training. This creates hard limits—they cannot access information added after training, and they cannot reference documents they never saw. RAG adds a **non-parametric** memory layer: at query time, the system retrieves relevant text from an external store and injects it into the prompt. The LLM then conditions its generation on both its parametric knowledge and the retrieved context.

This is not magic. It works because:

1. **Knowledge Cutoff → External Retrieval.** Instead of retraining a model (expensive, slow), you update the document store. The LLM sees fresh content at inference time.
2. **Hallucination → Grounding.** When the prompt contains relevant source text, the LLM is far more likely to paraphrase that text than to invent facts. This *reduces* hallucination probability but does **not eliminate it**. The LLM can still hallucinate if: the retrieved context is irrelevant, the context is ambiguous, or the model's parametric knowledge conflicts with the context.
3. **Private Data → Retrieval Without Training.** Your documents never enter model weights. They stay in your infrastructure. The LLM only sees chunks that match the current query—minimizing data exposure.
4. **Verifiability → Source Attribution.** Because each chunk has metadata (source file, page number, section), you can trace exactly which documents influenced the answer.

```python
# The basic RAG equation:
"""
Response = LLM(Query + Retrieved_Context)

Where:
- Query: User's question
- Retrieved_Context: Most relevant chunks from your documents
- Response: Answer grounded in your data (hallucination-reduced, not hallucination-proof)
"""
```

### When RAG Is the Wrong Choice

RAG is not universally correct. Use a different approach when:

| Situation | Why RAG Fails | Better Alternative |
|-----------|---------------|-------------------|
| **Task requires reasoning over the entire corpus** (e.g., "summarize all Q4 reports") | RAG retrieves fragments, not the full corpus | Map-reduce summarization, or fine-tuning |
| **Latency budget < 500ms** | Embedding + vector search + LLM generation adds 1-5s | Pre-computed answers, cached responses, or traditional search |
| **Document set fits in context window** (< 100K tokens) | Retrieval adds complexity without benefit | Stuff all documents directly into the prompt |
| **Task is style/format change, not knowledge retrieval** (e.g., "write in our brand voice") | RAG retrieves facts, not style | Fine-tuning on examples of desired style |
| **No clear query-document semantic match** (e.g., abstract reasoning, math) | Embedding similarity doesn't capture logical relationships | Chain-of-thought prompting, tool use, code execution |
| **Corpus changes every few seconds** (e.g., live stock prices) | Indexing latency makes embeddings stale | Direct API calls, real-time data feeds |

> **Rule of thumb:** If you can describe your task as "answer questions about a specific set of documents," RAG is likely a fit. If you cannot, question it before building.

---

## Document Processing

### Loading Documents

```python
"""
Document loading from various sources.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from pathlib import Path
import hashlib


@dataclass
class Document:
    """Represents a loaded document."""
    content: str
    metadata: Dict[str, Any]
    doc_id: str = None

    def __post_init__(self):
        if self.doc_id is None:
            self.doc_id = hashlib.md5(self.content.encode()).hexdigest()[:16]


class DocumentLoader:
    """
    Load documents from various sources.
    """

    @staticmethod
    def load_text(file_path: str) -> Document:
        """Load plain text file."""
        path = Path(file_path)
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()

        return Document(
            content=content,
            metadata={
                "source": str(path),
                "filename": path.name,
                "type": "text"
            }
        )

    @staticmethod
    def load_pdf(file_path: str) -> Document:
        """Load PDF file."""
        import pypdf

        path = Path(file_path)
        reader = pypdf.PdfReader(path)

        content = ""
        for page_num, page in enumerate(reader.pages):
            content += f"\n[Page {page_num + 1}]\n"
            content += page.extract_text()

        return Document(
            content=content,
            metadata={
                "source": str(path),
                "filename": path.name,
                "type": "pdf",
                "pages": len(reader.pages)
            }
        )

    @staticmethod
    def load_markdown(file_path: str) -> Document:
        """Load Markdown file."""
        path = Path(file_path)
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Extract title from first heading if present
        title = None
        for line in content.split('\n'):
            if line.startswith('# '):
                title = line[2:].strip()
                break

        return Document(
            content=content,
            metadata={
                "source": str(path),
                "filename": path.name,
                "type": "markdown",
                "title": title
            }
        )

    @staticmethod
    def load_html(file_path: str) -> Document:
        """Load HTML file, extracting text content."""
        from bs4 import BeautifulSoup

        path = Path(file_path)
        with open(path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f.read(), 'html.parser')

        # Remove script and style elements
        for element in soup(['script', 'style', 'nav', 'footer']):
            element.decompose()

        content = soup.get_text(separator='\n', strip=True)
        title = soup.title.string if soup.title else None

        return Document(
            content=content,
            metadata={
                "source": str(path),
                "filename": path.name,
                "type": "html",
                "title": title
            }
        )

    @staticmethod
    def load_directory(
        directory: str,
        extensions: List[str] = None
    ) -> List[Document]:
        """Load all supported documents from a directory."""
        extensions = extensions or ['.txt', '.pdf', '.md', '.html']
        documents = []

        path = Path(directory)
        for ext in extensions:
            for file_path in path.rglob(f"*{ext}"):
                try:
                    if ext == '.txt':
                        doc = DocumentLoader.load_text(str(file_path))
                    elif ext == '.pdf':
                        doc = DocumentLoader.load_pdf(str(file_path))
                    elif ext in ['.md', '.markdown']:
                        doc = DocumentLoader.load_markdown(str(file_path))
                    elif ext in ['.html', '.htm']:
                        doc = DocumentLoader.load_html(str(file_path))
                    else:
                        continue

                    documents.append(doc)
                    print(f"Loaded: {file_path}")
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")

        return documents
```

### Chunking Strategies

```python
"""
Chunking is critical - poor chunking = poor RAG.
"""

from dataclasses import dataclass
from typing import List
import re


@dataclass
class Chunk:
    """Represents a document chunk."""
    content: str
    metadata: Dict[str, Any]
    chunk_id: str = None

    def __post_init__(self):
        if self.chunk_id is None:
            self.chunk_id = hashlib.md5(self.content.encode()).hexdigest()[:16]


class ChunkingStrategy:
    """
    Different strategies for splitting documents into chunks.
    """

    @staticmethod
    def fixed_size(
        document: Document,
        chunk_size: int = 500,
        overlap: int = 50
    ) -> List[Chunk]:
        """
        Split into fixed-size chunks with overlap.

        Pros: Simple, predictable chunk sizes
        Cons: May split mid-sentence, ignores structure
        Best for: Uniform documents, fallback strategy
        """
        text = document.content
        chunks = []

        start = 0
        while start < len(text):
            end = start + chunk_size

            # Try to end at a sentence boundary
            if end < len(text):
                # Look for sentence end within last 20% of chunk
                search_start = max(start + int(chunk_size * 0.8), start)
                for delimiter in ['. ', '.\n', '! ', '? ']:
                    pos = text.rfind(delimiter, search_start, end)
                    if pos != -1:
                        end = pos + 1
                        break

            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append(Chunk(
                    content=chunk_text,
                    metadata={
                        **document.metadata,
                        "chunk_index": len(chunks),
                        "start_char": start,
                        "end_char": end
                    }
                ))

            start = end - overlap

        return chunks

    @staticmethod
    def semantic(
        document: Document,
        max_chunk_size: int = 1000,
        min_chunk_size: int = 100
    ) -> List[Chunk]:
        """
        Split based on semantic boundaries (paragraphs, sections).

        Pros: Preserves meaning, respects document structure
        Cons: Variable chunk sizes, more complex
        Best for: Well-structured documents (articles, docs)
        """
        text = document.content
        chunks = []

        # Split by double newlines (paragraphs)
        paragraphs = re.split(r'\n\s*\n', text)

        current_chunk = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # If adding this paragraph exceeds max size
            if len(current_chunk) + len(para) > max_chunk_size:
                # Save current chunk if it's big enough
                if len(current_chunk) >= min_chunk_size:
                    chunks.append(Chunk(
                        content=current_chunk.strip(),
                        metadata={**document.metadata, "chunk_index": len(chunks)}
                    ))
                    current_chunk = para
                else:
                    # Current chunk is too small, keep adding
                    current_chunk += "\n\n" + para
            else:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para

        # Don't forget the last chunk
        if current_chunk.strip():
            chunks.append(Chunk(
                content=current_chunk.strip(),
                metadata={**document.metadata, "chunk_index": len(chunks)}
            ))

        return chunks

    @staticmethod
    def hierarchical(
        document: Document,
        header_pattern: str = r'^#{1,3}\s+',
        max_chunk_size: int = 1500
    ) -> List[Chunk]:
        """
        Split by document structure (headers, sections).

        Pros: Preserves document hierarchy, great for docs/wikis
        Cons: Requires well-structured documents
        Best for: Documentation, markdown files, technical docs
        """
        text = document.content
        chunks = []

        # Find all headers
        lines = text.split('\n')
        sections = []
        current_section = {"header": None, "content": [], "level": 0}

        for line in lines:
            # Check if line is a header
            header_match = re.match(header_pattern, line)

            if header_match:
                # Save previous section
                if current_section["content"] or current_section["header"]:
                    sections.append(current_section)

                # Start new section
                level = len(header_match.group().strip())
                current_section = {
                    "header": line.strip(),
                    "content": [],
                    "level": level
                }
            else:
                current_section["content"].append(line)

        # Don't forget last section
        if current_section["content"] or current_section["header"]:
            sections.append(current_section)

        # Convert sections to chunks
        for section in sections:
            content = section["header"] or ""
            if section["content"]:
                content += "\n" + "\n".join(section["content"])

            content = content.strip()
            if not content:
                continue

            # Split if too large
            if len(content) > max_chunk_size:
                # Recursively split large sections
                sub_doc = Document(content=content, metadata=document.metadata)
                sub_chunks = ChunkingStrategy.semantic(sub_doc, max_chunk_size)
                for i, sub_chunk in enumerate(sub_chunks):
                    sub_chunk.metadata["section"] = section["header"]
                    sub_chunk.metadata["chunk_index"] = len(chunks)
                    chunks.append(sub_chunk)
            else:
                chunks.append(Chunk(
                    content=content,
                    metadata={
                        **document.metadata,
                        "section": section["header"],
                        "section_level": section["level"],
                        "chunk_index": len(chunks)
                    }
                ))

        return chunks

    @staticmethod
    def sentence(
        document: Document,
        sentences_per_chunk: int = 5,
        overlap_sentences: int = 1
    ) -> List[Chunk]:
        """
        Split by sentences.

        Pros: Never splits mid-sentence
        Cons: May create small chunks, requires sentence detection
        Best for: Conversational content, Q&A pairs
        """
        import nltk
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')

        sentences = nltk.sent_tokenize(document.content)
        chunks = []

        for i in range(0, len(sentences), sentences_per_chunk - overlap_sentences):
            chunk_sentences = sentences[i:i + sentences_per_chunk]
            if chunk_sentences:
                chunks.append(Chunk(
                    content=' '.join(chunk_sentences),
                    metadata={
                        **document.metadata,
                        "chunk_index": len(chunks),
                        "start_sentence": i,
                        "end_sentence": i + len(chunk_sentences)
                    }
                ))

        return chunks


# Choosing the right strategy
CHUNKING_GUIDE = {
    "documentation": {
        "strategy": "hierarchical",
        "params": {"max_chunk_size": 1500},
        "reason": "Technical docs have clear structure"
    },
    "articles": {
        "strategy": "semantic",
        "params": {"max_chunk_size": 1000},
        "reason": "Articles flow paragraph by paragraph"
    },
    "code": {
        "strategy": "fixed_size",
        "params": {"chunk_size": 500, "overlap": 100},
        "reason": "Code structure varies, fixed size is safer"
    },
    "conversations": {
        "strategy": "sentence",
        "params": {"sentences_per_chunk": 10},
        "reason": "Preserve complete exchanges"
    },
    "legal": {
        "strategy": "hierarchical",
        "params": {"max_chunk_size": 2000},
        "reason": "Legal docs have numbered sections"
    },
}
```

---

## Building the RAG Pipeline

### Complete RAG System

```python
"""
Complete RAG system implementation.
"""

from openai import OpenAI
import chromadb
from typing import List, Dict, Any, Optional


class RAGSystem:
    """
    Educational RAG system — complete pipeline for learning.

    NOTE: This is a teaching implementation. A production system would additionally need:
    - Retry logic with exponential backoff for API calls
    - Request caching (embedding cache, response cache)
    - Rate limiting and cost tracking
    - Structured logging and observability
    - Authentication and document-level access control
    - Async processing for indexing large corpora
    See the "Production Readiness Checklist" section below.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_model: str = "text-embedding-3-small",
        llm_model: str = "gpt-4o",
        chunk_strategy: str = "semantic",
        persist_directory: str = "./rag_db"
    ):
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.chunk_strategy = chunk_strategy

        # Initialize OpenAI client
        self.openai = OpenAI()

        # Initialize vector database
        self.chroma_client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    # ==================
    # INDEXING
    # ==================

    def index_documents(
        self,
        documents: List[Document],
        batch_size: int = 100
    ) -> Dict[str, int]:
        """
        Index documents into the RAG system.
        """
        stats = {"documents": len(documents), "chunks": 0}

        all_chunks = []

        # Chunk all documents
        for doc in documents:
            if self.chunk_strategy == "semantic":
                chunks = ChunkingStrategy.semantic(doc)
            elif self.chunk_strategy == "hierarchical":
                chunks = ChunkingStrategy.hierarchical(doc)
            elif self.chunk_strategy == "sentence":
                chunks = ChunkingStrategy.sentence(doc)
            else:
                chunks = ChunkingStrategy.fixed_size(doc)

            all_chunks.extend(chunks)

        stats["chunks"] = len(all_chunks)
        print(f"Created {len(all_chunks)} chunks from {len(documents)} documents")

        # Index in batches
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i:i + batch_size]

            # Generate embeddings
            texts = [chunk.content for chunk in batch]
            embeddings = self._get_embeddings(texts)

            # Prepare for ChromaDB
            ids = [chunk.chunk_id for chunk in batch]
            metadatas = []
            for chunk in batch:
                meta = chunk.metadata.copy()
                meta["text"] = chunk.content  # Store text in metadata for retrieval
                metadatas.append(meta)

            # Upsert to vector DB
            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas
            )

            print(f"Indexed {min(i + batch_size, len(all_chunks))}/{len(all_chunks)} chunks")

        return stats

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts."""
        response = self.openai.embeddings.create(
            input=texts,
            model=self.embedding_model
        )
        return [item.embedding for item in response.data]

    # ==================
    # RETRIEVAL
    # ==================

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: Dict[str, Any] = None,
        min_score: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks for a query.
        """
        # Generate query embedding
        query_embedding = self._get_embeddings([query])[0]

        # Query vector DB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filters
        )

        # Format results
        retrieved = []
        for i in range(len(results["ids"][0])):
            # Convert distance to similarity score.
            # ChromaDB with cosine space returns cosine distance = 1 - cosine_similarity.
            # So similarity = 1 - distance. This is exact when vectors are normalized
            # (which OpenAI embeddings are). For unnormalized vectors, this is an
            # approximation because cosine distance and L2 distance diverge.
            distance = results["distances"][0][i]
            score = 1 - distance

            if score >= min_score:
                retrieved.append({
                    "id": results["ids"][0][i],
                    "text": results["metadatas"][0][i].get("text", ""),
                    "score": score,
                    "metadata": {
                        k: v for k, v in results["metadatas"][0][i].items()
                        if k != "text"
                    }
                })

        return retrieved

    def retrieve_with_reranking(
        self,
        query: str,
        top_k: int = 5,
        initial_k: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Retrieve with cross-encoder reranking for better precision.

        WHY TWO STAGES?
        Stage 1 (bi-encoder / vector search): Encodes query and documents INDEPENDENTLY
        into vectors, then compares with cosine similarity. Fast (sub-second over millions
        of documents) but approximate — it cannot model fine-grained query-document
        interactions because query and document never "see" each other during encoding.

        Stage 2 (cross-encoder reranking): Takes the query-document PAIR as a single
        input and processes both together through a transformer. This allows full
        attention between query and document tokens, producing much more accurate
        relevance scores. But it's O(N) in compute — you can't run it over millions
        of documents. Hence the two-stage approach: cast a wide net with bi-encoder
        (cheap, fast), then re-score the top candidates with cross-encoder (expensive,
        accurate).

        Typical impact: Reranking improves Precision@5 by 10-25% over vector search
        alone, at a cost of ~50-200ms additional latency for 20 candidates.
        """
        # First-stage: retrieve more candidates
        candidates = self.retrieve(query, top_k=initial_k, min_score=0.3)

        if not candidates:
            return []

        # Second-stage: rerank with cross-encoder
        reranked = self._rerank(query, candidates)

        return reranked[:top_k]

    def _rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Rerank candidates using cross-encoder or LLM.
        """
        # Option 1: Use cross-encoder (faster, cheaper)
        try:
            from sentence_transformers import CrossEncoder
            model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

            pairs = [(query, c["text"]) for c in candidates]
            scores = model.predict(pairs)

            for i, score in enumerate(scores):
                candidates[i]["rerank_score"] = float(score)

            candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
            return candidates

        except ImportError:
            # Option 2: Use LLM for reranking (slower, more expensive)
            return self._llm_rerank(query, candidates)

    def _llm_rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Rerank using LLM (fallback when cross-encoder unavailable).
        """
        rerank_prompt = f"""
Rate the relevance of each document to the query on a scale of 1-10.

Query: {query}

Documents:
"""
        for i, candidate in enumerate(candidates):
            rerank_prompt += f"\n[{i}] {candidate['text'][:500]}...\n"

        rerank_prompt += """
Respond with JSON array of scores: [score1, score2, ...]
Only output the JSON array, nothing else.
"""

        response = self.openai.chat.completions.create(
            model="gpt-4o-mini",  # Use cheaper model for reranking
            messages=[{"role": "user", "content": rerank_prompt}],
            temperature=0
        )

        try:
            import json
            scores = json.loads(response.choices[0].message.content)
            for i, score in enumerate(scores):
                if i < len(candidates):
                    candidates[i]["rerank_score"] = score
        except (json.JSONDecodeError, ValueError, IndexError):
            pass  # Keep original scores if parsing fails

        candidates.sort(key=lambda x: x.get("rerank_score", x["score"]), reverse=True)
        return candidates

    # ==================
    # GENERATION
    # ==================

    def generate(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        system_prompt: str = None
    ) -> Dict[str, Any]:
        """
        Generate response using retrieved context.
        """
        # Build context string
        context = "\n\n".join([
            f"[Source {i+1}]: {chunk['text']}"
            for i, chunk in enumerate(context_chunks)
        ])

        # Default system prompt
        if system_prompt is None:
            system_prompt = """You are a helpful assistant that answers questions based on the provided context.

Rules:
1. Only use information from the provided context
2. If the context doesn't contain relevant information, say so
3. Cite your sources using [Source N] notation
4. Be concise but thorough"""

        # Build messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"""Context:
{context}

Question: {query}

Answer based on the context above:"""}
        ]

        # Generate response
        response = self.openai.chat.completions.create(
            model=self.llm_model,
            messages=messages,
            temperature=0.3,  # Lower temp for factual responses
        )

        answer = response.choices[0].message.content

        return {
            "answer": answer,
            "sources": [
                {
                    "text": chunk["text"][:200] + "...",
                    "source": chunk["metadata"].get("source", "Unknown"),
                    "score": chunk["score"]
                }
                for chunk in context_chunks
            ],
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
            }
        }

    # ==================
    # QUERY (Full Pipeline)
    # ==================

    def query(
        self,
        question: str,
        top_k: int = 5,
        use_reranking: bool = True,
        filters: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Full RAG pipeline: retrieve + generate.
        """
        # Retrieve relevant chunks
        if use_reranking:
            chunks = self.retrieve_with_reranking(question, top_k=top_k)
        else:
            chunks = self.retrieve(question, top_k=top_k, filters=filters)

        if not chunks:
            return {
                "answer": "I couldn't find relevant information to answer your question.",
                "sources": [],
                "usage": {}
            }

        # Generate response
        result = self.generate(question, chunks)

        return result

    # ==================
    # UTILITIES
    # ==================

    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        return {
            "total_chunks": self.collection.count(),
            "embedding_model": self.embedding_model,
            "llm_model": self.llm_model
        }

    def delete_collection(self):
        """Delete the entire collection."""
        self.chroma_client.delete_collection(self.collection.name)


# Usage example
def rag_example():
    """Complete RAG example."""
    # Initialize
    rag = RAGSystem(
        collection_name="knowledge_base",
        llm_model="gpt-4o"
    )

    # Load documents
    documents = DocumentLoader.load_directory("./docs", extensions=[".md", ".txt"])

    # Index
    stats = rag.index_documents(documents)
    print(f"Indexed: {stats}")

    # Query
    result = rag.query("How do I configure authentication?")

    print("\n" + "="*50)
    print("ANSWER:")
    print("="*50)
    print(result["answer"])

    print("\n" + "="*50)
    print("SOURCES:")
    print("="*50)
    for source in result["sources"]:
        print(f"- {source['source']} (score: {source['score']:.3f})")


# rag_example()
```

---

## Advanced RAG Techniques

### Query Transformation

```python
"""
Transform queries to improve retrieval.
"""

class QueryTransformer:
    """
    Transform queries for better retrieval.
    """

    def __init__(self, openai_client):
        self.openai = openai_client

    def expand_query(self, query: str) -> List[str]:
        """
        Expand query with related terms and phrasings.
        """
        prompt = f"""
Generate 3 alternative phrasings for this search query.
Keep the same meaning but use different words.

Query: {query}

Output as JSON array: ["phrase1", "phrase2", "phrase3"]
"""
        response = self.openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        )

        import json
        try:
            expansions = json.loads(response.choices[0].message.content)
            return [query] + expansions
        except (json.JSONDecodeError, ValueError, KeyError):
            return [query]

    def decompose_query(self, query: str) -> List[str]:
        """
        Break complex queries into sub-queries.
        """
        prompt = f"""
Break this complex question into simpler sub-questions that can be answered independently.

Question: {query}

Output as JSON array of sub-questions. If the question is already simple, return just that question.
"""
        response = self.openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )

        import json
        try:
            sub_queries = json.loads(response.choices[0].message.content)
            return sub_queries if isinstance(sub_queries, list) else [query]
        except (json.JSONDecodeError, ValueError, KeyError):
            return [query]

    def hypothetical_document(self, query: str) -> str:
        """
        Generate hypothetical document that would answer the query (HyDE).
        Then use this to find real similar documents.
        """
        prompt = f"""
Write a short passage that would perfectly answer this question.
Write as if this passage exists in a documentation site.

Question: {query}

Passage:
"""
        response = self.openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=200
        )

        return response.choices[0].message.content
```

### Hybrid Search (BM25 + Dense Vectors)

Dense vector search excels at semantic similarity ("car" matches "automobile") but can miss exact keyword matches that matter in technical domains. BM25 (sparse keyword search) excels at exact term matching but misses semantic relationships. Hybrid search combines both.

```python
"""
Hybrid search: combining dense vector search with sparse BM25.
"""

from collections import defaultdict
import math
import re
from typing import List, Dict, Any


class BM25:
    """
    BM25 sparse retrieval — term-frequency based ranking.

    BM25 scores documents by how well they match query keywords,
    weighted by how rare those keywords are across the corpus (IDF)
    and normalized by document length.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1  # Term frequency saturation parameter
        self.b = b     # Document length normalization (0=none, 1=full)
        self.doc_freqs = defaultdict(int)  # How many docs contain each term
        self.doc_lengths = []
        self.avg_doc_length = 0
        self.corpus_size = 0
        self.documents = []
        self.tokenized_docs = []

    def _tokenize(self, text: str) -> List[str]:
        """Simple whitespace + lowercase tokenization."""
        return re.findall(r'\w+', text.lower())

    def index(self, documents: List[Dict[str, Any]]):
        """Index documents for BM25 retrieval."""
        self.documents = documents
        self.tokenized_docs = []

        for doc in documents:
            tokens = self._tokenize(doc["text"])
            self.tokenized_docs.append(tokens)
            self.doc_lengths.append(len(tokens))

            # Count unique terms per document (for IDF)
            unique_terms = set(tokens)
            for term in unique_terms:
                self.doc_freqs[term] += 1

        self.corpus_size = len(documents)
        self.avg_doc_length = sum(self.doc_lengths) / self.corpus_size if self.corpus_size else 0

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Search using BM25 scoring."""
        query_tokens = self._tokenize(query)
        scores = []

        for idx, doc_tokens in enumerate(self.tokenized_docs):
            score = 0
            doc_len = self.doc_lengths[idx]

            for term in query_tokens:
                if term not in self.doc_freqs:
                    continue

                # Term frequency in this document
                tf = doc_tokens.count(term)

                # Inverse document frequency
                df = self.doc_freqs[term]
                idf = math.log((self.corpus_size - df + 0.5) / (df + 0.5) + 1)

                # BM25 formula
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_length)
                score += idf * numerator / denominator

            scores.append((idx, score))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in scores[:top_k]:
            result = self.documents[idx].copy()
            result["bm25_score"] = score
            results.append(result)

        return results


class HybridRetriever:
    """
    Combine dense vector search with BM25 sparse search.

    Uses Reciprocal Rank Fusion (RRF) to merge rankings from both systems.
    RRF is preferred over raw score combination because dense and sparse scores
    are on different scales and not directly comparable.
    """

    def __init__(self, rag_system, rrf_k: int = 60):
        self.rag = rag_system
        self.bm25 = BM25()
        self.rrf_k = rrf_k  # RRF constant (higher = less weight to top ranks)
        self._indexed = False

    def index_for_bm25(self):
        """Build BM25 index from existing vector store documents."""
        # Retrieve all documents from ChromaDB
        all_docs = self.rag.collection.get(include=["metadatas"])
        documents = []
        for i, doc_id in enumerate(all_docs["ids"]):
            documents.append({
                "id": doc_id,
                "text": all_docs["metadatas"][i].get("text", ""),
                "metadata": all_docs["metadatas"][i]
            })
        self.bm25.index(documents)
        self._indexed = True

    def search(
        self,
        query: str,
        top_k: int = 5,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search with Reciprocal Rank Fusion.

        dense_weight / sparse_weight control the relative importance.
        For technical docs with specific terminology, increase sparse_weight.
        For conversational queries, increase dense_weight.
        """
        if not self._indexed:
            self.index_for_bm25()

        # Get results from both systems
        dense_results = self.rag.retrieve(query, top_k=top_k * 3, min_score=0.0)
        sparse_results = self.bm25.search(query, top_k=top_k * 3)

        # Reciprocal Rank Fusion
        rrf_scores = defaultdict(float)
        doc_map = {}

        for rank, result in enumerate(dense_results):
            doc_id = result["id"]
            rrf_scores[doc_id] += dense_weight / (self.rrf_k + rank + 1)
            doc_map[doc_id] = result

        for rank, result in enumerate(sparse_results):
            doc_id = result["id"]
            rrf_scores[doc_id] += sparse_weight / (self.rrf_k + rank + 1)
            if doc_id not in doc_map:
                doc_map[doc_id] = result

        # Sort by fused score
        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for doc_id, score in ranked[:top_k]:
            result = doc_map[doc_id].copy()
            result["hybrid_score"] = score
            results.append(result)

        return results


# When to prefer hybrid over pure dense search:
HYBRID_SEARCH_GUIDE = """
Use hybrid search when:
- Your corpus has domain-specific terminology (legal, medical, code)
- Users search with exact identifiers (error codes, product SKUs, API names)
- You need to match acronyms or proper nouns exactly
- Pure vector search misses keyword-critical results

Use dense-only when:
- Queries are conversational and rarely use exact terms
- Your embedding model handles your domain well
- Latency budget is very tight (BM25 adds 10-50ms)
- Corpus is small enough that vector search alone has high recall
"""
```

### Context Window Management

```python
"""
Managing context window limits effectively.
"""

import tiktoken

class ContextManager:
    """
    Manage context to fit within LLM limits.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        max_context_tokens: int = 8000,
        reserved_output_tokens: int = 1000
    ):
        self.encoding = tiktoken.encoding_for_model(model)
        self.max_context_tokens = max_context_tokens
        self.reserved_output = reserved_output_tokens
        self.available_tokens = max_context_tokens - reserved_output_tokens

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.encoding.encode(text))

    def fit_chunks(
        self,
        chunks: List[Dict[str, Any]],
        query: str,
        system_prompt: str
    ) -> List[Dict[str, Any]]:
        """
        Select chunks that fit within context window.
        Prioritizes by relevance score.
        """
        # Count fixed tokens
        fixed_tokens = (
            self.count_tokens(system_prompt) +
            self.count_tokens(query) +
            100  # Overhead for formatting
        )

        available = self.available_tokens - fixed_tokens
        selected = []
        current_tokens = 0

        # Sort by score (assuming already sorted, but ensure)
        chunks = sorted(chunks, key=lambda x: x.get("score", 0), reverse=True)

        for chunk in chunks:
            chunk_tokens = self.count_tokens(chunk["text"])

            if current_tokens + chunk_tokens <= available:
                selected.append(chunk)
                current_tokens += chunk_tokens
            else:
                # Try to fit a truncated version
                remaining_tokens = available - current_tokens
                if remaining_tokens > 100:  # Minimum useful chunk
                    truncated_text = self._truncate_to_tokens(
                        chunk["text"],
                        remaining_tokens
                    )
                    truncated_chunk = chunk.copy()
                    truncated_chunk["text"] = truncated_text
                    truncated_chunk["truncated"] = True
                    selected.append(truncated_chunk)
                break

        return selected

    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within token limit."""
        tokens = self.encoding.encode(text)

        if len(tokens) <= max_tokens:
            return text

        truncated_tokens = tokens[:max_tokens - 3]
        truncated_text = self.encoding.decode(truncated_tokens)

        return truncated_text + "..."

    def create_context_string(
        self,
        chunks: List[Dict[str, Any]],
        include_metadata: bool = True
    ) -> str:
        """
        Create formatted context string from chunks.
        """
        context_parts = []

        for i, chunk in enumerate(chunks):
            part = f"[Source {i+1}]"

            if include_metadata:
                source = chunk.get("metadata", {}).get("source", "Unknown")
                part += f" (from: {source})"

            part += f"\n{chunk['text']}"

            if chunk.get("truncated"):
                part += "\n[Truncated for length]"

            context_parts.append(part)

        return "\n\n---\n\n".join(context_parts)
```

---

## RAG Evaluation

### Evaluation Metrics

```python
"""
Evaluate RAG system performance.
"""

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


@dataclass
class EvaluationResult:
    """Results from RAG evaluation."""
    retrieval_precision: float
    retrieval_recall: float
    answer_relevance: float
    answer_faithfulness: float
    overall_score: float


class RAGEvaluator:
    """
    Evaluate RAG system performance.
    """

    def __init__(self, openai_client):
        self.openai = openai_client

    def evaluate_retrieval(
        self,
        query: str,
        retrieved_chunks: List[str],
        relevant_chunks: List[str]
    ) -> dict:
        """
        Evaluate retrieval quality.
        """
        # Convert to sets for comparison
        retrieved_set = set(retrieved_chunks)
        relevant_set = set(relevant_chunks)

        # Calculate metrics
        true_positives = len(retrieved_set & relevant_set)
        precision = true_positives / len(retrieved_set) if retrieved_set else 0
        recall = true_positives / len(relevant_set) if relevant_set else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

    def evaluate_answer_relevance(
        self,
        question: str,
        answer: str
    ) -> float:
        """
        Evaluate if the answer is relevant to the question.
        Uses LLM-as-judge approach.
        """
        prompt = f"""
Rate how relevant the answer is to the question on a scale of 1-10.

Question: {question}

Answer: {answer}

Consider:
- Does the answer address the question directly?
- Is the answer complete?
- Is there irrelevant information?

Respond with just a number 1-10.
"""
        response = self.openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        try:
            score = int(response.choices[0].message.content.strip())
            return score / 10
        except (ValueError, AttributeError, IndexError):
            return 0.5

    def evaluate_faithfulness(
        self,
        answer: str,
        context: str
    ) -> float:
        """
        Evaluate if the answer is faithful to the context (no hallucination).
        """
        prompt = f"""
Evaluate if the answer is faithful to the provided context.
Check if all claims in the answer can be verified from the context.

Context:
{context}

Answer:
{answer}

Score from 1-10:
- 10: All claims are directly supported by context
- 5: Some claims are supported, some are not
- 1: Answer contains information not in context

Respond with just a number 1-10.
"""
        response = self.openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        try:
            score = int(response.choices[0].message.content.strip())
            return score / 10
        except (ValueError, AttributeError, IndexError):
            return 0.5

    def evaluate_full(
        self,
        question: str,
        answer: str,
        context: str,
        retrieved_ids: List[str] = None,
        relevant_ids: List[str] = None
    ) -> EvaluationResult:
        """
        Full RAG evaluation.
        """
        # Retrieval metrics
        if retrieved_ids and relevant_ids:
            retrieval = self.evaluate_retrieval(question, retrieved_ids, relevant_ids)
            retrieval_precision = retrieval["precision"]
            retrieval_recall = retrieval["recall"]
        else:
            retrieval_precision = retrieval_recall = None

        # Generation metrics
        relevance = self.evaluate_answer_relevance(question, answer)
        faithfulness = self.evaluate_faithfulness(answer, context)

        # Overall score
        scores = [relevance, faithfulness]
        if retrieval_precision is not None:
            scores.extend([retrieval_precision, retrieval_recall])
        overall = np.mean(scores)

        return EvaluationResult(
            retrieval_precision=retrieval_precision,
            retrieval_recall=retrieval_recall,
            answer_relevance=relevance,
            answer_faithfulness=faithfulness,
            overall_score=overall
        )


class RAGTestSuite:
    """
    Test suite for RAG systems.
    """

    def __init__(self, rag_system, evaluator):
        self.rag = rag_system
        self.evaluator = evaluator

    def run_tests(
        self,
        test_cases: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Run test suite on RAG system.

        test_cases: List of {"question": str, "expected_answer": str (optional)}
        """
        results = []

        for i, test in enumerate(test_cases):
            print(f"Running test {i+1}/{len(test_cases)}...")

            # Query RAG system
            response = self.rag.query(test["question"])

            # Build context string
            context = "\n\n".join([s["text"] for s in response["sources"]])

            # Evaluate
            evaluation = self.evaluator.evaluate_full(
                question=test["question"],
                answer=response["answer"],
                context=context
            )

            results.append({
                "question": test["question"],
                "answer": response["answer"],
                "evaluation": evaluation
            })

        # Aggregate metrics
        avg_relevance = np.mean([r["evaluation"].answer_relevance for r in results])
        avg_faithfulness = np.mean([r["evaluation"].answer_faithfulness for r in results])
        avg_overall = np.mean([r["evaluation"].overall_score for r in results])

        return {
            "results": results,
            "summary": {
                "avg_relevance": avg_relevance,
                "avg_faithfulness": avg_faithfulness,
                "avg_overall": avg_overall,
                "tests_run": len(results)
            }
        }
```

### Practical Measurement Guidance

The evaluation code above gives you the tools. Here is how to use them in practice:

**Build a golden test set.** Before optimizing anything, create 20-50 question-answer pairs from your actual documents. For each pair, record which chunks contain the answer. This is your ground truth. Without it, you are tuning blind.

**Measure retrieval and generation separately.** If your answers are bad, you need to know *why*. Is the system retrieving the wrong chunks (retrieval failure), or is the LLM ignoring good chunks (generation failure)? The `evaluate_retrieval` method measures the first; `evaluate_faithfulness` measures the second.

**Track these metrics over time:**

| Metric | What It Tells You | Target Range |
|--------|-------------------|--------------|
| Retrieval Precision@5 | Are the top 5 chunks relevant? | > 0.6 for most use cases |
| Retrieval Recall@10 | Are all relevant chunks in the top 10? | > 0.8 (you want to find everything) |
| Answer Faithfulness | Does the answer stick to the context? | > 0.8 (lower means hallucination risk) |
| Answer Relevance | Does the answer address the question? | > 0.7 |

> **Caveat on targets:** These ranges are rules of thumb, not universal standards. Your acceptable thresholds depend on your use case. A legal compliance system needs faithfulness above 0.95; an internal knowledge base might tolerate 0.7. Define your own thresholds based on the cost of errors in your domain.

**When to re-evaluate:** Re-run your test suite whenever you change chunking strategy, swap embedding models, modify the system prompt, or add significant new documents. Small changes to retrieval parameters (top_k, min_score) can also shift quality unexpectedly.

### Limitations of LLM-as-Judge (And What to Do About It)

The evaluation code above uses an LLM to judge answer quality. This approach has known failure modes you must understand:

1. **Self-preference bias.** GPT-4 rates GPT-4 outputs higher than equivalent outputs from other models. If your judge and generator are the same model family, scores will be inflated. **Mitigation:** Use a different model family for judging than for generation, or use multiple judges and average.

2. **Verbosity bias.** LLM judges tend to rate longer, more detailed answers higher, even when brevity is more appropriate. **Mitigation:** Include "penalize unnecessary verbosity" in your judge prompt, or normalize scores by answer length.

3. **Position bias.** When comparing multiple options, LLMs favor the first or last option presented. **Mitigation:** For A/B comparisons, run twice with swapped order and average.

4. **Inability to verify factual accuracy.** An LLM judge cannot reliably determine if a claim is factually correct—it can only check if the claim appears in the provided context. Factual errors that sound plausible will be missed. **Mitigation:** For high-stakes applications, supplement LLM-as-judge with human evaluation on a sample.

**Human evaluation baseline.** For any RAG system going to production, annotate at least 50 query-answer pairs with human judgments (relevant/not-relevant, faithful/hallucinated). Use these as a calibration set: if your LLM-judge scores diverge significantly from human scores (Pearson correlation < 0.7), your automated evaluation is unreliable and you need to adjust your judge prompt or switch models.

```python
# Example: comparing LLM-judge scores against human annotations
def calibrate_llm_judge(
    human_scores: List[float],  # Human ratings 0-1
    llm_scores: List[float]     # LLM judge ratings 0-1
) -> Dict[str, float]:
    """
    Check if your LLM judge correlates with human judgment.
    If correlation < 0.7, your automated eval is unreliable.
    """
    from scipy import stats

    correlation, p_value = stats.pearsonr(human_scores, llm_scores)
    mean_diff = np.mean(np.array(llm_scores) - np.array(human_scores))

    return {
        "pearson_correlation": correlation,
        "p_value": p_value,
        "mean_bias": mean_diff,  # Positive = LLM over-rates, negative = under-rates
        "reliable": correlation > 0.7 and p_value < 0.05
    }
```

---

## Common RAG Failures and Fixes

```python
"""
Debugging common RAG issues.
"""

RAG_FAILURE_MODES = {
    "Irrelevant retrieval": {
        "symptoms": [
            "Retrieved chunks don't relate to query",
            "Answer says 'context doesn't contain information'"
        ],
        "causes": [
            "Poor chunking (splitting important context)",
            "Embedding model mismatch with content type",
            "Query too different from document language"
        ],
        "fixes": [
            "Try different chunking strategy",
            "Add query expansion",
            "Use HyDE (hypothetical document embeddings)",
            "Tune retrieval threshold"
        ]
    },

    "Hallucination despite context": {
        "symptoms": [
            "Answer contains facts not in retrieved context",
            "Low faithfulness score"
        ],
        "causes": [
            "Context too short/incomplete",
            "Temperature too high",
            "System prompt doesn't enforce grounding"
        ],
        "fixes": [
            "Retrieve more chunks",
            "Lower generation temperature",
            "Strengthen 'only use context' instruction",
            "Add explicit citation requirement"
        ]
    },

    "Lost in the middle": {
        "symptoms": [
            "Important info in middle chunks is ignored",
            "Only first/last chunks influence answer"
        ],
        "causes": [
            "LLM attention bias toward beginning/end",
            "Too much context"
        ],
        "fixes": [
            "Rerank to put best chunks first",
            "Reduce number of chunks",
            "Summarize middle chunks"
        ]
    },

    "Context window overflow": {
        "symptoms": [
            "Error about token limits",
            "Incomplete responses"
        ],
        "causes": [
            "Too many/large chunks",
            "Long system prompt"
        ],
        "fixes": [
            "Implement context window management",
            "Truncate chunks intelligently",
            "Use model with larger context"
        ]
    },

    "Inconsistent answers": {
        "symptoms": [
            "Same question gives different answers",
            "Quality varies widely"
        ],
        "causes": [
            "Non-deterministic retrieval",
            "High generation temperature"
        ],
        "fixes": [
            "Lower temperature to 0 for retrieval ranking",
            "Cache retrieval results",
            "Ensemble multiple queries"
        ]
    }
}


def diagnose_rag_issue(
    question: str,
    retrieved_chunks: List[str],
    answer: str,
    expected_behavior: str
) -> List[str]:
    """
    Diagnose RAG issues.
    Returns list of potential problems and fixes.
    """
    issues = []

    # Check if retrieval is relevant
    if not any(relevant_term in chunk.lower()
               for chunk in retrieved_chunks
               for relevant_term in question.lower().split()):
        issues.append({
            "type": "Irrelevant retrieval",
            "fixes": RAG_FAILURE_MODES["Irrelevant retrieval"]["fixes"]
        })

    # Check for hallucination (simple heuristic)
    # In practice, use LLM-as-judge
    answer_words = set(answer.lower().split())
    context_words = set(' '.join(retrieved_chunks).lower().split())
    novel_words = answer_words - context_words

    if len(novel_words) / len(answer_words) > 0.5:
        issues.append({
            "type": "Potential hallucination",
            "fixes": RAG_FAILURE_MODES["Hallucination despite context"]["fixes"]
        })

    return issues
```

---

## Cost and Latency Analysis

RAG adds cost and latency at every stage. Ignoring this leads to systems that are too expensive or too slow for production.

### Cost Breakdown Per Query

| Component | Operation | Approximate Cost (OpenAI, Jan 2025) | Notes |
|-----------|-----------|-------------------------------------|-------|
| **Query Embedding** | Embed 1 query (~20 tokens) | ~$0.000004 (text-embedding-3-small) | Negligible |
| **Vector Search** | ChromaDB lookup | Free (self-hosted) or ~$0.01/1K queries (managed) | Depends on hosting |
| **Reranking** | Cross-encoder on 20 candidates | ~$0.001 (local model) or ~$0.003 (API) | GPU required for local |
| **LLM Generation** | GPT-4o with ~3K input + ~500 output tokens | ~$0.01-0.02 per query | Dominates total cost |
| **Total per query** | End-to-end | **~$0.01-0.03** | At 10K queries/day = $100-300/day |

### Cost Breakdown for Indexing

| Component | Operation | Approximate Cost | Notes |
|-----------|-----------|-----------------|-------|
| **Document Embedding** | Embed 100K chunks (~50M tokens) | ~$1.00 (text-embedding-3-small) | One-time + re-index |
| **Vector Storage** | Store 100K vectors (1536 dims) | ~500MB disk | Grows linearly with corpus |

### Latency Breakdown Per Query

| Stage | Typical Latency | What Drives It |
|-------|----------------|----------------|
| Query embedding | 50-150ms | API round-trip, model size |
| Vector search (100K docs) | 5-20ms | Index size, hardware |
| BM25 search (if hybrid) | 10-50ms | Corpus size |
| Cross-encoder reranking (20 docs) | 50-200ms | Model size, GPU vs CPU |
| LLM generation | 500-3000ms | Model, input length, output length |
| **Total** | **~700-3500ms** | Generation dominates |

> **Key insight:** LLM generation is 60-80% of both cost and latency. The most impactful optimizations are: (1) use a smaller/cheaper LLM (GPT-4o-mini cuts cost ~10x with moderate quality loss), (2) reduce context length by retrieving fewer, better chunks, and (3) cache frequent queries.

### Cost Optimization Strategies

```python
"""
Practical cost optimization patterns.
"""

import hashlib
import json
from functools import lru_cache


class CostAwareRAG:
    """
    Patterns for reducing RAG cost in production.
    """

    def __init__(self, rag_system):
        self.rag = rag_system
        self.query_cache = {}  # In production, use Redis or similar
        self.embedding_cache = {}

    def cached_query(self, question: str, cache_ttl_seconds: int = 3600) -> dict:
        """
        Cache RAG responses for identical queries.
        In production: use Redis with TTL, not in-memory dict.
        """
        cache_key = hashlib.md5(question.lower().strip().encode()).hexdigest()

        if cache_key in self.query_cache:
            return self.query_cache[cache_key]

        result = self.rag.query(question)
        self.query_cache[cache_key] = result
        return result

    def tiered_generation(self, question: str, chunks: list) -> dict:
        """
        Use cheaper models for simple queries, expensive models for complex ones.

        Simple heuristic: if top chunk has high relevance score and the question
        is straightforward, use GPT-4o-mini. Otherwise, use GPT-4o.
        """
        top_score = chunks[0]["score"] if chunks else 0
        is_simple = (
            top_score > 0.85 and
            len(question.split()) < 20 and
            "?" in question  # Direct question, not complex instruction
        )

        model = "gpt-4o-mini" if is_simple else "gpt-4o"
        # Cost difference: GPT-4o-mini is ~10x cheaper than GPT-4o

        return self.rag.generate(question, chunks, system_prompt=None)
```

## Production Readiness Checklist

Before deploying a RAG system, verify each item. This blog's implementation covers items marked with a checkmark; items marked with an X require additional engineering.

| Category | Item | Covered Here? | Priority |
|----------|------|:---:|----------|
| **Reliability** | Retry logic with exponential backoff for API calls | X | Critical |
| **Reliability** | Graceful degradation when vector DB is unavailable | X | Critical |
| **Performance** | Embedding cache (avoid re-embedding identical texts) | X | High |
| **Performance** | Response cache for frequent queries | X | High |
| **Performance** | Async document indexing for large batches | X | Medium |
| **Cost** | Per-query cost tracking and alerting | X | High |
| **Cost** | Model tiering (cheap model for simple queries) | Partial | Medium |
| **Observability** | Structured logging (query, retrieval scores, latency) | X | Critical |
| **Observability** | Retrieval quality dashboards | X | High |
| **Security** | Document-level access control | X | Critical for enterprise |
| **Security** | PII detection and filtering in responses | X | Critical for regulated industries |
| **Data** | Incremental document update pipeline | X | High |
| **Data** | Stale document detection and removal | X | Medium |
| **Quality** | Automated regression test suite | Partial | High |
| **Quality** | Human feedback collection loop | X | Medium |

---

## Interview Preparation

### Career Mapping

RAG knowledge maps directly to these industry roles:

| Role | What They Need from RAG | What This Blog Gives Them |
|------|------------------------|--------------------------|
| **ML/AI Engineer** | Build and optimize retrieval pipelines | Chunking strategies, reranking, hybrid search, evaluation metrics |
| **AI Platform Engineer** | Scale RAG infrastructure, manage vector DBs | Architecture overview, cost analysis, production checklist |
| **Solutions Architect** | Design RAG systems for enterprise clients | When-to-use/not-use analysis, RAG vs. fine-tuning tradeoffs |
| **Backend Engineer (AI-adjacent)** | Integrate RAG into existing applications | Complete pipeline code, API patterns, context management |
| **Data Engineer** | Build document ingestion and processing pipelines | Document loading, chunking, indexing strategies |

> **Framework context:** This blog builds RAG from scratch to teach the underlying mechanisms. In production, many teams use frameworks like **LangChain**, **LlamaIndex**, or **Haystack** which abstract these components. Understanding the internals taught here is essential for debugging and customizing those frameworks — they are convenience layers, not magic. When a LangChain retrieval chain returns bad results, you need to know whether the problem is chunking, embedding quality, or context construction. This blog gives you that diagnostic ability.

### Concept Questions

**Q1: Explain RAG and why it's important.**

*Answer:* RAG adds a non-parametric memory layer to LLMs. Instead of relying solely on knowledge baked into model weights (parametric), RAG retrieves relevant documents at query time and injects them into the prompt. The pipeline: (1) embed the user query, (2) retrieve top-K similar chunks from a vector database, (3) construct a prompt with query + retrieved context, (4) generate an answer conditioned on that context. RAG matters because it addresses three hard LLM limitations—knowledge cutoff, hallucination, and private data access—without retraining. But it is not a universal fix: it adds latency (1-3s), cost ($0.01-0.03/query), and fails when the task requires full-corpus reasoning rather than snippet retrieval.

**Q2: How would you choose a chunking strategy?**

*Answer:* It depends on document type and use case. For well-structured docs (technical documentation, wikis), use hierarchical chunking that respects headers/sections. For flowing prose (articles, blog posts), use semantic chunking by paragraphs. For code, fixed-size chunking with overlap is safest. For conversations, sentence-based chunking preserves context. Key principles: never split mid-sentence, preserve semantic units, include enough context per chunk (typically 200-500 tokens), and test with real queries.

**Q3: What's the "lost in the middle" problem?**

*Answer:* Studies (e.g., Liu et al., "Lost in the Middle," 2023) found that LLMs pay more attention to content at the beginning and end of their context window, often missing or underweighting information in the middle. In RAG, this means your 3rd-5th ranked chunks may be ignored even if relevant. Fixes include: reranking to put most relevant chunks first, reducing total chunks retrieved, using a recursive summarization approach, or explicitly instructing the model to consider all provided context equally.

**Q4: How do you evaluate a RAG system?**

*Answer:* Evaluate both retrieval and generation: For retrieval—measure precision (relevant chunks / retrieved chunks), recall (retrieved relevant / total relevant), and MRR (position of first relevant result). For generation—measure faithfulness (is the answer supported by context?), relevance (does it answer the question?), and completeness. Use LLM-as-judge for semantic evaluation or human evaluation for production. Build a test set with known answers and measure against it regularly.

**Q5: When would you NOT use RAG?**

*Answer:* RAG is wrong when: (1) The task requires reasoning over the entire corpus, not snippets — e.g., "summarize all 200 reports"; retrieval gives you fragments, not the full picture. (2) Latency is critical — RAG adds 1-3 seconds; if you need sub-200ms, use pre-computed answers or traditional search. (3) The document set fits in the context window — if your entire knowledge base is 50K tokens, just stuff it in the prompt; retrieval adds complexity without benefit. (4) The task is about style, not knowledge — fine-tuning is better for "write in our brand voice." (5) No semantic query-document match exists — for math problems or abstract reasoning, embedding similarity is meaningless.

**Q6: How does hybrid search differ from pure vector search, and when would you choose it?**

*Answer:* Pure vector search (dense retrieval) encodes queries and documents into embedding vectors and ranks by cosine similarity. It excels at semantic matching ("automobile" matches "car") but can miss exact keyword matches. BM25 (sparse retrieval) ranks by term frequency and inverse document frequency — it excels at exact matches but misses synonyms. Hybrid search combines both using rank fusion (typically Reciprocal Rank Fusion). Choose hybrid when your domain has specific terminology (legal, medical, code), when users search by exact identifiers (error codes, product SKUs), or when pure vector search is missing keyword-critical results. Choose dense-only when queries are conversational and latency budget is very tight.

### Coding Question

**Q7: Implement simple retrieval with reranking.**

```python
def retrieve_and_rerank(
    query: str,
    vector_store,  # Has .search(embedding, k) method
    embedding_model,  # Has .embed(text) method
    reranker,  # Has .score(query, doc) method
    initial_k: int = 20,
    final_k: int = 5
) -> list:
    """
    Two-stage retrieval with reranking.
    """
    # Stage 1: Vector search
    query_embedding = embedding_model.embed(query)
    candidates = vector_store.search(query_embedding, k=initial_k)

    # Stage 2: Rerank with cross-encoder
    scored_candidates = []
    for candidate in candidates:
        score = reranker.score(query, candidate["text"])
        scored_candidates.append({
            **candidate,
            "rerank_score": score
        })

    # Sort by rerank score
    scored_candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

    return scored_candidates[:final_k]
```

---

## Exercises

### Exercise 1: Build a Document QA System
Build a RAG system for a collection of PDFs:
- Load and chunk documents
- Index in ChromaDB
- Query with citations

### Exercise 2: Compare Chunking Strategies
Test different chunking strategies on the same document set:
- Measure retrieval precision
- Compare chunk sizes
- Evaluate answer quality

### Exercise 3: Implement Query Expansion
Build query expansion with:
- Synonym expansion
- Related term generation
- HyDE implementation

### Exercise 4: RAG Evaluation Pipeline
Create an automated evaluation pipeline:
- Generate test questions from documents
- Run RAG and evaluate
- Track metrics over time

### Exercise 5: Multi-Turn RAG
Extend RAG for conversations:
- Maintain conversation history
- Contextualize follow-up questions
- Track which sources were cited

---

## Summary

### Key Takeaways

1. **RAG adds non-parametric memory to LLMs:** It retrieves context at query time rather than baking knowledge into weights — understand this mechanism, not just the acronym
2. **RAG is not always the answer:** Know the six scenarios where RAG is the wrong tool (full-corpus reasoning, tight latency, small corpora, style tasks, abstract reasoning, rapidly changing data)
3. **Chunking is critical:** Poor chunking ruins everything downstream — choose strategy based on document type and test empirically
4. **Two-stage retrieval is worth the cost:** Bi-encoder for fast candidate retrieval, cross-encoder for accurate reranking — expect 10-25% precision improvement
5. **Hybrid search catches what vectors miss:** BM25 + dense vectors via Reciprocal Rank Fusion handles keyword-critical queries
6. **Manage context carefully:** Token-aware chunk selection, truncation, and prioritization prevent overflow and improve answer quality
7. **Evaluate retrieval and generation separately:** If answers are bad, know whether retrieval failed or generation failed — these have different fixes
8. **Calibrate your evaluation:** LLM-as-judge has known biases; validate against human annotations before trusting automated scores
9. **Know the cost:** LLM generation dominates both cost and latency; optimize with caching, model tiering, and fewer/better chunks

### What's Next

In Blog 18, we'll explore Function Calling and AI Agents:
- Tool use with LLMs
- Building autonomous agents
- ReAct and planning architectures

---

## Self-Assessment Rubric

Rate yourself honestly after completing this blog:

| Criteria | Excellent (9-10) | Good (7-8) | Needs Work (5-6) |
| ---------- | ------------------ | ------------ | ------------------ |
| **RAG Architecture** | Can explain the full pipeline and design choices for each component | Understands query-time retrieval + generation flow | Confuses RAG with fine-tuning |
| **Chunking Strategies** | Can select and justify strategy for different document types | Understands semantic vs. fixed-size tradeoffs | Uses one strategy for everything |
| **Retrieval Optimization** | Can implement reranking, query expansion, and hybrid search | Understands why two-stage retrieval helps | Only uses basic vector similarity |
| **Context Management** | Can fit chunks within token limits while preserving relevance | Understands token counting and truncation | Ignores context window limits |
| **Evaluation** | Can measure retrieval precision/recall and answer faithfulness | Can run the evaluation code and interpret results | Cannot distinguish good RAG from bad |
| **Debugging** | Can diagnose failure modes and apply targeted fixes | Recognizes common failure patterns | Cannot identify why RAG quality is poor |

### What This Blog Does Well

- Provides a complete, end-to-end RAG pipeline implementation (document loading through response generation)
- Covers four distinct chunking strategies with guidance on when to use each
- Includes a practical evaluation framework with both retrieval and generation metrics, plus guidance on LLM-judge calibration
- Documents five common failure modes with diagnostic signals and fixes
- Demonstrates advanced techniques (reranking, query transformation, HyDE, hybrid search) that materially improve RAG quality
- Provides explicit cost and latency analysis with per-query and per-indexing breakdowns
- Includes a "When RAG Is the Wrong Choice" section with concrete anti-patterns
- Maps RAG knowledge to specific career roles and contextualizes against production frameworks

### Where This Blog Falls Short

- All code uses the OpenAI API exclusively -- no examples for open-source embeddings or local LLMs, limiting portability
- The chunking strategies are demonstrated in isolation but never compared empirically on the same dataset
- No multi-turn conversation support -- the RAG system is stateless and handles only single-turn queries
- The production readiness checklist identifies gaps but does not implement solutions for caching, retries, or access control

### Architect Sanity Checks

### Check 1: Can They Build a Working RAG Pipeline?
**Question**: Could this person build a RAG system that ingests documents, retrieves relevant chunks, and generates grounded answers?
**Answer: Yes.** The blog provides a complete single-turn RAG pipeline covering ingestion, chunking, embedding, retrieval (including hybrid search with BM25), reranking, context management, and generation. The code is structurally sound and demonstrates good practices (source attribution, evaluation metrics, cost-aware query patterns). The implementation is tightly coupled to OpenAI APIs, but the blog explicitly contextualizes this as an educational choice and points readers toward production frameworks for abstraction. A reader who completes this blog can build a functional RAG system and understands what additional engineering is needed for production (via the Production Readiness Checklist).

### Check 2: Can They Debug RAG Quality Issues?
**Question**: When retrieval quality drops or hallucination increases, can they diagnose and fix the problem?
**Answer: Yes.** The failure modes section covers five common issues with diagnostic signals and fixes. The evaluation framework provides retrieval precision/recall, LLM-based faithfulness scoring, and a calibration method for validating LLM-judge reliability against human annotations. The reader understands the difference between retrieval failure and generation failure, knows how to measure both, and has concrete guidance on when to re-evaluate. The diagnostic code still uses a simplified heuristic, but the blog explicitly labels it as such and recommends LLM-as-judge for production-grade detection.

### Check 3: Do They Understand RAG vs. Alternatives?
**Question**: Can they recommend when to use RAG vs. fine-tuning vs. longer context windows vs. other approaches?
**Answer: Yes.** The blog includes a dedicated "When RAG Is the Wrong Choice" table covering six specific scenarios where RAG fails (full-corpus reasoning, tight latency budgets, small corpora, style tasks, abstract reasoning, rapidly changing data). The Manager's Summary compares RAG vs. fine-tuning. The interview Q&A addresses limitations explicitly. A reader can make an informed recommendation about whether RAG fits a given use case.
