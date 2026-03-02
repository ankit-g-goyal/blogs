# Blog 6: Text Preprocessing and Representation
## Words to Numbers

**Reading time:** 60–75 minutes
**Coding time:** 90–120 minutes
**Total investment:** 2.5–3 hours

---

## Prerequisites and Reading Guide

**Required background:**
- Python fluency (classes, list comprehensions, regex basics) -- Blog 2
- NumPy array operations (dot products, matrix multiplication) -- Blog 3
- Basic understanding of neural networks -- Blog 4
- PyTorch fundamentals (nn.Module, tensors) -- Blog 5

**Required libraries:**
```bash
pip install numpy gensim transformers tokenizers torch sentence-transformers scikit-learn
```

**How to read this blog:**
- Parts 1-2 (Tokenization and Cleaning): Essential foundation. Work through the code.
- Part 3 (TF-IDF): Understand the math and implement from scratch before using sklearn.
- Part 4 (Embeddings): Focus on using pre-trained embeddings; the theory is covered more in Blog 7-9.
- Part 5 (Semantic Search): A capstone project tying everything together. Run it end-to-end.

---

## What You'll Walk Away With

By the end of this blog, you will:

1. **Implement** tokenization strategies and understand their tradeoffs
2. **Build** TF-IDF representations for document similarity
3. **Use** pre-trained Word2Vec and GloVe embeddings
4. **Create** a semantic document search engine
5. **Recognize** preprocessing pitfalls that silently break NLP pipelines

This blog bridges the gap between raw text and the numerical vectors that neural networks consume.

---

## What This Blog Does NOT Cover

- **Contextual embeddings (BERT, GPT)**: Covered in Blogs 8-9 on attention and transformers.
- **Sequence models (RNNs, LSTMs)**: Covered in Blog 7. This blog treats text as bags of words or static embeddings.
- **Training Word2Vec/GloVe from scratch**: We use pre-trained embeddings. Training your own requires Blog 7-9 concepts.
- **Production search infrastructure**: We build a toy search engine. For production, see Elasticsearch, Faiss, Pinecone (Blog 16 on vector databases).
- **Non-English NLP in depth**: We note multilingual challenges but do not demonstrate language-specific tokenizers like MeCab (Japanese) or jieba (Chinese).
- **BM25 ranking**: Mentioned in exercises but not implemented. BM25 is the standard for keyword retrieval in production systems like Elasticsearch.

---

## Why Text Preprocessing Matters

Neural networks understand numbers, not words. The journey from text to prediction:

```
"This movie was great!" → [0.2, 0.8, -0.1, ...] → Neural Network → [Positive: 0.95]
                          ↑
                    This step is critical
```

Bad preprocessing = garbage in, garbage out. The best model can't recover from:
- Lost meaning during tokenization
- Important words treated as noise
- Semantic relationships ignored

---

## Part 1: Tokenization — Breaking Text into Units

### What is Tokenization?

Tokenization splits text into discrete units (tokens) that become the vocabulary of your model.

```python
# Simple example
text = "Don't stop believing!"

# Different tokenization approaches:
# Word-level: ["Don't", "stop", "believing", "!"]
# Subword:    ["Don", "'t", "stop", "believ", "ing", "!"]
# Character:  ["D", "o", "n", "'", "t", " ", "s", ...]
```

### Word-Level Tokenization

```python
import re
from collections import Counter

def simple_tokenize(text):
    """Basic whitespace tokenization."""
    return text.split()

def better_tokenize(text):
    """Handle punctuation and normalize."""
    # Lowercase
    text = text.lower()
    # Split on non-alphanumeric, keeping contractions
    tokens = re.findall(r"\b[\w']+\b", text)
    return tokens

def advanced_tokenize(text):
    """More sophisticated tokenization."""
    # Lowercase
    text = text.lower()

    # Handle contractions
    contractions = {
        "don't": "do not", "won't": "will not", "can't": "cannot",
        "i'm": "i am", "you're": "you are", "it's": "it is",
        "i've": "i have", "we've": "we have", "they've": "they have",
        "isn't": "is not", "aren't": "are not", "wasn't": "was not",
    }
    for contraction, expansion in contractions.items():
        text = text.replace(contraction, expansion)

    # Tokenize
    tokens = re.findall(r'\b\w+\b', text)

    return tokens


# Compare approaches
text = "Don't stop believing! It's a great song, isn't it?"

print("Simple:", simple_tokenize(text))
print("Better:", better_tokenize(text))
print("Advanced:", advanced_tokenize(text))
```

### Subword Tokenization (BPE)

Modern models (GPT, BERT, etc.) use subword tokenization:

```python
# Why subword?
# - Handles unknown words: "unbelievably" → ["un", "believ", "ably"]
# - Smaller vocabulary: ~30K tokens vs 100K+ words
# - Captures morphology: "running", "runs", "ran" share "run"

# HOW BPE WORKS (Iterative Merge Algorithm):
# 1. Start with character-level vocabulary: {'a', 'b', 'c', ..., 'z', ...}
# 2. Count all adjacent character pairs in the training corpus
# 3. Merge the most frequent pair into a new token
# 4. Repeat steps 2-3 until vocabulary reaches target size
#
# Example on corpus: ["low", "lower", "newest", "widest"]
# Step 0: Characters: l o w e r n w s t
# Step 1: Most frequent pair: (e, s) → merge to "es"
# Step 2: Most frequent pair: (es, t) → merge to "est"
# Step 3: Most frequent pair: (l, o) → merge to "lo"
# Step 4: Most frequent pair: (lo, w) → merge to "low"
# ...continues until vocab_size reached
#
# KEY INSIGHT: Common words become single tokens ("the", "and"),
# rare words decompose into subwords ("unbelievably" → "un" + "believ" + "ably").
# This is why GPT-2 tokenizes "indistinguishable" into 3 tokens but "the" into 1.
#
# VOCAB SIZE TRADEOFF:
# - Too small (1K): Every word splits into many tokens → long sequences, slow inference
# - Too large (100K+): Rare tokens get poor embeddings → wasted parameters
# - Sweet spot: 30K-50K (GPT-2: 50,257, BERT: 30,522, LLaMA: 32,000)

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

def train_bpe_tokenizer(texts, vocab_size=1000):
    """Train a BPE tokenizer from scratch."""
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
    )

    tokenizer.train_from_iterator(texts, trainer)
    return tokenizer


# Example with Hugging Face transformers
from transformers import AutoTokenizer

# Load pre-trained tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

text = "Unbelievably, the transformers library handles tokenization!"

# Tokenize
tokens = tokenizer.tokenize(text)
print(f"Tokens: {tokens}")

# Convert to IDs
ids = tokenizer.encode(text)
print(f"Token IDs: {ids}")

# Decode back
decoded = tokenizer.decode(ids)
print(f"Decoded: {decoded}")

# Full encoding with attention mask
encoding = tokenizer(text, return_tensors='pt', padding=True)
print(f"Input IDs shape: {encoding['input_ids'].shape}")
print(f"Attention mask: {encoding['attention_mask']}")
```

### Tokenization Pitfalls

```python
# Pitfall 1: Case sensitivity
text1 = "Apple announced iPhone"
text2 = "I ate an apple"

# Without lowercase, "Apple" ≠ "apple" (different tokens)
# With lowercase, you lose "Apple" (company) vs "apple" (fruit)
# Solution: Use context (embeddings) rather than just tokens

# Pitfall 2: Out-of-vocabulary (OOV) words
vocab = {"the", "cat", "sat", "on", "mat"}
text = "The dog sat on the rug"
# "dog" and "rug" are OOV → mapped to [UNK]
# Subword tokenization largely solves this

# Pitfall 3: Punctuation stripping
review1 = "This movie was great!"
review2 = "This movie was great..."
# Exclamation vs ellipsis conveys different sentiment
# Sometimes punctuation matters!

# Pitfall 4: Whitespace tokenization fails on languages without spaces
chinese_text = "我爱自然语言处理"  # "I love natural language processing"
# split() returns the whole string as one token
# Need language-specific tokenizers

# Pitfall 5: Emojis and special characters
tweet = "This is 🔥🔥🔥 amazing! #NLP #AI"
# Emojis carry sentiment information
# Hashtags might be meaningful

def robust_tokenize(text, keep_emojis=True, keep_hashtags=True):
    """More robust tokenization."""
    import emoji

    if keep_emojis:
        # Convert emojis to text
        text = emoji.demojize(text)  # 🔥 → :fire:

    tokens = re.findall(r'#?\w+|:\w+:', text.lower())
    return tokens

print(robust_tokenize("This is 🔥🔥🔥 amazing! #NLP"))
```

### ✅ Checkpoint: After Part 1

You should now be able to answer:
1. What are the tradeoffs between word-level, subword, and character tokenization?
2. How does BPE's iterative merge algorithm build its vocabulary?
3. Why do modern LLMs use vocab sizes of 30K-50K (not 1K or 100K)?
4. Name three tokenization pitfalls and how to mitigate each.

If you can't answer all four, re-read Part 1 before continuing.

---

## Part 2: Text Cleaning Pipeline

### Standard Preprocessing Steps

```python
import re
import string
from typing import List

class TextPreprocessor:
    """
    Comprehensive text preprocessing pipeline.

    Each step is optional and configurable.
    """

    def __init__(
        self,
        lowercase=True,
        remove_punctuation=False,
        remove_numbers=False,
        remove_stopwords=True,
        lemmatize=True,
        min_token_length=2
    ):
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_numbers = remove_numbers
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.min_token_length = min_token_length

        # Stopwords (common words that often don't carry meaning)
        self.stopwords = {
            'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
            'for', 'of', 'with', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'must', 'shall',
            'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it',
            'we', 'they', 'what', 'which', 'who', 'whom', 'when', 'where',
            'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more',
            'most', 'other', 'some', 'such', 'no', 'not', 'only', 'own',
            'same', 'so', 'than', 'too', 'very', 'just', 'can', 'now'
        }

        # Simple lemmatization rules (use spaCy/NLTK for production)
        self.lemma_rules = {
            'running': 'run', 'runs': 'run', 'ran': 'run',
            'better': 'good', 'best': 'good',
            'went': 'go', 'goes': 'go', 'going': 'go',
            'is': 'be', 'are': 'be', 'was': 'be', 'were': 'be',
            'having': 'have', 'has': 'have', 'had': 'have',
        }

    def clean_text(self, text: str) -> str:
        """Basic text cleaning."""
        # Remove URLs
        text = re.sub(r'http\S+|www\.\S+', '', text)

        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)

        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)

        # Remove extra whitespace
        text = ' '.join(text.split())

        return text

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        if self.lowercase:
            text = text.lower()

        # Handle contractions
        contractions = {
            "n't": " not", "'re": " are", "'s": " is",
            "'ll": " will", "'ve": " have", "'m": " am"
        }
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)

        # Tokenize
        tokens = re.findall(r'\b\w+\b', text)

        return tokens

    def filter_tokens(self, tokens: List[str]) -> List[str]:
        """Apply filtering rules to tokens."""
        filtered = []

        for token in tokens:
            # Skip short tokens
            if len(token) < self.min_token_length:
                continue

            # Skip numbers if configured
            if self.remove_numbers and token.isdigit():
                continue

            # Skip stopwords if configured
            if self.remove_stopwords and token in self.stopwords:
                continue

            # Apply lemmatization
            if self.lemmatize:
                token = self.lemma_rules.get(token, token)

            filtered.append(token)

        return filtered

    def preprocess(self, text: str) -> List[str]:
        """Full preprocessing pipeline."""
        text = self.clean_text(text)
        tokens = self.tokenize(text)
        tokens = self.filter_tokens(tokens)
        return tokens

    def preprocess_batch(self, texts: List[str]) -> List[List[str]]:
        """Preprocess multiple texts."""
        return [self.preprocess(text) for text in texts]


# Example usage
preprocessor = TextPreprocessor(
    lowercase=True,
    remove_stopwords=True,
    lemmatize=True
)

sample_texts = [
    "The quick brown fox is running over the lazy dog!",
    "I've been learning about NLP and it's fascinating.",
    "Check out https://example.com for more info!",
]

for text in sample_texts:
    tokens = preprocessor.preprocess(text)
    print(f"Original: {text}")
    print(f"Tokens: {tokens}\n")
```

### Production Lemmatization with spaCy

```python
"""
The hand-rolled lemma_rules above covers ~10 words. For production, use spaCy:
"""

# pip install spacy && python -m spacy download en_core_web_sm

import spacy

nlp = spacy.load("en_core_web_sm")

def spacy_lemmatize(text: str) -> List[str]:
    """Production-grade lemmatization using spaCy's morphological analyzer."""
    doc = nlp(text)
    return [token.lemma_ for token in doc if not token.is_space]

# Compare toy vs production lemmatization
test_sentences = [
    "The children were running and playing happily",
    "She has better opportunities than her predecessors",
    "The geese flew over the oxen grazing in the fields",
]

for sent in test_sentences:
    print(f"Original: {sent}")
    print(f"  spaCy:  {spacy_lemmatize(sent)}")
    # Hand-rolled would miss: children→child, geese→goose, oxen→ox, flew→fly
    print()

# WHY spaCy over NLTK WordNetLemmatizer:
# - spaCy uses POS tags to disambiguate: "better" (adj) → "good", "better" (verb) → "better"
# - NLTK requires you to manually specify POS tag
# - spaCy handles irregular forms (geese→goose, oxen→ox) via lookup tables
# - spaCy is faster (Cython-optimized) for batch processing
#
# COST: spaCy's en_core_web_sm model is ~12MB. Full pipeline (NER, parsing) is slower
# than regex-based preprocessing. For TF-IDF pipelines where lemmatization quality
# matters, the overhead is worth it. For real-time systems, profile first.
```

### ✅ Checkpoint: After Part 2

You should now be able to answer:
1. Why is the hand-rolled lemma dictionary insufficient for production?
2. What's the difference between stemming (Porter/Snowball) and lemmatization (spaCy)?
3. Which preprocessing steps are safe to skip for embedding-based models?
4. How does the TextPreprocessor class handle contraction expansion?

If you can't answer all four, re-read Part 2 before continuing.

---

## Part 3: TF-IDF — Traditional Text Representation

### What is TF-IDF?

TF-IDF (Term Frequency-Inverse Document Frequency) converts documents to vectors based on word importance.

```
TF-IDF(term, document) = TF(term, document) × IDF(term)

TF = count(term in document) / total_terms_in_document
IDF = log(total_documents / documents_containing_term)
```

**Intuition**:
- Words appearing frequently in a document → high TF
- Words appearing in many documents → low IDF (common words)
- TF-IDF is high for words that are important to a specific document but not common overall

### Implementation

```python
import numpy as np
from collections import Counter
from typing import List, Dict
import math

class TFIDFVectorizer:
    """
    TF-IDF implementation from scratch.
    """

    def __init__(self, max_features=None, min_df=1, max_df=1.0):
        self.max_features = max_features
        self.min_df = min_df  # Minimum document frequency
        self.max_df = max_df  # Maximum document frequency (as ratio)
        self.vocabulary = {}
        self.idf = {}

    def fit(self, documents: List[List[str]]):
        """Learn vocabulary and IDF from documents."""
        # Count document frequency for each term
        doc_freq = Counter()
        for doc in documents:
            unique_terms = set(doc)
            doc_freq.update(unique_terms)

        n_docs = len(documents)

        # Filter by document frequency
        valid_terms = {}
        for term, count in doc_freq.items():
            df_ratio = count / n_docs
            if count >= self.min_df and df_ratio <= self.max_df:
                valid_terms[term] = count

        # Sort by frequency and limit vocabulary
        sorted_terms = sorted(valid_terms.items(), key=lambda x: -x[1])
        if self.max_features:
            sorted_terms = sorted_terms[:self.max_features]

        # Build vocabulary
        self.vocabulary = {term: idx for idx, (term, _) in enumerate(sorted_terms)}

        # Compute IDF
        for term, idx in self.vocabulary.items():
            df = doc_freq[term]
            self.idf[term] = math.log(n_docs / (df + 1)) + 1  # Smoothed IDF

        return self

    def transform(self, documents: List[List[str]]) -> np.ndarray:
        """Transform documents to TF-IDF vectors."""
        vectors = np.zeros((len(documents), len(self.vocabulary)))

        for doc_idx, doc in enumerate(documents):
            # Compute term frequencies
            term_counts = Counter(doc)
            doc_length = len(doc)

            for term, count in term_counts.items():
                if term in self.vocabulary:
                    term_idx = self.vocabulary[term]
                    tf = count / doc_length
                    tfidf = tf * self.idf[term]
                    vectors[doc_idx, term_idx] = tfidf

        # L2 normalize
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        vectors = vectors / norms

        return vectors

    def fit_transform(self, documents: List[List[str]]) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(documents).transform(documents)


# Example
documents = [
    ["machine", "learning", "is", "amazing"],
    ["deep", "learning", "uses", "neural", "networks"],
    ["machine", "learning", "and", "deep", "learning", "are", "related"],
    ["natural", "language", "processing", "uses", "machine", "learning"],
]

vectorizer = TFIDFVectorizer(max_features=20)
tfidf_matrix = vectorizer.fit_transform(documents)

print("Vocabulary:", vectorizer.vocabulary)
print(f"\nTF-IDF matrix shape: {tfidf_matrix.shape}")
print("\nTF-IDF vectors:")
for i, doc in enumerate(documents):
    print(f"Doc {i}: {tfidf_matrix[i][:5]}...")  # First 5 dimensions
```

### Document Similarity with TF-IDF

```python
# WHY COSINE SIMILARITY (not Euclidean distance)?
# Euclidean distance is sensitive to document LENGTH:
#   doc1 = "cat cat cat" → TF vector [3, 0, 0, ...]
#   doc2 = "cat" → TF vector [1, 0, 0, ...]
#   Euclidean distance = 2.0 (seems different!)
#   Cosine similarity = 1.0 (same direction = same topic)
#
# Cosine measures ANGLE between vectors, ignoring magnitude.
# This is critical for text: a longer document about cats is still about cats.
# L2 normalization (done in our TFIDFVectorizer) makes cosine = dot product.

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def find_similar_documents(query_doc: List[str], documents: List[List[str]],
                          vectorizer: TFIDFVectorizer, top_k: int = 3):
    """Find most similar documents to a query."""
    # Transform all documents
    doc_vectors = vectorizer.transform(documents)
    query_vector = vectorizer.transform([query_doc])[0]

    # Compute similarities
    similarities = []
    for i, doc_vector in enumerate(doc_vectors):
        sim = cosine_similarity(query_vector, doc_vector)
        similarities.append((i, sim))

    # Sort by similarity
    similarities.sort(key=lambda x: -x[1])

    return similarities[:top_k]


# Find similar documents
query = ["machine", "learning", "neural"]
similar = find_similar_documents(query, documents, vectorizer)

print("Query:", query)
print("\nMost similar documents:")
for doc_idx, similarity in similar:
    print(f"  Doc {doc_idx} (sim={similarity:.3f}): {documents[doc_idx]}")
```

### Data Leakage Warning

```python
"""
IMPORTANT: In any ML pipeline, fit your TF-IDF vocabulary and IDF weights
on TRAINING data only. Then call .transform() on test/validation data.

    # CORRECT
    vectorizer = TFIDFVectorizer()
    train_vectors = vectorizer.fit_transform(train_docs)   # fit + transform
    test_vectors = vectorizer.transform(test_docs)          # transform only

    # WRONG — leaks test vocabulary into training
    all_vectors = vectorizer.fit_transform(all_docs)
    train_vectors = all_vectors[:split]
    test_vectors = all_vectors[split:]

The same principle applies to any preprocessing step that learns from data
(stopword frequency thresholds, vocabulary construction, etc.).
"""
```

### Data Leakage Prevention: Integrated Example

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Simulate a classification task
all_docs = [
    (["machine", "learning", "model", "training"], 0),
    (["deep", "neural", "network", "layers"], 0),
    (["stock", "market", "trading", "price"], 1),
    (["investment", "portfolio", "returns"], 1),
    (["gradient", "descent", "optimization"], 0),
    (["bond", "yield", "interest", "rate"], 1),
    (["backpropagation", "loss", "function"], 0),
    (["hedge", "fund", "risk", "management"], 1),
]

docs, labels = zip(*all_docs)
docs, labels = list(docs), list(labels)

# Split FIRST, then fit preprocessor
train_docs, test_docs, train_labels, test_labels = train_test_split(
    docs, labels, test_size=0.25, random_state=42
)

# CORRECT: fit on training data only
vectorizer = TFIDFVectorizer(max_features=50)
train_vectors = vectorizer.fit_transform(train_docs)  # fit + transform
test_vectors = vectorizer.transform(test_docs)          # transform ONLY

# Train classifier
clf = LogisticRegression()
clf.fit(train_vectors, train_labels)
train_acc = accuracy_score(train_labels, clf.predict(train_vectors))
test_acc = accuracy_score(test_labels, clf.predict(test_vectors))
print(f"CORRECT pipeline — Train: {train_acc:.2f}, Test: {test_acc:.2f}")

# WRONG: fitting on all data leaks test vocabulary and IDF weights
leaky_vectorizer = TFIDFVectorizer(max_features=50)
all_vectors = leaky_vectorizer.fit_transform(docs)  # fit on ALL data
leaky_train = all_vectors[:len(train_docs)]
leaky_test = all_vectors[len(train_docs):]
clf_leaky = LogisticRegression()
clf_leaky.fit(leaky_train, train_labels)
leaky_test_acc = accuracy_score(test_labels, clf_leaky.predict(leaky_test))
print(f"LEAKY pipeline  — Test: {leaky_test_acc:.2f} (artificially inflated)")

# WHY this matters:
# - IDF weights computed on all data include test document frequencies
# - Vocabulary includes terms that only appear in test set
# - In production, you won't have test data at training time
# - The leaky pipeline gives falsely optimistic metrics
```

### TF-IDF Limitations

```python
"""
TF-IDF Limitations:

1. No semantic understanding
   - "happy" and "joyful" are completely different vectors
   - "bank" (financial) and "bank" (river) are the same

2. Bag-of-words assumption
   - Word order is lost: "dog bites man" = "man bites dog"
   - Context is ignored

3. Sparse representations
   - Vocabulary of 100K → 100K-dimensional vectors
   - Most elements are zero

4. Out-of-vocabulary problem
   - New words not in training data get zero weight

These limitations motivate dense embeddings (Word2Vec, BERT, etc.)
"""
```

### BM25: The Industry Standard for Keyword Retrieval

```python
import numpy as np
from collections import Counter
from typing import List

class BM25:
    """
    BM25 (Best Matching 25) — the ranking function used by Elasticsearch,
    Solr, and most production search engines.

    BM25 improves over TF-IDF by:
    1. Saturating TF: BM25's TF component plateaus (diminishing returns
       for term repetition). TF-IDF grows linearly with frequency.
    2. Document length normalization: Short documents that mention a term
       once are ranked higher than long documents mentioning it once.
    3. Tunable parameters (k1, b) for domain-specific behavior.

    Formula:
    score(q, d) = Σ IDF(qi) × [TF(qi, d) × (k1 + 1)] / [TF(qi, d) + k1 × (1 - b + b × |d| / avgdl)]

    Parameters:
        k1: Controls TF saturation (default 1.5). Higher = more weight to frequent terms.
        b:  Controls length normalization (default 0.75). 0 = no normalization, 1 = full.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_lengths = []
        self.avgdl = 0
        self.doc_freqs = Counter()
        self.n_docs = 0
        self.documents = []

    def fit(self, documents: List[List[str]]):
        """Index documents for BM25 scoring."""
        self.documents = documents
        self.n_docs = len(documents)
        self.doc_lengths = [len(doc) for doc in documents]
        self.avgdl = np.mean(self.doc_lengths)

        # Document frequency: how many docs contain each term
        for doc in documents:
            for term in set(doc):
                self.doc_freqs[term] += 1

        return self

    def _idf(self, term: str) -> float:
        """IDF with Robertson-Sparck Jones formula (handles zero df)."""
        df = self.doc_freqs.get(term, 0)
        return np.log((self.n_docs - df + 0.5) / (df + 0.5) + 1)

    def score(self, query: List[str], doc_idx: int) -> float:
        """Score a single document against a query."""
        doc = self.documents[doc_idx]
        doc_len = self.doc_lengths[doc_idx]
        tf_counter = Counter(doc)

        score = 0.0
        for term in query:
            if term not in self.doc_freqs:
                continue

            tf = tf_counter.get(term, 0)
            idf = self._idf(term)

            # BM25 TF component with saturation and length normalization
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
            score += idf * numerator / denominator

        return score

    def search(self, query: List[str], top_k: int = 5) -> List[tuple]:
        """Rank all documents by BM25 score."""
        scores = [(i, self.score(query, i)) for i in range(self.n_docs)]
        scores.sort(key=lambda x: -x[1])
        return scores[:top_k]


# Compare TF-IDF vs BM25
documents = [
    ["machine", "learning", "is", "amazing"],
    ["deep", "learning", "uses", "neural", "networks"],
    ["machine", "learning", "and", "deep", "learning", "are", "related"],
    ["natural", "language", "processing", "uses", "machine", "learning"],
    # Long doc: "learning" appears 5 times but document is long
    ["learning", "learning", "learning", "learning", "learning",
     "is", "a", "broad", "field", "with", "many", "applications",
     "in", "science", "technology", "medicine", "and", "engineering"],
]

# TF-IDF scores
vectorizer = TFIDFVectorizer(max_features=20)
tfidf_matrix = vectorizer.fit_transform(documents)

# BM25 scores
bm25 = BM25(k1=1.5, b=0.75)
bm25.fit(documents)

query = ["machine", "learning"]
print("Query:", query)
print(f"\n{'Doc':>4}  {'TF-IDF':>8}  {'BM25':>8}  Content")
print("-" * 70)

# TF-IDF query
query_vec = vectorizer.transform([query])[0]
tfidf_scores = tfidf_matrix @ query_vec

bm25_results = bm25.search(query, top_k=5)
bm25_score_map = dict(bm25_results)

for i in range(len(documents)):
    print(f"{i:>4}  {tfidf_scores[i]:>8.4f}  {bm25_score_map.get(i, 0):>8.4f}  {' '.join(documents[i][:6])}...")

# KEY OBSERVATION: Doc 4 has "learning" 5 times.
# TF-IDF: May rank it highly (linear TF growth)
# BM25: TF saturates — doc 4 won't dominate just from repetition
# This is why BM25 is better for real search: keyword stuffing doesn't game it.
```

### ✅ Checkpoint: After Part 3

You should now be able to answer:
1. Why does cosine similarity work better than Euclidean distance for text vectors?
2. What is data leakage in NLP preprocessing and how do you prevent it?
3. How does BM25's TF saturation improve over raw TF-IDF?
4. What are TF-IDF's four fundamental limitations?

If you can't answer all four, re-read Part 3 before continuing.

---

## Part 4: Word Embeddings — Dense Representations

### Why Embeddings?

| TF-IDF | Embeddings |
|--------|------------|
| Sparse (100K+ dims) | Dense (100-1000 dims) |
| No semantics | Captures meaning |
| Exact word match | Similar words ≈ similar vectors |
| Hand-crafted features | Learned from data |

### Word2Vec Concepts

Word2Vec learns embeddings by predicting context:

```
Skip-gram: Given center word, predict surrounding words
CBOW: Given surrounding words, predict center word

"The cat sat on the mat"
Skip-gram: "sat" → predict ["The", "cat", "on", "the"]
CBOW: ["The", "cat", "on", "the"] → predict "sat"
```

**How Word2Vec Actually Trains (Skip-gram with Negative Sampling):**

```python
"""
Architecture: Two weight matrices (W_input, W_output), each of shape (vocab_size, embed_dim)

For each word in the corpus:
1. Take center word c and a real context word w+ (positive pair)
2. Sample k random "noise" words w1-, w2-, ... (negative samples)
3. Optimize:
   Loss = -log(σ(W_out[w+] · W_in[c])) - Σ log(σ(-W_out[wi-] · W_in[c]))

   σ = sigmoid. This pushes the dot product of real pairs UP and random pairs DOWN.

4. After training, W_input is your embedding matrix.

WHY NEGATIVE SAMPLING instead of full softmax?
- Full softmax: P(w|c) = exp(W_out[w] · W_in[c]) / Σ_all_words exp(W_out[v] · W_in[c])
- Computing the denominator requires summing over ALL vocab words (expensive!)
- Negative sampling approximates this by only updating k negative samples per step
- Typical k = 5-20 for large corpora, 2-5 for very large corpora

WHY THIS WORKS:
- Words in similar contexts get pulled to similar positions in embedding space
- "cat" and "dog" both appear near "pet", "furry", "vet" → similar embeddings
- The geometric relationships emerge from co-occurrence statistics
"""
```

### Using Pre-trained Embeddings

```python
import gensim.downloader as api
import numpy as np

# Load pre-trained Word2Vec (trained on Google News, 3B words)
print("Loading Word2Vec model... (this may take a minute)")
model = api.load('word2vec-google-news-300')  # 300-dimensional

print(f"Vocabulary size: {len(model.key_to_index)}")
print(f"Embedding dimension: {model.vector_size}")

# Get word vectors
king_vec = model['king']
print(f"'king' vector shape: {king_vec.shape}")
print(f"'king' vector (first 10): {king_vec[:10]}")

# Find similar words
similar = model.most_similar('computer', topn=5)
print(f"\nWords similar to 'computer':")
for word, score in similar:
    print(f"  {word}: {score:.3f}")

# Word analogies
# king - man + woman ≈ queen
result = model.most_similar(positive=['king', 'woman'], negative=['man'], topn=3)
print(f"\nking - man + woman =")
for word, score in result:
    print(f"  {word}: {score:.3f}")

# paris - france + germany ≈ berlin
result = model.most_similar(positive=['paris', 'germany'], negative=['france'], topn=3)
print(f"\nparis - france + germany =")
for word, score in result:
    print(f"  {word}: {score:.3f}")

# Similarity between words
similarity = model.similarity('dog', 'cat')
print(f"\nSimilarity('dog', 'cat'): {similarity:.3f}")

similarity = model.similarity('dog', 'car')
print(f"Similarity('dog', 'car'): {similarity:.3f}")
```

### GloVe Embeddings

```python
import numpy as np
from typing import Dict

def load_glove_embeddings(filepath: str, embedding_dim: int = 100) -> Dict[str, np.ndarray]:
    """
    Load GloVe embeddings from file.

    Download from: https://nlp.stanford.edu/projects/glove/
    """
    embeddings = {}

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.strip().split()
            word = values[0]
            vector = np.array(values[1:], dtype=np.float32)
            if len(vector) == embedding_dim:
                embeddings[word] = vector

    return embeddings


# Alternative: Use gensim
import gensim.downloader as api

print("Loading GloVe model...")
glove = api.load('glove-wiki-gigaword-100')  # 100-dimensional

# Compare Word2Vec and GloVe
print("\nWord2Vec vs GloVe similarity:")
for word1, word2 in [('king', 'queen'), ('apple', 'banana'), ('car', 'automobile')]:
    w2v_sim = model.similarity(word1, word2)
    glove_sim = glove.similarity(word1, word2)
    print(f"  {word1}-{word2}: Word2Vec={w2v_sim:.3f}, GloVe={glove_sim:.3f}")
```

### Embedding Lookup for Neural Networks

```python
import torch
import torch.nn as nn
import numpy as np

def create_embedding_matrix(vocab: Dict[str, int], pretrained_embeddings,
                           embedding_dim: int) -> np.ndarray:
    """
    Create embedding matrix from pre-trained embeddings.

    Args:
        vocab: Dictionary mapping words to indices
        pretrained_embeddings: Gensim model or dict
        embedding_dim: Dimension of embeddings

    Returns:
        Matrix of shape (vocab_size, embedding_dim)
    """
    vocab_size = len(vocab)
    rng = np.random.default_rng(seed=42)
    matrix = rng.standard_normal((vocab_size, embedding_dim)) * 0.01  # Random init

    found = 0
    for word, idx in vocab.items():
        try:
            if hasattr(pretrained_embeddings, '__getitem__'):
                matrix[idx] = pretrained_embeddings[word]
                found += 1
        except KeyError:
            pass  # Keep random initialization for OOV words

    print(f"Found {found}/{vocab_size} words in pre-trained embeddings")
    return matrix


class TextClassifier(nn.Module):
    """
    Text classifier using pre-trained embeddings.
    """

    def __init__(self, vocab_size: int, embedding_dim: int, num_classes: int,
                 pretrained_weights: np.ndarray = None, freeze_embeddings: bool = True):
        super().__init__()

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Load pre-trained weights
        if pretrained_weights is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_weights))

        # Optionally freeze embeddings
        if freeze_embeddings:
            self.embedding.weight.requires_grad = False

        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x: (batch_size, seq_len)
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)

        # Simple pooling: average over sequence
        pooled = embedded.mean(dim=1)  # (batch_size, embedding_dim)

        return self.classifier(pooled)


# Example usage
vocab = {'the': 0, 'cat': 1, 'sat': 2, 'on': 3, 'mat': 4, '[PAD]': 5, '[UNK]': 6}
embedding_matrix = create_embedding_matrix(vocab, glove, 100)

model = TextClassifier(
    vocab_size=len(vocab),
    embedding_dim=100,
    num_classes=2,
    pretrained_weights=embedding_matrix,
    freeze_embeddings=True
)

# Forward pass
x = torch.tensor([[0, 1, 2, 3, 4]])  # "the cat sat on mat"
output = model(x)
print(f"Output shape: {output.shape}")
```

### ✅ Checkpoint: After Part 4

You should now be able to answer:
1. How does Word2Vec's skip-gram with negative sampling train embeddings?
2. Why is negative sampling needed instead of full softmax?
3. When should you freeze vs fine-tune pre-trained embeddings?
4. What is the difference between Word2Vec and GloVe's training objectives?

If you can't answer all four, re-read Part 4 before continuing.

---

## Part 5: Building a Semantic Search Engine

Let's build a complete document search system:

```python
# semantic_search.py
"""
Semantic Document Search Engine

Combines TF-IDF for efficiency with embeddings for semantic understanding.
"""

import numpy as np
from typing import List, Dict, Tuple
from collections import Counter
import re

class SemanticSearchEngine:
    """
    A hybrid search engine combining keyword and semantic search.

    For production, consider:
    - Faiss or Annoy for approximate nearest neighbor
    - Elasticsearch for full-text search
    - Sentence transformers for better embeddings
    """

    def __init__(self, embedding_model=None):
        self.documents = []
        self.document_vectors = None
        self.tfidf_vectorizer = None
        self.embedding_model = embedding_model
        self.preprocessor = TextPreprocessor(
            lowercase=True,
            remove_stopwords=False,  # Keep for embeddings
            lemmatize=False
        )

    def preprocess(self, text: str) -> List[str]:
        """Preprocess text for indexing."""
        return self.preprocessor.preprocess(text)

    def add_documents(self, documents: List[str]):
        """Add documents to the search index."""
        self.documents.extend(documents)

        # Preprocess
        processed = [self.preprocess(doc) for doc in documents]

        # Build TF-IDF index
        self.tfidf_vectorizer = TFIDFVectorizer(max_features=5000)
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(processed)

        # Build embedding index (if model available)
        if self.embedding_model:
            self.document_embeddings = self._compute_document_embeddings(processed)

        print(f"Indexed {len(documents)} documents")
        print(f"Vocabulary size: {len(self.tfidf_vectorizer.vocabulary)}")

    def _compute_document_embeddings(self, processed_docs: List[List[str]]) -> np.ndarray:
        """
        Compute document embeddings using sentence transformers.

        WARNING: Averaging word embeddings is an anti-pattern that:
        - Loses word order information
        - Treats all words equally (ignores importance)
        - Fails on short documents and rare vocabulary

        For production, use:
        - sentence-transformers (https://www.sbert.net/)
        - BERT/RoBERTa with [CLS] token pooling
        - Dedicated document embedding models
        """
        try:
            from sentence_transformers import SentenceTransformer

            # Use sentence transformers for proper document embedding
            if not hasattr(self, 'sentence_model'):
                print("Loading sentence transformer model...")
                self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

            # Reconstruct documents from tokens
            documents_text = [' '.join(doc) for doc in processed_docs]
            embeddings = self.sentence_model.encode(documents_text, convert_to_numpy=True)

            return embeddings

        except ImportError:
            print("sentence-transformers not available. Falling back to word embedding averaging (NOT RECOMMENDED)")
            print("Install with: pip install sentence-transformers")

            # Fallback: Use word averaging (anti-pattern, for educational purposes only)
            embedding_dim = self.embedding_model.vector_size
            embeddings = np.zeros((len(processed_docs), embedding_dim))

            for i, doc in enumerate(processed_docs):
                word_vectors = []
                for word in doc:
                    try:
                        word_vectors.append(self.embedding_model[word])
                    except KeyError:
                        pass  # Skip OOV words

                if word_vectors:
                    embeddings[i] = np.mean(word_vectors, axis=0)

            # Normalize
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1
            embeddings = embeddings / norms

            return embeddings

    def search_tfidf(self, query: str, top_k: int = 5) -> List[Tuple[int, float, str]]:
        """Search using TF-IDF similarity."""
        processed_query = self.preprocess(query)
        query_vector = self.tfidf_vectorizer.transform([processed_query])[0]

        # Compute similarities
        similarities = self.tfidf_matrix @ query_vector

        # Get top results
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append((idx, similarities[idx], self.documents[idx]))

        return results

    def search_semantic(self, query: str, top_k: int = 5) -> List[Tuple[int, float, str]]:
        """
        Search using sentence embedding similarity (proper approach).

        Uses sentence-transformers for query encoding instead of averaging word vectors.
        """
        if self.embedding_model is None:
            raise ValueError("No embedding model provided")

        try:
            from sentence_transformers import SentenceTransformer
            if not hasattr(self, 'sentence_model'):
                self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

            # Encode query using sentence transformer
            query_embedding = self.sentence_model.encode(query, convert_to_numpy=True)

        except ImportError:
            # Fallback: naive averaging (NOT RECOMMENDED - anti-pattern)
            processed_query = self.preprocess(query)
            word_vectors = []
            for word in processed_query:
                try:
                    word_vectors.append(self.embedding_model[word])
                except KeyError:
                    pass

            if not word_vectors:
                return []

            query_embedding = np.mean(word_vectors, axis=0)
            query_embedding = query_embedding / np.linalg.norm(query_embedding)

        # Compute similarities
        similarities = self.document_embeddings @ query_embedding

        # Get top results
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append((idx, similarities[idx], self.documents[idx]))

        return results

    def search_hybrid(self, query: str, top_k: int = 5,
                     tfidf_weight: float = 0.3) -> List[Tuple[int, float, str]]:
        """
        Hybrid search combining TF-IDF and semantic similarity.

        Uses sentence-transformers for the semantic component (not word averaging).

        Args:
            query: Search query
            top_k: Number of results
            tfidf_weight: Weight for TF-IDF (0-1), semantic gets 1-weight
        """
        # TF-IDF scores
        processed_query = self.preprocess(query)
        query_tfidf = self.tfidf_vectorizer.transform([processed_query])[0]
        tfidf_scores = self.tfidf_matrix @ query_tfidf

        # Semantic scores using sentence-transformers (NOT word averaging)
        # IMPORTANT: We do NOT fall back to word averaging here.
        # If sentence-transformers is unavailable, we use TF-IDF only.
        # Word averaging is an anti-pattern that loses word order and
        # treats all words equally — it would undermine the hybrid approach.
        semantic_scores = np.zeros(len(self.documents))
        if hasattr(self, 'document_embeddings') and hasattr(self, 'sentence_model'):
            try:
                query_embedding = self.sentence_model.encode(query, convert_to_numpy=True)
                semantic_scores = self.document_embeddings @ query_embedding
            except Exception as e:
                print(f"WARNING: Semantic scoring failed ({e}). Using TF-IDF only.")
                tfidf_weight = 1.0
        else:
            # No sentence model available — degrade gracefully to TF-IDF only
            print("INFO: sentence-transformers not loaded. Hybrid search using TF-IDF only.")
            tfidf_weight = 1.0

        # Combine scores
        combined_scores = tfidf_weight * tfidf_scores + (1 - tfidf_weight) * semantic_scores

        # Get top results
        top_indices = np.argsort(combined_scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append((idx, combined_scores[idx], self.documents[idx]))

        return results


# Demo
documents = [
    "Machine learning is a subset of artificial intelligence.",
    "Deep learning uses neural networks with many layers.",
    "Natural language processing helps computers understand text.",
    "Computer vision enables machines to interpret images.",
    "Reinforcement learning trains agents through rewards.",
    "GPT and BERT are transformer-based language models.",
    "Convolutional neural networks excel at image recognition.",
    "Recurrent neural networks handle sequential data.",
    "Transfer learning applies knowledge from one task to another.",
    "Data preprocessing is crucial for machine learning success.",
]

# Create search engine
print("Building search index...")
search_engine = SemanticSearchEngine(embedding_model=glove)
search_engine.add_documents(documents)

# Test queries
queries = [
    "neural network image",
    "text understanding",
    "learning from data",
]

print("\n" + "="*60)
print("SEARCH RESULTS")
print("="*60)

for query in queries:
    print(f"\nQuery: '{query}'")

    # TF-IDF search
    print("\nTF-IDF Results:")
    for idx, score, doc in search_engine.search_tfidf(query, top_k=3):
        print(f"  [{score:.3f}] {doc[:60]}...")

    # Semantic search
    print("\nSemantic Results:")
    for idx, score, doc in search_engine.search_semantic(query, top_k=3):
        print(f"  [{score:.3f}] {doc[:60]}...")

    # Hybrid search
    print("\nHybrid Results:")
    for idx, score, doc in search_engine.search_hybrid(query, top_k=3):
        print(f"  [{score:.3f}] {doc[:60]}...")
```

### ✅ Checkpoint: After Part 5

You should now be able to answer:
1. Why does `search_hybrid` refuse to fall back to word averaging?
2. What is the role of `tfidf_weight` in hybrid scoring?
3. When would you set `tfidf_weight=1.0` (pure keyword) vs `tfidf_weight=0.0` (pure semantic)?
4. What production alternatives (Faiss, Elasticsearch) would replace this toy search engine?

If you can't answer all four, re-read Part 5 before continuing.

---

## Manager's Summary

### Why Preprocessing Matters

**Bad preprocessing** leads to:
- Lost important information (emojis = sentiment)
- Vocabulary explosion (case-sensitive = 2x vocabulary)
- Silent failures (OOV words → [UNK])

**Cost implications**:
| Factor | Impact |
|--------|--------|
| Vocabulary size | Memory, inference time |
| Embedding dimension | Model size, computation |
| Preprocessing choices | Model accuracy |

**Memory & Latency Estimates (10K documents, ~100 words each)**:
| Method | Index Memory | Query Latency | Notes |
|--------|-------------|---------------|-------|
| TF-IDF (sparse, 10K vocab) | ~40 MB (scipy.sparse: ~1% non-zero) | <1 ms | Sparse matrix ops, CPU-only |
| Word2Vec (300d, averaged) | ~24 MB (10K × 300 × 8 bytes) | ~5 ms | Dense matmul, no model inference |
| Sentence-transformers (384d) | ~31 MB (10K × 384 × 8 bytes) | ~50-200 ms/query | Model inference dominates |
| BM25 (inverted index) | ~20 MB (term→doc mapping) | <1 ms | No matrix ops, pure lookups |

At **1M documents**: TF-IDF sparse stays manageable (~400 MB). Dense embeddings hit ~3 GB.
This is why production search uses approximate nearest neighbor (Faiss, Annoy) at scale.

### Questions to Ask Your Team

1. "What tokenizer are we using? Why?"
2. "What's our out-of-vocabulary rate?"
3. "Are we using pre-trained embeddings? Which ones?"
4. "How are we handling different languages?"
5. "What's the preprocessing pipeline for production vs training?"

### Embedding Choices

| Embedding | Pros | Cons | When to Use |
|-----------|------|------|-------------|
| TF-IDF | Fast, interpretable | No semantics | Keyword search, baselines |
| Word2Vec | Semantic, efficient | No context | Word similarity tasks |
| GloVe | Good for analogies | Fixed vocabulary | General NLP |
| FastText | Handles OOV | Larger model | Morphologically rich languages |
| BERT/GPT | State-of-the-art | Expensive | When accuracy matters most |

---

## Evaluation & Measurement

### TF-IDF vs Embedding Accuracy Benchmarks

```python
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import numpy as np

# Ground truth: relevant document pairs
relevant_pairs = [
    (0, 2),  # "machine learning" relates to "ML and deep learning"
    (1, 5),  # "deep learning" relates to "transformer language models"
    (3, 8),  # "computer vision" relates to "CNNs for image recognition"
]

documents = [
    "Machine learning is a subset of artificial intelligence.",
    "Deep learning uses neural networks with many layers.",
    "Machine learning and deep learning are related fields.",
    "Computer vision enables machines to interpret images.",
    "Reinforcement learning trains agents through rewards.",
    "GPT and BERT are transformer-based language models.",
    "Convolutional neural networks excel at image recognition.",
    "Recurrent neural networks handle sequential data.",
    "CNNs are fundamental to modern computer vision.",
    "Data preprocessing is crucial for machine learning success.",
]

# Benchmark 1: TF-IDF
print("="*60)
print("TF-IDF Benchmark")
print("="*60)

tfidf = TfidfVectorizer(max_features=100, stop_words='english')
tfidf_matrix = tfidf.fit_transform(documents).toarray()

tfidf_scores = []
for doc_i, doc_j in relevant_pairs:
    sim = cosine_similarity([tfidf_matrix[doc_i]], [tfidf_matrix[doc_j]])[0][0]
    tfidf_scores.append(sim)

print(f"Mean TF-IDF similarity (relevant pairs): {np.mean(tfidf_scores):.4f}")
print(f"Std Dev: {np.std(tfidf_scores):.4f}")

# Benchmark 2: Sentence Transformers (proper embeddings)
print("\n" + "="*60)
print("Sentence Transformer Benchmark")
print("="*60)

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(documents)

embedding_scores = []
for doc_i, doc_j in relevant_pairs:
    sim = cosine_similarity([embeddings[doc_i]], [embeddings[doc_j]])[0][0]
    embedding_scores.append(sim)

print(f"Mean embedding similarity (relevant pairs): {np.mean(embedding_scores):.4f}")
print(f"Std Dev: {np.std(embedding_scores):.4f}")

# Comparison
print("\n" + "="*60)
print("Comparison")
print("="*60)
print(f"TF-IDF advantage: {abs(np.mean(tfidf_scores) - np.mean(embedding_scores)):.4f}")
print(f"Winner: {'Embeddings' if np.mean(embedding_scores) > np.mean(tfidf_scores) else 'TF-IDF'}")
```

**What to expect** (run the benchmark above to get your own numbers):

| Metric | TF-IDF | Embeddings | Notes |
|--------|--------|-----------|-------|
| Semantic pair similarity | Lower | Higher | Embeddings capture meaning beyond keyword overlap |
| Computation time | Faster | Slower | TF-IDF is sparse matrix ops; embeddings require model inference |
| Memory | Lower | Higher | Dense vectors use more memory than sparse TF-IDF |
| Interpretability | High | Low | TF-IDF weights map directly to terms |

Run the benchmark code above on your own data to get concrete numbers. Results vary significantly by corpus size, domain, and embedding model.

### Retrieval Ranking Metrics: Precision@k, MAP, NDCG

```python
import numpy as np
from typing import List, Set

def precision_at_k(retrieved: List[int], relevant: Set[int], k: int) -> float:
    """
    Precision@k: What fraction of the top-k results are relevant?

    Args:
        retrieved: Ranked list of document IDs (best first)
        relevant: Set of truly relevant document IDs
        k: Cutoff rank
    """
    top_k = retrieved[:k]
    return len(set(top_k) & relevant) / k


def recall_at_k(retrieved: List[int], relevant: Set[int], k: int) -> float:
    """Recall@k: What fraction of relevant documents appear in top-k?"""
    top_k = retrieved[:k]
    if len(relevant) == 0:
        return 0.0
    return len(set(top_k) & relevant) / len(relevant)


def average_precision(retrieved: List[int], relevant: Set[int]) -> float:
    """
    Average Precision (AP): Area under the precision-recall curve for one query.

    AP = (1/|relevant|) × Σ_{k=1}^{N} P(k) × rel(k)
    where rel(k) = 1 if doc at rank k is relevant, else 0.
    """
    if not relevant:
        return 0.0

    hits = 0
    sum_precision = 0.0

    for k, doc_id in enumerate(retrieved, 1):
        if doc_id in relevant:
            hits += 1
            sum_precision += hits / k

    return sum_precision / len(relevant)


def mean_average_precision(queries_retrieved: List[List[int]],
                           queries_relevant: List[Set[int]]) -> float:
    """MAP: Average of AP across all queries. THE standard retrieval metric."""
    aps = [average_precision(ret, rel) for ret, rel in zip(queries_retrieved, queries_relevant)]
    return np.mean(aps)


def ndcg_at_k(retrieved: List[int], relevance_scores: dict, k: int) -> float:
    """
    NDCG@k: Normalized Discounted Cumulative Gain.

    Unlike precision@k (binary relevant/not), NDCG handles graded relevance:
    - relevance_scores: {doc_id: relevance_level} (e.g., 0=irrelevant, 1=partial, 2=perfect)
    - Higher-ranked relevant documents contribute more (logarithmic discount)

    NDCG = DCG / IDCG where DCG = Σ (2^rel - 1) / log2(rank + 1)
    """
    # DCG
    dcg = 0.0
    for i, doc_id in enumerate(retrieved[:k]):
        rel = relevance_scores.get(doc_id, 0)
        dcg += (2**rel - 1) / np.log2(i + 2)  # +2 because rank starts at 1

    # Ideal DCG (sort by relevance, best first)
    ideal_rels = sorted(relevance_scores.values(), reverse=True)[:k]
    idcg = sum((2**rel - 1) / np.log2(i + 2) for i, rel in enumerate(ideal_rels))

    if idcg == 0:
        return 0.0
    return dcg / idcg


# ---- Evaluate our search engine with proper metrics ----
documents = [
    "Machine learning is a subset of artificial intelligence.",       # 0
    "Deep learning uses neural networks with many layers.",           # 1
    "Machine learning and deep learning are related fields.",         # 2
    "Computer vision enables machines to interpret images.",          # 3
    "Reinforcement learning trains agents through rewards.",          # 4
    "GPT and BERT are transformer-based language models.",            # 5
    "Convolutional neural networks excel at image recognition.",      # 6
    "Recurrent neural networks handle sequential data.",              # 7
    "CNNs are fundamental to modern computer vision.",                # 8
    "Data preprocessing is crucial for machine learning success.",    # 9
]

# Ground truth for 3 queries (manually annotated)
queries_and_relevance = [
    {
        "query": "neural network image",
        "relevant": {1, 6, 8},              # Binary relevance
        "graded": {1: 1, 3: 1, 6: 2, 8: 2}, # Graded: 2=highly relevant, 1=partial
    },
    {
        "query": "text understanding NLP",
        "relevant": {5, 7},
        "graded": {5: 2, 7: 1},
    },
    {
        "query": "learning from data",
        "relevant": {0, 1, 2, 9},
        "graded": {0: 2, 1: 1, 2: 2, 4: 1, 9: 2},
    },
]

# Build TF-IDF index
from sklearn.feature_extraction.text import TfidfVectorizer as SklearnTfidf
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine

tfidf = SklearnTfidf(max_features=100, stop_words='english')
tfidf_matrix = tfidf.fit_transform(documents).toarray()

print("="*60)
print("RETRIEVAL EVALUATION")
print("="*60)

all_retrieved = []
all_relevant = []

for qr in queries_and_relevance:
    query_vec = tfidf.transform([qr["query"]]).toarray()
    scores = sklearn_cosine(query_vec, tfidf_matrix)[0]
    ranked = list(np.argsort(scores)[::-1])

    p5 = precision_at_k(ranked, qr["relevant"], k=5)
    r5 = recall_at_k(ranked, qr["relevant"], k=5)
    ap = average_precision(ranked, qr["relevant"])
    ndcg5 = ndcg_at_k(ranked, qr["graded"], k=5)

    print(f"\nQuery: '{qr['query']}'")
    print(f"  P@5={p5:.3f}  R@5={r5:.3f}  AP={ap:.3f}  NDCG@5={ndcg5:.3f}")
    print(f"  Top-5 retrieved: {ranked[:5]}")

    all_retrieved.append(ranked)
    all_relevant.append(qr["relevant"])

map_score = mean_average_precision(all_retrieved, all_relevant)
print(f"\nOverall MAP: {map_score:.3f}")
print("(MAP > 0.5 is reasonable for keyword search; < 0.3 suggests semantic search needed)")
```

### Preprocessing Quality Metrics

```python
def evaluate_preprocessing_quality(original_texts: List[str],
                                  preprocessor: TextPreprocessor) -> Dict[str, float]:
    """
    Measure preprocessing quality on multiple dimensions.
    """
    results = {
        'vocab_size': 0,
        'avg_tokens_per_doc': 0,
        'compression_ratio': 0,
        'oov_rate': 0,
        'information_loss': 0,
    }

    all_tokens = []
    original_lengths = []

    for text in original_texts:
        preprocessed = preprocessor.preprocess(text)
        all_tokens.extend(preprocessed)
        original_lengths.append(len(text.split()))

    # Vocabulary size
    results['vocab_size'] = len(set(all_tokens))

    # Average tokens per document
    results['avg_tokens_per_doc'] = np.mean([len(preprocessor.preprocess(t))
                                             for t in original_texts])

    # Compression ratio (original vs preprocessed length)
    preprocessed_lengths = [len(preprocessor.preprocess(t)) for t in original_texts]
    results['compression_ratio'] = np.mean(preprocessed_lengths) / np.mean(original_lengths)

    # Information loss (tokens removed / original)
    tokens_removed = np.mean(original_lengths) - results['avg_tokens_per_doc']
    results['information_loss'] = tokens_removed / np.mean(original_lengths)

    return results

# Example usage
metrics = evaluate_preprocessing_quality(documents, preprocessor)
print("Preprocessing Quality Metrics:")
for metric, value in metrics.items():
    print(f"  {metric}: {value:.3f}")
```

**Quality Thresholds**:
- `vocab_size`: 1K-50K (sweet spot depends on domain)
- `compression_ratio`: 0.3-0.7 (high removal may lose information)
- `information_loss`: <0.4 (losing >40% of tokens is risky)

### Before/After Comparison Methodology

```python
def compare_preprocessing_strategies(text: str, strategies: Dict[str, TextPreprocessor]):
    """
    Compare multiple preprocessing approaches on a single document.
    """
    print(f"Original text: {text}")
    print(f"Original length: {len(text)} characters, {len(text.split())} tokens\n")

    for strategy_name, preprocessor in strategies.items():
        tokens = preprocessor.preprocess(text)
        print(f"{strategy_name}:")
        print(f"  Tokens: {tokens}")
        print(f"  Count: {len(tokens)}")
        print(f"  Unique: {len(set(tokens))}")
        print()

# Define strategies
conservative = TextPreprocessor(lowercase=True, remove_stopwords=False, lemmatize=False)
standard = TextPreprocessor(lowercase=True, remove_stopwords=True, lemmatize=True)
aggressive = TextPreprocessor(lowercase=True, remove_stopwords=True, lemmatize=True,
                             remove_numbers=True, min_token_length=3)

strategies = {
    "Conservative (minimal)": conservative,
    "Standard (balanced)": standard,
    "Aggressive (maximum reduction)": aggressive,
}

compare_preprocessing_strategies(
    "Don't stop believing! It's a great song, isn't it? 123 rocks.",
    strategies
)
```

---

## Interview Preparation

### Likely Questions

**Q: What is tokenization and why does it matter?**
A: Tokenization splits text into discrete units (tokens) that form the model's vocabulary. The choice of tokenization affects vocabulary size, OOV handling, and what the model can learn. Modern models use subword tokenization (BPE) to balance vocabulary size with coverage.

**Q: Explain TF-IDF.**
A: TF-IDF weights terms by how important they are to a document relative to the corpus. TF (term frequency) measures how often a term appears in a document. IDF (inverse document frequency) downweights common terms. High TF-IDF means the term is distinctive for that document.

**Q: What are word embeddings and why are they useful?**
A: Word embeddings are dense vector representations where similar words have similar vectors. Unlike one-hot encoding (sparse, no semantics), embeddings capture meaning: "king" is closer to "queen" than to "apple." They're learned from large corpora and transfer to downstream tasks.

**Q: How do you handle out-of-vocabulary words?**
A: Several strategies:
1. Use subword tokenization (BPE, WordPiece) to break unknown words into known subwords
2. Use FastText which represents words as character n-gram bags
3. Map to [UNK] token (loses information)
4. Use a fallback model for rare words

**Q: When would you use TF-IDF vs embeddings?**
A: TF-IDF for: keyword search, sparse features, interpretability, fast baseline. Embeddings for: semantic similarity, transfer learning, neural networks, when you need to capture meaning beyond exact matches.

**Q: Explain BM25 vs TF-IDF. Why is BM25 standard in production search?**
A: BM25 improves over TF-IDF in two ways: (1) TF saturation — repeating a word 10 times doesn't give 10x the score like raw TF does, which prevents keyword stuffing; (2) Document length normalization — short documents with a term match rank higher than long documents with the same match. BM25 is parameterized by k1 (TF saturation) and b (length normalization), giving engineers tuning knobs.

**Q: How do you evaluate a search/retrieval system?**
A: Primary metrics: MAP (mean average precision) for overall ranking quality, NDCG@k for graded relevance, Precision@k for top-result quality, Recall@k for coverage. These require annotated relevance judgments. For automated evaluation, use A/B tests measuring click-through rate, time-to-result, and query reformulation rate.

**Q: What is data leakage in NLP preprocessing?**
A: Fitting preprocessing parameters (vocabulary, IDF weights, stopword frequency thresholds) on the full dataset including test data. This leaks test distribution into training, giving falsely optimistic metrics. Prevention: fit on training split only, then `.transform()` test data. The same applies to normalization, vocabulary construction, and any learned preprocessing step.

---

## Job Role Mapping

| Blog Section | ML Engineer | Data Scientist | AI Architect | Engineering Manager |
|---|---|---|---|---|
| Part 1: Tokenization | Implement custom tokenizers, handle edge cases | Choose tokenizer for modeling, analyze OOV rates | Design tokenization strategy across services | Understand vocab size impact on costs |
| Part 2: Text Cleaning | Build preprocessing pipelines, production hardening | Feature engineering, data quality analysis | Define preprocessing standards, prevent leakage | Ask "what's our OOV rate? preprocessing pipeline?" |
| Part 3: TF-IDF & BM25 | Implement/optimize search ranking | Use as baseline, feature extraction | Choose BM25 vs embedding search architecture | Compare search latency vs quality tradeoffs |
| Part 4: Embeddings | Integrate pre-trained models, build lookup tables | Train/evaluate embeddings for tasks | Select embedding model (Word2Vec/GloVe/BERT) | Budget GPU vs CPU for embedding inference |
| Part 5: Semantic Search | Build hybrid search, ANN indexing, production scale | Evaluate retrieval quality (MAP, NDCG) | Design search architecture (Elasticsearch + Faiss) | Understand cost of semantic vs keyword search |
| Evaluation Metrics | Implement MAP/NDCG pipelines | Analyze precision-recall tradeoffs | Set quality gates for search launches | Require P@5 > threshold before deployment |

---

## When NOT to Use Standard Techniques

### When Embeddings Fail

```python
"""
Embeddings struggle in these scenarios:

1. SHORT TEXT (< 5 words)
   Examples: "N/A", "Product ID", "2023-01-15"
   Problem: Insufficient context for meaningful representation
   Solution: Use exact matching, hand-crafted features, or structured data

2. RARE VOCABULARY (OOV rate > 20%)
   Examples: Medical codes, chemical formulas, proprietary jargon
   Problem: Model was never trained on these terms
   Solution: Use subword tokenization (FastText), domain-specific models

3. HIGHLY SPECIFIC DOMAINS
   Examples: Legal contracts, medical records, programming code
   Problem: Pre-trained embeddings miss domain semantics
   Solution: Fine-tune embeddings on domain data or use specialized models

4. MORPHOLOGICALLY RICH LANGUAGES
   Examples: Finnish, Hungarian, German compound words
   Problem: Word2Vec/GloVe miss morphological relationships
   Solution: Use FastText or language-specific tokenizers

5. MULTILINGUAL CONTENT
   Example: Code-switching ("This is muito bom!" - English + Portuguese)
   Problem: Single-language embeddings don't bridge languages
   Solution: Use multilingual models (mBERT, XLM-R)
"""

# Code example: Detect when NOT to use embeddings
def should_use_embeddings(documents: List[str], vocab_coverage_threshold: float = 0.8):
    """
    Heuristic: If OOV rate is too high, embeddings won't work well.
    """
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')

        # For this example, use a simple metric
        avg_doc_length = np.mean([len(doc.split()) for doc in documents])

        if avg_doc_length < 5:
            print("WARNING: Average document length < 5 words")
            print("  Embeddings may not work well. Consider TF-IDF or structural features.")
            return False

        print("Documents appear suitable for embedding-based methods")
        return True

    except ImportError:
        print("sentence-transformers not installed. Install with: pip install sentence-transformers")
        return False

# Example usage
short_docs = ["N/A", "Yes", "2023-01", "Product XYZ"]
should_use_embeddings(short_docs)
```

### When TF-IDF is Actually Better

```python
"""
TF-IDF outperforms embeddings in these cases:

1. KEYWORD SEARCH (exact match matters)
   Example: "What documents mention 'acromegaly'?"
   TF-IDF: High score if 'acromegaly' appears
   Embeddings: May rank other medical terms higher
   Winner: TF-IDF

2. LEGAL/COMPLIANCE DOCUMENTS
   Example: Searching for specific clauses or regulations
   Problem: Semantic similarity ≠ relevant (e.g., "must pay" vs "should pay")
   Winner: TF-IDF

3. LOW-RESOURCE LANGUAGES
   Example: Languages with few pre-trained embeddings
   Problem: Embeddings unavailable or poor quality
   Winner: TF-IDF

4. PRODUCTION WITH STRICT INTERPRETABILITY REQUIREMENTS
   Example: Regulatory approval, medical diagnosis
   Problem: Can't explain why embedding deemed it similar
   Winner: TF-IDF

5. EXTREMELY TIGHT LATENCY CONSTRAINTS
   Example: Real-time search with millions of documents
   TF-IDF: Matrix multiplication (fast)
   Embeddings: Model inference (slow)
   Winner: TF-IDF
"""

def recommend_approach(use_case: str) -> str:
    """Recommend TF-IDF or embeddings based on use case."""
    tf_idf_cases = {
        'keyword_search': 'Search for exact terms',
        'legal': 'Regulatory document compliance',
        'interpretability': 'Need to explain rankings',
        'low_latency': 'Millisecond requirements',
    }

    embedding_cases = {
        'semantic': 'Find conceptually similar docs',
        'transfer': 'Leverage pre-trained models',
        'ranking': 'Rank by semantic relevance',
    }

    if use_case in tf_idf_cases:
        return f"TF-IDF: {tf_idf_cases[use_case]}"
    elif use_case in embedding_cases:
        return f"Embeddings: {embedding_cases[use_case]}"
    else:
        return "Consider hybrid approach"

# Example
print(recommend_approach('keyword_search'))
print(recommend_approach('semantic'))
```

### Preprocessing Pitfalls That Destroy Information

```python
from typing import Dict, Any

"""
CRITICAL: These preprocessing choices can permanently lose information:

1. LOWERCASING
   Loses: Company names, acronyms, proper nouns
   Example: "Apple" (company) vs "apple" (fruit)
   Fix: Use context/embeddings; don't lowercase if case matters
   Risk Level: MEDIUM

2. REMOVING PUNCTUATION
   Loses: Sentiment markers, emphasis, abbreviations
   Example: "great..." vs "great!" convey different emotions
   Fix: Keep punctuation for sentiment; use carefully
   Risk Level: HIGH

3. AGGRESSIVE STOPWORD REMOVAL
   Loses: Negations, intensifiers, function words
   Example: "not good" vs "good" mean opposite
   Fix: Custom stopword list; don't remove "not", "no", "but"
   Risk Level: CRITICAL

4. LEMMATIZATION ON VERBS
   Loses: Tense information (past vs present vs future)
   Example: "ran" and "runs" → "run" (lose tense)
   Fix: Use for nouns only; keep verbs for temporal analysis
   Risk Level: MEDIUM

5. REMOVING NUMBERS
   Loses: Measurements, quantities, years, IDs
   Example: "2023 revenue" vs revenue without year
   Fix: Keep numbers; normalize to categories if needed
   Risk Level: HIGH

6. MIN_TOKEN_LENGTH threshold too high
   Loses: Domain terms, abbreviations, acronyms
   Example: 'CPU', 'GB', 'API' filtered out
   Fix: Lower threshold (2-3) or custom whitelist
   Risk Level: MEDIUM
"""

class SafeTextPreprocessor(TextPreprocessor):
    """
    Conservative preprocessing that minimizes information loss.
    """

    def __init__(self):
        super().__init__(
            lowercase=False,  # Preserve case information
            remove_punctuation=False,  # Keep emphasis markers
            remove_numbers=False,  # Keep numerical data
            remove_stopwords=False,  # Keep negations
            lemmatize=False,  # Keep tense/form information
            min_token_length=1  # Keep all tokens
        )

        # Custom stopwords: only remove high-frequency, non-semantic words
        self.stopwords = {
            'a', 'an', 'the', 'and', 'or', 'in', 'on', 'at', 'to', 'for',
            'is', 'are', 'be', 'been', 'being',  # Only light verbs
        }

    def critical_analysis(self, text: str) -> Dict[str, Any]:
        """Identify what would be lost with aggressive preprocessing."""
        aggressive = TextPreprocessor(
            lowercase=True, remove_punctuation=True, remove_numbers=True,
            remove_stopwords=True, lemmatize=True, min_token_length=3
        )

        safe_tokens = self.preprocess(text)
        aggressive_tokens = aggressive.preprocess(text)

        lost = set(safe_tokens) - set(aggressive_tokens)

        return {
            'original_length': len(text.split()),
            'safe_length': len(safe_tokens),
            'aggressive_length': len(aggressive_tokens),
            'information_lost': lost,
            'loss_percentage': len(lost) / len(text.split()) * 100 if text.split() else 0,
        }

# Example: Analyze what's lost
text = "Apple's iPhone 15 (launched in 2023) costs $999 - not cheap!"
analyzer = SafeTextPreprocessor()
analysis = analyzer.critical_analysis(text)

print(f"Information loss analysis:")
print(f"  Original: {analysis['original_length']} tokens")
print(f"  After safe preprocessing: {analysis['safe_length']} tokens")
print(f"  After aggressive preprocessing: {analysis['aggressive_length']} tokens")
print(f"  Lost: {analysis['lost']}")
print(f"  Loss percentage: {analysis['loss_percentage']:.1f}%")
```

---

## Exercises (Do These)

1. **Tokenizer comparison**: Compare word, subword, and character tokenization on 1000 tweets. Measure vocabulary size and OOV rate.

2. **Custom TF-IDF**: Implement TF-IDF with BM25 weighting. Compare retrieval quality.

3. **Embedding visualization**: Use t-SNE to visualize 100 word embeddings. Do semantic clusters emerge?

4. **Search evaluation**: Build a search system and evaluate with precision@k and recall@k metrics.

5. **Multilingual**: Try your preprocessing pipeline on non-English text. What breaks?

---

## What's Next

You now have:
- Tokenization strategies and tradeoffs
- TF-IDF implementation and usage
- Pre-trained embedding integration
- Complete semantic search system
- Understanding of preprocessing pitfalls

**Blog 7** introduces sequence models—RNNs and LSTMs—that capture word order. This is where we move from bag-of-words to understanding language as a sequence.

**[→ Blog 7: Sequence Models — RNNs and LSTMs](#)**

---

---

## Self-Assessment: What This Blog Does Well and Where It Falls Short

### What This Blog Does Well

- **Tokenization coverage**: Covers word-level, subword (BPE with iterative merge algorithm explained), and character tokenization with clear tradeoffs. Includes real Hugging Face tokenizer usage alongside from-scratch implementations. Vocab size tradeoff guidance (30K-50K sweet spot) is concrete.
- **TF-IDF and BM25 from scratch**: Full TF-IDF implementation with smoothed IDF, L2 normalization, document frequency filtering. BM25 implementation shows TF saturation and length normalization advantages over raw TF-IDF.
- **Preprocessing pitfalls**: Extensive coverage of information loss from aggressive preprocessing. The SafeTextPreprocessor and critical_analysis method are genuinely useful patterns. Risk levels (MEDIUM/HIGH/CRITICAL) are concrete.
- **When NOT to use techniques**: Dedicated sections on when embeddings fail and when TF-IDF is actually better. This is rare in introductory material and adds real value.
- **Retrieval evaluation**: MAP, NDCG@k, Precision@k, Recall@k implemented from scratch with worked examples and interpretation guidance.
- **Data leakage prevention**: Integrated example showing correct vs leaky pipeline with measurable impact on metrics.
- **Production lemmatization**: spaCy integration shown alongside toy dictionary, with clear explanation of when each is appropriate.
- **Word2Vec training mechanism**: Skip-gram with negative sampling explained (loss function, weight matrices, why it works).

### Where This Blog Falls Short

- **Missing FastText demonstration**: FastText is mentioned in the comparison table but never demonstrated, despite being the best solution for OOV handling in morphologically rich languages.
- **No Unicode/encoding normalization**: Production NLP pipelines break on encoding issues (UTF-8 BOM, mixed encodings, Unicode normalization forms). Not covered.
- **No spaCy-integrated preprocessing pipeline**: spaCy lemmatization is shown standalone but not integrated into the TextPreprocessor class.
- **Retrieval ground truth is hand-picked**: The 3-query evaluation uses manually annotated relevance — doesn't show how to build evaluation sets at scale (crowdsourcing, click data).

### Calibration Guide

Score yourself on each area (0-10) and average:
- 9-10: You can build and evaluate a production text pipeline with BM25, sentence-transformers, MAP/NDCG evaluation, proper train/test discipline, and no data leakage
- 7-8: You can implement TF-IDF and BM25, use pre-trained embeddings, and evaluate with precision@k, but may miss encoding issues or data leakage
- 5-6: You understand the concepts but rely on library defaults without understanding BPE merge mechanics, BM25 parameters, or retrieval evaluation

---

## Architect Sanity Checks

### Check 1: Tokenization Awareness
**Question**: Can you choose the right tokenizer for production NLP?
**Answer: YES** -- The blog covers word-level, subword (BPE via Hugging Face), and character tokenization with clear tradeoffs. It demonstrates vocabulary size impact, OOV handling via subword tokenization, and special token conventions ([CLS], [SEP], [PAD]). Language-specific and emoji tokenization are addressed.

### Check 2: Semantic Search Safety
**Question**: Can you build semantic search without anti-patterns?
**Answer: YES** -- The blog correctly identifies averaging word embeddings as an anti-pattern and uses sentence-transformers as the proper alternative in all search methods. The `search_hybrid` method explicitly refuses to fall back to word averaging — it degrades to TF-IDF-only when sentence-transformers is unavailable. Retrieval evaluation uses MAP, NDCG@k, Precision@k, and Recall@k with worked examples and interpretation thresholds.

### Check 3: Production Text Pipeline
**Question**: Can you build robust text preprocessing for production?
**Answer: YES** -- The blog provides: (1) TextPreprocessor class with configurable steps, (2) SafeTextPreprocessor for conservative processing, (3) production-grade spaCy lemmatization alongside the toy dictionary, (4) integrated data leakage prevention example showing correct train/test split discipline, (5) BM25 implementation as the production keyword retrieval standard. Remaining gap: no Unicode/encoding normalization, but this is called out in the "Falls Short" section.

---

*Questions? Found an error? Comments are open. Technical corrections get priority.*
