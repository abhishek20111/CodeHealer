"""
══════════════════════════════════════════════════════════════════════════════════
VECTOR STORE — ChromaDB-based Session & Memory Storage
══════════════════════════════════════════════════════════════════════════════════

THIS IS THE KEY DIFFERENCE FROM THE ORIGINAL PROJECT.

Original (ai_db_agent_langchain):
    sessions.json → flat file, stores ALL messages as plain JSON
    build_history_context() → sliding window (last 6 msgs) + simple text summary
    Retrieval: SEQUENTIAL — always uses last N messages regardless of relevance

Vector version (this project):
    ChromaDB → vector database, stores messages as EMBEDDINGS
    get_relevant_history() → SEMANTIC SEARCH — finds most relevant past messages
    Retrieval: SEMANTIC — retrieves messages by MEANING, not just recency

═══════════════════════════════════════════════════════════════════════════════════

ARCHITECTURE COMPARISON:

    ┌─────────────── OLD (JSON) ───────────────┐
    │                                           │
    │  User asks: "how many employees?"         │
    │       │                                   │
    │       ▼                                   │
    │  sessions.json → load ALL messages        │
    │       │                                   │
    │       ▼                                   │
    │  Sliding window: [msg7, msg8, ..., msg12] │
    │  (last 6 — even if irrelevant)            │
    │       │                                   │
    │       ▼                                   │
    │  Send to LLM → response                  │
    └───────────────────────────────────────────┘

    ┌─────────── NEW (Vector DB) ──────────────┐
    │                                           │
    │  User asks: "how many employees?"         │
    │       │                                   │
    │       ▼                                   │
    │  ChromaDB → embed query with nomic-embed  │
    │       │                                   │
    │       ▼                                   │
    │  Similarity search: find TOP-K messages   │
    │  that are SEMANTICALLY related            │
    │  (e.g., past Q about employees table)     │
    │       │                                   │
    │       ▼                                   │
    │  Send relevant context to LLM → response  │
    └───────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════════════

INTERVIEW QUESTIONS & ANSWERS:

Q: "Why use a vector database instead of simple JSON for chat history?"
A: "Vector databases enable SEMANTIC retrieval. Instead of always sending the
   last N messages (which may be irrelevant), we embed each message and retrieve
   only messages that are semantically similar to the current question. This means
   if the user asked about employees 20 messages ago, that context is still
   retrievable — even though a sliding window would have dropped it."

Q: "What embedding model are you using and why?"
A: "nomic-embed-text via Ollama. It's a 274MB model that runs locally — no API
   key needed, no data leaves the machine. It produces 768-dimensional embeddings
   that capture semantic meaning. For production, you'd use OpenAI's
   text-embedding-3-small (1536-dim, better quality) but it requires an API key."

Q: "What is ChromaDB?"
A: "ChromaDB is an open-source, AI-native vector database. It stores documents
   as embeddings (dense vectors) and supports similarity search using cosine
   distance. It can run in-memory or persist to disk. It's like a 'semantic
   search engine' for your data."

Q: "How does similarity search work internally?"
A: "Each message is converted to a 768-dimensional vector using the embedding
   model. When searching, the query is also embedded, and ChromaDB finds vectors
   with the smallest cosine distance (most similar direction in 768-D space).
   This is an Approximate Nearest Neighbor (ANN) search using HNSW algorithm."

Q: "What's the difference between ChromaDB and Pinecone/Weaviate/Qdrant?"
A: "ChromaDB: Lightweight, embedded (runs in-process), great for prototyping.
   Pinecone: Managed cloud service, scales to billions of vectors, production-grade.
   Weaviate: Self-hosted, supports hybrid search (vector + keyword).
   Qdrant: Rust-based, very fast, good filtering. For local dev, ChromaDB is ideal."

Q: "What are the tradeoffs of semantic vs. sequential history?"
A: "Semantic: ✅ Finds relevant context regardless of when it occurred
              ✅ Works even for very long conversations
              ❌ May miss important sequential context (e.g., correction chains)
              ❌ Embedding adds ~50ms latency per message
   Sequential: ✅ Preserves conversation flow and corrections
               ✅ Zero overhead (just array slice)
               ❌ Loses old context when window slides
               ❌ Sends irrelevant messages to LLM"

═══════════════════════════════════════════════════════════════════════════════════

COLLECTIONS DESIGN:

We use TWO ChromaDB collections:

1. "sessions" — Session metadata (title, creation time)
   - id: session_id (e.g., "s_abc123")
   - document: session title
   - metadata: {created_at, message_count}

2. "messages" — ALL conversation messages across all sessions
   - id: unique message ID (e.g., "s_abc123_msg_0")
   - document: message content (what gets embedded)
   - metadata: {session_id, role, timestamp, index}

WHY THIS DESIGN:
- "messages" collection enables cross-session semantic search
  (find relevant past answers from ANY session)
- Metadata filtering (session_id) lets us scope to current session
- The embedding is on the content — so similar questions find similar past answers

══════════════════════════════════════════════════════════════════════════════════
"""

import os
import json
from datetime import datetime
from typing import Optional

import chromadb
from chromadb.config import Settings
from chromadb.api.types import EmbeddingFunction, Documents
try:
    # Temporarily disable proxy for Ollama import
    import os
    _old_http_proxy = os.environ.pop('HTTP_PROXY', None)
    _old_https_proxy = os.environ.pop('HTTPS_PROXY', None) 
    _old_http_proxy_lower = os.environ.pop('http_proxy', None)
    _old_https_proxy_lower = os.environ.pop('https_proxy', None)
    
    from langchain_ollama import OllamaEmbeddings
    OLLAMA_AVAILABLE = True
    
    # Restore proxy settings
    if _old_http_proxy:
        os.environ['HTTP_PROXY'] = _old_http_proxy
    if _old_https_proxy:
        os.environ['HTTPS_PROXY'] = _old_https_proxy
    if _old_http_proxy_lower:
        os.environ['http_proxy'] = _old_http_proxy_lower
    if _old_https_proxy_lower:
        os.environ['https_proxy'] = _old_https_proxy_lower
except (ImportError, Exception) as e:
    print(f"⚠️  Warning: Could not load Ollama embeddings: {e}")
    OLLAMA_AVAILABLE = False
    OllamaEmbeddings = None
import requests

# ── Logging ──────────────────────────────────────────────────────────────────

LOG_SEP = "─" * 60
LOG_BOX_TOP = "╔" + "═" * 58 + "╗"
LOG_BOX_BOT = "╚" + "═" * 58 + "╝"
LOG_BOX_MID = "╟" + "─" * 58 + "╢"

# ANSI colors for terminal
class C:
    PURPLE = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"


def log(tag: str, msg: str):
    """Structured console log with timestamp and colors."""
    colors = {
        "VECTOR": C.PURPLE,
        "EMBED": C.CYAN,
        "SEARCH": C.BLUE,
        "STORE": C.GREEN,
        "SESSION": C.YELLOW,
    }
    color = colors.get(tag, C.RESET)
    timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
    print(f"{C.DIM}[{timestamp}]{C.RESET} {color}[{tag}]{C.RESET} {msg}")


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

# Where ChromaDB persists data on disk
# This replaces sessions.json entirely
CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")

# Embedding model (already downloaded via `ollama pull nomic-embed-text`)
EMBED_MODEL = "nomic-embed-text"
OLLAMA_BASE_URL = "http://localhost:11434"

# How many relevant messages to retrieve for context
# INTERVIEW: "We retrieve top-K semantically similar messages. K=6 gives
# enough context without overwhelming the LLM's attention mechanism."
TOP_K_MESSAGES = 12

# Also keep the last N messages for sequential continuity
# INTERVIEW: "Hybrid approach: semantic retrieval + recency window.
# This ensures the agent sees both relevant AND recent context."
LAST_N_MESSAGES = 8


# ══════════════════════════════════════════════════════════════════════════════
# CHROMADB CLIENT & EMBEDDINGS
#
# KEY CONCEPT: ChromaDB runs as an embedded database (in-process).
# No separate server needed — unlike Postgres or Redis.
# Data persists to disk at CHROMA_PERSIST_DIR.
#
# INTERVIEW: "ChromaDB's PersistentClient is ideal for single-process apps.
# For multi-process or distributed systems, you'd use ChromaDB's client-server
# mode (chroma run) or a managed service like Pinecone."
# ══════════════════════════════════════════════════════════════════════════════

# Custom ChromaDB embedding function that uses Ollama
class OllamaChromaEmbeddingFunction(EmbeddingFunction[Documents]):
    """
    ChromaDB embedding function that uses Ollama's API directly.
    ChromaDB requires a specific interface, so we can't use LangChain's OllamaEmbeddings.
    """
    def __init__(self, model: str, base_url: str):
        self.model = model
        self.base_url = base_url.rstrip('/')
    
    def __call__(self, input: Documents) -> list[list[float]]:
        """Embed a list of documents using Ollama."""
        embeddings = []
        for text in input:
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model, "prompt": text}
            )
            response.raise_for_status()
            embeddings.append(response.json()["embedding"])
        return embeddings

# Initialize ChromaDB client with disk persistence
_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

# Initialize embedding function for ChromaDB collections
_chroma_embeddings = OllamaChromaEmbeddingFunction(
    model=EMBED_MODEL,
    base_url=OLLAMA_BASE_URL,
)

# Initialize embedding function using Ollama's nomic-embed-text
# This converts text → 768-dimensional vectors
#
# INTERVIEW: "OllamaEmbeddings wraps the /api/embeddings endpoint of Ollama.
# It sends text to the local nomic-embed-text model and receives a 768-dim
# float vector back. This vector captures semantic meaning — similar texts
# produce similar vectors (small cosine distance)."
if OLLAMA_AVAILABLE:
    _embeddings = OllamaEmbeddings(
        model=EMBED_MODEL,
        base_url=OLLAMA_BASE_URL,
    )
else:
    _embeddings = None


def _get_messages_collection():
    """
    Get or create the 'messages' collection.
    
    ChromaDB collection = like a table in SQL, but for vectors.
    Each document has: id, embedding, metadata, and the raw text.
    
    We use cosine similarity (default) for distance metric.
    INTERVIEW: "Cosine similarity measures the angle between vectors,
    ignoring magnitude. This makes it robust to different text lengths —
    a short question and a long answer about the same topic will still
    have high similarity."
    """
    return _client.get_or_create_collection(
        name="messages",
        metadata={"hnsw:space": "cosine"},  # Use cosine distance
        embedding_function=_chroma_embeddings,  # Use Ollama embeddings
    )


def _get_sessions_collection():
    """
    Get or create the 'sessions' collection.
    Stores session metadata (title, timestamps).
    
    NOTE: We don't really need embeddings for sessions — we just use
    ChromaDB as a convenient persistent key-value store here.
    """
    return _client.get_or_create_collection(
        name="sessions",
        metadata={"hnsw:space": "cosine"},
        embedding_function=_chroma_embeddings,  # Use Ollama embeddings
    )


# ── Public helpers for other modules (e.g., query caching in tools.py) ──────

def get_collection(name: str):
    """Get or create a named ChromaDB collection."""
    return _client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
        embedding_function=_chroma_embeddings,
    )


def embed_text(text: str) -> list[float]:
    """Embed a single text string using nomic-embed-text and return the vector."""
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text}
    )
    response.raise_for_status()
    return response.json()["embedding"]


# ══════════════════════════════════════════════════════════════════════════════
# SESSION MANAGEMENT
#
# Replaces: load_sessions(), save_sessions(), get_session(), update_session()
# from the original web_server.py
#
# DIFFERENCE:
#   OLD: Read/write entire JSON file on every operation (O(n) for n sessions)
#   NEW: ChromaDB handles indexing, only reads what's needed (O(1) lookup)
# ══════════════════════════════════════════════════════════════════════════════


def create_session(session_id: str, title: str) -> dict:
    """
    Create a new chat session in ChromaDB.
    
    Stores session metadata in the 'sessions' collection.
    Messages will be stored separately in the 'messages' collection.
    """
    print(f"\n{C.YELLOW}{LOG_BOX_TOP}{C.RESET}")
    print(f"{C.YELLOW}║{C.RESET} {C.BOLD}📁 CREATING NEW SESSION{C.RESET}")
    print(f"{C.YELLOW}{LOG_BOX_MID}{C.RESET}")
    print(f"{C.YELLOW}║{C.RESET}   Session ID : {C.CYAN}{session_id}{C.RESET}")
    print(f"{C.YELLOW}║{C.RESET}   Title      : {title[:60]}")
    print(f"{C.YELLOW}{LOG_BOX_BOT}{C.RESET}")
    
    sessions = _get_sessions_collection()
    now = datetime.now().isoformat()
    
    sessions.upsert(
        ids=[session_id],
        documents=[title],  # Title is the "document" (gets embedded, allows search)
        metadatas=[{
            "created_at": now,
            "message_count": 0,
            "title": title[:100],
        }],
    )
    
    log("SESSION", f"✅ Session created and stored in ChromaDB 'sessions' collection")
    log("SESSION", f"   Collection now has {_get_sessions_collection().count()} total sessions")
    
    return {
        "id": session_id,
        "title": title[:100],
        "created_at": now,
        "messages": [],
    }


def get_session(session_id: str) -> Optional[dict]:
    """
    Load a session and ALL its messages from ChromaDB.
    
    DIFFERENCE FROM OLD VERSION:
    - Old: json.load(file) → scan list for matching id → return
    - New: collection.get(id) → O(1) lookup by primary key
    
    Returns dict with {id, title, created_at, messages: [{role, content}]}
    """
    sessions = _get_sessions_collection()
    
    try:
        result = sessions.get(ids=[session_id])
        if not result["ids"]:
            return None
    except Exception:
        return None
    
    meta = result["metadatas"][0]
    
    # Load all messages for this session (ordered by index)
    messages = get_session_messages(session_id)
    
    return {
        "id": session_id,
        "title": meta.get("title", ""),
        "created_at": meta.get("created_at", ""),
        "messages": messages,
    }


def get_session_messages(session_id: str) -> list:
    """
    Get all messages for a session, ordered by index.
    
    Uses metadata filtering: WHERE session_id = X
    Then sorts by the 'index' field.
    
    INTERVIEW: "ChromaDB supports metadata filtering with $eq, $gt, $lt, etc.
    This lets us scope queries to a specific session without scanning all data."
    """
    messages_col = _get_messages_collection()
    
    try:
        result = messages_col.get(
            where={"session_id": session_id},
            include=["documents", "metadatas"],
        )
    except Exception:
        return []
    
    if not result["ids"]:
        return []
    
    # Zip and sort by index
    items = list(zip(result["ids"], result["documents"], result["metadatas"]))
    items.sort(key=lambda x: x[2].get("index", 0))
    
    return [
        {"role": item[2]["role"], "content": item[1]}
        for item in items
    ]


def list_sessions() -> list:
    """
    List all sessions with metadata (for sidebar display).
    
    Returns: [{id, title, created_at, message_count}, ...]
    Sorted by creation time (newest first).
    """
    sessions = _get_sessions_collection()
    
    result = sessions.get(include=["metadatas", "documents"])
    
    if not result["ids"]:
        return []
    
    items = []
    for sid, meta, doc in zip(result["ids"], result["metadatas"], result["documents"]):
        items.append({
            "id": sid,
            "title": meta.get("title", doc[:40] if doc else ""),
            "created_at": meta.get("created_at", ""),
            "message_count": meta.get("message_count", 0),
        })
    
    # Sort newest first
    items.sort(key=lambda x: x["created_at"], reverse=True)
    return items


def delete_session(session_id: str) -> bool:
    """
    Delete a session and all its messages.
    
    INTERVIEW: "We delete from both collections — the session metadata AND
    all associated messages. This is like CASCADE DELETE in SQL."
    """
    print(f"\n{C.RED}  ┌─── 🗑️  DELETING SESSION ─────────────────────────────────┐{C.RESET}")
    print(f"{C.RED}  │{C.RESET} Session: {session_id}")
    
    sessions = _get_sessions_collection()
    messages_col = _get_messages_collection()
    
    # Delete session metadata
    try:
        sessions.delete(ids=[session_id])
        print(f"{C.RED}  │{C.RESET} ✅ Session metadata deleted from 'sessions' collection")
    except Exception:
        print(f"{C.RED}  │{C.RESET} ❌ Failed to delete session metadata")
        print(f"{C.RED}  └──────────────────────────────────────────────────────────┘{C.RESET}")
        return False
    
    # Delete all messages for this session
    try:
        # Get IDs of messages belonging to this session
        result = messages_col.get(
            where={"session_id": session_id},
        )
        if result["ids"]:
            messages_col.delete(ids=result["ids"])
            print(f"{C.RED}  │{C.RESET} ✅ Deleted {len(result['ids'])} messages from 'messages' collection")
        else:
            print(f"{C.RED}  │{C.RESET} ℹ️  No messages found for this session")
    except Exception:
        pass  # Session messages may not exist
    
    print(f"{C.RED}  └──────────────────────────────────────────────────────────┘{C.RESET}")
    return True


# ══════════════════════════════════════════════════════════════════════════════
# MESSAGE STORAGE — Add messages with embeddings
#
# THIS IS WHERE THE MAGIC HAPPENS.
# Each message gets embedded (converted to a 768-dim vector) and stored.
# Later, we can find semantically similar messages via vector search.
#
# INTERVIEW: "Every message goes through the embedding model before storage.
# This adds ~50ms per message (local model) but enables O(1) semantic retrieval
# later. The tradeoff: slightly slower writes for much smarter reads."
# ══════════════════════════════════════════════════════════════════════════════


def add_message(session_id: str, role: str, content: str) -> str:
    """
    Add a single message to ChromaDB with embedding.
    
    Args:
        session_id: Which session this belongs to
        role: "user" or "assistant"
        content: The message text (this gets embedded)
    
    Returns:
        The message ID
    
    PROCESS:
    1. Generate unique ID: "s_abc123_msg_5"
    2. Compute embedding: nomic-embed-text(content) → [0.23, -0.45, ...] (768 floats)
    3. Store in ChromaDB: {id, embedding, content, metadata}
    4. Update session message count
    
    INTERVIEW: "We embed the raw content. For user messages, this captures
    the question semantics. For assistant messages, this captures the answer
    semantics. When searching later, a similar question will match both the
    original question AND its answer."
    """
    messages_col = _get_messages_collection()
    sessions = _get_sessions_collection()
    
    print(f"\n{C.GREEN}  ┌─── 💾 STORING MESSAGE ───────────────────────────────────┐{C.RESET}")
    print(f"{C.GREEN}  │{C.RESET} Session  : {C.CYAN}{session_id}{C.RESET}")
    print(f"{C.GREEN}  │{C.RESET} Role     : {C.BOLD}{'🧑 user' if role == 'user' else '🤖 assistant'}{C.RESET}")
    print(f"{C.GREEN}  │{C.RESET} Content  : \"{content[:70]}{'...' if len(content) > 70 else ''}\"")
    print(f"{C.GREEN}  │{C.RESET} Size     : {len(content)} characters")
    
    # Get current message count for this session (to set index)
    try:
        session_result = sessions.get(ids=[session_id])
        if session_result["ids"]:
            current_count = session_result["metadatas"][0].get("message_count", 0)
        else:
            current_count = 0
    except Exception:
        current_count = 0
    
    # Generate message ID
    msg_id = f"{session_id}_msg_{current_count}"
    now = datetime.now().isoformat()
    
    print(f"{C.GREEN}  │{C.RESET} Msg ID   : {C.CYAN}{msg_id}{C.RESET}")
    print(f"{C.GREEN}  │{C.RESET} Index    : {current_count} (position in session)")
    print(f"{C.GREEN}  │{C.RESET}")
    
    # Embed and store the message
    # ChromaDB will call our embedding function internally via add()
    # But since we're using raw chromadb (not langchain_chroma), we embed manually
    import time
    embed_start = time.time()
    print(f"{C.GREEN}  │{C.RESET} {C.CYAN}⏳ Embedding with nomic-embed-text...{C.RESET}")
    print(f"{C.GREEN}  │{C.RESET}    Input text ({min(len(content), 2000)} chars) → Ollama API → 768-dim vector")
    try:
        embedding = _embeddings.embed_query(content[:2000])  # Cap content for embedding
        embed_time = (time.time() - embed_start) * 1000
        print(f"{C.GREEN}  │{C.RESET}    {C.GREEN}✅ Embedded in {embed_time:.0f}ms{C.RESET}")
        print(f"{C.GREEN}  │{C.RESET}    Vector: [{embedding[0]:.4f}, {embedding[1]:.4f}, {embedding[2]:.4f}, ... ] ({len(embedding)} dims)")
    except Exception as e:
        log("EMBED", f"⚠️ Embedding failed: {e} — storing without embedding")
        embedding = None
    
    # Store in messages collection
    add_kwargs = {
        "ids": [msg_id],
        "documents": [content],
        "metadatas": [{
            "session_id": session_id,
            "role": role,
            "timestamp": now,
            "index": current_count,
        }],
    }
    if embedding:
        add_kwargs["embeddings"] = [embedding]
    
    messages_col.upsert(**add_kwargs)
    
    print(f"{C.GREEN}  │{C.RESET}")
    print(f"{C.GREEN}  │{C.RESET} {C.GREEN}✅ SAVED to ChromaDB 'messages' collection{C.RESET}")
    print(f"{C.GREEN}  │{C.RESET}    Total messages in DB: {messages_col.count()}")
    
    # Update session message count
    try:
        sessions.update(
            ids=[session_id],
            metadatas=[{
                "created_at": session_result["metadatas"][0].get("created_at", now) if session_result["ids"] else now,
                "message_count": current_count + 1,
                "title": session_result["metadatas"][0].get("title", "") if session_result["ids"] else "",
            }],
        )
    except Exception:
        pass
    
    print(f"{C.GREEN}  └────────────────────────────────────────────────────────────┘{C.RESET}")
    return msg_id


def add_messages_batch(session_id: str, messages: list[dict]) -> None:
    """
    Add multiple messages at once (e.g., user question + assistant answer).
    
    Args:
        messages: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    """
    for msg in messages:
        add_message(session_id, msg["role"], msg["content"])


# ══════════════════════════════════════════════════════════════════════════════
# SEMANTIC RETRIEVAL — The killer feature of vector storage
#
# This is what makes the vector version FUNDAMENTALLY different from JSON.
# Instead of "give me the last 6 messages", we ask:
# "give me the messages most RELEVANT to what the user is asking now"
#
# ARCHITECTURE:
#
#   User question: "how many employees are there?"
#        │
#        ▼
#   Embed with nomic-embed-text → [0.34, -0.12, 0.87, ...] (768 dims)
#        │
#        ▼
#   ChromaDB similarity search:
#   - Compare query vector against ALL stored message vectors
#   - Use cosine similarity (angle between vectors)
#   - Return top-K most similar messages
#        │
#        ▼
#   Results (ranked by relevance):
#   1. "Q: how many rows in employees table?" (sim=0.94) ← HIGHLY relevant!
#   2. "A: There are 150 employees in the..." (sim=0.89)
#   3. "Q: show me department names" (sim=0.45) ← less relevant
#
# INTERVIEW: "This is essentially the same architecture as RAG (Retrieval
# Augmented Generation), but applied to conversation history instead of
# documents. The LLM gets surgically relevant context, not just recency-based."
# ══════════════════════════════════════════════════════════════════════════════


def get_relevant_history(session_id: str, query: str, top_k: int = TOP_K_MESSAGES) -> list:
    """
    Retrieve the most semantically relevant past messages for the current query.
    
    This is the CORE FUNCTION that replaces build_history_context() from the old version.
    
    Args:
        session_id: Current session ID
        query: The user's current question
        top_k: How many relevant messages to retrieve
    
    Returns:
        List of {"role": ..., "content": ...} ordered by relevance
    
    ALGORITHM:
    1. Embed the query
    2. Search ChromaDB for similar messages in THIS session
    3. Also get the last N messages (for sequential continuity)
    4. Merge and deduplicate
    5. Sort by message index (chronological order)
    
    WHY HYBRID (semantic + recency):
    - Semantic alone might miss the last exchange (e.g., "can you clarify?")
    - Recency alone misses relevant old context (original problem)
    - Hybrid gives best of both worlds
    
    INTERVIEW: "I use a hybrid retrieval strategy: semantic search for relevance
    plus a recency window for continuity. This handles both 'tell me more about
    what you said 10 messages ago' AND 'yes, do that' (which needs recent context)."
    """
    import time
    messages_col = _get_messages_collection()
    
    print(f"\n{C.BLUE}{LOG_BOX_TOP}{C.RESET}")
    print(f"{C.BLUE}║{C.RESET} {C.BOLD}🔍 SEMANTIC RETRIEVAL (get_relevant_history){C.RESET}")
    print(f"{C.BLUE}{LOG_BOX_MID}{C.RESET}")
    print(f"{C.BLUE}║{C.RESET}   Session : {C.CYAN}{session_id}{C.RESET}")
    print(f"{C.BLUE}║{C.RESET}   Query   : \"{query[:60]}\"")
    print(f"{C.BLUE}║{C.RESET}   Top-K   : {top_k} (semantic) + {LAST_N_MESSAGES} (recent)")
    print(f"{C.BLUE}{LOG_BOX_MID}{C.RESET}")
    
    # Check if session has any messages
    try:
        all_session = messages_col.get(
            where={"session_id": session_id},
            include=["documents", "metadatas"],
        )
    except Exception:
        print(f"{C.BLUE}║{C.RESET}   {C.RED}❌ Failed to query session messages{C.RESET}")
        print(f"{C.BLUE}{LOG_BOX_BOT}{C.RESET}")
        return []
    
    if not all_session["ids"]:
        print(f"{C.BLUE}║{C.RESET}   📭 No messages found — fresh conversation")
        print(f"{C.BLUE}{LOG_BOX_BOT}{C.RESET}")
        return []
    
    total_messages = len(all_session["ids"])
    print(f"{C.BLUE}║{C.RESET}   📊 Session has {C.BOLD}{total_messages}{C.RESET} stored messages")
    
    # If session is small enough, just return all messages
    if total_messages <= top_k + LAST_N_MESSAGES:
        print(f"{C.BLUE}║{C.RESET}   ↳ Small session ({total_messages} ≤ {top_k + LAST_N_MESSAGES}) → returning ALL messages")
        items = list(zip(all_session["ids"], all_session["documents"], all_session["metadatas"]))
        items.sort(key=lambda x: x[2].get("index", 0))
        print(f"{C.BLUE}║{C.RESET}")
        for item in items:
            role_icon = "🧑" if item[2]["role"] == "user" else "🤖"
            print(f"{C.BLUE}║{C.RESET}     {role_icon} [{item[2]['role']:9}] \"{item[1][:50]}{'...' if len(item[1]) > 50 else ''}\"")
        print(f"{C.BLUE}{LOG_BOX_BOT}{C.RESET}")
        return [{"role": item[2]["role"], "content": item[1]} for item in items]
    
    # ── STEP 1: Semantic search — find relevant past messages ────────────
    print(f"{C.BLUE}║{C.RESET}")
    print(f"{C.BLUE}║{C.RESET}   {C.CYAN}STEP 1: Embedding query...{C.RESET}")
    embed_start = time.time()
    try:
        query_embedding = _embeddings.embed_query(query)
        embed_ms = (time.time() - embed_start) * 1000
        print(f"{C.BLUE}║{C.RESET}     ✅ Query embedded in {embed_ms:.0f}ms → [{query_embedding[0]:.4f}, {query_embedding[1]:.4f}, ...] ({len(query_embedding)} dims)")
    except Exception as e:
        print(f"{C.BLUE}║{C.RESET}     {C.RED}❌ Embedding failed: {e}{C.RESET}")
        print(f"{C.BLUE}║{C.RESET}     ↳ Falling back to recency-only")
        items = list(zip(all_session["ids"], all_session["documents"], all_session["metadatas"]))
        items.sort(key=lambda x: x[2].get("index", 0))
        print(f"{C.BLUE}{LOG_BOX_BOT}{C.RESET}")
        return [{"role": item[2]["role"], "content": item[1]} for item in items[-top_k:]]
    
    print(f"{C.BLUE}║{C.RESET}")
    print(f"{C.BLUE}║{C.RESET}   {C.CYAN}STEP 2: Cosine similarity search in ChromaDB...{C.RESET}")
    search_start = time.time()
    try:
        semantic_results = messages_col.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where={"session_id": session_id},
            include=["documents", "metadatas", "distances"],
        )
        search_ms = (time.time() - search_start) * 1000
        print(f"{C.BLUE}║{C.RESET}     ✅ Search completed in {search_ms:.0f}ms")
    except Exception as e:
        print(f"{C.BLUE}║{C.RESET}     {C.RED}❌ Search failed: {e}{C.RESET}")
        items = list(zip(all_session["ids"], all_session["documents"], all_session["metadatas"]))
        items.sort(key=lambda x: x[2].get("index", 0))
        print(f"{C.BLUE}{LOG_BOX_BOT}{C.RESET}")
        return [{"role": item[2]["role"], "content": item[1]} for item in items[-top_k:]]
    
    # Display semantic results with similarity bars
    print(f"{C.BLUE}║{C.RESET}")
    print(f"{C.BLUE}║{C.RESET}   {C.CYAN}STEP 3: Semantic results (ranked by similarity):{C.RESET}")
    if semantic_results["ids"] and semantic_results["ids"][0]:
        for i, (doc, meta, dist) in enumerate(zip(
            semantic_results["documents"][0],
            semantic_results["metadatas"][0],
            semantic_results["distances"][0],
        )):
            sim_pct = (1 - dist) * 100
            bar_len = int(sim_pct / 10)
            bar = "█" * bar_len + "░" * (10 - bar_len)
            role_icon = "🧑" if meta["role"] == "user" else "🤖"
            print(f"{C.BLUE}║{C.RESET}     #{i+1} [{bar}] {sim_pct:.1f}% {role_icon} \"{doc[:45]}{'...' if len(doc) > 45 else ''}\"")
    
    # ── STEP 2: Get last N messages for recency ──────────────────────────
    print(f"{C.BLUE}║{C.RESET}")
    print(f"{C.BLUE}║{C.RESET}   {C.CYAN}STEP 4: Adding last {LAST_N_MESSAGES} messages (recency window):{C.RESET}")
    items_all = list(zip(all_session["ids"], all_session["documents"], all_session["metadatas"]))
    items_all.sort(key=lambda x: x[2].get("index", 0))
    recent_items = items_all[-LAST_N_MESSAGES:]
    for msg_id, doc, meta in recent_items:
        role_icon = "🧑" if meta["role"] == "user" else "🤖"
        print(f"{C.BLUE}║{C.RESET}     {role_icon} [idx={meta.get('index',0)}] \"{doc[:45]}{'...' if len(doc) > 45 else ''}\"")
    
    # ── STEP 3: Merge semantic + recency (deduplicate by ID) ─────────────
    seen_ids = set()
    merged = []
    
    # Add semantic results
    if semantic_results["ids"] and semantic_results["ids"][0]:
        for msg_id, doc, meta, dist in zip(
            semantic_results["ids"][0],
            semantic_results["documents"][0],
            semantic_results["metadatas"][0],
            semantic_results["distances"][0],
        ):
            if msg_id not in seen_ids:
                seen_ids.add(msg_id)
                merged.append({
                    "id": msg_id,
                    "role": meta["role"],
                    "content": doc,
                    "index": meta.get("index", 0),
                    "distance": dist,
                })
    
    # Add recent messages (if not already in semantic results)
    recency_added = 0
    for msg_id, doc, meta in recent_items:
        if msg_id not in seen_ids:
            seen_ids.add(msg_id)
            recency_added += 1
            merged.append({
                "id": msg_id,
                "role": meta["role"],
                "content": doc,
                "index": meta.get("index", 0),
                "distance": 999,  # Mark as recency-based
            })
    
    # ── STEP 4: Sort by index (chronological order for LLM) ──────────────
    merged.sort(key=lambda x: x["index"])
    
    semantic_count = len(semantic_results['ids'][0]) if semantic_results['ids'] else 0
    print(f"{C.BLUE}║{C.RESET}")
    print(f"{C.BLUE}║{C.RESET}   {C.CYAN}STEP 5: Merge + Deduplicate + Sort{C.RESET}")
    print(f"{C.BLUE}║{C.RESET}     Semantic hits : {semantic_count}")
    print(f"{C.BLUE}║{C.RESET}     Recency added : {recency_added} (not already in semantic)")
    print(f"{C.BLUE}║{C.RESET}     Duplicates    : {semantic_count + len(recent_items) - len(merged)} removed")
    print(f"{C.BLUE}║{C.RESET}     {C.GREEN}Final context : {len(merged)} messages (sorted chronologically){C.RESET}")
    print(f"{C.BLUE}║{C.RESET}")
    print(f"{C.BLUE}║{C.RESET}   📤 Final messages being sent to LLM:")
    for m in merged:
        role_icon = "🧑" if m["role"] == "user" else "🤖"
        source = f"sim={1-m['distance']:.0%}" if m['distance'] != 999 else "recency"
        print(f"{C.BLUE}║{C.RESET}     {role_icon} [idx={m['index']}|{source}] \"{m['content'][:45]}{'...' if len(m['content']) > 45 else ''}\"")
    print(f"{C.BLUE}{LOG_BOX_BOT}{C.RESET}")
    
    return [{"role": m["role"], "content": m["content"]} for m in merged]


def search_across_sessions(query: str, top_k: int = 5) -> list:
    """
    Search ALL sessions for relevant messages (cross-session memory).
    
    This is a BONUS feature not possible with JSON storage.
    If the user asked something similar in a DIFFERENT session, we can find it.
    
    INTERVIEW: "Cross-session semantic search is like having a searchable
    knowledge base of all past conversations. The agent can recall relevant
    answers from weeks ago, even from different sessions."
    
    Returns: [{session_id, role, content, distance}, ...]
    """
    print(f"\n{C.PURPLE}  ┌─── 🌐 CROSS-SESSION SEARCH ─────────────────────────────┐{C.RESET}")
    print(f"{C.PURPLE}  │{C.RESET} Query: \"{query[:60]}\"")
    print(f"{C.PURPLE}  │{C.RESET} Searching ALL sessions (top {top_k})...")
    
    messages_col = _get_messages_collection()
    
    try:
        query_embedding = _embeddings.embed_query(query)
        results = messages_col.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as e:
        print(f"{C.PURPLE}  │{C.RESET} {C.RED}❌ Search failed: {e}{C.RESET}")
        print(f"{C.PURPLE}  └──────────────────────────────────────────────────────────┘{C.RESET}")
        return []
    
    if not results["ids"] or not results["ids"][0]:
        print(f"{C.PURPLE}  │{C.RESET} 📭 No results found")
        print(f"{C.PURPLE}  └──────────────────────────────────────────────────────────┘{C.RESET}")
        return []
    
    print(f"{C.PURPLE}  │{C.RESET} Found {len(results['ids'][0])} results:")
    for i, (doc, meta, dist) in enumerate(zip(results["documents"][0], results["metadatas"][0], results["distances"][0])):
        sim_pct = (1 - dist) * 100
        role_icon = "🧑" if meta["role"] == "user" else "🤖"
        print(f"{C.PURPLE}  │{C.RESET}   #{i+1} ({sim_pct:.0f}%) {role_icon} [{meta['session_id']}] \"{doc[:40]}...\"")
    print(f"{C.PURPLE}  └──────────────────────────────────────────────────────────┘{C.RESET}")
    
    return [
        {
            "session_id": meta["session_id"],
            "role": meta["role"],
            "content": doc,
            "distance": dist,
        }
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]


# ══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════


def get_stats() -> dict:
    """Get vector store statistics."""
    messages_col = _get_messages_collection()
    sessions_col = _get_sessions_collection()
    
    return {
        "total_messages": messages_col.count(),
        "total_sessions": sessions_col.count(),
        "persist_dir": CHROMA_PERSIST_DIR,
        "embed_model": EMBED_MODEL,
    }


def reset_all():
    """
    Delete all data (for testing/development).
    WARNING: This is irreversible!
    """
    log("VECTOR", "⚠️ RESETTING ALL DATA")
    _client.delete_collection("messages")
    _client.delete_collection("sessions")
    log("VECTOR", "All collections deleted")
