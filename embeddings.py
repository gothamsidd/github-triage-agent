"""
embeddings.py — Semantic duplicate detection via embeddings + FAISS.

Provider priority:
  1. Voyage AI  (VOYAGE_API_KEY)  — free 200M tokens/month, voyage-3-lite
  2. OpenAI     (OPENAI_API_KEY)  — text-embedding-3-small, paid
  3. TF-IDF fallback              — no key needed, keyword matching only

Falls back silently to TF-IDF (in tools.py) when no embedding provider
is available or faiss-cpu is not installed.

The FAISS index is persisted to .faiss_cache/ so embeddings are never
recomputed for issues already indexed. Cache filename includes the model
name to prevent dimension mismatches when switching providers.
"""

import os
import pickle
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Optional dependencies — graceful degradation if not installed
# ---------------------------------------------------------------------------

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

# ── Voyage AI (preferred — free tier) ────────────────────────────────────────
_voyage_client = None
try:
    import voyageai
    _voyage_key = os.getenv("VOYAGE_API_KEY")
    if _voyage_key:
        _voyage_client = voyageai.Client(api_key=_voyage_key)
except ImportError:
    pass

# ── OpenAI (fallback — paid) ─────────────────────────────────────────────────
_openai_client = None
try:
    from openai import OpenAI as _OAI
    _oai_key = os.getenv("OPENAI_API_KEY")
    if _oai_key:
        _openai_client = _OAI(api_key=_oai_key)
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Active provider — Voyage wins if both keys are set
# ---------------------------------------------------------------------------

if _voyage_client:
    EMBEDDING_MODEL = "voyage-3-lite"
    EMBEDDING_DIM   = 512
    _provider       = "voyage"
elif _openai_client:
    EMBEDDING_MODEL = "text-embedding-3-small"
    EMBEDDING_DIM   = 1536
    _provider       = "openai"
else:
    EMBEDDING_MODEL = ""
    EMBEDDING_DIM   = 512
    _provider       = None

EMBEDDINGS_READY    = FAISS_AVAILABLE and (_provider is not None)
DUPLICATE_THRESHOLD = 0.85

CACHE_DIR = Path(__file__).parent / ".faiss_cache"


# ---------------------------------------------------------------------------
# Core embedding functions
# ---------------------------------------------------------------------------

def embed(text: str) -> Optional[np.ndarray]:
    """Convert text to a unit-normalized embedding vector."""
    result = embed_batch([text])
    return result[0] if result else None


def embed_batch(texts: list[str]) -> list[Optional[np.ndarray]]:
    """
    Convert a list of texts to unit-normalized vectors in a single API call.

    One call regardless of batch size — 40× faster than calling embed() in a loop.
    Returns a list of the same length; individual entries are None on failure.
    """
    if not _provider or not texts:
        return [None] * len(texts)

    try:
        if _provider == "voyage":
            result = _voyage_client.embed(
                [t[:8000] for t in texts],
                model=EMBEDDING_MODEL,
                input_type="document",
            )
            raw_vecs = result.embeddings
        else:
            resp     = _openai_client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=[t[:8000] for t in texts],
            )
            raw_vecs = [item.embedding for item in resp.data]

        out = []
        for raw in raw_vecs:
            vec  = np.array(raw, dtype="float32")
            norm = np.linalg.norm(vec)
            out.append(vec / norm if norm > 0 else vec)
        return out

    except Exception as exc:
        print(f"[WARN] Batch embedding API call failed ({_provider}): {exc}")
        return [None] * len(texts)


# ---------------------------------------------------------------------------
# Persistent FAISS index per repository
# ---------------------------------------------------------------------------

class IssueIndex:
    """
    FAISS index + metadata store for one GitHub repository.

    Cache filename includes the model name so switching providers
    (e.g. OpenAI → Voyage) automatically creates a fresh index with
    the correct dimensions rather than loading a mismatched one.
    """

    def __init__(self, repo_name: str):
        self.repo_name = repo_name
        safe           = repo_name.replace("/", "_")
        model_tag      = EMBEDDING_MODEL.replace("/", "-").replace(".", "-")
        self._path     = CACHE_DIR / f"{safe}_{model_tag}.pkl"
        self.index: Optional[object] = None
        self.metadata: list[dict]    = []
        self._load()

    def _load(self):
        CACHE_DIR.mkdir(exist_ok=True)
        if self._path.exists() and FAISS_AVAILABLE:
            try:
                with open(self._path, "rb") as f:
                    saved = pickle.load(f)
                self.index    = saved["index"]
                self.metadata = saved["metadata"]
                return
            except Exception:
                pass
        if FAISS_AVAILABLE:
            self.index = faiss.IndexFlatIP(EMBEDDING_DIM)

    def _save(self):
        if not FAISS_AVAILABLE or self.index is None:
            return
        with open(self._path, "wb") as f:
            pickle.dump({"index": self.index, "metadata": self.metadata}, f)

    def add(self, issues: list[dict]) -> int:
        """Embed and index a batch of issues. Skips already-indexed issues."""
        if not EMBEDDINGS_READY or self.index is None:
            return 0

        existing = {m["number"] for m in self.metadata}
        new      = [i for i in issues if i["number"] not in existing]

        if not new:
            return 0

        texts = [f"{i['title']} {i.get('body', '')}" for i in new]
        vecs  = embed_batch(texts)
        added = 0

        for issue, vec in zip(new, vecs):
            if vec is None:
                continue
            self.index.add(vec.reshape(1, -1))
            self.metadata.append({
                "number": issue["number"],
                "title":  issue["title"],
                "url":    issue.get("url", f"https://github.com/{self.repo_name}/issues/{issue['number']}"),
            })
            added += 1

        if added:
            self._save()
        return added

    def search(self, query: str, k: int = 3) -> list[dict]:
        """Return the k most semantically similar issues to the query string."""
        if not EMBEDDINGS_READY or self.index is None or self.index.ntotal == 0:
            return []

        vec = embed(query)
        if vec is None:
            return []

        k = min(k, self.index.ntotal)
        scores, indices = self.index.search(vec.reshape(1, -1), k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            meta    = self.metadata[int(idx)]
            score_f = float(score)
            results.append({
                "issue_number":        meta["number"],
                "title":               meta["title"],
                "similarity_score":    round(score_f, 4),
                "url":                 meta["url"],
                "is_likely_duplicate": score_f > DUPLICATE_THRESHOLD,
            })
        return results

    @property
    def size(self) -> int:
        return len(self.metadata)
