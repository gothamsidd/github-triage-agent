# GitHub Issue Triage Agent

An autonomous Python agent that triages GitHub issues end-to-end — detecting duplicates, identifying the affected component, scoring severity, posting a structured report, and applying labels. Runs fully without API keys using mock data and keyword heuristics.

```
New issue opened
      │
      ▼
find_duplicate_issues ──► similarity > 0.75 ? ──► post duplicate notice + stop
      │
      ▼
search_codebase       (which source files are involved?)
      │
      ▼
score_severity        (critical / high / medium / low)
      │
      ▼
post_triage_comment   (structured Markdown report on the issue)
      │
      ▼
apply_labels          (bug, critical, needs-info, …)
```

---

## Quick start

```bash
git clone <this-repo> && cd github-triage-agent
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Triage all 5 mock issues (no API keys required)
python main.py

# Triage a single issue
python main.py --issue 1003

# Launch the web dashboard
python server.py        # → http://localhost:8080
```

---

## Configuration

Copy `.env.example` to `.env` and fill in your keys. Every key is optional — see the table below.

| Keys set | Duplicate detection | Severity scoring | Output |
|---|---|---|---|
| None | TF-IDF keyword match | Keyword heuristics | Printed to stdout |
| `GITHUB_TOKEN` only | TF-IDF | Heuristics | Posted to GitHub |
| `VOYAGE_API_KEY` + `GROQ_API_KEY` | Semantic (FAISS) — **free** | Llama-3.3-70b — **free** | Printed / posted |
| `OPENAI_API_KEY` | Semantic (FAISS) | GPT-4o-mini | Printed / posted |

**Free full-power setup (recommended):**
- `VOYAGE_API_KEY` → [dash.voyageai.com](https://dash.voyageai.com) — 200M embedding tokens/month free
- `GROQ_API_KEY` → [console.groq.com](https://console.groq.com) — 14k requests/day free

---

## Features

- **Semantic duplicate detection** — Voyage AI / OpenAI embeddings + FAISS. Catches duplicates that share no vocabulary ("app crashes on login" ≈ "segfault when authenticating"). Falls back to TF-IDF when no key is set.
- **LLM severity scoring** — Groq Llama-3.3-70b or GPT-4o-mini. Falls back to keyword heuristics.
- **Codebase search** — ripgrep / grep on a local repo clone. Mock results when no clone exists.
- **Human approval queue** — enable "Stage for review" in the dashboard to queue reports for human sign-off before posting to GitHub.
- **Real-time web dashboard** — Flask SSE streams every log line to the browser as it happens.
- **LangSmith tracing** — set `LANGCHAIN_TRACING_V2=true` to record every agent run.
- **GitHub Actions** — runs on `issues: opened` and hourly schedule.
- **Dry-run mode** — `--dry-run` flag previews everything without touching GitHub.
- **Parallel workers** — `--workers N` triages N issues simultaneously.

---

## Architecture decisions

### Dual operating modes
The LangGraph ReAct agent (when `GROQ_API_KEY` or `OPENAI_API_KEY` is set) lets the LLM decide tool order and handle edge cases. The manual fallback (`triage_manually`) hard-codes the same 5-step workflow deterministically and works with zero API keys. Both call the same 6 tools and produce identical output.

### Semantic duplicate detection (embeddings.py)
TF-IDF matches words. Embeddings match meaning. Provider priority: **Voyage AI → OpenAI → TF-IDF**.

Issue texts are unit-normalized before insertion into a FAISS `IndexFlatIP` index, making inner-product search equivalent to cosine similarity without extra computation. The index is persisted to `.faiss_cache/` keyed by repo + model name — issues are never re-embedded across restarts, and switching providers automatically creates a fresh index with the correct dimensions.

Entire issue batches are embedded in a **single API call** (`embed_batch`) rather than one call per issue — reduces cold-start from ~15s to ~2s.

Duplicate threshold: **0.85** (tighter than TF-IDF's 0.75 because embedding similarity is more precise).

### Severity scoring (tools.py)
LLM call with `response_format={"type": "json_object"}` guarantees structured output. Provider priority: **Groq → OpenAI → keyword heuristics**. Groq's `llama-3.3-70b-versatile` is preferred — free tier, lower latency than OpenAI.

### Steps 2 + 3 run in parallel
`search_codebase` and `score_severity` are independent — both fire simultaneously in a 2-worker thread pool. Total time = `max(t_search, t_score)` instead of `t_search + t_score`.

### Session cache for GitHub issues
`find_duplicate_issues` fetches existing issues from GitHub once per session and caches them in memory. Triaging 10 issues costs 1 API round-trip instead of 10.

### Web dashboard (server.py)
`builtins.print` is patched with a thread-local router before any project imports. Each triage worker thread sets `threading.local().sse_queue = q`; the patched print routes output to that queue instead of stdout. The Flask route streams queue items to the browser as Server-Sent Events. SSE is used over WebSockets — unidirectional, no extra library, auto-reconnects.

### Human approval queue
When `TRIAGE_STAGE_MODE=1`, `post_triage_comment` writes the formatted report to `_staged_reports` (an in-process dict) instead of posting to GitHub. `apply_labels` attaches the label list to the same entry. The `/api/pending` endpoint exposes the queue; `/api/approve` posts the (optionally edited) comment and applies labels atomically; `/api/reject` discards the entry.

### GitHub Actions
Triggers on `issues: opened` (real-time) and `schedule: '0 * * * *'` (hourly sweep for issues missed during downtime). Manual trigger accepts an optional `issue_number` for debugging.

---

## File structure

```
.
├── main.py               # Agent orchestration + manual ReAct fallback
├── tools.py              # 6 LangChain @tool functions
├── embeddings.py         # Voyage AI / OpenAI + FAISS semantic index
├── server.py             # Flask SSE dashboard + approval queue
├── templates/
│   └── index.html        # Dark GitHub-themed single-page dashboard
├── tests/
│   └── test_tools.py     # 38 pytest unit tests
├── mock_issues.json      # 5 test issues covering all triage scenarios
├── .github/
│   └── workflows/
│       └── triage.yml    # GitHub Actions workflow
├── requirements.txt
└── .env.example          # All supported env vars with descriptions
```

---

## Mock issues

| # | Title | Tests |
|---|---|---|
| 1001 | Login fails with OAuth tokens on mobile | Duplicate detection |
| 1002 | App crashes when I use it | `needs-info` — vague, no stack trace |
| 1003 | CRITICAL: Silent data loss for files > 100MB | Critical severity, storage component |
| 1004 | Feature request: dark mode | Low severity, feature label |
| 1005 | Typo in README: 'recieve' | Documentation label |

---

## CLI reference

```
python main.py [--repo owner/repo] [--issue N] [--dry-run] [--workers N]

  --repo         GitHub repository (default: psf/requests)
  --local-repo   Path to local clone for codebase search
  --issue        Triage one mock issue by number
  --dry-run      Preview only — never post comments or apply labels
  --workers      Triage N issues in parallel (default: 1)
```

---

## Running tests

```bash
pytest tests/ -v
```
