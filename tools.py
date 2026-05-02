"""
tools.py — All 6 triage tools used by the GitHub Issue Triage Agent.

Each tool gracefully degrades to mock/heuristic mode when API credentials
are absent, so the agent can be demoed without a real GitHub token.
"""

import os
import json
import subprocess
from pathlib import Path
from typing import Any

# Staging area: post_triage_comment writes here instead of GitHub when
# TRIAGE_STAGE_MODE=1. server.py reads from this dict after triage completes.
_staged_reports: dict = {}  # key: (repo_name, issue_number)

# Session-level cache for existing issues used by find_duplicate_issues.
# Avoids re-fetching the same GitHub pages for every issue in a triage session.
# Resets on server restart so it never goes stale across deployments.
_issue_cache: dict[str, list] = {}  # key: repo_name

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Use LangChain's @tool decorator when available; fall back to a no-op wrapper
# so tools.py works even without langchain_core installed.
try:
    from langchain_core.tools import tool
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    def tool(fn):  # type: ignore[misc]
        """Minimal passthrough when LangChain is not installed."""
        fn.invoke = lambda args: fn(**args)
        return fn

try:
    from github import Github, GithubException
    PYGITHUB_AVAILABLE = True
except ImportError:
    PYGITHUB_AVAILABLE = False
    print("[WARN] PyGithub not installed — will use mock data for all GitHub calls.")

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

# ---------------------------------------------------------------------------
# Client initialisation (conditional on environment variables)
# ---------------------------------------------------------------------------

GITHUB_TOKEN   = os.getenv("GITHUB_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY   = os.getenv("GROQ_API_KEY")

gh_client     = Github(GITHUB_TOKEN) if (GITHUB_TOKEN and PYGITHUB_AVAILABLE) else None
openai_client = OpenAI(api_key=OPENAI_API_KEY) if (OPENAI_API_KEY and OPENAI_AVAILABLE) else None
groq_client   = Groq(api_key=GROQ_API_KEY) if (GROQ_API_KEY and GROQ_AVAILABLE) else None

# Prefer Groq for chat completions (free); fall back to OpenAI
_chat_client  = groq_client or openai_client
_chat_model   = "llama-3.3-70b-versatile" if groq_client else "gpt-4o-mini"

MOCK_DATA_PATH = Path(__file__).parent / "mock_issues.json"

# ---------------------------------------------------------------------------
# Static mock data used when GITHUB_TOKEN is absent
# ---------------------------------------------------------------------------

# Simulates the "already-triaged" history that duplicate detection searches through.
MOCK_EXISTING_ISSUES: list[dict] = [
    {
        "number": 142,
        "title": "Login fails with OAuth tokens",
        "body": (
            "When using OAuth tokens the login endpoint returns 500. "
            "Steps: 1. Set up OAuth 2. Call /auth/login. Error: Internal Server Error. "
            "KeyError access_token in authenticate()."
        ),
        "url": "https://github.com/psf/requests/issues/142",
    },
    {
        "number": 89,
        "title": "Request timeout not configurable",
        "body": "There is no way to set a custom timeout for HTTP requests. "
                "Would be nice to add a timeout parameter to the request function.",
        "url": "https://github.com/psf/requests/issues/89",
    },
    {
        "number": 201,
        "title": "Memory leak in session handling",
        "body": "The session object is not being properly garbage collected, "
                "causing memory usage to grow over time in long-running applications.",
        "url": "https://github.com/psf/requests/issues/201",
    },
    {
        "number": 55,
        "title": "Typo in README documentation",
        "body": "There is a typo in README.md: 'recieve' should be 'receive'. Minor fix.",
        "url": "https://github.com/psf/requests/issues/55",
    },
    {
        "number": 310,
        "title": "Add HTTP/2 support",
        "body": "Feature request: please add HTTP/2 support to improve performance for modern apps.",
        "url": "https://github.com/psf/requests/issues/310",
    },
    {
        "number": 405,
        "title": "Dark mode feature request for UI",
        "body": "Would love dark mode support in the dashboard for accessibility and eye comfort.",
        "url": "https://github.com/psf/requests/issues/405",
    },
    {
        "number": 512,
        "title": "Data loss when saving large files over 100MB",
        "body": (
            "Files larger than 100MB get silently truncated during upload. "
            "No error is raised. MemoryError in chunk_upload. Production impact."
        ),
        "url": "https://github.com/psf/requests/issues/512",
    },
]

# Label definitions: name -> (hex color, description)
STANDARD_LABELS: dict[str, dict] = {
    "bug":           {"color": "d73a4a", "description": "Something isn't working"},
    "feature":       {"color": "a2eeef", "description": "New feature or request"},
    "duplicate":     {"color": "cfd3d7", "description": "This issue already exists"},
    "needs-info":    {"color": "e4e669", "description": "More information is needed"},
    "critical":      {"color": "b60205", "description": "Critical severity"},
    "high":          {"color": "d93f0b", "description": "High severity"},
    "medium":        {"color": "fbca04", "description": "Medium severity"},
    "low":           {"color": "0075ca", "description": "Low severity"},
    "documentation": {"color": "0075ca", "description": "Documentation related"},
    "auth-module":   {"color": "e4e669", "description": "Related to authentication"},
    "storage-module":{"color": "e4e669", "description": "Related to storage/upload"},
    "ui":            {"color": "bfd4f2", "description": "Related to the UI"},
}


def load_mock_issues() -> list[dict]:
    """Load test issues from mock_issues.json."""
    with open(MOCK_DATA_PATH, "r") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Tool 1 — fetch_new_issues
# ---------------------------------------------------------------------------

@tool
def fetch_new_issues(repo_name: str) -> str:
    """
    Fetch open GitHub issues for triage.

    Strategy (in priority order):
      1. Unlabelled issues (truly awaiting triage) — up to FETCH_LIMIT
      2. If fewer than MIN_RESULTS found, backfill with recently-opened issues
         that have only auto-applied labels (bug, enhancement, etc.) so the
         dashboard is never nearly empty on well-maintained repos like prisma/prisma.

    Uses PyGithub when GITHUB_TOKEN is set; loads mock_issues.json otherwise.

    Args:
        repo_name: GitHub repository in "owner/repo" format, e.g. "psf/requests".

    Returns:
        JSON array where each element has:
        issue_number, title, body, created_at, author, labels.
    """
    FETCH_LIMIT = 200   # how many open issues to scan
    MAX_RETURN  = 20    # cap results shown in the dashboard
    MIN_RESULTS = 5     # backfill if fewer unlabelled found

    if gh_client:
        try:
            repo = gh_client.get_repo(repo_name)
            unlabelled = []
            recent     = []  # backfill pool: issues with ≤2 auto-labels

            scanned = 0
            for issue in repo.get_issues(state="open", sort="created", direction="desc"):
                if scanned >= FETCH_LIMIT:
                    break
                # GitHub's issues endpoint includes PRs — skip them.
                if issue.pull_request:
                    continue
                scanned += 1

                labels     = [lb.name for lb in issue.labels]
                issue_data = {
                    "issue_number": issue.number,
                    "title":        issue.title,
                    "body":         issue.body or "",
                    "created_at":   issue.created_at.isoformat(),
                    "author":       issue.user.login,
                    "labels":       labels,
                }

                if len(labels) == 0:
                    unlabelled.append(issue_data)
                elif len(labels) <= 2:
                    # Light auto-labelling (e.g. "bug") still benefits from triage.
                    recent.append(issue_data)

                if len(unlabelled) >= MAX_RETURN:
                    break

            # Backfill so dashboard is never nearly empty
            result = unlabelled
            if len(result) < MIN_RESULTS:
                needed = MIN_RESULTS - len(result)
                result = result + recent[:needed]

            return json.dumps(result[:MAX_RETURN], indent=2)

        except GithubException as exc:
            return json.dumps({"error": f"GitHub API error {exc.status}: {exc.data}"})

    # Mock mode
    print("[MOCK] Loading issues from mock_issues.json")
    return json.dumps(load_mock_issues(), indent=2)


# ---------------------------------------------------------------------------
# Tool 2 — find_duplicate_issues
# ---------------------------------------------------------------------------

@tool
def find_duplicate_issues(issue_text: str, repo_name: str, issue_number: int = 0) -> str:
    """
    Detect duplicate issues using semantic embeddings (OpenAI + FAISS) when
    available, falling back to TF-IDF cosine similarity otherwise.

    Embeddings catch semantic duplicates that share no words:
      "app crashes on login" ≈ "segfault when authenticating"  (score 0.91)
    TF-IDF would score these near zero.

    Threshold: score > 0.85 (embeddings) or > 0.75 (TF-IDF) = likely duplicate.

    Args:
        issue_text:    Combined title + body of the new issue.
        repo_name:     GitHub repository in "owner/repo" format.
        issue_number:  The number of the issue being triaged — excluded from
                       the corpus so an issue never matches itself.

    Returns:
        JSON array of up to 3 objects, each with:
        issue_number, title, similarity_score, url, is_likely_duplicate.
    """
    from embeddings import IssueIndex, EMBEDDINGS_READY

    existing: list[dict] = []

    if gh_client:
        if repo_name in _issue_cache:
            existing = _issue_cache[repo_name]
        else:
            try:
                repo = gh_client.get_repo(repo_name)
                for issue in repo.get_issues(state="open")[:20]:
                    if not issue.pull_request:
                        existing.append({
                            "number": issue.number,
                            "title":  issue.title,
                            "body":   issue.body or "",
                            "url":    issue.html_url,
                        })
                for issue in repo.get_issues(state="closed")[:20]:
                    if not issue.pull_request:
                        existing.append({
                            "number": issue.number,
                            "title":  issue.title,
                            "body":   issue.body or "",
                            "url":    issue.html_url,
                        })
                _issue_cache[repo_name] = existing
            except GithubException as exc:
                print(f"[WARN] GitHub API error ({exc.status}), falling back to mock history.")
                existing = MOCK_EXISTING_ISSUES
    else:
        print("[MOCK] Using mock issue history for duplicate detection.")
        existing = MOCK_EXISTING_ISSUES

    if not existing:
        return json.dumps([])

    # Never match an issue against itself.
    if issue_number:
        existing = [i for i in existing if i["number"] != issue_number]

    # ── Path A: Embeddings + FAISS (semantic search) ─────────────────
    if EMBEDDINGS_READY:
        print(f"  🔍 Using semantic embeddings (text-embedding-3-small + FAISS)")
        index = IssueIndex(repo_name)
        newly_indexed = index.add(existing)
        if newly_indexed:
            print(f"  📥 Indexed {newly_indexed} new issues into FAISS")
        results = index.search(issue_text, k=3)
        if results:
            return json.dumps(results, indent=2)
        # Empty results = new index with no vectors yet — fall through to TF-IDF

    # ── Path B: TF-IDF (keyword matching fallback) ───────────────────
    print(f"  🔍 Using TF-IDF similarity (install faiss-cpu + OPENAI_API_KEY for semantic search)")
    corpus = [f"{i['title']} {i.get('body', '')}" for i in existing]
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    try:
        matrix = vectorizer.fit_transform([issue_text] + corpus)
    except ValueError:
        return json.dumps([])

    scores = cosine_similarity(matrix[0], matrix[1:])[0]
    top3   = np.argsort(scores)[::-1][:3]

    results = []
    for idx in top3:
        item  = existing[idx]
        score = float(scores[idx])
        results.append({
            "issue_number":       item["number"],
            "title":              item["title"],
            "similarity_score":   round(score, 4),
            "url":                item.get("url", f"https://github.com/{repo_name}/issues/{item['number']}"),
            "is_likely_duplicate": score > 0.75,
        })

    return json.dumps(results, indent=2)


# ---------------------------------------------------------------------------
# Tool 3 — search_codebase
# ---------------------------------------------------------------------------

@tool
def search_codebase(keywords: str, local_repo_path: str) -> str:
    """
    Search a local repository clone for keywords in file names, function/class
    definitions, and comments. Uses ripgrep (rg) when available, falls back
    to grep. Searches only Python files.

    If the path doesn't exist, returns plausible mock results so the agent
    can still produce a useful triage without a local clone.

    Args:
        keywords:        Space-separated search terms (e.g. "login oauth token").
        local_repo_path: Absolute path to the local clone of the repository.

    Returns:
        JSON array of up to 5 objects with: file (relative path), matches (lines).
    """
    repo_path = Path(local_repo_path)

    if not repo_path.exists() or not repo_path.is_dir():
        print(f"[MOCK] Local repo not found at '{local_repo_path}'. Using mock search results.")
        return _mock_search(keywords)

    hits: dict[str, list[str]] = {}  # relative_file -> [matching lines]

    for keyword in keywords.strip().split():
        lines = _run_search(keyword, repo_path)
        for raw_line in lines:
            if not raw_line:
                continue
            parts = raw_line.split(":", 2)
            if len(parts) < 2:
                continue
            rel_file = parts[0].replace(str(repo_path), "").lstrip("/\\")
            match_text = ":".join(parts[1:]).strip()
            hits.setdefault(rel_file, [])
            if len(hits[rel_file]) < 3:
                hits[rel_file].append(match_text)

    # Rank by total number of keyword hits (more hits = more relevant).
    ranked = sorted(hits.items(), key=lambda kv: len(kv[1]), reverse=True)[:5]
    output = [{"file": f, "matches": m} for f, m in ranked]

    if not output:
        output = [{"file": "No matches found", "matches": [f"No Python files match: {keywords}"]}]

    return json.dumps(output, indent=2)


def _run_search(keyword: str, repo_path: Path) -> list[str]:
    """Try ripgrep, fall back to grep, return raw output lines."""
    try:
        result = subprocess.run(
            ["rg", "--type", "py", "-i", "-n", "--max-count", "5", keyword, str(repo_path)],
            capture_output=True, text=True, timeout=10,
        )
        return result.stdout.strip().splitlines()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    try:
        result = subprocess.run(
            ["grep", "-r", "-i", "-n", "--include=*.py", keyword, str(repo_path)],
            capture_output=True, text=True, timeout=10,
        )
        return result.stdout.strip().splitlines()[:20]
    except subprocess.TimeoutExpired:
        return []


def _mock_search(keywords: str) -> str:
    """Return plausible mock search results based on common keyword patterns."""
    kw = keywords.lower()

    # Keys sorted by length (longest first) so multi-word phrases like
    # "data loss" are tested before their constituent words ("data", "loss"),
    # preventing the generic data model from winning over storage/upload.
    file_map = {
        "data loss":    ("src/storage/upload.py", ["line 89: def chunk_upload(file, size):", "line 134: def validate_file_size(f):"]),
        "file upload":  ("src/storage/upload.py", ["line 89: def chunk_upload(file, size):", "line 201: class StorageBackend:"]),
        "memory error": ("src/storage/upload.py", ["line 89: def chunk_upload(file, size):", "line 33: def allocate_buffer(size):"]),
        "dark mode":    ("src/ui/theme.py",        ["line 5: THEMES = ['light', 'dark']", "line 23: def apply_theme(name):"]),
        "oauth token":  ("src/auth/oauth.py",      ["line 34: def validate_token(access_token):", "line 12: class OAuthProvider:"]),
        "upload":   ("src/storage/upload.py", ["line 89: def chunk_upload(file, size):", "line 134: def validate_file_size(f):"]),
        "storage":  ("src/storage/upload.py", ["line 89: def chunk_upload(file, size):", "line 201: class StorageBackend:"]),
        "truncat":  ("src/storage/upload.py", ["line 89: def chunk_upload(file, size):", "line 134: def validate_file_size(f):"]),
        "silent":   ("src/storage/upload.py", ["line 89: def chunk_upload(file, size):", "line 67: # silent failure path"]),
        "auth":     ("src/auth/login.py",     ["line 42: def authenticate(token):", "line 89: class OAuthHandler:"]),
        "login":    ("src/auth/login.py",     ["line 42: def authenticate(token):", "line 156: def login_user(request):"]),
        "oauth":    ("src/auth/oauth.py",     ["line 12: class OAuthProvider:", "line 67: def exchange_token(code):"]),
        "token":    ("src/auth/oauth.py",     ["line 34: def validate_token(access_token):", "line 78: # token expiry check"]),
        "session":  ("src/auth/session.py",   ["line 23: class SessionManager:", "line 91: def cleanup_expired_sessions():"]),
        "timeout":  ("src/http/client.py",    ["line 34: DEFAULT_TIMEOUT = 30", "line 78: def set_timeout(seconds):"]),
        "readme":   ("README.md",             ["line 42: # Contributing — receieve -> receive (typo)", "line 89: Documentation section"]),
        "typo":     ("README.md",             ["line 42: receieve should be receive"]),
        "memory":   ("src/core/memory.py",    ["line 33: def allocate_buffer(size):", "line 78: # GC pressure warning"]),
        "crash":    ("src/core/runner.py",    ["line 19: def run(self):", "line 104: except Exception as e: # unhandled"]),
        "dark":     ("src/ui/theme.py",       ["line 5: THEMES = ['light', 'dark']", "line 23: def apply_theme(name):"]),
        "ui":       ("src/ui/components.py",  ["line 12: class DashboardLayout:", "line 55: def render_settings_panel():"]),
        "data":     ("src/models/data.py",    ["line 15: class DataModel:", "line 78: def save(self, data):"]),
    }

    found: dict[str, list[str]] = {}

    # Iterate keys longest-first so specific phrases win over generic tokens.
    for key in sorted(file_map, key=len, reverse=True):
        filepath, lines = file_map[key]
        if key in kw:
            found.setdefault(filepath, lines)

    if not found:
        found = {
            "src/core/utils.py":    ["line 1: # Core utilities module", "line 45: def process_request():"],
            "src/api/endpoints.py": ["line 23: # API endpoint definitions"],
        }

    result = [{"file": fp, "matches": ml} for fp, ml in list(found.items())[:5]]
    return json.dumps(result, indent=2)


# ---------------------------------------------------------------------------
# Tool 4 — score_severity
# ---------------------------------------------------------------------------

@tool
def score_severity(issue_title: str, issue_body: str, affected_files: str) -> str:
    """
    Score the severity of an issue as critical / high / medium / low and
    suggest relevant labels. Makes an LLM call when OPENAI_API_KEY is set;
    falls back to keyword heuristics otherwise.

    Args:
        issue_title:    The issue title.
        issue_body:     The issue description / body.
        affected_files: JSON string from search_codebase describing touched files.

    Returns:
        JSON object with: severity, reason, suggested_labels, missing_info.
    """
    combined = f"{issue_title}\n{issue_body}".lower()

    if _chat_client:
        prompt = f"""You are a GitHub issue triage assistant. Analyze the following issue and assign a severity level.

Issue Title: {issue_title}
Issue Body: {issue_body}
Affected Files: {affected_files}

Severity guidelines:
- critical : crashes, data loss, security vulnerabilities, production outages
- high     : significant functionality broken, no obvious workaround
- medium   : functionality impaired but a workaround exists
- low      : typos, docs, cosmetic, feature requests, minor improvements

Respond ONLY with a valid JSON object (no markdown fences) in this exact shape:
{{
  "severity": "critical|high|medium|low",
  "reason": "One sentence explaining the chosen severity.",
  "suggested_labels": ["label1", "label2"],
  "missing_info": "What is missing that would help diagnose this (empty string if nothing)."
}}"""

        try:
            response = _chat_client.chat.completions.create(
                model=_chat_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                response_format={"type": "json_object"},
            )
            return response.choices[0].message.content
        except Exception as exc:
            print(f"[WARN] LLM call failed ({exc}). Using heuristic fallback.")

    return _heuristic_score(issue_title, issue_body, combined)


def _heuristic_score(title: str, body: str, combined: str) -> str:
    """Keyword-based severity scoring — no LLM required."""
    word_count = len(body.split())
    has_trace = (
        "traceback" in combined
        or 'file "' in combined
        or "at line" in combined
        or "error:" in combined
    )

    critical_kw = ["crash", "data loss", "data corrupt", "security", "vulnerability",
                   "breach", "production down", "critical", "silent", "truncat"]
    high_kw     = ["fail", "broken", "memory leak", "exception", "500", "regression",
                   "authentication", "auth", "error", "allocat"]
    low_kw      = ["typo", "spelling", "documentation", "readme", "cosmetic",
                   "minor", "dark mode", "feature request", "add support", "would be nice"]

    if any(kw in combined for kw in critical_kw):
        severity = "critical"
        reason   = "Issue contains keywords indicating a crash, data loss, or security risk."
        labels   = ["bug", "critical"]
    elif any(kw in combined for kw in high_kw):
        severity = "high"
        reason   = "Issue describes a significant functional failure or error condition."
        labels   = ["bug", "high"]
    elif any(kw in combined for kw in low_kw):
        severity = "low"
        if "typo" in combined or "readme" in combined or "spelling" in combined:
            labels = ["documentation", "low"]
            reason = "Issue is a minor documentation fix (typo or wording)."
        else:
            labels = ["feature", "low"]
            reason = "Issue is a non-critical feature request or cosmetic improvement."
    else:
        severity = "medium"
        reason   = "Issue describes a functional problem without clear severity indicators."
        labels   = ["bug", "medium"]

    missing_info = ""
    if word_count < 50 and not has_trace:
        missing_info = (
            "Issue body is very short and contains no stack trace. "
            "Please provide: Python version, OS, steps to reproduce, and full error output."
        )
    elif not has_trace and severity in ("critical", "high"):
        missing_info = "A stack trace would help maintainers reproduce and fix this faster."

    return json.dumps({
        "severity": severity,
        "reason": reason,
        "suggested_labels": labels,
        "missing_info": missing_info,
    }, indent=2)


# ---------------------------------------------------------------------------
# Tool 5 — post_triage_comment
# ---------------------------------------------------------------------------

@tool
def post_triage_comment(issue_number: int, repo_name: str, triage_report: str) -> str:
    """
    Post a structured triage report as a comment on a GitHub issue.

    Uses PyGithub when GITHUB_TOKEN is set; prints a preview to stdout otherwise.

    Args:
        issue_number:  The GitHub issue number.
        repo_name:     GitHub repository in "owner/repo" format.
        triage_report: JSON string produced by score_severity (may be enriched
                       by the agent with duplicate/component information).

    Returns:
        JSON object with success flag and a message.
    """
    try:
        report: dict[str, Any] = (
            json.loads(triage_report) if isinstance(triage_report, str) else triage_report
        )
    except json.JSONDecodeError:
        report = {"raw": triage_report}

    comment_body = _format_comment(report)
    dry_run    = bool(os.getenv("TRIAGE_DRY_RUN"))
    stage_mode = bool(os.getenv("TRIAGE_STAGE_MODE"))

    if stage_mode:
        _staged_reports[(repo_name, issue_number)] = {
            "repo":             repo_name,
            "issue_number":     issue_number,
            "comment":          comment_body,
            "report":           report,
            "suggested_labels": report.get("suggested_labels", []),
        }
        print(f"[STAGED] Report queued for human approval — #{issue_number}")
        return json.dumps({"success": True, "staged": True, "issue_number": issue_number})

    if gh_client and not dry_run:
        try:
            repo   = gh_client.get_repo(repo_name)
            issue  = repo.get_issue(issue_number)
            issue.create_comment(comment_body)
            return json.dumps({"success": True, "message": f"Comment posted on #{issue_number}."})
        except GithubException as exc:
            return json.dumps({"success": False, "error": f"GitHub API {exc.status}: {exc.data}"})

    # Dry-run or mock mode — print preview, touch nothing.
    tag = "[DRY-RUN]" if (gh_client and dry_run) else "[MOCK]"
    border = "=" * 62
    print(f"\n{border}")
    print(f"{tag} Comment preview for {repo_name}#{issue_number}:")
    print(border)
    print(comment_body)
    print(border)
    return json.dumps({
        "success": True,
        "message": f"{tag} Comment would be posted on #{issue_number}.",
        "preview": comment_body,
    })


def _format_comment(report: dict) -> str:
    """Render a triage dict into a human-readable GitHub Markdown comment."""
    if "raw" in report:
        return f"## Agent Triage Report\n\n{report['raw']}"

    SEV_EMOJI = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}

    lines = [
        "## Agent Triage Report",
        "_Automated analysis by the GitHub Issue Triage Agent._",
        "",
    ]

    if "severity" in report:
        sev   = report["severity"]
        emoji = SEV_EMOJI.get(sev, "⚪")
        lines.append(f"**Severity**: {emoji} {sev.capitalize()}")

    if report.get("reason"):
        lines.append(f"**Reason**: {report['reason']}")

    if report.get("likely_component"):
        lines.append(f"**Likely component**: `{report['likely_component']}`")

    if report.get("duplicates"):
        top = report["duplicates"][0]
        score_pct = f"{top['similarity_score']:.0%}"
        lines.append(
            f"**Possible duplicate of**: "
            f"[{top['title']} (#{top['issue_number']})]({top['url']}) "
            f"— similarity {score_pct}"
        )

    if report.get("missing_info"):
        lines.append(f"**Missing info**: {report['missing_info']}")

    if report.get("suggested_labels"):
        badge_str = ", ".join(f"`{lb}`" for lb in report["suggested_labels"])
        lines.append(f"**Suggested labels**: {badge_str}")

    lines += [
        "",
        "---",
        "_This report was generated automatically. A human maintainer will review shortly._",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool 6 — apply_labels
# ---------------------------------------------------------------------------

@tool
def apply_labels(issue_number: int, repo_name: str, labels: str) -> str:
    """
    Apply labels to a GitHub issue, creating any that do not exist yet.

    Uses PyGithub when GITHUB_TOKEN is set; prints what would happen otherwise.

    Args:
        issue_number: The GitHub issue number.
        repo_name:    GitHub repository in "owner/repo" format.
        labels:       JSON array of label name strings, e.g. '["bug", "high"]'.

    Returns:
        JSON object with success flag and the list of applied labels.
    """
    # Accept a JSON array string, a comma-separated string, or a plain single label.
    try:
        label_list: list[str] = json.loads(labels) if isinstance(labels, str) else labels
        if isinstance(label_list, str):
            label_list = [lb.strip() for lb in label_list.split(",")]
    except (json.JSONDecodeError, AttributeError):
        # Not JSON — treat as comma-separated plain text.
        label_list = [lb.strip() for lb in labels.split(",")] if labels else []

    if not label_list:
        return json.dumps({"success": False, "error": "No labels provided."})

    dry_run    = bool(os.getenv("TRIAGE_DRY_RUN"))
    stage_mode = bool(os.getenv("TRIAGE_STAGE_MODE"))

    if stage_mode:
        # Attach labels to any pending staged report for this issue.
        key = (repo_name, issue_number)
        if key in _staged_reports:
            _staged_reports[key]["suggested_labels"] = label_list
        print(f"[STAGED] Labels {label_list} queued for #{issue_number} — pending approval")
        return json.dumps({"success": True, "staged": True, "applied_labels": label_list})

    if gh_client and not dry_run:
        try:
            repo  = gh_client.get_repo(repo_name)
            issue = repo.get_issue(issue_number)

            existing_names = {lb.name for lb in repo.get_labels()}
            for name in label_list:
                if name not in existing_names:
                    cfg = STANDARD_LABELS.get(name, {"color": "ededed", "description": ""})
                    repo.create_label(name=name, color=cfg["color"], description=cfg["description"])

            issue.set_labels(*label_list)
            return json.dumps({"success": True, "applied_labels": label_list, "issue": issue_number})

        except GithubException as exc:
            return json.dumps({"success": False, "error": f"GitHub API {exc.status}: {exc.data}"})

    tag = "[DRY-RUN]" if (gh_client and dry_run) else "[MOCK]"
    print(f"{tag} Would apply labels {label_list} to {repo_name}#{issue_number}")
    return json.dumps({
        "success": True,
        "message": f"{tag} Labels {label_list} would be applied to #{issue_number}.",
        "applied_labels": label_list,
    })
