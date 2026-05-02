"""
main.py — GitHub Issue Triage Agent

Two operating modes:
  1. LangChain + OpenAI agent (when OPENAI_API_KEY is set): the LLM decides
     which tools to call and in what order. A system prompt enforces the
     required workflow and duplicate-short-circuit rule.
  2. Manual orchestration (no OPENAI_API_KEY): a hard-coded ReAct-style loop
     that follows the same workflow deterministically.

Either mode works without GITHUB_TOKEN — it will use mock_issues.json and
print comment/label previews to stdout instead of hitting the real API.

Usage:
    python main.py                         # triage all unlabelled issues
    python main.py --repo psf/requests     # specify repo (real or mock mode)
    python main.py --issue 1003            # triage one mock issue by number
"""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv

load_dotenv()  # reads .env if present

GITHUB_TOKEN   = os.getenv("GITHUB_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY   = os.getenv("GROQ_API_KEY")

# ── LangSmith Tracing ───────────────────────────────────────────────────────
if os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true":
    os.environ.setdefault("LANGCHAIN_PROJECT", "github-triage-agent")
    print("[INFO] LangSmith tracing enabled — runs visible at smith.langchain.com")

# Use LangChain agent when either Groq (free) or OpenAI key is available.
# Groq is preferred — faster and free.
USE_LANGCHAIN = bool(GROQ_API_KEY or OPENAI_API_KEY)

if USE_LANGCHAIN:
    try:
        from langgraph.prebuilt import create_react_agent
        if GROQ_API_KEY:
            from langchain_groq import ChatGroq
        else:
            from langchain_openai import ChatOpenAI
    except ImportError:
        print("[WARN] langchain-groq / langchain-openai / langgraph not installed. Falling back to manual mode.")
        USE_LANGCHAIN = False

from tools import (
    fetch_new_issues,
    find_duplicate_issues,
    load_mock_issues,
    search_codebase,
    score_severity,
    post_triage_comment,
    apply_labels,
)

# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------

DEFAULT_REPO       = "psf/requests"
DEFAULT_LOCAL_REPO = os.getenv("LOCAL_REPO_PATH", "/tmp/requests")  # set to your local clone

# ---------------------------------------------------------------------------
# LangChain agent
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert GitHub issue triage agent. Your sole responsibility is to
analyse a single GitHub issue and produce a thorough, accurate triage report.

You have access to these tools (always use them in this order):
  1. find_duplicate_issues  — check whether the issue is a near-duplicate
  2. search_codebase        — locate which source files are relevant
  3. score_severity         — get a severity rating + label suggestions
  4. post_triage_comment    — publish the structured report as a GitHub comment
  5. apply_labels           — apply the chosen labels to the issue

Important rules:
- If find_duplicate_issues returns any issue with similarity_score > 0.75:
    • Do NOT run the full triage workflow.
    • Call post_triage_comment with a short "possible duplicate" message.
    • Call apply_labels with ["duplicate"] and stop.
- If the issue body is fewer than 50 words AND contains no stack trace,
  always include a request for more information in the triage_report JSON
  you pass to post_triage_comment.
- Pass triage_report as a JSON string that includes at minimum:
    severity, reason, likely_component, duplicates (list), missing_info,
    suggested_labels.
- Be concise. Maintainers are busy.
"""


def build_agent():
    """Construct a LangGraph ReAct agent — uses Groq if available, else OpenAI."""
    if GROQ_API_KEY:
        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, groq_api_key=GROQ_API_KEY)
    else:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=OPENAI_API_KEY)

    all_tools = [
        find_duplicate_issues,
        search_codebase,
        score_severity,
        post_triage_comment,
        apply_labels,
    ]

    # create_react_agent from langgraph replaces the deprecated
    # create_openai_tools_agent + AgentExecutor pattern.
    return create_react_agent(llm, all_tools, prompt=SYSTEM_PROMPT)


def _agent_task(issue: dict, repo_name: str, local_repo: str) -> str:
    """Format the per-issue task prompt for the LangChain agent."""
    return (
        f"Triage this GitHub issue from repository '{repo_name}'.\n\n"
        f"Issue #{issue['issue_number']}\n"
        f"Title: {issue['title']}\n"
        f"Author: {issue['author']}\n"
        f"Created: {issue['created_at']}\n"
        f"Body:\n{issue['body']}\n\n"
        f"Local repository path for codebase search: {local_repo}\n\n"
        f"Complete the full triage workflow now."
    )


# ---------------------------------------------------------------------------
# Manual orchestration (no-LLM path)
# ---------------------------------------------------------------------------

_STOP_WORDS = frozenset(
    "a an the in on at is it to for of and or with when issue bug error "
    "please help fix this".split()
)


def _extract_keywords(title: str) -> str:
    """
    Extract search terms from an issue title.
    Bigrams (adjacent word pairs) are prepended so that specific phrases like
    "data loss" match storage files before the generic "data" token does.
    """
    words = [w for w in title.lower().split() if w not in _STOP_WORDS and len(w) > 2]
    bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]
    return " ".join(bigrams + words) or title


def triage_manually(issue: dict, repo_name: str, local_repo: str) -> None:
    """
    Hard-coded triage workflow executed when no OpenAI key is available.
    Mirrors the agent's intended tool call sequence exactly.
    """
    num   = issue["issue_number"]
    title = issue["title"]
    body  = issue.get("body", "")

    print(f"\n{'='*62}")
    print(f"  Triaging Issue #{num}: {title}")
    print(f"{'='*62}")

    issue_text = f"{title} {body}"
    word_count = len(body.split())
    has_trace  = (
        "traceback" in body.lower()
        or 'file "' in body.lower()
        or "error:" in body.lower()
    )

    # --- Step 1: duplicate check -------------------------------------------
    print("\n[1/5] Checking for duplicates…")
    raw_dups = find_duplicate_issues.invoke({
        "issue_text":   issue_text,
        "repo_name":    repo_name,
        "issue_number": num,
    })
    duplicates: list[dict] = json.loads(raw_dups)
    top = duplicates[0] if duplicates else None

    if top and top["similarity_score"] > 0.75:
        pct = f"{top['similarity_score']:.0%}"
        print(f"  ⚠  Likely duplicate of #{top['issue_number']} ({pct} similar).")

        dup_report = json.dumps({
            "severity": "low",
            "reason": f"Appears to be a duplicate of #{top['issue_number']}.",
            "duplicates": [top],
            "missing_info": "",
            "suggested_labels": ["duplicate"],
        })

        print("\n[2/5] Posting duplicate notice…")
        post_triage_comment.invoke({
            "issue_number": num,
            "repo_name": repo_name,
            "triage_report": dup_report,
        })

        print("\n[3/5] Applying 'duplicate' label…")
        apply_labels.invoke({
            "issue_number": num,
            "repo_name": repo_name,
            "labels": json.dumps(["duplicate"]),
        })

        print(f"\n✅  Done (duplicate flagged).\n")
        return

    # --- Steps 2 + 3: run codebase search and severity scoring in parallel --
    print("\n[2/5] Searching codebase + scoring severity in parallel…")
    keywords = _extract_keywords(title)

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_files = pool.submit(search_codebase.invoke, {
            "keywords": keywords,
            "local_repo_path": local_repo,
        })
        # Score severity with empty files first; we'll refine below if needed.
        # (severity rarely changes based on files — title+body carry the signal)
        fut_sev = pool.submit(score_severity.invoke, {
            "issue_title": title,
            "issue_body": body,
            "affected_files": "[]",
        })
        raw_files = fut_files.result()
        raw_sev   = fut_sev.result()

    files: list[dict] = json.loads(raw_files)
    likely_component = files[0]["file"] if files else "unknown"
    print(f"  📁 Likely component: {likely_component}")

    severity_data: dict = json.loads(raw_sev)
    sev = severity_data["severity"]
    print(f"  🎯 Severity: {sev.upper()} — {severity_data['reason']}")

    # --- Step 4: build report and post comment -----------------------------
    print("\n[4/5] Posting triage report…")

    missing_info = severity_data.get("missing_info", "")
    # Always ask for more info when the body is too short or lacks a trace.
    if word_count < 50 and not has_trace and not missing_info:
        missing_info = (
            "Issue body is too short (< 50 words) and has no stack trace. "
            "Please provide: Python version, OS, steps to reproduce, and full error output."
        )

    triage_report = json.dumps({
        "severity": sev,
        "reason": severity_data["reason"],
        "likely_component": likely_component,
        # Show potential (non-duplicate) near-matches as context for maintainers.
        "duplicates": [d for d in duplicates if 0.3 < d["similarity_score"] <= 0.75],
        "missing_info": missing_info,
        "suggested_labels": severity_data.get("suggested_labels", ["medium"]),
    })

    post_triage_comment.invoke({
        "issue_number": num,
        "repo_name": repo_name,
        "triage_report": triage_report,
    })

    # --- Step 5: apply labels ----------------------------------------------
    print("\n[5/5] Applying labels…")
    final_labels: list[str] = severity_data.get("suggested_labels", ["medium"])
    if word_count < 50 and not has_trace and "needs-info" not in final_labels:
        final_labels.append("needs-info")

    apply_labels.invoke({
        "issue_number": num,
        "repo_name": repo_name,
        "labels": json.dumps(final_labels),
    })

    print(f"\n✅  Triage complete for issue #{num}.\n")


# ---------------------------------------------------------------------------
# Concurrent triage helpers
# ---------------------------------------------------------------------------

def _agent_output(result: dict) -> str:
    """Extract the final text reply from a LangGraph agent result."""
    messages = result.get("messages", [])
    return messages[-1].content if messages else "(no output)"


def _triage_one(issue: dict, repo_name: str, local_repo: str, agent) -> None:
    """Triage a single issue — safe to call from a thread pool worker."""
    if agent:
        print(f"\n{'='*62}")
        print(f"  Triaging Issue #{issue['issue_number']}: {issue['title']}")
        print(f"{'='*62}")
        try:
            result = agent.invoke({"messages": [("human", _agent_task(issue, repo_name, local_repo))]})
            print(f"\nAgent summary: {_agent_output(result)}")
        except Exception as exc:
            print(f"Agent error for #{issue['issue_number']}: {exc}. Falling back to manual triage.")
            triage_manually(issue, repo_name, local_repo)
    else:
        triage_manually(issue, repo_name, local_repo)


def _run_concurrent(issues: list, repo_name: str, local_repo: str, agent, workers: int) -> None:
    """Triage up to `workers` issues simultaneously using a thread pool."""
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_triage_one, issue, repo_name, local_repo, agent): issue["issue_number"]
            for issue in issues
        }
        for future in as_completed(futures):
            issue_num = futures[future]
            try:
                future.result()
            except Exception as exc:
                print(f"[ERROR] Issue #{issue_num} failed: {exc}")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(
    repo_name: str = DEFAULT_REPO,
    local_repo: str = DEFAULT_LOCAL_REPO,
    workers: int = 1,
    dry_run: bool = False,
) -> None:
    """Fetch all unlabelled issues and triage each one."""
    if dry_run:
        os.environ["TRIAGE_DRY_RUN"] = "1"

    mode_gh  = "Real GitHub API" if GITHUB_TOKEN else "Mock (no GITHUB_TOKEN)"
    mode_llm = "OpenAI gpt-4o-mini" if USE_LANGCHAIN else "Heuristic fallback (no OPENAI_API_KEY)"
    mode_dry = " [DRY-RUN — no writes]" if dry_run else ""

    print("\n🤖  GitHub Issue Triage Agent")
    print(f"    Repository : {repo_name}")
    print(f"    GitHub mode: {mode_gh}{mode_dry}")
    print(f"    LLM mode   : {mode_llm}")
    print(f"    Local repo : {local_repo}")
    print(f"    Workers    : {workers}")
    print()

    print("Fetching new unlabelled issues…")
    raw = fetch_new_issues.invoke({"repo_name": repo_name})

    try:
        issues = json.loads(raw)
    except json.JSONDecodeError:
        print(f"Could not parse issue list: {raw}")
        sys.exit(1)

    if isinstance(issues, dict) and "error" in issues:
        print(f"Error: {issues['error']}")
        sys.exit(1)

    if not issues:
        print("No unlabelled issues found — nothing to triage.")
        return

    print(f"Found {len(issues)} issue(s) to triage.\n")

    agent = build_agent() if USE_LANGCHAIN else None

    if workers > 1:
        print(f"Running with {workers} parallel workers…\n")
        _run_concurrent(issues, repo_name, local_repo, agent, workers)
    else:
        for issue in issues:
            _triage_one(issue, repo_name, local_repo, agent)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Autonomous GitHub Issue Triage Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Triage all mock issues (no tokens required):
  python main.py

  # Triage a single mock issue by number:
  python main.py --issue 1003

  # Triage real issues (requires GITHUB_TOKEN):
  python main.py --repo pallets/flask --local-repo /path/to/flask

  # Read from real GitHub but post nothing (safe for testing):
  python main.py --repo pallets/flask --dry-run

  # Triage 3 issues at a time in parallel:
  python main.py --workers 3
""",
    )
    p.add_argument("--repo",       default=DEFAULT_REPO,       help="GitHub repo (owner/repo)")
    p.add_argument("--local-repo", default=DEFAULT_LOCAL_REPO, help="Path to local repo clone")
    p.add_argument(
        "--issue", type=int, metavar="N",
        help="Triage a single issue by number (uses mock data when no GITHUB_TOKEN)",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print what would happen; never post comments or apply labels (safe with real tokens)",
    )
    p.add_argument(
        "--workers", type=int, default=1, metavar="N",
        help="Number of issues to triage in parallel (default: 1)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.dry_run:
        os.environ["TRIAGE_DRY_RUN"] = "1"

    if args.issue:
        # Single-issue mode: always pulls from mock_issues.json so specific
        # scenarios can be tested without touching the real GitHub API.
        mock_issues = load_mock_issues()
        target = next((i for i in mock_issues if i["issue_number"] == args.issue), None)
        if target is None:
            print(f"Issue #{args.issue} not found in mock_issues.json.")
            sys.exit(1)

        if USE_LANGCHAIN:
            agent = build_agent()
            result = agent.invoke({"messages": [("human", _agent_task(target, args.repo, args.local_repo))]})
            print(f"\nAgent summary: {_agent_output(result)}")
        else:
            triage_manually(target, args.repo, args.local_repo)
    else:
        run(
            repo_name=args.repo,
            local_repo=args.local_repo,
            workers=args.workers,
            dry_run=args.dry_run,
        )
