"""
server.py — Local web dashboard for the GitHub Issue Triage Agent.

Run:  python server.py
Open: http://localhost:5000

The server patches builtins.print with a thread-local router BEFORE importing
any project modules, so every print() call made by the triage tools is
automatically captured per-thread and streamed to the browser via SSE.
No changes to tools.py or main.py are required.
"""

import builtins
import json
import os
import queue
import threading

# ---------------------------------------------------------------------------
# Patch builtins.print FIRST — before any project imports fire their prints.
# Uses threading.local so concurrent triage requests never cross-contaminate.
# ---------------------------------------------------------------------------

_original_print = builtins.print
_thread_local   = threading.local()


def _smart_print(*args, sep=" ", end="\n", file=None, flush=False):
    q = getattr(_thread_local, "sse_queue", None)
    if q is not None and file is None:
        text = sep.join(str(a) for a in args) + end
        q.put(text)
    else:
        _original_print(*args, sep=sep, end=end, file=file, flush=flush)


builtins.print = _smart_print

# ---------------------------------------------------------------------------
# Now safe to import project modules
# ---------------------------------------------------------------------------

from dotenv import load_dotenv
load_dotenv()

from flask import Flask, Response, jsonify, render_template, request, stream_with_context

from main import DEFAULT_REPO, USE_LANGCHAIN, GROQ_API_KEY, triage_manually
from tools import fetch_new_issues, load_mock_issues, _staged_reports, gh_client

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
LOCAL_REPO   = os.getenv("LOCAL_REPO_PATH", "/tmp/requests")

app = Flask(__name__)

# ---------------------------------------------------------------------------
# In-memory pending approval queue
# Populated by triage workers when TRIAGE_STAGE_MODE=1.
# Key: (repo_name, issue_number)  Value: staged report dict
# ---------------------------------------------------------------------------
_pending: dict = _staged_reports  # same object — workers write directly here


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    if GROQ_API_KEY:
        llm_mode = "Groq Llama-3.3-70b"
    elif USE_LANGCHAIN:
        llm_mode = "OpenAI GPT-4o-mini"
    else:
        llm_mode = "Heuristics"

    return render_template(
        "index.html",
        default_repo=DEFAULT_REPO,
        github_mode="Real API" if GITHUB_TOKEN else "Mock",
        llm_mode=llm_mode,
    )


@app.route("/api/issues")
def get_issues():
    repo         = request.args.get("repo", DEFAULT_REPO)
    issue_number = request.args.get("issue_number", type=int)

    # Single-issue lookup — used when the user pastes a full GitHub URL
    if issue_number and gh_client:
        try:
            from github import GithubException as GHEx
            gh_repo = gh_client.get_repo(repo)
            issue   = gh_repo.get_issue(issue_number)
            data    = [{
                "issue_number": issue.number,
                "title":        issue.title,
                "body":         issue.body or "",
                "created_at":   issue.created_at.isoformat(),
                "author":       issue.user.login,
                "labels":       [lb.name for lb in issue.labels],
            }]
            return jsonify(data)
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

    raw = fetch_new_issues.invoke({"repo_name": repo})
    try:
        return jsonify(json.loads(raw))
    except json.JSONDecodeError:
        return jsonify({"error": raw}), 500


@app.route("/api/triage/stream/<int:issue_number>")
def triage_stream(issue_number):
    """SSE endpoint — streams triage log lines for a single issue."""
    repo       = request.args.get("repo", DEFAULT_REPO)
    dry_run    = request.args.get("dry_run", "false").lower() == "true"
    stage_mode = request.args.get("stage_mode", "false").lower() == "true"

    if dry_run:
        os.environ["TRIAGE_DRY_RUN"] = "1"
    else:
        os.environ.pop("TRIAGE_DRY_RUN", None)

    if stage_mode:
        os.environ["TRIAGE_STAGE_MODE"] = "1"
    else:
        os.environ.pop("TRIAGE_STAGE_MODE", None)

    q = queue.Queue()

    def worker():
        _thread_local.sse_queue = q
        try:
            # Always include mock issues as a fallback for the demo numbers
            mock = {i["issue_number"]: i for i in load_mock_issues()}

            if GITHUB_TOKEN:
                raw    = fetch_new_issues.invoke({"repo_name": repo})
                issues = json.loads(raw)
                issue  = next((i for i in issues if i.get("issue_number") == issue_number), None)
                # Fall back to mock if the real repo doesn't have this number
                if issue is None:
                    issue = mock.get(issue_number)
            else:
                issue = mock.get(issue_number)

            if issue is None:
                q.put(f"❌ Issue #{issue_number} not found.\n")
            else:
                triage_manually(issue, repo, LOCAL_REPO)
                # If staging mode was active, notify the browser that approval is waiting.
                key = (repo, issue_number)
                if key in _pending:
                    q.put(json.dumps({"__staged__": True, "issue_number": issue_number}))

        except Exception as exc:
            import traceback
            q.put(f"❌ Error: {exc}\n")
            q.put(traceback.format_exc())
        finally:
            _thread_local.sse_queue = None
            q.put(None)  # sentinel

    threading.Thread(target=worker, daemon=True).start()
    return _sse_response(q)


@app.route("/api/triage/all")
def triage_all_stream():
    """SSE endpoint — streams triage log for every unlabelled issue."""
    repo       = request.args.get("repo", DEFAULT_REPO)
    dry_run    = request.args.get("dry_run", "false").lower() == "true"
    stage_mode = request.args.get("stage_mode", "false").lower() == "true"

    if dry_run:
        os.environ["TRIAGE_DRY_RUN"] = "1"
    else:
        os.environ.pop("TRIAGE_DRY_RUN", None)

    if stage_mode:
        os.environ["TRIAGE_STAGE_MODE"] = "1"
    else:
        os.environ.pop("TRIAGE_STAGE_MODE", None)

    q = queue.Queue()

    def worker():
        _thread_local.sse_queue = q
        try:
            raw    = fetch_new_issues.invoke({"repo_name": repo})
            issues = json.loads(raw)
            if isinstance(issues, dict) and "error" in issues:
                q.put(f"❌ {issues['error']}\n")
                return
            for issue in issues:
                triage_manually(issue, repo, LOCAL_REPO)
                key = (repo, issue["issue_number"])
                if key in _pending:
                    q.put(json.dumps({"__staged__": True, "issue_number": issue["issue_number"]}))
        except Exception as exc:
            import traceback
            q.put(f"❌ Error: {exc}\n")
            q.put(traceback.format_exc())
        finally:
            _thread_local.sse_queue = None
            q.put(None)

    threading.Thread(target=worker, daemon=True).start()
    return _sse_response(q)


# ---------------------------------------------------------------------------
# Human Approval Queue endpoints
# ---------------------------------------------------------------------------

@app.route("/api/pending")
def get_pending():
    """Return all staged reports awaiting human approval."""
    items = [
        {
            "repo":             v["repo"],
            "issue_number":     v["issue_number"],
            "comment_preview":  v["comment"],
            "suggested_labels": v["suggested_labels"],
        }
        for v in _pending.values()
    ]
    return jsonify(items)


@app.route("/api/approve", methods=["POST"])
def approve():
    """
    Approve a staged report and post it to GitHub.

    Body JSON: { "repo": "owner/repo", "issue_number": 123,
                 "comment": "...(optional edited comment)..." }
    """
    data         = request.get_json(force=True)
    repo_name    = data.get("repo", DEFAULT_REPO)
    issue_number = int(data["issue_number"])
    key          = (repo_name, issue_number)

    entry = _pending.pop(key, None)
    if entry is None:
        return jsonify({"error": "No pending report for that issue."}), 404

    # Allow the browser to send an edited comment body.
    comment_body   = data.get("comment", entry["comment"])
    label_list     = entry["suggested_labels"]

    if gh_client:
        try:
            from github import GithubException as GHEx
            gh_repo = gh_client.get_repo(repo_name)
            issue   = gh_repo.get_issue(issue_number)
            issue.create_comment(comment_body)

            # Ensure labels exist then apply them.
            from tools import STANDARD_LABELS
            existing = {lb.name for lb in gh_repo.get_labels()}
            for name in label_list:
                if name not in existing:
                    cfg = STANDARD_LABELS.get(name, {"color": "ededed", "description": ""})
                    gh_repo.create_label(name=name, **cfg)
            issue.set_labels(*label_list)

            return jsonify({"success": True, "message": f"Comment posted and labels applied to #{issue_number}."})
        except Exception as exc:
            return jsonify({"success": False, "error": str(exc)}), 500

    return jsonify({"success": False, "error": "No GitHub token configured — cannot post."}), 400


@app.route("/api/reject", methods=["POST"])
def reject():
    """Discard a staged report without posting anything to GitHub."""
    data         = request.get_json(force=True)
    repo_name    = data.get("repo", DEFAULT_REPO)
    issue_number = int(data["issue_number"])
    key          = (repo_name, issue_number)

    if key in _pending:
        del _pending[key]
        return jsonify({"success": True, "message": f"Report for #{issue_number} discarded."})
    return jsonify({"error": "No pending report found."}), 404


# ---------------------------------------------------------------------------
# SSE helper
# ---------------------------------------------------------------------------

def _sse_response(q: queue.Queue) -> Response:
    """Consume a queue and emit each item as a Server-Sent Event."""
    def generate():
        while True:
            line = q.get()
            if line is None:
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
                break
            if not line.strip():
                continue
            # Internal sentinel emitted by the worker when staging mode is active.
            try:
                obj = json.loads(line)
                if isinstance(obj, dict) and obj.get("__staged__"):
                    yield f"data: {json.dumps({'type': 'staged', 'issue_number': obj['issue_number']})}\n\n"
                    continue
            except (json.JSONDecodeError, TypeError):
                pass
            yield f"data: {json.dumps({'type': 'log', 'text': line.rstrip()})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _original_print("🤖  Triage Agent Server")
    _original_print(f"    GitHub : {'Real API  — ' + 'comments/labels WILL be posted' if GITHUB_TOKEN else 'Mock mode — nothing will be posted'}")
    _original_print(f"    LLM    : {'OpenAI GPT-4o-mini' if USE_LANGCHAIN else 'Keyword heuristics (no OPENAI_API_KEY)'}")
    _original_print(f"    Open   : http://localhost:8080\n")
    port = int(os.getenv("PORT", 8080))
    app.run(debug=False, threaded=True, host="0.0.0.0", port=port)
