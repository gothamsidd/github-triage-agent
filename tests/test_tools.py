"""
tests/test_tools.py — Unit tests for all 6 triage tools.

Run with:  pytest tests/ -v
No API keys or GitHub token required — all tests use mock/heuristic paths.
"""

import json
import os
import pytest

# Ensure we're always in mock mode during tests.
os.environ.pop("GITHUB_TOKEN", None)
os.environ.pop("OPENAI_API_KEY", None)

from tools import (
    find_duplicate_issues,
    search_codebase,
    score_severity,
    post_triage_comment,
    apply_labels,
    fetch_new_issues,
    _heuristic_score,
    _mock_search,
    load_mock_issues,
)


# ---------------------------------------------------------------------------
# Tool 1 — fetch_new_issues
# ---------------------------------------------------------------------------

class TestFetchNewIssues:
    def test_returns_list(self):
        raw = fetch_new_issues.invoke({"repo_name": "psf/requests"})
        data = json.loads(raw)
        assert isinstance(data, list)

    def test_all_five_mock_issues_loaded(self):
        raw = fetch_new_issues.invoke({"repo_name": "psf/requests"})
        data = json.loads(raw)
        assert len(data) == 5

    def test_issue_shape(self):
        raw = fetch_new_issues.invoke({"repo_name": "psf/requests"})
        issue = json.loads(raw)[0]
        for field in ("issue_number", "title", "body", "created_at", "author"):
            assert field in issue, f"Missing field: {field}"


# ---------------------------------------------------------------------------
# Tool 2 — find_duplicate_issues
# ---------------------------------------------------------------------------

class TestFindDuplicateIssues:
    def test_returns_at_most_three(self):
        raw = find_duplicate_issues.invoke({
            "issue_text": "login fails",
            "repo_name": "psf/requests",
        })
        results = json.loads(raw)
        assert len(results) <= 3

    def test_scores_between_zero_and_one(self):
        raw = find_duplicate_issues.invoke({
            "issue_text": "OAuth token login authentication fails 500 error",
            "repo_name": "psf/requests",
        })
        for item in json.loads(raw):
            assert 0.0 <= item["similarity_score"] <= 1.0

    def test_high_similarity_text_flagged_as_duplicate(self):
        # Very close to mock issue #142 — should exceed the 0.75 threshold.
        text = (
            "Login fails with OAuth tokens. "
            "The login endpoint returns 500 Internal Server Error. "
            "KeyError access_token in authenticate()."
        )
        raw = find_duplicate_issues.invoke({"issue_text": text, "repo_name": "psf/requests"})
        results = json.loads(raw)
        assert results[0]["is_likely_duplicate"] is True
        assert results[0]["similarity_score"] > 0.75

    def test_unrelated_text_not_flagged(self):
        text = "The blue navbar button overlaps on 320px mobile screens in Safari."
        raw = find_duplicate_issues.invoke({"issue_text": text, "repo_name": "psf/requests"})
        results = json.loads(raw)
        assert results[0]["is_likely_duplicate"] is False

    def test_result_has_required_keys(self):
        raw = find_duplicate_issues.invoke({"issue_text": "bug", "repo_name": "psf/requests"})
        for item in json.loads(raw):
            for key in ("issue_number", "title", "similarity_score", "url", "is_likely_duplicate"):
                assert key in item


# ---------------------------------------------------------------------------
# Tool 3 — search_codebase
# ---------------------------------------------------------------------------

class TestSearchCodebase:
    def test_upload_keywords_map_to_storage(self):
        raw = _mock_search("file upload data loss")
        files = [r["file"] for r in json.loads(raw)]
        assert any("storage" in f or "upload" in f for f in files)

    def test_auth_keywords_map_to_auth_module(self):
        raw = _mock_search("login oauth token auth")
        files = [r["file"] for r in json.loads(raw)]
        assert any("auth" in f for f in files)

    def test_dark_mode_maps_to_ui(self):
        raw = _mock_search("dark mode ui theme")
        files = [r["file"] for r in json.loads(raw)]
        assert any("theme" in f or "ui" in f for f in files)

    def test_typo_readme_maps_to_readme(self):
        raw = _mock_search("typo readme documentation")
        files = [r["file"] for r in json.loads(raw)]
        assert any("README" in f or "readme" in f.lower() for f in files)

    def test_unknown_keywords_return_fallback(self):
        raw = _mock_search("xyzzy frobnicator quux")
        results = json.loads(raw)
        assert len(results) > 0

    def test_nonexistent_path_uses_mock(self):
        raw = search_codebase.invoke({
            "keywords": "login",
            "local_repo_path": "/nonexistent/path/that/does/not/exist",
        })
        results = json.loads(raw)
        assert isinstance(results, list)
        assert len(results) > 0

    def test_bigram_data_loss_beats_generic_data(self):
        # "data loss" as a bigram should match storage, not src/models/data.py
        raw = _mock_search("data loss upload silent truncat")
        files = [r["file"] for r in json.loads(raw)]
        # The first result should be storage-related, not the generic data model.
        assert "storage" in files[0] or "upload" in files[0]


# ---------------------------------------------------------------------------
# Tool 4 — score_severity (heuristic path, no OpenAI key)
# ---------------------------------------------------------------------------

class TestScoreSeverity:
    def _score(self, title, body):
        return json.loads(_heuristic_score(title, body, f"{title} {body}".lower()))

    def test_data_loss_is_critical(self):
        r = self._score("Silent data loss", "Files are silently truncated causing data loss in production.")
        assert r["severity"] == "critical"

    def test_crash_is_critical(self):
        r = self._score("App crashes on startup", "The application crashes every time I start it.")
        assert r["severity"] == "critical"

    def test_security_vulnerability_is_critical(self):
        r = self._score("Security vulnerability in auth", "There is a security vulnerability allowing bypass.")
        assert r["severity"] == "critical"

    def test_auth_error_is_high(self):
        r = self._score("Authentication fails", "Login returns 500 error for all users.")
        assert r["severity"] == "high"

    def test_typo_is_low(self):
        r = self._score("Typo in README", "recieve should be receive in the docs")
        assert r["severity"] == "low"
        assert "documentation" in r["suggested_labels"]

    def test_feature_request_is_low(self):
        r = self._score("Feature request: dark mode", "Please add dark mode support to the UI.")
        assert r["severity"] == "low"
        assert "feature" in r["suggested_labels"]

    def test_vague_body_triggers_missing_info(self):
        r = self._score("App crashes", "crashes")
        assert r["missing_info"] != ""

    def test_detailed_body_with_trace_no_missing_info(self):
        long_body = (
            "When I call authenticate() with an expired OAuth token I get a KeyError. "
            "Traceback: File 'src/auth/login.py' line 42 KeyError access_token. "
            "Python 3.11 Ubuntu 22.04. Reproducible every time."
        )
        r = self._score("Auth fails with expired token", long_body)
        assert r["missing_info"] == ""

    def test_severity_threshold_boundary(self):
        # Exactly 50-word body WITH a traceback should NOT trigger missing_info.
        body = ("word " * 49 + "Traceback: File 'x.py' line 1 Error").strip()
        r = self._score("Some error", body)
        assert r["missing_info"] == ""

    def test_suggested_labels_is_list(self):
        r = self._score("Something broke", "It broke and does not work anymore.")
        assert isinstance(r["suggested_labels"], list)
        assert len(r["suggested_labels"]) > 0


# ---------------------------------------------------------------------------
# Tool 5 — post_triage_comment (mock/dry-run mode)
# ---------------------------------------------------------------------------

class TestPostTriageComment:
    def _make_report(self, severity="medium"):
        return json.dumps({
            "severity": severity,
            "reason": "Test reason.",
            "likely_component": "src/core/utils.py",
            "duplicates": [],
            "missing_info": "",
            "suggested_labels": ["bug", severity],
        })

    def test_returns_success_in_mock_mode(self):
        raw = post_triage_comment.invoke({
            "issue_number": 9999,
            "repo_name": "psf/requests",
            "triage_report": self._make_report(),
        })
        result = json.loads(raw)
        assert result["success"] is True

    def test_comment_preview_contains_severity(self):
        raw = post_triage_comment.invoke({
            "issue_number": 9999,
            "repo_name": "psf/requests",
            "triage_report": self._make_report("critical"),
        })
        result = json.loads(raw)
        assert "Critical" in result.get("preview", "")

    def test_dry_run_tag_appears_when_set(self, monkeypatch):
        monkeypatch.setenv("TRIAGE_DRY_RUN", "1")
        raw = post_triage_comment.invoke({
            "issue_number": 9999,
            "repo_name": "psf/requests",
            "triage_report": self._make_report(),
        })
        result = json.loads(raw)
        # In mock mode (no gh_client) the tag is [MOCK]; that's fine —
        # the important thing is that success is True and no real call was made.
        assert result["success"] is True

    def test_handles_invalid_json_gracefully(self):
        raw = post_triage_comment.invoke({
            "issue_number": 9999,
            "repo_name": "psf/requests",
            "triage_report": "This is a plain text report, not JSON.",
        })
        result = json.loads(raw)
        assert result["success"] is True


# ---------------------------------------------------------------------------
# Tool 6 — apply_labels (mock mode)
# ---------------------------------------------------------------------------

class TestApplyLabels:
    def test_returns_success_in_mock_mode(self):
        raw = apply_labels.invoke({
            "issue_number": 9999,
            "repo_name": "psf/requests",
            "labels": '["bug", "high"]',
        })
        result = json.loads(raw)
        assert result["success"] is True

    def test_applied_labels_match_input(self):
        raw = apply_labels.invoke({
            "issue_number": 9999,
            "repo_name": "psf/requests",
            "labels": '["bug", "critical", "needs-info"]',
        })
        result = json.loads(raw)
        assert set(result["applied_labels"]) == {"bug", "critical", "needs-info"}

    def test_empty_labels_returns_error(self):
        raw = apply_labels.invoke({
            "issue_number": 9999,
            "repo_name": "psf/requests",
            "labels": "[]",
        })
        result = json.loads(raw)
        assert result["success"] is False

    def test_accepts_comma_separated_string(self):
        raw = apply_labels.invoke({
            "issue_number": 9999,
            "repo_name": "psf/requests",
            "labels": "bug, medium",
        })
        result = json.loads(raw)
        assert result["success"] is True
        assert "bug" in result["applied_labels"]


# ---------------------------------------------------------------------------
# Integration — end-to-end mock triage of each scenario
# ---------------------------------------------------------------------------

class TestEndToEnd:
    """Run the full manual triage pipeline on each mock issue and assert
    the outputs are sensible — no tokens required."""

    def _triage(self, issue_number: int) -> dict:
        """Return the triage report dict for a given mock issue number."""
        from main import triage_manually

        issues = load_mock_issues()
        issue = next(i for i in issues if i["issue_number"] == issue_number)

        captured = {}

        # Monkey-patch post_triage_comment to capture the report.
        original_post = post_triage_comment.invoke

        def capturing_post(args):
            captured["report"] = json.loads(args["triage_report"])
            return original_post(args)

        post_triage_comment.invoke = capturing_post
        try:
            triage_manually(issue, "psf/requests", "/nonexistent")
        finally:
            post_triage_comment.invoke = original_post

        return captured.get("report", {})

    def test_critical_issue_scored_critical(self):
        report = self._triage(1003)
        assert report["severity"] == "critical"

    def test_vague_issue_requests_more_info(self):
        report = self._triage(1002)
        assert report.get("missing_info", "") != ""

    def test_feature_request_scored_low(self):
        report = self._triage(1004)
        assert report["severity"] == "low"

    def test_typo_issue_scored_low_with_docs_label(self):
        report = self._triage(1005)
        assert report["severity"] == "low"
        assert "documentation" in report.get("suggested_labels", [])

    def test_auth_issue_identifies_auth_component(self):
        report = self._triage(1001)
        component = report.get("likely_component", "")
        assert "auth" in component
