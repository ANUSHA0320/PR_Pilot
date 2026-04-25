"""
gradio_app.py
=============
Interactive Gradio web UI for CodeReviewEnv-v0.

Each browser tab gets its own isolated session via gr.State(),
so concurrent users never corrupt each other's episodes.

Run locally
-----------
    pip install gradio
    python gradio_app.py

Then open: http://localhost:7860
"""

from __future__ import annotations

import sys
import os
import subprocess
import re
from pathlib import Path
sys.path.insert(0, os.path.dirname(__file__))

import gradio as gr
import uuid

import code_review_env
from code_review_env.env import CodeReviewEnv
from code_review_env.actions import ACTION_LABELS

# ── Per-session state ─────────────────────────────────────────────────────────

def _make_session() -> dict:
    """Create a fresh isolated session. Called once per browser tab."""
    return {
        "id":   str(uuid.uuid4())[:8],
        "envs": {},   # difficulty → CodeReviewEnv
        "obs":  {},   # difficulty → last observation
        "pr_queue": {},   # difficulty → queued PR descriptors
        "pr_index": {},   # difficulty → active PR index
        "log":  [],   # completed episode records
    }


def _get_env(session: dict, difficulty: str) -> CodeReviewEnv:
    if difficulty not in session["envs"]:
        session["envs"][difficulty] = CodeReviewEnv(
            difficulty=difficulty, seed=None, render_mode="ansi"
        )
    return session["envs"][difficulty]


def _format_agent_view(agent_reports: dict, debate_summary: str, author_reply: str, history) -> str:
    bug = agent_reports.get("bug", "")
    sec = agent_reports.get("security", "")
    perf = agent_reports.get("performance", "")
    summary = debate_summary or ""
    reply = author_reply or ""
    if isinstance(history, list):
        history_text = "\n".join(history)
    else:
        history_text = history or ""
    return (
        f"**BugAgent:** {bug}\n\n"
        f"**SecurityAgent:** {sec}\n\n"
        f"**PerformanceAgent:** {perf}\n\n"
        f"**Debate Summary:** {summary}\n\n"
        f"**Author Reply:** {reply}\n\n"
        f"**Conversation History:**\n{history_text}"
    )


def _format_diff(diff_text: str) -> str:
    if not diff_text:
        return "(empty diff)"
    return f"```diff\n{diff_text}\n```"


def _get_pr_diff(repo_path: str, base_ref: str, head_ref: str) -> str:
    head_fetch = head_ref
    if head_ref.startswith("origin/"):
        head_fetch = head_ref.split("/", 1)[1]
    subprocess.check_output(
        ["git", "fetch", "origin", base_ref, head_fetch],
        cwd=repo_path,
    )
    diff = subprocess.check_output(
        ["git", "diff", base_ref, "FETCH_HEAD"],
        cwd=repo_path,
    )
    return diff.decode()


def _resolve_repo_path(repo_input: str) -> str:
    """Return a local repo path, cloning a URL to a cache directory if needed."""
    if repo_input.startswith("http://") or repo_input.startswith("https://") or repo_input.endswith(".git"):
        cache_root = Path(".cache") / "pr_pilot_repos"
        cache_root.mkdir(parents=True, exist_ok=True)
        repo_name = repo_input.rstrip("/").split("/")[-1].removesuffix(".git")
        local_path = cache_root / repo_name

        if local_path.is_dir():
            subprocess.check_output(["git", "-C", str(local_path), "fetch", "--all"])
        else:
            subprocess.check_output(["git", "clone", repo_input, str(local_path)])

        return str(local_path)

    return repo_input


def _parse_head_refs(head_refs_text: str) -> list[str]:
    refs: list[str] = []
    for raw in re.split(r"[\n,]+", head_refs_text or ""):
        ref = raw.strip()
        if ref and ref not in refs:
            refs.append(ref)
    return refs


def _store_pr_queue(session: dict, difficulty: str, queue: list[dict]) -> None:
    session.setdefault("pr_queue", {})[difficulty] = queue
    session.setdefault("pr_index", {})[difficulty] = 0


def _current_queue_entry(session: dict, difficulty: str) -> tuple[dict | None, int, int]:
    queue = session.get("pr_queue", {}).get(difficulty, [])
    if not queue:
        return None, 0, 0
    idx = session.get("pr_index", {}).get(difficulty, 0)
    idx = max(0, min(idx, len(queue) - 1))
    session.setdefault("pr_index", {})[difficulty] = idx
    return queue[idx], idx, len(queue)


def _queue_position_text(session: dict, difficulty: str) -> str:
    _, idx, total = _current_queue_entry(session, difficulty)
    if total <= 0:
        return "No loaded PR queue"
    return f"Loaded PR {idx + 1}/{total}"


def _queue_summary_text(session: dict, difficulty: str) -> str:
    queue = session.get("pr_queue", {}).get(difficulty, [])
    if not queue:
        return "No loaded PRs from this repository."
    entries = []
    for i, item in enumerate(queue, 1):
        marker = "→" if session.get("pr_index", {}).get(difficulty, 0) == i - 1 else "•"
        entries.append(f"{marker} {i}. {item.get('head_ref', '?')}")
    return "\n".join(entries)


def _build_loaded_pr(pr_source: dict, diff: str, repo_label: str, base_ref: str, head_ref: str) -> dict:
    pr = dict(pr_source)
    pr["diff_patch"] = diff
    pr["repository_context"] = repo_label
    pr["base_ref"] = base_ref
    pr["head_ref"] = head_ref
    pr["source_repo"] = repo_label
    return pr


def _prepare_loaded_pr(env: CodeReviewEnv, pr: dict, repo_label: str):
    raw_env = env.unwrapped
    raw_env._state.current_pull_request = pr
    raw_env._state.agent_reports, raw_env._state.debate_summary = raw_env._run_multi_agent(pr)
    inferred = _infer_issues_from_reports(raw_env._state.agent_reports, raw_env._state.debate_summary)
    if not inferred:
        inferred = _infer_issues_from_diff(pr.get("diff_patch", ""))

    diff_lower = pr.get("diff_patch", "").lower()
    if "return price - 0" in diff_lower or "return 0" in diff_lower:
        inferred.append("off_by_one_logical_bug")
    if "password" in diff_lower or "api_key" in diff_lower or "secret" in diff_lower:
        inferred.append("hardcoded_secret")
    if "select" in diff_lower and "%s" not in diff_lower and "sql" in diff_lower:
        inferred.append("sql_injection")

    pr["expected_issues"] = _dedupe_issues(inferred)
    pr["has_bug"] = bool(pr["expected_issues"])

    raw_env._state.agent_reports, raw_env._state.debate_summary = raw_env._run_multi_agent(pr)
    raw_env._state.conversation_history.clear()
    raw_env._record_dialogue("BugAgent", raw_env._state.agent_reports.get("bug", ""))
    raw_env._record_dialogue("SecurityAgent", raw_env._state.agent_reports.get("security", ""))
    raw_env._record_dialogue("PerformanceAgent", raw_env._state.agent_reports.get("performance", ""))
    raw_env._record_dialogue("Debate", raw_env._state.debate_summary)

    obs = raw_env._build_obs(pr)
    info = {
        "pr_id": pr.get("id", "unknown"),
        "difficulty": pr.get("difficulty", "unknown"),
    }
    pr_info = (
        f"**PR ID:** `{pr.get('id', 'unknown')}`  |  "
        f"**Source:** `{repo_label}`  |  "
        f"**Base:** `{base_ref}`  |  **Head:** `{head_ref}`\n\n"
        f"**File:** `{pr.get('repository_context', 'unknown')}`"
    )
    agent_view = _format_agent_view(
        obs.get("agent_reports", {}),
        obs.get("debate_summary", ""),
        obs.get("author_reply", ""),
        obs.get("conversation_history", ""),
    )
    log_msg = f"[RESET] Loaded PR diff from `{repo_label}` ({base_ref}..{head_ref}). Choose an action below."
    return obs, info, pr_info, agent_view, log_msg, pr.get("diff_patch", "")


def _infer_issues_from_reports(agent_reports: dict, debate_summary: str) -> list[str]:
    text = " ".join(agent_reports.values()).lower() + " " + (debate_summary or "").lower()
    issues: list[str] = []

    def add(issue: str) -> None:
        if issue not in issues:
            issues.append(issue)

    explicit_labels = [
        "hardcoded_password",
        "hardcoded_secret",
        "sql_injection",
        "infinite_loop",
        "division_by_zero",
        "off_by_one_logical_bug",
        "assignment_in_condition",
        "mutable_default_argument",
        "string_format_error",
        "null_pointer_dereference",
        "debug_print_left_in",
        "non_pythonic_loop",
        "missing_context_manager",
        "use_pathlib_over_os_path",
    ]
    for label in explicit_labels:
        if label in text:
            add(label)

    if any(k in text for k in ("secret", "token", "apikey", "api_key", "password")):
        add("hardcoded_secret")
    if "sql" in text and "injection" in text:
        add("sql_injection")
    if "infinite" in text or "loop" in text:
        add("infinite_loop")
    if "division" in text and "zero" in text:
        add("division_by_zero")
    if "off by" in text or "logic" in text or "boundary" in text:
        add("off_by_one_logical_bug")
    if "assignment in condition" in text or "assignment_in_condition" in text:
        add("assignment_in_condition")
    if "mutable default" in text:
        add("mutable_default_argument")
    if "string format" in text:
        add("string_format_error")
    if "null pointer" in text or "none check" in text:
        add("null_pointer_dereference")
    if "debug print" in text or ("print" in text and "debug" in text):
        add("debug_print_left_in")
    if "range(len(" in text or "enumerate" in text:
        add("non_pythonic_loop")
    if "open(" in text and "with" not in text:
        add("missing_context_manager")
    if "os.path" in text:
        add("use_pathlib_over_os_path")

    return issues


def _infer_issues_from_diff(diff_text: str) -> list[str]:
    text = diff_text.lower()
    issues: list[str] = []

    def add(issue: str) -> None:
        if issue not in issues:
            issues.append(issue)

    if any(k in text for k in ("password", "secret", "api_key", "apikey", "token")):
        add("hardcoded_secret")
    if "sql" in text and ("execute(" in text or "select" in text):
        add("sql_injection")
    if "division" in text and "0" in text:
        add("division_by_zero")
    if re.search(r'^-\s*if\b[^\n]*>\s*\d+', diff_text, re.MULTILINE) and re.search(r'^\+\s*if\b[^\n]*>=\s*\d+', diff_text, re.MULTILINE):
        add("off_by_one_logical_bug")
    if re.search(r'^-\s*if\b[^\n]*=\s*[^=]', diff_text, re.MULTILINE):
        add("assignment_in_condition")
    if "lst=[]" in text or "dict={}" in text:
        add("mutable_default_argument")
    if "% ()" in text or "%()" in text:
        add("string_format_error")
    if "null" in text or "none" in text:
        add("null_pointer_dereference")
    if "print(" in text and "debug" in text:
        add("debug_print_left_in")
    if "for i in range(len(" in text:
        add("non_pythonic_loop")
    if "open(" in text and "with" not in text:
        add("missing_context_manager")
    if "os.path.join" in text:
        add("use_pathlib_over_os_path")

    return issues


_CRITICAL_ISSUES = {
    "hardcoded_password",
    "hardcoded_secret",
    "sql_injection",
    "infinite_loop",
    "division_by_zero",
    "null_pointer_dereference",
    "assignment_in_condition",
    "off_by_one_logical_bug",
}


def _dedupe_issues(issues: list[str]) -> list[str]:
    deduped: list[str] = []
    for issue in issues:
        if issue and issue not in deduped:
            deduped.append(issue)
    return deduped


def _suggest_action_index(pr: dict, obs: dict, agent_reports: dict, debate_summary: str, severity_score: float = 0.0) -> int:
    issues = _dedupe_issues(
        list(pr.get("expected_issues", []))
        or _infer_issues_from_reports(agent_reports, debate_summary)
        or _infer_issues_from_diff(pr.get("diff_patch", ""))
    )

    tests_passed = bool(obs.get("test_results", {}).get("tests_passed", 1))
    lint_clean = not bool(obs.get("lint_report", {}).get("unused_variable", 0))
    suspicious = bool(issues) or not tests_passed or not lint_clean or severity_score >= 0.25

    if not suspicious:
        return 0

    if any(issue in _CRITICAL_ISSUES for issue in issues) or severity_score >= 0.6:
        return 1

    # Prefer a conservative reject signal over approve for anything ambiguous.
    return 1


# ── Core functions ─────────────────────────────────────────────────────────────

def reset_env(difficulty: str, session: dict):
    """Load a new PR and return UI components."""
    env = _get_env(session, difficulty)
    obs, info = env.reset()
    session["obs"][difficulty] = obs

    diff_text = obs["diff_patch"]
    ctx_text  = obs["repository_context"] or "unknown"
    file_type = obs["file_type"] or "python"
    tests_ok  = "Passing" if obs["test_results"]["tests_passed"] else "Failing"
    lint_warn = "Unused variable detected" if obs["lint_report"]["unused_variable"] else "No lint warnings"

    pr_info = (
        f"**PR ID:** `{info.get('pr_id', 'unknown')}`  |  "
        f"**Difficulty:** `{difficulty}`  |  "
        f"**File:** `{ctx_text}`  |  "
        f"**Language:** `{file_type}`\n\n"
        f"**Tests:** {tests_ok}  |  **Lint:** {lint_warn}"
    )
    agent_view = _format_agent_view(
        obs.get("agent_reports", {}),
        obs.get("debate_summary", ""),
        obs.get("author_reply", ""),
        obs.get("conversation_history", ""),
    )
    log_msg = f"[RESET] Loaded PR `{info.get('pr_id')}` ({difficulty}). Choose an action below."
    suggested_action = _suggest_action_index(
        env._state.current_pull_request,
        obs,
        obs.get("agent_reports", {}),
        obs.get("debate_summary", ""),
        getattr(env._state, "severity_score", 0.0),
    )
    return (
        _format_diff(diff_text),
        pr_info,
        agent_view,
        log_msg,
        "Step: 0 / 5 | Score: 0.000 | Decision: pending",
        gr.update(value=suggested_action),
        session,
    )


def load_pr_from_git(repo_path: str, base_ref: str, head_ref: str, difficulty: str, session: dict):
    """Load PR diff from a local git clone and inject into the env."""
    resolved_path = _resolve_repo_path(repo_path)
    if not os.path.isdir(resolved_path):
        return "", "Repo not found.", "", "Invalid repo path.", "No active episode", gr.update(value=0), session

    env = _get_env(session, difficulty)
    raw_env = env.unwrapped

    diff = _get_pr_diff(resolved_path, base_ref, head_ref)
    obs, info = env.reset()

    pr = raw_env._state.current_pull_request
    pr["diff_patch"] = diff
    pr["repository_context"] = os.path.basename(resolved_path)
    pr["file_type"] = "python"

    raw_env._state.agent_reports, raw_env._state.debate_summary = raw_env._run_multi_agent(pr)
    inferred = _infer_issues_from_reports(raw_env._state.agent_reports, raw_env._state.debate_summary)
    if not inferred:
        inferred = _infer_issues_from_diff(diff)

    diff_lower = diff.lower()
    if "return price - 0" in diff_lower or "return 0" in diff_lower:
        inferred.append("off_by_one_logical_bug")
    if "password" in diff_lower or "api_key" in diff_lower or "secret" in diff_lower:
        inferred.append("hardcoded_secret")
    if "select" in diff_lower and "%s" not in diff_lower and "sql" in diff_lower:
        inferred.append("sql_injection")
    if not inferred and getattr(raw_env._state, "severity_score", 0.0) >= 0.25:
        inferred.append("review_issue")
    pr["expected_issues"] = inferred
    pr["has_bug"] = bool(inferred)

    raw_env._state.agent_reports, raw_env._state.debate_summary = raw_env._run_multi_agent(pr)
    raw_env._state.conversation_history.clear()
    raw_env._record_dialogue("BugAgent", raw_env._state.agent_reports.get("bug", ""))
    raw_env._record_dialogue("SecurityAgent", raw_env._state.agent_reports.get("security", ""))
    raw_env._record_dialogue("PerformanceAgent", raw_env._state.agent_reports.get("performance", ""))
    raw_env._record_dialogue("Debate", raw_env._state.debate_summary)

    obs = raw_env._build_obs(pr)
    session["obs"][difficulty] = obs

    diff_text = diff
    ctx_text = obs["repository_context"] or "unknown"
    file_type = obs["file_type"] or "python"
    tests_ok = "Passing" if obs["test_results"]["tests_passed"] else "Failing"
    lint_warn = "Unused variable detected" if obs["lint_report"]["unused_variable"] else "No lint warnings"

    pr_info = (
        f"**PR ID:** `{info.get('pr_id', 'unknown')}`  |  "
        f"**Difficulty:** `{difficulty}`  |  "
        f"**File:** `{ctx_text}`  |  "
        f"**Language:** `{file_type}`\n\n"
        f"**Tests:** {tests_ok}  |  **Lint:** {lint_warn}"
    )
    agent_view = _format_agent_view(
        obs.get("agent_reports", {}),
        obs.get("debate_summary", ""),
        obs.get("author_reply", ""),
        obs.get("conversation_history", ""),
    )
    log_msg = f"[RESET] Loaded PR diff from `{repo_path}` ({base_ref}..{head_ref}). Choose an action below."
    status = "Step: 0 / 5 | Score: 0.000 | Decision: pending"
    suggested_action = _suggest_action_index(
        pr,
        obs,
        obs.get("agent_reports", {}),
        obs.get("debate_summary", ""),
        getattr(raw_env._state, "severity_score", 0.0),
    )

    return _format_diff(diff_text), pr_info, agent_view, log_msg, status, gr.update(value=suggested_action), session


def load_prs_from_git(repo_path: str, base_ref: str, head_refs_text: str, difficulty: str, session: dict):
    """Load multiple PR diffs from one repo and create a navigable queue."""
    resolved_path = _resolve_repo_path(repo_path)
    if not os.path.isdir(resolved_path):
        return "", "Repo not found.", "", "Invalid repo path.", "No active episode", gr.update(value=0), session

    head_refs = _parse_head_refs(head_refs_text)
    if not head_refs:
        return "", "No PR refs provided.", "", "Add one or more head refs.", "No active episode", gr.update(value=0), session

    queue = [
        {
            "repo_path": resolved_path,
            "repo_label": os.path.basename(resolved_path),
            "base_ref": base_ref,
            "head_ref": head_ref,
        }
        for head_ref in head_refs
    ]
    _store_pr_queue(session, difficulty, queue)

    first = queue[0]
    single_result = load_pr_from_git(first["repo_path"], first["base_ref"], first["head_ref"], difficulty, session)
    diff_box, pr_info, agent_view, log_msg, status, action_choice, session = single_result

    pr_info = f"{pr_info}\n\n**Queued PRs:** `{len(queue)}`"
    log_msg = f"{log_msg}\n\n**Queue:** loaded {len(queue)} PR(s) from this repository."
    status = f"{status} | Queue: 1/{len(queue)}"

    return diff_box, pr_info, agent_view, log_msg, status, action_choice, session


def switch_loaded_pr(difficulty: str, direction: int, session: dict):
    """Move to the previous/next queued PR for the active repository."""
    entry, idx, total = _current_queue_entry(session, difficulty)
    if not entry:
        return "No loaded PR queue. Use **Load PRs** first.", "No active episode", "", "No active episode", gr.update(value=0), session

    idx = (idx + direction) % total
    session.setdefault("pr_index", {})[difficulty] = idx
    entry = session["pr_queue"][difficulty][idx]
    diff_box, pr_info, agent_view, log_msg, status, action_choice, session = load_pr_from_git(
        entry["repo_path"],
        entry["base_ref"],
        entry["head_ref"],
        difficulty,
        session,
    )
    pr_info = f"{pr_info}\n\n**Queued PRs:** `{total}` | **Active:** `{idx + 1}/{total}`"
    status = f"{status} | Queue: {idx + 1}/{total}"
    return diff_box, pr_info, agent_view, log_msg, status, action_choice, session


def take_action(difficulty: str, action_idx: int, session: dict):
    """Submit an action to the env and return updated UI state."""
    env = _get_env(session, difficulty)

    if env._state.current_pull_request is None:
        return "No active episode. Click **Load PR** first.", "No active episode", session

    obs, reward, terminated, truncated, info = env.step(action_idx)
    session["obs"][difficulty] = obs
    done  = terminated or truncated
    label = ACTION_LABELS[action_idx]
    score = info.get("task_score", 0.0)
    steps = info["steps_taken"]
    decision = info.get("review_decision") or "pending"
    status = f"Step: {steps} / 5 | Score: {score:.3f} | Decision: {decision}"

    if done:
        grade = "CORRECT" if score >= 0.7 else ("PARTIAL" if score >= 0.4 else "WRONG")
        log_msg = (
            f"[DONE] Action: **{label}**  ->  Reward: `{reward:+.3f}`\n\n"
            f"**Episode Score: {score:.3f}** - {grade}\n\n"
            f"Click **Load PR** to start a new episode."
        )
        session["log"].append({
            "difficulty": difficulty,
            "pr_id":      info.get("pr_id", "?"),
            "task_score": round(score, 3),
            "decision":   decision,
            "steps":      steps,
        })
    else:
        log_msg = (
            f"[STEP {steps}] Action: **{label}**  ->  Reward: `{reward:+.3f}`  "
            f"| Score so far: `{score:.3f}`"
        )
    agent_view = _format_agent_view(
        info.get("agent_reports", {}),
        info.get("debate_summary", ""),
        info.get("author_reply", ""),
        info.get("conversation_history", []),
    )
    return log_msg, status, agent_view, session


def show_leaderboard(session: dict):
    """Returns a markdown leaderboard table for this session."""
    logs = session.get("log", [])
    if not logs:
        return "No completed episodes yet. Play a round first!"

    rows = ["| # | Difficulty | PR ID | Score | Decision | Steps |",
            "|---|-----------|-------|-------|---------|-------|"] 
    for i, ep in enumerate(logs[-20:], 1):
        rows.append(
            f"| {i} | {ep['difficulty']} | {ep['pr_id']} "
            f"| {ep['task_score']:.3f} | {ep['decision']} | {ep['steps']} |"
        )
    for diff in ("easy", "medium", "hard"):
        sc = [e["task_score"] for e in logs if e["difficulty"] == diff]
        if sc:
            rows.append(f"\n**{diff.capitalize()} avg score:** `{sum(sc)/len(sc):.3f}` over {len(sc)} episodes")
    return "\n".join(rows)


def heuristic_demo(difficulty: str, session: dict):
    """Run one full heuristic episode and return a summary."""
    env = _get_env(session, difficulty)
    obs, info = env.reset()
    session["obs"][difficulty] = obs

    pr   = env._state.current_pull_request
    diff = pr.get("diff_patch", "")

    suggested_action = _suggest_action_index(
        pr,
        obs,
        obs.get("agent_reports", {}),
        obs.get("debate_summary", ""),
        getattr(env._state, "severity_score", 0.0),
    )

    log_lines = [f"**Auto-demo ({difficulty} PR `{info['pr_id']}`)**\n"]

    action = suggested_action

    if action in (2, 3, 4):
        _, r2, _, _, _ = env.step(action)
        log_lines.append(f"- Step 1: `{ACTION_LABELS[action]}` → reward `{r2:+.3f}`")
        terminal = 1 if action != 0 else 0
        _, r3, _, _, final_info = env.step(terminal)
        log_lines.append(f"- Step 2: `{ACTION_LABELS[terminal]}` → reward `{r3:+.3f}`")
    else:
        _, r2, _, _, final_info = env.step(action)
        log_lines.append(f"- Step 1: `{ACTION_LABELS[action]}` → reward `{r2:+.3f}`")

    score = final_info.get("task_score", 0.0)
    grade = "CORRECT" if score >= 0.7 else ("PARTIAL" if score >= 0.4 else "WRONG")
    log_lines.append(f"\n**Final Score: {score:.3f}** - {grade}")

    session["log"].append({
        "difficulty": difficulty,
        "pr_id":      info["pr_id"],
        "task_score": round(score, 3),
        "decision":   final_info.get("review_decision", "unknown"),
        "steps":      final_info["steps_taken"],
    })
    status = (
        f"Step: {final_info['steps_taken']} / 5 | "
        f"Score: {score:.3f} | Decision: {final_info.get('review_decision', '?')}"
    )
    agent_view = _format_agent_view(
        env._state.agent_reports,
        env._state.debate_summary,
        env._state.author_reply,
        env._state.conversation_history,
    )
    return "\n".join(log_lines), status, agent_view, gr.update(value=suggested_action), session


# ── Gradio UI ──────────────────────────────────────────────────────────────────

ACTION_CHOICES = [
    ("0 - Approve (end)", 0),
    ("1 - Reject (end)", 1),
    ("2 - Request Changes", 2),
    ("3 - Comment Bug", 3),
    ("4 - Suggest Patch", 4),
]

with gr.Blocks(title="PR Pilot") as demo:

    # Per-session isolated state — each browser tab gets its own copy
    session_state = gr.State(_make_session)

    gr.Markdown(
        """# PR Pilot
**A Reinforcement Learning Environment for Automated Pull Request Review**

Simulate a real code review workflow. Load a PR, detect bugs, suggest patches, and score your decisions.
Each browser session is fully isolated — multiple users can play simultaneously.
"""
    )

    with gr.Row():
        github_mode_btn = gr.Button("PR from GitHub", variant="primary")
        dataset_mode_btn = gr.Button("Load from Dataset", variant="secondary")

    github_panel = gr.Column(visible=True)
    with github_panel:
        gr.Markdown(
            """
**Load a real PR diff**

Paste a GitHub repo URL or a local clone path. Provide the base ref and one or more head refs.
"""
        )
        repo_path_tb = gr.Textbox(label="Repo URL or local path", placeholder="https://github.com/user/repo.git or C:/path/to/repo")
        with gr.Row():
            base_ref_tb = gr.Textbox(label="Base ref", value="main")
        head_refs_tb = gr.Textbox(
            label="Head refs / PR branches",
            value="origin/Branch1",
            lines=4,
            placeholder="origin/Branch1\norigin/Branch2\norigin/Branch3",
        )
        with gr.Row():
            load_prs_btn = gr.Button("Load PRs", variant="primary")
            prev_pr_btn = gr.Button("Previous PR", variant="secondary")
            next_pr_btn = gr.Button("Next PR", variant="secondary")

    dataset_panel = gr.Column(visible=False)
    with dataset_panel:
        gr.Markdown(
            """
**Load a benchmark PR**

This uses the built-in dataset and difficulty setting to start a fresh episode.
"""
        )
        with gr.Row():
            difficulty_dd = gr.Dropdown(
                choices=["easy", "medium", "hard", "adaptive"],
                value="easy",
                label="Difficulty",
                interactive=True,
            )
            demo_btn = gr.Button("Auto Demo", variant="secondary")
        reset_btn = gr.Button("Load PR", variant="primary")

    pr_info_md = gr.Markdown("Click **Load PR Diff** or **Load PR** to start.")
    agent_md = gr.Markdown("Agent analysis will appear here.")
    diff_box = gr.Markdown("(diff will appear here)")

    with gr.Row():
        action_radio = gr.Radio(
            choices=ACTION_CHOICES,
            label="Choose Action",
            value=3,
        )
        submit_btn = gr.Button("Submit Action", variant="primary", scale=1)

    status_bar = gr.Textbox(
        label="Episode Status",
        value="No active episode",
        interactive=False,
    )
    log_box = gr.Markdown("Action log will appear here.")

    # ── Event handlers ─────────────────────────────────────────────────────────
    reset_btn.click(
        fn=reset_env,
        inputs=[difficulty_dd, session_state],
        outputs=[diff_box, pr_info_md, agent_md, log_box, status_bar, action_radio, session_state],
    )

    load_prs_btn.click(
        fn=load_prs_from_git,
        inputs=[repo_path_tb, base_ref_tb, head_refs_tb, difficulty_dd, session_state],
        outputs=[diff_box, pr_info_md, agent_md, log_box, status_bar, action_radio, session_state],
    )

    prev_pr_btn.click(
        fn=lambda difficulty, session: switch_loaded_pr(difficulty, -1, session),
        inputs=[difficulty_dd, session_state],
        outputs=[diff_box, pr_info_md, agent_md, log_box, status_bar, action_radio, session_state],
    )

    next_pr_btn.click(
        fn=lambda difficulty, session: switch_loaded_pr(difficulty, 1, session),
        inputs=[difficulty_dd, session_state],
        outputs=[diff_box, pr_info_md, agent_md, log_box, status_bar, action_radio, session_state],
    )

    def _show_github():
        return gr.update(visible=True), gr.update(visible=False)

    def _show_dataset():
        return gr.update(visible=False), gr.update(visible=True)

    github_mode_btn.click(
        fn=_show_github,
        inputs=[],
        outputs=[github_panel, dataset_panel],
    )

    dataset_mode_btn.click(
        fn=_show_dataset,
        inputs=[],
        outputs=[github_panel, dataset_panel],
    )

    submit_btn.click(
        fn=take_action,
        inputs=[difficulty_dd, action_radio, session_state],
        outputs=[log_box, status_bar, agent_md, session_state],
    )

    demo_btn.click(
        fn=heuristic_demo,
        inputs=[difficulty_dd, session_state],
        outputs=[log_box, status_bar, agent_md, action_radio, session_state],
    )



if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        inbrowser=True,
    )
