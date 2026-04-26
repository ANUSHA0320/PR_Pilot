"""
tests/test_agents.py
====================
Coverage-focused tests for the multi-agent review heuristics.
"""

from code_review_env.agents import (
    Agent,
    BugAgent,
    DebateMechanism,
    Finding,
    PerformanceAgent,
    SecurityAgent,
    _findings_from_llm_text,
    _llm_available,
    _llm_backend_preference,
    _safe_json_loads,
    _strip_code_fences,
    run_multi_agent_analysis,
)


class DummyAgent(Agent):
    def __init__(self):
        super().__init__("Dummy")

    def analyze(self, diff: str, context: dict):
        return []


def test_json_helpers_and_llm_fallback(monkeypatch):
    monkeypatch.delenv("PR_PILOT_AGENT_BACKEND", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    assert _llm_backend_preference() == "hybrid"
    assert _llm_available() is False
    assert _strip_code_fences("```json\n{\"a\": 1}\n```") == '{"a": 1}'
    assert _safe_json_loads("not json") is None

    findings = _findings_from_llm_text(
        '{"findings": [{"issue_type": "x", "severity": "HIGH", '
        '"description": "d", "confidence": 0.5, "line_number": 12}]}'
    )
    assert len(findings) == 1
    assert findings[0].issue_type == "x"
    assert findings[0].severity == "high"
    assert findings[0].line_number == 12


def test_generate_report_groups_severities():
    agent = DummyAgent()

    assert agent.generate_report([]) == "No issues detected."

    findings = [
        Finding("critical_a", "critical", "", 1.0),
        Finding("high_a", "high", "", 1.0),
        Finding("medium_a", "medium", "", 1.0),
        Finding("medium_b", "medium", "", 1.0),
        Finding("medium_c", "medium", "", 1.0),
    ]
    report = agent.generate_report(findings)
    assert "CRITICAL: critical_a" in report
    assert "HIGH: high_a" in report
    assert "MEDIUM: medium_a, medium_b" in report
    assert "medium_c" not in report


def test_bug_security_performance_agents_cover_key_paths():
    bug_diff = (
        "def broken(x)\n"
        "if x\n"
        "    return none\n"
        "value = 10 / 0\n"
        "for i in range(n + 1):\n"
        "    unused_value = 1  # unused\n"
        "def f(items=[]):\n"
        "    pass\n"
        "if count = 3\n"
        "    print(\"debug mode\")\n"
    )
    bug_findings = BugAgent().analyze(bug_diff, {"test_results": {"tests_passed": 0}})
    bug_types = {f.issue_type for f in bug_findings}
    assert {"missing_colon", "division_by_zero", "off_by_one", "unused_variable", "mutable_default_argument", "assignment_in_condition", "debug_print", "test_failure"}.issubset(bug_types)

    security_diff = (
        "password = \"secret\"\n"
        "api_key = \"12345678901234567890\"\n"
        "cursor.execute(\"SELECT * FROM users WHERE id=\" + user_id)\n"
        "pickle.loads(data)\n"
        "os.system(\"ls \" + path)\n"
        "eval(user_input)\n"
        "md5(data)\n"
    )
    security_findings = SecurityAgent().analyze(security_diff, {})
    security_types = {f.issue_type for f in security_findings}
    assert {"hardcoded_password", "hardcoded_secret", "sql_injection", "unsafe_deserialization", "command_injection", "dangerous_eval", "weak_cryptography"}.issubset(security_types)

    performance_diff = (
        "for i in range(len(items)):\n"
        "    total += 'x'\n"
        "for item in items:\n"
        "    results.append(item)\n"
        "open('data.txt')\n"
        "for x in xs:\n"
        "    for y in ys:\n"
        "        pass\n"
        "global config\n"
        "os.path.join('a', 'b')\n"
        "if key in mydict.keys():\n"
        "    pass\n"
    )
    performance_findings = PerformanceAgent().analyze(performance_diff, {})
    performance_types = {f.issue_type for f in performance_findings}
    assert {"non_pythonic_loop", "string_concat_in_loop", "use_list_comprehension", "missing_context_manager", "nested_loops", "global_variable", "use_pathlib", "inefficient_dict_operation"}.issubset(performance_types)


def test_debate_and_multi_agent_analysis(monkeypatch):
    summary, score = DebateMechanism.debate([], [], [])
    assert "approve is reasonable" in summary
    assert score == 0.0

    crit_summary, crit_score = DebateMechanism.debate(
        [Finding("bug", "critical", "", 1.0)],
        [Finding("sec", "high", "", 0.9)],
        [Finding("perf", "medium", "", 0.8)],
    )
    assert "CRITICAL CONSENSUS" in crit_summary
    assert crit_score > 0

    monkeypatch.setenv("PR_PILOT_AGENT_BACKEND", "heuristic")
    reports, debate_summary, severity_score = run_multi_agent_analysis(
        {
            "diff_patch": "password = \"secret\"\nfor i in range(len(items)):\n    pass\n",
            "test_results": {"tests_passed": 0},
            "lint_report": {"unused_variable": 1},
            "file_type": "py",
        }
    )
    assert set(reports) == {"bug", "security", "performance"}
    assert isinstance(debate_summary, str) and debate_summary
    assert severity_score >= 0
