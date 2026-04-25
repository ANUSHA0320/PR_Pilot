"""
agents.py
=========
Multi-agent code review system with specialized agents.

Each agent analyzes code diffs independently with detailed heuristics,
then participates in a debate mechanism to reach consensus.
"""

from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional


def _llm_backend_preference() -> str:
    """Return the preferred agent backend: 'llm', 'heuristic', or 'hybrid'."""
    return os.getenv("PR_PILOT_AGENT_BACKEND", "hybrid").strip().lower()


def _llm_available() -> bool:
    return bool(os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY"))


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _safe_json_loads(text: str) -> Optional[object]:
    try:
        return json.loads(_strip_code_fences(text))
    except Exception:
        return None


class LLMClient:
    """Thin wrapper around OpenAI / Anthropic for agent analysis."""

    def __init__(self) -> None:
        self.provider = None
        self.model = None
        self._client = None

        if os.getenv("OPENAI_API_KEY"):
            try:
                from openai import OpenAI

                self._client = OpenAI()
                self.provider = "openai"
                self.model = os.getenv("PR_PILOT_OPENAI_MODEL", "gpt-4o-mini")
            except Exception:
                self._client = None

        if self._client is None and os.getenv("ANTHROPIC_API_KEY"):
            try:
                from anthropic import Anthropic

                self._client = Anthropic()
                self.provider = "anthropic"
                self.model = os.getenv("PR_PILOT_ANTHROPIC_MODEL", "claude-3-5-sonnet-latest")
            except Exception:
                self._client = None

    @property
    def available(self) -> bool:
        return self._client is not None and self.provider is not None and self.model is not None

    def analyze(self, system_prompt: str, user_prompt: str) -> str:
        if not self.available:
            raise RuntimeError("No LLM backend is configured.")

        if self.provider == "openai":
            response = self._client.chat.completions.create(
                model=self.model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return response.choices[0].message.content or "{}"

        if self.provider == "anthropic":
            response = self._client.messages.create(
                model=self.model,
                temperature=0,
                max_tokens=1024,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            return "".join(block.text for block in response.content if hasattr(block, "text"))

        raise RuntimeError(f"Unsupported provider: {self.provider}")


def _findings_from_llm_text(text: str) -> List["Finding"]:
    data = _safe_json_loads(text)
    findings: List[Finding] = []

    if isinstance(data, dict):
        items = data.get("findings", data.get("issues", []))
    elif isinstance(data, list):
        items = data
    else:
        items = []

    if not isinstance(items, list):
        return findings

    for item in items:
        if not isinstance(item, dict):
            continue
        findings.append(
            Finding(
                issue_type=str(item.get("issue_type", item.get("type", "llm_finding"))),
                severity=str(item.get("severity", "medium")).lower(),
                description=str(item.get("description", item.get("message", "LLM-detected issue"))),
                confidence=float(item.get("confidence", 0.75)),
                line_number=int(item.get("line_number", 0) or 0),
            )
        )

    return findings


@dataclass
class Finding:
    """A single issue detected by an agent."""
    issue_type: str
    severity: str  # 'critical', 'high', 'medium', 'low'
    description: str
    confidence: float  # 0.0 to 1.0
    line_number: int = 0


class Agent(ABC):
    """Base class for code review agents."""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def analyze(self, diff: str, context: dict) -> List[Finding]:
        """Analyze code and return findings."""
        pass
    
    def generate_report(self, findings: List[Finding]) -> str:
        """Generate human-readable report from findings."""
        if not findings:
            return "No issues detected."
        
        # Group by severity
        critical = [f for f in findings if f.severity == 'critical']
        high = [f for f in findings if f.severity == 'high']
        medium = [f for f in findings if f.severity == 'medium']
        low = [f for f in findings if f.severity == 'low']
        
        report_parts = []
        if critical:
            issues = ", ".join([f.issue_type for f in critical])
            report_parts.append(f"CRITICAL: {issues}")
        if high:
            issues = ", ".join([f.issue_type for f in high])
            report_parts.append(f"HIGH: {issues}")
        if medium:
            issues = ", ".join([f.issue_type for f in medium[:2]])  # Limit to 2
            report_parts.append(f"MEDIUM: {issues}")
        
        return "; ".join(report_parts) if report_parts else "Minor issues detected."


class BugAgent(Agent):
    """Detects syntax errors, logical bugs, and runtime issues."""
    
    def __init__(self):
        super().__init__("BugAgent")
        
    def analyze(self, diff: str, context: dict) -> List[Finding]:
        findings = []
        diff_lower = diff.lower()
        
        # 1. Syntax errors
        if re.search(r'def\s+\w+\([^)]*\)\s*\n', diff):  # missing colon
            findings.append(Finding(
                issue_type="missing_colon",
                severity="critical",
                description="Function definition missing colon",
                confidence=0.9
            ))
        
        if re.search(r'if\s+[^:]+\n\s+return', diff):  # missing colon in if
            findings.append(Finding(
                issue_type="syntax_error",
                severity="critical",
                description="If statement missing colon",
                confidence=0.85
            ))
        
        # 2. Division by zero
        if re.search(r'\/\s*0\b', diff) or 'divide' in diff_lower and 'zero' in diff_lower:
            findings.append(Finding(
                issue_type="division_by_zero",
                severity="critical",
                description="Potential division by zero",
                confidence=0.8
            ))
        
        # 3. Null/None pointer dereference
        if any(pattern in diff_lower for pattern in ['if x:', 'if not x:', 'x = none', 'return none']):
            if 'x.' in diff_lower or 'x[' in diff_lower:
                findings.append(Finding(
                    issue_type="null_pointer_dereference",
                    severity="high",
                    description="Possible null pointer access",
                    confidence=0.7
                ))
        
        # 4. Off-by-one errors
        if re.search(r'range\([^)]*\+\s*1\)', diff) or re.search(r'range\([^)]*-\s*1\)', diff):
            findings.append(Finding(
                issue_type="off_by_one",
                severity="medium",
                description="Potential off-by-one error in range",
                confidence=0.6
            ))
        
        # 5. Unused variables
        if 'unused' in diff_lower or 'unused_' in diff_lower or '# unused' in diff_lower:
            findings.append(Finding(
                issue_type="unused_variable",
                severity="low",
                description="Unused variable detected",
                confidence=0.8
            ))
        
        # 6. Mutable default arguments
        if re.search(r'def\s+\w+\([^)]*=\s*\[\]', diff) or re.search(r'def\s+\w+\([^)]*=\s*\{\}', diff):
            findings.append(Finding(
                issue_type="mutable_default_argument",
                severity="high",
                description="Mutable default argument (list/dict)",
                confidence=0.95
            ))
        
        # 7. Assignment in condition
        if re.search(r'if\s+\w+\s*=\s*[^=]', diff):
            findings.append(Finding(
                issue_type="assignment_in_condition",
                severity="high",
                description="Assignment in conditional (should be ==)",
                confidence=0.9
            ))
        
        # 8. Debug print statements
        if re.search(r'print\s*\(\s*["\']debug', diff_lower):
            findings.append(Finding(
                issue_type="debug_print",
                severity="low",
                description="Debug print statement left in code",
                confidence=0.95
            ))
        
        # 9. Test failures indicate bugs
        if context.get('test_results', {}).get('tests_passed') == 0:
            findings.append(Finding(
                issue_type="test_failure",
                severity="high",
                description="Tests failing, indicating potential bugs",
                confidence=0.8
            ))
        
        return findings


class SecurityAgent(Agent):
    """Detects security vulnerabilities and hardcoded secrets."""
    
    def __init__(self):
        super().__init__("SecurityAgent")
        
    def analyze(self, diff: str, context: dict) -> List[Finding]:
        findings = []
        diff_lower = diff.lower()
        
        # 1. Hardcoded passwords
        password_patterns = [
            r'password\s*=\s*["\'][^"\']+["\']',
            r'passwd\s*=\s*["\'][^"\']+["\']',
            r'pwd\s*=\s*["\'][^"\']+["\']'
        ]
        for pattern in password_patterns:
            if re.search(pattern, diff_lower):
                # Check if it's NOT using environment variable
                if 'os.getenv' not in diff and 'env[' not in diff_lower:
                    findings.append(Finding(
                        issue_type="hardcoded_password",
                        severity="critical",
                        description="Hardcoded password detected",
                        confidence=0.95
                    ))
                    break
        
        # 2. Hardcoded API keys/tokens
        secret_patterns = [
            r'api[_-]?key\s*=\s*["\'][^"\']{20,}["\']',
            r'secret[_-]?key\s*=\s*["\'][^"\']{20,}["\']',
            r'token\s*=\s*["\'][^"\']{20,}["\']',
            r'sk_live_',  # Stripe live key
            r'pk_live_',  # Public live key
        ]
        for pattern in secret_patterns:
            if re.search(pattern, diff_lower):
                if 'os.getenv' not in diff and 'env[' not in diff_lower:
                    findings.append(Finding(
                        issue_type="hardcoded_secret",
                        severity="critical",
                        description="Hardcoded API key or secret detected",
                        confidence=0.9
                    ))
                    break
        
        # 3. SQL injection
        sql_injection_patterns = [
            r'execute\s*\([^)]*\+',  # String concatenation in SQL
            r'cursor\.execute\s*\([^)]*%',  # Old-style formatting
            r'select\s+\*\s+from.*\+',  # SELECT * with concatenation
            r'SELECT.*WHERE.*\+',  # WHERE clause with concatenation
        ]
        for pattern in sql_injection_patterns:
            if re.search(pattern, diff, re.IGNORECASE):
                findings.append(Finding(
                    issue_type="sql_injection",
                    severity="critical",
                    description="Potential SQL injection vulnerability",
                    confidence=0.85
                ))
                break
        
        # 4. Unsafe deserialization
        if 'pickle.loads' in diff or 'yaml.load(' in diff:
            if 'yaml.safe_load' not in diff:
                findings.append(Finding(
                    issue_type="unsafe_deserialization",
                    severity="high",
                    description="Unsafe deserialization (use safe_load)",
                    confidence=0.8
                ))
        
        # 5. Command injection
        if re.search(r'os\.system\s*\([^)]*\+', diff) or re.search(r'subprocess\.(call|run|Popen)\s*\([^)]*\+', diff):
            findings.append(Finding(
                issue_type="command_injection",
                severity="critical",
                description="Potential command injection via string concatenation",
                confidence=0.85
            ))
        
        # 6. Eval usage
        if 'eval(' in diff or 'exec(' in diff:
            findings.append(Finding(
                issue_type="dangerous_eval",
                severity="high",
                description="Use of eval() or exec() is dangerous",
                confidence=0.9
            ))
        
        # 7. Weak cryptography
        if 'md5' in diff_lower or 'sha1' in diff_lower:
            findings.append(Finding(
                issue_type="weak_cryptography",
                severity="medium",
                description="Weak hash algorithm (use SHA256+)",
                confidence=0.7
            ))
        
        return findings


class PerformanceAgent(Agent):
    """Detects performance issues and non-idiomatic code."""
    
    def __init__(self):
        super().__init__("PerformanceAgent")
        
    def analyze(self, diff: str, context: dict) -> List[Finding]:
        findings = []
        diff_lower = diff.lower()
        
        # 1. Non-Pythonic loops
        if re.search(r'for\s+i\s+in\s+range\s*\(\s*len\s*\(', diff):
            findings.append(Finding(
                issue_type="non_pythonic_loop",
                severity="medium",
                description="Use enumerate() instead of range(len())",
                confidence=0.85
            ))
        
        # 2. String concatenation in loops
        if re.search(r'for\s+.*:\s*\n\s+.*\+=.*["\']', diff):
            findings.append(Finding(
                issue_type="string_concat_in_loop",
                severity="medium",
                description="String concatenation in loop (use join())",
                confidence=0.7
            ))
        
        # 3. List append vs comprehension
        if re.search(r'for\s+\w+\s+in.*:\s*\n\s+\w+\.append\(', diff):
            findings.append(Finding(
                issue_type="use_list_comprehension",
                severity="low",
                description="Consider using list comprehension",
                confidence=0.6
            ))
        
        # 4. Manual file handling (missing context manager)
        if 'open(' in diff and 'with' not in diff:
            findings.append(Finding(
                issue_type="missing_context_manager",
                severity="medium",
                description="Use 'with' statement for file handling",
                confidence=0.8
            ))
        
        # 5. Nested loops (O(n²) complexity)
        if re.search(r'for.*:\s*\n\s+for.*:', diff):
            findings.append(Finding(
                issue_type="nested_loops",
                severity="medium",
                description="Nested loops may cause O(n²) complexity",
                confidence=0.7
            ))
        
        # 6. Global variables
        if re.search(r'^global\s+\w+', diff, re.MULTILINE):
            findings.append(Finding(
                issue_type="global_variable",
                severity="low",
                description="Global variable usage (prefer passing arguments)",
                confidence=0.8
            ))
        
        # 7. os.path vs pathlib
        if 'os.path.join' in diff:
            findings.append(Finding(
                issue_type="use_pathlib",
                severity="low",
                description="Consider using pathlib instead of os.path",
                confidence=0.6
            ))
        
        # 8. Inefficient dictionary operations
        if '.keys()' in diff and 'in' in diff:
            findings.append(Finding(
                issue_type="inefficient_dict_operation",
                severity="low",
                description="Unnecessary .keys() (use 'in dict' directly)",
                confidence=0.7
            ))
        
        return findings


class LLMBugAgent(BugAgent):
    """BugAgent powered by an LLM when available."""

    SYSTEM_PROMPT = (
        "You are BugAgent, a specialized code-review assistant. "
        "Inspect the provided diff for syntax errors, logical bugs, runtime issues, "
        "test failures, and debugging leftovers. Return JSON only."
    )

    def analyze(self, diff: str, context: dict) -> List[Finding]:
        if _llm_backend_preference() in ("llm", "hybrid") and _llm_available():
            try:
                client = LLMClient()
                prompt = (
                    "Diff:\n"
                    f"{diff}\n\n"
                    "Context:\n"
                    f"{json.dumps(context, ensure_ascii=False)}\n\n"
                    "Return JSON with a top-level key 'findings' containing a list of objects. "
                    "Each object must include: issue_type, severity, description, confidence, line_number. "
                    "Only report findings you are reasonably confident about."
                )
                text = client.analyze(self.SYSTEM_PROMPT, prompt)
                findings = _findings_from_llm_text(text)
                if findings:
                    return findings
            except Exception:
                pass
        return super().analyze(diff, context)


class LLMSecurityAgent(SecurityAgent):
    """SecurityAgent powered by an LLM when available."""

    SYSTEM_PROMPT = (
        "You are SecurityAgent, a specialized code-review assistant. "
        "Inspect the provided diff for hardcoded secrets, SQL injection, command injection, unsafe deserialization, "
        "eval/exec usage, and weak cryptography. Return JSON only."
    )

    def analyze(self, diff: str, context: dict) -> List[Finding]:
        if _llm_backend_preference() in ("llm", "hybrid") and _llm_available():
            try:
                client = LLMClient()
                prompt = (
                    "Diff:\n"
                    f"{diff}\n\n"
                    "Context:\n"
                    f"{json.dumps(context, ensure_ascii=False)}\n\n"
                    "Return JSON with a top-level key 'findings' containing a list of objects. "
                    "Each object must include: issue_type, severity, description, confidence, line_number."
                )
                text = client.analyze(self.SYSTEM_PROMPT, prompt)
                findings = _findings_from_llm_text(text)
                if findings:
                    return findings
            except Exception:
                pass
        return super().analyze(diff, context)


class LLMPerformanceAgent(PerformanceAgent):
    """PerformanceAgent powered by an LLM when available."""

    SYSTEM_PROMPT = (
        "You are PerformanceAgent, a specialized code-review assistant. "
        "Inspect the provided diff for inefficient loops, missing context managers, nested loops, global variables, "
        "os.path usage, dictionary inefficiencies, and non-idiomatic Python. Return JSON only."
    )

    def analyze(self, diff: str, context: dict) -> List[Finding]:
        if _llm_backend_preference() in ("llm", "hybrid") and _llm_available():
            try:
                client = LLMClient()
                prompt = (
                    "Diff:\n"
                    f"{diff}\n\n"
                    "Context:\n"
                    f"{json.dumps(context, ensure_ascii=False)}\n\n"
                    "Return JSON with a top-level key 'findings' containing a list of objects. "
                    "Each object must include: issue_type, severity, description, confidence, line_number."
                )
                text = client.analyze(self.SYSTEM_PROMPT, prompt)
                findings = _findings_from_llm_text(text)
                if findings:
                    return findings
            except Exception:
                pass
        return super().analyze(diff, context)


class DebateMechanism:
    """Coordinates agent findings and builds consensus."""
    
    @staticmethod
    def debate(bug_findings: List[Finding], 
               security_findings: List[Finding], 
               performance_findings: List[Finding]) -> Tuple[str, float]:
        """
        Aggregate findings from all agents and determine consensus.
        
        Returns:
            (summary: str, severity_score: float) 
            severity_score: 0.0 (clean) to 1.0 (critical issues)
        """
        all_findings = bug_findings + security_findings + performance_findings
        
        if not all_findings:
            return "All agents agree: PR appears clean; approve is reasonable.", 0.0
        
        # Calculate weighted severity score
        severity_weights = {
            'critical': 1.0,
            'high': 0.7,
            'medium': 0.4,
            'low': 0.2
        }
        
        severity_score = sum(
            severity_weights.get(f.severity, 0) * f.confidence 
            for f in all_findings
        ) / max(len(all_findings), 1)
        
        # Categorize by severity
        critical_count = sum(1 for f in all_findings if f.severity == 'critical')
        high_count = sum(1 for f in all_findings if f.severity == 'high')
        medium_count = sum(1 for f in all_findings if f.severity == 'medium')
        
        # Build consensus summary
        if critical_count > 0:
            summary = f"CRITICAL CONSENSUS: {critical_count} blocking issue(s) detected; reject or request changes immediately."
        elif high_count > 0:
            summary = f"HIGH CONSENSUS: {high_count} serious issue(s) detected; request changes strongly recommended."
        elif medium_count > 2:
            summary = f"MODERATE CONSENSUS: {medium_count} issues detected; request changes or comment for clarification."
        else:
            summary = f"LOW CONSENSUS: {len(all_findings)} minor issue(s); approve with notes or request minor fixes."
        
        return summary, severity_score


def run_multi_agent_analysis(pr: dict) -> Tuple[Dict[str, str], str, float]:
    """
    Run all agents on a PR and generate reports + debate summary.
    
    Args:
        pr: Pull request dict with 'diff_patch' and context
        
    Returns:
        (reports: dict, debate_summary: str, severity_score: float)
    """
    diff = pr.get('diff_patch', '')
    context = {
        'test_results': pr.get('test_results', {}),
        'lint_report': pr.get('lint_report', {}),
        'file_type': pr.get('file_type', ''),
    }
    
    backend = _llm_backend_preference()
    use_llm = backend in ("llm", "hybrid") and _llm_available()

    # Run agents in parallel (LLM-backed when available, heuristic fallback otherwise)
    bug_agent = LLMBugAgent() if use_llm else BugAgent()
    security_agent = LLMSecurityAgent() if use_llm else SecurityAgent()
    performance_agent = LLMPerformanceAgent() if use_llm else PerformanceAgent()
    
    bug_findings = bug_agent.analyze(diff, context)
    security_findings = security_agent.analyze(diff, context)
    performance_findings = performance_agent.analyze(diff, context)
    
    # Generate reports
    reports = {
        'bug': bug_agent.generate_report(bug_findings),
        'security': security_agent.generate_report(security_findings),
        'performance': performance_agent.generate_report(performance_findings),
    }
    
    # Run debate mechanism
    debate_summary, severity_score = DebateMechanism.debate(
        bug_findings, security_findings, performance_findings
    )

    if use_llm:
        debate_summary = "LLM-backed agents reviewed the PR. " + debate_summary
    
    return reports, debate_summary, severity_score
