# PR Pilot: Teaching AI to Review Code Through Multi-Agent Collaboration

**OpenEnv Hackathon 2026 Submission** | [GitHub](https://github.com/ANUSHA0320/CodeReviewEnv) | [Live Demo](https://huggingface.co/spaces/anu1720/code-review-gym) | [Training Notebook](https://colab.research.google.com/github/ANUSHA0320/CodeReviewEnv/blob/main/training_trl_colab.ipynb)

---

## 🎯 The Problem: Code Review is Hard for AI

Code review requires understanding bugs, security risks, AND performance issues simultaneously. Single-agent LLMs struggle because they:
- Miss context-specific vulnerabilities (e.g., SQL injection in one file, hardcoded secrets in another)
- Can't weigh trade-offs (security vs. readability)
- Provide generic feedback instead of actionable suggestions

**The idea**: What if we trained a *team of specialist agents* to collaborate like human reviewers do?

---

## 🏗️ The Solution: Multi-Agent RL Environment

**PR Pilot** is a Gymnasium-compatible environment where three specialist agents analyze pull requests, debate findings, and train a lead reviewer to make optimal decisions.

### Architecture

```
┌─────────────────┐
│  Pull Request   │  
│   (Diff + Tests)│
└────────┬────────┘
         │
    ┌────▼─────┐
    │ 3 Agents │ ◄── Parallel Analysis
    └────┬─────┘
         │
    ┌────▼──────────┐
    │ Debate Phase  │ ◄── Structured Discussion
    └────┬──────────┘
         │
    ┌────▼──────────┐
    │ Lead Reviewer │ ◄── RL-Trained Decision Maker
    └───────────────┘
```

**1. BugAgent**: Detects syntax errors, unused variables, type mismatches  
**2. SecurityAgent**: Identifies hardcoded secrets, SQL injection, XSS risks  
**3. PerformanceAgent**: Spots inefficient loops, memory leaks, N+1 queries  

**4. LeadReviewer** (RL agent): Synthesizes findings → approve/reject/request_changes

---

## 🎓 What Makes This Unique?

### 1. Multi-Agent Debate Simulation
Agents don't just report findings—they *debate* them:

**Example Conversation:**
```
SecurityAgent: "Found hardcoded API key on line 42 - CRITICAL"
PerformanceAgent: "Inefficient loop detected but not blocking"
BugAgent: "No syntax errors - code runs fine"
→ Debate consensus: Security issue is blocking → reject PR
```

### 2. Reviewer-Author Dialogue Loop
When the reviewer requests changes, a simulated author replies:
```
Reviewer: "request_changes" (SQL injection risk)
Author: "I've parameterized the query, please re-review"
Reviewer: Must decide whether fix is adequate → approve/escalate
```

### 3. Self-Improving Curriculum
The environment adapts based on agent performance:
- **Easy**: Detect obvious bugs (hardcoded secrets, unused imports)
- **Medium**: Logical bugs (off-by-one errors, null pointer risks)
- **Hard**: Suggest Pythonic refactors (context managers, comprehensions)

---

## 📊 Training Results

We trained a **distilgpt2** model (82M parameters) using **policy gradient** on 30 episodes:

| Metric | Value |
|--------|-------|
| **Average Reward** | 0.140 |
| **Final 5 Episodes** | 0.360 (▲ +156% improvement) |
| **Range** | -0.3 to +0.8 |

### Reward Curves

![Training Progress](results/TrainingProgress_RewardDistribution.png)

**Left**: Episode rewards with moving average (clear upward trend)  
**Right**: Reward distribution (agent learned to avoid negative rewards)

### Trained vs Random Baseline

![Baseline Comparison](results/TrainingAgent_RandomPolicy.png)

**Random policy** (untrained): ~0.0 avg reward  
**Trained agent**: 0.140 avg reward → **agent learned meaningful strategies**

---

## 🔍 What Did the Agent Learn?

### Before Training (Random Actions)
```
Observation: Diff with hardcoded API key + test failures
Action: approve ❌ (wrong - should reject)
Reward: -0.30 (penalty for approving buggy code)
```

### After Training (Learned Policy)
```
Observation: Same scenario
Action: reject ✓ (correct - security issue is blocking)
Reward: +0.30 (bonus for correct review decision)
```

**Key insight**: The agent learned to:
1. Recognize when test failures indicate real bugs (not flaky tests)
2. Prioritize security issues over performance suggestions
3. Request changes instead of outright rejection when bugs are fixable

---

## 🛠️ Technical Implementation

### Observation Space (15 fields)
```python
{
  "diff_patch":          "...",               # Code changes
  "agent_reports":       {                    # Specialist findings
    "bug":         "Unused variable x",
    "security":    "Hardcoded API key",
    "performance": "O(n²) loop"
  },
  "debate_summary":      "Security blocker",  # Consensus
  "test_results":        {"tests_passed": 0}, # CI status
  "conversation_history": "...",              # Iterative feedback
  "iteration":           2                    # Review round
}
```

### Action Space (5 actions)
```python
0 = approve           # Merge PR (terminal)
1 = reject            # Block PR (terminal)
2 = request_changes   # Ask for revision (non-terminal)
3 = comment_bug       # Annotate specific issue
4 = suggest_patch     # Propose improved code
```

### Reward Shaping
```python
+0.40  Correct bug detection
+0.30  Correct review decision (approve/reject at right time)
+0.20  Useful code suggestion
-0.30  Approve buggy code (critical error)
-0.20  Reject clean code (false positive)
```

---

## 🚀 Try It Yourself

### 1. Quick Start (Google Colab)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ANUSHA0320/CodeReviewEnv/blob/main/training_trl_colab.ipynb)

**5-step process** (~15 minutes with free T4 GPU):
1. Click Colab badge → Runtime → T4 GPU
2. Run all cells
3. Watch training curves update in real-time
4. Download results from `results/` folder

### 2. Live Demo (HuggingFace Space)
🌐 **[Try the interactive Gradio UI](https://huggingface.co/spaces/anu1720/code-review-gym)**

Paste a code diff → See multi-agent analysis → Watch debate → Get review decision

### 3. Install Locally
```bash
git clone https://github.com/ANUSHA0320/CodeReviewEnv.git
cd CodeReviewEnv
pip install -e .
python -m code_review_env  # Starts Gradio UI
```

---

## 🎬 What's Next?

### Short-term
- [ ] Add more specialist agents (TypeCheckerAgent, DocumentationAgent)
- [ ] Train on real GitHub PR data (currently synthetic)
- [ ] Implement multi-turn dialogue (full back-and-forth negotiation)

### Long-term Vision
- **Production deployment**: GitHub App that auto-reviews PRs
- **Human-in-the-loop**: Reviewers override agent decisions → collect feedback for retraining
- **Cross-repo learning**: Train on multiple codebases to generalize

---

## 📚 Resources

- **GitHub**: [ANUSHA0320/CodeReviewEnv](https://github.com/ANUSHA0320/CodeReviewEnv)
- **Live Demo**: [HuggingFace Space](https://huggingface.co/spaces/anu1720/code-review-gym)
- **Training Notebook**: [Colab](https://colab.research.google.com/github/ANUSHA0320/CodeReviewEnv/blob/main/training_trl_colab.ipynb)
- **Documentation**: [Full README](https://github.com/ANUSHA0320/CodeReviewEnv/blob/main/README.md)

---

## 🙋 Why This Matters

**Code review is a bottleneck** in software development. This environment:
1. **Bridges research → production**: First RL environment for multi-agent code review
2. **Demonstrates real learning**: Not just static analysis—agent improves with training
3. **Opens new research directions**: Multi-agent collaboration, curriculum learning, human feedback integration

**Built for OpenEnv Hackathon 2026** | Multi-Agent Collaboration + Self-Improving Systems

---

**Questions? Suggestions?** Open an issue on [GitHub](https://github.com/ANUSHA0320/CodeReviewEnv/issues) or tag me on X [@<your-handle>]
