# PR Pilot - Slide Deck
**OpenEnv Hackathon 2026 Submission**

---

## Slide 1: The Problem

### Code Review is Hard for AI 🤖

**Current LLM limitations:**
- Miss context-specific bugs
- Can't weigh security vs. performance trade-offs
- Provide generic feedback

**Our solution:**  
Train a *team of specialist agents* to collaborate like human reviewers

---

## Slide 2: The Architecture

```
Pull Request (Diff + Tests)
         │
    ┌────▼─────────────────┐
    │   3 Specialist Agents │
    │  ┌────────────────┐   │
    │  │ 🐛 BugAgent    │   │
    │  │ 🔒 Security    │   │
    │  │ ⚡ Performance │   │
    │  └────────────────┘   │
    └────┬──────────────────┘
         │
    ┌────▼──────────────┐
    │  Debate Phase     │ ◄── Structured Discussion
    └────┬──────────────┘
         │
    ┌────▼──────────────┐
    │  Lead Reviewer    │ ◄── RL-Trained (Policy Gradient)
    └───────────────────┘
         │
    Approve / Reject / Request Changes
```

---

## Slide 3: What Makes It Unique?

### 🎯 Multi-Agent Debate

**Example:**
- SecurityAgent: "Hardcoded API key - CRITICAL"
- PerformanceAgent: "Inefficient loop - not blocking"
- BugAgent: "No syntax errors"
- **→ Consensus: Security issue is blocking → reject PR**

### 🔄 Reviewer-Author Dialogue

```
Reviewer: "request_changes" (SQL injection)
Author: "Parameterized query, please re-review"
Reviewer: Decides if fix is adequate
```

### 📈 Adaptive Curriculum

Easy → Medium → Hard (based on performance)

---

## Slide 4: Training Results

| Metric | Value |
|--------|-------|
| **Model** | distilgpt2 (82M params) |
| **Episodes** | 30 |
| **Avg Reward** | 0.140 |
| **Final 5 Avg** | 0.360 (+156% improvement) |

### Key Learning:
- ✅ Learned to prioritize security over performance
- ✅ Recognize when test failures = real bugs
- ✅ Request changes instead of outright rejection

---

## Slide 5: Reward Curves

![Training Progress](results/TrainingProgress_RewardDistribution.png)

**Left**: Clear upward trend in episodes  
**Right**: Agent learned to avoid negative rewards

---

## Slide 6: Baseline Comparison

![Trained vs Random](results/TrainingAgent_RandomPolicy.png)

**Random policy**: ~0.0 avg reward  
**Trained agent**: 0.140 → **Measurable improvement**

---

## Slide 7: Technical Details

### Observation Space (15 fields)
- Diff patch, agent reports, debate summary
- Test results, lint reports, conversation history

### Action Space (5 actions)
- approve, reject, request_changes, comment_bug, suggest_patch

### Reward Shaping
- +0.40 correct bug detection
- +0.30 correct decision
- -0.30 approve buggy code (penalty)

---

## Slide 8: Demo

### 🚀 Try It:

🌐 **Live Demo**: https://huggingface.co/spaces/anu1720/code-review-gym

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ANUSHA0320/CodeReviewEnv/blob/main/training_trl_colab.ipynb)

**5-min training**: Free T4 GPU on Colab

---

## Slide 9: Why It Matters

### Code review is a bottleneck in software development

**This environment:**
1. First RL environment for multi-agent code review
2. Demonstrates *real learning* (not just static analysis)
3. Opens research: multi-agent collab + human feedback

**Production vision**: GitHub App that auto-reviews PRs

---

## Slide 10: Thank You!

### PR Pilot - Multi-Agent Code Review Environment

📂 **GitHub**: github.com/ANUSHA0320/CodeReviewEnv  
🌐 **HF Space**: huggingface.co/spaces/anu1720/code-review-gym  
📓 **Colab**: [Training Notebook]  

**Built for OpenEnv Hackathon 2026**  
*Multi-Agent Collaboration + Self-Improving Systems*

---

**Questions?** Open an issue on GitHub!
