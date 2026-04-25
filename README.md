---
title: PR Pilot
emoji: 🚀
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
license: mit
short_description: Multi-Agent AI Code Review with RL Training
tags:
  - reinforcement-learning
  - code-review
  - multi-agent
  - trl
  - ppo
---

# PR Pilot · Multi-Agent Code Review Environment

[![Tests](https://github.com/ANUSHA0320/CodeReviewEnv/actions/workflows/tests.yml/badge.svg)](https://github.com/ANUSHA0320/CodeReviewEnv/actions/workflows/tests.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Gymnasium](https://img.shields.io/badge/gymnasium-1.x-green.svg)](https://gymnasium.farama.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ANUSHA0320/CodeReviewEnv/blob/main/training_trl_colab.ipynb)

> **OpenEnv Hackathon 2026 Submission**: Multi-Agent Collaboration + Self-Improving Systems

**PR Pilot** is a Gymnasium-compatible RL environment where AI agents learn to review code through multi-agent collaboration. Three specialist agents (Bug, Security, Performance) analyze PRs, debate findings, and train a lead reviewer to make optimal decisions.

---

## 🎯 Key Features

### Multi-Agent Architecture
- **BugAgent**: Detects syntax errors, unused variables, type issues
- **SecurityAgent**: Identifies hardcoded secrets, SQL injection risks
- **PerformanceAgent**: Spots inefficient algorithms, memory leaks
- **LeadReviewer**: Synthesizes findings and makes final decision

### Collaborative Workflow
1. Each specialist agent analyzes the PR independently
2. Agents debate their findings (structured discussion)
3. Consensus reached through weighted voting
4. LeadReviewer trained via PPO to optimize decisions

### Self-Improving System
- **Adaptive Curriculum**: Difficulty scales based on performance (easy → medium → hard)
- **Reviewer-Author Dialogue**: Iterative negotiation loop for PR improvements
- **Reward Shaping**: Multi-objective optimization (correctness, false positives, review quality)

---

## 📂 Project Structure

```
PR-Pilot/
├── code_review_env/         ← Core Gymnasium environment
│   ├── env.py               ← Multi-agent orchestration & PPO integration
│   ├── state.py             ← Agent reports, debate, dialogue tracking
│   ├── actions.py           ← Review actions (approve/reject/comment)
│   └── reward.py            ← Shaped reward with negotiation bonuses
├── tasks/                   ← Difficulty-specific evaluation logic
├── graders/                 ← Task graders for scoring
├── data/                    ← 24 GitHub PRs (8 easy, 8 medium, 8 hard)
├── baseline/                ← Heuristic + LLM baselines
├── app/                     ← FastAPI REST API
├── gradio_app.py            ← Interactive demo UI
├── training_trl_colab.ipynb ← TRL PPO training notebook
├── results/                 ← Training evidence (plots, metrics)
├── tests/                   ← 74 pytest tests
└── openenv.yaml             ← OpenEnv manifest
```

---

## 🚀 Quick Start

### 1. Install

```bash
git clone https://github.com/ANUSHA0320/CodeReviewEnv
cd CodeReviewEnv
pip install -r requirements.txt
pip install -e .  # editable install
```

### 2. Try the Demo (Gradio UI)

```bash
python gradio_app.py
# Open: http://localhost:7860
```

**Features**:
- Load GitHub PRs directly from URLs
- View multi-agent analysis (Bug/Security/Performance)
- See debate summary and recommendations
- Test different review scenarios

### 3. Train an Agent (Google Colab)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ANUSHA0320/CodeReviewEnv/blob/main/training_trl_colab.ipynb)

**Steps**:
1. Click button above (opens notebook in Colab)
2. Runtime → Change runtime type → T4 GPU
3. Runtime → Run all (~15 minutes)
4. Download training plots from `results/` folder

See [COLAB_INSTRUCTIONS.md](COLAB_INSTRUCTIONS.md) for details.

### 4. Use the Environment Directly

```python
import gymnasium as gym
import code_review_env   # registers CodeReviewEnv-v0

env = gym.make("CodeReviewEnv-v0", difficulty="easy")
obs, info = env.reset()

done = False
while not done:
    action = env.action_space.sample()   # replace with your agent
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    env.render()

print("Score:", info["task_score"])
env.close()
```

---

## Difficulties

| Difficulty | Objective | Target Score |
|------------|-----------|:------------:|
| easy | Detect surface bugs (hardcoded secrets, syntax errors, unused vars) | 0.80 |
| medium | Detect logical bugs (off-by-one, SQL injection, null dereference) | 0.60 |
| hard | Suggest code-quality patches (Pythonic rewrites, context managers) | 0.40 |

---

## 📖 Storytelling & Presentation

**OpenEnv Hackathon 2026 Submission**

Choose your format:

### 📝 Blog Post
📄 **[Read the full story on HuggingFace](TBD)**  
*Alternative*: [Local blog post](HACKATHON_BLOG.md) (ready to publish)

### 🎬 Video (< 2 minutes)
🎥 **[Watch on YouTube](TBD)**  
*Alternative*: [Video script](HACKATHON_VIDEO_SCRIPT.md) (ready to record)

### 📊 Slide Deck
📑 **[View slides on Google Slides](TBD)**  
*Alternative*: [Slide deck markdown](HACKATHON_SLIDES.md) (ready to present)

**What's covered:**
- Problem: Why code review is hard for single-agent AI
- Solution: Multi-agent collaboration with debate simulation
- Results: Training curves showing 156% improvement (0.0 → 0.36 final avg)
- Demo: Live HuggingFace Space + Colab training notebook
- Impact: First RL environment for multi-agent code review

---

## Observation Space

```python
spaces.Dict({
    "diff_patch":          spaces.Text(min_length=0, max_length=4096, charset=string.printable),
    "repository_context":  spaces.Text(min_length=0, max_length=512,  charset=string.printable),
    "test_results":        spaces.Dict({"tests_passed":    spaces.Discrete(2)}),  # 0=fail 1=pass
    "lint_report":         spaces.Dict({"unused_variable": spaces.Discrete(2)}),  # 0=clean 1=warn
    "file_type":           spaces.Text(min_length=1, max_length=64,   charset=string.printable),
  "agent_reports":        spaces.Dict({
    "bug":         spaces.Text(min_length=0, max_length=512, charset=string.printable),
    "security":    spaces.Text(min_length=0, max_length=512, charset=string.printable),
    "performance": spaces.Text(min_length=0, max_length=512, charset=string.printable),
  }),
  "debate_summary":       spaces.Text(min_length=0, max_length=512,  charset=string.printable),
  "conversation_history": spaces.Text(min_length=0, max_length=2048, charset=string.printable),
  "author_reply":         spaces.Text(min_length=0, max_length=512,  charset=string.printable),
  "iteration":            spaces.Discrete(6),
})
```

---

## Action Space

```python
spaces.Discrete(5)
```

| Action | Label | Terminal? | When to use |
|:------:|-------|:---------:|-------------|
| 0 | `approve` | ✅ Yes | PR looks clean — no issues found |
| 1 | `reject` | ✅ Yes | PR has a critical bug that blocks merge |
| 2 | `request_changes` | No | Issues found but fixable — ask for revision |
| 3 | `comment_bug` | No | Annotate a specific bug in the diff |
| 4 | `suggest_patch` | No | Propose an improved version of the code |

---

## Reward Shaping

| Event | Reward |
|-------|-------:|
| Correct bug detection | +0.40 |
| Correct review decision | +0.30 |
| Useful code suggestion | +0.20 |
| Request changes on buggy PR | +0.10 |
| Approve buggy code | −0.30 |
| Reject clean code | −0.20 |
| Terminal bonus (task score × 0.5) | +0–0.50 |

The reward also includes small bonuses for reviewer decisions made after
AuthorAgent feedback during request-changes negotiation.

---

## Multi-Agent Flow

1. PR arrives
2. BugAgent analyzes
3. SecurityAgent analyzes
4. PerformanceAgent analyzes
5. Agents debate
6. LeadReviewer decides

The environment logs a conversation history and the AuthorAgent reply so
training scripts can learn from negotiation dynamics.

---

## Adaptive Curriculum

Use `difficulty="adaptive"` to enable self-improving difficulty sampling.
The environment promotes from easy → medium → hard as recent scores improve.

---

## REST API

A FastAPI server exposes the environment over HTTP with a Swagger UI:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 7860
# Open: http://localhost:7860/docs
```

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Welcome page |
| `/health` | GET | Health check |
| `/reset` | GET | Start new episode (`?difficulty=easy`) |
| `/step` | POST | Submit action `{"action": 3, "difficulty": "easy"}` |
| `/render` | GET | Current state summary |
| `/scores` | GET | Leaderboard of completed episodes |
| `/actions` | GET | List all actions |
| `/docs` | GET | Interactive Swagger UI |

---

## Docker

```bash
# Build
docker build -t code-review-env .

# Run Gradio UI on port 7860 (default)
docker run -p 7860:7860 code-review-env

# Run FastAPI REST server instead
docker run -p 7860:7860 code-review-env \
  uvicorn app.main:app --host 0.0.0.0 --port 7860

# Run heuristic baseline only
docker run code-review-env \
  python baseline/run_agent.py --no-llm --episodes 5
```

---

## Gradio UI

The interactive Gradio app (`gradio_app.py`) ships with three tabs:

| Tab | Description |
|-----|-------------|
| **▶ Play** | Load a PR, choose an action via buttons, see live reward + score feedback |
| **🏆 Leaderboard** | Table of all completed episodes with per-difficulty averages |
| **ℹ️ About** | Full API reference, observation/action space docs, quick-start guide |

The **Auto Demo** button runs a full heuristic episode automatically so evaluators can see a complete trajectory without manual input.

---

## 📊 Training Results with TRL PPO

### Setup & Configuration
- **Algorithm**: Proximal Policy Optimization (PPO) via Hugging Face TRL
- **Model**: distilgpt2 with value head (82M parameters)
- **Environment**: PR Pilot (CodeReviewEnv-v0), easy difficulty  
- **Episodes**: 30+ for demonstration, 100+ recommended for convergence

### Key Metrics (Latest Colab Run):
| Metric | Value |
|--------|------:|
| Average Reward | 0.140 |
| Reward Range | [-0.300, 0.800] |
| Final 5 Episodes Avg | 0.360 |
| Improvement vs Baseline | **+156%** (0.0 → 0.360 final avg) |
| Standard Deviation | 0.539 |
| Training Time | ~15 min (T4 GPU) |

**Evidence**: [Training Progress](results/TrainingProgress_RewardDistribution.png) | [Baseline Comparison](results/TrainingAgent_RandomPolicy.png) | [Summary](results/training_summary.txt)

### What the Agent Learns:
1. ✅ **Bug Detection**: Syntax errors, unused variables, type mismatches
2. ✅ **Security Analysis**: Hardcoded secrets, SQL injection patterns
3. ✅ **Review Strategy**: Optimal balance of approve/reject/request changes
4. ✅ **Multi-Agent Synthesis**: Integrate Bug/Security/Performance findings

### Training Resources:
- 📓 **Notebook**: [training_trl_colab.ipynb](training_trl_colab.ipynb)
- 📊 **Results**: [results/](results/) folder (plots + metrics)
- 📖 **Guide**: [COLAB_INSTRUCTIONS.md](COLAB_INSTRUCTIONS.md)

> **Note**: For full TRL training, run notebook on Google Colab (free T4 GPU). Local runs use demo mode as fallback due to SSL restrictions.

---

## 🏆 OpenEnv Hackathon 2026 Alignment

### Theme: Multi-Agent Collaboration ✅
- **Specialist Agents**: BugAgent, SecurityAgent, PerformanceAgent analyze independently
- **Debate Mechanism**: Structured discussion reaches consensus
- **Synthesis**: LeadReviewer integrates findings via weighted voting

### Theme: Self-Improving Systems ✅  
- **Adaptive Curriculum**: Auto-adjusts difficulty (easy → medium → hard)
- **Dialogue Loop**: Reviewer-Author negotiation refines PRs iteratively
- **Reward Shaping**: Multi-objective optimization drives exploration

### Innovation Highlights:
- 🥇 **First multi-agent** RL environment for code review with debate
- 🌍 **Real-world data**: 24 GitHub PRs (Python/JavaScript/Java)
- 🤖 **LLM-ready**: Rich textual observations for transformer policies
- 🚀 **Production-grade**: FastAPI + Gradio UI + 74 tests + CI/CD

---

## Measured Baseline Scores

Scores measured over 8 episodes (full dataset) with `--seed 42`:

| Agent | Easy | Medium | Hard |
|-------|:----:|:------:|:----:|
| HeuristicAgent (no LLM) | 1.000 | 1.000 | 0.580 |
| GPT-3.5-turbo | ~0.90 | ~0.75 | ~0.60 |

> Hard score for HeuristicAgent reflects patch similarity from observable diff lines only (no access to reference patch).

---

## 🔧 API & Integration

The repository ships with a GitHub Actions workflow (`.github/workflows/tests.yml`) that runs on every push and pull request:

- **Matrix:** Python 3.10, 3.11, 3.12 on `ubuntu-latest`
- **Steps:** install deps → `pytest` with coverage → `check_env` on all difficulties → heuristic baseline smoke test
🔧 API & Integration

### FastAPI REST Server

Start the server:
```bash
uvicorn app.main:app --reload
# Open: http://localhost:8000/docs
```

**Endpoints**:
- `POST /reset` - Start new episode, returns observation
- `POST /step` - Submit action, returns reward + done flag
- `POST /render` - Get current state visualization

### 
---

## Gymnasium Compliance

The environment passes `gymnasium.utils.env_checker.check_env` on all three difficulties:

```python
from gymnasium.utils.env_checker import check_env
import code_review_env
from code_review_env.env import CodeReviewEnv

for difficulty in ("easy", "medium", "hard"):
    env = CodeReviewEnv(difficulty=difficulty)
    check_env(env)   # no errors
    env.close()
    print(f"check_env({difficulty}) PASSED")
```

---

## 🚀 Deployment

### Hugging Face Spaces

This repo is configured for [Hugging Face Spaces](https://huggingface.co/spaces/anu1720/PR_Pilot):

```
🌐 Live Demo: https://huggingface.co/spaces/anu1720/PR_Pilot
```

**Docker SDK** deployment (automatically builds and serves on port 7860):

```bash
git remote add hf https://huggingface.co/spaces/<your-user>/pr-pilot
git push hf main
```

### Local Docker

```bash
docker build -t pr-pilot .
docker run -p 7860:7860 pr-pilot  # Gradio UI
```

---

## 📚 Documentation

- 📓 **Training Guide**: [COLAB_INSTRUCTIONS.md](COLAB_INSTRUCTIONS.md)
- ⚡ **Quick Start**: [QUICK_START_COLAB.md](QUICK_START_COLAB.md)  
- 📊 **Results**: [results/README.md](results/README.md)
- 🔧 **OpenEnv Manifest**: [openenv.yaml](openenv.yaml)

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **OpenEnv Hackathon 2026** for the challenge themes
- **Hugging Face TRL** for PPO training framework
- **Gymnasium** for RL environment standardization
- **GitHub** for real PR data inspiration

---

**Built with ❤️ for the OpenEnv Hackathon 2026**
