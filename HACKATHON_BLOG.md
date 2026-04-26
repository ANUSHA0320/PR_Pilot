# PR Pilot: Training Evidence and Demo Story

PR Pilot is a reinforcement learning environment for automated pull-request review.
It turns code review into a measurable task where an agent must inspect diffs,
coordinate specialist reviewers, and choose a final review action.

## Why this project matters

Manual code review is valuable, but it is slow and inconsistent.
PR Pilot explores whether a structured review environment can help an agent learn
to identify bugs, security issues, and code-quality problems more reliably.

The project is designed as a research-style benchmark rather than a simple UI demo.
Each episode gives the agent partial information, specialist agent feedback,
and delayed reward based on review quality.

## What the system does

- Loads a pull request diff from a dataset or GitHub repository
- Runs specialist reviewers:
  - BugAgent
  - SecurityAgent
  - PerformanceAgent
- Produces a debate summary and author reply
- Lets the reviewer choose among approve, reject, request changes,
  comment bug, or suggest patch
- Scores the decision with shaped rewards

## Training evidence

The notebook training run was executed in Google Colab and produced artifacts
that can be imported into the repository for submission evidence.

Committed or generated artifacts:

- results/TrainingProgress_Reward.png
- results/TrainingAgent_RandomPolicy.png
- results/training_summary.txt

Latest saved summary:

- Episodes: 100
- Average Reward: 0.052
- Std Dev: 0.513
- Range: [-0.300, 0.800]
- Final 5 episodes avg: 0.360

## Demo highlights

- Gradio UI for interactive review
- Multi-PR navigation from a GitHub repo
- Isolated browser sessions for concurrent use
- Leaderboard-style episode tracking

## How the Colab results are imported

If the notebook cannot be executed in the local environment, the Colab outputs
can still be brought into the repo by copying the generated files into the
results/ directory and keeping the summary visible in the notebook and README.

That gives judges both the live demo and the training evidence in one place.

## Final takeaway

PR Pilot is a compact but complete research prototype:

- a working environment
- a training pipeline
- evidence of reward behavior
- and an interactive review demo

The main goal is to show that code review can be modeled, trained, and evaluated
as a learning problem.