# Making Real Machines Work: Modelling with States, Actions and Observations

Tutorial materials for the April 23rd 2026 session of the MIT PSFC Machine Learning Working Group.

> [!WARNING]
This repo was made for illustrative/didactic purposes. Most of the code was vibe-coded with Claude Code. Major bugs/issues may be present.

## Setup

You need [uv](https://docs.astral.sh/uv/getting-started/installation/) installed.

```bash
# Clone the repo
git clone <repo-url>
cd mlwg-ssm

# Create a virtual environment and install dependencies
uv sync

```

Then open the notebooks in the `notebooks/` directory.

## Cartpole CLI

```bash
uv run cartpole-frame   # render a single static frame
uv run cartpole-sim     # simulate and animate with zero force
```
