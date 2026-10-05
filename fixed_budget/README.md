# fixed_budget/

Fixed-budget study: stochastic binary neural network (SBNN) inference under
autocorrelated sMTJ randomness: what happens to classification accuracy when
p-bit devices are sampled faster than their correlation time, and what
recovers the loss. The adaptive-stopping study lives in the repository root.

## Layout
- `src/smtjnn/`: model (SBNN + STE), telegraph sMTJ source, protocols, training
- `tests/`: physics + protocol validation (pytest)
- `experiments/`: run_v2.py (all phases), analysis / figure scripts
- `results/v2/`: one JSON per (config, seed), stamped with its git SHA
- `results/*.pt`: trained checkpoints

## Reproduce (run inside fixed_budget/)
    pip install -e ".[dev]"
    python -m pytest tests/
    python experiments/run_v2.py --phase 0   # train (val split; test untouched)
    python experiments/run_v2.py --phase 1   # ideal-randomness anchors
    python experiments/run_v2.py --phase 2   # sampling-ratio sweep
    python experiments/run_v2.py --phase 3   # device variability
    python experiments/run_v2.py --phase 4   # correlation-aware training
    python experiments/run_v2.py --phase 5   # lane-count invariance
    python experiments/run_v2.py --phase 2 --dataset fashion

MNIST and Fashion-MNIST are downloaded by torchvision into `data/` on first
use. CPU-only; a few hours on a laptop. Git SHAs in the result records refer
to the private development history, available to editors and reviewers on
request.
