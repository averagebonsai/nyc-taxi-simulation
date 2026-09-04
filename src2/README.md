# Modular Dynamic-Pricing Pipeline

`src2` is the supported replacement for the legacy scripts in `src/`. It has one source of truth for pricing actions: `(0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.2, 2.5)`.

The current prepared `data/` artifacts contain 262 pickup zones.  Zone-level
Q-learning keeps a separate table for each zone, with default shape
`(262, 10, 8)`: 262 zones, 10 discretized taxi-inventory states, and eight
surge-multiplier actions. The zone count is loaded from the prepared
predictor rather than hard-coded by the simulation.

## Files

- `market/` — raw-trip preparation, runtime assets, market configuration, and the shared taxi-market environment.
- `policies/base.py` — common observation, action, transition, and policy lifecycle contracts.
- `policies/algorithms.py` — reusable zone-level Q-learning and EXP3 agents, including frozen EXP3 snapshots.
- `experiments/rollout.py` — the shared episode loop used by every market experiment.
- `experiments/` — monopoly/oligopoly runners, Courthoud intervention workflow, configurations, and reporting.
- `run_experiments.py` — supported CLI.
- `run_courthoud_experiment.py` and `run_parallel_experiments.py` — Courthoud experiment CLIs.

Run a prepared-data experiment from the repository root:

```bash
python -m src2.run_experiments --episodes 70 --steps 120
```

Build assets from the raw trip parquet before an initial run or after changing demand assumptions:

```bash
python -m src2.run_experiments --prepare-assets --theta 0.4
```

Run a Courthoud experiment with four firms. Use `--reset-frozen` to erase the
three frozen agents' memory immediately before unfreezing:

```bash
python -m src2.run_courthoud_experiment --episodes 100 --freeze-episodes 70
python -m src2.run_courthoud_experiment --episodes 100 --freeze-episodes 70 --reset-frozen
```

During the freeze phase, platforms retain a sampled snapshot of their learned
EXP3 action probabilities. They still choose stochastic prices, but their
weights are not updated until the unfreeze phase.

Use `python -m src2.run_parallel_experiments --deviators 0 1 2 3 --seeds 42 43`
to run the same modular CLI across several seeds; extra Courthoud options are
forwarded to every subprocess.
