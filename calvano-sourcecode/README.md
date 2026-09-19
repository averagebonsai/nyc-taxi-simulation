# Calvano baseline Python implementation

This directory ports the `119462-V1/baseline` model needed by the supplied
Figure 4 script: configurable Logit and Singh--Vives payoffs, repeated-game
Q-learning, and a unilateral static-best-response impulse response.

```bash
python calvano-sourcecode/main.py --input 119462-V1/A_InputParametersModified.txt --output-dir results
```

The command writes `results/A_irToBR.txt`. It provides the 15 impulse periods
expected by `119462-V1/figure_4.R`. Do not use the supplied full input as a
smoke test; create the scaled `A_InputParametersModified.txt` first.

Use `--impulse-cycles N` to impose `N` repeated deviations by the same agent.
Each deviation starts a 15-period block: the first period is the forced static
best response, followed by 14 periods in which both agents follow their learned
policies. The default, `N = 1`, is the original Figure 4 behavior.
Each run writes `N_eqm.csv` (or `N_eqm_P_periods.csv` when a cycle has `P != 15`
periods), followed by a PNG rendered from that CSV. The CSV contains only the
pre-deviation point and the final learned-policy price in each cycle, so the
chart can be regenerated without running the simulation again.

It also writes `A_trainingVisits.csv` plus `A_trainingStateVisits.csv`.
With two agents and memory two, a state is ordered as
`(agent_1_t-1, agent_2_t-1, agent_1_t-2, agent_2_t-2)`, giving `15**4 =
50,625` possible states. Each agent therefore has a `50,625 by 15` Q-table.
The state-visit file has one row for each state; the joint-action column sums
to all training iterations across sessions.

Set `Memory:` to `2` in an input file to use this state space. The supplied
`A_InputParametersModified.txt` is configured for two-period memory.

To preserve training for a later impulse-response run, pass an empty archive
directory with `--save-q-tables`:

```bash
python calvano-sourcecode/main.py --input calvano-sourcecode/A_InputParametersModified.txt \
  --output-dir results --save-q-tables results/q_tables
```

This writes one compressed `session_XXXX.npz` file per training session plus a
compatibility manifest. Each file contains the Q-table, its learned policy,
terminal state, and visit counts. Reuse those trained tables without learning
again with:

```bash
python calvano-sourcecode/main.py --input calvano-sourcecode/A_InputParametersModified.txt \
  --output-dir post_training_results --load-q-tables results/q_tables
```

The current input's two-period, 10-price Q-table is about 1.5 MiB per session
before compression. The archive loader rejects incompatible numbers of agents,
prices, memory periods, or price grids.

Each completed session is an atomic checkpoint: `session_0000.npz` is written
only after its seed has trained successfully. If a VM is stopped or one seed
raises an exception, inspect `session_XXXX.error.json` and resume only the
missing seeds using the same input file and seed:

```bash
python calvano-sourcecode/main.py --input calvano-sourcecode/A_InputParametersModified.txt \
  --output-dir results --seed 1 --resume-q-tables results/q_tables
```

`Number of cores` is the number of simultaneous one-core sessions. Numerical
library worker threads are limited to one per process, preventing CPU
oversubscription on a cloud VM.

After training, generate the complete one-, ten-, and fifty-cycle report set
from an archive in one command (this works with either memory setting):

```bash
python calvano-sourcecode/generate_reports.py \
  --input calvano-sourcecode/A_InputParametersModified.txt \
  --q-tables results/q_tables \
  --output-dir results/reports
```

It writes Figure 4-style PDF and PNG files for the one-cycle and ten-cycle
responses, plus the 15- and 20-episode equilibrium-cycle PNGs and source CSVs
for ten and fifty cycles. It does not perform any additional Q-learning.

For a single command that trains a new archive (or resumes an interrupted one)
and then creates the full report set, use:

```bash
python calvano-sourcecode/run_all_reports.py \
  --input calvano-sourcecode/A_InputParametersTwoPeriod.txt \
  --output-dir 119462-V1/results/2-period \
  --seed 1
```

Re-run exactly the same command after a VM interruption. Existing completed
session checkpoints are retained and only missing seeds are trained.

`Number of cores` controls session-level parallelism. The runner creates up to
that many worker processes (capped by the number of sessions and available CPU
cores); results remain deterministic for a fixed `--seed`.
