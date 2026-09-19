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

`Number of cores` controls session-level parallelism. The runner creates up to
that many worker processes (capped by the number of sessions and available CPU
cores); results remain deterministic for a fixed `--seed`.
