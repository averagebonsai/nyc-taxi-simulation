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

`Number of cores` controls session-level parallelism. The runner creates up to
that many worker processes (capped by the number of sessions and available CPU
cores); results remain deterministic for a fixed `--seed`.
