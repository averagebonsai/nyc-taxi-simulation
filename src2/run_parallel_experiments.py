"""Run modular Courthoud experiments across deviators and seeds in parallel."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

COURTHOUD_MODULE = "src2.run_courthoud_experiment"


def run_single_experiment(module: str, deviator: int, seed: int, output_dir: Path, forwarded_args: list[str]) -> tuple[int, int, float, bool]:
    """Run one subprocess and persist its combined output beside its artifacts."""
    run_dir = output_dir / f"deviator_{deviator}_seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "simulation.log"
    command = [sys.executable, "-u", "-m", module, "--deviator", str(deviator), "--seed", str(seed), "--out-dir", str(run_dir), *forwarded_args]
    started = time.monotonic()
    with log_path.open("w") as log_file:
        completed = subprocess.run(command, stdout=log_file, stderr=subprocess.STDOUT, text=True, check=False)
    return deviator, seed, time.monotonic() - started, completed.returncode == 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Run modular Courthoud freeze/unfreeze experiments in parallel.")
    parser.add_argument("--deviators", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--cores", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/parallel_runs"))
    args, forwarded_args = parser.parse_known_args()
    tasks = [(COURTHOUD_MODULE, deviator, seed, args.out_dir, forwarded_args) for seed in args.seeds for deviator in args.deviators]
    print(f"Starting {len(tasks)} run(s) on {args.cores or 'all available'} core(s).")
    failures: list[tuple[int, int]] = []

    def report(result: tuple[int, int, float, bool]) -> None:
        deviator, seed, elapsed, succeeded = result
        label = "completed" if succeeded else "failed"
        print(f"{label}: deviator={deviator}, seed={seed}, elapsed={elapsed:.1f}s")
        if not succeeded:
            failures.append((deviator, seed))

    if args.cores == 1:
        for task in tasks:
            report(run_single_experiment(*task))
    else:
        with ProcessPoolExecutor(max_workers=args.cores) as executor:
            futures = [executor.submit(run_single_experiment, *task) for task in tasks]
            for future in as_completed(futures):
                report(future.result())
    if failures:
        raise SystemExit(f"Failed runs: {failures}")


if __name__ == "__main__":
    main()
