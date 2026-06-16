"""Turn results.json into a markdown report.

Usage:
    uv run python -m experiments.filtered_bc.make_report \\
        --results experiments/filtered_bc/results.json \\
        --out experiments/filtered_bc/overnight_report.md
"""

from __future__ import annotations

import dataclasses
import json
import pathlib

import tyro


@dataclasses.dataclass
class Args:
    results: str = "experiments/filtered_bc/results.json"
    out: str = "experiments/filtered_bc/overnight_report.md"


def _fmt_pct(x: float | None) -> str:
    if x is None:
        return "-"
    return f"{100 * x:.0f}%"


def _fmt_time(x: float | None) -> str:
    if x is None:
        return "-"
    if x >= 60:
        return f"{x / 60:.1f}m"
    return f"{x:.0f}s"


def main(args: Args) -> None:
    path = pathlib.Path(args.results)
    if not path.exists():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text())

    lines: list[str] = []
    lines.append("# Overnight run: filtered-BC on MetaWorld ML45 — results")
    lines.append("")
    lines.append(f"Started:  {data.get('started_at', '?')}")
    lines.append(f"Finished: {data.get('finished_at', '(in progress)')}")
    lines.append(f"Updated:  {data.get('last_updated', '?')}")
    lines.append("")
    lines.append("## Config")
    lines.append("```json")
    lines.append(json.dumps(data.get("args", {}), indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## Per-task results")
    lines.append("")
    lines.append(
        "| task | n_train_samples | rollout_success | eval_success (n/n) | eval% | t_rollout | t_train | t_eval | status |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    tasks = data.get("tasks", {})
    ok_rates: list[float] = []
    for task, rec in tasks.items():
        status = rec.get("status", "?")
        if status == "ok":
            eval_rate = rec.get("eval_success_rate")
            lines.append(
                "| {task} | {nsamp} | {rs} | {succ}/{n} | {er} | {tr} | {tt} | {te} | {st} |".format(
                    task=task,
                    nsamp=rec.get("num_train_samples", "-"),
                    rs=_fmt_pct(rec.get("rollout_success_rate")),
                    succ=rec.get("eval_num_success", "-"),
                    n=rec.get("eval_num_episodes", "-"),
                    er=_fmt_pct(eval_rate),
                    tr=_fmt_time(rec.get("t_rollout_sec")),
                    tt=_fmt_time(rec.get("t_train_sec")),
                    te=_fmt_time(rec.get("t_eval_sec")),
                    st=status,
                )
            )
            if eval_rate is not None:
                ok_rates.append(eval_rate)
        elif status == "skipped_no_successes":
            lines.append(f"| {task} | 0 | 0% | - | - | - | - | - | skipped |")
        elif status == "failed":
            err = rec.get("error", "?")[:50]
            lines.append(f"| {task} | - | - | - | - | - | - | - | FAILED: {err} |")
        else:
            lines.append(f"| {task} | - | - | - | - | - | - | - | {status} |")

    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Completed tasks: **{len([r for r in tasks.values() if r.get('status') == 'ok'])}** / {len(tasks)}")
    lines.append(
        f"- Skipped (no rollout successes): **{len([r for r in tasks.values() if r.get('status') == 'skipped_no_successes'])}**"
    )
    lines.append(f"- Failed: **{len([r for r in tasks.values() if r.get('status') == 'failed'])}**")
    if ok_rates:
        mean = sum(ok_rates) / len(ok_rates)
        lines.append(f"- Mean eval success rate (completed tasks only): **{100 * mean:.1f}%**")

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    tyro.cli(main)
