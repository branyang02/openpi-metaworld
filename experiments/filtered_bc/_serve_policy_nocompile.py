"""Shim that disables ``torch.compile(mode="max-autotune")`` before running ``scripts/serve_policy.py``.

``PI0Pytorch.__init__`` wraps ``sample_actions`` with ``torch.compile(mode="max-autotune")``
at construction time (``src/openpi/models_pytorch/pi0_pytorch.py:150``). On a cold L40 GPU
this costs ~5 minutes of Triton autotune on the first inference call — and the filtered-BC
subprocess pipeline launches a fresh server for every (rollout, eval) pair, so that cost
would be paid twice per task. Since our flow doesn't need the autotuned kernels (the
merged checkpoint will be thrown away in minutes), we patch ``torch.compile`` into a
no-op before importing anything that triggers the pi0_pytorch module load.

Invocation mirrors serve_policy.py exactly:

    python _serve_policy_nocompile.py --pytorch --port=... policy:checkpoint ...
"""

from __future__ import annotations

import runpy
import sys

import torch


def _noop_compile(fn=None, *_args, **_kwargs):
    if fn is None:
        return lambda f: f
    return fn


torch.compile = _noop_compile  # type: ignore[assignment]


if __name__ == "__main__":
    # Rewrite argv so scripts/serve_policy.py sees its own name + our flags, then run it
    # as __main__ (triggers its tyro.cli entry point).
    sys.argv = ["scripts/serve_policy.py", *sys.argv[1:]]
    runpy.run_path("scripts/serve_policy.py", run_name="__main__")
