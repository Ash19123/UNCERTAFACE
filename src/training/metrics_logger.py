"""Per-epoch metrics to CSV + JSON (+ TensorBoard when installed)."""
from __future__ import annotations

import csv
import json
from pathlib import Path


class MetricsLogger:
    def __init__(self, run_dir, tensorboard: bool = True):
        self.dir = Path(run_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.writer = None
        if tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(str(self.dir / "tb"))
            except Exception:                      # tensorboard is optional
                self.writer = None

    def log(self, epoch: int, row: dict) -> None:
        if self.writer:
            for k, v in row.items():
                if isinstance(v, (int, float)):
                    self.writer.add_scalar(k, v, epoch)

    def write_history(self, history: list[dict]) -> None:
        """Rewrite history.json and metrics.csv from the full history (resume-safe)."""
        (self.dir / "history.json").write_text(json.dumps(history, indent=1))
        if history:
            keys = sorted({k for h in history for k, v in h.items() if isinstance(v, (int, float, str))})
            with open(self.dir / "metrics.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                for h in history:
                    w.writerow({k: h.get(k, "") for k in keys})

    def close(self) -> None:
        if self.writer:
            self.writer.close()
