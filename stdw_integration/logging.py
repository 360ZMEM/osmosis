from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class STDWCSVLogger:
    file_path: Path
    fieldnames: list[str]
    rows: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.file_path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self.fieldnames)
        self._writer.writeheader()

    def append(self, row: dict[str, Any]) -> None:
        self.rows.append(row)
        self._writer.writerow(self._prepare_row(row))
        self._file.flush()

    def _prepare_row(self, row: dict[str, Any]) -> dict[str, Any]:
        prepared: dict[str, Any] = {}
        for key in self.fieldnames:
            value = row.get(key, "")
            if isinstance(value, (list, tuple, dict)):
                prepared[key] = json.dumps(value)
            else:
                prepared[key] = value
        return prepared

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)

    def close(self) -> None:
        self._file.close()
