from __future__ import annotations

from datetime import date, timedelta

from ..models import Paper


class Source:
    name = "base"

    def __init__(self, scfg: dict, cfg: dict):
        self.scfg = scfg
        self.cfg = cfg
        self.days = int(cfg.get("lookback_days", 3))

    @property
    def since(self) -> date:
        return date.today() - timedelta(days=self.days)

    def fetch(self, topics) -> list[Paper]:  # pragma: no cover
        raise NotImplementedError
