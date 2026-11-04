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

    def window_days(self, minimum: int = 0) -> int:
        """Wider of the run window (--days) and this source's own minimum.

        Per-source ``lookback_days`` is a floor, not a cap: a normal daily run
        can look further back than 3 days (S2 recommendations, journal indexing),
        while ``--days 30`` still widens every source for backfill.
        """
        own = int(self.scfg.get("lookback_days") or 0)
        return max(self.days, own, minimum)

    def fetch(self, topics) -> list[Paper]:  # pragma: no cover
        raise NotImplementedError
