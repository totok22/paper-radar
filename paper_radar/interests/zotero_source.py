"""Optional Zotero adapter (pip install pyzotero).

Maps Zotero collections to topics via config:

zotero:
  enabled: true
  library_id: "1234567"        # or env ZOTERO_ID
  api_key: ""                  # or env ZOTERO_KEY
  library_type: user
  collection_map:              # Zotero collection name -> topic id
    "规划-LLM基元": planning_llm_primitives
    "场景-扩散": scen_diffusion

Every item in a mapped collection becomes a positive seed (arXiv ID / DOI)
and its title+abstract is added to the topic profile text.
"""
from __future__ import annotations

import logging
import re

from .base import InterestSource, Topic

log = logging.getLogger("paper_radar")
ARXIV_RE = re.compile(r"(\d{4}\.\d{4,5})")


class ZoteroInterestSource(InterestSource):
    name = "zotero"

    def __init__(self, zcfg: dict):
        self.cfg = zcfg

    def contribute(self, topics: dict[str, Topic]) -> None:
        try:
            from pyzotero import zotero
        except ImportError:
            log.warning("zotero.enabled=true but pyzotero not installed: pip install pyzotero")
            return
        if not (self.cfg.get("library_id") and self.cfg.get("api_key")):
            log.warning("zotero enabled but ZOTERO_ID / ZOTERO_KEY missing; skipped")
            return
        zot = zotero.Zotero(self.cfg["library_id"], self.cfg.get("library_type", "user"), self.cfg["api_key"])
        cmap = self.cfg.get("collection_map", {}) or {}
        cols = {c["data"]["name"]: c["key"] for c in zot.everything(zot.collections())}
        for cname, tid in cmap.items():
            if cname not in cols or tid not in topics:
                log.warning("zotero collection %r or topic %r not found", cname, tid)
                continue
            for it in zot.everything(zot.collection_items_top(cols[cname])):
                d = it["data"]
                ref = None
                m = ARXIV_RE.search(" ".join(str(d.get(k, "")) for k in ("url", "extra", "archiveID", "DOI")))
                if d.get("DOI"):
                    ref = f"DOI:{d['DOI']}"
                elif m:
                    ref = f"ARXIV:{m.group(1)}"
                if ref:
                    topics[tid].positive_ids.append(ref)
                topics[tid].positive_texts.append(f"{d.get('title', '')}. {d.get('abstractNote', '')[:500]}")
