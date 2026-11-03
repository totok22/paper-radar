from __future__ import annotations

import os
from pathlib import Path

import yaml

from .validate import validate_config


def load_config(path: str | Path, *, check: bool = True) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    env = os.environ.get
    llm = cfg.setdefault("llm", {})
    llm["api_key"] = env("LLM_API_KEY") or llm.get("api_key") or ""
    llm["base_url"] = env("LLM_BASE_URL") or llm.get("base_url") or "https://api.deepseek.com"
    llm["model"] = env("LLM_MODEL") or llm.get("model") or "deepseek-flash"
    emb = cfg.setdefault("embedding", {})
    emb["api_key"] = env("EMBED_API_KEY") or emb.get("api_key") or ""
    emb["base_url"] = env("EMBED_BASE_URL") or emb.get("base_url") or ""
    emb["model"] = env("EMBED_MODEL") or emb.get("model") or ""
    fs = cfg.setdefault("feishu", {})
    fs["webhook"] = env("FEISHU_WEBHOOK") or fs.get("webhook") or ""
    fs["secret"] = env("FEISHU_SECRET") or fs.get("secret") or ""
    fs["app_id"] = env("FEISHU_APP_ID") or fs.get("app_id") or ""
    fs["app_secret"] = env("FEISHU_APP_SECRET") or fs.get("app_secret") or ""
    fs["receive_id"] = env("FEISHU_RECEIVE_ID") or fs.get("receive_id") or ""
    fs["receive_id_type"] = env("FEISHU_RECEIVE_ID_TYPE") or fs.get("receive_id_type") or "open_id"
    fs["open_base"] = env("FEISHU_OPEN_BASE") or fs.get("open_base") or "https://open.feishu.cn"
    src = cfg.setdefault("sources", {})
    s2 = src.setdefault("semantic_scholar", {})
    s2["api_key"] = env("S2_API_KEY") or s2.get("api_key") or ""
    ds = src.setdefault("deepseek_search", {})
    ds["api_key"] = env("LLM_API_KEY") or ds.get("api_key") or llm.get("api_key") or ""
    oa = src.setdefault("openalex", {})
    oa["api_key"] = env("OPENALEX_API_KEY") or oa.get("api_key") or ""
    oa["email"] = env("OPENALEX_EMAIL") or oa.get("email") or ""
    ds["email"] = env("OPENALEX_EMAIL") or ds.get("email") or oa.get("email") or ""
    journals = src.setdefault("journals", {})
    journals["api_key"] = journals.get("api_key") or oa.get("api_key") or ""
    journals["email"] = journals.get("email") or oa.get("email") or ""
    pdf = cfg.setdefault("pdf", {})
    pdf["email"] = env("UNPAYWALL_EMAIL") or pdf.get("email") or oa.get("email") or ""
    z = cfg.setdefault("zotero", {})
    z["library_id"] = env("ZOTERO_ID") or z.get("library_id") or ""
    z["api_key"] = env("ZOTERO_KEY") or z.get("api_key") or ""
    if check:
        cfg["_warnings"] = validate_config(cfg)
    return cfg
