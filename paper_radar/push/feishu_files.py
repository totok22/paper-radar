"""Optional PDF delivery through a Feishu self-built app.

Custom-bot webhooks cannot attach files. When FEISHU_APP_ID, FEISHU_APP_SECRET
and FEISHU_RECEIVE_ID are all set, the top-N direct PDFs are downloaded
(size-capped), uploaded with im/v1/files (file_type=pdf), and sent as file
messages. Otherwise this module does nothing.
"""
from __future__ import annotations

import json
import logging
import re

from ..http import download_capped, request

log = logging.getLogger("paper_radar")


def enabled(fs: dict) -> bool:
    return bool(fs.get("app_id") and fs.get("app_secret") and fs.get("receive_id"))


def pdf_filename(p) -> str:
    stem = p.arxiv_id or ((p.doi or "paper").split("/")[-1])
    title = re.sub(r"[^A-Za-z0-9\u4e00-\u9fff]+", "_", p.title or "").strip("_")[:36]
    name = f"{stem}_{title}.pdf" if title else f"{stem}.pdf"
    return name[:80]


def _token(fs: dict) -> str | None:
    base = fs.get("open_base", "https://open.feishu.cn").rstrip("/")
    r = request(
        "POST", f"{base}/open-apis/auth/v3/tenant_access_token/internal",
        json={"app_id": fs["app_id"], "app_secret": fs["app_secret"]},
        retries=2, timeout=20, pace=0,
    )
    if r is None or r.status_code != 200:
        log.error("Feishu tenant_access_token failed: %s", None if r is None else r.text[:200])
        return None
    try:
        data = r.json()
    except ValueError:
        return None
    if data.get("code") not in (0, None) or not data.get("tenant_access_token"):
        log.error("Feishu token error: %s", data)
        return None
    return data["tenant_access_token"]


def _upload(fs: dict, token: str, filename: str, blob: bytes) -> str | None:
    base = fs.get("open_base", "https://open.feishu.cn").rstrip("/")
    r = request(
        "POST", f"{base}/open-apis/im/v1/files",
        headers={"Authorization": f"Bearer {token}"},
        data={"file_type": "pdf", "file_name": filename},
        files={"file": (filename, blob, "application/pdf")},
        retries=2, timeout=60, pace=0.3,
    )
    if r is None or r.status_code != 200:
        log.error("Feishu file upload failed: %s", None if r is None else r.text[:200])
        return None
    try:
        data = r.json()
    except ValueError:
        return None
    if data.get("code") != 0:
        log.error("Feishu file upload error: %s", data)
        return None
    return (data.get("data") or {}).get("file_key")


def _send_file(fs: dict, token: str, file_key: str) -> bool:
    base = fs.get("open_base", "https://open.feishu.cn").rstrip("/")
    receive_type = fs.get("receive_id_type") or "open_id"
    r = request(
        "POST", f"{base}/open-apis/im/v1/messages",
        params={"receive_id_type": receive_type},
        headers={"Authorization": f"Bearer {token}"},
        json={
            "receive_id": fs["receive_id"],
            "msg_type": "file",
            "content": json.dumps({"file_key": file_key}),
        },
        retries=2, timeout=30, pace=0.3,
    )
    if r is None or r.status_code != 200:
        log.error("Feishu file message failed: %s", None if r is None else r.text[:200])
        return False
    try:
        data = r.json()
    except ValueError:
        return False
    if data.get("code") != 0:
        log.error("Feishu file message error: %s", data)
        return False
    return True


def deliver(papers, fs: dict, pdf_cfg: dict | None = None, dry_run: bool = False) -> dict:
    """Upload and send top-N PDFs. No-op unless the app env vars are set."""
    pdf_cfg = pdf_cfg or {}
    if not enabled(fs):
        return {"enabled": False, "sent": 0}
    top_n = int(pdf_cfg.get("push_top_n", 5))
    max_bytes = int(pdf_cfg.get("max_bytes", 20 * 1024 * 1024))
    chosen = [p for p in papers if p.pdf_link][:top_n]
    plan = [{"key": p.key, "url": p.pdf_link, "name": pdf_filename(p)} for p in chosen]
    if dry_run:
        log.info("dry-run: would upload %d PDFs via Feishu app", len(plan))
        return {"enabled": True, "sent": 0, "dry_run": True, "would_send": plan}
    if not chosen:
        return {"enabled": True, "sent": 0, "would_send": []}
    token = _token(fs)
    if not token:
        return {"enabled": True, "sent": 0, "error": "token"}
    sent = 0
    for p in chosen:
        blob = download_capped(p.pdf_link, max_bytes)
        if not blob:
            continue
        file_key = _upload(fs, token, pdf_filename(p), blob)
        if file_key and _send_file(fs, token, file_key):
            sent += 1
            log.info("Feishu PDF sent: %s", p.key)
    return {"enabled": True, "sent": sent}
