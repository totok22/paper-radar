"""Beijing Institute of Technology (BIT) access links for closed papers.

WebVPN (网瑞达) encrypts the target host with AES-128-CFB. The key and IV
are the vendor string ``wrdvpnisthebest!``. CARSI WAYFless links are only
built when the article URL is on IEEE Xplore or ScienceDirect (Elsevier
needs a PII, not a DOI). Nothing here reaches Sci-Hub or LibGen.
"""
from __future__ import annotations

import re
from urllib.parse import quote, urlsplit, urlunsplit

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

DEFAULT_KEY = "wrdvpnisthebest!"
DEFAULT_BASE = "https://webvpn.bit.edu.cn"
DEFAULT_IDP = "https://idp.bit.edu.cn/idp/shibboleth"

_IEEE_DOC = re.compile(r"ieeexplore\.ieee\.org/(?:document|abstract/document)/(\d+)", re.I)
_IEEE_ARN = re.compile(r"[?&]arnumber=(\d+)", re.I)
_PII = re.compile(r"/pii/([A-Z0-9]{10,20})", re.I)
_PII_BARE = re.compile(r"^S\d{16,17}$")


def encrypt_host(host: str, key: str = DEFAULT_KEY) -> str:
    """hex(iv) + hex(AES-128-CFB(host)). IV and key are the same 16 bytes."""
    raw = key.encode("utf-8")
    if len(raw) != 16:
        raise ValueError("WebVPN key must be 16 bytes")
    enc = Cipher(algorithms.AES(raw), modes.CFB(raw)).encryptor()
    ct = enc.update(host.encode("utf-8")) + enc.finalize()
    return raw.hex() + ct.hex()


def _scheme_token(scheme: str, port: int | None) -> str:
    scheme = (scheme or "https").lower()
    if port and not ((scheme == "https" and port == 443) or (scheme == "http" and port == 80)):
        return f"{scheme}-{port}"
    return scheme


def webvpn_url(target: str, *, base: str = DEFAULT_BASE, key: str = DEFAULT_KEY) -> str:
    """Rewrite an absolute article URL onto the BIT WebVPN host."""
    parts = urlsplit(target.strip())
    host = parts.hostname or ""
    if not host:
        return ""
    token = _scheme_token(parts.scheme, parts.port)
    path = parts.path or ""
    if path and not path.startswith("/"):
        path = "/" + path
    url = f"{base.rstrip('/')}/{token}/{encrypt_host(host, key)}{path}"
    if parts.query:
        url += "?" + parts.query
    return url


def ieee_document_url(url: str) -> str | None:
    if not url:
        return None
    m = _IEEE_DOC.search(url) or _IEEE_ARN.search(url)
    if not m:
        return None
    return f"https://ieeexplore.ieee.org/document/{m.group(1)}"


def sciencedirect_pii_url(url: str) -> str | None:
    if not url:
        return None
    m = _PII.search(url)
    if not m:
        return None
    return f"https://www.sciencedirect.com/science/article/pii/{m.group(1)}"


def publisher_url_from(url: str) -> str | None:
    """Prefer a stable publisher article URL over a DOI landing page."""
    if not url:
        return None
    ieee = ieee_document_url(url)
    if ieee:
        return ieee
    sd = sciencedirect_pii_url(url)
    if sd:
        return sd
    parts = urlsplit(url)
    host = (parts.hostname or "").lower()
    if host.endswith("link.springer.com") and parts.path.startswith("/"):
        return urlunsplit((parts.scheme or "https", host, parts.path, "", ""))
    return None


def publisher_from_crossref(message: dict, doi: str = "") -> str | None:
    if not isinstance(message, dict):
        return None
    for link in message.get("link") or []:
        if isinstance(link, dict):
            got = publisher_url_from(link.get("URL") or "")
            if got:
                return got
    for alt in message.get("alternative-id") or []:
        text = str(alt).strip()
        if _PII_BARE.match(text):
            return f"https://www.sciencedirect.com/science/article/pii/{text}"
        if text.isdigit() and (doi or "").lower().startswith("10.1109/"):
            return f"https://ieeexplore.ieee.org/document/{text}"
    return None


def carsi_url(article_url: str, *, idp: str = DEFAULT_IDP) -> str | None:
    """WAYFless CARSI. Elsevier needs a ScienceDirect PII URL, not a DOI."""
    if not article_url or not idp:
        return None
    host = (urlsplit(article_url).hostname or "").lower()
    if host.endswith("ieeexplore.ieee.org"):
        return (
            "https://ieeexplore.ieee.org/servlet/wayf.jsp?entityId="
            + quote(idp, safe="")
            + "&url="
            + quote(article_url, safe="")
        )
    if "sciencedirect.com" in host and "/pii/" in article_url:
        return (
            "https://auth.elsevier.com/ShibAuth/institutionLogin?entityID="
            + quote(idp, safe="")
            + "&appReturnURL="
            + quote(article_url, safe="")
        )
    return None


def access_buttons(target: str, icfg: dict | None) -> list[tuple[str, str]]:
    """(label, url) pairs for a journal article that has no free PDF."""
    if not icfg or not icfg.get("enabled", False) or not target:
        return []
    base = icfg.get("webvpn_base") or DEFAULT_BASE
    key = icfg.get("webvpn_key") or DEFAULT_KEY
    idp = icfg.get("idp") or DEFAULT_IDP
    buttons = []
    vpn = webvpn_url(target, base=base, key=key)
    if vpn:
        buttons.append(("WebVPN", vpn))
    carsi = carsi_url(target, idp=idp)
    if carsi:
        buttons.append(("CARSI", carsi))
    return buttons
