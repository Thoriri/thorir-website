#!/usr/bin/env python3
"""Update citation counts, h-index, and BioFoundation GitHub stats in site content.

Fetches metrics directly from Google Scholar (with a proxy fallback to avoid
anti-bot blocks) and GitHub, then patches:
- content/home/metrics.md (citations + h-index counters)
- content/home/featured-work.md (BioFoundation stars/forks)
- content/project/biofoundation/index.md (BioFoundation stars/forks bullet)

Usage:
    python scripts/update_metrics.py

Relies only on the Python standard library.

Environment variables:
    SCHOLAR_PROXY_BASE (optional): Override proxy prefix (default: r.jina.ai)
    GITHUB_TOKEN: GitHub token for API access (auto-provided in GitHub Actions)

Reference:
    Google Scholar scraping strategies inspired by:
    https://www.scrapeless.com/en/blog/scrape-google-scholar
"""

from __future__ import annotations

import json
import os
import re
import sys
import random
import time
from pathlib import Path
from typing import Optional
from urllib.error import URLError, HTTPError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
SCHOLAR_USER = "TyRxmUkAAAAJ"
GITHUB_REPO = "pulp-bio/biofoundation"
GH_TOKEN = os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN")
DEFAULT_PROXY_BASE = "https://r.jina.ai/http://"
SCHOLAR_PROXY_BASE = os.getenv("SCHOLAR_PROXY_BASE", DEFAULT_PROXY_BASE)
PROXY_ENABLED = SCHOLAR_PROXY_BASE.lower() not in {"", "none", "off"}
BLOCK_INDICATORS = [
    "our systems have detected unusual traffic",
    "unusual traffic from your computer network",
    "enable javascript to view the page",
    "please show you're not a robot",
    "forbidden",
]
USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
]


def fetch_text(url: str, retries: int = 2, delay: float = 1.0, extra_headers: Optional[dict] = None) -> Optional[str]:
    """Fetch text from URL with retry logic and browser-like headers."""
    headers = {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }
    if "api.github.com" in url and GH_TOKEN:
        headers["Authorization"] = f"Bearer {GH_TOKEN}"
        headers["Accept"] = "application/vnd.github+json"
    if extra_headers:
        headers.update(extra_headers)
    
    for attempt in range(retries + 1):
        req = Request(url, headers=headers)
        try:
            with urlopen(req, timeout=20) as resp:
                charset = resp.headers.get_content_charset() or "utf-8"
                return resp.read().decode(charset, errors="ignore")
        except (HTTPError, URLError, TimeoutError) as exc:
            if attempt < retries:
                sleep_for = delay * (attempt + 1)
                print(f"[warn] Attempt {attempt + 1} failed for {url}: {exc}, retrying in {sleep_for:.1f}s...")
                time.sleep(sleep_for)
            else:
                print(f"[warn] Failed to fetch {url} after {retries + 1} attempts: {exc}")
                return None
    return None


def looks_blocked(html: str) -> bool:
    """Return True if the HTML appears to be a Google anti-bot page."""
    lowered = html.lower()
    return any(indicator in lowered for indicator in BLOCK_INDICATORS)


def build_proxy_url(url: str) -> str:
    """Return the proxied URL using the configured proxy base."""
    if not PROXY_ENABLED:
        return ""
    base = SCHOLAR_PROXY_BASE.rstrip("/")
    if "{url}" in base:
        return base.replace("{url}", url)
    stripped = url.replace("https://", "").replace("http://", "")
    if base.endswith("http://") or base.endswith("https://"):
        return f"{base}{stripped}"
    return f"{base}/{url.lstrip('/')}"


def fetch_google_scholar_html(user_id: str) -> Optional[str]:
    """Fetch Google Scholar profile HTML, falling back to a proxy if blocked."""
    base_url = f"https://scholar.google.com/citations?hl=en&user={user_id}"
    headers = {
        "Referer": "https://scholar.google.com/",
    }
    html = fetch_text(base_url, extra_headers=headers)
    if html and not looks_blocked(html):
        return html

    if html:
        print("[warn] Direct Google Scholar fetch looked blocked; trying proxy.")
    else:
        print("[warn] Direct Google Scholar fetch failed; trying proxy.")

    proxy_url = build_proxy_url(base_url)
    if not proxy_url:
        print("[warn] Proxy disabled; cannot retry Google Scholar fetch.")
        return None

    return fetch_text(proxy_url, extra_headers=headers)


def fetch_scholar_metrics(user_id: str) -> Optional[tuple[int, int]]:
    """Fetch citation count and h-index from Google Scholar."""
    html = fetch_google_scholar_html(user_id)
    if not html:
        return None

    patterns = [
        (
            r"Citations</a></td><td class=\"gsc_rsb_std\">(\d+)",
            r"h-index</a></td><td class=\"gsc_rsb_std\">(\d+)",
        ),
        (
            r"Citations</a>.*?<td class=\"gsc_rsb_std\">(\d+)",
            r"h-index</a>.*?<td class=\"gsc_rsb_std\">(\d+)",
        ),
        (r"aria-label=\"Total citations\">(\d+)", r"aria-label=\"H-index\">(\d+)"),
        (r"\"Citations\".*?(\d+)", r"\"h-index\".*?(\d+)"),
    ]

    for cit_pattern, h_pattern in patterns:
        citations_match = re.search(cit_pattern, html, re.IGNORECASE | re.DOTALL)
        hindex_match = re.search(h_pattern, html, re.IGNORECASE | re.DOTALL)

        if citations_match and hindex_match:
            try:
                return int(citations_match.group(1)), int(hindex_match.group(1))
            except (ValueError, IndexError):
                continue

    print("[warn] Unable to parse Scholar metrics; page layout may have changed.")
    return None


def fetch_github_stats(repo: str) -> Optional[tuple[int, int]]:
    data = fetch_text(f"https://api.github.com/repos/{repo}")
    if not data:
        return None

    try:
        payload = json.loads(data)
        stars = int(payload.get("stargazers_count", 0))
        forks = int(payload.get("forks_count", 0))
    except (ValueError, TypeError) as exc:
        print(f"[warn] Could not parse GitHub response: {exc}")
        return None

    return stars, forks


def write_if_changed(path: Path, new_text: str) -> bool:
    """Write new_text to path if content differs. Returns True if file updated."""
    if not path.exists():
        print(f"[warn] File not found, creating: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(new_text, encoding="utf-8")
        return True

    old_text = path.read_text(encoding="utf-8")
    if old_text != new_text:
        path.write_text(new_text, encoding="utf-8")
        return True
    return False


def update_metrics_card(citations: int, h_index: int) -> bool:
    path = ROOT / "content/home/metrics.md"
    if not path.exists():
        print(f"[warn] Missing metrics card, skipping: {path}")
        return False

    original = path.read_text(encoding="utf-8")
    text = original

    text = re.sub(
        r'(data-target=")\d+(" aria-live="polite" aria-label="Total citations")',
        rf'\g<1>{citations}\g<2>',
        text,
    )
    text = re.sub(
        r'(aria-label="Total citations">)\d+',
        rf'\g<1>{citations}',
        text,
    )
    text = re.sub(
        r'(data-target=")\d+(" aria-live="polite" aria-label="H-index")',
        rf'\g<1>{h_index}\g<2>',
        text,
    )
    text = re.sub(
        r'(aria-label="H-index">)\d+',
        rf'\g<1>{h_index}',
        text,
    )

    changed = write_if_changed(path, text)
    print(f"[info] metrics.md {'updated' if changed else 'no change'} "
          f"(citations={citations}, h-index={h_index})")
    return changed


def update_biofoundation_stats(stars: int, forks: int) -> bool:
    changed_any = False

    featured = ROOT / "content/home/featured-work.md"
    if featured.exists():
        original = featured.read_text(encoding="utf-8")
        text = re.sub(
            r"⭐\s*\d+\s*stars • 🍴\s*\d+\s*forks",
            f"⭐ {stars} stars • 🍴 {forks} forks",
            original,
        )
        changed = write_if_changed(featured, text)
        print(f"[info] featured-work.md {'updated' if changed else 'no change'} "
              f"(stars={stars}, forks={forks})")
        changed_any = changed_any or changed
    else:
        print(f"[warn] Missing: {featured}")

    project = ROOT / "content/project/biofoundation/index.md"
    if project.exists():
        original = project.read_text(encoding="utf-8")
        text = re.sub(
            r"- ⭐\s*\d+\s*GitHub stars • 🍴\s*\d+\s*forks",
            f"- ⭐ {stars} GitHub stars • 🍴 {forks} forks",
            original,
        )
        changed = write_if_changed(project, text)
        print(f"[info] project biofoundation index.md "
              f"{'updated' if changed else 'no change'} (stars={stars}, forks={forks})")
        changed_any = changed_any or changed
    else:
        print(f"[warn] Missing: {project}")

    return changed_any


def main() -> int:
    changed = False

    # Fetch metrics from Google Scholar (with proxy fallback)
    print(f"[info] Fetching metrics from Google Scholar (user ID: {SCHOLAR_USER})...")
    scholar_metrics = fetch_scholar_metrics(SCHOLAR_USER)
    
    if scholar_metrics:
        citations, h_index = scholar_metrics
        print(f"[info] Successfully fetched metrics: citations={citations}, h-index={h_index}")
        changed = update_metrics_card(citations, h_index) or changed
    else:
        print("[warn] Skipped updating citation metrics.")
        if PROXY_ENABLED:
            print("[info] Tip: Double-check proxy prefix or run locally to refresh metrics.")
        else:
            print("[info] Tip: Set SCHOLAR_PROXY_BASE (default r.jina.ai) to enable proxy fallback.")

    gh_metrics = fetch_github_stats(GITHUB_REPO)
    if gh_metrics:
        stars, forks = gh_metrics
        changed = update_biofoundation_stats(stars, forks) or changed
    else:
        print("[warn] Skipped updating GitHub stats.")

    print(f"[info] Done. Changes written: {changed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())