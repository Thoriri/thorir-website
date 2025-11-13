#!/usr/bin/env python3
"""Update citation counts, h-index, and BioFoundation GitHub stats in site content.

Fetches metrics from Google Scholar and GitHub, then patches:
- content/home/metrics.md (citations + h-index counters)
- content/home/featured-work.md (BioFoundation stars/forks)
- content/project/biofoundation/index.md (BioFoundation stars/forks bullet)

Usage:
    python scripts/update_metrics.py

Relies only on the Python standard library.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Optional
from urllib.error import URLError, HTTPError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
SCHOLAR_USER = "TyRxmUkAAAAJ"
GITHUB_REPO = "pulp-bio/biofoundation"
USER_AGENT = "Mozilla/5.0 (MetricsUpdater)"


def fetch_text(url: str) -> Optional[str]:
    req = Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urlopen(req, timeout=20) as resp:
            charset = resp.headers.get_content_charset() or "utf-8"
            return resp.read().decode(charset, errors="ignore")
    except (HTTPError, URLError, TimeoutError) as exc:
        print(f"[warn] Failed to fetch {url}: {exc}")
        return None


def fetch_scholar_metrics(user_id: str) -> Optional[tuple[int, int]]:
    html = fetch_text(f"https://scholar.google.com/citations?hl=en&user={user_id}")
    if not html:
        return None

    citations_match = re.search(
        r"Citations</a></td><td class=\"gsc_rsb_std\">(\d+)", html,
        re.IGNORECASE,
    )
    hindex_match = re.search(
        r"h-index</a></td><td class=\"gsc_rsb_std\">(\d+)", html,
        re.IGNORECASE,
    )

    if not citations_match or not hindex_match:
        print("[warn] Unable to parse Scholar metrics; page layout may have changed.")
        return None

    return int(citations_match.group(1)), int(hindex_match.group(1))


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


def update_metrics_card(citations: int, h_index: int) -> None:
    path = ROOT / "content/home/metrics.md"
    text = path.read_text(encoding="utf-8")

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

    path.write_text(text, encoding="utf-8")


def update_biofoundation_stats(stars: int, forks: int) -> None:
    featured = ROOT / "content/home/featured-work.md"
    featured_text = featured.read_text(encoding="utf-8")
    featured_text = re.sub(
        r"⭐\s*\d+\s*stars • 🍴\s*\d+\s*forks",
        f"⭐ {stars} stars • 🍴 {forks} forks",
        featured_text,
    )
    featured.write_text(featured_text, encoding="utf-8")

    project = ROOT / "content/project/biofoundation/index.md"
    project_text = project.read_text(encoding="utf-8")
    project_text = re.sub(
        r"- ⭐\s*\d+\s*GitHub stars • 🍴\s*\d+\s*forks",
        f"- ⭐ {stars} GitHub stars • 🍴 {forks} forks",
        project_text,
    )
    project.write_text(project_text, encoding="utf-8")


def main() -> int:
    scholar_metrics = fetch_scholar_metrics(SCHOLAR_USER)
    if scholar_metrics:
        citations, h_index = scholar_metrics
        update_metrics_card(citations, h_index)
        print(f"[info] Updated citations={citations}, h-index={h_index}")
    else:
        print("[warn] Skipped updating citation metrics.")

    gh_metrics = fetch_github_stats(GITHUB_REPO)
    if gh_metrics:
        stars, forks = gh_metrics
        update_biofoundation_stats(stars, forks)
        print(f"[info] Updated GitHub stats: stars={stars}, forks={forks}")
    else:
        print("[warn] Skipped updating GitHub stats.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
