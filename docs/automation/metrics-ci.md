# Automating Citation & GitHub Metrics Updates

This repository includes `scripts/update_metrics.py`, which pulls:

- **Citation metrics** directly from Google Scholar (with an optional proxy fallback)
  - Total citations and h-index
  - Uses a low-frequency fetch + human-like headers to avoid anti-bot triggers
  - Falls back to the open `r.jina.ai` reader proxy if Google blocks the CI IP
- **GitHub stats** for `pulp-bio/biofoundation` (stars and forks)

It then rewrites:

- `content/home/metrics.md` (animated counters on the homepage)
- `content/home/tm-biofoundation.md` (active homepage BioFoundation stats)
- `content/home/featured-work.md` (BioFoundation card)
- `content/project/biofoundation/index.md` (resource summary)

The script relies only on the Python standard library (`urllib`, `json`). It exits gracefully if a fetch fails, leaving existing numbers untouched.

## Workflow routing

GitHub schedules workflows from the repository default branch. The scheduled
workflow therefore lives there but checks out and commits to `redesign`, the
deployed content branch. Keep the workflow definition on the default branch in
sync with the copy on `redesign` whenever its routing changes.

## Optional Proxy Configuration

The workflow usually succeeds with a single request every two days. If Google still blocks the GitHub Actions IP, the script automatically retries via the public `r.jina.ai` reader proxy. You can override the proxy prefix (or point to your own relay) by defining `SCHOLAR_PROXY_BASE` in repository **Variables** (not secrets), e.g.:

- Default: `https://r.jina.ai/http://`
- Custom pattern: `https://your-proxy.example.com?url={url}`

Set `SCHOLAR_PROXY_BASE` to `off` to disable the proxy entirely.

## Triggering the Script in CI/CD

1. **Ensure Python is installed** in the build environment (Python 3.8+ recommended).
2. **Before running `hugo`**, call the script:

   ```bash
   python scripts/update_metrics.py
   ```

3. **Proceed with the static site build**:

   ```bash
   hugo --minify
   ```

4. **Commit/Deploy** the resulting changes if your workflow pushes artifacts back to the repository.

## Example (GitHub Actions)

```yaml
name: Build site

on:
  push:
    branches: [ main ]
  workflow_dispatch:

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Update Scholar & GitHub metrics
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          SCHOLAR_PROXY_BASE: ${{ vars.SCHOLAR_PROXY_BASE }}
        run: python scripts/update_metrics.py

      - name: Install Hugo
        uses: peaceiris/actions-hugo@v2
        with:
          hugo-version: '0.111.3'

      - name: Build site
        run: hugo --minify

      # Optional: deploy or upload artifacts
      # - uses: peaceiris/actions-gh-pages@v3
      #   with:
      #     github_token: ${{ secrets.GITHUB_TOKEN }}
      #     publish_dir: ./public
```

## Notes & Troubleshooting

- The script minimizes anti-bot triggers by rotating User-Agents, adding realistic headers, and running only once every 48 hours.
- If Google still blocks the request, the script automatically retries via the proxy prefix (`SCHOLAR_PROXY_BASE`, default `r.jina.ai`). Disable or replace it if you manage your own relay.
- If both direct and proxy fetches fail, the script prints a warning and leaves the existing metrics untouched. GitHub stats continue to update.
- For private pipelines, consider caching the last-successful response to avoid unnecessary hits.
- If you need additional metrics, extend `update_metrics.py` with new selectors and target files.
- References:
  - https://www.scrapeless.com/en/blog/scrape-google-scholar (general anti-bot guidance)
