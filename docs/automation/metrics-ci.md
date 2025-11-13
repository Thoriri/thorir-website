# Automating Citation & GitHub Metrics Updates

This repository includes `scripts/update_metrics.py`, which pulls:

- Google Scholar totals (citations and h-index) for the profile `TyRxmUkAAAAJ`
- GitHub star/fork counts for `pulp-bio/biofoundation`

It then rewrites:

- `content/home/metrics.md` (animated counters on the homepage)
- `content/home/featured-work.md` (BioFoundation card)
- `content/project/biofoundation/index.md` (resource summary)

The script relies only on the Python standard library (`urllib`). It exits gracefully if a fetch fails, leaving existing numbers untouched.

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

- Google Scholar occasionally rate-limits requests. The script will print a warning and skip updates if the page layout changes or the request fails.
- For private pipelines, consider caching the API responses to avoid repeated calls.
- If you need additional metrics, extend `update_metrics.py` with new selectors and target files.
