# thorirmar.com

Source for Thorir Mar Ingolfsson's personal research website. It is a Hugo site
using the pinned Wowchemy v5 module, with custom content and styling for the
research portfolio, publications, talks, and student-project pages.

## Local preview

Install Hugo Extended **0.111.3**, then run:

```sh
hugo server --buildFuture
```

Open the local URL printed by Hugo. `--buildFuture` keeps upcoming dated
content visible during review, matching deploy-preview behavior.

## Production build

```sh
hugo --gc --minify --buildFuture --baseURL https://thorirmar.com/
```

Netlify uses the same Hugo version, configured in `netlify.toml`. The GitHub
Actions site check runs this build for pull requests and pushes to `redesign`.

## Publishing branches

`redesign` is the production-content branch reviewed by Netlify. The public
repository's default branch hosts scheduled workflow definitions; the metrics
workflow must check out and commit back to `redesign` so generated metrics
reach the deployed site.

## Automated metrics

`scripts/update_metrics.py` refreshes Google Scholar citations/h-index and
BioFoundation GitHub stars/forks. It updates homepage, project, and legacy
feature content without requiring third-party Python packages.

For details and manual use, see [the metrics automation guide](docs/automation/metrics-ci.md).

## Key directories

- `content/` — pages, posts, publications, talks, and homepage sections.
- `assets/scss/custom.scss` — site-specific presentation styles.
- `layouts/` — Hugo/Wowchemy template overrides.
- `static/uploads/` — downloadable assets and static media.
- `.github/workflows/` — build validation and metrics automation.
