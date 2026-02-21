---
widget: blank
headless: true
weight: 24

title: ""
subtitle: ""

design:
  columns: '1'
  spacing:
    padding: ['30px', '0', '40px', '0']
---

<div class="metrics-card" id="stats-section" role="region" aria-label="Research impact metrics">
  <div class="metric">
    <span class="metric-value" data-target="868" aria-live="polite" aria-label="Total citations">868</span>
    <span class="metric-label">Citations</span>
    <a class="metric-link" href="https://scholar.google.com/citations?user=TyRxmUkAAAAJ&hl=en" target="_blank" rel="noopener">
      <i class="ai ai-google-scholar" aria-hidden="true"></i>
      Google Scholar
    </a>
  </div>

  <div class="metric">
    <span class="metric-value" data-target="12" aria-live="polite" aria-label="H-index">12</span>
    <span class="metric-label">h-index</span>
    <span class="metric-caption">Core research impact</span>
  </div>

  <div class="metric">
    <span class="metric-value" data-target="24" aria-live="polite" aria-label="Publications">0</span>
    <span class="metric-label">Publications</span>
    <a class="metric-link" href="/publication/">
      View publications →
    </a>
  </div>
</div>

<script>
  // Animated counter function
  function animateCounter(element, target, duration = 2000) {
    const start = 0;
    const increment = target / (duration / 16); // 60fps
    let current = start;

    const timer = setInterval(() => {
      current += increment;
      if (current >= target) {
        element.textContent = target.toLocaleString();
        clearInterval(timer);
      } else {
        element.textContent = Math.floor(current).toLocaleString();
      }
    }, 16);
  }

  // Intersection Observer for scroll-triggered animation
  const observerOptions = {
    threshold: 0.4,
    rootMargin: '0px'
  };

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting && !entry.target.classList.contains('counted')) {
        entry.target.classList.add('counted');
        const metrics = entry.target.querySelectorAll('.metric-value');
        metrics.forEach(metric => {
          const target = parseInt(metric.getAttribute('data-target'), 10);
          animateCounter(metric, target);
        });
      }
    });
  }, observerOptions);

  // Observe the stats section when DOM is ready
  document.addEventListener('DOMContentLoaded', () => {
    const statsSection = document.getElementById('stats-section');
    if (statsSection) {
      observer.observe(statsSection);
    }
  });
</script>
