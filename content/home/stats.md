---
widget: blank
headless: true
weight: 22

title: ""
subtitle: ""

design:
  columns: '1'
  spacing:
    padding: ['20px', '0', '20px', '0']
---

<div class="metrics-casual" id="stats-section">
  <div class="metric">
    <span class="metric-value" data-target="687">0</span>
    <div class="metric-label">Citations</div>
  </div>
  <div class="metric">
    <span class="metric-value" data-target="12">0</span>
    <div class="metric-label">h-index</div>
  </div>
  <div class="metric">
    <span class="metric-value" data-target="15">0</span>
    <div class="metric-label">Publications</div>
  </div>
  <div class="metric">
    <span class="metric-value" data-target="27">0</span>
    <div class="metric-label">GitHub Stars</div>
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
  threshold: 0.5,
  rootMargin: '0px'
};

const observer = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting && !entry.target.classList.contains('counted')) {
      entry.target.classList.add('counted');
      const metrics = entry.target.querySelectorAll('.metric-value');
      metrics.forEach(metric => {
        const target = parseInt(metric.getAttribute('data-target'));
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

