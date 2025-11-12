# AI Agent Prompt: Personal Academic Website Enhancement

## Mission Statement

You are tasked with enhancing a personal academic website for **Thorir Mar Ingolfsson**, a Postdoctoral Researcher and ML Group Leader at ETH Zurich. The website has been redesigned from a PhD student portfolio to a research group leader presence, and now needs visual and interactive enhancements to make it more attractive and engaging.

---

## Required Reading (In Order)

Before starting any work, read these documents in the repository to understand the full context:

### 1. **FINAL-SUMMARY.md** (CRITICAL - Read First)
This is the master document containing:
- Complete overview of what has been implemented
- All files created and modified
- Technical fixes applied
- Hugo version compatibility issues
- Comprehensive design enhancement ideas (Section: "Ideas to Make the Site More Beautiful & Attractive")

**Key sections to focus on:**
- "Successfully Implemented Changes" - What's already done
- "Ideas to Make the Site More Beautiful & Attractive" - Your roadmap
- "Recommended Priority Order" - Where to start

### 2. **high-level-plan.md**
Strategic analysis and recommendations for the transition from PhD to postdoc positioning.

### 3. **website-layout-redesign.md**
Detailed layout design with group leader positioning philosophy.

### 4. **IMPLEMENTATION-SUMMARY.md**
Technical implementation details and next steps.

---

## Current State

### ✅ What's Already Done

**Profile & Content:**
- Author profile updated to "Postdoctoral Researcher | ML Group Leader"
- Biography rewritten with leadership-focused language
- Three new homepage sections created:
  - Research Interests (3 cards)
  - Featured Work (LUNA/BioFoundation showcase)
  - Available Projects (student recruitment)
- BioFoundation project page with YouTube embed
- Navigation restructured (Research → Publications → Projects → Opportunities)

**Technical:**
- Hugo v0.111.3 compatible (NOT v0.152 - critical!)
- Ocean theme applied
- Custom CSS in `assets/scss/custom.scss`
- Google Analytics fixed with custom partial
- All Hugo compatibility issues resolved

**Branch:** `claude/analyze-personal-website-011CV3o1H7QAN4BzgyteG6pK`

### 🎯 Your Mission

Implement the visual and interactive enhancements outlined in **FINAL-SUMMARY.md** under "Ideas to Make the Site More Beautiful & Attractive" to transform the site from functional to exceptional.

---

## Technical Environment & Constraints

### CRITICAL Technical Requirements

**⚠️ Hugo Version:**
- **MUST USE Hugo v0.111.3 or compatible (NOT v0.152+)**
- The Wowchemy theme v5 (Aug 2021) is incompatible with Hugo v0.152
- Netlify uses Hugo 0.87.0 (specified in netlify.toml)
- Local development requires Hugo v0.111.3 or similar

**Technology Stack:**
- Hugo Static Site Generator (v0.111.3)
- Wowchemy Academic Theme v5
- Current theme: `ocean` (professional blue)
- Deployment: Netlify (automatic on push to master)
- Repository: thorir-website-private

**Working Directory Structure:**
```
thorir-website-private/
├── config/_default/          # Hugo configuration
├── content/
│   ├── authors/admin/        # Author profile
│   ├── home/                 # Homepage widgets
│   ├── publication/          # Publications
│   ├── project/              # Projects (includes BioFoundation)
│   └── event/                # Talks/conferences
├── layouts/partials/         # Custom HTML partials
├── assets/scss/              # Custom CSS (custom.scss)
├── data/themes/              # Theme definitions
└── static/                   # Static assets
```

**Key Files to Know:**
- `content/authors/admin/_index.md` - Main profile
- `content/home/*.md` - Homepage section widgets
- `config/_default/params.yaml` - Site parameters
- `config/_default/menus.yaml` - Navigation
- `assets/scss/custom.scss` - Custom styles
- `layouts/partials/marketing/google_analytics.html` - Custom GA partial

---

## Priority Tasks (Recommended Order)

### Phase 1: Quick Visual Polish (1-2 hours) - IMMEDIATE IMPACT

**Priority 1: Professional Photography**
- Replace `content/authors/admin/avatar.jpg` with high-quality professional photo
- Recommended: Lab setting, conference presentation, or studio shot
- Size: Square aspect ratio, minimum 400×400px
- **Impact:** Single biggest visual upgrade

**Priority 2: Visual Abstracts for Publications**
Create visual abstracts for top 3 publications:
- `content/publication/2024-ieee-tbcas/` (LUNA/EEGformer)
- `content/publication/2024-ieee-tim/` (Muscle pennation)
- One more recent publication

**Tools:** Canva (free), Figma, PowerPoint
- Size: 1200×630px
- File name: `featured.png` in each publication folder
- Include: Title, key finding, impact statement, author info
- Template in FINAL-SUMMARY.md

**Priority 3: Color Scheme Enhancement**
Choose and implement one of the color schemes from FINAL-SUMMARY.md:
- **Recommended:** Nordic Minimalist (fits Icelandic background)
- Update `data/themes/custom.toml` or modify existing theme
- Test in both light and dark modes

**Priority 4: Typography Refinement**
- Select font pairing from recommendations (Inter + Source Sans Pro suggested)
- Update in `config/_default/params.yaml`
- Test readability on mobile

**Priority 5: Micro-animations**
Add to `assets/scss/custom.scss`:
- Card hover effects (lift + shadow)
- Smooth transitions
- Scroll animations
- Code examples provided in FINAL-SUMMARY.md

### Phase 2: Content Richness (3-5 hours)

**Task 1: Add Student Projects**
Edit `content/home/available-projects.md`:
- Add 2-3 real project opportunities
- Use template from FINAL-SUMMARY.md
- Include prerequisites, duration, type

**Task 2: Research Journey Narrative**
Create new section or blog post:
- "From Iceland to ETH Zurich"
- Tell the story of your research evolution
- 300-500 words, engaging tone

**Task 3: Blog Posts**
Write 2-3 technical blog posts:
- NeurIPS 2024 acceptance announcement
- Behind-the-scenes of LUNA development
- Research philosophy or methodology

**Task 4: Infographics**
Create at least one:
- Research process flowchart
- Technology stack diagram
- LUNA efficiency comparison chart

**Task 5: Testimonials**
If available, add quotes from:
- Prof. Benini
- Collaborators
- Students

### Phase 3: Interactive Elements (5-10 hours)

**Task 1: Animated Statistics Counters**
Create JavaScript for counting animations:
- Citations: 687
- Publications: 15+
- Students supervised: [number]
- Trigger on scroll into view

**Task 2: Enhanced Hero Section**
Create custom widget `layouts/partials/widgets/custom-hero.html`:
- Full-width banner
- Large professional photo (50% width)
- Stats cards
- Gradient background
- Reference design in FINAL-SUMMARY.md

**Task 3: Publication Filtering**
Add JavaScript filtering to publications:
- By year (2024, 2023, 2022, etc.)
- By topic (Foundation Models, TinyML, Biosignals)
- By venue (Journal, Conference)
- Smooth animations

**Task 4: Research Timeline**
Create interactive horizontal timeline:
- BSc (University of Iceland) → MSc (ETH) → PhD (ETH) → Postdoc (ETH)
- Key papers at each milestone
- Clickable for details

**Task 5: Live GitHub Stats**
Integrate live stats for BioFoundation:
- Star count (currently 27)
- Fork count (currently 2)
- Recent commits
- Use GitHub API or widget

### Phase 4: Advanced Features (10+ hours)

**Task 1: Interactive Demos**
- Embed Google Colab notebook for LUNA
- Create "Try it yourself" section
- Link to HuggingFace demo if available

**Task 2: 3D Visualizations**
- Network architecture explorer
- Interactive diagram of LUNA model
- Use Three.js or similar

**Task 3: Custom Illustrations**
- Hire designer or create custom graphics
- Illustrate research concepts
- Create branded iconography

**Task 4: Video Content**
- 2-3 minute research explainer
- Record and embed conference presentations
- Lab tour or behind-the-scenes

**Task 5: Performance Optimization**
- Image optimization (WebP format)
- Lazy loading
- Code splitting
- Lighthouse score 90+

---

## Content & Assets Needed

### Provided Information

**Research Details:**
- h-index: 12
- Citations: 687
- Position: Postdoctoral Researcher, ML Group Leader
- Institution: ETH Zurich, working with Prof. Luca Benini

**Featured Work:**
- LUNA paper: https://arxiv.org/abs/2510.22257
- GitHub: https://github.com/pulp-bio/biofoundation (27 stars, 2 forks)
- YouTube: https://www.youtube.com/watch?v=1KPFJlJaXTI
- NeurIPS 2024 acceptance (poster)

**Research Interests:**
1. Foundation Models for Biosignals
2. Tiny Recursion Models & Deep Supervision
3. Edge AI Deployment on Constrained Devices

### Assets to Request/Create

**From Thorir:**
- [ ] Professional photographs (hero shot, action shots, lab photos)
- [ ] Student project descriptions (2-3 opportunities)
- [ ] Testimonial quotes (if available)
- [ ] Any awards or grants to highlight
- [ ] Preferred color scheme choice

**To Create:**
- [ ] Visual abstracts for top 3 publications
- [ ] Infographics (research process, tech stack)
- [ ] Blog post content
- [ ] Research journey narrative
- [ ] Custom icons/illustrations

---

## Design Guidelines

### Visual Style

**Tone:** Professional, modern, memorable but not flashy

**Inspiration:**
- Chris Olah's blog (https://colah.github.io/) - Excellent visualization
- Distill.pub - Beautiful research communication
- Microsoft Research pages - Clean, professional

**Colors:**
- Primary: Professional blue/teal (current: ocean theme)
- Accent: Vibrant but tasteful (orange, bright blue, or coral)
- Backgrounds: White/light gray with subtle variations

**Typography:**
- Headings: Bold, modern (Inter, Montserrat, Raleway)
- Body: Highly readable (Source Sans Pro, Open Sans)
- Minimum 16px body text, 18px preferred

### Interactive Elements

**Animations:**
- Subtle, not distracting
- Smooth transitions (0.3s ease)
- Respect `prefers-reduced-motion`

**Hover States:**
- Cards: Lift + shadow
- Buttons: Slight lift + color shift
- Links: Underline or color change

**Loading States:**
- Skeleton screens preferred
- No blank pages
- Progressive enhancement

### Accessibility Requirements

**Must Have:**
- Contrast ratio minimum 4.5:1
- All images with alt text
- Keyboard navigation
- Screen reader friendly
- Focus indicators visible
- Touch targets 44×44px minimum (mobile)

---

## Testing Checklist

Before considering any task complete, verify:

### Local Testing
- [ ] Hugo server runs without errors (`hugo server --disableFastRender`)
- [ ] All pages render correctly
- [ ] No console errors in browser
- [ ] Links work (internal and external)
- [ ] Images load and display correctly
- [ ] Responsive on mobile (iPhone, Android)
- [ ] Responsive on tablet (iPad)
- [ ] Responsive on desktop (various widths)

### Visual Testing
- [ ] Typography is readable
- [ ] Colors have sufficient contrast
- [ ] Spacing is consistent
- [ ] Animations are smooth
- [ ] No layout shifts
- [ ] Images are optimized (not too large)

### Functional Testing
- [ ] Navigation works (all menu items)
- [ ] Forms work (if applicable)
- [ ] Filtering works (publications, etc.)
- [ ] Interactive elements respond correctly
- [ ] Videos play (YouTube embeds)
- [ ] Dark mode works (if enabled)

### Performance Testing
- [ ] Lighthouse score 80+ (preferably 90+)
- [ ] Page load under 3 seconds
- [ ] Images lazy load
- [ ] No render-blocking resources

### Cross-Browser Testing
- [ ] Chrome/Chromium
- [ ] Firefox
- [ ] Safari (macOS/iOS)
- [ ] Edge

---

## Git Workflow

**Current Branch:** `claude/analyze-personal-website-011CV3o1H7QAN4BzgyteG6pK`

**Workflow:**
```bash
# Pull latest
git pull origin claude/analyze-personal-website-011CV3o1H7QAN4BzgyteG6pK

# Make changes
# Test locally with: hugo server --disableFastRender

# Commit
git add .
git commit -m "Clear description of changes"

# Push
git push origin claude/analyze-personal-website-011CV3o1H7QAN4BzgyteG6pK
```

**Commit Message Guidelines:**
- Clear, descriptive messages
- Reference what was changed and why
- Example: "Add visual abstracts for top 3 publications"
- Example: "Implement Nordic Minimalist color scheme"

**When to Commit:**
- After completing each discrete task
- After significant progress on larger tasks
- Before making risky changes (so you can revert)
- At end of work session

---

## Communication & Documentation

### Progress Updates

Create a `PROGRESS.md` file to track your work:

```markdown
# Enhancement Progress

## Completed
- [x] Task name - Date completed - Notes

## In Progress
- [ ] Task name - Current status - Blockers/questions

## Planned
- [ ] Task name - Dependencies - Estimated time
```

### Questions & Blockers

If you encounter issues:
1. Check FINAL-SUMMARY.md for solutions
2. Review relevant planning documents
3. Search Hugo/Wowchemy documentation
4. Document the blocker in PROGRESS.md
5. Propose alternative approaches

### Asset Requests

If you need assets from Thorir, document clearly:
- What you need
- Why you need it
- Specifications (size, format, content)
- Priority/urgency
- Where it will be used

---

## Quality Standards

### Code Quality
- Follow existing code style
- Comment complex logic
- Use semantic HTML
- Organize CSS logically
- No inline styles (use classes)

### Content Quality
- Proofread all text
- Verify all facts/numbers
- Check all links
- Optimize all images
- Consistent tone and voice

### Design Quality
- Consistent spacing
- Aligned elements
- Readable typography
- Accessible colors
- Professional appearance

---

## Success Metrics

The redesign is successful when:

**Visual Appeal:**
- [ ] Site looks professional and modern
- [ ] Distinct from generic academic templates
- [ ] Reflects leadership positioning
- [ ] Memorable and engaging

**User Experience:**
- [ ] Easy to navigate
- [ ] Fast loading
- [ ] Mobile-friendly
- [ ] Accessible to all

**Content:**
- [ ] Research clearly presented
- [ ] Collaboration opportunities obvious
- [ ] Student projects visible
- [ ] Recent work highlighted

**Technical:**
- [ ] No errors or warnings
- [ ] Builds successfully
- [ ] Works on all devices
- [ ] Lighthouse score 90+

**Impact:**
- [ ] Positions Thorir as group leader
- [ ] Attracts potential students
- [ ] Encourages collaboration
- [ ] Showcases research effectively

---

## Resources & Tools

### Free Design Tools
- **Canva** - Visual abstracts, infographics (https://canva.com)
- **Figma** - UI/UX design (https://figma.com)
- **Unsplash** - Stock photos (https://unsplash.com)
- **Pexels** - Stock photos (https://pexels.com)
- **Coolors** - Color palettes (https://coolors.co)
- **Google Fonts** - Typography (https://fonts.google.com)
- **Font Awesome** - Icons (https://fontawesome.com)

### Development Tools
- **Hugo Documentation** - https://gohugo.io/documentation/
- **Wowchemy Docs** - https://wowchemy.com/docs/
- **CodePen** - Test snippets (https://codepen.io)
- **TinyPNG** - Image compression (https://tinypng.com)
- **WebAIM** - Contrast checker (https://webaim.org/resources/contrastchecker/)

### Testing Tools
- **Lighthouse** - Performance audit (built into Chrome DevTools)
- **PageSpeed Insights** - https://pagespeed.web.dev/
- **BrowserStack** - Cross-browser testing (paid, has free tier)
- **Responsive Design Checker** - Various online tools

### Inspiration
- **Dribbble** - Search "academic portfolio"
- **Behance** - Search "researcher website"
- **Awwwards** - Web design excellence
- **Academic websites** in FINAL-SUMMARY.md

---

## Troubleshooting Common Issues

### Hugo Build Errors

**"File.UniqueID" errors:**
- This means Hugo version is too new (v0.152+)
- Downgrade to Hugo v0.111.3 or similar
- The theme is incompatible with v0.152

**"deprecated paginate" warning:**
- Already fixed in config.yaml
- Uses `pagination.pagerSize` instead

**"WC_POST_CSS" security error:**
- Already fixed in config.yaml
- Security whitelist added

**"GoogleAnalytics" field error:**
- Already fixed with custom partial
- Uses `layouts/partials/marketing/google_analytics.html`

### Theme Issues

**Custom theme not working:**
- Check `data/themes/` file syntax
- Ensure all required color variables defined
- For now, use built-in "ocean" theme
- Custom theme needs ALL Wowchemy variables

**CSS not applying:**
- Check `assets/scss/custom.scss` syntax
- Verify Hugo can compile SCSS
- Clear browser cache
- Check browser console for errors

### Content Issues

**Images not showing:**
- Check file path (relative to content file)
- Verify image exists in correct folder
- Check file name spelling/case
- Use browser dev tools to inspect

**YouTube not embedding:**
- Verify video ID is correct
- Check embed code in project file
- Ensure responsive wrapper div is present

---

## Final Notes

### Philosophy

Remember the goal: Transform a functional academic website into an **exceptional showcase of research leadership** that:
- Positions Thorir as an emerging leader in TinyML and biomedical AI
- Makes collaboration opportunities immediately obvious
- Attracts talented students to open projects
- Demonstrates expertise through beautiful presentation
- Remains professional, not flashy

### Balance

Strive for:
- **Professional** yet approachable
- **Modern** yet timeless
- **Visual** yet content-focused
- **Interactive** yet performant
- **Distinctive** yet accessible

### Iteration

Start with high-impact, low-effort improvements (Phase 1), then progressively enhance. Get feedback, iterate, improve. The website is never "done" - it evolves with the research.

---

## Quick Start Checklist

To begin work immediately:

**Setup (5 minutes)**
- [ ] Clone repository
- [ ] Install Hugo v0.111.3 (or compatible)
- [ ] Run `hugo server --disableFastRender`
- [ ] Verify site builds at http://localhost:1313

**Research (15 minutes)**
- [ ] Read FINAL-SUMMARY.md completely
- [ ] Skim high-level-plan.md and website-layout-redesign.md
- [ ] Review current site (http://localhost:1313)
- [ ] Identify quick wins

**First Task (1 hour)**
- [ ] Choose Phase 1, Priority 1-3 from above
- [ ] Gather necessary assets/tools
- [ ] Implement
- [ ] Test
- [ ] Commit and push

**Ongoing**
- [ ] Update PROGRESS.md daily
- [ ] Test frequently
- [ ] Commit often
- [ ] Document decisions
- [ ] Request feedback

---

## Contact & Questions

If you need clarification or encounter blockers:
1. Re-read FINAL-SUMMARY.md (most answers are there)
2. Check other planning documents
3. Document specific questions clearly
4. Include context and what you've tried
5. Propose potential solutions

---

**You have everything you need to create an outstanding academic website. Good luck!** 🚀

---

*Last updated: November 12, 2024*
*For: Personal website enhancement project*
*Subject: Thorir Mar Ingolfsson - Academic Portfolio*
