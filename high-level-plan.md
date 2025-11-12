# Website Redesign: High-Level Plan
## Transition from PhD Student to High-Profile Postdoctoral Researcher

---

## Executive Summary

**Current State**: Functional but forgettable academic website presenting a competent PhD student who finished their work.

**Desired State**: Dynamic, engaging website showcasing an exciting postdoctoral researcher that academics and students want to collaborate with.

**Core Challenge**: Strong research credentials (IEEE TBioCAS, TIM publications) are buried under passive language and generic academic presentation.

---

## Current Strengths ✓

### Technical Foundation
- Clean Hugo/Wowchemy setup with proper deployment pipeline
- Well-structured publication management
- Professional theme (minimal, academic-focused)
- Google Analytics tracking in place

### Strong Research Profile
- **Impressive publications**: IEEE TBioCAS and IEEE TIM (2024) are high-quality venues
- **Clear research focus**: TinyML, wearable devices, biomedical signal processing
- **Practical impact**: Real-world applications (seizure detection, muscle monitoring)
- **Technical depth**: Hardware-aware ML, embedded systems expertise

### Good Academic Fundamentals
- Links to Google Scholar, GitHub, LinkedIn
- Contact information properly displayed
- Conference talks documented

---

## Critical Issues (Fix Immediately) ⚠️

### 1. Outdated Title & Status
**Location**: `/content/authors/admin/_index.md` Line 3
```yaml
role: "PhD Student in Robust and Practical Machine Learning "
```
**Fix**: Update to postdoc status immediately.

### 2. Outdated Biography
**Location**: `/content/authors/admin/_index.md` Lines 44-45
> "Hi! I am Thorir, and I am currently a Ph.D. student at ETH Zurich..."

**Fix**: Rewrite to reflect postdoc status.

### 3. PhD Graduation Missing
Education section shows PhD enrollment but no completion year - unclear whether finished.

---

## Major Strategic Issues (High Priority) 🎯

### 1. Weak Personal Branding & Positioning

**Current** (humble/generic):
> "My research interest is applying Robust and Practical Machine Learning approaches..."

**What high-profile researchers do** (confident/specific):
> "I develop efficient machine learning systems for biomedical wearables that operate under extreme resource constraints. My work enables real-time seizure detection and physiological monitoring on devices consuming less than 50mW."

**Action**: Rewrite bio to:
- Lead with unique value proposition
- Highlight quantifiable achievements (e.g., "0.31 mJ per inference," "88% seizure detection")
- Use active, confident language
- Mention awards, grants, or recognition

### 2. Missing Research Impact Narrative

**Problem**: Strong publications but no story connecting them. Visitors can't quickly understand:
- Why your work matters
- What problems you solve
- What makes your approach unique
- Where your research is heading

**Action**: Add "Research Vision" section that:
- Articulates research mission
- Explains real-world impact
- Shows progression of work
- Positions you as thought leader in TinyML + biomedical applications

### 3. Insufficient Visibility Features

**What's Missing**:
- Research highlights/metrics (citation count, h-index)
- Media/press section
- Visible collaborations
- Grants/Awards display
- Teaching/Mentoring evidence

**Action**: Add these sections to demonstrate standing in community.

### 4. Weak Visual Presentation

**Current Issues**:
- Theme is "minimal" but feels bare/austere rather than "elegant minimal"
- No visual identity beyond standard academic template
- No hero section or compelling visual entry point
- Featured publications not optimized

**Action**:
- Add striking hero section with professional photo and tagline
- Consider custom color scheme
- Add visual abstracts to publications
- Optimize publication layouts

### 5. Limited Content Strategy

**Current State**:
- Only 2 blog posts (seemingly old)
- Only 1 project (from 2019)
- No recent news/updates section
- No evidence of community engagement

**Action**: Establish regular content updates and community presence.

---

## Design & User Experience Issues 🎨

### 1. Navigation Structure
**Current**: `Home → Posts → Projects → Talks → Publications → Contact`

**Issues**:
- "Posts" prioritized over "Publications"
- No "Research" overview page
- Empty "Projects" section

**Recommended**: `Home → Research → Publications → Projects → Blog → Contact`

### 2. Homepage Widget Order
**Current** (by weight):
```
Biography (20) → Posts (60) → Projects (65) → Talks (70) → Featured (80) → Publications (90) → Contact (130)
```

**Issues**:
- Featured publications come late (weight: 80)
- Blog posts prioritized over research outputs

**Recommended**:
```
Biography (20) → Featured Publications (30) → Research Overview (40) → Recent Publications (50) → Projects (60) → Talks (70) → Blog (80) → Contact (130)
```

### 3. No Social Proof
- No testimonials from collaborators
- No mention of paper citations or impact
- No visible peer recognition

---

## Content Gaps for "High-Profile" Positioning 📊

### Must-Have Sections

1. **Research Statement/Vision Page**
   - 5-year research agenda
   - Big problems being tackled
   - Unique approach/methodology

2. **Active Blog/News Section**
   - Conference attendance updates
   - Paper acceptances
   - Collaboration opportunities
   - Technical insights (biweekly posts ideal)

3. **Collaboration Page**
   - "I'm looking for collaborators in..."
   - "Open positions" (if mentoring students)
   - "Available for..." (talks, reviews, etc.)

4. **Media & Outreach**
   - Press mentions
   - Invited talks
   - Podcast appearances
   - Social media engagement

5. **Teaching & Mentoring**
   - Supervised students
   - Courses taught
   - Workshop organization

6. **Code & Datasets**
   - Open-source contributions
   - Released datasets
   - Reproducibility efforts
   - GitHub stars/forks

---

## Specific Technical Recommendations 🔧

1. Update theme to "ocean" or "mr robot" (more distinctive than "minimal")
2. Add avatar hover effect with contact icons
3. Enable dark mode toggle
4. Add Google Scholar stats widget/badge
5. Create custom homepage with hero banner
6. Add paper thumbnails (visual abstracts) to all publications
7. Remove example publication (`/content/publication/example/`)
8. Update or remove Projects section if not actively maintaining
9. Add PDF links for all publications (upload to `/static/uploads/`)
10. Create prominent CV download link

---

## Competitive Positioning Analysis 🎓

### Competition
- Other postdocs in TinyML/embedded ML space
- Early-career researchers at top institutions
- Industry ML engineers transitioning to academia

### Your Unique Angle
- **Intersection of three hot areas**: TinyML + Biomedical + Edge Computing
- **Practical deployments**: Systems that work on real hardware, not just papers
- **Quantifiable efficiency**: Energy/latency metrics that matter
- **Medical impact**: Solving real patient problems

### Amplification Strategy
1. **Tagline**: "Building intelligent wearable systems that save lives and consume microwatts"
2. **Lead with numbers**: "My work enables AI on devices 1000x more power-efficient than smartphones"
3. **Show the stack**: "From algorithm design to silicon validation"

---

## Implementation Roadmap 📋

### Phase 1: Essential Updates (Immediate - Day 1)
1. ✅ Update title to "Postdoctoral Researcher" with specific focus
2. ✅ Rewrite biography with confident, impact-focused language
3. ✅ Add PhD completion year to education
4. ✅ Update meta description in params.yaml

### Phase 2: Content Enhancement (Week 1)
5. ✅ Create Research Vision/Overview page
6. ✅ Add research metrics (citations, h-index) if favorable
7. ✅ Write 2-3 new blog posts about recent work
8. ✅ Add visual abstracts to featured publications
9. ✅ Create "Looking for Collaborators" section
10. ✅ Remove or update outdated content

### Phase 3: Design Overhaul (Weeks 2-3)
11. ✅ Redesign homepage with hero section
12. ✅ Change theme to something more distinctive
13. ✅ Reorganize content priority (publications first)
14. ✅ Add custom CSS for personality
15. ✅ Professional photo shoot if current avatar is dated

### Phase 4: Ongoing Maintenance (Continuous)
16. ✅ Bi-weekly blog posts
17. ✅ Update news section with paper acceptances
18. ✅ Engage on Twitter/academic social media
19. ✅ Share work-in-progress and behind-the-scenes content

---

## Key Messaging Framework

### Elevator Pitch (30 seconds)
"I develop ultra-efficient machine learning systems for medical wearables. My research enables real-time AI on devices consuming less power than a hearing aid - making life-saving applications like seizure detection and physiological monitoring accessible anywhere."

### Research Mission (2 minutes)
"The future of healthcare depends on intelligent wearable devices that can monitor and respond to our bodies 24/7. But current AI systems are too power-hungry for true wearability. I bridge this gap by co-designing algorithms and hardware, achieving 1000x efficiency improvements while maintaining clinical-grade accuracy. My work spans the full stack - from neural architecture search to silicon deployment - with real-world impact in epilepsy care and rehabilitation monitoring."

### Collaboration Value Proposition
"I bring expertise at the intersection of machine learning, embedded systems, and biomedical engineering. I'm looking for collaborations that push the boundaries of:
- TinyML for medical applications
- Hardware-aware neural architecture design
- Real-time biosignal processing on edge devices
- Practical deployment of AI in resource-constrained environments"

---

## Metrics to Track

### Research Impact
- Citation count and h-index
- Paper downloads
- GitHub stars/forks for code releases
- Dataset downloads
- Media mentions

### Website Engagement
- Monthly unique visitors
- Time on site
- Pages per session
- Collaboration inquiries received
- Student interest in projects

### Community Presence
- Conference presentations (invited vs. regular)
- Workshop organization
- Program committee memberships
- Collaborations initiated through website

---

## Bottom Line

**You have strong research credentials** but they're **buried under passive language and generic presentation**.

**To "sell yourself" effectively in academia**, you need to:
1. **Be more assertive** about your expertise and impact
2. **Tell a compelling story** about your research vision
3. **Show personality and distinctiveness** in design and content
4. **Demonstrate active engagement** with the community
5. **Make collaboration opportunities explicit**

**The technical foundation is solid** - this is primarily a **content and positioning challenge**, not a technical rebuild.
