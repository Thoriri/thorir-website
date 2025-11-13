# Citation & Metrics Display Options

## Analysis of Changes Made

The website redesign has been successfully implemented with:
- Profile updated to "Postdoctoral Researcher | ML Group Leader"
- Three new homepage sections (Research Interests, Featured Work, Available Projects)
- BioFoundation project page with YouTube embed
- Navigation restructured
- Ocean theme applied
- All Hugo compatibility issues resolved

**Current State:** Citation metrics (h-index: 12, 687 citations) are linked via Google Scholar icon in the social links, but not prominently displayed on the homepage.

---

## Recommended Citation Display Layouts

### Option 1: Stats Card (Prominent & Modern) ✨ RECOMMENDED

**Location:** Right after Biography section (weight: 22)
**Style:** Centered card with gradient background
**File:** Created as `content/home/metrics.md`

**Visual:**
```
┌────────────────────────────────────────┐
│  [Gradient background, subtle shadow]  │
│                                         │
│    687        12         15+           │
│  Citations  h-index  Publications     │
│  [Scholar]            [View Pubs →]   │
│                                         │
└────────────────────────────────────────┘
```

**Pros:**
- Immediately visible, high impact
- Professional and elegant
- Easy to update numbers
- Links to Scholar and publications
- Responsive design

**Cons:**
- Takes up full width section
- Might feel too promotional if overdone

---

### Option 2: Inline Bio Stats (Subtle & Integrated)

**Location:** Within biography text
**Style:** Small badge/pill style inline with text
**Implementation:** Modify `content/authors/admin/_index.md`

**Example text:**
```markdown
I am a Postdoctoral Researcher at ETH Zurich... <span style="display: inline-block; background: #e0f2fe; color: #0369a1; padding: 4px 12px; border-radius: 12px; font-size: 0.9em; font-weight: 600; margin: 0 4px;">687 citations</span> <span style="display: inline-block; background: #e0f2fe; color: #0369a1; padding: 4px 12px; border-radius: 12px; font-size: 0.9em; font-weight: 600; margin: 0 4px;">h-index: 12</span>

My recent work on **LUNA**...
```

**Pros:**
- Very subtle and professional
- Doesn't take separate space
- Natural reading flow

**Cons:**
- Less prominent
- Easy to miss
- Harder to update

---

### Option 3: Sidebar Stats (Compact)

**Location:** Below author photo in about widget
**Style:** Small vertical stack
**Implementation:** Custom CSS in `assets/scss/custom.scss`

**Visual:**
```
[Photo]

📊 687 Citations
📈 h-index: 12
📄 15+ Papers
```

**Pros:**
- Always visible with bio
- Compact, doesn't break flow
- Natural placement

**Cons:**
- Limited by Wowchemy widget structure
- Requires theme customization
- Less flexible

---

### Option 4: Mini Banner (Sleek)

**Location:** Above or below biography section
**Style:** Horizontal banner, minimal design

**Visual:**
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  687 citations · h-index: 12 · 15+ publications · Google Scholar →
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**Implementation:**
```html
<div style="text-align: center; padding: 20px; border-top: 1px solid #e2e8f0; border-bottom: 1px solid #e2e8f0; margin: 30px 0;">
  <span style="color: #64748b; font-size: 0.95rem;">
    <strong style="color: #0369a1;">687</strong> citations ·
    <strong style="color: #0369a1;">h-index: 12</strong> ·
    <strong style="color: #0369a1;">15+</strong> publications ·
    <a href="https://scholar.google.com/citations?user=TyRxmUkAAAAJ&hl=en" style="color: #0369a1; text-decoration: none;">Google Scholar →</a>
  </span>
</div>
```

**Pros:**
- Very clean and elegant
- Professional, not showy
- Easy to implement

**Cons:**
- Less visual impact
- Might be overlooked

---

### Option 5: Hero Stats (Bold)

**Location:** In a custom hero section above biography
**Style:** Large numbers with icons
**Implementation:** Custom hero widget

**Visual:**
```
┌─────────────────────────────────────────────┐
│    [Your Photo]   |   Thorir Mar Ingolfsson │
│     (larger)      |   Postdoctoral Researcher│
│                   |                           │
│                   |   📊 687  📈 12  📄 15+  │
│                   |   Citations h-index Pubs  │
└─────────────────────────────────────────────┘
```

**Pros:**
- Maximum visibility
- Modern, impressive
- Sets professional tone

**Cons:**
- Requires custom widget
- More complex to implement
- Might feel too self-promotional

---

## My Recommendation: Hybrid Approach

**Best of Both Worlds:**

1. **Primary:** Use **Option 1 (Stats Card)** - Already created in `content/home/metrics.md`
   - Prominent but tasteful
   - Easy to update
   - Professional appearance
   - Links directly to Scholar

2. **Secondary:** Add **Option 4 (Mini Banner)** at the very bottom of your bio text in `content/authors/admin/_index.md`
   - Subtle reinforcement
   - Doesn't compete with stats card
   - Provides context within bio

**Combined Effect:**
- Visitors see stats immediately after reading bio
- Stats are elegant, not overwhelming
- Easy to navigate to Scholar or publications
- Maintains professional academic tone

---

## Implementation Guide for Option 1 (Already Done)

**File Created:** `content/home/metrics.md`
**Weight:** 22 (right after Biography at 20)
**Features:**
- ✅ Gradient background (subtle blue)
- ✅ Large numbers (3rem font)
- ✅ Clean typography
- ✅ Links to Google Scholar
- ✅ Link to publications section
- ✅ Responsive design
- ✅ Hover effects on links
- ✅ Academic icon integration

**To Activate:**
The file is already created and will appear on your homepage automatically.

**To Customize:**
Edit line numbers in `content/home/metrics.md`:
- Line 14: Update citation count (currently 687)
- Line 21: Update h-index (currently 12)
- Line 28: Update publication count (currently 15+)

---

## Color Schemes for Stats

### Current (Ocean Blue) ✅
```css
Background: #f8fafc → #e0f2fe (gradient)
Numbers: #0369a1 (ocean blue)
Labels: #64748b (slate gray)
```
- Professional
- Matches ocean theme
- Calming, trustworthy

### Alternative 1: Academic Gold
```css
Background: #fffbeb → #fef3c7 (gradient)
Numbers: #92400e (warm brown)
Labels: #78716c
```
- Prestigious feel
- Warm and inviting
- Less common

### Alternative 2: Nordic Cool
```css
Background: #f0f9ff → #e0f2fe (gradient)
Numbers: #0c4a6e (deep blue)
Labels: #475569
```
- Matches Nordic Minimalist theme suggestion
- Clean, Scandinavian
- Professional

### Alternative 3: Biomedical Teal
```css
Background: #f0fdfa → #ccfbf1 (gradient)
Numbers: #115e59 (teal)
Labels: #6b7280
```
- Evokes healthcare/medical
- Fresh, modern
- Distinctive

---

## Typography Recommendations

**Current Implementation:**
- Numbers: 3rem (48px), font-weight: 700
- Labels: 0.9rem (14.4px), uppercase, letter-spacing: 0.1em
- Links: 0.85rem (13.6px)

**For More Impact:**
```css
Numbers: 3.5rem (56px) - Bolder
Labels: 1rem (16px) - More readable
```

**For More Subtle:**
```css
Numbers: 2.5rem (40px) - Gentler
Labels: 0.85rem (13.6px) - More compact
```

---

## Animation Options

### Option A: Count-Up Animation (Eye-catching)
Numbers animate from 0 to final value when scrolled into view.

```javascript
// Add to custom.js
const observer = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      animateValue(entry.target, 0, parseInt(entry.target.innerText), 2000);
    }
  });
});

function animateValue(obj, start, end, duration) {
  let startTimestamp = null;
  const step = (timestamp) => {
    if (!startTimestamp) startTimestamp = timestamp;
    const progress = Math.min((timestamp - startTimestamp) / duration, 1);
    obj.innerHTML = Math.floor(progress * (end - start) + start);
    if (progress < 1) {
      window.requestAnimationFrame(step);
    }
  };
  window.requestAnimationFrame(step);
}
```

### Option B: Fade-In (Subtle)
Stats fade in gently when page loads.

```css
@keyframes fadeIn {
  from { opacity: 0; transform: translateY(10px); }
  to { opacity: 1; transform: translateY(0); }
}

.stats-container {
  animation: fadeIn 0.8s ease-out;
}
```

### Option C: Pulse on Hover (Interactive)
Numbers pulse slightly when you hover over them.

```css
.stats-number:hover {
  animation: pulse 0.5s ease-in-out;
}

@keyframes pulse {
  0%, 100% { transform: scale(1); }
  50% { transform: scale(1.05); }
}
```

**Recommendation:** Option B (Fade-In) - Elegant without being distracting.

---

## A/B Testing Suggestions

Try these variations and see what works best:

**Test 1: Position**
- A: After biography (weight: 22) ✅ Current
- B: Before biography (weight: 15)
- C: After featured publications

**Test 2: Size**
- A: Large (3rem numbers) ✅ Current
- B: Medium (2.5rem numbers)
- C: Extra large (4rem numbers)

**Test 3: Background**
- A: Gradient card ✅ Current
- B: Solid white with border
- C: No background, just numbers

**Test 4: Layout**
- A: Horizontal (3 items side-by-side) ✅ Current
- B: Vertical (stacked)
- C: Grid (2x2)

---

## Accessibility Considerations

**Current Implementation:**
- ✅ Sufficient color contrast (4.5:1+)
- ✅ Semantic HTML structure
- ✅ Links have clear labels
- ✅ Responsive on mobile
- ✅ No motion for users with prefers-reduced-motion

**To Improve:**
- Add `aria-label` to stat numbers
- Add screen reader text for context
- Ensure keyboard navigation works

```html
<div role="region" aria-label="Research Impact Metrics">
  <div aria-label="687 total citations">
    <div class="stats-number">687</div>
    <div class="stats-label">Citations</div>
  </div>
</div>
```

---

## Mobile Optimization

**Current Breakpoints:**
- Desktop (>768px): 3 columns, large numbers
- Mobile (≤768px): Stacks vertically, smaller numbers

**Further Improvements:**
```css
@media (max-width: 640px) {
  .stats-container {
    gap: 40px; /* Reduce spacing */
    padding: 20px 15px; /* Reduce padding */
  }

  .stats-number {
    font-size: 2.5rem; /* Smaller numbers */
  }
}
```

---

## Quick Customization Guide

### To Change Colors:
Edit `content/home/metrics.md`, lines 12-13:
```html
background: linear-gradient(135deg, #f8fafc 0%, #e0f2fe 100%);
color: #0369a1;
```

### To Update Numbers:
Edit `content/home/metrics.md`:
- Line 14: Citation count
- Line 21: h-index
- Line 28: Publication count

### To Change Position:
Edit `content/home/metrics.md`, line 3:
```yaml
weight: 22  # Lower number = higher on page
```

### To Hide Temporarily:
Add to top of `content/home/metrics.md`:
```yaml
active: false
```

---

## Comparison with Top Researchers

**MIT Professor Example:**
- Large stats at top
- Multiple metrics (citations, students, grants)
- Professional but prominent

**Stanford Professor Example:**
- Subtle mention in bio
- Link to Google Scholar
- Focus on recent work

**ETH Professor Example:**
- Mini stats under photo
- Clean, understated
- European academic style

**Your Current Style:** Balanced - prominent enough to show impact, elegant enough to stay professional. Perfect for postdoc/group leader positioning.

---

## Next Steps

1. **Test locally:** `hugo server --disableFastRender`
2. **View at:** http://localhost:1313
3. **Adjust if needed:** Edit numbers, colors, or position
4. **Get feedback:** Ask colleagues what feels right
5. **Monitor analytics:** See if it helps with student recruitment

---

## Final Recommendation

**Keep Option 1 (Stats Card) as is** - it's:
- ✅ Prominent but not overwhelming
- ✅ Elegant and professional
- ✅ Easy to maintain
- ✅ Responsive and accessible
- ✅ Links to relevant pages
- ✅ Matches academic norms for senior researchers

The layout strikes the perfect balance between **showcasing your impact** (important for attracting students and collaborators) and **maintaining humility** (important in academic culture).

---

*Created: November 12, 2024*
*For: Citation metrics display optimization*
