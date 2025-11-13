# Visual Abstracts Guide
## Phase 1, Priority 2: Visual Abstracts for Publications

### Overview

Visual abstracts are graphical summaries that make research immediately understandable and shareable. They're perfect for social media, presentations, and making your publications stand out.

### Target Publications

Based on your recent work, create visual abstracts for:

1. **2024 IEEE TBioCAS - EEGformer** (`content/publication/2024-ieee-tbcas/`)
   - Title: "Reducing False Alarms in Wearable Seizure Detection With EEGformer"
   - Key metrics: 20% latency reduction, 73% detection probability, 0.15 FP/h
   - Focus: Efficient transformer for MCU deployment

2. **2024 IEEE TIM - Muscle Pennation** (`content/publication/2024-ieee-tim/`)
   - Title: "A Muscle Pennation Angle Estimation Framework From Raw Ultrasound Data"
   - Key metrics: 1.6 RMSE, 11 kB memory, 1.31 ms inference, 43.32 uJ energy
   - Focus: Real-time muscle analysis on wearable devices

3. **2023 IEEE BioCAS - EEGformer Original** (`content/publication/2023-ieee-biocas/`)
   - Or choose another recent publication that's most impactful

### Technical Specifications

**File Requirements:**
- **Size:** 1200×630px (perfect for social media sharing)
- **Format:** PNG or JPEG
- **File Size:** Optimize to <500KB for web
- **Aspect Ratio:** 1.91:1 (landscape)
- **Resolution:** 72-150 DPI (web standard)
- **Color Space:** sRGB

**File Location:**
- Place as `featured.png` in each publication folder
- Example: `content/publication/2024-ieee-tbcas/featured.png`
- Replace existing `featured.PNG` if present

### Design Template Structure

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  [Your Logo/ETH Zurich]                    NeurIPS 2024   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐  │
│  │                                                       │  │
│  │  [Simple Diagram/Illustration]                      │  │
│  │  (Architecture, process, or key concept)            │  │
│  │                                                       │  │
│  └─────────────────────────────────────────────────────┘  │
│                                                             │
│  LUNA: Foundation Model for EEG Analysis                   │
│                                                             │
│  Key Finding: 300× fewer FLOPs                            │
│  Impact: Real-time EEG on wearables                        │
│                                                             │
│  Ingolfsson et al. | ETH Zurich                            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Content Guidelines

**Must Include:**
- ✅ Paper title (shortened if needed)
- ✅ 1-3 key findings/metrics
- ✅ Visual element (diagram, chart, or illustration)
- ✅ Venue/year (e.g., "IEEE TBioCAS 2024")
- ✅ Author attribution

**Should Include:**
- ✅ Impact statement (1 sentence)
- ✅ Visual metaphor or icon
- ✅ Your branding/logo

**Avoid:**
- ❌ Too much text (keep it minimal)
- ❌ Complex diagrams (simplify)
- ❌ Small fonts (minimum 24pt)
- ❌ Low contrast colors
- ❌ Cluttered layout

### Design Tools & Resources

#### Free Tools (Recommended)
1. **Canva** (https://canva.com)
   - Free academic templates
   - Easy drag-and-drop
   - Pre-sized templates (1200×630px)
   - Search: "research poster" or "social media graphic"

2. **Figma** (https://figma.com)
   - More design control
   - Free for personal use
   - Professional vector graphics

3. **PowerPoint/Keynote**
   - Set slide size to 1200×630px
   - Export as PNG
   - Good for simple layouts

#### Paid Options
- **Adobe Illustrator** - Professional vector graphics
- **BioRender** - Scientific diagrams (if needed)
- **Fiverr** - Hire designer ($20-50 per abstract)

### Template Designs for Each Publication

#### Template 1: EEGformer (2024 TBioCAS)

**Layout:**
```
[Left 60%]              [Right 40%]
┌─────────────┐         ┌──────────┐
│             │         │ Key Stats│
│ Transformer │         │          │
│ Architecture│         │ 20% ↓    │
│ Diagram     │         │ Latency  │
│             │         │          │
│ EEG Input → │         │ 73%      │
│ Model →     │         │ Detection│
│ Seizure Out │         │          │
│             │         │ 0.15 FP/h│
└─────────────┘         └──────────┘

Title: Reducing False Alarms in Wearable Seizure Detection
Venue: IEEE TBioCAS 2024
```

**Color Scheme:**
- Primary: Deep blue (#1e40af)
- Accent: Orange (#f97316) for metrics
- Background: White with subtle gradient

#### Template 2: Muscle Pennation (2024 TIM)

**Layout:**
```
[Top 40%]                    [Bottom 60%]
┌────────────────────────┐
│ Ultrasound Image       │
│ + Muscle Diagram       │
└────────────────────────┘
┌────────────────────────────────────┐
│ Key Metrics (4 boxes)              │
│ ┌────┐ ┌────┐ ┌────┐ ┌────┐       │
│ │1.6 │ │11kB│ │1.3 │ │43μJ│       │
│ │RMSE│ │Mem │ │ms  │ │En  │       │
│ └────┘ └────┘ └────┘ └────┘       │
└────────────────────────────────────┘

Title: Muscle Pennation Angle Estimation
Venue: IEEE TIM 2024
```

**Color Scheme:**
- Primary: Medical teal (#00897b)
- Accent: Coral (#ff6b6b)
- Background: Light gray (#f5f5f5)

#### Template 3: General Template

**Simple 3-Column Layout:**
```
┌──────────┬──────────┬──────────┐
│          │          │          │
│ Problem  │ Solution │ Impact   │
│          │          │          │
│ [Icon]   │ [Diagram]│ [Chart]  │
│          │          │          │
└──────────┴──────────┴──────────┘

Title | Venue | Authors
```

### Step-by-Step Creation Process

#### Using Canva (Easiest)

1. **Sign up** at canva.com (free)
2. **Create custom size:** 1200×630px
3. **Choose template** or start blank
4. **Add elements:**
   - Background (gradient or solid color)
   - Title text (large, bold)
   - Key metrics (highlighted boxes)
   - Simple diagram/icon
   - Venue/year
   - Your logo
5. **Export** as PNG (high quality)
6. **Optimize** using TinyPNG.com
7. **Save** as `featured.png` in publication folder

#### Using PowerPoint

1. **Set slide size:**
   - Design → Slide Size → Custom
   - Width: 12 inches, Height: 6.3 inches
2. **Design layout:**
   - Use shapes and text boxes
   - Insert icons from Insert → Icons
   - Use SmartArt for diagrams
3. **Export:**
   - File → Export → Change File Type → PNG
   - Select "Export All Slides" or current slide
4. **Optimize** and save

### Content for Each Publication

#### 1. EEGformer (2024 TBioCAS)

**Title:** "Reducing False Alarms in Wearable Seizure Detection With EEGformer"

**Key Points:**
- Compact transformer for MCU deployment
- 20% reduction in detection latency
- 73% seizure detection probability
- 0.15 false positives per hour
- Deployed on GAP8, GAP9, Apollo4 MCUs
- 13.7 ms inference, 0.31 mJ per inference

**Visual Elements:**
- Transformer architecture diagram
- EEG signal visualization
- MCU chip icon
- Performance comparison chart

**Impact Statement:**
"Enabling real-time seizure detection on wearable devices with multi-day battery life"

#### 2. Muscle Pennation (2024 TIM)

**Title:** "A Muscle Pennation Angle Estimation Framework From Raw Ultrasound Data"

**Key Points:**
- Direct estimation from raw ultrasound data
- 1.6° RMSE (comparable to expert annotations)
- Only 11 kB memory footprint
- 1.31 ms inference time
- 43.32 μJ energy consumption
- Real-time on GAP9 processor

**Visual Elements:**
- Ultrasound image of muscle
- Pennation angle diagram
- GAP9 processor icon
- Performance metrics visualization

**Impact Statement:**
"Real-time muscle analysis directly on wearable ultrasound probes"

### Design Best Practices

**Typography:**
- **Title:** Bold, 48-60pt, high contrast
- **Body text:** 24-32pt, readable
- **Metrics:** Extra bold, 36-48pt
- **Fonts:** Sans-serif (Arial, Helvetica, Inter, Montserrat)

**Colors:**
- **High contrast:** Text should be clearly readable
- **Brand colors:** Use your theme colors (blue/teal)
- **Accent colors:** Use for metrics/highlights
- **Background:** Light (white/light gray) or dark (if appropriate)

**Layout:**
- **Rule of thirds:** Divide into 3 sections
- **Visual hierarchy:** Most important info largest
- **White space:** Don't overcrowd
- **Alignment:** Keep elements aligned

**Accessibility:**
- **Contrast ratio:** Minimum 4.5:1 for text
- **Alt text:** Will be added in markdown
- **Readable fonts:** Avoid decorative fonts

### Testing Checklist

After creating visual abstracts:

- [ ] Size is exactly 1200×630px
- [ ] File size is <500KB (optimized)
- [ ] Text is readable at small sizes
- [ ] Colors have sufficient contrast
- [ ] All key information is included
- [ ] Visual elements are clear
- [ ] Branding/logo is present
- [ ] Saved as `featured.png` in correct folder
- [ ] Tested on website (displays correctly)
- [ ] Looks good in both light and dark modes

### Implementation Steps

1. **Create visual abstracts** using your chosen tool
2. **Save as `featured.png`** in each publication folder:
   - `content/publication/2024-ieee-tbcas/featured.png`
   - `content/publication/2024-ieee-tim/featured.png`
   - `content/publication/2023-ieee-biocas/featured.png` (or chosen third)
3. **Update publication markdown** if needed (usually not required)
4. **Test locally:**
   ```bash
   hugo server --disableFastRender
   ```
5. **Verify:**
   - Abstracts display on publication pages
   - Abstracts appear in publication listings
   - Images load quickly
   - Text is readable

### Example HTML/CSS Template (For Reference)

If you want to create a web-based template, here's a starting point:

```html
<!DOCTYPE html>
<html>
<head>
  <style>
    .visual-abstract {
      width: 1200px;
      height: 630px;
      background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
      padding: 40px;
      font-family: 'Arial', sans-serif;
    }
    .title {
      font-size: 48px;
      font-weight: bold;
      color: #1e40af;
      margin-bottom: 20px;
    }
    .metrics {
      display: flex;
      gap: 20px;
      margin: 30px 0;
    }
    .metric-box {
      background: white;
      padding: 20px;
      border-radius: 8px;
      text-align: center;
    }
    .metric-value {
      font-size: 36px;
      font-weight: bold;
      color: #f97316;
    }
  </style>
</head>
<body>
  <div class="visual-abstract">
    <div class="title">Paper Title Here</div>
    <div class="metrics">
      <div class="metric-box">
        <div class="metric-value">300×</div>
        <div>FLOPs Reduction</div>
      </div>
      <!-- More metrics -->
    </div>
  </div>
</body>
</html>
```

### Resources & Inspiration

**Academic Visual Abstract Examples:**
- Search "visual abstract" on Twitter/X
- Check Nature, Science, Cell journal social media
- Browse #VisualAbstract on social platforms

**Design Inspiration:**
- Dribbble: Search "research poster"
- Behance: Search "academic design"
- Pinterest: "scientific infographic"

**Icons & Graphics:**
- **Flaticon** (https://flaticon.com) - Free icons
- **The Noun Project** (https://thenounproject.com) - Icon library
- **Unsplash** (https://unsplash.com) - Free photos (if needed)

### Next Steps

1. **Choose your tool** (Canva recommended for beginners)
2. **Create first visual abstract** (start with EEGformer)
3. **Get feedback** (show to colleagues)
4. **Create remaining abstracts** using same style
5. **Optimize all images** (TinyPNG.com)
6. **Replace featured images** in publication folders
7. **Test on website**
8. **Update PROGRESS.md** when complete

---

**Estimated Time:** 2-3 hours for all 3 abstracts (if using Canva)

**Impact:** Makes publications immediately shareable and more engaging

**Priority:** HIGH - Visual abstracts significantly improve engagement

