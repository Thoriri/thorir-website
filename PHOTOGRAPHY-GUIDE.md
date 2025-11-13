# Professional Photography Guide
## Phase 1, Priority 1: Avatar Enhancement

### Current Status
- **Location:** `content/authors/admin/avatar.jpg`
- **Current Size:** 1365×1365px (square, meets requirements)
- **File Format:** JPEG

### Requirements for New Professional Photo

#### Technical Specifications
- **Aspect Ratio:** Square (1:1) - **REQUIRED**
- **Minimum Size:** 400×400px (current is 1365×1365px, which is excellent)
- **Recommended Size:** 800×800px to 2000×2000px
- **File Format:** JPEG or PNG
- **File Size:** Optimize to under 500KB for web performance
- **Color Space:** sRGB
- **Resolution:** 72-150 DPI (web standard)

#### Content Guidelines

**Recommended Settings:**
1. **Lab Setting** - Working with equipment, in research environment
2. **Conference Presentation** - Presenting at a podium or poster session
3. **Studio Shot** - Professional headshot with clean background
4. **Office Setting** - In your workspace at ETH Zurich

**What to Include:**
- ✅ Professional attire (business casual or formal)
- ✅ Good lighting (natural or professional)
- ✅ Clear, in-focus face
- ✅ Friendly, approachable expression
- ✅ Square crop (can crop from larger photo)

**What to Avoid:**
- ❌ Casual selfies
- ❌ Blurry or low-resolution images
- ❌ Busy backgrounds that distract
- ❌ Extreme angles or filters
- ❌ Group photos (should be solo)

### How to Replace the Photo

#### Option 1: Direct Replacement (Recommended)
1. Obtain your new professional photo
2. Ensure it's square (crop if needed)
3. Resize to 800-2000px square (maintain aspect ratio)
4. Optimize for web (compress to <500KB)
5. Replace `content/authors/admin/avatar.jpg` with the new file
6. Keep the same filename: `avatar.jpg`

#### Option 2: Using the Alternative Photo
If `16797511_10211917259905005_5197078804592533351_o.jpg` is better quality:
1. Copy it: `cp content/authors/admin/16797511_10211917259905005_5197078804592533351_o.jpg content/authors/admin/avatar.jpg`
2. Test locally with `hugo server --disableFastRender`
3. Verify it displays correctly

### Image Optimization Tools

**Free Online Tools:**
- **TinyPNG** (https://tinypng.com) - Compress JPEG/PNG
- **Squoosh** (https://squoosh.app) - Advanced compression with preview
- **ImageOptim** (Mac app) - Batch optimization

**Command Line (if available):**
```bash
# Using ImageMagick (if installed)
convert input.jpg -resize 1000x1000 -quality 85 -strip avatar.jpg

# Using sips (macOS built-in)
sips -Z 1000 input.jpg --out avatar.jpg
```

### Testing Checklist

After replacing the photo:
- [ ] Photo displays correctly on homepage
- [ ] Photo appears in author profile page
- [ ] Photo is square (not stretched)
- [ ] Photo loads quickly (<1 second)
- [ ] Photo looks good in both light and dark mode
- [ ] Photo is clear and professional
- [ ] File size is optimized (<500KB)

### Where the Photo Appears

The avatar photo is used in:
1. **Homepage** - About section (circular crop)
2. **Author Profile Page** - Full display
3. **Publication Pages** - Author attribution
4. **Social Media** - If sharing pages (Open Graph)

### Next Steps After Photo Replacement

Once the professional photo is in place:
1. Test locally: `hugo server --disableFastRender`
2. Verify on mobile devices (responsive check)
3. Check in both light and dark themes
4. Update PROGRESS.md to mark as complete
5. Move to Phase 1, Priority 2: Visual Abstracts

### Professional Photography Resources

**If you need to take a new photo:**

**Option 1: Professional Photographer**
- ETH Zurich may have photography services
- Local Zurich photographers (1-2 hour session)
- Cost: ~$200-500 for professional headshots

**Option 2: DIY Professional Setup**
- Use good natural lighting (near window)
- Clean, neutral background
- Use smartphone with portrait mode
- Square crop in post-processing
- Free apps: Snapseed, VSCO for editing

**Option 3: Conference/Event Photos**
- Use photos from recent presentations
- Ask event photographer for high-res version
- Crop to square format

### Example Professional Academic Photos

Good examples to reference:
- Clean, professional appearance
- Good lighting on face
- Neutral or relevant background
- Square format
- High resolution

---

**Priority:** HIGH - This is the single biggest visual upgrade for the website

**Estimated Time:** 30 minutes (if photo is ready) to 2 hours (if new photo needed)

**Impact:** Immediate visual improvement, more professional appearance

