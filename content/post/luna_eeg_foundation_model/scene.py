"""
LUNA cross-attention compression — manim animation.

Tells the visual story:
  1. EEG input has C=24 channels (rows) and S=8 time patches per channel (columns).
  2. Each electrode has a 3D position; those coordinates are sinusoidally encoded
     and fused with the channel features (NeRF-style positional encoding).
  3. A small set of N=8 *learned* query tokens are introduced — model parameters,
     not produced from the input.
  4. Channels cross-attend to those queries, producing a fixed-size token sequence
     of shape N x d.
  5. Then we shrink the input from 24 → 4 channels — the token output is unchanged.
  6. After training, queries specialize: Q1 to frontal, Q4 to central, Q8 to
     occipital, etc.

Two scenes share this body:
  * CrossAttentionCompression           — paper-aligned palette (white background)
  * CrossAttentionCompression3B1B       — 3Blue1Brown-style (dark background, bright
                                          accent colors)

Uses only Text (no LaTeX needed).
"""
from manim import *
import numpy as np

# Both variants share frame dimensions (must be set at module load).
config.frame_width = 16
config.frame_height = 9


# ---------------------------------------------------------------------------
# Palettes
# ---------------------------------------------------------------------------

PALETTE_PAPER = {
    "background": WHITE,
    "channel": "#E08660",
    "query": "#7FA89E",
    "token": "#D67945",
    "module_bg": "#E7F1DA",
    "module_outline": "#7AA86A",
    "dim": "#9AA0A6",
    "text": "#1F1F1F",
    "attn": "#B68466",
    "highlight": "#1F6F47",
    "encode": "#A8B7E0",
    "head_outline": "#7C8390",
}

PALETTE_3B1B = {
    "background": "#0E1014",
    "channel": "#58C4DD",       # BLUE_C
    "query": "#F8E25C",         # YELLOW_C
    "token": "#5CD0B3",          # TEAL_C
    "module_bg": "#1A2B26",
    "module_outline": "#5CD0B3",
    "dim": "#7A8390",
    "text": "#F4F4F4",
    "attn": "#B5BAC4",
    "highlight": "#FFC857",
    "encode": "#C9A6E0",         # bright lavender — for sin/cos PE
    "head_outline": "#A8B0BA",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def patch_block(rows: int, cols: int, color: str,
                w: float = 0.22, h: float = 0.22, gap: float = 0.04) -> VGroup:
    grid = VGroup()
    for r in range(rows):
        for c in range(cols):
            sq = RoundedRectangle(
                corner_radius=0.04,
                width=w, height=h,
                stroke_color=color, stroke_width=1.2,
                fill_color=color, fill_opacity=0.55,
            )
            sq.move_to(np.array([c * (w + gap), -r * (h + gap), 0]))
            grid.add(sq)
    grid.move_to(ORIGIN)
    return grid


def column_blocks(n: int, color: str,
                  w: float = 0.55, h: float = 0.32, gap: float = 0.10) -> VGroup:
    col = VGroup()
    for i in range(n):
        sq = RoundedRectangle(
            corner_radius=0.06,
            width=w, height=h,
            stroke_color=color, stroke_width=1.5,
            fill_color=color, fill_opacity=0.65,
        )
        sq.move_to(np.array([0, -i * (h + gap), 0]))
        col.add(sq)
    col.move_to(ORIGIN)
    return col


def head_schematic(color: str, dot_color: str, radius: float = 0.45) -> VGroup:
    """
    Top-down head outline with nose at top and 8 electrode dots in a clean,
    symmetric layout (one outer ring of 6 + 2 mid dots).
    """
    head = Circle(radius=radius, color=color, stroke_width=1.6)
    nose = Triangle(color=color, stroke_width=1.6, fill_opacity=0)
    nose.scale(0.10).move_to(head.get_top() + UP * 0.05)

    # Eight dots: outer ring of 6 evenly spaced + 2 mid dots
    dots = VGroup()
    outer = 6
    for k in range(outer):
        theta = -np.pi / 2 + 2 * np.pi * k / outer
        x = radius * 0.78 * np.cos(theta)
        y = radius * 0.78 * np.sin(theta)
        dots.add(Dot(point=np.array([x, y, 0]), radius=0.045, color=dot_color))
    # Two mid dots on the central axis (left and right)
    dots.add(Dot(point=np.array([-radius * 0.30, 0, 0]), radius=0.045, color=dot_color))
    dots.add(Dot(point=np.array([radius * 0.30, 0, 0]), radius=0.045, color=dot_color))

    return VGroup(head, nose, dots)


# ---------------------------------------------------------------------------
# Shared scene body
# ---------------------------------------------------------------------------

class _BaseCrossAttention(Scene):
    """Subclasses set ``palette`` to one of PALETTE_PAPER / PALETTE_3B1B."""

    palette: dict = PALETTE_3B1B

    def label(self, text: str, scale: float = 0.40, color=None) -> Text:
        return Text(text, color=color or self.palette["text"],
                    weight=NORMAL, font="DejaVu Sans").scale(scale)

    def construct(self):
        P = self.palette
        self.camera.background_color = P["background"]

        # ---- Title --------------------------------------------------------
        title = self.label(
            "LUNA: cross-attention compresses variable channels into a fixed latent",
            0.50,
        )
        title.to_edge(UP, buff=0.30)
        self.play(FadeIn(title, shift=DOWN * 0.2), run_time=1.1)
        self.wait(0.4)

        # ---- Initial layout: C = 24 ---------------------------------------
        C_FULL = 24
        S_PATCHES = 8     # time patches per channel
        N = 8

        eeg = patch_block(rows=C_FULL, cols=S_PATCHES, color=P["channel"])
        eeg.scale(0.78)
        eeg.to_edge(LEFT, buff=1.9).shift(DOWN * 0.5)
        eeg_label = self.label("EEG input", 0.34)
        eeg_label.next_to(eeg, UP, buff=0.16)

        # Two braces: rows = C channels (left), columns = S time patches (bottom)
        c_brace = Brace(eeg, LEFT, color=P["dim"], buff=0.10)
        c_brace_label = self.label(f"C = {C_FULL}", 0.26).next_to(c_brace, LEFT, buff=0.08)
        s_brace = Brace(eeg, DOWN, color=P["dim"], buff=0.10)
        s_brace_label = self.label(f"{S_PATCHES} patches per channel — time →", 0.26).next_to(s_brace, DOWN, buff=0.08)

        # Queries / tokens columns + CUM box
        queries = column_blocks(N, P["query"])
        queries.move_to(ORIGIN + RIGHT * 1.2 + DOWN * 0.5)

        tokens = column_blocks(N, P["token"])
        tokens.to_edge(RIGHT, buff=1.4).shift(DOWN * 0.5)
        t_label = self.label("Tokens  ·  N × d (fixed)", 0.32)
        t_label.next_to(tokens, UP, buff=0.16)

        cum_bg = RoundedRectangle(
            corner_radius=0.18,
            width=queries.get_width() + 1.0,
            height=queries.get_height() + 0.45,
            stroke_color=P["module_outline"], stroke_width=1.6,
            fill_color=P["module_bg"], fill_opacity=1.0,
        )
        cum_bg.move_to(queries.get_center() + LEFT * 0.18)
        cum_bg.set_z_index(-1)
        q_label = self.label(f"Learned queries  ·  N = {N}", 0.32)
        q_label.next_to(cum_bg, UP, buff=0.18)
        cum_label = self.label("Channel Unification Module", 0.28,
                                color=P["module_outline"])
        cum_label.next_to(cum_bg, DOWN, buff=0.12)

        out_arrow = Arrow(
            start=queries.get_right() + RIGHT * 0.05,
            end=tokens.get_left() + LEFT * 0.05,
            color=P["dim"], buff=0.08, stroke_width=2.2,
            max_tip_length_to_length_ratio=0.12,
        )
        out_arrow_label = self.label("self-attention over N tokens", 0.28)
        out_arrow_label.next_to(out_arrow, UP, buff=0.06)

        # ---- Phase 1: Reveal EEG input + axis braces ---------------------
        self.play(
            FadeIn(eeg, shift=RIGHT * 0.2),
            FadeIn(eeg_label),
            run_time=1.1,
        )
        self.wait(0.35)
        self.play(
            LaggedStart(
                AnimationGroup(FadeIn(c_brace), FadeIn(c_brace_label)),
                AnimationGroup(FadeIn(s_brace), FadeIn(s_brace_label)),
                lag_ratio=0.5,
            ),
            run_time=1.3,
        )
        self.wait(1.0)

        # ---- Phase 2: Positional encoding of electrode coordinates -------
        head_g = head_schematic(P["head_outline"], P["channel"], radius=0.45)
        head_caption = self.label("electrode 3D coords", 0.22, color=P["dim"])

        xyz = self.label("(xᵢ, yᵢ, zᵢ)", 0.26, color=P["channel"])

        pe_box = RoundedRectangle(
            corner_radius=0.08,
            width=1.6, height=0.65,
            stroke_color=P["encode"], stroke_width=1.6,
            fill_color=P["encode"], fill_opacity=0.18,
        )
        pe_text = self.label("sin / cos PE", 0.24, color=P["encode"])
        pe_text.move_to(pe_box.get_center())
        pe_group = VGroup(pe_box, pe_text)

        # Lay out in a row above the EEG: [head] (xyz) [PE box] ↓ into EEG
        head_g.next_to(eeg, UP, buff=0.95).shift(LEFT * 0.6)
        head_caption.next_to(head_g, DOWN, buff=0.05)
        xyz.next_to(head_g, RIGHT, buff=0.30)
        pe_group.next_to(xyz, RIGHT, buff=0.30)

        pe_arrow_start = pe_group.get_bottom() + DOWN * 0.05
        pe_arrow_end = np.array([pe_group.get_center()[0], eeg.get_top()[1] + 0.05, 0])
        pe_arrow = Arrow(
            start=pe_arrow_start, end=pe_arrow_end,
            color=P["encode"], buff=0.0, stroke_width=2.0,
            max_tip_length_to_length_ratio=0.18,
        )

        # Hide eeg_label briefly so the PE row has space
        self.play(eeg_label.animate.set_opacity(0.0), run_time=0.15)

        self.play(FadeIn(head_g, shift=DOWN * 0.05), FadeIn(head_caption), run_time=0.7)
        self.wait(0.5)
        self.play(FadeIn(xyz, shift=RIGHT * 0.05), run_time=0.55)
        self.wait(0.4)

        # xyz flows into PE box
        self.play(
            FadeIn(pe_group),
            xyz.animate.move_to(pe_box.get_center()).set_opacity(0.0),
            run_time=0.7,
        )
        self.remove(xyz)
        self.play(pe_group.animate.scale(1.08), run_time=0.25)
        self.play(pe_group.animate.scale(1 / 1.08), run_time=0.25)
        self.wait(0.3)

        self.play(GrowArrow(pe_arrow), run_time=0.6)
        self.wait(0.25)

        # Channel "fusion" shimmer — slower so it reads as a deliberate event
        for sq in eeg:
            sq.save_state()
        self.play(
            *[sq.animate.set_stroke(P["encode"], width=1.6).set_fill(P["channel"], opacity=0.85)
              for sq in eeg],
            run_time=0.7,
        )
        self.play(*[sq.animate.restore() for sq in eeg], run_time=0.7)

        pe_caption = self.label(
            "3D electrode coords  →  sinusoidal PE  →  fused into channel features",
            0.26, color=P["encode"],
        )
        pe_caption.next_to(s_brace_label, DOWN, buff=0.30)
        self.play(FadeIn(pe_caption, shift=UP * 0.05), run_time=0.55)
        self.wait(1.7)

        pe_cleanup = VGroup(head_g, head_caption, pe_group, pe_arrow, pe_caption)
        self.play(
            FadeOut(pe_cleanup),
            eeg_label.animate.set_opacity(1.0),
            run_time=0.55,
        )

        # ---- Phase 3: CUM box + queries reveal ---------------------------
        self.play(
            FadeIn(cum_bg),
            FadeIn(cum_label),
            FadeIn(q_label),
            run_time=0.45,
        )

        q_subscripts = VGroup()
        for i in range(N):
            tag = self.label(f"Q{i+1}", 0.20, color=P["query"]).set_opacity(0.85)
            tag.next_to(queries[i], LEFT, buff=0.10)
            q_subscripts.add(tag)

        self.play(
            LaggedStart(
                *[
                    AnimationGroup(
                        FadeIn(queries[i], shift=UP * 0.10),
                        FadeIn(q_subscripts[i], shift=RIGHT * 0.05),
                    )
                    for i in range(N)
                ],
                lag_ratio=0.10,
            ),
            run_time=1.4,
        )
        self.wait(0.2)

        # Training shimmer — quicker, still legible as a sweep
        for i in range(N):
            self.play(queries[i].animate.set_fill(P["query"], opacity=1.0).scale(1.06),
                      run_time=0.06)
            self.play(queries[i].animate.set_fill(P["query"], opacity=0.65).scale(1 / 1.06),
                      run_time=0.06)

        learned_note = self.label(
            "model parameters — pre-trained, shared across every EEG layout",
            0.26, color=P["module_outline"],
        )
        q_label.generate_target()
        q_label.target.shift(UP * 0.30)
        learned_note.next_to(q_label.target, UP, buff=0.10)
        self.play(MoveToTarget(q_label), FadeIn(learned_note, shift=DOWN * 0.05),
                  run_time=0.45)
        self.wait(0.95)
        self.play(FadeOut(learned_note),
                  q_label.animate.shift(DOWN * 0.30),
                  run_time=0.4)

        # ---- Phase 4: Tokens column + arrow -------------------------------
        self.play(
            FadeIn(tokens, shift=UP * 0.15),
            FadeIn(t_label),
            FadeIn(out_arrow),
            FadeIn(out_arrow_label),
            run_time=0.7,
        )
        self.wait(0.15)

        # ---- Phase 5: Cross-attention pulse (C = 24) ----------------------
        attn_lines = VGroup()
        right_col = [eeg[r * S_PATCHES + (S_PATCHES - 1)] for r in range(C_FULL)]
        for patch in right_col:
            for q_idx in range(N):
                line = Line(
                    patch.get_right(),
                    queries[q_idx].get_left(),
                    stroke_color=P["attn"],
                    stroke_width=0.85,
                    stroke_opacity=0.0,
                )
                attn_lines.add(line)

        # Place the cross-attn caption in the empty band between title and EEG/queries
        attn_label = self.label("cross-attention:  N queries  ←  C channels", 0.30)
        attn_label.move_to(np.array([
            (eeg.get_right()[0] + queries.get_left()[0]) / 2,
            title.get_bottom()[1] - 0.45, 0,
        ]))

        self.play(FadeIn(attn_label), run_time=0.35)
        self.play(*[ln.animate.set_stroke(opacity=0.65) for ln in attn_lines],
                  run_time=0.85)
        self.play(*[queries[i].animate.set_fill(P["query"], opacity=1.0).scale(1.06)
                    for i in range(N)], run_time=0.4)
        self.play(*[queries[i].animate.set_fill(P["query"], opacity=0.65).scale(1 / 1.06)
                    for i in range(N)], run_time=0.4)
        self.wait(0.25)

        # ---- Phase 6: Shrink to C = 4 ------------------------------------
        self.play(*[ln.animate.set_stroke(opacity=0.0) for ln in attn_lines],
                  FadeOut(attn_label), run_time=0.4)

        keep_rows = 4
        keep_idx = set()
        for r in range(keep_rows):
            for c in range(S_PATCHES):
                keep_idx.add(r * S_PATCHES + c)

        dropped_patches = [eeg[i] for i in range(len(eeg)) if i not in keep_idx]
        for sq in dropped_patches:
            sq.save_state()

        new_eeg_label = self.label("Same model, fewer channels", 0.32).move_to(eeg_label)
        new_brace_label = self.label(f"C = {keep_rows}", 0.26).move_to(c_brace_label)

        self.play(
            *[FadeOut(sq, shift=LEFT * 0.15) for sq in dropped_patches],
            Transform(eeg_label, new_eeg_label),
            Transform(c_brace_label, new_brace_label),
            run_time=0.85,
        )

        kept = VGroup(*[eeg[i] for i in sorted(keep_idx)])
        new_brace = Brace(kept, LEFT, color=P["dim"], buff=0.12)
        new_s_brace = Brace(kept, DOWN, color=P["dim"], buff=0.12)
        self.play(
            Transform(c_brace, new_brace),
            Transform(s_brace, new_s_brace),
            s_brace_label.animate.next_to(new_s_brace, DOWN, buff=0.08),
            run_time=0.45,
        )

        right_col_small = [eeg[r * S_PATCHES + (S_PATCHES - 1)] for r in range(keep_rows)]
        new_attn_lines = VGroup()
        for patch in right_col_small:
            for q_idx in range(N):
                ln = Line(patch.get_right(), queries[q_idx].get_left(),
                          stroke_color=P["attn"], stroke_width=1.0,
                          stroke_opacity=0.0)
                new_attn_lines.add(ln)

        self.play(*[ln.animate.set_stroke(opacity=0.75) for ln in new_attn_lines],
                  run_time=0.65)

        same_label = self.label("Tokens: still N × d — unchanged.",
                                 0.38, color=P["highlight"])
        same_label.next_to(tokens, DOWN, buff=0.40)
        if same_label.get_right()[0] > 7.6:
            same_label.shift(LEFT * (same_label.get_right()[0] - 7.6))

        self.play(
            *[tokens[i].animate.set_fill(P["token"], opacity=1.0).scale(1.10)
              for i in range(N)],
            FadeIn(same_label, shift=UP * 0.1),
            run_time=0.5,
        )
        self.play(
            *[tokens[i].animate.set_fill(P["token"], opacity=0.65).scale(1 / 1.10)
              for i in range(N)],
            run_time=0.5,
        )
        self.wait(1.0)

        # ---- Phase 7: Specialization payoff ------------------------------
        # Restore full 24 channels so the spatial focus reads clearly.
        full_eeg_label = self.label("EEG input", 0.34).move_to(eeg_label)
        full_brace_label = self.label(f"C = {C_FULL}", 0.26).move_to(c_brace_label)
        full_s_label = self.label(f"{S_PATCHES} patches per channel — time →", 0.26)
        big_brace = Brace(eeg, LEFT, color=P["dim"], buff=0.10)
        big_s_brace = Brace(eeg, DOWN, color=P["dim"], buff=0.10)
        full_s_label.next_to(big_s_brace, DOWN, buff=0.08)

        self.play(
            *[FadeOut(ln) for ln in new_attn_lines],
            FadeOut(same_label),
            FadeOut(out_arrow),
            FadeOut(out_arrow_label),
            *[sq.animate.restore() for sq in dropped_patches],
            Transform(eeg_label, full_eeg_label),
            Transform(c_brace_label, full_brace_label),
            Transform(c_brace, big_brace),
            Transform(s_brace, big_s_brace),
            Transform(s_brace_label, full_s_label),
            run_time=0.7,
        )

        spec_caption = self.label(
            "after pre-training, each query specialises to a brain region",
            0.32, color=P["highlight"],
        )
        spec_caption.move_to(np.array([
            (eeg.get_right()[0] + queries.get_left()[0]) / 2,
            title.get_bottom()[1] - 0.45, 0,
        ]))
        self.play(FadeIn(spec_caption, shift=DOWN * 0.05), run_time=0.45)

        faint_lines = VGroup()
        for r in range(C_FULL):
            patch = eeg[r * S_PATCHES + (S_PATCHES - 1)]
            for q_idx in range(N):
                ln = Line(patch.get_right(), queries[q_idx].get_left(),
                          stroke_color=P["attn"], stroke_width=0.45,
                          stroke_opacity=0.10)
                faint_lines.add(ln)
        self.play(FadeIn(faint_lines), run_time=0.35)

        spec_groups = [
            (0, list(range(0, 8)),    "frontal"),
            (3, list(range(8, 16)),   "central"),
            (7, list(range(16, 24)),  "occipital"),
        ]

        for q_idx, channel_rows, region_name in spec_groups:
            bright = []
            for r in channel_rows:
                patch = eeg[r * S_PATCHES + (S_PATCHES - 1)]
                ln = Line(patch.get_right(), queries[q_idx].get_left(),
                          stroke_color=P["query"], stroke_width=1.7,
                          stroke_opacity=0.0)
                bright.append(ln)
            self.play(
                *[ln.animate.set_stroke(opacity=0.85) for ln in bright],
                queries[q_idx].animate.set_fill(P["query"], opacity=1.0).scale(1.10),
                run_time=0.45,
            )
            tag = self.label(f"→ {region_name}", 0.28, color=P["query"])
            tag.next_to(queries[q_idx], RIGHT, buff=0.20).set_z_index(2)
            self.play(FadeIn(tag, shift=LEFT * 0.05), run_time=0.30)
            self.wait(0.40)
            self.play(
                *[ln.animate.set_stroke(opacity=0.55) for ln in bright],
                queries[q_idx].animate.scale(1 / 1.10),
                run_time=0.25,
            )

        self.wait(1.5)


class CrossAttentionCompression(_BaseCrossAttention):
    """Paper-aligned palette — white background, paper Figure 1 colors."""
    palette = PALETTE_PAPER


class CrossAttentionCompression3B1B(_BaseCrossAttention):
    """3Blue1Brown-style — dark background, bright accent colors."""
    palette = PALETTE_3B1B
