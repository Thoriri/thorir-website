"""
The pre-training objective is a dial — manim animation for the LeJEPA / LuMamba post.

Tells the visual story behind the post:
  1. The pre-training objective is a single knob: masked reconstruction on the
     left, LeJEPA on the right, a mix in the middle.
  2. As the knob slides from reconstruction → LeJEPA, the latent space morphs
     from four tight, well-separated clusters into one diffuse *isotropic
     Gaussian* ball (the dashed target ring fades in at the LeJEPA extreme).
  3. Two competing virtues cross over on the right: in-distribution *structure*
     falls while cross-montage *generalisation* rises. The robust regime is the
     mix in the middle.

A single ValueTracker `lam` ∈ [0, 1] drives everything (knob, point cloud, and
the two trade-off dots) through one set of updaters, so the whole scene stays in
sync.

Two scenes share the body:
  * ObjectiveDial         — paper palette (white background)
  * ObjectiveDial3B1B     — 3Blue1Brown style (dark background, bright accents)

Render, e.g.:
    manim -qh scene.py ObjectiveDial3B1B
    manim -qh scene.py ObjectiveDial

Uses only Text / MathTex-free primitives (no LaTeX needed).
"""
from manim import *
import numpy as np

config.frame_width = 16
config.frame_height = 9

# ---------------------------------------------------------------------------
# Palettes (mirrors the conventions in the LUNA post's scene.py)
# ---------------------------------------------------------------------------
PALETTE_PAPER = {
    "background": WHITE,
    "text": "#1F1F1F",
    "dim": "#8A929B",
    "grid": "#E4E7EA",
    "panel": "#F6F7F4",
    "recon": "#E07A52",       # masked reconstruction
    "lejepa": "#2BA89A",      # LeJEPA
    "mix": "#C9A227",
    "clusters": ["#58779A", "#2BA89A", "#C9A227", "#E07A52"],
    "ring": "#8A929B",
    "knob": "#1F1F1F",
}

PALETTE_3B1B = {
    "background": "#0E1014",
    "text": "#F4F4F4",
    "dim": "#8A93A3",
    "grid": "#262B34",
    "panel": "#15181F",
    "recon": "#E08660",
    "lejepa": "#5CD0B3",
    "mix": "#F8E25C",
    "clusters": ["#58C4DD", "#5CD0B3", "#F8E25C", "#E08660"],
    "ring": "#8A93A3",
    "knob": "#F4F4F4",
}

N_PER = 45            # points per cluster
N_CLASS = 4
N = N_PER * N_CLASS


def smoothstep(t):
    return t * t * (3 - 2 * t)


def _make_configs(rng, span=2.05, spread=0.32, ball_r=1.55):
    """Return (cluster_pos, ball_pos) arrays of shape (N, 3) in manim units."""
    ang = np.linspace(0, TAU, N_CLASS, endpoint=False) + PI / 4
    centres = np.stack([span * np.cos(ang), span * np.sin(ang)], axis=1)

    cluster = np.zeros((N, 2))
    for c in range(N_CLASS):
        cluster[c * N_PER:(c + 1) * N_PER] = centres[c] + rng.normal(0, spread, (N_PER, 2))

    ball = rng.normal(0, 1.0, (N, 2))
    ball -= ball.mean(0)
    cov = np.cov(ball.T)
    w, V = np.linalg.eigh(cov)
    ball = ball @ V @ np.diag(1 / np.sqrt(w)) @ V.T * ball_r    # whiten → isotropic

    pad = lambda a: np.hstack([a, np.zeros((N, 1))])
    return pad(cluster), pad(ball)


class _BaseObjectiveDial(Scene):
    palette = PALETTE_PAPER

    def label(self, txt, s=0.42, col=None, weight=NORMAL):
        return Text(txt, font_size=s * 56, color=col or self.palette["text"], weight=weight)

    def construct(self):
        P = self.palette
        self.camera.background_color = P["background"]
        rng = np.random.default_rng(7)
        cluster_pos, ball_pos = _make_configs(rng)

        lam = ValueTracker(0.0)   # 0 = reconstruction, 1 = LeJEPA

        # ---- title ----
        title = self.label("The pre-training objective is a dial", s=0.62, weight=BOLD)
        title.to_edge(UP, buff=0.45)
        self.play(FadeIn(title, shift=DOWN * 0.2), run_time=1.0)

        # ---- dial ----
        dial = Line(LEFT * 2.6, RIGHT * 2.6, color=P["grid"], stroke_width=7)
        dial.move_to(UP * 2.55)
        ticks = VGroup()
        tick_specs = [(-2.6, "masked\nreconstruction", P["recon"]),
                      (0.0, "mix", P["mix"]),
                      (2.6, "LeJEPA", P["lejepa"])]
        for tx, lab, col in tick_specs:
            d = Dot([dial.get_center()[0] + tx, dial.get_center()[1], 0],
                    radius=0.07, color=col)
            t = Text(lab, font_size=22, color=col, weight=BOLD, line_spacing=0.8)
            t.next_to(d, DOWN, buff=0.18)
            ticks.add(VGroup(d, t))
        knob = Dot(dial.get_left(), radius=0.16, color=P["knob"])
        knob.set_stroke(P["background"], width=4, background=True)
        lam_txt = always_redraw(
            lambda: Text(f"λ = {lam.get_value():0.2f}", font_size=24, color=P["text"])
            .next_to(dial, RIGHT, buff=0.5))

        def knob_updater(m):
            x0, x1 = dial.get_left()[0], dial.get_right()[0]
            m.move_to([x0 + (x1 - x0) * lam.get_value(), dial.get_center()[1], 0])
        knob.add_updater(knob_updater)

        self.play(Create(dial), *[FadeIn(t) for t in ticks], run_time=0.9)
        self.play(FadeIn(knob), FadeIn(lam_txt), run_time=0.5)

        # ---- left latent panel ----
        panel = RoundedRectangle(width=5.4, height=5.0, corner_radius=0.12,
                                 stroke_color=P["grid"], stroke_width=2,
                                 fill_color=P["panel"], fill_opacity=1.0)
        panel.move_to([-4.0, -1.0, 0])
        panel_lab = Text("latent space (t-SNE)", font_size=24, color=P["dim"], weight=BOLD)
        panel_lab.next_to(panel, UP, buff=0.12)
        centre = panel.get_center()

        dots = VGroup()
        for i in range(N):
            c = P["clusters"][i // N_PER]
            dots.add(Dot(centre + cluster_pos[i] * 0.9, radius=0.045, color=c,
                         fill_opacity=0.92))

        ring = DashedVMobject(
            Circle(radius=1.55 * 0.9 * 2.0, color=P["ring"], stroke_width=1.6),
            num_dashes=44)
        ring.move_to(centre).set_opacity(0.0)

        def dots_updater(group):
            t = smoothstep(lam.get_value())
            for i, d in enumerate(group):
                d.move_to(centre + (cluster_pos[i] * (1 - t) + ball_pos[i] * t) * 0.9)
        dots.add_updater(dots_updater)

        def ring_updater(m):
            l = lam.get_value()
            m.set_opacity(np.clip((l - 0.6) / 0.4, 0, 1) * 0.6)
        ring.add_updater(ring_updater)

        caption = always_redraw(lambda: self._caption(lam.get_value(), panel))

        self.play(FadeIn(panel), FadeIn(panel_lab), run_time=0.6)
        self.play(LaggedStart(*[FadeIn(d, scale=0.5) for d in dots],
                              lag_ratio=0.004), run_time=1.1)
        self.add(ring, caption)

        # ---- right trade-off panel ----
        axes = Axes(x_range=[0, 1, 0.5], y_range=[0.5, 1.0, 0.5],
                    x_length=4.4, y_length=3.0,
                    axis_config={"include_tip": False, "stroke_color": P["grid"],
                                 "stroke_width": 2})
        axes.move_to([4.2, -1.1, 0])
        struct = axes.plot(lambda x: 0.93 - 0.30 * x ** 1.3, color=P["recon"],
                           stroke_width=5)
        gener = axes.plot(lambda x: 0.60 + 0.37 * (1 - np.exp(-3.2 * x)),
                          color=P["lejepa"], stroke_width=5)
        sweet = axes.get_area(axes.plot(lambda x: 1.0), x_range=[0.40, 0.60],
                              color=P["mix"], opacity=0.10)
        to_title = Text("the trade-off", font_size=24, color=P["text"], weight=BOLD)
        to_title.next_to(axes, UP, buff=0.12)
        s_lab = Text("in-distribution\nstructure", font_size=18, color=P["recon"],
                     weight=BOLD, line_spacing=0.8).next_to(axes.c2p(0, 0.93), UL, buff=0.05)
        g_lab = Text("cross-montage\ngeneralisation", font_size=18, color=P["lejepa"],
                     weight=BOLD, line_spacing=0.8).next_to(axes.c2p(1, 0.97), UR, buff=0.05)

        dotS = always_redraw(lambda: Dot(
            axes.c2p(lam.get_value(), 0.93 - 0.30 * lam.get_value() ** 1.3),
            radius=0.09, color=P["recon"]).set_stroke(P["background"], width=3, background=True))
        dotG = always_redraw(lambda: Dot(
            axes.c2p(lam.get_value(), 0.60 + 0.37 * (1 - np.exp(-3.2 * lam.get_value()))),
            radius=0.09, color=P["lejepa"]).set_stroke(P["background"], width=3, background=True))

        self.play(FadeIn(sweet), Create(axes), FadeIn(to_title), run_time=0.7)
        self.play(Create(struct), Create(gener),
                  FadeIn(s_lab), FadeIn(g_lab), run_time=1.0)
        self.add(dotS, dotG)
        self.wait(0.6)

        # ---- the sweep ----
        self.play(lam.animate.set_value(1.0), run_time=3.2, rate_func=smooth)
        self.wait(0.8)
        self.play(lam.animate.set_value(0.5), run_time=2.0, rate_func=smooth)
        self.wait(0.4)
        sweet_lab = Text("robust mix", font_size=20, color=P["mix"], weight=BOLD)
        sweet_lab.move_to(axes.c2p(0.5, 0.56))
        self.play(FadeIn(sweet_lab, shift=UP * 0.1), run_time=0.6)
        self.wait(1.2)
        self.play(lam.animate.set_value(0.0), run_time=2.0, rate_func=smooth)
        self.wait(1.0)

    def _caption(self, l, panel):
        P = self.palette
        if l < 0.2:
            txt, col = "masked reconstruction  →  tight, separated clusters", P["recon"]
        elif l > 0.8:
            txt, col = "LeJEPA  →  diffuse, isotropic Gaussian", P["lejepa"]
        else:
            txt, col = "the mix  →  structured but well-conditioned", P["mix"]
        t = Text(txt, font_size=20, color=col, weight=BOLD, slant=ITALIC)
        t.next_to(panel, DOWN, buff=0.18)
        return t


class ObjectiveDial(_BaseObjectiveDial):
    palette = PALETTE_PAPER


class ObjectiveDial3B1B(_BaseObjectiveDial):
    palette = PALETTE_3B1B
