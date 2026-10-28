"""Paper figures from the summary tables docs/paper/out/v2/*.csv → docs/paper/latex/fig_*.pdf|png.
ACL style: Times, 7.5–8 pt (no text smaller than 7.5 pt: MINFS), column width 3.03 in, page width 6.3 in;
Okabe–Ito palette, each label keeps the same colour across all figures. The figure size equals its LaTeX
width exactly (no bbox tight), margins are set in inches; check() verifies that no text falls outside
the figure or overlaps other text.

Figure 2 of the paper is fig_bias.pdf and Figure 3 is fig_auroc.pdf. fig_anatomy.pdf (classes of the
evidence document in false misses) is not a figure of the paper; the classes are given in the text of
Section 4.2. Qwen3-14B appears by itself when its rows are in the CSV files (label_v2.py uses a judge
file only if it has all 858 rows).

Environment variables:
  RETECO_FIG_SMALL=1       thin grey lines for the four small judges (small_judges.csv, small_judges.py)
                           in panel (b) of fig_bias; the legend then goes above both panels. On by
                           default in the narrow layout, off otherwise.
  RETECO_FIG_EXCLUDE=a,b   readers to leave out of fig_auroc (the paper leaves out OCC-RAG-1.7B).
  RETECO_FIG_COL, RETECO_FIG_PAGE
                           column and page width in inches (default 3.03 and 6.3, the ACL template).
                           A page width below 5.5 in gives a narrow layout: shorter labels, one legend
                           above the panels of fig_bias, fig_auroc as a 2 x 2 grid for four readers.
  RETECO_FIG_OUT           output folder (default docs/paper/latex).
  RETECO_V2                input folder (default docs/paper/out/v2; for testing).

Figures 2 and 3 of the paper:
    RETECO_FIG_SMALL=1 RETECO_FIG_EXCLUDE=OCC-RAG-1.7B python docs/paper/scripts/paper_figs.py"""
import csv, os
from decimal import Decimal, ROUND_HALF_UP
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.transforms import blended_transform_factory as blend

HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.environ.get("RETECO_V2", os.path.join(HERE, "..", "out", "v2"))        # override for testing
OUT = os.environ.get("RETECO_FIG_OUT", os.path.join(HERE, "..", "latex"))
# widths in inches: ACL column 3.03, page 6.3; a narrow page, e.g. RETECO_FIG_COL=RETECO_FIG_PAGE=4.8
COL = float(os.environ.get("RETECO_FIG_COL", 3.03))
PAGE = float(os.environ.get("RETECO_FIG_PAGE", 6.3))
NARROW = PAGE < 5.5                           # narrow layout: shorter labels, AUROC as a 2×2 grid
MINFS = 7.5                                   # minimum font size of any text in the figures, pt

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": MINFS, "ytick.labelsize": MINFS, "legend.fontsize": MINFS, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "xtick.major.pad": 2, "ytick.major.pad": 2, "axes.labelpad": 2.5, "axes.titlepad": 3,
    "lines.linewidth": 1.1, "lines.markersize": 3.8, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.handlelength": 2.2, "legend.borderaxespad": 0.3,
    "legend.labelspacing": 0.3, "pdf.fonttype": 42, "savefig.pad_inches": 0,
})
# label → (legend text, colour, marker, line style)
STYLE = {
    "qrels": ("qrels: no gold in top-5", "#D55E00", "s", "--"),
    "phi-4": ("content: phi-4", "#0072B2", "o", "-"),
    "Qwen3-14B": ("content: Qwen3-14B", "#009E73", "^", "-"),
    "audit": ("blind audit (Claude Opus 5.5)", "#000000", "D", ":"),
}
JUDGES = ["phi-4", "Qwen3-14B"]
STAGE = {"bm25_query": "BM25-turn", "bm25_hist": "BM25-hist", "bm25_fuse3": "Fusion", "hybrid": "Hybrid",
         "gold": "Gold (2b)"}
ORDER = ["bm25_query", "bm25_hist", "bm25_fuse3", "hybrid"]
KEY = {v: k for k, v in STAGE.items()} | {"gold": "gold", "Gold": "gold"}   # stage label → key
GUIDE, GRID, MUTED = "#cfcfcf", "#e4e4e4", "#555555"


def rows(name):
    return list(csv.DictReader(open(os.path.join(V2, name), encoding="utf-8")))


def f(x):
    return float(x) if x not in ("", None, "nan") else float("nan")


def num(v, signed=False):
    """.68 / +.08 / −.04 — as in the paper text (no leading zero, typographic minus); rounds
    half-up on the decimal representation from the CSV (0.185 → .19), as in tables_v2.py."""
    v = float(Decimal(repr(float(v))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
    s = f"{v:+.2f}" if signed else f"{v:.2f}"
    return s.replace("0.", ".", 1).replace("-", "−")


def errbar(ax, x, y, lo, hi, key, label=True, dx=0.0, ms=None):
    lab, c, m, ls = STYLE[key]
    xs = [v + dx for v in x]
    ax.errorbar(xs, y, yerr=[[a - b for a, b in zip(y, lo)], [b - a for a, b in zip(y, hi)]], color=c, marker=m,
                ls=ls, ms=ms or plt.rcParams["lines.markersize"], capsize=1.5, elinewidth=0.6, capthick=0.6,
                label=lab if label else None, mfc=c if key != "audit" else "white", mec=c, mew=0.9, zorder=3)


def handle(key, label=None):
    lab, c, m, ls = STYLE[key]
    return Line2D([], [], color=c, marker=m, ls=ls, ms=plt.rcParams["lines.markersize"], mec=c, mew=0.9,
                  mfc=c if key != "audit" else "white", label=label or lab)


def axes_in(fig, W, H, left, bottom, width, height):
    """Axes with margins in inches, so the figure size does not depend on bbox tight."""
    return fig.add_axes([left / W, bottom / H, width / W, height / H])


def check(fig, name):
    """Check that text (labels, legends, numbers) stays inside the figure and does not overlap. Prints violations."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    fb = fig.bbox
    texts = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]
    boxes, bad = [], 0
    for t in dict.fromkeys(texts):
        bb = t.get_window_extent(r)
        if bb.width == 0:
            continue
        if bb.x0 < fb.x0 - 0.5 or bb.y0 < fb.y0 - 0.5 or bb.x1 > fb.x1 + 0.5 or bb.y1 > fb.y1 + 0.5:
            print(f"  [{name}] outside the figure: {t.get_text()!r}")
            bad += 1
        boxes.append((t.get_text(), bb.shrunk(0.97, 0.9), t.get_rotation() % 360 != 0))
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if boxes[i][2] and boxes[j][2]:     # rotated axis labels: boxes are coarse, inspect visually
                continue
            if boxes[i][1].overlaps(boxes[j][1]):
                print(f"  [{name}] overlap: {boxes[i][0]!r} and {boxes[j][0]!r}")
                bad += 1
    return bad


def save(fig, name):
    fig.set_dpi(300)       # check text at the same resolution as the PNG (at 100 dpi hinting underestimates width)
    bad = check(fig, name)
    fig.savefig(os.path.join(OUT, f"{name}.pdf"))
    fig.savefig(os.path.join(OUT, f"{name}.png"), dpi=300)
    w, h = fig.get_size_inches()
    plt.close(fig)
    print(f"ok {name}: {w:.2f} x {h:.2f} in, text problems: {bad}")


def stage_guides(ax, xs_labels, top_labels=True):
    """Vertical guides at the stage nDCG values and stage labels above the axes; close neighbours
    are pushed left/right so the labels do not overlap."""
    xs_labels = sorted(xs_labels)
    tr = blend(ax.transData, ax.transAxes)
    for i, (x, lab) in enumerate(xs_labels):
        ax.axvline(x, color=GUIDE, lw=0.5, ls=(0, (1, 1.5)), zorder=0)
        if not top_labels:
            continue
        near_next = i + 1 < len(xs_labels) and xs_labels[i + 1][0] - x < 0.08
        near_prev = i > 0 and x - xs_labels[i - 1][0] < 0.08
        ha, dx = ("right", 1.5) if near_next else ("left", -1.5) if near_prev else ("center", 0)
        ax.annotate(lab, (x, 1.0), xycoords=tr, xytext=(dx, 2), textcoords="offset points", ha=ha, va="bottom",
                    fontsize=MINFS, color=MUTED, annotation_clip=False)


def fig_bias():
    """(a) share of contexts labelled insufficient under qrels and under content labels;
    (b) precision of the qrels label against every annotator."""
    lab = rows("label_stage.csv")        # judge, stage, ndcg, share_none(+lo,hi), share_no(+lo,hi), P(+lo,hi)
    aud = [r for r in rows("audit_stage.csv") if r["annotator"] == "claude"]   # stage, ndcg, P, P_lo, P_hi
    by = defaultdict(list)
    for r in lab:
        by[r["judge"]].append(r)
    judges = [j for j in JUDGES if j in by] + [j for j in by if j not in JUDGES]
    for j in judges:
        by[j].sort(key=lambda r: ORDER.index(r["stage"]))
    small_on = os.environ.get("RETECO_FIG_SMALL", "1" if NARROW else "0") == "1"
    TOP = NARROW or small_on             # one legend in two rows above the panels (in (b) it would cover the grey lines)
    LEG = 0.34 if TOP else 0.0
    W, H = PAGE, 2.2 + LEG
    L, R, B, T, GAP = 0.40, 0.13, 0.34, 0.36 + LEG, 0.50
    aw = (W - L - R - GAP) / 2
    ah = H - B - T
    a = axes_in(plt.figure(figsize=(W, H)), W, H, L, B, aw, ah)
    fig = a.figure
    b = axes_in(fig, W, H, L + aw + GAP, B, aw, ah)

    # (a): draw qrels once (it does not depend on the judge), then the share of "no" for each judge
    q = by[judges[0]]
    x = [f(r["ndcg"]) for r in q]
    errbar(a, x, [100 * f(r["share_none"]) for r in q], [100 * f(r["share_none_lo"]) for r in q],
           [100 * f(r["share_none_hi"]) for r in q], "qrels")
    for j in judges:
        rs = by[j]
        errbar(a, [f(r["ndcg"]) for r in rs], [100 * f(r["share_no"]) for r in rs],
               [100 * f(r["share_no_lo"]) for r in rs], [100 * f(r["share_no_hi"]) for r in rs], j)
    # (b): judges and the audit side by side, shifted along x so the error bars do not merge
    series = [(j, by[j]) for j in judges] + [("audit", sorted(aud, key=lambda r: ORDER.index(r["stage"])))]
    for i, (k, rs) in enumerate(series):
        dx = (i - (len(series) - 1) / 2) * 0.008
        errbar(b, [f(r["ndcg"]) for r in rs], [f(r["P"]) for r in rs], [f(r["P_lo"]) for r in rs],
               [f(r["P_hi"]) for r in rs], k, dx=dx)

    short = {"bm25_query": "Turn", "bm25_hist": "Hist"}
    # (b), narrow layout or RETECO_FIG_SMALL=1: thin grey lines for the small judges (small_judges.py), if their labels exist
    small = [r for r in (rows("small_judges.csv") if small_on and os.path.exists(os.path.join(V2, "small_judges.csv")) else [])
             if r["judge"] not in JUDGES + ["panel"] and r["stage"] in KEY]
    nd = {r["stage"]: f(r["ndcg"]) for r in q}
    for j in dict.fromkeys(r["judge"] for r in small):
        rs = sorted([r for r in small if r["judge"] == j], key=lambda r: ORDER.index(KEY[r["stage"]]))
        b.plot([nd[KEY[r["stage"]]] for r in rs], [f(r["P"]) for r in rs], color="#9a9a9a", lw=0.7, marker="o",
               ms=2.0, zorder=1)
    stages = [(f(r["ndcg"]), short.get(r["stage"], STAGE[r["stage"]]) if NARROW else STAGE[r["stage"]]) for r in q]
    for ax in (a, b):
        ax.set_xlabel("first-stage nDCG@10")
        ax.set_xlim(0.14, 0.63)
        ax.set_xticks([0.2, 0.3, 0.4, 0.5, 0.6], [".2", ".3", ".4", ".5", ".6"])
        ax.grid(axis="y", lw=0.4, color=GRID)
        ax.set_axisbelow(True)
        stage_guides(ax, stages)
    a.set_ylabel("contexts labelled insufficient (%)")
    a.set_ylim(0, 80)
    a.set_title("(a) Contexts labelled insufficient" if NARROW else
                "(a) Share of top-5 contexts labelled insufficient", loc="left", pad=11)
    b.set_ylabel("precision of “no gold”")
    b.set_ylim(0, 1)
    b.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], ["0", ".2", ".4", ".6", ".8", "1"])
    b.set_title("(b) Precision of “no gold”" if NARROW else
                "(b) Precision of “no gold” as a detector of insufficiency", loc="left", pad=11)
    if TOP:      # one legend for both panels, above them
        extra = [Line2D([], [], color="#9a9a9a", lw=0.7, marker="o", ms=2.0, label="small judges (4)")] if small else []
        alab = "audit (Claude Opus 5.5)" if NARROW else "blind audit (Claude Opus 5.5), 50 per stage"
        hs = [handle("qrels")] + [handle(j) for j in judges] + [handle("audit", alab)] + extra
        fig.legend(handles=hs, loc="upper left", bbox_to_anchor=(L / W - 0.012, 1.0), ncol=3, handlelength=2.2,
                   columnspacing=1.2, borderaxespad=0.15, handletextpad=0.5)
    else:
        a.legend(handles=[handle("qrels")] + [handle(j) for j in judges], loc="lower left")
        b.legend(handles=[handle(j) for j in judges] + [handle("audit", "blind audit (Claude Opus 5.5), 50 per stage")],
                 loc="lower left")
    save(fig, "fig_bias")


ANAT = [("near_dup", "near-duplicate of gold", "#0072B2", "white"),
        ("other_turn", "gold of another turn", "#56B4E9", "black"),
        ("same_source", "same source page", "#009E73", "white"),
        ("pool_neg", "same-thread negative", "#E69F00", "black"),
        ("other", "other document", "#BBBBBB", "black")]


ANAT_LEG = dict(fontsize=MINFS, handlelength=1.0, handleheight=0.9, columnspacing=0.9, labelspacing=0.3,
                handletextpad=0.4, borderaxespad=0)


def anat_handles():
    return [plt.Rectangle((0, 0), 1, 1, color=c, label=l) for _, l, c, _ in ANAT]


def fig_anatomy():
    """Classes of the evidence document in false misses, by stage and domain group."""
    rs = rows("anatomy.csv")             # row (row label), class, share, n
    labels = list(dict.fromkeys(r["row"] for r in rs))
    share = {(r["row"], r["class"]): f(r["share"]) for r in rs}
    n = {r["row"]: r["n"] for r in rs}
    main = [l for l in labels if ":" not in l]
    split = [l for l in labels if ":" in l]
    # row positions: stages, a gap, then stage × domain group
    ypos, y = {}, 0.0
    for l in main:
        ypos[l] = y
        y += 1
    y += 0.55
    for l in split:
        ypos[l] = y
        y += 1
    nrow = y
    names = {l: l.replace(": ", ", ") for l in labels}
    W, RH = COL, 0.165                   # row height, inches
    probe = plt.figure(figsize=(W, 1), dpi=300)
    rr = probe.canvas.get_renderer()
    wmax = max(probe.text(0, 0, names[l], fontsize=plt.rcParams["ytick.labelsize"]).get_window_extent(rr).width
               for l in labels) / probe.dpi
    # legend: 3 columns if it fits the column width at MINFS, otherwise 2; the bottom margin follows its height
    for ncol in (3, 2):
        lg = probe.legend(handles=anat_handles(), loc="upper left", ncol=ncol, **ANAT_LEG)
        probe.canvas.draw()
        lbb = lg.get_window_extent(rr)
        lg.remove()
        if lbb.width / probe.dpi <= W - 0.04:
            break
    plt.close(probe)
    L, R, T, B = wmax + 0.07, 0.24, 0.15, max(0.62, 0.31 + lbb.height / probe.dpi + 0.05)
    H = T + B + RH * nrow
    fig = plt.figure(figsize=(W, H))
    ax = axes_in(fig, W, H, L, B, W - L - R, RH * nrow)
    for row in labels:
        i, left = ypos[row], 0
        for key, _, c, tc in ANAT:
            v = 100 * share.get((row, key), 0)
            ax.barh(i, v, left=left, color=c, height=0.72, edgecolor="white", lw=0.4)
            if v >= 8:
                ax.text(left + v / 2, i, f"{v:.0f}", ha="center", va="center", fontsize=MINFS, color=tc)
            left += v
        ax.text(101.5, i, f"{n[row]}", va="center", ha="left", fontsize=MINFS, color=MUTED)
    ax.text(101.5, -0.75, "$n$", va="center", ha="left", fontsize=MINFS, color=MUTED)
    ticks = [ypos[l] for l in labels]
    ax.set_yticks(ticks, [names[l] for l in labels])
    ax.set_ylim(nrow - 0.5, -0.5)
    if split:   # thin separator between the overall rows and the per-domain-group breakdown
        ax.axhline(len(main) - 0.5 + 0.275, color="#999999", lw=0.4, xmin=-(L - 0.03) / (W - L - R), clip_on=False)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("share of false misses (%)")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=3)
    fig.legend(handles=anat_handles(), loc="upper left", bbox_to_anchor=(0.02 / W, (B - 0.31) / H), ncol=ncol,
               **ANAT_LEG)
    save(fig, "fig_anatomy")


READER = {"OCC-RAG-1.7B": "OCC-RAG-1.7B", "Qwen3-1.7B": "Qwen3-1.7B", "Qwen3-4B": "Qwen3-4B-Instruct-2507",
          "Qwen3-8B": "Qwen3-8B"}
XLAB = {"bm25_query": "BM25-turn", "bm25_hist": "BM25-hist", "bm25_fuse3": "Fusion", "hybrid": "Hybrid",
        "gold": "Gold (2b)"}


def fig_auroc():
    """AUROC of p(answerable) under qrels and under content labels: one panel per reader, stages + gold (2b)."""
    drop = set(filter(None, os.environ.get("RETECO_FIG_EXCLUDE", "").split(",")))   # the paper: OCC-RAG-1.7B
    rs = [r for r in rows("auroc.csv") if r.get("score", "p") == "p" and r["reader"] not in drop]
    for r in rs:
        r["key"] = KEY.get(r["stage"], r["stage"])
    readers = list(dict.fromkeys(r["reader"] for r in rs))
    stages = [s for s in ORDER + ["gold"] if any(r["key"] == s for r in rs)]
    labels = ["qrels"] + [j for j in JUDGES if any(r["label"] == j for r in rs)]
    ncol = len(readers) if not NARROW or len(readers) <= 3 else 2
    nrow = -(-len(readers) // ncol)
    hs = [handle("qrels", "qrels label (all vs. none)" if len(labels) == 2 else "qrels label")]
    hs += [handle(j, f"content label: {j}" + (" (yes vs. no)" if len(labels) == 2 else "")) for j in labels[1:]]
    LEGKW = dict(ncol=len(hs), handlelength=2.2, columnspacing=1.2, borderaxespad=0.15, handletextpad=0.5)
    L, R, B, GAP, VGAP = 0.40, 0.03, 0.46, 0.10, 0.24
    # note on the numbers: at the right end of the legend row if it fits there, otherwise as a second line
    note = "numbers: content − qrels (bold: 95% CI excludes 0)"
    probe = plt.figure(figsize=(PAGE, 1), dpi=300)
    rr = probe.canvas.get_renderer()
    lg, nt = probe.legend(handles=hs, **LEGKW), probe.text(0, 0, note, fontsize=MINFS)
    probe.canvas.draw()
    below = NARROW or (L - 0.012 * PAGE) + (lg.get_window_extent(rr).width + nt.get_window_extent(rr).width) / probe.dpi \
        + 0.15 > PAGE - R
    plt.close(probe)
    W, H = PAGE, (2.75 if NARROW and len(readers) > 3 else 2.25 if NARROW else 2.15 if below else 2.05)
    T = 0.44 if NARROW else 0.50 if below else 0.40
    aw = (W - L - R - GAP * (ncol - 1)) / ncol
    ah = (H - B - T - VGAP * (nrow - 1)) / nrow
    fig = plt.figure(figsize=(W, H))
    axes = [axes_in(fig, W, H, L + (i % ncol) * (aw + GAP), B + (nrow - 1 - i // ncol) * (ah + VGAP), aw, ah)
            for i in range(len(readers))]
    step = 0.24 if len(labels) == 2 else 0.2
    for ax, rd in zip(axes, readers):
        for i, key in enumerate(labels):
            dx = (i - (len(labels) - 1) / 2) * step
            pts = sorted([r for r in rs if r["reader"] == rd and r["label"] == key], key=lambda r: stages.index(r["key"]))
            if not pts:
                continue
            # retrieval stages are joined by a line; gold (2b) is a separate point without a line
            ret = [r for r in pts if r["key"] != "gold"]
            errbar(ax, [stages.index(r["key"]) for r in ret], [f(r["auroc"]) for r in ret],
                   [f(r["lo"]) for r in ret], [f(r["hi"]) for r in ret], key, label=False, dx=dx)
            for r in pts:
                if r["key"] == "gold":
                    lab, c, m, _ = STYLE[key]
                    ax.errorbar([stages.index("gold") + dx], [f(r["auroc"])],
                                yerr=[[f(r["auroc"]) - f(r["lo"])], [f(r["hi"]) - f(r["auroc"])]], color=c,
                                marker=m, ls="none", ms=plt.rcParams["lines.markersize"], capsize=1.5,
                                elinewidth=0.6, capthick=0.6, mec=c, mew=0.9, zorder=3)
        for r in rs:   # paired difference content − qrels above each stage (main judge only)
            if r["reader"] == rd and r["label"] == "phi-4" and r.get("delta") not in ("", None, "nan"):
                d, lo, hi = f(r["delta"]), f(r["d_lo"]), f(r["d_hi"])
                sig = r["d_sig"] == "True" if r.get("d_sig") in ("True", "False") else (lo > 0 or hi < 0)
                ax.text(stages.index(r["key"]), 0.985, num(d, signed=True), ha="center", va="top", fontsize=MINFS,
                        color=STYLE["phi-4"][1] if sig else "#8a8a8a", fontweight="bold" if sig else "normal")
        ax.axhline(0.5, color="#999999", lw=0.5, ls=":", zorder=0)
        if "gold" in stages:
            ax.axvline(stages.index("gold") - 0.5, color=GUIDE, lw=0.5, zorder=0)
        bottom_row = axes.index(ax) // ncol == nrow - 1
        ax.set_xticks(range(len(stages)), [XLAB[s] for s in stages] if bottom_row else [""] * len(stages),
                      rotation=35, ha="right", rotation_mode="anchor")
        ax.set_xlim(-0.55, len(stages) - 0.45)
        ax.set_title(READER.get(rd, rd), loc="left")
        ax.set_ylim(0.4, 1.0)
        ax.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        ax.grid(axis="y", lw=0.4, color=GRID)
        ax.set_axisbelow(True)
        if axes.index(ax) % ncol:
            ax.tick_params(axis="y", labelleft=False)
    for ax in axes[::ncol]:
        ax.set_yticklabels([".4", ".5", ".6", ".7", ".8", ".9", "1"])
        ax.set_ylabel("AUROC" if NARROW else "AUROC of $p$(answerable)")
    fig.legend(handles=hs, loc="upper left", bbox_to_anchor=(L / W - 0.012, 1.0), **LEGKW)
    if below:    # below the legend, as a second line
        fig.text(L / W - 0.004, 1 - 0.215 / H, "numbers: content (phi-4) − qrels; bold: 95% CI excludes 0",
                 ha="left", va="top", fontsize=MINFS, color=MUTED)
    else:
        fig.text(1 - R / W, 1 - 0.035 / H, note, ha="right", va="top", fontsize=MINFS, color=MUTED)
    save(fig, "fig_auroc")


if __name__ == "__main__":
    for fn in (fig_bias, fig_anatomy, fig_auroc):
        try:
            fn()
        except FileNotFoundError as e:
            print("no data for", fn.__name__, "—", e.filename)
