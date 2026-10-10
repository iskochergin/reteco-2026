"""LaTeX tables for the paper: docs/paper/out/v2/*.csv → docs/paper/latex/tables/tab_*.tex.

tab_bias        main table (Table 1): nDCG, share labelled insufficient, P/R/F1, false misses, P under Qwen3-14B and the audit
tab_contrast    2b/2c contrast: 4 readers × 4 retrievers, qrels vs. content label (phi-4)
tab_detectors   AUROC of insufficiency detectors on BM25-fusion and the hybrid under the two labels
tab_agreement   agreement of the blind audit, phi-4 and Qwen3-14B, and the audit × phi-4 matrix (250 items)
tab_sameturn    within-turn 2b/2c contrast (contrast.csv, def=same_turn): appendix
tab_audit_auroc AUROC of OCC and Qwen3-4B on the 100 audited BM25-fusion contexts (out/refusal_flips.json, audit100)

All tables are single-column (\\columnwidth), booktabs, minus sign as $-$. They need the main.tex preamble macros
\\ci, \\qnone, \\cno, \\cyes, \\cpart, \\pend. label_v2.py uses a judge file only if it has all 858 rows;
a missing judge is shown as \\pend{--}.
Widths are checked with Times New Roman metrics (≈ Times from the times package): the natural width of each
table is printed against \\columnwidth = 7.7 cm. This is an estimate without running pdflatex, with a 3% margin.
    python tables_v2.py"""
import csv, json, math, os, re
from decimal import Decimal, ROUND_HALF_UP
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
V2 = os.environ.get("RETECO_V2", os.path.join(HERE, "..", "out", "v2"))
OUT = os.environ.get("RETECO_TAB_OUT", os.path.join(HERE, "..", "latex", "tables"))
SUFF = os.path.join(ROOT, "data", "drive", "reteco", "suff")
HYB = "hyb_dense_hanhainebula_reason-embed-qwen3-4b-0928_q2d"
ST = ["bm25_query", "bm25_hist", "bm25_fuse3", "hybrid"]
DISP = {"bm25_query": "BM25-turn", "bm25_hist": "BM25-hist", "bm25_fuse3": "Fusion", "hybrid": "Hybrid"}
COLW, TEXTW = 7.7 * 28.4528, 16 * 28.4528            # ACL: \columnwidth, \textwidth in pt
SIZE = {"small": 10.0, "footnotesize": 9.0, "scriptsize": 8.0}   # article 11pt
PEND = r"\pend{--}"


def rows(name):
    p = os.path.join(V2, name)
    return list(csv.DictReader(open(p, encoding="utf-8"))) if os.path.exists(p) else []


def fl(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def isnan(x):
    return x != x


# ───────────────────────── numbers ─────────────────────────
def hu(x, nd=2, scale=1):
    """Round half up on the decimal representation of the CSV number (0.185 → .19, as in the text),
    without the binary noise of round(); scale=100 for percentages."""
    d = Decimal(repr(float(x))) * scale
    return float(d.quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP))


def pc(x, nd=0):
    """Fraction → percentage string: 0.2818 → '28'."""
    return f"{hu(x, nd, 100):.{nd}f}"


def n2(x, signed=False, nd=2, negz=False):
    """.68 / $-$.04 / +.08; zero has no sign. negz (CI bounds): a negative number that
    rounds to zero is printed as $-$.00, so that the bound is visibly below zero."""
    if isnan(x):
        return "--"
    v = hu(x, nd)
    s = f"{abs(v):.{nd}f}"
    s = s[1:] if s.startswith("0") else s
    if v == 0:
        return "$-$" + s if negz and math.copysign(1.0, x) < 0 else s
    if v < 0:
        return "$-$" + s
    return ("+" if signed else "") + s


def ci(lo, hi, nd=2, pct=False):
    if isnan(lo) or isnan(hi):
        return ""
    if pct:
        return rf"\ci{{{pc(lo)}}}{{{pc(hi)}}}"
    return rf"\ci{{{n2(lo, nd=nd, negz=True)}}}{{{n2(hi, nd=nd, negz=True)}}}"


def cirow(lo, hi, nd=2, pct=False):
    """CI on a separate row below the value: [lo, hi] in \\scriptsize."""
    if isnan(lo) or isnan(hi):
        return ""
    a, b = (pc(lo), pc(hi)) if pct else (n2(lo, nd=nd, negz=True), n2(hi, nd=nd, negz=True))
    return rf"{{\scriptsize[{a},\,{b}]}}"


# ───────────────────────── table assembly and width estimate ─────────────────────────
class Tab:
    """Rows: a list of cells (str or ('mc', k, align, text)), or ('raw', latex)."""

    def __init__(self, cols, size="small", sep=4.0, env="table", width="columnwidth", pos="t"):
        self.cols, self.size, self.sep, self.env, self.width, self.pos, self.body = cols, size, sep, env, width, pos, []

    def row(self, *cells):
        self.body.append(list(cells))

    def raw(self, s):
        self.body.append(("raw", s))

    def latex(self, caption, label):
        out = [rf"\begin{{{self.env}}}[{self.pos}]", r"\centering", "\\" + self.size, rf"\setlength{{\tabcolsep}}{{{self.sep:g}pt}}"]
        out.append(rf"\begin{{tabularx}}{{\{self.width}}}{{{self.cols}}}")
        for r in self.body:
            if isinstance(r, tuple):
                out.append(r[1])
                continue
            cells = [rf"\multicolumn{{{c[1]}}}{{{c[2]}}}{{{c[3]}}}" if isinstance(c, tuple) else c for c in r]
            out.append(" & ".join(cells) + r" \\")
        out += [r"\end{tabularx}", rf"\caption{{{caption}}}", rf"\label{{{label}}}", rf"\end{{{self.env}}}"]
        return "\n".join(out) + "\n"


_FONTS = {}


def _font(bold, ital):
    from matplotlib.font_manager import FontProperties
    key = (bold, ital)
    if key not in _FONTS:
        _FONTS[key] = FontProperties(family="Times New Roman", weight="bold" if bold else "normal",
                                     style="italic" if ital else "normal")
    return _FONTS[key]


def text_width(s, size, bold=False, ital=False):
    from matplotlib.textpath import TextToPath
    if not s:
        return 0.0
    w, _, _ = TextToPath().get_text_width_height_descent(s, _font(bold, ital), ismath=False)
    return w * size / _font(bold, ital).get_size_in_points()   # w is measured at the font size of prop


def latex_width(s, size):
    """Rough width of a LaTeX cell in pt: parses only the macros that occur in these tables."""
    segs = []   # (text, size, bold, italic)

    def walk(t, sz, b, i):
        pos = 0
        while pos < len(t):
            m = re.match(r"\\ci\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", t[pos:])
            if m:
                segs.append(("\u2009", sz, b, i))
                walk("[" + m.group(1) + ", " + m.group(2) + "]", SIZE["scriptsize"], b, i)
                pos += m.end()
                continue
            m = re.match(r"\{\\(scriptsize|footnotesize|small)\s*", t[pos:])
            if m:
                depth, j = 1, pos + m.end()
                while depth:
                    depth += {"{": 1, "}": -1}.get(t[j], 0)
                    j += 1
                walk(t[pos + m.end():j - 1], SIZE[m.group(1)], b, i)
                pos = j
                continue
            m = re.match(r"\\(textbf|textit|textsc|pend|textnormal|mathrm|emph)\{", t[pos:])
            if m:
                depth, j = 1, pos + m.end()
                while depth:
                    depth += {"{": 1, "}": -1}.get(t[j], 0)
                    j += 1
                inner = t[pos + m.end():j - 1]
                if m.group(1) == "textsc":
                    walk(inner.upper(), sz * 0.8, b, i)
                else:
                    walk(inner, sz, b or m.group(1) == "textbf", i or m.group(1) in ("textit", "emph"))
                pos = j
                continue
            for pat, rep, scale, it in ((r"\\qnone(\{\})?", "NONE", 0.8, False), (r"\\cno(\{\})?", "no", 1, True),
                                        (r"\\cyes(\{\})?", "yes", 1, True), (r"\\cpart(\{\})?", "partial", 1, True),
                                        (r"\$-\$", "\u2212", 1, False), (r"\$\\times\$", "\u00d7", 1, False),
                                        (r"\$\^\\dagger\$", "\u2020", 0.7, False),
                                        (r"\$\^\{?\\(uparrow|downarrow)\}?\$", "\u2191", 0.7, False),
                                        (r"\$\\kappa\$", "\u03ba", 1, True), (r"\$\\tau\$", "\u03c4", 1, True),
                                        (r"\$n_I\$", "nI", 1, True), (r"\$n\$", "n", 1, True), (r"\$z\$", "z", 1, True),
                                        (r"\$p\$", "p", 1, True), (r"\\quad\s*", "\u2003", 1, False),
                                        (r"\\,", "\u2009", 1, False), (r"\\%", "%", 1, False), (r"~", " ", 1, False),
                                        (r"\\ ", " ", 1, False), (r"\\textsuperscript\{([^}]*)\}", None, 0.7, False),
                                        (r"--", "\u2013", 1, False)):
                m = re.match(pat, t[pos:])
                if m:
                    segs.append((rep if rep is not None else m.group(1), sz * scale, b, i or it))
                    pos += m.end()
                    break
            else:
                ch = t[pos]
                if ch in "{}":
                    pos += 1
                    continue
                if ch == "\\":   # other macros are not measured, but print a warning
                    m = re.match(r"\\[A-Za-z]+", t[pos:])
                    print("   ? macro without a width estimate:", m.group(0) if m else t[pos:pos + 8])
                    pos += m.end() if m else 1
                    continue
                segs.append((ch, sz, b, i))
                pos += 1

    walk(s, size, False, False)
    # merge consecutive segments with the same style so that kerning is taken into account
    out, cur = 0.0, None
    for txt, sz, b, i in segs + [("", 0, None, None)]:
        if cur and (sz, b, i) == cur[1:]:
            cur[0] += txt
            continue
        if cur:
            out += text_width(cur[0], cur[1], cur[2], cur[3])
        cur = [txt, sz, b, i]
    return out


def check_width(tab, name):
    """Natural column widths; X is the first column and gets the remaining width."""
    spec = re.findall(r"[lcrX]", tab.cols.replace("@{}", ""))
    ncol, sz = len(spec), SIZE[tab.size]
    nat = [0.0] * ncol
    spans = []
    for r in tab.body:
        if isinstance(r, tuple):
            continue
        j = 0
        for c in r:
            if isinstance(c, tuple):
                spans.append((j, c[1], latex_width(c[3], sz), c[2]))
                j += c[1]
            else:
                nat[j] = max(nat[j], latex_width(c, sz))
                j += 1
    target = COLW if tab.width == "columnwidth" else TEXTW
    seps = 2 * tab.sep * ncol - (2 * tab.sep if tab.cols.startswith("@{}") and tab.cols.endswith("@{}") else 0)
    fixed = sum(w for w, s in zip(nat, spec) if s != "X")
    xi = [k for k, s in enumerate(spec) if s == "X"]
    xavail = target - fixed - seps
    ok = True
    total = fixed + seps + (nat[xi[0]] if xi else 0)
    if xi and nat[xi[0]] > 0.97 * xavail:
        ok = False
    if not xi and total > 0.97 * target:
        ok = False
    for j, k, w, al in spans:   # \multicolumn cells: check whether they are wider than their columns combined
        inner = sum(nat[j:j + k]) + 2 * tab.sep * (k - 1) + (0 if "@{}" in al else 2 * tab.sep)
        if xi and xi[0] in range(j, j + k):
            inner = sum(nat[j:j + k]) - nat[xi[0]] + xavail + 2 * tab.sep * (k - 1)
        if w > 0.99 * inner + 0.5:
            print(f"   [{name}] \\multicolumn wider than its columns: {w:.1f} > {inner:.1f} pt")
            ok = False
    print(f"{'ok ' if ok else 'TOO WIDE'} {name}: columns {' '.join(f'{w:.0f}' for w in nat)} pt; "
          f"fixed+padding {fixed + seps:.0f} pt; X needs {nat[xi[0]] if xi else 0:.0f} of {xavail:.0f} pt "
          f"(width {target:.0f} pt)")
    return ok


def write(tab, name, caption, label):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name + ".tex"), "w", encoding="utf-8") as fh:
        fh.write(f"% generated by docs/paper/scripts/tables_v2.py from docs/paper/out/v2 — do not edit by hand\n")
        fh.write(tab.latex(caption, label))
    check_width(tab, name)


# ───────────────────────── tab_bias ─────────────────────────
def tab_bias():
    lab = rows("label_stage.csv")
    aud = {r["stage"]: r for r in rows("audit_stage.csv") if r["annotator"] == "claude"}
    by = defaultdict(dict)
    for r in lab:
        by[r["judge"]][r["stage"]] = r
    p4 = by["phi-4"]
    q14 = by.get("Qwen3-14B", {})
    t = Tab("@{}Xcccc@{}", size="small", sep=3.5)
    t.raw(r"\toprule")
    t.row("", "Turn", "Hist", "Fusion", "Hybrid")
    t.raw(r"\midrule")
    t.row("nDCG@10", *[n2(fl(p4[s]["ndcg"]), nd=3) for s in ST])
    t.raw(r"\addlinespace[2pt]")
    t.row(("mc", 5, "@{}l", r"\textit{Contexts labelled insufficient (\%)}"))
    t.row("qrels: no gold", *[pc(fl(p4[s]['share_none']), 1) for s in ST])
    t.row(r"content: \cno", *[pc(fl(p4[s]['share_no']), 1) for s in ST])
    t.row(r"ratio qrels\,/\,\cno", *[f"{hu(fl(p4[s]['ratio']), 1):.1f}$\\times$" for s in ST])
    if q14:
        t.row(r"\quad Qwen3-14B \cno", *[pc(fl(q14[s]['share_no']), 1) if s in q14 else PEND for s in ST])
    t.raw(r"\addlinespace[2pt]")
    t.row(("mc", 5, "@{}l", r"\textit{qrels \qnone{} against content \cno}"))
    t.row(r"false misses (\%)", *[pc(fl(p4[s]['fm'])) for s in ST])
    t.row("", *[cirow(fl(p4[s]["fm_lo"]), fl(p4[s]["fm_hi"]), pct=True) for s in ST])
    t.row("precision", *[n2(fl(p4[s]["P"])) for s in ST])
    t.row("", *[cirow(fl(p4[s]["P_lo"]), fl(p4[s]["P_hi"])) for s in ST])
    t.row("recall", *[n2(fl(p4[s]["R"])) for s in ST])
    t.row(r"false alarms (\%)", *[pc(fl(p4[s]["fpr"])) for s in ST])
    t.row("F1", *[n2(fl(p4[s]["F1"])) for s in ST])
    t.raw(r"\addlinespace[2pt]")
    t.row(("mc", 5, "@{}l", r"\textit{Precision under other labellers}"))
    if q14:
        t.row("Qwen3-14B", *[n2(fl(q14[s]["P"])) if s in q14 else PEND for s in ST])
        t.row("", *[cirow(fl(q14[s]["P_lo"]), fl(q14[s]["P_hi"])) if s in q14 else "" for s in ST])
    else:
        t.row("Qwen3-14B", *[PEND] * 4)
    t.row("blind audit", *[n2(fl(aud[s]["P"])) for s in ST])
    t.row("", *[cirow(fl(aud[s]["P_lo"]), fl(aud[s]["P_hi"])) for s in ST])
    t.raw(r"\bottomrule")
    cap = (r"The qrels label against the content label (phi-4 unless noted) for the four first-stage retrievers "
           r"(BM25-turn, BM25-hist, BM25-fusion, hybrid). Ratio: qrels no-gold share divided by the phi-4 \cno{} "
           r"share. False misses: share of \qnone{} contexts judged \cyes. Shares and ratio count only \cno{} as "
           r"insufficient (\cpart{} as sufficient; see Appendix~\ref{app:partial}). False alarms: share of \cyes{} "
           r"contexts that are \qnone. Precision and recall of \qnone{} as a detector of content \cno, with \cpart{} excluded. "
           r"Blind audit: 50 \qnone{} contexts per retriever. Brackets: 95\% bootstrap CIs "
           r"(over turns; for the audit, over its items).")
    write(t, "tab_bias", cap, "tab:bias")


# ───────────────────────── tab_contrast ─────────────────────────
READER = [("OCC-RAG-1.7B", "OCC-RAG-1.7B"), ("Qwen3-1.7B", "Qwen3-1.7B"),
          ("Qwen3-4B", "Qwen3-4B-Instruct-2507"), ("Qwen3-8B", "Qwen3-8B")]


def tab_contrast():
    rs = [r for r in rows("contrast.csv") if r["refusal"] == "verdict" and r["def"] == "v1"]
    get = {(r["reader"], r["stage"], r["label"]): r for r in rs}
    t = Tab("@{}Xccc@{}", size="small", sep=3.5)
    t.raw(r"\toprule")
    t.row("Retriever", "qrels label", "content label", r"{\scriptsize $n_I$ q/c}")
    for k, (rd, name) in enumerate(READER):
        t.raw(r"\midrule" if k == 0 else r"\addlinespace[3pt]")
        t.row(("mc", 4, "@{}l", rf"\textit{{{name}}}"))
        for s in ST:
            q, c = get.get((rd, DISP[s], "qrels")), get.get((rd, DISP[s], "phi-4"))
            if not q or not c:
                t.row(DISP[s], PEND, PEND, "")
                continue
            sq, sc = q["sig"] == "True", c["sig"] == "True"
            vq, vc = n2(fl(q["contrast"]), signed=True), n2(fl(c["contrast"]), signed=True)
            if sc and not sq:
                vc = rf"\textbf{{{vc}}}"
            if sq and not sc:
                vq += r"$^\dagger$"
            t.row(DISP[s], vq + ci(fl(q["lo"]), fl(q["hi"])), vc + ci(fl(c["lo"]), fl(c["hi"])),
                  rf"{{\scriptsize {q['n_I']}/{c['n_I']}}}")
    t.raw(r"\bottomrule")
    cap = (r"The 2b/2c contrast: refusal rate (\texttt{UNANSWERABLE} status) on insufficient retrieved contexts "
           r"minus refusal rate on sufficient gold (2b) contexts, under the qrels label (\qnone; every gold "
           r"context counts as sufficient) and the content label (phi-4 \cno{} vs.\ \cyes). "
           r"$n_I$: insufficient retrieved contexts under each label. Bold: significant under the content label "
           r"only; $^\dagger$: significant under the qrels label only. Responses without a status line are "
           r"excluded. Qwen3-1.7B and Qwen3-8B refuse rarely (1--9\% of contexts, 21\% for Qwen3-8B on BM25-turn).")
    write(t, "tab_contrast", cap, "tab:contrast")


# ───────────────────────── tab_detectors ─────────────────────────
DET = [("Retrieval score (no reader)", [("ret_top1_raw", "top-1 score"), ("ret_top1_dz", r"\quad domain-$z$"),
                                       ("ret_margin12_raw", "top-1 margin"), ("ret_margin12_dz", r"\quad domain-$z$"),
                                       ("ret_mean5_raw", "mean of top-5"), ("ret_mean5_dz", r"\quad domain-$z$")]),
       (r"Reader $p$(\textnormal{\textsc{answerable}})", [("OCC_p", "OCC-RAG-1.7B"), ("Qwen3-1.7B_p", "Qwen3-1.7B"),
                                             ("Qwen3-4B_p", "Qwen3-4B"), ("Qwen3-8B_p", "Qwen3-8B")]),
       ("OCC hidden-state probe, trained on", [("xfer_probe_synth_L20", "SQuAD\\,2.0/MuSiQue"),
                                               ("xfer_probe_retecotrain_L12", "RECOR train (qrels)")])]
DSTAGES = ["Fusion", "Hybrid"]


def tab_detectors():
    rs = rows("detectors.csv")
    bj = json.load(open(os.path.join(V2, "baselines.json")))
    # responses without a status (p_answerable None) are excluded from reader AUROCs: count them on the table's stages
    emp = {rd: max(bj["reader_empty_status"][f"{rd}|{s}"] for s in DSTAGES)
           for rd in ("OCC", "Qwen3-1.7B", "Qwen3-4B", "Qwen3-8B")}
    worst = max(emp, key=emp.get)
    zero = [rd for rd, v in emp.items() if v == 0]
    emp_txt = (f"up to {emp[worst]} of {bj['n_turns']} turns for {worst}"
               + (f", none for {' and '.join(zero)}" if zero else ""))
    content = [l for l in dict.fromkeys(r["label"] for r in rs) if l.startswith("content_")]
    main_c = "content_phi-4" if "content_phi-4" in content else content[0]
    get = {(r["detector"], r["stage"], r["label"]): r for r in rs}
    best = {}
    for s in DSTAGES:
        for L in ("qrels", main_c):
            cand = [(fl(r["auroc"]), r["detector"]) for r in rs if r["stage"] == s and r["label"] == L
                    and r["source"] == "computed" and (r["detector"].endswith("_p") or r["detector"].startswith("ret_"))]
            best[s, L] = max(cand)[1] if cand else None
    t = Tab("@{}Xcccc@{}", size="small", sep=3)
    t.raw(r"\toprule")
    t.row("", ("mc", 2, "c", "Fusion"), ("mc", 2, "c", "Hybrid"))
    t.raw(r"\cmidrule(lr){2-3}\cmidrule(l){4-5}")
    t.row("Detector", "qrels", "content", "qrels", "content")
    for g, (head, dets) in enumerate(DET):
        t.raw(r"\midrule" if g == 0 else r"\addlinespace[2pt]")
        t.row(("mc", 5, "@{}l", rf"\textit{{{head}}}"))
        for det, name in dets:
            cells = []
            for s in DSTAGES:
                for L in ("qrels", main_c):
                    r = get.get((det, s, L))
                    if not r or isnan(fl(r["auroc"])):
                        cells.append("--")
                        continue
                    v = n2(fl(r["auroc"]))
                    if best.get((s, L)) == det:
                        v = rf"\textbf{{{v}}}"
                    if L != "qrels" and not isnan(fl(r["d_lo"])):
                        if fl(r["d_lo"]) > 0:
                            v += r"$^{\uparrow}$"
                        elif fl(r["d_hi"]) < 0:
                            v += r"$^{\downarrow}$"
                    cells.append(v)
            t.row(name, *cells)
    # Kendall τ between the rankings of the 11 detectors under the two labels (baselines.json)
    tau = {r["stage"]: r["kendall_tau_vs_qrels"] for r in bj["rankings"] if r["label"] == main_c}
    t.raw(r"\midrule")
    t.row(r"Kendall $\tau$",
          *[("mc", 2, "c", n2(tau[s][0]) + ci(tau[s][1], tau[s][2])) if s in tau else ("mc", 2, "c", "--")
            for s in DSTAGES])
    t.raw(r"\bottomrule")
    cap = (r"AUROC of insufficiency detectors on the two strongest retrievers under the qrels label "
           r"(\textsc{all} vs.\ \qnone) and the content label (phi-4 \cyes{} vs.\ \cno). Retrieval scores need no "
           r"reader; domain-$z$: z-scored within retriever and domain on dev (label-free). Bold: best detector in "
           r"the column, probes excluded. Readers' AUROCs omit responses without a status line (" + emp_txt + r"). "
           r"Arrows: paired content$-$qrels difference with 95\% CI above or below zero. Probes are "
           r"point estimates on OCC's hidden states (Fusion only; turns without a status excluded); the RECOR-train "
           r"probe was trained with the qrels label. "
           r"Kendall $\tau$ compares the ranking of 11 detectors (three domain-$z$ scores, four readers' $p$ and "
           r"verdicts) under the two labels.")
    write(t, "tab_detectors", cap, "tab:detectors")


# ───────────────────────── tab_agreement ─────────────────────────
def audit_items():
    """The 250 audit items as (stage, domain, topic_id, blind-audit label), plus labels from judges with 858-row files."""
    key = {r["n"]: r for r in csv.DictReader(open(os.path.join(ROOT, "data/drive/audit/sample_key.csv")))}
    items = []
    for r in csv.DictReader(open(os.path.join(ROOT, "docs/audit_llm.csv"))):
        k = key[r["n"]]
        items.append(("bm25_fuse3", k["domain"], k["topic_id"], r["final"]))
    smap = {"bm25_query": "bm25_query", "bm25_hist": "bm25_hist", HYB: "hybrid", "hybrid": "hybrid"}
    for r in csv.DictReader(open(os.path.join(ROOT, "docs/audit_llm2.csv"))):
        items.append((smap[r["stage"]], r["domain"], r["topic_id"], r["final"]))
    job = {"bm25_query": "2c-bm25_query", "bm25_hist": "2c-bm25_hist", "bm25_fuse3": "2c-bm25_fuse3",
           "hybrid": f"2c-{HYB}"}
    judges = defaultdict(dict)
    for jd in ("phi-4", "Qwen3-14B"):
        for s, jb in job.items():
            p = os.path.join(SUFF, f"{jd}_{jb}_dev.jsonl")
            if not os.path.exists(p):
                continue
            lines = open(p, encoding="utf-8").read().splitlines()
            if len(lines) != 858:
                continue
            for ln in lines:
                d = json.loads(ln)
                lab = d.get("label")
                judges[jd][(s, d["domain"], d["topic_id"])] = lab if lab in ("yes", "partial", "no") else "invalid"
    return items, judges


def tab_agreement():
    ag = rows("agreement.csv")
    get = {(r["pair"], r["subset"]): r for r in ag}
    items, judges = audit_items()
    L3 = ["yes", "partial", "no"]
    CMAC = {"yes": r"\cyes", "partial": r"\cpart", "no": r"\cno"}

    def stats(r, kci=False):
        if not r:
            return [PEND] * 4
        k = n2(fl(r["kappa3"])) + (ci(fl(r["kappa3_lo"]), fl(r["kappa3_hi"])) if kci else "")
        return [f"{int(r['n']):,}".replace(",", "{,}"), n2(fl(r["agree3"])), k, n2(fl(r["ac1_3"]))]

    def get_pair(a, b):
        return lambda sub: get.get((f"{a} vs {b}", sub)) or get.get((f"{b} vs {a}", sub))

    t = Tab("@{}Xcccc@{}", size="small", sep=3.5, pos="H")
    t.raw(r"\toprule")
    t.row("Items", "$n$", "agree", r"$\kappa$", "AC1")
    t.raw(r"\midrule")
    cp = get_pair("claude", "phi-4")
    t.row(("mc", 5, "@{}l", r"\textit{Blind audit (Claude) vs.\ phi-4}"))
    t.row("all audited", *stats(cp("all_audited"), kci=True))
    t.row(r"\quad \qnone{} only", *stats(cp("none_audited")))
    for s, name in (("bm25_query", "BM25-turn"), ("bm25_hist", "BM25-hist"), ("bm25_fuse3", "Fusion"),
                    ("hybrid", "Hybrid")):
        t.row(rf"\quad {name}" + (r"\textsuperscript{a}" if s == "bm25_fuse3" else ""),
              *stats(cp(f"{s}_audited")))
    r = cp("all_audited")
    t.row(r"\cyes{} vs.\ \cno{} only", r["n_strict"], n2(fl(r["agree_yes_no"])), n2(fl(r["kappa_yes_no"])),
          n2(fl(r["ac1_yes_no"])))
    t.raw(r"\addlinespace[2pt]")
    t.row(("mc", 5, "@{}l", r"\textit{Claude run A vs.\ run B (before adjudication)}"))
    t.row("all audited", *stats(get.get(("claude_A vs claude_B", "all_audited"))))
    t.raw(r"\addlinespace[2pt]")
    qp = get_pair("phi-4", "Qwen3-14B")
    t.row(("mc", 5, "@{}l", r"\textit{Qwen3-14B vs.\ phi-4 (all contexts)}"))
    if qp("all_jobs_pooled"):
        t.row(r"gold + 4 retrievers", *stats(qp("all_jobs_pooled"), kci=True))
        for s, name in (("gold", "gold (2b)"), ("bm25_query", "BM25-turn"), ("bm25_hist", "BM25-hist"),
                        ("bm25_fuse3", "Fusion"), ("hybrid", "Hybrid")):
            if qp(f"{s}_all858"):
                t.row(rf"\quad {name}", *stats(qp(f"{s}_all858")))
        cq = get_pair("claude", "Qwen3-14B")
        if cq("all_audited"):
            t.raw(r"\addlinespace[2pt]")
            t.row(("mc", 5, "@{}l", r"\textit{Blind audit (Claude) vs.\ Qwen3-14B}"))
            t.row("all audited", *stats(cq("all_audited"), kci=True))
    else:
        t.row(r"gold + 4 retrievers", PEND, PEND, PEND, PEND)
    # blind audit × phi-4 matrix over all 250 items
    cm = Counter()
    for s, d, tid, c in items:
        j = judges["phi-4"].get((s, d, tid))
        if j in L3 and c in L3:
            cm[c, j] += 1
    t.raw(r"\midrule")
    t.row(("mc", 5, "@{}l", r"\textit{Claude (rows) $\times$ phi-4 (columns), all audited}"))
    t.row("", "", r"\cyes", r"\cpart", r"\cno")
    for c in L3:
        t.row(rf"\quad Claude {CMAC[c]}", "", *[str(cm[c, j]) for j in L3])
    t.raw(r"\bottomrule")
    tot = sum(cm.values())
    diag = sum(cm[c, c] for c in L3)
    r0 = cp("all_audited")
    assert r0 and int(r0["n"]) == tot and abs(diag / tot - fl(r0["agree3"])) < 1e-6, (tot, diag, r0 and r0["agree3"])
    cap = (r"Agreement between labellers on the three-way sufficiency label: raw agreement, Cohen's $\kappa$ "
           r"(95\% bootstrap CI over items where shown) and Gwet's AC1. The audit covers 250 contexts: 50 random "
           r"\qnone{} contexts per retriever and, for Fusion\textsuperscript{a}, 50 \qnone{} plus 50 \textsc{all}. "
           r"\cyes{} vs.\ \cno: items where neither labeller says \cpart. Run A read each item holistically, run B "
           r"checked the reference answer claim by claim.")
    write(t, "tab_agreement", cap, "tab:agreement")


# ───────────────────────── tab_sameturn ─────────────────────────
def tab_sameturn():
    """Within-turn 2b/2c contrast: refusal on an insufficient retrieved context minus refusal on the
    sufficient gold context of the same turn (contrast.csv, def=same_turn, refusal=verdict)."""
    rs = [r for r in rows("contrast.csv") if r["refusal"] == "verdict" and r["def"] == "same_turn"]
    get = {(r["reader"], r["stage"], r["label"]): r for r in rs}
    t = Tab("@{}Xccc@{}", size="small", sep=3.5, pos="ht")
    t.raw(r"\toprule")
    t.row("Retriever", "qrels label", "content label", r"{\scriptsize $n_I$ q/c}")
    for k, (rd, name) in enumerate(READER):
        t.raw(r"\midrule" if k == 0 else r"\addlinespace[3pt]")
        t.row(("mc", 4, "@{}l", rf"\textit{{{name}}}"))
        for s in ST:
            q, c = get.get((rd, DISP[s], "qrels")), get.get((rd, DISP[s], "phi-4"))
            if not q or not c:
                t.row(DISP[s], PEND, PEND, "")
                continue
            sq, sc = q["sig"] == "True", c["sig"] == "True"
            vq, vc = n2(fl(q["contrast"]), signed=True), n2(fl(c["contrast"]), signed=True)
            if sc and not sq:
                vc = rf"\textbf{{{vc}}}"
            if sq and not sc:
                vq += r"$^\dagger$"
            t.row(DISP[s], vq + ci(fl(q["lo"]), fl(q["hi"])), vc + ci(fl(c["lo"]), fl(c["hi"])),
                  rf"{{\scriptsize {q['n_I']}/{c['n_I']}}}")
    t.raw(r"\bottomrule")
    cap = (r"Within-turn variant of Table~\ref{tab:contrast}: refusal rate on insufficient retrieved contexts minus "
           r"refusal rate on the sufficient gold (2b) context of the same turns, under the qrels label (\qnone) and "
           r"the content label (phi-4 \cno{} retrieved, \cyes{} gold). $n_I$: turns in each comparison. Bold: "
           r"significant under the content label only; $^\dagger$: significant under the qrels label only. "
           r"Responses without a status line are excluded.")
    write(t, "tab_sameturn", cap, "tab:sameturn")


# ───────────────────────── tab_audit_auroc ─────────────────────────
def tab_audit_auroc():
    """AUROC of p(answerable) for OCC and Qwen3-4B on the 100 audited Fusion contexts under three labels.
    Source: out/refusal_flips.json, audit100 (refusal_flips.py; "Qwen" there is Qwen3-4B-Instruct-2507);
    the unrounded *_exact fields are used to avoid rounding twice."""
    au = json.load(open(os.path.join(V2, "..", "refusal_flips.json")))["audit100"]
    get = {(r["reader"], r["label"]): r for r in au}
    LBL = [("qrels", "qrels"), ("phi4", "phi-4"), ("claude", "blind audit")]
    t = Tab("@{}Xccc@{}", size="small", sep=3.5, pos="ht")
    t.raw(r"\toprule")
    t.row("Reader", *[h for _, h in LBL])
    t.raw(r"\midrule")
    for rd, name, signed in (("OCC", "OCC-RAG-1.7B", False), ("Qwen", "Qwen3-4B", False),
                             ("Qwen-OCC", r"Qwen3-4B $-$ OCC", True)):
        t.row(name, *[n2(get[(rd, L)]["auc_exact"], signed=signed) for L, _ in LBL])
        t.row("", *[cirow(get[(rd, L)]["lo_exact"], get[(rd, L)]["hi_exact"]) for L, _ in LBL])
    t.raw(r"\midrule")
    t.row(r"insufficient contexts", *[str(get[("OCC", L)]["n_I"]) for L, _ in LBL])
    t.raw(r"\bottomrule")
    n_ctx = get[("OCC", "qrels")]["n_S"] + get[("OCC", "qrels")]["n_P"] + get[("OCC", "qrels")]["n_I"]
    assert all(get[("OCC", L)]["n_I"] == get[("Qwen", L)]["n_I"] for L, _ in LBL)
    cap = (rf"AUROC of the reader's $p$(\textsc{{answerable}}) on the {n_ctx} audited BM25-fusion contexts "
           rf"({get[('OCC', 'qrels')]['n_I']} \qnone, {get[('OCC', 'qrels')]['n_S']} \textsc{{all}}) under the "
           r"qrels label, the phi-4 content label and the blind audit (\cyes{} vs.\ \cno). "
           rf"95\% bootstrap CIs over the {n_ctx} contexts.")
    write(t, "tab_audit_auroc", cap, "tab:audit-auroc")


# ───────────────────────── tab_mtrag ─────────────────────────
MTRAG = os.path.join(ROOT, "data", "drive", "reteco", "mtrag", "summary.csv")
MT_DISP = {"bm25_questions": "Qs", "bm25_lastturn": "Last", "bm25_rewrite": "Rewr.",
           "hybrid_rewrite": "Hybrid", "dense_rewrite": "Dense"}
MT_READ = [("OCC", "OCC-RAG-1.7B"), ("Qwen3-4B", "Qwen3-4B-Instruct-2507")]


def tab_mtrag():
    """MTRAG (mtrag.ipynb → summary.csv): retrievers in ascending order of nDCG@10, phi-4 label, AUROC of two readers.
    The content-label AUROC is bold if the CI of the difference "content − qrels" excludes zero."""
    if not os.path.exists(MTRAG):
        print("missing", MTRAG, "— skipping tab_mtrag")
        return
    S = defaultdict(dict)
    for r in csv.DictReader(open(MTRAG, encoding="utf-8")):
        S[(r["section"], r["job"], r["reader"])][r["metric"]] = (fl(r["value"]), fl(r["lo"]), fl(r["hi"]))
    nan3 = (float("nan"),) * 3
    lab = lambda j, m: S[("labels", j, "")].get(m, nan3)
    js = sorted([j for j in MT_DISP if ("labels", j, "") in S], key=lambda j: lab(j, "ndcg10")[0])
    t = Tab("@{}X" + "c" * len(js) + "@{}", size="small", sep=4.0)
    t.raw(r"\toprule")
    t.row("", *[MT_DISP[j] for j in js])
    t.raw(r"\midrule")
    t.row("nDCG@10", *[n2(lab(j, "ndcg10")[0]) for j in js])
    # label shares come from mtrag_anatomy.json (mtrag_v2.py): summary.csv is rounded to 4 decimals, so using it would round twice
    an = json.load(open(os.path.join(V2, "mtrag_anatomy.json")))
    t.row(r"no gold (\%)", *[pc(an[j]["share_qrels_none"]) for j in js])
    t.row(r"content \cno{} (\%)", *[pc(an[j]["share_judge_no"]) for j in js])
    t.row(r"false misses (\%)", *[pc(an[j]["fm"]) for j in js])
    t.row("precision", *[n2(an[j]["precision"]) for j in js])
    t.raw(r"\addlinespace[2pt]")
    t.row(("mc", len(js) + 1, "@{}l", r"\textit{AUROC of $p$(\textnormal{\textsc{answerable}})}"))
    for disp, rd in MT_READ:
        g = lambda j, m: S[("readers", j, rd)].get(m, nan3)
        if not any(("readers", j, rd) in S for j in js):
            t.row(disp, *[PEND] * len(js))
            continue
        def cont(j):
            v, (d, lo, hi) = n2(g(j, "auroc_content")[0]), g(j, "d_auroc")
            return rf"\textbf{{{v}}}" if not isnan(lo) and (lo > 0 or hi < 0) else v
        t.row(f"{disp}: qrels", *[n2(g(j, "auroc_qrels")[0]) for j in js])
        t.row(r"\quad content", *[cont(j) for j in js])
    t.raw(r"\bottomrule")
    cap = (r"MTRAG (777 answerable and partially answerable turns): the qrels label against the phi-4 content label "
           r"for five first-stage retrievers, ordered by nDCG@10. Qs: BM25 over all user questions; Last: BM25 over "
           r"the last turn; Rewr.: BM25 over MTRAG's rewritten question; Dense: Qwen3-Embedding-0.6B over the rewrite; "
           r"Hybrid: equal-weight min-max fusion of Rewr.\ and Dense. Definitions as in Table~\ref{tab:bias}. "
           r"Bold: content-label AUROC whose difference from the qrels-label AUROC has a 95\% bootstrap CI "
           r"(over turns, stratified by domain) excluding zero.")
    write(t, "tab_mtrag", cap, "tab:mtrag")


if __name__ == "__main__":
    for fn in (tab_bias, tab_contrast, tab_detectors, tab_agreement, tab_sameturn, tab_audit_auroc, tab_mtrag):
        fn()
