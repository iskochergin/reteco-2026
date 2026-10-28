"""Reranker harm vs. holes in the qrels (dev, 858 turns, 11 domains). Not reported in the paper.

What is computed:
  1. Per-domain nDCG@10: bm25_fuse3 (first stage), Qwen3-Reranker-4B, ReasonRank-7B; deltas with
     turn-level bootstrap CIs (1000 resamples, seed 0, strata = domains).
  2. Per-domain hole proxy from phi-4 labels on bm25_fuse3 contexts (top-5): the share of qrels-none
     contexts where phi-4 says yes (yes/(yes+no), partial reported separately); and a breakdown of
     these "false misses" by evidence documents: near-duplicate of gold / adjacent chunk of the same
     page (BRIGHT) / gold of another turn in the dialogue / independent document.
  3. Spearman (n=11 domains) of the rerank delta vs. the hole proxy, vs. first-stage strength
     (confounder), and partial correlation controlling for first-stage strength. Permutation p and
     turn-level bootstrap CIs. SE vs. BRIGHT groups.
  4. In the top-10 of each run: how many unjudged documents are near-duplicates of gold
     (5-gram Jaccard >= 0.7) or adjacent chunks of the same page; nDCG@10 with qrels expanded by
     such documents (identically for all runs; pool = union of the top-10 lists).
CPU only, precomputed files only. Output: docs/paper/out/reranker_holes*.{json,csv,tex,png}.
"""
import json, re, os, collections, itertools
import numpy as np
import pytrec_eval
from scipy.stats import spearmanr, rankdata

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
DATA = f'{ROOT}/data/reteco/track2_recor'
RUNS = f'{ROOT}/data/drive/reteco/runs'
SUFF = f'{ROOT}/data/drive/reteco/suff/phi-4_2c-bm25_fuse3_dev.jsonl'
OUT = f'{ROOT}/docs/paper/out'
SE = ['drones', 'hardware', 'law', 'medicalsciences', 'politics']
BRIGHT = ['biology', 'earth_science', 'economics', 'psychology', 'robotics', 'sustainable_living']
DOMS = SE + BRIGHT
RUN = {'fuse3': 'bm25_fuse3_dev', 'qwen_rr': 'rerank_bm25_fuse3_dev_qd',
       'reasonrank': 'llmrr_reasonrank-7B_bm25_fuse3_dev_qd_k20w20s10'}
RR = ['qwen_rr', 'reasonrank']
B, SEED, JAC = 1000, 0, 0.7
os.makedirs(OUT, exist_ok=True)


def read_qrels(d):
    q = collections.defaultdict(dict)
    for l in open(f'{DATA}/{d}/qrels_dev.txt'):
        t, _, doc, r = l.split()
        q[t][doc] = int(r)
    return dict(q)


def read_run(path):
    r = collections.defaultdict(dict)
    for l in open(path):
        t, _, doc, _, s, _ = l.split()
        r[t][doc] = float(s)
    return dict(r)


def top(run_t, k=10):
    # same ordering as trec_eval: score descending, ties broken by doc_id descending
    return [d for d, _ in sorted(run_t.items(), key=lambda x: (x[1], x[0]), reverse=True)[:k]]


def ndcg(qrels, run):
    ev = pytrec_eval.RelevanceEvaluator(qrels, {'ndcg_cut.10'})
    res = ev.evaluate(run)
    return {t: res.get(t, {}).get('ndcg_cut_10', 0.0) for t in qrels}


def page(doc):
    # BRIGHT: 'folder/Color_2_5.txt' -> 'folder/Color'; earth_science: 'f/Sahara3_14.txt' -> 'f/Sahara'
    # (trailing digits and _<number> are the part/chunk index within the page; a heuristic)
    return re.sub(r'\d*(_\d+)*\.txt$', '', doc) if '/' in doc else None


def dialog(t):
    return t.rsplit('_turn_', 1)[0]


def shingles(text):
    w = re.findall(r'\w+', text.lower())
    if len(w) < 5:
        return {tuple(w)} if w else set()
    return {tuple(w[i:i + 5]) for i in range(len(w) - 4)}


def jac(a, b):
    return len(a & b) / len(a | b) if a and b else 0.0


# ---------- loading ----------
qrels = {d: read_qrels(d) for d in DOMS}
runs = {d: {k: read_run(f'{RUNS}/{v}/{d}/run.trec') for k, v in RUN.items()} for d in DOMS}
phi = {}
for l in open(SUFF):
    r = json.loads(l)
    phi[(r['domain'], r['topic_id'])] = r
assert len(phi) == 858

need = collections.defaultdict(set)
for d in DOMS:
    for t, g in qrels[d].items():
        need[d] |= set(g)
        for k in RUN:
            need[d] |= set(top(runs[d][k].get(t, {})))
        need[d] |= set(phi[(d, t)]['context_ids'])
sh = {}
for d in DOMS:
    for l in open(f'{DATA}/{d}/documents.jsonl'):
        i = l.find('"doc_id": "')
        did = l[i + 11:l.find('"', i + 11)]
        if did in need[d]:
            r = json.loads(l)
            sh[(d, r['doc_id'])] = shingles(r['content'])
    miss = [x for x in need[d] if (d, x) not in sh]
    assert not miss, (d, miss[:3])

# gold of the other turns in the same dialogue
gold_dialog = {d: collections.defaultdict(set) for d in DOMS}
for d in DOMS:
    for t, g in qrels[d].items():
        gold_dialog[d][dialog(t)] |= set(g)


def classify(d, t, doc, cache={}):
    """Unjudged doc: 'dup' (Jaccard >= 0.7 with gold), 'page' (same page, BRIGHT),
    'other_turn' (gold of another turn in the dialogue) or None."""
    key = (d, t, doc)
    if key in cache:
        return cache[key]
    g = qrels[d][t]
    if doc in g:
        c = 'gold'
    elif max(jac(sh[(d, doc)], sh[(d, x)]) for x in g) >= JAC:
        c = 'dup'
    elif page(doc) and page(doc) in {page(x) for x in g}:
        c = 'page'
    elif doc in gold_dialog[d][dialog(t)]:
        c = 'other_turn'
    else:
        c = None
    cache[key] = c
    return c


# ---------- 1. Per-turn nDCG: original and expanded qrels ----------
per = {}  # (variant, run, d) -> np.array over turns (in sorted(topics) order)
topics = {d: sorted(qrels[d]) for d in DOMS}
for d in DOMS:
    pool = {t: set().union(*(top(runs[d][k].get(t, {})) for k in RUN)) for t in topics[d]}
    q_dup, q_lib = {}, {}
    for t in topics[d]:
        cls = {x: classify(d, t, x) for x in pool[t]}
        q_dup[t] = dict(qrels[d][t], **{x: 1 for x, c in cls.items() if c == 'dup'})
        q_lib[t] = dict(q_dup[t], **{x: 1 for x, c in cls.items() if c == 'page'})
    for name, q in [('orig', qrels[d]), ('plus_dup', q_dup), ('plus_dup_page', q_lib)]:
        for k in RUN:
            nd = ndcg(q, runs[d][k])
            per[(name, k, d)] = np.array([nd[t] for t in topics[d]])

# ---------- 2. Hole proxy from phi-4 ----------
holes = {}
fm_cat = collections.Counter()
for d in DOMS:
    cnt = collections.Counter(); cat = collections.Counter(); hole_vec = []
    for t in topics[d]:
        r = phi[(d, t)]
        if r['suff_qrels'] != 'none':
            continue
        cnt[r['label']] += 1
        if r['label'] in ('yes', 'no'):
            hole_vec.append(1.0 if r['label'] == 'yes' else 0.0)
        if r['label'] == 'yes':
            ctx = r['context_ids']
            ev = [ctx[i - 1] for i in r['evidence_docs'] if 1 <= i <= len(ctx)] or ctx
            cs = {classify(d, t, x) for x in ev}
            c = next((c for c in ('dup', 'page', 'other_turn') if c in cs), 'independent')
            cat[c] += 1
    fm_cat.update(cat)
    holes[d] = dict(n_none=sum(cnt.values()), yes=cnt['yes'], partial=cnt['partial'], no=cnt['no'],
                    hole_rate=cnt['yes'] / max(1, cnt['yes'] + cnt['no']),
                    hole_incidence=cnt['yes'] / len(topics[d]),
                    vec=np.array(hole_vec), **{f'fm_{c}': cat[c] for c in ('dup', 'page', 'other_turn', 'independent')})

# ---------- 3. Unjudged near-duplicates / neighbours in the top-10 ----------
top10 = {}
for d in DOMS:
    for k in RUN:
        dup = []; pg = []; ot = []
        for t in topics[d]:
            cs = [classify(d, t, x) for x in top(runs[d][k].get(t, {}))]
            dup.append(cs.count('dup')); pg.append(cs.count('page')); ot.append(cs.count('other_turn'))
        top10[(k, d)] = dict(dup=np.array(dup, float), page=np.array(pg, float), other_turn=np.array(ot, float))

# ---------- Bootstrap (strata = domains, the same indices for all quantities) ----------
rng = np.random.default_rng(SEED)
idx = {d: rng.integers(0, len(topics[d]), size=(B, len(topics[d]))) for d in DOMS}
# separate indices over qrels-none contexts for the hole proxy
idx_h = {d: rng.integers(0, max(1, len(holes[d]['vec'])), size=(B, max(1, len(holes[d]['vec'])))) for d in DOMS}


def ci(a):
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]


def dom_mean_bs(v, d):  # v: per-turn array -> (point estimate, B bootstrap samples)
    return float(v.mean()), v[idx[d]].mean(1)


def hole_bs(d):
    v = holes[d]['vec']
    return v[idx_h[d]].mean(1) if len(v) else np.full(B, np.nan)


rows = []
bs_delta = {k: {} for k in RR}; bs_fs = {}; bs_hole = {}; bs_dup = {k: {} for k in RR}
for d in DOMS:
    fs, fs_b = dom_mean_bs(per[('orig', 'fuse3', d)], d); bs_fs[d] = fs_b
    bs_hole[d] = hole_bs(d)
    row = dict(domain=d, group='SE' if d in SE else 'BRIGHT', n_turns=len(topics[d]), ndcg_fuse3=fs)
    for k in RR:
        rr, _ = dom_mean_bs(per[('orig', k, d)], d)
        dv = per[('orig', k, d)] - per[('orig', 'fuse3', d)]
        dl, dl_b = dom_mean_bs(dv, d); bs_delta[k][d] = dl_b
        row[f'ndcg_{k}'] = rr; row[f'delta_{k}'] = dl
        row[f'delta_{k}_lo'], row[f'delta_{k}_hi'] = ci(dl_b)
        for var in ('plus_dup', 'plus_dup_page'):
            row[f'delta_{k}_{var}'] = float((per[(var, k, d)] - per[(var, 'fuse3', d)]).mean())
        # gain in unjudged duplicates + neighbours in the top-10 relative to the first stage
        ex = (top10[(k, d)]['dup'] + top10[(k, d)]['page']) - (top10[('fuse3', d)]['dup'] + top10[('fuse3', d)]['page'])
        row[f'dupnb_gain_{k}'], bs_dup[k][d] = dom_mean_bs(ex, d)
    for k in RUN:
        for c in ('dup', 'page', 'other_turn'):
            row[f'top10_{c}_{k}'] = float(top10[(k, d)][c].mean())
    h = holes[d]
    row.update({x: h[x] for x in ('n_none', 'yes', 'partial', 'no', 'hole_rate', 'hole_incidence',
                                  'fm_dup', 'fm_page', 'fm_other_turn', 'fm_independent')})
    row['hole_rate_lo'], row['hole_rate_hi'] = ci(bs_hole[d])
    rows.append(row)


def partial_spearman(x, y, z):
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    res = lambda a: a - np.polyval(np.polyfit(rz, a, 1), rz)
    return float(np.corrcoef(res(rx), res(ry))[0, 1])


def perm_p(x, y, n=100000, seed=SEED):
    g = np.random.default_rng(seed); obs = abs(spearmanr(x, y)[0])
    ry = rankdata(y); rx = rankdata(x)
    cnt = sum(abs(np.corrcoef(rx, g.permutation(ry))[0, 1]) >= obs - 1e-12 for _ in range(n))
    return (cnt + 1) / (n + 1)


corr = {}
H = np.array([r['hole_rate'] for r in rows]); FS = np.array([r['ndcg_fuse3'] for r in rows])
HI = np.array([r['hole_incidence'] for r in rows])
Hb = np.stack([bs_hole[d] for d in DOMS], 1); FSb = np.stack([bs_fs[d] for d in DOMS], 1)
for k in RR:
    D = np.array([r[f'delta_{k}'] for r in rows]); Db = np.stack([bs_delta[k][d] for d in DOMS], 1)
    G = np.array([r[f'dupnb_gain_{k}'] for r in rows]); Gb = np.stack([bs_dup[k][d] for d in DOMS], 1)
    Dd = np.array([r[f'delta_{k}_plus_dup_page'] for r in rows])
    out = {}
    for name, x, xb in [('hole_rate', H, Hb), ('ndcg_fuse3', FS, FSb), ('dupnb_gain', G, Gb)]:
        rho, p = spearmanr(x, D)
        rb = np.array([spearmanr(xb[i], Db[i])[0] for i in range(B)])
        out[name] = dict(rho=float(rho), p_scipy=float(p), p_perm=perm_p(x, D), ci_turn_bs=ci(rb[~np.isnan(rb)]))
    out['hole_incidence'] = dict(rho=float(spearmanr(HI, D)[0]), p_perm=perm_p(HI, D),
                                 partial_given_ndcg_fuse3=partial_spearman(HI, D, FS))
    out['hole_rate|ndcg_fuse3_partial'] = partial_spearman(H, D, FS)
    out['hole_rate_vs_ndcg_fuse3'] = float(spearmanr(H, FS)[0])
    out['hole_rate_vs_delta_plus_dup_page'] = float(spearmanr(H, Dd)[0])
    for g, ds in [('SE', SE), ('BRIGHT', BRIGHT)]:
        rho = spearmanr([holes[d]['hole_rate'] for d in ds], [r[f'delta_{k}'] for r in rows if r['domain'] in ds])[0]
        out[f'within_{g}_rho'] = float(rho)
    corr[k] = out

# ---------- Groups and macro averages ----------
groups = {}
for g, ds in [('SE', SE), ('BRIGHT', BRIGHT), ('ALL', DOMS)]:
    gi = {}
    gi['ndcg_fuse3'] = float(np.mean([per[('orig', 'fuse3', d)].mean() for d in ds]))
    for var in ('orig', 'plus_dup', 'plus_dup_page'):
        for k in RR:
            pt = np.mean([(per[(var, k, d)] - per[(var, 'fuse3', d)]).mean() for d in ds])
            bs = np.mean([(per[(var, k, d)] - per[(var, 'fuse3', d)])[idx[d]].mean(1) for d in ds], 0)
            gi[f'delta_{k}_{var}'] = dict(mean=float(pt), ci=ci(bs))
        for k in RUN:
            gi[f'ndcg_{k}_{var}'] = float(np.mean([per[(var, k, d)].mean() for d in ds]))
    yes = sum(holes[d]['yes'] for d in ds); no = sum(holes[d]['no'] for d in ds)
    pa = sum(holes[d]['partial'] for d in ds)
    pooled = np.concatenate([holes[d]['vec'] for d in ds])
    pb = np.concatenate([holes[d]['vec'][idx_h[d]] for d in ds if len(holes[d]['vec'])], 1).mean(1)
    gi['hole'] = dict(n_none=yes + no + pa, yes=yes, partial=pa, no=no, rate_pooled=float(pooled.mean()), ci=ci(pb),
                      rate_macro=float(np.mean([holes[d]['hole_rate'] for d in ds])))
    gi['false_miss_cats'] = {c: sum(holes[d][f'fm_{c}'] for d in ds) for c in ('dup', 'page', 'other_turn', 'independent')}
    for k in RUN:
        for c in ('dup', 'page', 'other_turn'):
            gi[f'top10_{c}_{k}'] = float(np.mean([top10[(k, d)][c].mean() for d in ds]))
    groups[g] = gi

# ---------- Output ----------
cols = list(rows[0].keys())
with open(f'{OUT}/reranker_holes_domains.csv', 'w') as f:
    f.write(','.join(cols) + '\n')
    for r in rows:
        f.write(','.join(f'{r[c]:.4f}' if isinstance(r[c], float) else str(r[c]) for c in cols) + '\n')
json.dump(dict(config=dict(runs=RUN, suff=os.path.relpath(SUFF, ROOT), jaccard=JAC, ngram=5, bootstrap=B, seed=SEED,
                           hole_rate='phi-4 yes/(yes+no) among bm25_fuse3 top-5 contexts with no qrels-gold',
                           pool='union of top-10 of the three runs; expansions identical for all runs'),
               domains=rows, groups=groups, spearman=corr),
          open(f'{OUT}/reranker_holes.json', 'w'), indent=1, ensure_ascii=False)

# LaTeX table
f3 = lambda x: f'{x:+.3f}'.replace('+-', '-')
lines = [r'\begin{tabular}{llrrrrr}', r'\toprule',
         r'Group & Domain & nDCG$_{\mathrm{BM25}}$ & $\Delta$Qwen3-RR & $\Delta$ReasonRank & hole rate & $\Delta$dup+page@10 (Qwen / RR) \\',
         r'\midrule']
for r in rows:
    lines.append(f"{r['group']} & {r['domain'].replace('_', ' ')} & {r['ndcg_fuse3']:.3f} & {f3(r['delta_qwen_rr'])} & "
                 f"{f3(r['delta_reasonrank'])} & {r['hole_rate']:.2f} ({r['yes']}/{r['yes'] + r['no']}) & "
                 f"{r['dupnb_gain_qwen_rr']:+.2f} / {r['dupnb_gain_reasonrank']:+.2f} \\\\")
lines += [r'\midrule']
for g in ('SE', 'BRIGHT'):
    gi = groups[g]
    lines.append(f"{g} & macro & {gi['ndcg_fuse3']:.3f} & {f3(gi['delta_qwen_rr_orig']['mean'])} & "
                 f"{f3(gi['delta_reasonrank_orig']['mean'])} & {gi['hole']['rate_pooled']:.2f} & \\\\")
lines += [r'\bottomrule', r'\end{tabular}']
open(f'{OUT}/reranker_holes.tex', 'w').write('\n'.join(lines) + '\n')

# terminal summary
print('domain      fs    dQwen   dRR    hole  (y/p/n)  fm dup/page/ot/ind  top10 dup+page fs/qw/rr')
for r in rows:
    print(f"{r['domain'][:11]:11s} {r['ndcg_fuse3']:.3f} {r['delta_qwen_rr']:+.3f} {r['delta_reasonrank']:+.3f} "
          f"{r['hole_rate']:.2f} ({r['yes']}/{r['partial']}/{r['no']})  {r['fm_dup']}/{r['fm_page']}/{r['fm_other_turn']}/{r['fm_independent']}"
          f"   {r['top10_dup_fuse3'] + r['top10_page_fuse3']:.2f}/{r['top10_dup_qwen_rr'] + r['top10_page_qwen_rr']:.2f}/"
          f"{r['top10_dup_reasonrank'] + r['top10_page_reasonrank']:.2f}")
print(json.dumps(groups, indent=0)[:4000])
print(json.dumps(corr, indent=1))

# ---------- Figure: rerank delta vs. first-stage strength and vs. the hole proxy ----------
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
COL = {'SE': '#2a78d6', 'BRIGHT': '#eb6834'}
MK = {'SE': 'o', 'BRIGHT': 's'}
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({'font.size': 8, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
                     'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.7), sharey=True)
for ax, (xk, xl) in zip(axes, [('ndcg_fuse3', 'first-stage nDCG@10 (BM25 fuse3)'),
                               ('hole_rate', 'hole rate: P(phi-4 yes | no gold in top-5)')]):
    ax.axhline(0, color=GRID, lw=1, zorder=0)
    for r in rows:
        g = r['group']
        ax.plot([r[xk]] * 2, [r['delta_qwen_rr'], r['delta_reasonrank']], color=GRID, lw=1, zorder=1)
        ax.scatter(r[xk], r['delta_reasonrank'], s=28, marker=MK[g], color=COL[g], edgecolor='#fcfcfb', lw=1, zorder=3)
        ax.scatter(r[xk], r['delta_qwen_rr'], s=28, marker=MK[g], facecolor='#fcfcfb', edgecolor=COL[g], lw=1.2, zorder=2)
        if xk == 'ndcg_fuse3':
            nm = {'medicalsciences': 'med. sci.', 'sustainable_living': 'sust. living',
                  'earth_science': 'earth sci.'}.get(r['domain'], r['domain'])
            dx = -4 if r['domain'] in ('psychology', 'sustainable_living') else 4
            ax.annotate(nm, (r[xk], r['delta_reasonrank']), xytext=(dx, 2), textcoords='offset points',
                        fontsize=6, color=INK2, ha='right' if dx < 0 else 'left')
    rq = corr['qwen_rr'][xk]['rho']; rr = corr['reasonrank'][xk]['rho']
    ax.set_title(f'Spearman ρ: ReasonRank {rr:+.2f}, Qwen3-RR {rq:+.2f} (n=11)', fontsize=7, color=INK2, loc='left')
    ax.set_xlabel(xl)
axes[0].set_ylabel('Δ nDCG@10 vs first stage')
from matplotlib.lines import Line2D
h = [Line2D([], [], ls='', marker='o', color=COL['SE'], label='SE domains'),
     Line2D([], [], ls='', marker='s', color=COL['BRIGHT'], label='BRIGHT domains'),
     Line2D([], [], ls='', marker='o', color=INK2, label='ReasonRank-7B (filled)'),
     Line2D([], [], ls='', marker='o', markerfacecolor='white', markeredgecolor=INK2, label='Qwen3-Reranker-4B (hollow)')]
fig.legend(handles=h, loc='lower center', ncol=4, frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig(f'{OUT}/reranker_holes.png', dpi=200, facecolor='#fcfcfb')
