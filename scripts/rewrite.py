#!/usr/bin/env python3
"""LLM-generated query variants: rewrite, topic switch and a HyDE / Query2doc passage per turn.

One LLM call per batch of turns. For each turn the model sees ONLY the query
and conversation_history (no answer, subquestion_reasoning, original_query or
doc_id) and returns three fields:

    rewrite       the turn rewritten as a self-contained question         (QD, CHIQ)
    topic_switch  whether the sub-topic changed since the previous turn   (TS, CHIQ)
    passage       a hypothetical reference-document passage answering it  (HyDE / Query2Doc)

If the model refuses a batch (safety filter), the batch is split into single
turns; a turn that is refused again keeps its original text without generation
and is marked "fallback": true (rewrite = query, passage = "").

A turn is identified by the pair (domain, topic_id): in the six domains taken
from BRIGHT, conversations are numbered 0, 1, 2, ... so the same topic_id
occurs in several domains.

Raw responses are cached line by line in runs/rewrites/raw/<split>.jsonl, so an
interrupted run resumes where it stopped. Query variants are built from them as
runs/rewrites/<variant>/<split>.jsonl (domain, topic_id, text) for retrieve.py --queries:

    qd        rewrite
    qd_hist   rewrite + history
    ts        turn + history, but without history if the topic switched
    hyde      passage instead of the turn
    q2d       turn + history + passage (official hist + pseudo-document)
    qd_hyde   rewrite + passage
    qd_q2d    rewrite + history + passage

    python scripts/rewrite.py --split dev                # generate and build
    python scripts/rewrite.py --split dev --build-only   # rebuild variants only

The paper's raw responses (runs/rewrites/raw/{dev,train}.jsonl with
config_{dev,train}.json) are released as reteco-expansions.zip. Unzipped at the
repository root, they let --build-only rebuild the paper's variants without
calling a model; it needs only them and the RETECO data in data/reteco/.

call_llm() calls the `claude` command-line client in print mode (`claude -p`);
the paper's expansions were generated this way with Claude Sonnet 5 at low
reasoning effort, in batches of ten turns (the defaults below). Any
chat-completion client can be substituted: call_llm() only needs to send the
system prompt SYSTEM and the task text and return the response text (and,
optionally, the names of the models that answered). main() also records
`claude --version` in raw/config_<split>.json; change that field as well.
"""
import argparse
import datetime as dt
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "reteco", "track2_recor")
OUT = os.path.join(ROOT, "runs", "rewrites")
DOMAINS = ["biology", "drones", "earth_science", "economics", "hardware", "law",
           "medicalsciences", "politics", "psychology", "robotics", "sustainable_living"]
NO_HIST = "No previous conversation."

SYSTEM = ("You prepare search queries for a document retrieval system. The corpus consists of "
          "reference documents: encyclopedia articles, textbook passages, forum answers. You never "
          "answer the user yourself; you only produce texts that will be sent to a search engine.")

TASK = """Each item below is one turn of a conversation: the conversation so far (earlier user questions with the assistant's answers) and the user's latest question.

For every item produce:
- "rewrite": the latest question rewritten as a single self-contained question that is understandable without the conversation. Resolve pronouns, ellipsis and references ("it", "those", "that approach") using the conversation. Keep only what the latest question asks; do not add the earlier questions and do not answer it.
- "topic_switch": true if the latest question moves to a different sub-topic, so that the earlier questions and answers would not help to find documents for it; false if it continues, narrows or follows up on the earlier discussion. Always false when there is no previous conversation.
- "passage": a 60-100 word passage that could appear in a reference document and directly answers the latest question. Declarative, factual, encyclopedic tone; no chat phrasing; do not restate the question.

Return only a JSON array with one object per item, in the same order:
[{"id": "...", "rewrite": "...", "topic_switch": false, "passage": "..."}]

<items>
%s
</items>"""


def load_turns(split):
    """All turns of a split in file order: (key "domain/topic_id", query, history)."""
    out = []
    for dom in DOMAINS:
        for c in json.load(open(os.path.join(DATA, dom, f"benchmark_{split}.json"), encoding="utf-8")):
            for t in c["turns"]:
                out.append((f"{dom}/{c['id']}_turn_{t['turn_id']}", t["query"],
                            t.get("conversation_history") or NO_HIST))
    return out


def key_of(r):
    return f"{r['domain']}/{r['topic_id']}"


def load_raw(path):
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return {key_of(r): r for r in map(json.loads, filter(str.strip, f))}


def render(batch):
    items = []
    for tid, q, h in batch:
        items.append(f'<item id="{tid}">\n<conversation>\n{h}\n</conversation>\n'
                     f"<latest_question>{q}</latest_question>\n</item>")
    return TASK % "\n".join(items)


def call_llm(prompt, model, effort, timeout):
    env = {k: v for k, v in os.environ.items()
           if k not in ("ANTHROPIC_BASE_URL", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")}
    cmd = ["claude", "-p", "--model", model, "--effort", effort, "--tools", "",
           "--no-session-persistence", "--setting-sources", "", "--system-prompt", SYSTEM,
           "--output-format", "json", prompt]
    # separate process group: on timeout, also kill the CLI's child processes; otherwise
    # they keep the output pipe open and the wait hangs forever
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                         cwd=tempfile.gettempdir(), env=env, stdin=subprocess.DEVNULL,
                         start_new_session=True)
    try:
        out, _ = p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
        p.communicate()
        raise
    d = json.loads(out)
    if d.get("is_error"):
        raise RuntimeError(str(d.get("result"))[:300])
    return d["result"], list((d.get("modelUsage") or {}).keys())


def parse(text, batch):
    """JSON array from the model response; keeps only valid objects with ids from the batch."""
    a, b = text.find("["), text.rfind("]")
    arr = json.loads(text[a:b + 1])
    want = {tid for tid, _, _ in batch}
    good = {}
    for o in arr:
        if (isinstance(o, dict) and o.get("id") in want and isinstance(o.get("rewrite"), str)
                and o["rewrite"].strip() and isinstance(o.get("topic_switch"), bool)
                and isinstance(o.get("passage"), str) and o["passage"].strip()):
            good[o["id"]] = o
    return good


def generate(split, model, effort, batch_size, workers, timeout):
    raw_path = os.path.join(OUT, "raw", f"{split}.jsonl")
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    done = set(load_raw(raw_path))
    todo = [t for t in load_turns(split) if t[0] not in done]
    print(f"{split}: done {len(done)}, remaining {len(todo)}", flush=True)
    if not todo:
        return
    batches = [todo[i:i + batch_size] for i in range(0, len(todo), batch_size)]
    lock = threading.Lock()
    stats = {"ok": 0, "fail": 0}
    t0 = time.time()

    def write(recs):
        with lock, open(raw_path, "a", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                stats["ok"] += 1

    def record(key, o, models):
        dom, tid = key.split("/", 1)
        return {"domain": dom, "topic_id": tid, "rewrite": o["rewrite"].strip(),
                "topic_switch": o["topic_switch"], "passage": o["passage"].strip(),
                "model": models[0] if models else model}

    def work(batch):
        pending = list(batch)
        for attempt in range(3 if len(batch) > 1 else 2):
            if not pending:
                break
            try:
                text, models = call_llm(render(pending), model, effort, timeout)
                got = parse(text, pending)
            except Exception as e:
                print(f"  error ({attempt + 1}/3): {type(e).__name__}: {str(e)[:160]}", flush=True)
                got, models = {}, []
                time.sleep(5 * (attempt + 1))
            write([record(tid, got[tid], models) for tid, _, _ in pending if tid in got])
            pending = [p for p in pending if p[0] not in got]
        if not pending:
            return
        if len(batch) > 1:                    # batch refused: retry one turn at a time
            for item in pending:
                work([item])
            return
        key, q, _ = pending[0]                # single turn refused: keep the original text
        print(f"  {key}: no generation, using the original turn", flush=True)
        dom, tid = key.split("/", 1)
        write([{"domain": dom, "topic_id": tid, "rewrite": q, "topic_switch": False,
                "passage": "", "model": None, "fallback": True}])
        with lock:
            stats["fail"] += 1

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(work, b) for b in batches]
        for i, _ in enumerate(as_completed(futs), 1):
            if i % 5 == 0 or i == len(futs):
                print(f"  batches {i}/{len(futs)}, turns ok {stats['ok']}, "
                      f"failed {stats['fail']}, {time.time() - t0:.0f}s", flush=True)
    if stats["fail"]:
        print(f"!! {stats['fail']} turns without generation (model refusal); original text used", flush=True)


def with_hist(text, h):
    return text if h == NO_HIST else f"{text}\n\nConversation History:\n{h}"


def build(split):
    raw = load_raw(os.path.join(OUT, "raw", f"{split}.jsonl"))
    turns = load_turns(split)
    missing = [t for t, _, _ in turns if t not in raw]
    if missing:
        sys.exit(f"no LLM response for {len(missing)} turns, e.g. {missing[:3]}; run generation first")
    variants = {
        "qd": lambda q, h, o: o["rewrite"],
        "qd_hist": lambda q, h, o: with_hist(o["rewrite"], h),
        "ts": lambda q, h, o: q if o["topic_switch"] else with_hist(q, h),
        "hyde": lambda q, h, o: o["passage"] or q,
        "q2d": lambda q, h, o: with_hist(q, h) + ("\n\n" + o["passage"] if o["passage"] else ""),
        "qd_hyde": lambda q, h, o: o["rewrite"] + ("\n\n" + o["passage"] if o["passage"] else ""),
        "qd_q2d": lambda q, h, o: with_hist(o["rewrite"], h) + ("\n\n" + o["passage"] if o["passage"] else ""),
    }
    for name, fn in variants.items():
        path = os.path.join(OUT, name, f"{split}.jsonl")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for key, q, h in turns:
                dom, tid = key.split("/", 1)
                f.write(json.dumps({"domain": dom, "topic_id": tid, "text": fn(q, h, raw[key])},
                                   ensure_ascii=False) + "\n")
    sw = sum(raw[t]["topic_switch"] for t, _, h in turns if h != NO_HIST)
    later = sum(1 for _, _, h in turns if h != NO_HIST)
    fb = sum(1 for t, _, _ in turns if raw[t].get("fallback"))
    print(f"{split}: variants {', '.join(variants)} -> runs/rewrites/<variant>/{split}.jsonl; "
          f"topic switch in {sw} of {later} T2+ turns ({100 * sw / max(later, 1):.0f}%); "
          f"without generation {fb}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="dev", choices=["train", "dev"])
    ap.add_argument("--model", default="sonnet")
    ap.add_argument("--effort", default="low", help="simple task, low is enough")
    ap.add_argument("--batch", type=int, default=10, help="turns per call")
    ap.add_argument("--workers", type=int, default=6, help="parallel CLI calls")
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--build-only", action="store_true")
    a = ap.parse_args()
    if not a.build_only:
        generate(a.split, a.model, a.effort, a.batch, a.workers, a.timeout)
        cfg = {"script": "scripts/rewrite.py", "args": vars(a), "system": SYSTEM, "task": TASK,
               "cli": subprocess.run(["claude", "--version"], capture_output=True, text=True).stdout.strip(),
               "date": dt.datetime.now().isoformat(timespec="seconds")}
        json.dump(cfg, open(os.path.join(OUT, "raw", f"config_{a.split}.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
    build(a.split)


if __name__ == "__main__":
    main()
