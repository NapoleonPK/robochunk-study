"""Shared README / Hugging Face card sections, generated from runs/results_*.json, git and docs/timeline_evidence.json.
Every result number comes from the result files; nothing is typed by hand.
usage: python scripts/make_cards.py   (rewrites the generated block of README.md; the HF cards are built by stage_hf.py)"""
import hashlib
import json
import subprocess
from pathlib import Path


R = Path("runs")
GH = "https://github.com/NapoleonPK/robochunk-study"
HF_USER = "PAOP123"
MODEL, DATA = f"{HF_USER}/robochunk", f"{HF_USER}/robochunk-sim"
FIG1 = "figures/fig1_episodes.png"
NOT_AFFILIATED = ("**Independent re-implementation of H-Net-style dynamic chunking for robot trajectories. Not affiliated "
                  "with the H-Net authors (Sukjun Hwang, Brandon Wang, Albert Gu) or Cartesia.**")
NAMES = dict(fixed="Fixed", random="Random", entropy="Entropy", velocity="Velocity minima")
START, END = "<!-- generated:start (scripts/make_cards.py) -->", "<!-- generated:end -->"

sg = lambda x: ("+" if x >= 0 else "−") + f"{abs(x):.3f}"
ci = lambda c: f"[{sg(c[0])}, {sg(c[1])}]"


def results_table():
    A = json.load(open(R / "results_partA.json"))
    REAL = json.load(open(R / "results_real.json"))
    B = json.load(open(R / "results_partB.json"))
    rc = json.load(open(R / "results_partB_posthoc.json"))["rho_ci"]
    V2 = json.load(open(R / "results_v2.json"))
    S2, Q2 = V2["sim"], V2["real"]
    h1, h2, h4 = A["H1"], A["H2"], B["h4"]
    h1_pass = all(d["pass_"] for d in h1["diffs"].values())
    yes = lambda b: "**pass**" if b else "**fail**"
    rows = [r for r in h2["rows"]]
    h2txt = "; ".join(f"N={r['N']}: {r['learned_bits']:.3f} vs {r['fixed_bits']:.3f}" +
                      ("" if r["compression_matched"] else f" (compression {r['learned_compression']:.1f} < 0.8 N, counts as fail)")
                      for r in rows)
    h3 = "; ".join(f"{nm} {REAL[d]['H3']['ratio']:.2f}× (p = {REAL[d]['H3']['p']:.3f})" for d, nm in (("so100", "SO-100"), ("koch", "Koch")))
    b1 = S2["H1b_v2"]["diffs"]
    lo_b = min(b1, key=lambda k: b1[k]["mean"])
    L = ["| test | criterion | key numbers (from `runs/results_*.json`) | verdict |", "|---|---|---|---|",
         f"| H1 | learned router − every baseline ≥ 0.10 boundary F1 (±3), CI excludes 0 | learned {h1['mean_f1']['learned']:.3f} vs "
         f"Fixed {h1['mean_f1']['fixed']:.3f}; difference {sg(h1['diffs']['fixed']['mean'])} {ci(h1['diffs']['fixed']['ci'])} | {yes(h1_pass)} |",
         f"| H2 | learned bits/step < Fixed at ≥ 2 of 3 N, compression matched | {h2txt} | {yes(h2['pass_'])} |",
         f"| H3 | ≥ 2× chance near gripper events and p < 0.01, on both datasets | {h3} | **significant but below threshold** |",
         f"| H4 | Spearman ρ(novelty, gain) ≥ 0.6 over 11 clean candidates | ρ = {h4['rho']:.3f} (p = {h4['p']:.4f}); 95% CI "
         f"[{rc['fisher_bonett_wright_95'][0]:.2f}, {rc['fisher_bonett_wright_95'][1]:.2f}] (post hoc); corrupted data scored most "
         f"novel in {h4['failure_mode_count']} of 4 | {yes(h4['pass_'])} |",
         f"| H1-v2 | refractory router − every baseline ≥ 0.10 raw F1 (fresh data) | vs Fixed {sg(S2['H1_v2']['diffs']['fixed']['mean'])} "
         f"{ci(S2['H1_v2']['diffs']['fixed']['ci'])} | {yes(S2['H1_v2']['pass_'])} (predicted) |",
         f"| H1b-v2 | chance-corrected (circular shift) difference > 0 vs every baseline, CI excludes 0 | vs Fixed "
         f"{sg(b1['fixed']['mean'])} {ci(b1['fixed']['ci'])}; smallest (vs {NAMES[lo_b]}) {sg(b1[lo_b]['mean'])} {ci(b1[lo_b]['ci'])} | "
         f"{yes(S2['H1b_v2']['pass_'])} |",
         f"| H3-v2 | refractory router, ≥ 2× chance near gripper events, SO-100 and Koch | " +
         "; ".join(f"{nm} {Q2[d]['test']['ratio']:.2f}× (p = {Q2[d]['test']['p']:.3f})" for d, nm in (("so100", "SO-100"), ("koch", "Koch"))) +
         f" | **{Q2['H3_v2']}** |",
         f"| H3b-v2 | same rule on a fresh dataset (SO-101, 5 test episodes) | {Q2['so101']['test']['ratio']:.2f}× "
         f"(p = {Q2['so101']['test']['p']:.3f}) | {yes(Q2['H3b_v2'] == 'pass')} |"]
    return "\n".join(L)


def git(*a):
    return subprocess.run(["git", *a], capture_output=True, text=True).stdout


def prereg_hashes():
    L = ["| file | SHA-256 | tag (commit) |", "|---|---|---|"]
    for tag in ("prereg", "prereg-amend1", "prereg-v2"):
        c = git("rev-parse", "--short", f"{tag}^{{commit}}").strip()
        for f in ("PREREGISTRATION.md", "PREREGISTRATION_v2.md"):
            blob = subprocess.run(["git", "show", f"{tag}:{f}"], capture_output=True).stdout
            if blob:
                L.append(f"| `{f}` | `{hashlib.sha256(blob).hexdigest()}` | `{tag}` (`{c}`) |")
    for f in ("PREREGISTRATION.md", "PREREGISTRATION_v2.md"):
        L.append(f"| `{f}` | `{hashlib.sha256(Path(f).read_bytes()).hexdigest()}` | current file |")
    return ("SHA-256 of the pre-registration files at each tag and now (`shasum -a 256 PREREGISTRATION*.md`). "
            "`PREREGISTRATION.md` changed once, by Amendment 1, before any main result; neither file has changed since its "
            "last tag.\n\n" + "\n".join(L))


def timeline():
    ev = json.load(open("docs/timeline_evidence.json"))
    L = ["Timestamps recorded by third parties (UTC). Source details: `docs/timeline_evidence.json`.", "",
         "| UTC | source | what it shows |", "|---|---|---|"]
    for e in ev["events"]:
        L.append(f"| {e['utc'].replace('T', ' ').replace('Z', '')} | {e['source']} | {e['what']} |")
    L += ["", "Self-reported (git and local run logs, not independent): " +
          "; ".join(f"{e['utc'].replace('T', ' ').replace('Z', '')} {e['what']}" for e in ev["self_reported"]) + ".", "",
          "**History rewrite and commit map.** Before the repository was made public (2026-10-30), its git history was "
          "rewritten once with `git filter-repo --replace-text`, only to replace the author's personal e-mail address in "
          "`docs/BRIEF.md`. File contents, dates, messages and tags are otherwise unchanged, so the pre-registration files "
          "have the same SHA-256 in the old and the new commits (checked for all three tags), but every commit hash "
          "changed. Emails, the earlier private repository and run `git.txt` files cite the old hashes; "
          "`docs/HISTORY.md` and `docs/commit-map.txt` map them to the new ones (for example `1f1faa0` → `7ffc024`, "
          "tag `prereg`). " + ev["note"].split(". ", 1)[1]]
    return "\n".join(L)


def readme_block():
    return "\n\n".join([
        START, "## Results", results_table(), f"![Episode boundaries]({FIG1})",
        "*True motion phases (bands) and the boundaries each rule places on six simulated test episodes, N = 10.*",
        "![v2 results](figures/fig7_v2_f1.png)",
        "## Pre-registration hashes", prereg_hashes(), "## Timeline evidence", timeline(), END])


def sim_counts():
    from robochunk.data_sim import load
    rows = ["| split | episodes | steps | Parquet | seeds |", "|---|---|---|---|---|"]
    seeds = dict(train="0–2399", val="100000–100299", test="200000–200299", fresh="700000–700299 (v2)",
                 pilot="900000–900199")
    for s in ("train", "val", "test", "fresh", "pilot"):
        e = load(f"data/sim/{s}.npz")
        rows.append(f"| `{s}` | {len(e)} | {sum(len(x['action']) for x in e)} | {'yes' if s in ('test', 'fresh') else 'npz only'} | {seeds[s]} |")
    pb = ["| Part B file | episodes |", "|---|---|"]
    for f in sorted(Path("data/partb").glob("*.npz"), key=lambda p: (p.stem[0], len(p.stem), p.stem)):
        pb.append(f"| `{f.stem}` | {len(load(str(f)))} |")
    return "\n".join(rows), "\n".join(pb)


def fig_url(repo, typ):
    pre = "datasets/" if typ == "dataset" else ""
    return f"https://huggingface.co/{pre}{repo}/resolve/main/{FIG1}"


if __name__ == "__main__":
    t = Path("README.md").read_text()
    if START in t:
        a, b = t.index(START), t.index(END) + len(END)
        t = t[:a] + readme_block() + t[b:]
    else:
        raise SystemExit("README.md has no generated block markers")
    Path("README.md").write_text(t)
    print("README.md generated block updated")
