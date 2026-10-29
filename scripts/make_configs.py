"""Write one YAML per pre-registered run. Settings frozen in PREREGISTRATION.md come from configs/frozen.yaml."""
from pathlib import Path

import yaml

FROZEN = yaml.safe_load(open("configs/frozen.yaml"))
N_MAIN = FROZEN["N_main"]
NS = [N_MAIN // 2, N_MAIN, 2 * N_MAIN]
SEEDS = [0, 1, 2]


def write(group, run_id, **kw):
    d = Path("configs") / group
    d.mkdir(parents=True, exist_ok=True)
    ds = kw.get("dataset", "sim")
    budget = "flat" if kw.get("method") == "flat" and ds == "sim" else (
        "real" if ds.startswith("real:") else "partb" if ds.startswith("partb:") else "sim")
    cfg = dict(run_id=run_id, **FROZEN["train"], **FROZEN["budget"][budget], **kw)
    yaml.safe_dump(cfg, open(d / f"{run_id}.yaml", "w"), sort_keys=False)


def lr(method):
    return FROZEN["lr"][method]


if __name__ == "__main__":
    router = FROZEN["router"]
    # entropy proxies: one per dataset
    write("proxy", "proxy_sim", method="proxy", lr=lr("proxy"), seed=0)
    for name in FROZEN["real_datasets"]:
        write("proxy", f"proxy_{name}", method="proxy", dataset=f"real:{name}", lr=lr("proxy"), seed=0)
    # Part A simulation: 5 boundary methods x 3 N x 3 seeds, flat x 3 seeds
    for s in SEEDS:
        for N in NS:
            for m in ["learned", "fixed", "random", "entropy", "velocity"]:
                extra = dict(router) if m == "learned" else {}
                if m == "entropy":
                    extra["proxy_run"] = "runs/proxy_sim"
                write("sim", f"sim_{m}_N{N}_s{s}", method=m, N=N, seed=s, lr=lr(m), **extra)
        write("sim", f"sim_flat_s{s}", method="flat", N=1, seed=s, lr=lr("flat"))
        # ablations at N_main
        write("ablation", f"abl_object_N{N_MAIN}_s{s}", method="learned", N=N_MAIN, seed=s, lr=lr("learned"),
              inputs="robot+object", **router)
        write("ablation", f"abl_conv_N{N_MAIN}_s{s}", method="learned", N=N_MAIN, seed=s, lr=lr("learned"),
              encoder="conv", **router)
        # exploratory: refractory router (no boundary within r steps after a boundary)
        write("exploratory", f"exp_refractory_N{N_MAIN}_s{s}", method="learned", N=N_MAIN, seed=s, lr=lr("learned"),
              refractory=FROZEN["exploratory"]["refractory_steps"], **router)
        # real data: RoboChunk, Fixed, Entropy at N_real
        for name, n_real in FROZEN["real_datasets"].items():
            for m in ["learned", "fixed", "entropy"]:
                extra = dict(router) if m == "learned" else {}
                if m == "entropy":
                    extra["proxy_run"] = f"runs/proxy_{name}"
                write("real", f"real_{name}_{m}_N{n_real}_s{s}", method=m, dataset=f"real:{name}", N=n_real, seed=s,
                      lr=lr(m), **extra)
        # Part B novelty models trained on R only
        partb = dict(dataset="partb:R", splits=dict(train="R", val="R_val", test="R_heldout"))
        write("partb", f"partb_robochunk_N{N_MAIN}_s{s}", method="learned", N=N_MAIN, seed=s, lr=lr("learned"),
              **router, **partb)
        write("partb", f"partb_flat_s{s}", method="flat", N=1, seed=s, lr=lr("flat"), **partb)
    print("configs written")
