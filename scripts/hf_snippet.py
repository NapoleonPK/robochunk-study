"""Model-card snippet (scripts/stage_hf.py embeds it): load a checkpoint from the Hub and plot its boundaries on one test episode."""
import matplotlib.pyplot as plt, torch, yaml
from huggingface_hub import snapshot_download
from robochunk.data import Normalizer
from robochunk.datasets import load_dataset
from robochunk.train import DEFAULTS, build_model, evaluate

run = "exp_refractory_N10_s0"  # any folder under runs/ in this repo
ckpt = snapshot_download("PAOP123/robochunk", allow_patterns=f"runs/{run}/*")
snapshot_download("PAOP123/robochunk-sim", repo_type="dataset", allow_patterns="sim/test.npz", local_dir="data")

d = f"{ckpt}/runs/{run}"
cfg = {**DEFAULTS, **yaml.safe_load(open(f"{d}/config.yaml"))}
eps = load_dataset({**cfg, "splits": {"test": "test"}})["test"]
model = build_model(cfg, eps[0]["x"].shape[1])
model.load_state_dict(torch.load(f"{d}/best.pt", map_location="cpu"))
ev = evaluate(model, cfg, eps[:1], None, Normalizer.load(f"{d}/norm.json"), torch.device("cpu"), keep=True)

ep, b = eps[0], ev["bounds"][0]
plt.step(range(len(ep["phase"])), ep["phase"], where="post", label="true phase")
plt.vlines(b[b > 0], -0.5, 6.5, color="C1", label="router boundaries")
plt.yticks(range(7), ["approach", "descend", "close", "lift", "transport", "open", "retreat"])
plt.xlabel("step (25 Hz)"); plt.legend(); plt.show()
