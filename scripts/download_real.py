"""Download a real dataset's parquet tables and metadata (no videos) at a pinned revision.
usage: python scripts/download_real.py so101"""
import json
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

from robochunk.data_real import DATASETS, ROOT

PINNED = {  # name: revision (v1 datasets were downloaded at the revisions in downloads/real/revisions.json)
    "so101": "f641879e22172be7e8161d5e6c1503c2d2feb657",
}

if __name__ == "__main__":
    name = sys.argv[1]
    rid, rev = DATASETS[name][0], PINNED[name]
    out = ROOT / rid.replace("/", "__")
    snapshot_download(rid, repo_type="dataset", revision=rev, allow_patterns=["data/*", "meta/*"], local_dir=out)
    revs_path = ROOT / "revisions.json"
    revs = json.load(open(revs_path))
    revs[rid] = rev
    json.dump(revs, open(revs_path, "w"), indent=1)
    print(name, rid, rev, "->", out, sum(f.stat().st_size for f in out.rglob("*") if f.is_file()), "bytes")
