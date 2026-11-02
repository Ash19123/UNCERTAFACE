"""Fetch selected MUG500+ skull meshes from Figshare WITHOUT downloading the 210 GB of CT volumes.

Dataset: Li J. et al., MUG500+ Repository, Figshare 2021, DOI 10.6084/m9.figshare.9616319 (licence CC BY 4.0).
Paper:   Li J. et al., MUG500+: Database of 500 high-resolution healthy human skulls and 29 craniotomy skulls and
         implants, Data in Brief 2021, DOI 10.1016/j.dib.2021.107524. Please cite both when using the data.

Real, anonymised patient-derived CT skull meshes. They are kept OUTSIDE git (external_data/ is git-ignored);
do not commit or redistribute them from this repository.

Each zip is read with HTTP range requests (the zip's central directory, then only the requested members), so only
the chosen .stl files cross the network. Downloaded files are treated as untrusted data: they go into their own
directory and are only ever parsed (never executed).

  python scripts/fetch_mug500.py --cases 1 --variants A    # pilot: one case
  python scripts/fetch_mug500.py --first 20 --variants clear
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

from remotezip import RemoteZip

API = "https://api.figshare.com/v2/articles/9616319"


def file_table() -> dict[str, dict]:
    with urllib.request.urlopen(API, timeout=60) as r:
        meta = json.load(r)
    return {f["name"]: f for f in meta["files"]}, meta["license"]["name"], meta["doi"], meta["version"]


def zip_for(case: int, table: dict) -> str:
    """Name of the zip that holds healthy case ``case`` (A0001..A0500)."""
    for name in table:
        m = re.fullmatch(r"A(\d{4})-A(\d{4})\.zip", name)
        if m and int(m.group(1)) <= case <= int(m.group(2)):
            return name
    raise KeyError(f"no zip for case {case}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="external_data/mug500/raw")
    ap.add_argument("--cases", type=int, nargs="*", help="explicit case numbers")
    ap.add_argument("--first", type=int, help="cases 1..N")
    ap.add_argument("--variants", default="clear", choices=["A", "clear", "both"],
                    help="A -> A####.stl, clear -> A####_clear.stl")
    a = ap.parse_args()
    cases = a.cases or list(range(1, (a.first or 1) + 1))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    table, lic, doi, ver = file_table()
    (out.parent / "SOURCE.json").write_text(json.dumps(dict(
        dataset="MUG500+ Repository", doi=doi, figshare_version=ver, licence=lic,
        citation=["Li J. et al. MUG500+ Repository. Figshare 2021. DOI 10.6084/m9.figshare.9616319",
                  "Li J. et al. MUG500+: Database of 500 High-resolution Healthy Human Skulls and 29 Craniotomy Skulls and Implants. Data in Brief 2021. DOI 10.1016/j.dib.2021.107524"],
        note="real anonymised CT-derived skull meshes; local use only, not redistributed by this repository"), indent=2))
    want = {"A": ["{n}.stl"], "clear": ["{n}_clear.stl"], "both": ["{n}.stl", "{n}_clear.stl"]}[a.variants]
    by_zip: dict[str, list[int]] = {}
    for c in cases:
        by_zip.setdefault(zip_for(c, table), []).append(c)
    t0, total = time.time(), 0
    for zname, cs in by_zip.items():
        url = table[zname]["download_url"]
        with RemoteZip(url) as z:
            members = {Path(i.filename).name: i for i in z.infolist() if not i.is_dir()}
            for c in cs:
                n = f"A{c:04d}"
                for pat in want:
                    fn = pat.format(n=n)
                    dest = out / n / fn
                    if dest.exists() and dest.stat().st_size == members[fn].file_size:
                        continue
                    z.extract(members[fn], path=out / "_tmp")
                    src = next((out / "_tmp").rglob(fn))
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    src.replace(dest)
                    total += members[fn].compress_size
                    print(f"{fn}: {members[fn].file_size / 1e6:.0f} MB  ({total / 1e6:.0f} MB transferred, {time.time() - t0:.0f}s)", flush=True)
    tmp = out / "_tmp"
    if tmp.exists():
        for p in sorted(tmp.rglob("*"), reverse=True):
            p.rmdir() if p.is_dir() else p.unlink()
        tmp.rmdir()


if __name__ == "__main__":
    sys.exit(main())
