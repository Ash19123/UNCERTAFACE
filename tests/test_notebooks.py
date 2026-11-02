import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_notebooks_are_valid_json_and_code_compiles(tmp_path):
    subprocess.run([sys.executable, str(ROOT / "scripts/make_notebooks.py")], check=True, cwd=ROOT)
    files = sorted((ROOT / "notebooks").glob("*.ipynb"))
    assert [f.name[:2] for f in files] == [f"{i:02d}" for i in range(1, 10)]
    for f in files:
        nb = json.loads(f.read_text())
        assert nb["nbformat"] == 4 and any(c["cell_type"] == "markdown" for c in nb["cells"])
        for i, c in enumerate(nb["cells"]):
            if c["cell_type"] == "code":
                compile("".join(c["source"]), f"{f.name}[{i}]", "exec")      # syntax check only (not executed)
