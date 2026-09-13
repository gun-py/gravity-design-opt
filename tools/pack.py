# build & validate the 6 competition ZIPs (files flat at ZIP root)
import argparse
import ast
import importlib.util
import os
import re
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
L2D = ROOT / "l2d"
ENTRIES = ROOT / "entries"
DIST = ROOT / "dist"

SHARED = ["name_gen.py", "warm.py", "core.py", "prior.npz"]
FORBIDDEN = re.compile(r"^\s*(from|import)\s+differometor\b", re.MULTILINE)


def count_algo(text):
    n = 0
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.ClassDef):
            for b in node.bases:
                if (isinstance(b, ast.Name) and b.id == "OptimizationAlgorithm") \
                   or (isinstance(b, ast.Attribute) and b.attr == "OptimizationAlgorithm"):
                    n += 1; break
    return n


def check(files):
    errs = []
    for f in files:
        if f.suffix != ".py": continue
        text = f.read_text()
        if FORBIDDEN.search(text):
            errs.append(f"forbidden `import differometor` in {f.name}")
    return errs


def import_smoke(tmp, mods):
    errs = []; op = list(sys.path); sys.path.insert(0, str(tmp))
    try:
        for m in mods:
            spec = importlib.util.spec_from_file_location(f"_smoke_{m}", tmp / f"{m}.py")
            try: spec.loader.exec_module(importlib.util.module_from_spec(spec))
            except Exception as e: errs.append(f"{m}.py: {e}")
    finally:
        sys.path[:] = op
    return errs


def pack(entry, extra=None, out=DIST):
    d = ENTRIES / entry
    sub = d / "submission.py"; req = d / "requirements.txt"
    if not sub.is_file(): raise SystemExit(f"missing {sub}")
    if not req.is_file(): raise SystemExit(f"missing {req}")
    if (n := count_algo(sub.read_text())) != 1:
        raise SystemExit(f"{sub}: {n} OptimizationAlgorithm subclasses (need 1)")

    files = [sub, req] + [L2D / f for f in SHARED]
    if extra:
        for x in extra:
            p = d / x
            if not p.is_file(): raise SystemExit(f"extra missing: {p}")
            files.append(p)

    if errs := check(files): raise SystemExit("\n  ".join(["validation:"] + errs))

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for f in files: shutil.copy(f, tmp / f.name)
        if errs := import_smoke(tmp, ["name_gen", "warm", "core"]):
            raise SystemExit("\n  ".join(["import smoke:"] + errs))

        out.mkdir(parents=True, exist_ok=True)
        z = out / f"{entry}.zip"
        if z.exists(): z.unlink()
        with zipfile.ZipFile(z, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for f in files: zf.write(f, arcname=f.name)
        mb = z.stat().st_size / 1e6
        if mb > 50: raise SystemExit(f"{z} is {mb:.1f} MB (>50 limit)")
        print(f"OK  {z}  ({mb:.2f} MB, {len(files)} files)")
    return z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("entry", nargs="?")
    ap.add_argument("--extra", nargs="*", default=[])
    a = ap.parse_args()
    if not a.entry or a.entry == "all":
        for d in sorted(ENTRIES.iterdir()):
            if d.is_dir() and (d / "submission.py").exists():
                x = ["weights.npz"] if (d / "weights.npz").exists() else []
                try: pack(d.name, x)
                except SystemExit as e: print(f"FAIL {d.name}: {e}")
        return
    pack(a.entry, a.extra)


if __name__ == "__main__":
    main()
