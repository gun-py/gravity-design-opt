"""Build four isolated Champion V3 experiments without touching the V2 ZIP."""

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

from official.validate_submission import validate_zip


ROOT = Path(__file__).resolve().parents[1]
ENTRY = ROOT / 'entries/champion_v3'
VARIANTS = ('control', 'rescue', 'late_lr', 'polish')
PLACEHOLDER = 'VARIANT = "control"  # Packaging replaces this exact line for each isolated ZIP.'
ENTRY_FILES = ('submission.py', 'core.py', 'warm.py', 'rescue.py', 'requirements.txt')
SHARED_FILES = ('name_gen.py', 'box.py', 'fit.py', 'prior.npz')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def contents(variant):
    if variant not in VARIANTS:
        raise ValueError(f'unknown variant {variant}')
    files = {name: (ENTRY / name).read_bytes() for name in ENTRY_FILES}
    source = files['submission.py'].decode()
    if source.count(PLACEHOLDER) != 1:
        raise ValueError('variant placeholder missing or duplicated')
    files['submission.py'] = source.replace(PLACEHOLDER, PLACEHOLDER.replace('"control"', f'"{variant}"')).encode()
    files.update({name: (ROOT / 'l2d' / name).read_bytes() for name in SHARED_FILES})
    return files


def build(out, replace=False):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    baseline = ROOT / 'artifacts/baselines/local_champion.zip'
    v2 = ROOT / 'dist/champion_v2.zip'
    before = {str(path): digest(path.read_bytes()) for path in (baseline, v2)}
    rows = {}
    for variant in VARIANTS:
        path = out / f'champion_v3_{variant}.zip'
        payload = contents(variant)
        if path.exists() and not replace:
            with zipfile.ZipFile(path) as archive:
                existing = {name: archive.read(name) for name in archive.namelist()}
            if existing != payload:
                raise FileExistsError(f'{path} differs; pass --replace to rebuild deliberately')
        else:
            with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
                for name, data in payload.items():
                    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                    info.compress_type = zipfile.ZIP_DEFLATED
                    info.external_attr = 0o644 << 16
                    archive.writestr(info, data)
        failures = [issue.render() for issue in validate_zip(str(path)) if issue.severity == 'FAIL']
        if failures:
            raise ValueError(f'{path}: {failures}')
        rows[variant] = {'path': str(path), 'sha256': digest(path.read_bytes()),
                         'files_sha256': {name: digest(data) for name, data in payload.items()},
                         'features': ['screen', 'basin', 'state', 'reinject'],
                         'experiment': variant}
    after = {str(path): digest(path.read_bytes()) for path in (baseline, v2)}
    if before != after:
        raise RuntimeError('preserved Champion or V2 archive changed during packaging')
    manifest = {'preserved': before, 'variants': rows}
    (out / 'champion_v3_experiments.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=ROOT / 'dist')
    parser.add_argument('--replace', action='store_true')
    args = parser.parse_args()
    result = build(args.out, args.replace)
    print(json.dumps({name: row['sha256'] for name, row in result['variants'].items()}, indent=2))


if __name__ == '__main__':
    main()
