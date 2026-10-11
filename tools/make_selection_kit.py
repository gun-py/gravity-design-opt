import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FILES = [
    'tools/bench.py',
    'tools/choose_submission.py',
    'tools/official/validate_submission.py',
    'competition_data/round2/seeds.csv',
    'competition_data/round2/source.json',
    'artifacts/baselines/local_champion.zip',
    'artifacts/baselines/provenance.json',
    'dist/champion_repair.zip',
    'dist/n5_rescue.zip',
]


def main():
    out = ROOT / 'dist' / 'l2d_selection_kit.zip'
    hashes = {}
    with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:
        for name in FILES:
            path = ROOT / name
            if not path.is_file():
                raise FileNotFoundError(path)
            hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
            z.write(path, name)
        z.writestr('manifest.json', json.dumps({'sha256':hashes},indent=2))
    print(json.dumps({'path':str(out),'bytes':out.stat().st_size,
                      'sha256':hashlib.sha256(out.read_bytes()).hexdigest()},indent=2))


if __name__ == '__main__':
    main()
