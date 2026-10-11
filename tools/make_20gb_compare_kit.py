import hashlib
import json
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FILES = (
    'tools/bench.py',
    'tools/colab_20gb_compare.py',
    'tools/official/validate_submission.py',
    'artifacts/baselines/local_champion.zip',
    'dist/champion_v2.zip',
    'dist/n7_fsq.zip',
    'analysis/final_holdout_seeds.csv',
    'competition_data/round3/seeds.csv',
)


def main():
    destination = ROOT / 'dist/l2d_20gb_1h_compare_kit.zip'
    checksums = {}
    with zipfile.ZipFile(destination, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in FILES:
            source = ROOT / name
            if not source.is_file():
                raise FileNotFoundError(source)
            checksums[name] = hashlib.sha256(source.read_bytes()).hexdigest()
            archive.write(source, name)
        archive.writestr('manifest.json', json.dumps({'files_sha256': checksums}, indent=2))
    print(json.dumps({'path': str(destination), 'bytes': destination.stat().st_size,
                      'sha256': hashlib.sha256(destination.read_bytes()).hexdigest()}, indent=2))


if __name__ == '__main__':
    main()
