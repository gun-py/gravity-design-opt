"""Create a self-contained Colab kit for the staged V3 experiment."""

import hashlib
import json
import zipfile
from pathlib import Path

from pack_champion_experiments import ROOT, VARIANTS, build


FILES = (
    'tools/bench.py', 'tools/colab_20gb_compare.py', 'tools/champion_experiment.py',
    'tools/smoke_champion_experiment.py',
    'tools/official/validate_submission.py',
    'analysis/champion_v3_experiment_plan.json',
    'analysis/final_holdout_seeds.csv', 'competition_data/round3/seeds.csv',
) + tuple(f'dist/champion_v3_{name}.zip' for name in VARIANTS)


def main():
    build(ROOT / 'dist')
    destination = ROOT / 'dist/l2d_champion_v3_experiment_kit.zip'
    checksums = {}
    with zipfile.ZipFile(destination, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in FILES:
            source = ROOT / name
            content = source.read_bytes()
            checksums[name] = hashlib.sha256(content).hexdigest()
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, content)
        archive.writestr('manifest.json', json.dumps({'files_sha256': checksums}, indent=2))
    print(json.dumps({'path': str(destination), 'bytes': destination.stat().st_size,
                      'sha256': hashlib.sha256(destination.read_bytes()).hexdigest()}, indent=2))


if __name__ == '__main__':
    main()
