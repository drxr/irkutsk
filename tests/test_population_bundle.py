"""Поставляемое население сохраняет проверенные значения и происхождение."""
import gzip
import hashlib
import json
from pathlib import Path

from data_access import optional_csv
from indicators import load_population

ROOT = Path(__file__).resolve().parents[1]


def test_bundled_population_is_complete_and_lossless():
    folder = ROOT / 'data/population'
    manifest = json.loads((folder / 'bundle_manifest.json').read_text())
    for item in manifest['files']:
        path = folder / item['file']
        raw = gzip.decompress(path.read_bytes()) if path.suffix == '.gz' else path.read_bytes()
        assert len(raw) == item['uncompressed_bytes']
        assert hashlib.sha256(raw).hexdigest() == item['uncompressed_sha256']
    population = load_population(folder / 'population_denominators_2013_2024.csv.gz')
    assert len(population) == 2056
    assert set(population.year) == set(range(2013, 2025))
    assert population.source_trace.ne('').all()
    municipal = optional_csv(ROOT, 'population/population_municipal_long.csv')
    assert len(municipal) == 2375
