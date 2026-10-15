"""Проверка синхронизации с новым парсером без изменения рабочего набора."""
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def test_population_and_numerator_mismatch():
    from indicators import incidence_series, load_population, normalize_series
    from analytics import load_frames
    import pytest
    from data_access import medical_inputs, population_input
    data = load_frames(medical_inputs(ROOT))
    path = population_input(ROOT)
    if path is None:
        pytest.skip('Население не подключено')
    population = load_population(path)
    with pytest.raises(ValueError, match='разные разрезы'):
        normalize_series(incidence_series(data, sex='female'), population)


def test_update_copies_only_after_validation(tmp_path):
    application = tmp_path / 'app'
    application.mkdir()
    for name in ['analytics.py', 'indicators.py', 'update_data.py']:
        shutil.copy2(ROOT / name, application / name)
    (application / 'docs').mkdir()
    # Источник не содержит необходимых файлов — существующие данные целы.
    destination = application / 'data/medical'
    destination.mkdir(parents=True)
    (destination / 'old.txt').write_text('keep')
    process = subprocess.run([sys.executable, str(application / 'update_data.py'), str(tmp_path / 'missing')], capture_output=True)
    assert process.returncode != 0
    assert (destination / 'old.txt').read_text() == 'keep'
