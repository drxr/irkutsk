"""Пути данных: текущий комплект парсера, прежний формат и локальная папка."""
import os
from pathlib import Path

import pandas as pd


def data_root(root):
    """Разрешить внешние данные без копирования в публичный репозиторий."""
    configured = os.environ.get('IRKUTSK_DATA_DIR')
    return Path(configured).expanduser().resolve() if configured else Path(root) / 'data'


def medical_inputs(root):
    """Найти две формы нового или старого комплекта; не смешивать версии."""
    folder = data_root(root)
    for candidate in [folder / 'medical', folder]:
        paths = [candidate / f'form_{number}_all_years_long.csv' for number in [10, 36]]
        if all(path.is_file() for path in paths):
            return paths
    for name in ['psychiatry_all_long.csv', 'psychiatry_all_long.csv.gz']:
        for candidate in [folder / 'medical', folder]:
            if (candidate / name).is_file():
                return [candidate / name]
    return []


def empty_population():
    """Типизированный пустой набор сохраняет пропуски относительных метрик."""
    columns = ['population_type', 'sex', 'residence', 'age_group', 'method',
               'source_trace', 'reference_date', 'territory_name', 'territory_level']
    frame = pd.DataFrame({name: pd.Series(dtype='str') for name in columns})
    frame['year'] = pd.Series(dtype='int64')
    frame['population'] = pd.Series(dtype='float64')
    return frame


def population_input(root):
    """Найти знаменатели: обновлённый CSV имеет приоритет перед сжатым комплектом."""
    path = data_root(root) / 'population/population_denominators_2013_2024.csv'
    if path.is_file():
        return path
    compressed = path.with_suffix(path.suffix + '.gz')
    return compressed if compressed.is_file() else None


def optional_csv(root, relative):
    """Вернуть дополнительную таблицу или None, если она не передана."""
    path = data_root(root) / relative
    if not path.is_file():
        path = path.with_suffix(path.suffix + '.gz')
    return pd.read_csv(path, keep_default_na=False) if path.is_file() else None
