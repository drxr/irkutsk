"""Фактические инсайты и переходы от итога к группе и исходной ячейке.

Функции не зависят от Streamlit. Рост учёта не считается автоматически
плохим, снижение — хорошим. Пол и возраст не пересекаются без числителя.
"""
from copy import deepcopy
from collections.abc import Mapping

import numpy as np
import pandas as pd

from analytics import DEFAULT_POLICY, diagnostic_rows, point
from indicators import (AGE_LABELS, RESIDENCE_LABELS, SEX_LABELS,
                        incidence_series, normalize_series)

DIAGNOSIS_LABELS = {'all': 'Все нозологии', 'psychoses': 'Психозы и состояния слабоумия',
                    'nonpsychotic': 'Непсихотические расстройства',
                    'intellectual': 'Умственная отсталость'}
AXES = {'age': 'Возраст · оба пола', 'sex': 'Пол · все возрасты',
        'residence': 'Город и село', 'diagnosis': 'Основные группы заболеваний',
        'source': 'Источник и следующий шаг'}
STATE_FIELDS = ['sex', 'age_group', 'residence', 'diagnosis', 'axis']


def number(value, digits=0):
    """Русское представление конечного числа."""
    return f'{value:,.{digits}f}'.replace(',', '\u202f').replace('.', ',') if np.isfinite(value) else '—'


def compare_years(series, year):
    """Сравнить ровно два соседних календарных года, без заполнения дыр."""
    current, previous = point(series, year), point(series, year - 1)
    valid = np.isfinite(current.value) and np.isfinite(previous.value)
    change = current.value - previous.value if valid else np.nan
    pct = change / previous.value * 100 if valid and previous.value > 0 else np.nan
    return dict(current=current.value, previous=previous.value, difference=change,
                change_pct=pct, comparable=bool(valid), year=int(year), previous_year=int(year - 1),
                reason=str(current.get('reason', '')))


def insight(series, year):
    """Понятный вывод по видимому ряду; не приписывать изменению причину."""
    change = compare_years(series, year)
    unit = str(series.unit.iloc[0]) if not series.empty else ''
    if not np.isfinite(change['current']):
        return dict(title='Для выбранной группы нет допустимого значения',
                    detail=change['reason'] or 'Нужен источник нужного года и разреза.', tone='neutral')
    if not change['comparable']:
        return dict(title=f"За {year} год: {number(change['current'], 1 if unit.startswith('на ') else 0)} {unit}",
                    detail=f'Сравнение с {year - 1} годом недоступно. Пропуск не заменён ближайшим годом.', tone='neutral')
    delta = change['difference']
    direction = 'увеличилось' if delta > 0 else 'снизилось' if delta < 0 else 'не изменилось'
    if unit == '%':
        title = f"Изменение к {year - 1} году: {number(delta, 1)} п.п."
    elif delta == 0:
        title = f'Значение к {year - 1} году не изменилось'
    else:
        title = f"Значение {direction} на {number(abs(delta), 1 if unit.startswith('на ') else 0)} {unit}"
    detail = f"{year - 1}: {number(change['previous'], 1)} → {year}: {number(change['current'], 1)}."
    if np.isfinite(change['change_pct']) and unit != '%':
        detail += f" Изменение: {number(change['change_pct'], 1)}%."
    elif change['previous'] == 0:
        detail += ' При нулевой базе процент не рассчитывается.'
    detail += ' Причины изменения требуют отдельной проверки.'
    return dict(title=title, detail=detail, tone='neutral')


def signal_tone(value, previous, threshold=15, invalid=False):
    """Красный — нарушение проверки, зелёный — пройдена проверка, не рост/спад."""
    if invalid:
        return 'negative'
    if not np.isfinite(value) or not np.isfinite(previous):
        return 'neutral'
    if previous > 0 and min(value, previous) >= 20 and abs((value / previous - 1) * 100) >= threshold:
        return 'attention'
    return 'positive'


def initial_slice(axis='age'):
    """Начало маршрута; все группы относятся к одной территории."""
    return dict(sex='both', age_group='all', residence='all', diagnosis='all',
                axis=axis, history=[])


def selected_code(event, allowed):
    """Принимать только одиночный клик по каноническому коду текущего графика."""
    if not isinstance(event, Mapping):
        return None
    selection = event.get('selection', {})
    if not isinstance(selection, Mapping):
        return None
    points = selection.get('points', [])
    if not isinstance(points, list) or len(points) != 1 or not isinstance(points[0], Mapping):
        return None
    custom = points[0].get('customdata', [])
    if not isinstance(custom, (list, tuple)) or not custom:
        return None
    code = str(custom[0])
    return code if code in {str(value) for value in allowed} else None


def drill_selection(state, event, axis, allowed):
    """Перейти на следующий уровень, исключая недоступный пол × возраст."""
    code = selected_code(event, allowed)
    result = deepcopy(state)
    if code is None:
        return result
    valid = {'age': set(AGE_LABELS) - {'all', '18+'}, 'sex': {'female', 'male'},
             'residence': {'urban', 'rural'}, 'diagnosis': set(DIAGNOSIS_LABELS) - {'all'}}
    if code not in valid.get(axis, set()):
        return result
    result['history'] = result.get('history', []) + [{key: result[key] for key in STATE_FIELDS}]
    if axis == 'age':
        result.update(age_group=code, sex='both', axis='residence')
    elif axis == 'sex':
        result.update(sex=code, age_group='all', axis='residence')
    elif axis == 'residence':
        result.update(residence=code, axis='diagnosis')
    else:
        result.update(diagnosis=code, axis='source')
    return result


def previous_slice(state):
    """Вернуть предыдущий уровень вместе с его фильтрами."""
    result = deepcopy(state)
    if result.get('history'):
        snapshot = result['history'].pop()
        result.update(snapshot)
    return result


def diagnosis_reference(data, year, table, code):
    if code == 'all':
        return None
    row = diagnostic_rows(data, year, table).get(code)
    selected = data.loc[(data.year == year) & (data.form_number == '10') &
                        (data.table_code == table) & (data.row_number == row)]
    return selected.iloc[0].to_dict() if not selected.empty else False


def slice_series(data, population, year, table, state, policy=DEFAULT_POLICY,
                 rates=False, calculated=False):
    """Одна и та же фильтрация для графика, вывода, источника и выгрузки."""
    reference = diagnosis_reference(data, year, table, state['diagnosis'])
    result = incidence_series(data, table, state['sex'], state['residence'],
                              state['age_group'], policy, reference if reference is not False else None)
    if reference is False:
        result['value'] = np.nan
        result['reason'] = 'Не найдено однозначное определение группы заболеваний'
        result['source_file'] = ''
        result['mapping'] = ''
    if rates:
        result = normalize_series(result, population, sex=state['sex'], residence=state['residence'],
                                  age_group=state['age_group'], allow_calculated=calculated)
    return result


def breakdown(data, population, year, table, state, axis, policy=DEFAULT_POLICY,
              rates=False, calculated=False):
    """Непересекающиеся группы одного разреза; доля всегда от полного итога."""
    options = {'age': {k: v for k, v in AGE_LABELS.items() if k not in {'all', '18+'}},
               'sex': {k: v for k, v in SEX_LABELS.items() if k != 'both'},
               'residence': {k: v for k, v in RESIDENCE_LABELS.items() if k != 'all'},
               'diagnosis': {k: v for k, v in DIAGNOSIS_LABELS.items() if k != 'all'}}[axis]
    base = dict(state)
    if axis == 'age':
        base.update(age_group='all', sex='both')
    if axis == 'sex':
        base.update(age_group='all', sex='both')
    if axis == 'residence':
        base['residence'] = 'all'
    if axis == 'diagnosis':
        base['diagnosis'] = 'all'
    total = point(slice_series(data, population, year, table, base, policy), year).value
    records = []
    for code, label in options.items():
        child = dict(base)
        child[{'age': 'age_group', 'sex': 'sex', 'residence': 'residence', 'diagnosis': 'diagnosis'}[axis]] = code
        series = slice_series(data, population, year, table, child, policy, rates, calculated)
        current = point(series, year)
        change = compare_years(series, year)
        absolute = current.get('absolute_value', current.value)
        records.append(dict(code=code, label=label, value=current.value,
                            absolute_value=absolute, difference=change['difference'],
                            previous=change['previous'], change_pct=change['change_pct'],
                            share_pct=absolute / total * 100 if np.isfinite(total) and total > 0 and np.isfinite(absolute) and absolute <= total else np.nan,
                            reason=current.reason, quality_flag=current.quality_flag,
                            source_file=current.get('source_file', ''), mapping=current.get('mapping', ''),
                            unit=str(series.unit.iloc[0])))
    return pd.DataFrame(records)
