"""Показатели для брифа: сопоставимые группы, знаменатели и прогноз.

Не заполняем пропуски, не складываем диагнозы с вложенными подстроками.
Расчёты отделены от интерфейса и повторно используются в CLI и СППР.
"""
from io import BytesIO
import re

import numpy as np
import pandas as pd

from analytics import CATALOG, DEFAULT_POLICY, cell_value, diagnostic_rows, normalize_text, years_of

POP_KEYS = ['year', 'population_type', 'sex', 'residence', 'age_group']
AGE_COLS = {'0-14': ['6'], '15-17': ['7'], '18-19': ['8'],
            '20-39': ['9'], '40-59': ['10'], '60+': ['11'],
            '18+': ['8', '9', '10', '11']}
AGE_LABELS = {'all': 'Все возрасты', '0-14': 'Дети 0–14', '15-17': 'Подростки 15–17',
              '18-19': '18–19', '20-39': '20–39', '40-59': '40–59', '60+': '60+', '18+': 'Взрослые 18+'}
SEX_LABELS = {'both': 'Оба пола', 'female': 'Женщины', 'male': 'Мужчины'}
RESIDENCE_LABELS = {'all': 'Вся область', 'urban': 'Городское население', 'rural': 'Сельское население'}
TYPE_LABELS = {'annual_average': 'Среднегодовая численность', 'beginning_year': 'На начало года', 'end_year': 'На конец года (1 января следующего)'}


def load_population(source):
    """Проверить уникальность, территорию, разрезы и происхождение населения."""
    frame = pd.read_csv(BytesIO(source) if isinstance(source, bytes) else source,
                        dtype=str, keep_default_na=False)
    required = set(POP_KEYS + ['population', 'territory_name', 'territory_level', 'method', 'source_trace'])
    missing = required - set(frame)
    if missing:
        raise ValueError('В населении отсутствуют поля: ' + ', '.join(sorted(missing)))
    frame['year'] = pd.to_numeric(frame.year, errors='raise')
    if not frame.year.between(1900, 2100).all() or not frame.year.mod(1).eq(0).all():
        raise ValueError('Население: нужен целый год в допустимом диапазоне.')
    frame['year'] = frame.year.astype(int)
    frame['population'] = pd.to_numeric(frame.population, errors='coerce')
    if not np.isfinite(frame.population).all() or not frame.population.gt(0).all():
        raise ValueError('Население: численность должна быть положительным конечным числом.')
    if frame.duplicated(POP_KEYS).any():
        raise ValueError('Население: повторяющиеся ключи знаменателей.')
    if set(frame.territory_name) != {'Иркутская область'} or set(frame.territory_level) != {'region'}:
        raise ValueError('Для этой СППР нужны областные данные Иркутской области.')
    for field, allowed in [('sex', set(SEX_LABELS)), ('residence', set(RESIDENCE_LABELS)),
                           ('age_group', set(AGE_LABELS)), ('population_type', set(TYPE_LABELS))]:
        if not set(frame[field]).issubset(allowed):
            raise ValueError('Население: неизвестный разрез ' + field)
    allowed_methods = {'published', 'sum_single_ages', 'sum_published_mean_ages',
                       'average_adjacent_january', 'next_january_as_year_end'}
    if not set(frame.method).issubset(allowed_methods) or frame.source_trace.eq('').any():
        raise ValueError('Население: неизвестный метод или отсутствует происхождение.')
    if not frame.get('census_basis', pd.Series('', index=frame.index)).eq('census_2020').all():
        raise ValueError('Население: требуется сопоставимый областной пересчёт по ВПН 2020.')
    return frame


def combine(parts, operation='sum'):
    """Сложить непересекающиеся компоненты или вычесть подмножество."""
    result = parts[0].copy()
    result.update(value=np.nan, source_file='; '.join(dict.fromkeys(p.get('source_file', '') for p in parts)),
                  mapping=' | '.join(p.get('mapping', '') for p in parts),
                  quality_flag='digital' if all(p['quality_flag'] == 'digital' for p in parts) else 'mixed',
                  numerator_method='sum_components' if operation == 'sum' else 'total_minus_subset')
    if any(str(p['quality_flag']).startswith('ocr') for p in parts):
        result['quality_flag'] = 'ocr_derived'
    result['reason'] = '; '.join(dict.fromkeys(p.get('reason', '') for p in parts if p.get('reason')))
    if all(np.isfinite(p['value']) for p in parts):
        value = sum(p['value'] for p in parts) if operation == 'sum' else parts[0]['value'] - parts[1]['value']
        if value < 0:
            result['reason'] = 'Подмножество больше итога; требуется сверка'
        else:
            result.update(value=value, reason='')
    return result


def form10_count(data, year, table='2000', row='1', sex='both', residence='all', age_group='all', policy=DEFAULT_POLICY):
    """Получить разрез формы 10; мужской и городской — разности итогов.

    Форма не содержит совместного разреза пола и возраста. Его не создаём.
    Графы: 4 общий итог, 5 женщины, 6–11 возраста, 14–21 сельские.
    """
    if sex != 'both' and age_group != 'all':
        return dict(year=year, value=np.nan, reason='В форме нет совместного разреза пола и возраста',
                    quality_flag='missing', source_file='', mapping='', numerator_method='unavailable')
    if residence == 'urban':
        return combine([form10_count(data, year, table, row, sex, 'all', age_group, policy),
                        form10_count(data, year, table, row, sex, 'rural', age_group, policy)], 'difference')
    if sex == 'male':
        return combine([form10_count(data, year, table, row, 'both', residence, age_group, policy),
                        form10_count(data, year, table, row, 'female', residence, age_group, policy)], 'difference')
    offset = 10 if residence == 'rural' else 0
    columns = ['5'] if sex == 'female' else (['4'] if age_group == 'all' else AGE_COLS[age_group])
    parts = []
    for col in columns:
        actual = str(int(col) + offset)
        # Проверка заголовка защищает от смещения граф в новой редакции.
        pattern = 'женщ' if sex == 'female' else ''
        if age_group not in {'all', '18+'}:
            pattern = re.escape(age_group.replace('-', '')) if age_group != '60+' else '60'
        elif age_group == 'all' and sex == 'both':
            pattern = 'всего'
        item = cell_value(data, year, '10', table, row, actual, policy, pattern)
        item['numerator_method'] = 'source_cell'
        parts.append(item)
    return parts[0] if len(parts) == 1 else combine(parts)


def match_diagnosis(data, year, table, reference, form='10'):
    """Сопоставить диагноз по МКБ и определению, учитывая сдвиг строк."""
    name = normalize_text(reference['entity_name'])
    if reference['row_number'] == '1':
        return '1'
    patterns = {'psychoses': r'^психозы', 'nonpsychotic': r'^психическиерасстройстванепсихотичес|^непсихотическиепсихическиерасстройств',
                'intellectual': r'^умственнаяотсталость'}
    for key, pattern in patterns.items():
        if form == '10' and re.search(pattern, name) and 'изних' not in name:
            return diagnostic_rows(data, year, table).get(key)
    rows = data.loc[(data.year == year) & (data.form_number == form) & (data.table_code == table),
                    ['row_number', 'entity_name', 'icd10']].drop_duplicates()
    code = normalize_text(reference['icd10'])
    matched = rows.loc[rows.icd10.map(normalize_text).eq(code)] if code else rows.loc[rows.entity_name.map(normalize_text).eq(name)]
    if len(matched) > 1:
        matched = matched.loc[matched.entity_name.map(normalize_text).eq(name)]
    return str(matched.iloc[0].row_number) if len(matched) == 1 else None


def incidence_series(data, table='2000', sex='both', residence='all', age_group='all', policy=DEFAULT_POLICY, reference=None):
    """Абсолютный ряд заболеваемости/учёта с явными причинами пропусков."""
    records = []
    for year in years_of(data):
        row = match_diagnosis(data, year, table, reference) if reference is not None else '1'
        if row is None:
            item = dict(year=year, value=np.nan, reason='Нет однозначного соответствия диагноза',
                        quality_flag='definition_changed', mapping='', source_file='', numerator_method='unavailable')
        else:
            item = form10_count(data, year, table, row, sex, residence, age_group, policy)
        records.append(item)
    frame = pd.DataFrame(records)
    frame['label'] = 'Зарегистрированные за год' if table == '2000' else 'Впервые в жизни установленный диагноз'
    frame['unit'] = 'чел. по форме'
    frame['sex'], frame['residence'], frame['age_group'] = sex, residence, age_group
    frame['yoy_pct'] = frame.value.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan) * 100
    return frame


def service_diagnosis_series(data, metric_id, reference, policy=DEFAULT_POLICY):
    """Ряд нозологии формы 36 с теми же графами и проверкой смысла."""
    spec = CATALOG[metric_id]
    form, table, _, old_column, new_column, pattern = spec['slot']
    records = []
    for year in years_of(data):
        row = match_diagnosis(data, year, table, reference, form)
        if row is None:
            item = dict(year=year, value=np.nan, reason='Нет однозначного соответствия диагноза',
                        quality_flag='definition_changed', source_file='', mapping='')
        else:
            item = cell_value(data, year, form, table, row, new_column if year >= 2023 else old_column, policy, pattern)
        records.append(item)
    result = pd.DataFrame(records)
    result['label'] = reference['entity_name']
    result['unit'] = spec['unit']
    result['yoy_pct'] = result.value.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan) * 100
    return result


def normalize_series(series, population, population_type='annual_average', sex='both', residence='all', age_group='all', scale=100000, allow_calculated=False):
    """Соединить по году и точным разрезам; не подменять отсутствующую группу."""
    for field, expected in [('sex', sex), ('residence', residence), ('age_group', age_group)]:
        if field in series and not series[field].eq(expected).all():
            raise ValueError('Числитель и знаменатель имеют разные разрезы: ' + field)
    selected = population.loc[(population.population_type == population_type) & (population.sex == sex)
                              & (population.residence == residence) & (population.age_group == age_group)]
    columns = ['year', 'population', 'method', 'source_trace', 'reference_date']
    denominator = selected[columns].rename(columns={'method': 'denominator_method',
                    'source_trace': 'denominator_source', 'reference_date': 'denominator_date'})
    result = series.merge(denominator, on='year', how='left', validate='one_to_one').sort_values('year')
    result['absolute_value'] = result.value
    result['denominator_type'] = population_type
    result['denominator_sex'], result['denominator_residence'], result['denominator_age_group'] = sex, residence, age_group
    result['rate_reason'] = ''
    result.loc[result.population.isna(), 'rate_reason'] = 'Нет населения нужного года и разреза'
    calculated = result.denominator_method.eq('average_adjacent_january')
    if not allow_calculated:
        result.loc[calculated, 'rate_reason'] = 'Расчётный знаменатель выключен: метод нужно подтвердить'
    result.loc[result.absolute_value.isna(), 'rate_reason'] = result.loc[result.absolute_value.isna(), 'reason']
    result['value'] = result.absolute_value / result.population * scale
    result.loc[result.rate_reason.ne(''), 'value'] = np.nan
    result['reason'] = result.rate_reason
    result['unit'] = f'на {scale:,.0f} населения'.replace(',', ' ')
    result['yoy_pct'] = result.value.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan) * 100
    return result


def forecast_series(series, horizon=2):
    """Rolling-origin проверка линейного тренда и последнего значения.

    Берём только непрерывный хвост до последнего года выбранного периода.
    Не продолжаем устаревший ряд. Нужны 8 лет и минимум 3 тестовые точки.
    """
    ordered = series.sort_values('year')[['year', 'value']].copy()
    if ordered.empty or pd.isna(ordered.iloc[-1].value):
        return {'status': 'blocked', 'reason': 'В последнем году периода нет значения', 'forecast': pd.DataFrame(), 'backtest': pd.DataFrame()}
    tail = []
    expected = int(ordered.iloc[-1].year)
    for row in ordered.iloc[::-1].itertuples(index=False):
        if int(row.year) != expected or not np.isfinite(row.value):
            break
        tail.append((int(row.year), float(row.value)))
        expected -= 1
    tail.reverse()
    if len(tail) < 8:
        return {'status': 'blocked', 'reason': f'Непрерывный хвост — {len(tail)} лет; нужно минимум 8', 'forecast': pd.DataFrame(), 'backtest': pd.DataFrame()}
    x, y = np.array(tail, dtype=float).T
    back = []
    for split in range(5, len(y)):
        coeff = np.polyfit(x[:split] - x[0], y[:split], 1)
        trend = max(0., float(np.polyval(coeff, x[split] - x[0])))
        back.append(dict(train_last_year=int(x[split-1]), test_year=int(x[split]), actual=y[split],
                         last_value=y[split-1], linear_trend=trend))
    bt = pd.DataFrame(back)
    errors = {m: float((bt[m] - bt.actual).abs().mean()) for m in ['last_value', 'linear_trend']}
    selected = 'linear_trend' if errors['linear_trend'] < errors['last_value'] else 'last_value'
    coeff = np.polyfit(x - x[0], y, 1)
    points = []
    for step in range(1, horizon + 1):
        year = int(x[-1]) + step
        val = max(0., float(np.polyval(coeff, year - x[0]))) if selected == 'linear_trend' else float(y[-1])
        points.append(dict(year=year, value=val, model=selected))
    # Ошибка относится к однолетнему бэктесту. Не выдаём её за интервал горизонта 2 года.
    return dict(status='exploratory', reason='Исследовательский прогноз; метод выбран по однолетнему MAE',
                model=selected, mae=errors, observations=len(y), first_year=int(x[0]),
                forecast=pd.DataFrame(points), backtest=bt)


def brief_coverage():
    """Реестр исполнения запросов заказчика и конкретных недостающих данных."""
    return pd.DataFrame([
        ('Общая и первичная заболеваемость', 'Реализовано с ограничениями', '2013–2024; OCR исключён по умолчанию; средние 2022–2024 расчётные'),
        ('Пол, возраст, город/село и нозологии', 'Реализовано с ограничениями', 'Форма 10; совместного пола и возраста нет; возрастные средние 2022–2024 отсутствуют'),
        ('Диспансерное и консультативное наблюдение', 'Реализовано', 'Контингенты на конец года, движение, структура среди зарегистрированных по форме 10'),
        ('Инвалидность и госпитализация', 'Частично', 'Доступные показатели формы 36; охват уникальных пациентов госпитализацией не вычислен из событий поступления'),
        ('Кадры, укомплектованность, койки', 'Частично', 'Занятые должности и дневные места; нужен ряд формы 30, штатные должности, физлица, круглосуточные койки'),
        ('Россия и федеральные округа', 'Нет исходных данных', 'Нужны сопоставимые медицинские показатели и знаменатели РФ/ФО'),
        ('Муниципальные сравнения и карты', 'Нет медицинских данных и геометрии', 'Население есть; нужны муниципальные числители, коды, границы и согласование пересчётов'),
        ('Прогноз заболеваемости', 'Проверка реализована', 'В строгом режиме непрерывный хвост короткий; нужен проверенный PDF-ряд; исследовательский режим отделён'),
        ('Ожидаемая экономия времени 70%/80%', 'Требует измерения', 'Замерить длительность процесса до и после внедрения; это цель брифа, а не подтверждённый эффект'),
    ], columns=['request', 'status', 'limitation'])
