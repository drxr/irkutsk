"""Экраны нового брифа; вычисления находятся в indicators.py."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analytics import CATALOG, cell_value, diagnostic_rows, point
from indicators import (AGE_LABELS, RESIDENCE_LABELS, SEX_LABELS, TYPE_LABELS,
                        brief_coverage, forecast_series, incidence_series, normalize_series, service_diagnosis_series)


def show_series(series, year, years, ui, title='Показатель'):
    """Единый вид ряда: выбранная точка, график, причины пропусков и CSV."""
    selected = series.loc[series.year.between(*years)]
    record = point(series, year)
    st.metric(title, ui['fmt'](record.value, 2 if series.unit.iloc[0].startswith('на ') else 0))
    if record.reason:
        st.info(str(record.reason))
    if not selected.empty:
        ui['wide'](st.plotly_chart, ui['line'](selected, events=False), config={'displayModeBar': False})
    with st.expander('Значения, источники и причины пропусков'):
        columns = [c for c in ['year', 'value', 'absolute_value', 'unit', 'population',
            'denominator_type', 'denominator_method', 'denominator_date', 'quality_flag',
            'numerator_method', 'reason', 'source_file', 'mapping', 'denominator_source'] if c in series]
        ui['wide'](st.dataframe, selected[columns], hide_index=True)
        st.download_button('Скачать выбранный ряд', ui['csv'](selected), 'selected_indicator.csv', 'text/csv', key='selected_series_download')


def overview_rates(data, population, year, policy, calculated, ui):
    st.subheader('Заболеваемость и учёт на 100 тысяч населения')
    for col, table, title in zip(st.columns(2), ['2000', '3000'], ['Зарегистрированные за год', 'Впервые установленный диагноз']):
        absolute = incidence_series(data, table, policy=policy)
        rate = normalize_series(absolute, population, allow_calculated=calculated)
        r = point(rate, year)
        col.metric(title, ui['fmt'](r.value, 1))
        if r.reason:
            col.caption(str(r.reason))
        elif r.denominator_method == 'average_adjacent_january':
            col.caption('Использовано расчётное среднее двух январских срезов.')
    st.caption('Знаменатель — среднегодовая численность. Непроверенный OCR исключён в обычном режиме.')


def incidence_page(data, population, year, policy, years, calculated, ui):
    controls = st.columns(4)
    table = controls[0].selectbox('Показатель', ['2000', '3000'], format_func=lambda x: 'Общая заболеваемость / зарегистрированные' if x == '2000' else 'Первичная заболеваемость', key='inc_table')
    sex = controls[1].selectbox('Пол', list(SEX_LABELS), format_func=SEX_LABELS.get, key='inc_sex')
    age = controls[2].selectbox('Возраст', list(AGE_LABELS), format_func=AGE_LABELS.get, key='inc_age')
    residence = controls[3].selectbox('Место проживания', list(RESIDENCE_LABELS), format_func=RESIDENCE_LABELS.get, key='inc_residence')
    current = data.loc[(data.year == year) & (data.form_number == '10') & (data.table_code == table), ['row_number', 'entity_name', 'icd10']].drop_duplicates()
    if current.empty:
        st.info('В выбранном году нет нужной таблицы формы 10.')
        return
    current['_sort'] = pd.to_numeric(current.row_number, errors='coerce')
    current = current.sort_values('_sort')
    refs = {str(r['row_number']): r for r in current.to_dict('records')}
    row = st.selectbox('Нозология / строка формы', list(refs), format_func=lambda r: f"{r}. {refs[r]['entity_name']}", key='inc_diagnosis')
    mode = st.radio('Единицы', ['На 100 тыс. соответствующей группы', 'Абсолютные значения'], horizontal=True, key='inc_mode')
    absolute = incidence_series(data, table, sex, residence, age, policy, refs[row])
    result = normalize_series(absolute, population, sex=sex, residence=residence, age_group=age,
                              allow_calculated=calculated) if mode.startswith('На ') else absolute
    show_series(result, year, years, ui, refs[row]['entity_name'])
    if sex != 'both' and age != 'all':
        st.caption('Возраст и пол нельзя пересечь: форма 10 не содержит такого числителя.')
    if sex == 'male' or residence == 'urban':
        st.caption('Мужчины = все − женщины; городские жители = все − сельские. Разности считаются только при допустимых компонентах.')
    st.caption('Суммарные строки и «из них» не складываются. Определения сопоставлены по МКБ/названию, основные группы — по смыслу строки.')
    with st.expander('Структура основных диагностических групп'):
        groups = diagnostic_rows(data, year, table)
        labels = {'psychoses': 'Психозы и состояния слабоумия', 'nonpsychotic': 'Непсихотические расстройства', 'intellectual': 'Умственная отсталость'}
        total = cell_value(data, year, '10', table, '1', '4', policy)['value']
        rows = []
        for key, number in groups.items():
            r = cell_value(data, year, '10', table, number, '4', policy)
            rows.append(dict(group=labels[key], patients=r['value'], share_pct=r['value']/total*100 if np.isfinite(total) and total > 0 else np.nan, quality=r['quality_flag']))
        structure = pd.DataFrame(rows, columns=['group', 'patients', 'share_pct', 'quality'])
        ui['wide'](st.dataframe, structure, hide_index=True)
        st.caption('Структура относится ко всей области, всем возрастам и обоим полам. Пропуски не исключаются из итогового знаменателя.')


def observation_page(data, population, series_map, year, policy, years, calculated, ui):
    for col, key in zip(st.columns(4), ['dispensary', 'consultative', 'disp_taken', 'disp_removed']):
        col.metric(CATALOG[key]['label'], ui['fmt'](point(series_map[key], year).value))
    key = st.selectbox('Показатель наблюдения', ['dispensary', 'consultative', 'disp_taken', 'disp_removed', 'consult_started', 'consult_stopped', 'disability_stock', 'disability_new', 'admissions'], format_func=lambda k: CATALOG[k]['label'], key='observation_metric')
    mode = st.radio('Показать', ['Абсолютные значения', 'На 100 тыс. населения'], horizontal=True, key='observation_mode')
    form, table_code, _, _, _, _ = CATALOG[key]['slot']
    candidates = data.loc[(data.year == year) & (data.form_number == form) & (data.table_code == table_code), ['row_number', 'entity_name', 'icd10']].drop_duplicates()
    result = series_map[key]
    if not candidates.empty:
        candidates = candidates.assign(_sort=pd.to_numeric(candidates.row_number, errors='coerce')).sort_values('_sort')
        refs = {str(r['row_number']): r for r in candidates.to_dict('records')}
        chosen = st.selectbox('Нозология наблюдения / помощи', list(refs), format_func=lambda r: f"{r}. {refs[r]['entity_name']}", key='observation_diagnosis_' + key)
        if chosen != '1':
            result = service_diagnosis_series(data, key, refs[chosen], policy)
    if mode.startswith('На '):
        stock = key in {'dispensary', 'consultative', 'disability_stock'}
        kind = 'end_year' if stock else 'annual_average'
        result = normalize_series(result, population, population_type=kind, allow_calculated=calculated)
        st.caption('Знаменатель: ' + TYPE_LABELS[kind] + '. ' + ('Реальная дата среза сохранена в выгрузке.' if stock else 'Расчётное среднее включается отдельно.'))
    show_series(result, year, years, ui, CATALOG[key]['label'])
    st.caption(CATALOG[key]['note'])
    st.subheader('Диспансерная и консультативная группы среди пациентов формы 10')
    table = st.selectbox('Основа структуры', ['2000', '3000'], format_func=lambda t: 'Все зарегистрированные за год' if t == '2000' else 'Впервые выявленные в жизни', key='observation_structure')
    total = cell_value(data, year, '10', table, '1', '4', policy)['value']
    pieces = []
    for label, column in [('Диспансерная группа', '12'), ('Консультативная группа', '13')]:
        r = cell_value(data, year, '10', table, '1', column, policy)
        share = r['value']/total*100 if np.isfinite(total) and total > 0 and np.isfinite(r['value']) and r['value'] <= total else np.nan
        pieces.append(dict(group=label, patients=r['value'], share_pct=share, source_file=r['source_file'], cell=r['mapping'], reason=r['reason']))
    ui['wide'](st.dataframe, pd.DataFrame(pieces), hide_index=True)
    st.caption('Числитель и знаменатель взяты из одной таблицы формы 10. Для таблицы 2000 это продолжение наблюдения на конец года среди зарегистрированных за год; доли не обязаны давать 100%.')


def resource_additions(series_map, population, year, years, calculated, ui):
    st.subheader('Ресурсы относительно населения')
    key = st.selectbox('Ресурс', ['adult_positions', 'day_places'], format_func=lambda k: CATALOG[k]['label'], key='resource_rate')
    adult = key == 'adult_positions'
    rate = normalize_series(series_map[key], population, population_type='end_year' if adult else 'annual_average',
                            age_group='18+' if adult else 'all', scale=10000, allow_calculated=calculated)
    show_series(rate, year, years, ui, CATALOG[key]['label'] + (' на 10 тыс. взрослых' if adult else ' на 10 тыс. жителей'))
    st.caption('Занятые должности — не физические лица врачей. Дневные места — не круглосуточные койки. Для обеспеченности врачами и укомплектованности по брифу нужен ряд формы 30.')


def population_page(population, root, year, ui):
    st.subheader('Население: опубликованное и расчётное')
    kinds = st.columns(3)
    for col, kind in zip(kinds, TYPE_LABELS):
        rows = population.loc[(population.year == year) & (population.population_type == kind) & (population.sex == 'both') & (population.residence == 'all') & (population.age_group == 'all')]
        if not rows.empty:
            r = rows.iloc[0]
            col.metric(TYPE_LABELS[kind], ui['fmt'](r.population, 1 if r.population % 1 else 0))
            col.caption('Расчёт по двум датам' if r.method == 'average_adjacent_january' else ('1 января следующего года' if kind == 'end_year' else 'Опубликовано'))
    ui['wide'](st.dataframe, population.groupby(['population_type', 'year']).agg(rows=('population', 'size'), age_groups=('age_group', 'nunique')).reset_index(), hide_index=True)
    st.info('Возрастные среднегодовые знаменатели 2022–2024 отсутствуют. Не используем население всей области вместо численности детей или подростков.')
    for name, title in [('missing_denominators.csv', 'Недостающие среднегодовые разрезы'), ('municipal_region_comparison.csv', 'Муниципальные суммы и областной ряд'), ('age_mean_method_comparison.csv', 'Сопоставимость опубликованных и расчётных средних')]:
        with st.expander(title):
            frame = pd.read_csv(root / 'data/population' / name)
            ui['wide'](st.dataframe, frame, hide_index=True)
            st.download_button('Скачать таблицу', ui['csv'](frame), name, 'text/csv', key='pop_' + name)
    st.caption('За 2013–2021 годы муниципальные суммы ниже областного пересчёта на 1 470–6 025 человек. До сверки версий не используем их как единую согласованную сумму.')
    st.download_button('Скачать текущие областные знаменатели', ui['csv'](population), 'population_denominators.csv', 'text/csv')


def forecast_page(data, population, policy, years, calculated, ui):
    table = st.selectbox('Что прогнозировать', ['2000', '3000'], format_func=lambda t: 'Зарегистрированные за год' if t == '2000' else 'Впервые установленный диагноз', key='forecast_metric')
    mode = st.radio('Ряд для модели', ['Абсолютные значения', 'На 100 тыс. населения'], horizontal=True, key='forecast_mode')
    source = incidence_series(data, table, policy=policy)
    if mode.startswith('На '):
        source = normalize_series(source, population, allow_calculated=calculated)
    source = source.loc[source.year.between(*years)]
    result = forecast_series(source)
    if result['status'] == 'blocked':
        st.info(result['reason'])
        st.write('Для прогноза нужно дополнить и проверить PDF-ряд. Сейчас строгий режим не соединяет годы через непроверенные значения.')
        return
    st.warning('Исследовательский прогноз по агрегированной отчётности. Однолетняя проверка не подтверждает точность двухлетнего горизонта.')
    st.caption(f"Непрерывный ряд: {result['first_year']}–{int(source.year.max())}; {result['observations']} наблюдений. Метод: {result['model']}.")
    scores = pd.DataFrame([dict(model=k, MAE=v) for k, v in result['mae'].items()])
    ui['wide'](st.dataframe, scores, hide_index=True)
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=source.year, y=source.value, mode='lines+markers', name='Факт', connectgaps=False))
    fc = result['forecast']
    last = source.iloc[-1]
    figure.add_trace(go.Scatter(x=[int(last.year)] + fc.year.tolist(), y=[last.value] + fc.value.tolist(), mode='lines+markers', line={'dash': 'dash'}, name='Прогноз'))
    figure.update_layout(xaxis={'dtick': 1}, yaxis_title=source.unit.iloc[0], height=400)
    ui['wide'](st.plotly_chart, figure)
    ui['wide'](st.dataframe, fc, hide_index=True)
    with st.expander('Проверка на прошлых годах'):
        ui['wide'](st.dataframe, result['backtest'], hide_index=True)
    st.download_button('Скачать прогноз', ui['csv'](fc), 'forecast.csv', 'text/csv')
    st.download_button('Скачать проверку моделей', ui['csv'](result['backtest']), 'forecast_backtest.csv', 'text/csv')


def comparison_page(root, year, ui):
    st.subheader('Готовность территориальных сравнений')
    st.write('Население муниципальных территорий загружено. Для сравнения заболеваемости нужны медицинские числители по тем же территориям и годам. Для карт дополнительно нужны согласованные коды и границы.')
    municipal = pd.read_csv(root / 'data/population/population_municipal_long.csv', keep_default_na=False)
    selected = municipal.loc[(municipal.year == year) & (municipal.population_type == 'annual_average') & (municipal.territory_level == 'municipality')]
    ui['wide'](st.dataframe, selected[['territory_name', 'year', 'population', 'value_status', 'source_file']], hide_index=True)
    st.download_button('Скачать население муниципалитетов', ui['csv'](selected), f'municipal_population_{year}.csv', 'text/csv')
    st.info('Показатели России и федеральных округов не загружены. Областные числа не выдаются за федеральные и не копируются на муниципальные территории.')
    st.subheader('Запросы заказчика: что реализовано и что нужно получить')
    coverage = brief_coverage()
    ui['wide'](st.dataframe, coverage.rename(columns={'request': 'Запрос', 'status': 'Статус', 'limitation': 'Что учитывать / получить'}), hide_index=True)
    st.download_button('Скачать реестр запросов', ui['csv'](coverage), 'brief_coverage.csv', 'text/csv')

