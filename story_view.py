"""Главный экран: событие → группы → место проживания → нозология → источник."""
from functools import partial
from hashlib import sha256
from html import escape

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from analytics import CATALOG, point, quality_checks
from design_tokens import (AGE, DIAGNOSIS, REGION, RESIDENCE, SEX, STATUS,
                           chart_layout)
from indicators import AGE_LABELS, RESIDENCE_LABELS, SEX_LABELS, normalize_series
from narrative import (AXES, DIAGNOSIS_LABELS, breakdown, compare_years,
                       drill_selection, initial_slice, insight, number,
                       previous_slice, selected_code, slice_series)

SLICE_KEY = 'story_slice'


def render_insight(item):
    """Текстовый вывод рядом с числами, а не только подсказка при наведении."""
    color = STATUS[item.get('tone', 'neutral')]
    st.markdown(f'<div class="insight" style="border-left-color:{color}"><strong>{escape(item["title"])}</strong>'
                f'<p>{escape(item["detail"])}</p></div>', unsafe_allow_html=True)


def change_axis(axis):
    st.session_state[SLICE_KEY] = initial_slice(axis)
    st.session_state['story_revision'] = st.session_state.get('story_revision', 0) + 1


def go_back():
    st.session_state[SLICE_KEY] = previous_slice(st.session_state[SLICE_KEY])
    st.session_state['story_revision'] = st.session_state.get('story_revision', 0) + 1


def switch_section(section):
    st.session_state['section'] = section


def set_absolute():
    st.session_state['story_measure'] = 'Абсолютные значения'


def on_chart_selection(key, axis, allowed):
    """Callback исполняется до построения виджетов нового прогона."""
    event = st.session_state.get(key, {})
    if axis == 'year':
        code = selected_code(event, allowed)
        if code is not None:
            st.session_state['year'] = int(code)
    else:
        state = st.session_state.get(SLICE_KEY, initial_slice())
        st.session_state[SLICE_KEY] = drill_selection(state, event, axis, allowed)
        if st.session_state[SLICE_KEY] != state:
            st.session_state['story_revision'] = st.session_state.get('story_revision', 0) + 1


def context_key(prefix, *values):
    """Новый контекст получает новый выбор Plotly, старый клик не повторяется."""
    return prefix + '_' + sha256(repr(values).encode()).hexdigest()[:12]


def profile_chart(frame, axis):
    """Канонический код в customdata; подписи и порядок не управляют фильтром."""
    palettes = {'age': AGE, 'sex': SEX, 'residence': RESIDENCE, 'diagnosis': DIAGNOSIS}
    shown = frame.loc[frame.value.notna()].sort_values('value', ascending=True)
    figure = go.Figure(go.Bar(x=shown.value.tolist(), y=shown.label.tolist(), orientation='h',
        customdata=[[row.code, row.absolute_value, row.share_pct] for row in shown.itertuples()],
        marker_color=[palettes[axis][code] for code in shown.code],
        text=[number(value, 1 if str(shown.unit.iloc[0]).startswith('на ') else 0) for value in shown.value] if not shown.empty else [],
        textposition='auto', selected=dict(marker=dict(opacity=1)), unselected=dict(marker=dict(opacity=.65)),
        hovertemplate='%{y}<br>%{x:,.1f}<br>Доля полного итога: %{customdata[2]:.1f}%<extra></extra>'))
    chart_layout(figure, 360 if axis == 'age' else 280)
    figure.update_layout(clickmode='event+select', dragmode=False,
                         xaxis_title=frame.unit.iloc[0] if not frame.empty else '',
                         yaxis=dict(showgrid=False, automargin=True))
    return figure, shown.code.tolist()


def breadcrumbs(state, year):
    labels = [f'Иркутская область · {year}']
    if state['age_group'] != 'all':
        labels.append(AGE_LABELS[state['age_group']] + ' · оба пола')
    elif state['sex'] != 'both':
        labels.append(SEX_LABELS[state['sex']] + ' · все возрасты')
    if state['residence'] != 'all':
        labels.append(RESIDENCE_LABELS[state['residence']])
    if state['diagnosis'] != 'all':
        labels.append(DIAGNOSIS_LABELS[state['diagnosis']])
    return ' → '.join(labels)


def profile_insight(frame, year, rates):
    """Выделить максимум и изменение только среди реально доступных групп."""
    available = frame.loc[frame.value.notna()]
    if available.empty:
        return dict(title='Сравнение групп пока недоступно',
                    detail='Нет допустимых числителей или знаменателей выбранных разрезов.', tone='neutral')
    largest = available.sort_values('value', ascending=False).iloc[0]
    prefix = 'Наибольший показатель на 100 тыс.' if rates else 'Больше всего зарегистрировано в группе'
    detail = f"{largest.label}: {number(largest.value, 1 if rates else 0)} {largest.unit}."
    comparable = available.loc[available.difference.notna()]
    if not comparable.empty:
        mover = comparable.loc[comparable.difference.abs().idxmax()]
        sign = '+' if mover.difference > 0 else ''
        detail += f" Наибольшее абсолютное изменение к {year - 1} году среди {len(comparable)} сопоставимых групп: {mover.label}, {sign}{number(mover.difference, 1 if rates else 0)}."
    if len(available) < len(frame):
        detail += f' Доступно {len(available)} из {len(frame)} групп; сравнение неполное.'
    detail += ' Это описание учёта, причины нужно проверить.'
    return dict(title=f'{prefix}: {largest.label}', detail=detail, tone='neutral')


def source_and_action(series, state, year, ui):
    st.subheader('3. Проверить число и сформулировать следующий шаг')
    current = series.loc[series.year.eq(year)]
    columns = [c for c in ['year', 'value', 'unit', 'absolute_value', 'population', 'quality_flag',
                           'numerator_method', 'source_file', 'source_page', 'mapping',
                           'denominator_method', 'denominator_date', 'denominator_source', 'reason'] if c in current]
    ui['wide'](st.dataframe, current[columns], hide_index=True)
    question = 'Сверить изменения первичного выявления с доступностью консультаций и полнотой регистрации.'
    if state['age_group'] in {'0-14', '15-17'}:
        question = 'Сверить выявление в детско-подростковой группе, направления на консультацию и доступность профильного приёма.'
    elif state['residence'] == 'rural':
        question = 'Проверить доступность консультаций для сельских жителей, маршруты направлений и полноту учёта.'
    st.write('**Что обсудить на рабочей встрече:** ' + question)
    st.caption('Данные ставят вопрос для проверки. Причина изменения и решение о ресурсах не устанавливаются одним показателем.')
    st.download_button('Скачать выбранный срез с источниками', ui['csv'](series),
                       'decision_slice.csv', 'text/csv', key='story_export')


def narrative_overview(data, population, series_map, alerts, year, policy, years, calculated, threshold, ui):
    """Единый маршрут анализа с крупными KPI и реальным выбором на графике."""
    registered, new = point(series_map['registered'], year), point(series_map['new'], year)
    headline = f"За {year} год зарегистрировано {number(registered.value)} пациентов"
    lead = insight(series_map['new'], year)
    st.markdown(f'<div class="hero"><div class="eyebrow" style="color:#82CFD0">1. ОБЩАЯ КАРТИНА</div>'
        f'<div class="hero-title">{escape(headline)}</div>'
        f'<div class="hero-text">Первый диагноз в жизни: {number(new.value)}. {escape(lead["title"])}.</div></div>', unsafe_allow_html=True)
    with st.container(key='primary_kpis'):
        for col, key in zip(st.columns(2), ['registered', 'new']):
            change = compare_years(series_map[key], year)
            delta = f"{number(change['change_pct'], 1)}% к {year - 1}" if np.isfinite(change['change_pct']) else None
            col.metric(CATALOG[key]['label'], number(change['current']), delta, delta_color='off')
    if not population.empty:
        columns = st.columns(2)
        for col, key in zip(columns, ['registered', 'new']):
            rate = normalize_series(series_map[key], population, allow_calculated=calculated)
            current = point(rate, year)
            if np.isfinite(current.value):
                col.metric(CATALOG[key]['label'] + ' · на 100 тыс.', number(current.value, 1))
            else:
                col.caption(CATALOG[key]['label'] + ': ' + str(current.reason))
    render_insight(lead)
    st.caption('Рост и снижение учёта показывают изменение, а не автоматическую оценку качества помощи.')
    left, right = st.columns([1.7, 1], gap='large')
    with left:
        st.subheader('Динамика: выбери год прямо на графике')
        table = st.selectbox('Что разбираем', ['3000', '2000'],
            format_func=lambda t: 'Впервые в жизни установленный диагноз' if t == '3000' else 'Все зарегистрированные за год', key='story_metric')
        metric_id = 'new' if table == '3000' else 'registered'
        history = series_map[metric_id].loc[series_map[metric_id].year.between(*years)].copy()
        fig = ui['line'](history, events=False, height=330)
        fig.data[0].customdata = [[str(row.year), row.quality_flag, row.reason] for row in history.itertuples()]
        fig.update_layout(clickmode='event+select', dragmode=False)
        chart_key = context_key('story_year', table, policy, years)
        ui['wide'](st.plotly_chart, fig, theme=None, key=chart_key,
                   on_select=partial(on_chart_selection, chart_key, 'year', history.year.tolist()),
                   selection_mode='points', config={'displayModeBar': False})
        st.caption('Клик по точке меняет отчётный год и все связанные детали. Пропуски остаются разрывами.')
    with right:
        st.subheader('Что требует внимания')
        current_alerts = alerts.loc[alerts.year.eq(year)]
        if current_alerts.empty:
            st.info('Пороговых сигналов нет. Это не оценка качества помощи.')
        for item in current_alerts.head(2).itertuples():
            render_insight(dict(title=f'{item.indicator}: {number(item.change_pct, 1)}%',
                                detail=item.action, tone='attention'))
        checks = quality_checks(data, policy)
        checks = checks.loc[checks.year.eq(year)]
        if checks.empty:
            render_insight(dict(title='Выполняемые проверки не выявили расхождений',
                detail='Это результат имеющихся проверок, а не подтверждение ручной сверки всех PDF.', tone='positive'))
        else:
            render_insight(dict(title=f'Есть замечания к данным: {len(checks)}',
                detail='Проверь источники перед интерпретацией изменений.', tone='negative'))
        st.button('Перейти к качеству данных', on_click=switch_section, args=('Аномалии и качество',), key='story_quality')
    st.subheader('2. Где сосредоточено изменение?')
    context = (int(year), table)
    if st.session_state.get('story_context') != context:
        st.session_state[SLICE_KEY] = initial_slice()
        st.session_state['story_context'] = context
    state = st.session_state.setdefault(SLICE_KEY, initial_slice())
    st.radio('Единицы разбора', ['Абсолютные значения', 'На 100 тыс. соответствующей группы'],
             horizontal=True, key='story_measure')
    rates = st.session_state['story_measure'].startswith('На ')
    route_buttons = st.columns(3)
    for column, axis, text in zip(route_buttons, ['age', 'sex', 'residence'],
                                  ['Разобрать по возрасту', 'Разобрать по полу', 'Разобрать по месту проживания']):
        column.button(text, on_click=change_axis, args=(axis,), key='story_start_' + axis)
    st.caption(breadcrumbs(state, year))
    navigation = st.columns([1, 1, 3])
    navigation[0].button('← Назад', on_click=go_back, disabled=not state['history'], key='story_back')
    navigation[1].button('Сбросить путь', on_click=change_axis, args=('age',), key='story_reset')
    chosen = slice_series(data, population, year, table, state, policy, rates, calculated)
    if state['axis'] != 'source':
        axis = state['axis']
        st.markdown('**' + AXES[axis] + '**')
        frame = breakdown(data, population, year, table, state, axis, policy, rates, calculated)
        render_insight(profile_insight(frame, year, rates))
        figure, allowed = profile_chart(frame, axis)
        if allowed:
            key = context_key('story_groups', year, table, policy, rates, calculated,
                              st.session_state.get('story_revision', 0),
                              tuple(state[field] for field in ['sex', 'age_group', 'residence', 'diagnosis', 'axis']))
            ui['wide'](st.plotly_chart, figure, theme=None, key=key,
                       on_select=partial(on_chart_selection, key, axis, allowed),
                       selection_mode='points', config={'displayModeBar': False})
            st.caption('Нажми на столбец, чтобы перейти к следующему уровню. Цвета категорий постоянны во всех разборах.')
            # Равноправный способ выбора для клавиатуры и небольших экранов.
            select_key = key + '_accessible'
            selection = st.selectbox('Выбрать группу без клика по графику', [''] + allowed,
                format_func=lambda code: 'Выбери группу' if not code else frame.set_index('code').loc[code, 'label'], key=select_key)
            if selection:
                st.button('Открыть детали группы', key=key + '_open',
                    on_click=lambda code=selection, ax=axis, codes=allowed: _choose_group(code, ax, codes))
        else:
            st.info('; '.join(dict.fromkeys(str(reason) for reason in frame.reason if reason)) or 'Нет допустимых значений.')
            if rates:
                st.button('Посмотреть абсолютные значения', on_click=set_absolute, key='story_switch_absolute')
        with st.expander('Весь разрез, включая отсутствующие значения'):
            ui['wide'](st.dataframe, frame, hide_index=True)
        st.caption('Возраст и пол — отдельные маршруты: совместный разрез отсутствует в медицинской форме. Группы 55+ в форме также нет.')
    else:
        render_insight(insight(chosen, year))
        ui['wide'](st.plotly_chart, ui['line'](chosen.loc[chosen.year.between(*years)], events=False), theme=None)
    with st.expander('Выбранный срез и его источник', expanded=state['axis'] == 'source'):
        source_and_action(chosen, state, year, ui)
    with st.expander('Ресурсы и нагрузка: следующий управленческий вопрос'):
        for col, key in zip(st.columns(3), ['admissions', 'adult_positions', 'visits_per_position']):
            col.metric(CATALOG[key]['label'], number(point(series_map[key], year).value, 1 if key != 'admissions' else 0))
        st.button('Открыть нагрузку и ресурсы', on_click=switch_section, args=('Нагрузка и ресурсы',), key='story_resources')
    st.caption('Легенда сигналов: зелёный — пройдены доступные проверки; красный — есть замечания к данным; янтарный — изменение требует разбора; серый — нет достаточных данных.')


def _choose_group(code, axis, allowed):
    event = {'selection': {'points': [{'customdata': [code]}]}}
    st.session_state[SLICE_KEY] = drill_selection(st.session_state[SLICE_KEY], event, axis, allowed)
    st.session_state['story_revision'] = st.session_state.get('story_revision', 0) + 1
