"""Нарратив, сопоставимость групп и настоящий путь выбора графика."""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from analytics import DEFAULT_POLICY, load_frames
from data_access import empty_population, medical_inputs
from narrative import (breakdown, compare_years, drill_selection, initial_slice,
                       insight, previous_slice, selected_code, signal_tone, slice_series)
from story_view import profile_chart
from design_tokens import AGE

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def medical():
    return load_frames(medical_inputs(ROOT))


def event(code):
    return {'selection': {'points': [{'customdata': [code]}]}}


def test_drill_route_and_exact_backtracking():
    start = initial_slice()
    child = drill_selection(start, event('0-14'), 'age', ['0-14'])
    assert child['axis'] == 'residence' and child['age_group'] == '0-14'
    rural = drill_selection(child, event('rural'), 'residence', ['urban', 'rural'])
    diagnosis = drill_selection(rural, event('psychoses'), 'diagnosis', ['psychoses'])
    assert diagnosis['axis'] == 'source' and diagnosis['residence'] == 'rural'
    assert previous_slice(diagnosis) == rural
    assert previous_slice(rural) == child
    assert previous_slice(child) == start
    assert start == initial_slice()  # функция не мутирует исходное состояние


def test_no_invented_age_and_sex_intersection():
    male = drill_selection(initial_slice('sex'), event('male'), 'sex', ['male'])
    child = drill_selection(male, event('0-14'), 'age', ['0-14'])
    assert child['sex'] == 'both'
    assert drill_selection(child, event('female'), 'sex', ['female'])['age_group'] == 'all'
    assert drill_selection(child, event('55+'), 'age', ['55+']) == child


@pytest.mark.parametrize('payload', [{}, {'selection': None}, {'selection': {'points': [None]}},
    {'selection': {'points': []}}, {'selection': {'points': [{'customdata': ['0-14']}, {'customdata': ['15-17']}]}},
    {'selection': {'points': [{'point_index': 0}]}}, {'selection': {'points': [{'customdata': ['foreign']}]}}])
def test_stale_or_ambiguous_click_is_ignored(payload):
    assert selected_code(payload, ['0-14']) is None
    assert drill_selection(initial_slice(), payload, 'age', ['0-14']) == initial_slice()


def test_missing_year_and_zero_base_have_no_fictitious_percent():
    frame = pd.DataFrame({'year': [2022, 2023, 2024], 'value': [100, np.nan, 90],
                          'reason': ['', '', ''], 'quality_flag': ['digital'] * 3, 'unit': ['чел.'] * 3})
    assert not compare_years(frame, 2024)['comparable']
    assert '2023' in insight(frame, 2024)['detail']
    frame.loc[1, 'value'] = 0
    change = compare_years(frame, 2024)
    assert change['difference'] == 90 and np.isnan(change['change_pct'])
    assert 'нулевой' in insight(frame, 2024)['detail']
    assert signal_tone(200, 100) == signal_tone(50, 100) == 'attention'
    assert signal_tone(100, 100, invalid=True) == 'negative'


def test_age_profile_matches_form_and_preserves_full_denominator(medical):
    frame = breakdown(medical, empty_population(), 2024, '3000', initial_slice(), 'age')
    assert len(frame) == 6
    assert frame.value.sum() == 7007
    assert frame.loc[frame.code.eq('0-14'), 'value'].iloc[0] == 3243
    assert frame.share_pct.sum() == pytest.approx(100)
    figure, codes = profile_chart(frame, 'age')
    assert set(codes) == set(frame.code)
    assert list(figure.data[0].marker.color) == [AGE[code] for code in codes]
    assert [p[0] for p in figure.data[0].customdata] == codes


def test_missing_population_keeps_absolute_slice_and_trace(medical):
    state = initial_slice()
    state.update(age_group='0-14', residence='rural')
    absolute = slice_series(medical, empty_population(), 2024, '3000', state)
    current = absolute.loc[absolute.year.eq(2024)].iloc[0]
    assert current.value == 426
    assert 'графа 16' in current.mapping
    rate = slice_series(medical, empty_population(), 2024, '3000', state, rates=True)
    current = rate.loc[rate.year.eq(2024)].iloc[0]
    assert np.isnan(current.value) and current.absolute_value == 426
    assert 'Нет населения' in current.reason
