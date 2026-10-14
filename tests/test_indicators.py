"""Проверки ошибок, меняющих показатель или управленческий вывод."""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from analytics import DEFAULT_POLICY, POLICIES, build_series, load_frames, point
from indicators import (forecast_series, form10_count, incidence_series,
                        load_population, normalize_series)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def data():
    return load_frames([ROOT / 'data/medical/psychiatry_all_long.csv.gz'])


@pytest.fixture(scope='module')
def population():
    return load_population(ROOT / 'data/population/population_denominators_2013_2024.csv')


def test_all_reports_and_confirmed_2017(data):
    assert len(data) == 15279 and data.source_file.nunique() == 24
    assert point(incidence_series(data), 2017).value == 80546
    assert len(data[(data.year == 2022) & (data.form_number == '36')]) == 484


def test_known_2024_rates(data, population):
    absolute = incidence_series(data)
    strict = normalize_series(absolute, population)
    assert np.isnan(point(strict, 2024).value)
    enabled = normalize_series(absolute, population, allow_calculated=True)
    assert point(enabled, 2024).population == 2326414.5
    assert point(enabled, 2024).value == pytest.approx(64515 / 2326414.5 * 100000)
    assert point(enabled, 2024).denominator_method == 'average_adjacent_january'
    primary = normalize_series(incidence_series(data, '3000'), population, allow_calculated=True)
    assert point(primary, 2024).value == pytest.approx(7007 / 2326414.5 * 100000)


def test_sex_and_residence_are_differences(data):
    both = form10_count(data, 2024)
    female = form10_count(data, 2024, sex='female')
    male = form10_count(data, 2024, sex='male')
    assert both['value'] == female['value'] + male['value']
    rural = form10_count(data, 2024, residence='rural')
    urban = form10_count(data, 2024, residence='urban')
    assert urban['value'] + rural['value'] == both['value']
    assert male['numerator_method'] == 'total_minus_subset'


def test_age_missing_never_uses_total_population(data, population):
    child = incidence_series(data, age_group='0-14')
    assert point(child, 2024).value == 12839
    rate = normalize_series(child, population, age_group='0-14', allow_calculated=True)
    assert np.isnan(point(rate, 2024).value)
    assert point(rate, 2021).value > 0
    assert np.isnan(form10_count(data, 2021, sex='female', age_group='0-14')['value'])


def test_age_partition_and_adults(data):
    for year in [2013, 2021, 2024]:
        groups = ['0-14', '15-17', '18-19', '20-39', '40-59', '60+']
        assert sum(form10_count(data, year, age_group=g)['value'] for g in groups) == form10_count(data, year)['value']
        assert form10_count(data, year, age_group='18+')['value'] == sum(form10_count(data, year, age_group=g)['value'] for g in groups[2:])


def test_stock_uses_next_january(data, population):
    source = build_series(data)['dispensary']
    result = normalize_series(source, population, population_type='end_year')
    selected = population.query("year == 2024 and population_type == 'end_year' and sex == 'both' and residence == 'all' and age_group == 'all'").iloc[0]
    assert point(result, 2024).value == pytest.approx(28374 / selected.population * 100000)
    assert point(result, 2024).denominator_date == '2025-01-01'


def test_movements_shift_columns_in_2023(data):
    series = build_series(data)
    assert point(series['disp_taken'], 2024).value == 2167
    assert point(series['disp_removed'], 2024).value == 3656
    assert point(series['consult_started'], 2024).value == 8051
    assert point(series['consult_stopped'], 2024).value == 8649
    assert point(series['disp_removed'], 2022).value > 0


def test_wrong_population_duplicate_and_geography_rejected(population):
    for changed in [pd.concat([population, population.iloc[:1]]), population.assign(territory_name='г. Иркутск'), population.assign(population=0)]:
        with pytest.raises(ValueError):
            load_population(changed.to_csv(index=False).encode())


def test_ocr_and_negative_difference_remain_gaps(data):
    assert np.isnan(point(incidence_series(data, '3000'), 2019).value)
    assert point(incidence_series(data, '3000', policy=list(POLICIES)[1]), 2019).value == 7255
    changed = data.copy()
    mask = (changed.year == 2024) & (changed.form_number == '10') & (changed.table_code == '2000') & (changed.row_number == '1') & (changed.column_number == '5')
    changed.loc[mask, 'value_num'] = 100000
    assert np.isnan(form10_count(changed, 2024, sex='male')['value'])


def test_real_forecast_blocked_and_synthetic_validated(data):
    assert forecast_series(incidence_series(data))['status'] == 'blocked'
    synthetic = pd.DataFrame({'year': range(2013, 2025), 'value': range(100, 220, 10)})
    result = forecast_series(synthetic)
    assert result['model'] == 'linear_trend'
    assert result['mae']['linear_trend'] < result['mae']['last_value']
    assert result['forecast'].year.tolist() == [2025, 2026]
    assert result['forecast'].value.tolist() == pytest.approx([220, 230])
    assert (result['backtest'].train_last_year < result['backtest'].test_year).all()
    synthetic.loc[11, 'value'] = np.nan
    assert forecast_series(synthetic)['status'] == 'blocked'
    synthetic.loc[11, 'value'] = 210
    synthetic.loc[6, 'value'] = np.nan
    assert forecast_series(synthetic)['status'] == 'blocked'


def test_diagnoses_match_changed_rows(data):
    ref = data.loc[(data.year == 2024) & (data.form_number == '10') & (data.table_code == '2000') & (data.row_number == '15')].iloc[0].to_dict()
    series = incidence_series(data, reference=ref)
    expected = data.loc[(data.year == 2013) & (data.form_number == '10') & (data.table_code == '2000') & (data.row_number == '14') & (data.column_number == '4')].iloc[0].value_num
    assert point(series, 2013).value == expected
