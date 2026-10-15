"""Реальный запуск Streamlit и взаимодействия с важными фильтрами."""
from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
SECTIONS = ['Обзор', 'Заболеваемость и структура', 'Наблюдение и помощь',
            'Нагрузка и ресурсы', 'Динамика и события', 'Прогноз',
            'Население и покрытия', 'Сравнения и запросы', 'Аномалии и качество', 'Данные и методика']


@pytest.mark.parametrize('section', SECTIONS)
def test_each_screen(section):
    at = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=40).run()
    assert not any('Население не подключено' in i.value for i in at.info)
    assert any('Население подключено' in c.value for c in at.caption)
    assert not at.exception
    at.radio(key='section').set_value(section).run()
    assert not at.exception, [e.message for e in at.exception]


def test_calculated_population_and_missing_child_denominator():
    from data_access import population_input
    if population_input(ROOT) is None:
        pytest.skip('Население не подключено')
    at = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=40).run()
    at.checkbox(key='allow_calculated_population').check().run()
    assert not at.exception
    assert any('2 773' in m.value or '2773' in m.value or '2\u202f773' in m.value for m in at.metric)
    at.radio(key='section').set_value('Заболеваемость и структура').run()
    at.selectbox(key='inc_age').set_value('0-14').run()
    assert not at.exception
    assert any('Нет населения' in i.value for i in at.info)
    at.radio(key='inc_mode').set_value('Абсолютные значения').run()
    assert not at.exception
    assert any(''.join(m.value.split()) == '12839' for m in at.metric)


def test_stock_rate_and_research_forecast_mode():
    at = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=40).run()
    at.radio(key='section').set_value('Наблюдение и помощь').run()
    at.radio(key='observation_mode').set_value('На 100 тыс. населения').run()
    assert not at.exception
    at.radio(key='section').set_value('Прогноз').run()
    assert not at.exception
    assert any('Непрерывный хвост' in i.value for i in at.info)
    at.selectbox(key='quality_policy').set_value('Добавить OCR без ручной сверки').run()
    assert not at.exception
