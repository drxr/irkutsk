from pathlib import Path
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def test_accessible_drilldown_back_reset_and_sources():
    at = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=60).run()
    assert not at.exception
    selector = next(widget for widget in at.selectbox if widget.label == 'Выбрать группу без клика по графику')
    selector.set_value('0-14').run()
    next(button for button in at.button if button.label == 'Открыть детали группы').click().run()
    assert not at.exception
    assert at.session_state['story_slice']['age_group'] == '0-14'
    selector = next(widget for widget in at.selectbox if widget.label == 'Выбрать группу без клика по графику')
    selector.set_value('rural').run()
    next(button for button in at.button if button.label == 'Открыть детали группы').click().run()
    assert at.session_state['story_slice']['residence'] == 'rural'
    selector = next(widget for widget in at.selectbox if widget.label == 'Выбрать группу без клика по графику')
    selector.set_value('psychoses').run()
    next(button for button in at.button if button.label == 'Открыть детали группы').click().run()
    assert not at.exception
    assert at.session_state['story_slice']['axis'] == 'source'
    assert any('графа 16' in str(table.value) for table in at.dataframe)
    at.button(key='story_back').click().run()
    assert at.session_state['story_slice']['axis'] == 'diagnosis'
    at.button(key='story_reset').click().run()
    assert at.session_state['story_slice']['age_group'] == 'all'
    assert at.session_state['story_slice']['residence'] == 'all'


def test_year_selection_callback_updates_global_year():
    from story_view import on_chart_selection
    # Callback проверяется в реальном Streamlit-контексте с форматом PlotlyState.
    at = AppTest.from_string('''
import streamlit as st
from story_view import on_chart_selection
st.session_state['year'] = 2024
st.session_state['chart_test'] = {'selection': {'points': [{'customdata': ['2017', 'visually_checked', '']}]}}
on_chart_selection('chart_test', 'year', [2017, 2024])
st.write(st.session_state['year'])
''', default_timeout=60).run()
    assert not at.exception
    assert at.session_state['year'] == 2017
