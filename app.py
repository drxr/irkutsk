"""Streamlit: региональная поддержка управленческих решений по формам 10/36."""

from base64 import b64encode
from html import escape
from inspect import signature
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analytics import (
    CATALOG, DEFAULT_POLICY, METRICS, POLICIES, build_series, capacity_scenario,
    cell_value, detect_anomalies, diagnostic_rows, diagnosis_history, load_frames, per_100k, period_comparison,
    point, population_frame, quality_checks, years_of,
)


ROOT = Path(__file__).resolve().parent
TEAL, CORAL, NAVY, PURPLE = "#087F83", "#C84A40", "#152D3D", "#6750A4"
SECTIONS = ["Обзор", "Динамика и события", "Структура пациентов",
            "Нагрузка и ресурсы", "Аномалии и качество", "Данные и методика"]

st.set_page_config(page_title="Волна данных · Психиатрия", page_icon="◉", layout="wide")


def full_width(element, *args, **kwargs):
    """Выбрать способ заполнения контейнера по API установленного Streamlit.

    Старые версии принимают ширину только в пикселях и используют
    use_container_width=True. Новые версии поддерживают width="stretch".
    Проверяем сигнатуру каждого элемента: таблицы и графики менялись
    в разных выпусках. Строку в старый числовой параметр не передаем.
    """
    width = signature(element).parameters.get("width")
    if width is not None and isinstance(width.default, str):
        kwargs["width"] = "stretch"
    else:
        kwargs["use_container_width"] = True
    return element(*args, **kwargs)


def style():
    """Воздушный интерфейс: крупная иерархия, контрастные смысловые акценты."""
    st.markdown("""<style>
    .stApp { background:#F6F8FA; color:#152D3D; }
    .block-container {max-width:1500px;padding:2rem 2.8rem 3rem;}
    [data-testid="stSidebar"] {background:#FFFFFF;border-right:1px solid #E2E8EC;}
    [data-testid="stSidebar"] .block-container{padding-top:2rem;}
    h1,h2,h3 {color:#152D3D;letter-spacing:-.025em;}
    h1{font-size:2.5rem !important;font-weight:700 !important;}
    h2{font-size:1.65rem !important;} h3{font-size:1.2rem !important;}
    [data-testid="stMetric"]{background:white;border:1px solid #E2E8EC;border-radius:18px;padding:20px;}
    [data-testid="stMetricValue"]{font-size:2rem;}
    [data-testid="stMetricLabel"] p {white-space:normal!important;overflow:visible!important;text-overflow:clip!important;}
    [data-testid="stVerticalBlockBorderWrapper"]{border-radius:18px;}
    .brand{margin-bottom:24px;}
    .brand img{display:block;width:190px;max-width:100%;height:auto;}
    .brand small{display:block;font-size:11px;letter-spacing:1.5px;color:#536977;margin:12px 0 0;font-weight:600;}
    .sidebar-signature{margin-top:18px;color:#536977;font-size:13px;line-height:1.7;}
    .sidebar-signature strong{font-weight:600;color:#152D3D;}
    .eyebrow{text-transform:uppercase;letter-spacing:2px;font-size:11px;font-weight:750;color:#536977;margin-bottom:8px;}
    .hero{background:#152D3D;border-radius:22px;color:white;padding:28px 32px;margin:8px 0 24px;}
    .hero-title{font-size:24px;font-weight:650;line-height:1.35;max-width:950px;}
    .hero-text{color:#C7D9E2;font-size:14px;line-height:1.65;margin-top:12px;max-width:990px;}
    .chip{display:inline-block;background:#E3F3F2;color:#075E62;border-radius:30px;padding:5px 11px;font-size:12px;font-weight:600;margin:0 8px 8px 0;}
    .signal{padding:18px 20px;background:white;border:1px solid #E2E8EC;border-left:4px solid #C84A40;border-radius:14px;margin-bottom:12px;}
    .signal strong{font-size:15px;display:block;color:#152D3D;}
    .signal p{font-size:13px;line-height:1.55;color:#536977;margin:8px 0 0;}
    .signal .delta{color:#A1322C;font-weight:700;font-size:22px;margin-top:6px;}
    .timeline{padding:20px;border:1px solid #E2E8EC;background:#FFFFFF;border-radius:16px;min-height:178px;}
    .timeline b{display:block;font-size:19px;margin:6px 0;}
    .timeline p{font-size:13px;line-height:1.6;color:#536977;}
    .footnote{font-size:12px;line-height:1.6;color:#536977;}
    .stButton button,.stDownloadButton button{border-radius:12px;}
    @media(max-width:800px){.block-container{padding:1rem;}h1{font-size:1.9rem!important;}.hero{padding:22px;}}
    </style>""", unsafe_allow_html=True)


def fmt(value, digits=0):
    """Русский формат числа; прочерк означает отсутствие допустимого значения."""
    if not np.isfinite(value):
        return "—"
    return f"{value:,.{digits}f}".replace(",", " ").replace(".", ",")


def csv_bytes(frame):
    return frame.to_csv(index=False).encode("utf-8-sig")


@st.cache_data
def prepared_data(contents):
    return load_frames(contents)


@st.cache_data
def prepared_series(data, policy):
    return build_series(data, policy)


@st.cache_data
def logo_data_uri():
    """Встроить векторный логотип из комплекта без сетевого запроса."""
    encoded = b64encode((ROOT / "assets" / "volna-logo.svg").read_bytes()).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"


def line_chart(series, color=TEAL, height=330, events=True):
    """Годовые точки на конец отчетного года; пропуски разрывают линию."""
    plot = go.Figure()
    dates = [f"{year}-12-31" for year in series.year]
    hover = np.column_stack([series.year, series.quality_flag, series.reason])
    plot.add_trace(go.Scatter(
        x=dates, y=series.value, mode="lines+markers", connectgaps=False,
        line=dict(color=color, width=3), marker=dict(size=8, color=color),
        customdata=hover,
        hovertemplate="%{customdata[0]} год<br>%{y:,.2f}<br>Качество: %{customdata[1]}<br>%{customdata[2]}<extra></extra>",
        name=series.label.iloc[0],
    ))
    ocr = series.quality_flag.str.contains("ocr") & series.value.notna()
    if ocr.any():
        plot.add_trace(go.Scatter(x=[dates[i] for i in range(len(dates)) if ocr.iloc[i]],
                                  y=series.loc[ocr, "value"], mode="markers",
                                  marker=dict(symbol="diamond-open", size=13, color=CORAL, line=dict(width=2)),
                                  name="OCR: требует сверки", hoverinfo="skip"))
    if events:
        minimum, maximum = int(series.year.min()), int(series.year.max())
        if minimum <= 2021 and maximum >= 2020:
            plot.add_vrect(x0="2020-01-01", x1="2021-12-31", fillcolor="#F2C17B", opacity=.13, line_width=0, layer="below")
            plot.add_annotation(x="2020-11-01", y=1.09, xref="x", yref="paper", text="COVID · контекст 2020–2021", showarrow=False, font=dict(size=11, color="#84531C"))
        if minimum <= 2022 <= maximum:
            plot.add_shape(type="line", x0="2022-02-24", x1="2022-02-24", y0=0, y1=1, xref="x", yref="paper", line=dict(color=PURPLE, width=1.5, dash="dash"))
            plot.add_annotation(x="2022-02-24", y=1.18, xref="x", yref="paper", text="24.02.2022 · начало СВО", showarrow=False, xanchor="left", font=dict(size=11, color=PURPLE))
    plot.update_layout(height=height, margin=dict(t=65, b=35, l=10, r=15),
                       paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                       font=dict(family="Arial, sans-serif", color=NAVY), showlegend=bool(ocr.any()),
                       legend=dict(orientation="h", y=-.2), hovermode="x unified",
                       xaxis=dict(type="date", tickmode="array", tickvals=dates,
                                  ticktext=[str(y) for y in series.year], showgrid=False),
                       yaxis=dict(gridcolor="#E4EAF0", zeroline=False, title=series.unit.iloc[0]),)
    return plot


def bar_chart(labels, values, color=TEAL, height=320):
    plot = go.Figure(go.Bar(x=values, y=labels, orientation="h",
                           marker_color=color, text=[fmt(v) for v in values], textposition="auto",
                           hovertemplate="%{y}<br>%{x:,.0f}<extra></extra>"))
    plot.update_layout(height=height, margin=dict(t=10, b=20, l=10, r=15),
                       paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                       font=dict(family="Arial, sans-serif", color=NAVY),
                       yaxis=dict(autorange="reversed"), xaxis=dict(gridcolor="#E4EAF0", zeroline=False))
    return plot


def metric_card(key, series_map, year, label=None):
    selected = point(series_map[key], year)
    digits = 1 if key in {"length_of_stay", "visits_per_position", "adult_positions", "repeat_share", "employment_share"} else 0
    delta = f"{selected.yoy_pct:+.1f}% к предыдущему году" if np.isfinite(selected.yoy_pct) else None
    displayed = fmt(selected.value, digits)
    if CATALOG[key]["unit"] == "%" and np.isfinite(selected.value):
        displayed += "%"
        previous = point(series_map[key], year - 1).value
        delta = f"{selected.value - previous:+.1f} п.п. к предыдущему году" if np.isfinite(previous) else None
    st.metric(label or CATALOG[key]["label"], displayed, delta=delta,
              delta_color="off", help=CATALOG[key]["note"])
    if not np.isfinite(selected.value):
        st.caption(selected.reason)


def trace_details(key, series_map, year):
    selected = point(series_map[key], year)
    st.caption(CATALOG[key]["note"])
    st.caption(f"{selected.get('mapping', '')} · {selected.get('source_file', '')}")


def decision_list(alerts, year):
    """Сигнал ведет к проверяемому управленческому вопросу, а не диагнозу."""
    current = alerts[alerts.year == year]
    if current.empty:
        st.info("По выбранному правилу нет годовых сигналов. Это не подтверждает отсутствие проблем службы.")
    for _, item in current.head(3).iterrows():
        st.markdown(f'<div class="signal"><strong>{escape(item.indicator)}</strong><div class="delta">{item.change_pct:+.1f}%</div><p>{escape(item.reason)}</p><p><b>Следующий шаг:</b> {escape(item.action)}</p></div>', unsafe_allow_html=True)


def overview(data, series_map, alerts, year, policy, selected_years):
    a = point(series_map["registered"], year)
    b = point(series_map["new"], year)
    message = "Сопоставь динамику учета с доступностью помощи и ресурсами службы"
    if np.isfinite(a.yoy_pct) and np.isfinite(b.yoy_pct):
        if a.yoy_pct < 0 < b.yoy_pct:
            message = "Пациентов в учете меньше, впервые установленных диагнозов больше"
        elif b.yoy_pct < 0 < a.yoy_pct:
            message = "Пациентов в учете больше, впервые установленных диагнозов меньше"
    st.markdown(f'<div class="hero"><div class="eyebrow" style="color:#82CFD0">СНАЧАЛА ВАЖНОЕ · {year}</div><div class="hero-title">{message}</div><div class="hero-text">Годовые отчёты помогают заметить изменения и поставить вопросы службе. Причины проверяются по исходникам, маршрутизации и данным об ожидании помощи.</div></div>', unsafe_allow_html=True)
    columns = st.columns(4, gap="medium")
    for target, key in zip(columns[:3], ["registered", "new", "admissions"]):
        with target:
            metric_card(key, series_map, year, {"new": "Впервые установлен диагноз", "admissions": "Поступления в стационар"}.get(key))
    with columns[3]:
        fatal_a, fatal_b = (point(series_map[k], year).value for k in ["fatal_disp", "fatal_cons"])
        st.metric("Завершённые суицидальные попытки", f"{fmt(fatal_a)} / {fmt(fatal_b)}", help="Диспансерная / консультативная группы отдельно. Таблица 2150, графы 2 и 4. Это не смертность всего региона.")
        st.caption("Диспансерная / консультативная группы")
    st.markdown("")
    left, right = st.columns([1.75, 1], gap="large")
    with left:
        with st.container(border=True):
            st.subheader("Как менялся поток")
            key = st.selectbox("Показатель динамики", ["registered", "new", "admissions", "adult_visits"], format_func=lambda k: CATALOG[k]["label"], key="overview_metric")
            selected = series_map[key].query("@selected_years[0] <= year <= @selected_years[1]")
            full_width(st.plotly_chart, line_chart(selected, height=350), config={"displayModeBar": False})
            trace_details(key, series_map, year)
    with right:
        st.subheader("Что требует внимания")
        decision_list(alerts, year)
        st.caption("↑ и ↓ — изменение, без автоматической оценки «лучше / хуже». Сигналы не доказывают причин.")
    st.subheader("Уточняющие показатели")
    for column, key in zip(st.columns(4), ["adult_positions", "visits_per_position", "length_of_stay", "day_discharge"]):
        with column:
            metric_card(key, series_map, year)
    with st.expander("Числа первого экрана: значения и происхождение"):
        overview_rows = pd.concat([series_map[key].loc[series_map[key].year == year] for key in ["registered", "new", "admissions", "fatal_disp", "fatal_cons"]])
        full_width(st.dataframe, overview_rows[["label", "value", "yoy_pct", "quality_flag", "source_file", "mapping", "reason"]], hide_index=True)
    with st.expander("План разбора на рабочую встречу", expanded=False):
        items = alerts.loc[alerts.year == year, ["indicator", "action"]].head(8).rename(columns={"indicator": "Сигнал", "action": "Что проверить"})
        if items.empty:
            items = pd.DataFrame([{"Сигнал": "Доступность и учет", "Что проверить": "Сверить динамику учёта с очередями и маршрутизацией"}])
        items["Ответственный"] = ""
        items["Срок"] = ""
        items["Статус"] = "К разбору"
        edited = full_width(st.data_editor, items, hide_index=True, key=f"actions_{year}_{policy}", num_rows="dynamic")
        st.download_button("Скачать план разбора", csv_bytes(edited), f"decision_plan_{year}.csv", "text/csv")
        st.caption("Изменения в плане действуют в текущем сеансе; для сохранения скачай CSV.")


def events_page(data, series_map, year, selected_years):
    st.subheader("События — контекст для сравнения")
    texts = [
        ("COVID · самоизоляция", "31 марта 2020, 20:00", "Начало режима в области. Документы подтверждают продление до 31 мая; это не окончание всех ограничений."),
        ("Начало СВО", "24 февраля 2022", "2022 год содержит время до и после даты. Для основной сравнительной таблицы берём полные годы с 2023-го."),
        ("Изменение формы 36", "С 2023 года", "Номера ряда граф изменились. Сравнение выполнено по смыслу метрик; сырой номер графы не является показателем."),
    ]
    for column, (title, date, text) in zip(st.columns(3), texts):
        with column:
            st.markdown(f'<div class="timeline"><div class="eyebrow">{title}</div><b>{date}</b><p>{text}</p></div>', unsafe_allow_html=True)
    st.caption("Жёлтое окно 2020–2021 на графиках — аналитический пандемический контекст, а не единый юридический срок карантина. Источники дат доступны в методике.")
    st.caption("Годовые точки расположены на 31 декабря соответствующего отчётного года; вертикальная линия показывает фактическую дату события.")
    key = st.selectbox("Какой ряд сравнить", list(CATALOG), format_func=lambda k: CATALOG[k]["label"], key="event_metric")
    selected = series_map[key].query("@selected_years[0] <= year <= @selected_years[1]")
    full_width(st.plotly_chart, line_chart(selected, height=400), config={"displayModeBar": False})
    trace_details(key, series_map, year)
    st.subheader("Есть ли изменения в полные годы после 24.02.2022?")
    controls = st.columns([1, 1, 1])
    with controls[0]:
        baseline = st.selectbox("База сравнения", ["2019–2021", "2013–2019", "Только 2021"], key="baseline")
    with controls[1]:
        include_transition = st.checkbox("Включить смешанный 2022 год", value=False)
    with controls[2]:
        st.caption("Без контрольного региона, учета тренда и месячных данных причинный эффект не оценивается.")
    before = {"2019–2021": [2019, 2020, 2021], "2013–2019": list(range(2013, 2020)), "Только 2021": [2021]}[baseline]
    after = list(range(2022 if include_transition else 2023, year + 1))
    comparison = period_comparison(series_map, before, after)
    row = comparison.loc[comparison.metric_id == key].iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Среднее до", fmt(row.before, 1), help=f"Фактически использованы годы: {row.years_before or 'нет'}")
    c2.metric("Среднее после", fmt(row.after, 1), help=f"Фактически использованы годы: {row.years_after or 'нет'}")
    c3.metric("Описательная разница", f"{fmt(row.change_pct, 1)}%" if np.isfinite(row.change_pct) else "—", help="Сравнение средних без причинной интерпретации и без проверки статистической значимости.")
    st.info(f"Использовано {row.n_before} точек до и {row.n_after} после. База 2019–2021 пересекается с COVID; проверь результат также на базе 2013–2019. Для формы 36 отсутствует 2022 год. Участники СВО в данных не выделены.")
    columns = {"indicator": "Показатель", "before": "Среднее до", "after": "Среднее после", "change_pct": "Разница, %", "n_before": "Точек до", "n_after": "Точек после", "years_before": "Годы до", "years_after": "Годы после", "unit": "Единица"}
    full_width(st.dataframe, comparison[list(columns)].rename(columns=columns), hide_index=True)
    st.download_button("Скачать сравнение периодов", csv_bytes(comparison), "event_comparison.csv", "text/csv")
    st.subheader("COVID: изменение 2020 к 2019 году")
    covid = series_map[key].set_index("year")
    if 2019 in covid.index and 2020 in covid.index and np.isfinite(covid.at[2019, "value"]) and np.isfinite(covid.at[2020, "value"]):
        st.write(f"{fmt(covid.at[2019, 'value'], 1)} → {fmt(covid.at[2020, 'value'], 1)}. Годовое изменение: {fmt(covid.at[2020, 'yoy_pct'], 1)}%.")
    else:
        st.write("Нет двух допустимых значений для сравнения. 2019 год получен OCR: при включении этого режима сравнение будет исследовательским.")
    st.caption("Годовой отчет не позволяет выделить эффект конкретных недель самоизоляции. Снижение обращений может отражать ограничения доступа и учета.")


def structure_page(data, year, policy):
    st.subheader("Кого видит служба")
    controls = st.columns(2)
    with controls[0]:
        table = st.selectbox("Контингент", ["2000", "3000"], format_func=lambda c: "Зарегистрированные за год" if c == "2000" else "Впервые установленный диагноз")
    with controls[1]:
        rural = st.selectbox("Место проживания", ["Все жители области", "Сельские жители"])
    offset = 10 if rural == "Сельские жители" else 0
    def value(row, column):
        return cell_value(data, year, "10", table, str(row), str(column), policy)["value"]
    total = value(1, 4 + offset)
    child = [value(1, 6 + offset), value(1, 7 + offset)]
    young_share = sum(child) / total * 100 if total > 0 and all(np.isfinite(v) for v in child) else np.nan
    c1, c2, c3 = st.columns(3)
    c1.metric("Пациенты выбранного контингента", fmt(total))
    c2.metric("Доля детей 0–17 лет", f"{fmt(young_share, 1)}%" if np.isfinite(young_share) else "—")
    c3.metric("Женщины", fmt(value(1, 5 + offset)))
    ages = ["0–14 лет", "15–17 лет", "18–19 лет", "20–39 лет", "40–59 лет", "60 лет и старше"]
    groups = ["Психозы / состояния слабоумия", "Непсихотические расстройства", "Интеллектуальные нарушения"]
    mapped_rows = diagnostic_rows(data, year, table)
    group_values = [value(mapped_rows[key], 4 + offset) if key in mapped_rows else np.nan for key in ["psychoses", "nonpsychotic", "intellectual"]]
    a, b = st.columns(2, gap="large")
    with a:
        with st.container(border=True):
            st.subheader("Возраст")
            full_width(st.plotly_chart, bar_chart(ages, [value(1, col + offset) for col in range(6, 12)]), config={"displayModeBar": False})
    with b:
        with st.container(border=True):
            st.subheader("Три непересекающиеся группы")
            full_width(st.plotly_chart, bar_chart(groups, group_values, color=PURPLE), config={"displayModeBar": False})
    st.caption(f"Итоговые группы найдены по названиям: {mapped_rows}. Номера строк разных редакций могут отличаться. Подстроки «из них» не прибавляются к родительским итогам. Сельская доля не является сравнением рисков городского и сельского населения.")
    st.subheader("Уточнить диагноз / группу")
    rows = data.loc[(data.form_number == "10") & (data.table_code == table) & (data.year == year), ["row_number", "entity_name", "icd10"]].drop_duplicates("row_number")
    if not rows.empty:
        names = dict(zip(rows.row_number, rows.entity_name))
        row = st.selectbox("Строка формы", list(names), format_func=lambda r: f"{r}. {names[r]}")
        history = diagnosis_history(data, table, year, row, str(4 + offset), policy)
        full_width(st.plotly_chart, line_chart(history, color=PURPLE), config={"displayModeBar": False})
        st.caption("Строки разных лет сопоставлены по совпадению МКБ или названия, а не номеру строки. Изменившиеся или неоднозначные определения оставлены без точки.")


def resources_page(series_map, year):
    st.subheader("Нагрузка, доступность, реабилитация")
    for column, key in zip(st.columns(3), ["adult_positions", "adult_visits", "visits_per_position"]):
        with column:
            metric_card(key, series_map, year)
    st.caption("Показатели выше относятся к участковым психиатрам для взрослых. Освидетельствования входят в число посещений. Норматив нагрузки и кадровый дефицит из этих данных не вычисляются.")
    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("Стационар и непрерывность помощи")
        metric_card("length_of_stay", series_map, year)
        metric_card("repeat_share", series_map, year)
        full_width(st.plotly_chart, line_chart(series_map["repeat_share"], height=310), config={"displayModeBar": False})
        st.caption(CATALOG["repeat_share"]["note"])
    with right:
        st.subheader("Альтернативы и социальное участие")
        metric_card("day_discharge", series_map, year)
        metric_card("employment_share", series_map, year)
        full_width(st.plotly_chart, line_chart(series_map["day_discharge"], color=PURPLE, height=310), config={"displayModeBar": False})
        st.caption(CATALOG["employment_share"]["note"])
    st.subheader("Что изменится в арифметике нагрузки?")
    with st.container(border=True):
        st.caption("Сценарий распределения посещений: одинаковая производительность и состав приема. Это расчёт при заданных условиях, не прогноз и не рекомендация штатного норматива.")
        c1, c2 = st.columns(2)
        with c1:
            demand = st.slider("Изменение числа посещений, %", -30, 50, 10, step=5)
        with c2:
            extra = st.number_input("Изменение занятых должностей", min_value=-30.0, max_value=50.0, value=5.0, step=.5)
        visits = point(series_map["adult_visits"], year).value
        positions = point(series_map["adult_positions"], year).value
        scenario = capacity_scenario(visits, positions, demand, extra)
        a, b, c = st.columns(3)
        a.metric("Сейчас · посещений / должность", fmt(scenario["current"]))
        b.metric("В сценарии · посещений / должность", fmt(scenario["scenario"]))
        c.metric("Заданный поток посещений", fmt(scenario["visits"]))
    st.subheader("Безопасность и права — отдельные ряды")
    key = st.selectbox("Показатель безопасности", ["attempts_disp", "fatal_disp", "attempts_cons", "fatal_cons", "involuntary"], format_func=lambda k: CATALOG[k]["label"])
    full_width(st.plotly_chart, line_chart(series_map[key], color=CORAL), config={"displayModeBar": False})
    trace_details(key, series_map, year)
    st.caption("Для редких событий показывай абсолютные значения. Даже крупный процент при малом числе не означает устойчивый тренд или индивидуальный риск.")


def quality_page(data, series_map, alerts, year, policy):
    st.subheader("Разделяем изменение службы и ошибку источника")
    numerical = data.value_num.notna().sum()
    allowed = data.quality_flag.isin(POLICIES[policy]) & data.value_num.notna()
    checks = quality_checks(data, policy)
    a, b, c = st.columns(3)
    a.metric("Исходных ячеек", fmt(len(data)))
    b.metric("Числовых ячеек в разрешённых флагах", fmt(allowed.sum()), help="Это обзор флагов. Метрики дополнительно проверяют статус числа, смысл графы и номер строки.")
    c.metric("Служебных проверок с замечаниями", fmt(len(checks)))
    st.caption(f"Числовых ячеек в CSV: {numerical}. Флаг digital означает извлечение из цифрового отчета, а не полную ручную верификацию отчетности.")
    st.subheader("Объяснимые аномалии")
    only_selected = st.checkbox("Только выбранный год", value=True)
    shown = alerts.loc[alerts.year == year] if only_selected else alerts
    if shown.empty:
        st.info("По выбранным порогам и качеству данных сигналов нет.")
    else:
        full_width(st.dataframe, shown[["year", "indicator", "value", "previous", "change_pct", "reason", "quality_flag", "action"]].rename(columns={"year": "Год", "indicator": "Показатель", "value": "Значение", "previous": "Прошлый год", "change_pct": "Изменение, %", "reason": "Почему подсвечено", "quality_flag": "Качество", "action": "Что проверить"}), hide_index=True)
        st.download_button("Скачать сигналы", csv_bytes(shown), "anomalies.csv", "text/csv")
    st.caption("Алгоритм использует изменения к соседнему календарному году и прошлые значения MAD. Малые числа <20 подавляют процентные сигналы; разрыв отчетности не заполняется. Это правила внимания, без p-value и клинических нормативов.")
    st.subheader("Полнота отчетов")
    coverage = data.groupby(["year", "form_number"]).size().unstack("form_number").reindex(years_of(data)).fillna(0).astype(int)
    full_width(st.dataframe, coverage.rename(columns={"10": "Ячеек формы 10", "36": "Ячеек формы 36"}))
    st.caption("Количество ячеек показывает объем выгрузки, а не доказательство наличия всех исходных таблиц. Нет данных и нулевое значение различаются.")
    if not checks.empty:
        full_width(st.dataframe, checks.rename(columns={"year": "Год", "check": "Проверка", "detail": "Наблюдение", "severity": "Тип"}), hide_index=True)
    st.subheader("Какие точки не попали в показатели")
    excluded = pd.concat(series_map.values(), ignore_index=True)
    excluded = excluded.loc[excluded.value.isna(), ["year", "label", "reason", "quality_flag", "source_file", "mapping"]]
    full_width(st.dataframe, excluded, hide_index=True)
    st.download_button("Скачать причины исключения", csv_bytes(excluded), "excluded_metric_points.csv", "text/csv")
    suspicious = data.loc[data.quality_flag.isin(["ocr_review", "layout_review", "source_annotation_review", "conflict_review"]) | data.value_status.isin(["text", "conflict"])]
    with st.expander("Сомнительные исходные ячейки"):
        full_width(st.dataframe, suspicious[["year", "form_number", "table_code", "row_number", "column_number", "value_raw", "value_num", "quality_flag", "source_annotation", "source_file"]], hide_index=True)
        st.download_button("Скачать ячейки для сверки", csv_bytes(suspicious), "quality_review.csv", "text/csv")


def methodology_page(data, series_map, policy):
    st.subheader("Данные, формулы и основания решений")
    st.write("СППР предназначена для управленческого разбора психиатрической службы Иркутской области. Она не ставит диагнозы, не оценивает индивидуальный риск и не выбирает лечение. Город Иркутск отдельно в этих файлах не выделен.")
    with st.expander("Исследования и принятые принципы", expanded=True):
        sources = json.loads((ROOT / "docs" / "sources.json").read_text(encoding="utf-8"))
        for source in sources:
            st.markdown(f"**[{source['title']}]({source['url']})** — {source['application']}")
        st.caption("Перенос выводов исследований на этот агрегированный региональный дашборд — проектное решение. Работа не является испытанием клинической эффективности системы.")
    with st.expander("Словарь показателей и формулы"):
        dictionary = []
        for key, item in CATALOG.items():
            formula = " / ".join(item.get("dependencies", []))
            if "slot" in item:
                f, t, r, old, new, _ = item["slot"]
                formula = f"Форма {f}, таблица {t}, строка {r}; графа {old} до 2023, {new} с 2023"
                if t == "2150":
                    formula += f"; Excel 2020/2023: графа {int(old) + 2}"
                if key in {"working_age", "working_age_employed"}:
                    formula += f"; Excel 2020/2023: графа {4 if key == 'working_age' else 6}"
            if key == "repeat_share":
                formula = "100 × (admissions − first_admission_year) / admissions"
            if key == "employment_share":
                formula = "100 × working_age_employed / working_age"
            dictionary.append(dict(metric_id=key, indicator=item["label"], unit=item["unit"], formula=formula, note=item["note"]))
        full_width(st.dataframe, pd.DataFrame(dictionary), hide_index=True)
        st.download_button("Скачать словарь показателей", csv_bytes(pd.DataFrame(dictionary)), "metric_dictionary.csv", "text/csv")
    with st.expander("Чего не хватает для следующих решений"):
        st.markdown("""
        - Среднегодовая численность населения и возрастные знаменатели — для корректного сравнения на 100 тыс.
        - Муниципалитеты, место обращения, очередь и время ожидания — для оценки территориальной доступности.
        - Штатные вакансии, физические лица и среднегодовые FTE — для кадрового планирования.
        - Форма 36 за 2022 год, месячная динамика и контрольные регионы — для более сильного анализа временных изменений.
        - Связанные эпизоды лечения и 30-дневные повторные госпитализации — для непрерывности помощи.
        - PROMs/PREMs, функциональные исходы и доступность реабилитации — для оценки результата для пациента.
        - Независимая статистика причин смерти — для суицидальной смертности всего населения.
        """)
    st.subheader("Уточнить любую таблицу")
    a, b, c = st.columns(3)
    with a:
        form = st.selectbox("Форма", sorted(data.form_number.unique()), key="raw_form")
    with b:
        table = st.selectbox("Код таблицы", sorted(data.loc[data.form_number == form, "table_code"].unique()), key=f"raw_table_{form}")
    with c:
        years = sorted(data.loc[(data.form_number == form) & (data.table_code == table), "year"].unique())
        chosen_year = st.selectbox("Год таблицы", years, index=len(years) - 1, key=f"raw_year_{form}_{table}")
    selected = data.loc[(data.form_number == form) & (data.table_code == table) & (data.year == chosen_year)]
    full_width(st.dataframe, selected[["row_number", "column_number", "entity_name", "icd10", "metric_name", "value_raw", "value_num", "quality_flag", "source_file", "source_page"]], hide_index=True)
    st.download_button("Скачать таблицу в длинном виде", csv_bytes(selected), f"form_{form}_table_{table}_{chosen_year}.csv", "text/csv")
    all_metrics = pd.concat(series_map.values(), ignore_index=True)
    st.download_button("Скачать рассчитанные ряды", csv_bytes(all_metrics), "derived_series.csv", "text/csv")
    st.caption(f"Текущий режим качества: {policy}. При смене режима расчёты и экспорты пересчитываются.")


def main():
    style()
    with st.sidebar:
        st.markdown(f'<div class="brand"><img src="{logo_data_uri()}" alt="Волна"><small>ПСИХИАТРИЧЕСКАЯ СЛУЖБА</small></div>', unsafe_allow_html=True)
        section = st.radio("Раздел", SECTIONS, key="section", label_visibility="collapsed")
        st.divider()
        # Место фильтров создаём раньше загрузки, а заполняем после чтения CSV.
        # Так фильтры учитывают новые данные и остаются выше загрузки в меню.
        controls = st.container()
        st.divider()
        with st.expander("Загрузка данных", expanded=False):
            uploads = st.file_uploader("Обновить сводные CSV", type=["csv"], accept_multiple_files=True, key="summary_csv_uploads", help="Загруженные файлы полностью заменяют комплект примера. Выбери форму 10 и форму 36.")
        st.markdown('<div class="sidebar-signature">С уважением,<br><strong>команда Волны</strong></div>', unsafe_allow_html=True)
    try:
        content = tuple(file.getvalue() for file in uploads) if uploads else tuple((ROOT / "data" / f"form_{number}_all_years_long.csv").read_bytes() for number in [10, 36])
        data = prepared_data(content)
    except (ValueError, OSError, pd.errors.ParserError) as error:
        st.error(f"Не удалось прочитать данные: {error}")
        st.stop()
    regions = [r for r in data.region.unique() if r]
    region = regions[0] if regions else "Регион не указан"
    with controls:
        year = st.selectbox("Отчетный год", sorted(data.year.unique()), index=len(data.year.unique()) - 1, key="year")
        policy = st.selectbox("Качество для расчётов", list(POLICIES), key="quality_policy", help="По умолчанию OCR исключен. Разрешение OCR не подтверждает правильность распознавания.")
        minimum, maximum = int(data.year.min()), int(data.year.max())
        selected_years = st.slider("Период графиков", minimum, maximum, (minimum, maximum), key="history") if minimum < maximum else (minimum, maximum)
        threshold = st.slider("Порог годового изменения, %", 5, 50, 15, step=5, help="Порог сигнала внимания, не норматив здравоохранения.")
        st.caption(f"{region}\n\nГодовые формы 10 и 36. Последний отчет в комплекте: {maximum}.")
    series_map = prepared_series(data, policy)
    alerts = detect_anomalies(series_map, threshold)
    st.markdown('<div class="eyebrow">РЕГИОНАЛЬНАЯ СИСТЕМА ПОДДЕРЖКИ РЕШЕНИЙ</div>', unsafe_allow_html=True)
    st.title({"Обзор": "Психиатрическая служба: главное", "Динамика и события": "Динамика в контексте событий", "Структура пациентов": "Структура пациентов", "Нагрузка и ресурсы": "Ресурсы и пути помощи", "Аномалии и качество": "Сигналы и доверие к данным", "Данные и методика": "От показателя — к источнику"}[section])
    st.markdown(f'<span class="chip">{escape(region)}</span><span class="chip">{year} год</span><span class="chip">{escape(policy)}</span>', unsafe_allow_html=True)
    if "Иркут" not in region:
        st.warning("В загруженных CSV указан другой регион или регион не указан. Заголовок данных обновлен; события в разделе контекста относятся к Иркутской области.")
    if policy != DEFAULT_POLICY:
        st.warning("Исследовательский режим: разрешено использование OCR. Проверь исходники перед управленческими решениями.")
    if not ((data.year == year) & (data.form_number == "36")).any():
        st.info(f"Форма 36 за {year} год отсутствует. Показатели службы за этот год оставлены без значения.")
    if section == "Обзор":
        overview(data, series_map, alerts, year, policy, selected_years)
    elif section == "Динамика и события":
        events_page(data, series_map, year, selected_years)
    elif section == "Структура пациентов":
        structure_page(data, year, policy)
    elif section == "Нагрузка и ресурсы":
        resources_page(series_map, year)
    elif section == "Аномалии и качество":
        quality_page(data, series_map, alerts, year, policy)
    else:
        methodology_page(data, series_map, policy)
    # Нормирование включается только при появлении соответствующего знаменателя.
    if section == "Динамика и события":
        with st.expander("Нормировать учетные показатели на 100 тыс. жителей"):
            st.caption("Для численности населения нужна среднегодовая оценка той же территории. Это показатели зарегистрированного учета, а не оценка истинной распространенности.")
            template = pd.DataFrame({"year": years_of(data), "population_total": [""] * len(years_of(data))})
            st.download_button("Скачать шаблон населения", csv_bytes(template), "population_template.csv", "text/csv")
            population = st.file_uploader("CSV с year и population_total", type=["csv"], key="population")
            if population:
                try:
                    denominator = population_frame(population.getvalue())
                    key = st.selectbox("Показатель на 100 тыс.", ["registered", "new"], format_func=lambda k: CATALOG[k]["label"], key="rate_metric")
                    normalized = per_100k(series_map[key], denominator)
                    full_width(st.plotly_chart, line_chart(normalized))
                    st.download_button("Скачать нормированный ряд", csv_bytes(normalized), "rates_per_100k.csv", "text/csv")
                except (ValueError, pd.errors.ParserError) as error:
                    st.error(str(error))
    st.divider()
    st.markdown('<div class="footnote">Управленческий мониторинг · Годовая отчетность · Никакой интерполяции отсутствующих отчетов · Каждое число можно сверить с источником</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
