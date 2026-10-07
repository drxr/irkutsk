"""Расчёты СППР: каждая метрика имеет явный адрес ячейки и источник.

Функциональный модуль не зависит от Streamlit. Пропуски сохраняются,
подстроки «из них» не суммируются с родительскими итогами.
"""

from io import BytesIO
from pathlib import Path
import re

import numpy as np
import pandas as pd


POLICIES = {
    "Только цифровые и сверенные": {"digital", "visually_checked"},
    "Добавить OCR без ручной сверки": {
        "digital", "visually_checked", "ocr_unverified",
    },
    "Все числовые, включая сомнительные": {
        "digital", "visually_checked", "ocr_unverified", "ocr_review",
    },
}
DEFAULT_POLICY = next(iter(POLICIES))
REQUIRED = {
    "source_file", "form_number", "year", "table_code", "row_number",
    "column_number", "metric_name", "value_num", "value_status", "quality_flag",
}
CELL_KEY = ["form_number", "year", "table_code", "row_number", "column_number"]

# slot=(форма, таблица, строка, графа до 2023, графа с 2023, контроль смысла).
# С 2023 года в ряде таблиц добавлена графа детей из учреждений соцобслуживания.
METRICS = {
    "registered": dict(label="Зарегистрированные пациенты", unit="чел.",
                       slot=("10", "2000", "1", "4", "4", "з[а]?регистр"),
                       note="Все зарегистрированные за год, F00–F09 и F20–F99. Это учет службы, а не популяционная распространённость.",
                       action="Сверить полноту учета, маршрутизацию и доступность амбулаторной помощи.", priority=1),
    "new": dict(label="Впервые установленный диагноз", unit="чел.",
                slot=("10", "3000", "1", "4", "4", "впервые"),
                note="Впервые в жизни установленный диагноз; на выявление влияет доступность службы.",
                action="Проверить охват выявления и доступность первичной консультации.", priority=1),
    "admissions": dict(label="Поступления в стационар", unit="по форме, чел.",
                       slot=("36", "2300", "1", "4", "4", "поступ"),
                       note="Строка 1: психические расстройства. Повторные поступления возможны; строка ПАВ «кроме того» не включена.",
                       action="Сопоставить изменение потока с амбулаторной доступностью и составом поступлений.", priority=2),
    "dispensary": dict(label="Диспансерное наблюдение", unit="чел. на конец года",
                       slot=("36", "2100", "1", "10", "11", "конец"),
                       note="Контингент на конец года; графа 10 до 2023 года, 11 с 2023 года.",
                       action="Проверить снятия, переводы и непрерывность наблюдения.", priority=3),
    "consultative": dict(label="Консультативная помощь", unit="чел. на конец года",
                         slot=("36", "2110", "1", "10", "11", "помощ|конец"),
                         note="Пациенты, которым продолжает оказываться консультативно-лечебная помощь.",
                         action="Проверить причины прекращения обращений и маршруты повторного доступа.", priority=3),
    "adult_visits": dict(label="Посещения участковых психиатров для взрослых", unit="посещений",
                         slot=("36", "2200", "1", "4", "4", "посещ"),
                         note="Все посещения, включая освидетельствования; это не число пациентов.",
                         action="Разделить лечебные посещения и освидетельствования; проверить очереди.", priority=2),
    "adult_positions": dict(label="Занятые должности участковых психиатров для взрослых", unit="должностей на конец года",
                            slot=("36", "2200", "1", "3", "3", "занят"),
                            note="Должности, а не физические лица; нет штатных вакансий и среднегодовых FTE.",
                            action="Сверить кадровый учет и фактическое распределение приема.", priority=2),
    "attempts_disp": dict(label="Суицидальные попытки · диспансерная группа", unit="чел. по форме",
                          slot=("36", "2150", "1", "1", "1", "суицид"),
                          note="Только пациенты, отраженные в таблице 2150; не все жители региона.",
                          action="Сверить полноту регистрации и организацию кризисной помощи.", priority=0),
    "fatal_disp": dict(label="Завершённые попытки · диспансерная группа", unit="чел. по форме",
                       slot=("36", "2150", "1", "2", "2", "заверш"),
                       note="Подмножество попыток диспансерной группы. Не региональная статистика смертности.",
                       action="Проверить учет и маршруты кризисной помощи; разбор организует ответственный специалист.", priority=0),
    "attempts_cons": dict(label="Суицидальные попытки · консультативная группа", unit="чел. по форме",
                          slot=("36", "2150", "1", "3", "3", "суицид|консультатив"),
                          note="Консультативная группа отдельно: уникальность людей между группами за год не установлена.",
                          action="Сверить учет и доступность кризисной помощи.", priority=0),
    "fatal_cons": dict(label="Завершённые попытки · консультативная группа", unit="чел. по форме",
                       slot=("36", "2150", "1", "4", "4", "заверш"),
                       note="Подмножество попыток консультативной группы; не суммируется в показатель всех жителей.",
                       action="Проверить учет и организацию кризисного реагирования.", priority=0),
    "discharges": dict(label="Выбыло из стационара", unit="чел. по форме",
                       slot=("36", "2300", "1", "10", "11", "выбыло"),
                       note="Выбывшие пациенты строки 1, включая умерших.",
                       action="Сопоставить поток выписок с поступлениями.", priority=3),
    "discharge_days": dict(label="Койко-дни выбывших", unit="койко-дней",
                           slot=("36", "2300", "1", "11", "12", "койко|койк"),
                           note="Койко-дни выписанных и умерших, а не все занятые койко-дни календарного года.",
                           action="Изучить длительность пребывания и структуру диагнозов.", priority=3),
    "first_admission_year": dict(label="Поступили впервые в данном году", unit="чел. по форме",
                                 slot=("36", "2300", "1", "7", "8", "впервые.*году|впервые.*года"),
                                 note="Основа приближённой доли повторных поступлений, а не 30-дневных повторных госпитализаций.",
                                 action="Проверить переход от стационарной к амбулаторной помощи.", priority=3),
    "involuntary": dict(label="Недобровольные поступления", unit="чел. по форме",
                        slot=("36", "2300", "1", "9", "10", "недобро"),
                        note="Строка 1, графа поступлений в соответствии со ст. 29. Это мониторинг учета, без правовой оценки случая.",
                        action="Проверить сопоставимость регистрации и организацию помощи.", priority=3),
    "day_discharge": dict(label="Дневной стационар · выписано", unit="чел. по форме",
                          slot=("36", "2600", "1", "5", "5", "выпис"),
                          note="Дневной стационар отдельно от круглосуточного; категории нельзя суммировать как уникальных пациентов.",
                          action="Проверить доступность дневных и амбулаторных альтернатив.", priority=2),
    "day_places": dict(label="Дневной стационар · среднегодовые места", unit="мест",
                       slot=("36", "2600", "1", "4", "4", "средне"),
                       note="Места дневного стационара. Коечный фонд круглосуточного стационара в этих CSV не задан.",
                       action="Сопоставить доступность мест с маршрутизацией пациентов.", priority=3),
    "disability_new": dict(label="Впервые признаны инвалидами", unit="чел. по форме",
                           slot=("36", "2180", "1", "4", "4", "впервые"),
                           note="Не исход лечения; показатель зависит от правил и полноты учета.",
                           action="Оценить доступность социальной и реабилитационной помощи.", priority=3),
    "disability_stock": dict(label="Имеют группу инвалидности", unit="чел. на конец года",
                             slot=("36", "2180", "1", "7", "8", "конец"),
                             note="Сдвиг графы 7 → 8 в 2023 году. Общесоматическая инвалидность «кроме того» не включена.",
                             action="Проверить доступность реабилитации и социальной поддержки.", priority=3),
    "working_age": dict(label="Трудоспособный возраст · наблюдаемые", unit="чел.",
                        slot=("36", "2120", "1", "2", "2", "трудоспособ"),
                        note="Контингент таблицы 2120, а не все зарегистрированные за год.",
                        action="Уточнить потребность в сопровождении занятости.", priority=4),
    "working_age_employed": dict(label="Работающие в трудоспособном возрасте", unit="чел.",
                                 slot=("36", "2120", "1", "4", "4", "трудоспособ"),
                                 note="Числитель доли занятости в том же контингенте таблицы 2120.",
                                 action="Оценить организацию социальной реабилитации.", priority=4),
    "military_exams": dict(label="Военная психиатрическая экспертиза", unit="чел. по форме",
                           slot=("36", "2500", "1", "4", "4", "военную"),
                           note="Вид экспертизы. Данные не идентифицируют участников СВО и не измеряют последствия для этой группы.",
                           action="Уточнить состав потока экспертиз и правила учета.", priority=4),
}

DERIVED = {
    "visits_per_position": dict(label="Посещений на занятую должность · взрослые", unit="посещений / должность",
                                dependencies=["adult_visits", "adult_positions"], operation="ratio", scale=1,
                                note="Прокси нагрузки: годовые посещения / должности на конец года. Это не норматив нагрузки и не среднегодовой FTE.",
                                action="Проверить расписания, очереди и годовую динамику кадров.", priority=1),
    "length_of_stay": dict(label="Среднее пребывание выбывших", unit="дней",
                           dependencies=["discharge_days", "discharges"], operation="ratio", scale=1,
                           note="Койко-дни выбывших / число выбывших в строке 1. Длительность сама по себе не определяет качество помощи.",
                           action="Сверить состав случаев и условия выписки.", priority=2),
    "repeat_share": dict(label="Доля повторных поступлений в данном году", unit="%",
                         dependencies=["admissions", "first_admission_year"], operation="repeat", scale=100,
                         note="(Все поступления − впервые в данном году) / все поступления. Не показатель повторной госпитализации за 30 дней.",
                         action="Проверить маршруты после выписки и причины повторных поступлений.", priority=2),
    "employment_share": dict(label="Доля работающих · трудоспособный контингент", unit="%",
                             dependencies=["working_age_employed", "working_age"], operation="ratio", scale=100,
                             note="Работающие трудоспособного возраста / все трудоспособного возраста в таблице 2120.",
                             action="Обсудить доступность сопровождения занятости и социальной реабилитации.", priority=3),
}
CATALOG = {**METRICS, **DERIVED}


def load_frames(inputs):
    """Прочитать CSV из путей или байтов, проверить схему и уникальность."""
    frames = []
    for source in inputs:
        buffer = BytesIO(source) if isinstance(source, bytes) else source
        frame = pd.read_csv(buffer, dtype=str, keep_default_na=False)
        missing = REQUIRED - set(frame.columns)
        if missing:
            raise ValueError(f"В CSV отсутствуют поля: {', '.join(sorted(missing))}")
        for column in ["form_number", "table_code", "row_number", "column_number"]:
            frame[column] = frame[column].str.strip().str.replace(r"\.0$", "", regex=True)
        frame["year"] = pd.to_numeric(frame.year, errors="raise").astype(int)
        if not frame.year.between(1900, 2100).all():
            raise ValueError("Год отчета выходит за допустимый диапазон.")
        if not set(frame.form_number).issubset({"10", "36"}):
            raise ValueError("Приложение ожидает только формы 10 и 36.")
        if "period_type" in frame and not frame.period_type.eq("annual").all():
            raise ValueError("Приложение рассчитано на годовые отчеты; другие периоды не объединяются.")
        for column in ["region", "entity_name", "icd10", "source_page", "source_format", "source_annotation", "row_number_inferred", "metric_name_reference"]:
            if column not in frame:
                frame[column] = ""
        frame["value_num"] = pd.to_numeric(frame.value_num, errors="coerce")
        if "value_raw" not in frame:
            frame["value_raw"] = frame.value_num.fillna("").astype(str)
        frames.append(frame)
    if not frames:
        raise ValueError("Нет CSV для анализа.")
    data = pd.concat(frames, ignore_index=True)
    if data.duplicated(CELL_KEY).any():
        raise ValueError("Есть повторяющиеся ключи ячеек. Загрузи по одному отчету формы за год; одинаковые CSV нельзя складывать.")
    regions = set(data.region.str.strip()) - {""}
    if len(regions) > 1:
        raise ValueError("CSV содержат несколько регионов. Их нельзя смешивать в региональном дашборде.")
    return data


def years_of(data):
    """Полная ось лет; отсутствующие годы остаются пропусками."""
    return list(range(int(data.year.min()), int(data.year.max()) + 1))


def cell_value(data, year, form, table, row, column, policy=DEFAULT_POLICY, pattern=""):
    """Вернуть значение и происхождение; не превращать плохую ячейку в ноль."""
    subset = data.loc[(data.year == year) & (data.form_number == str(form))
                      & (data.table_code == str(table)) & (data.row_number == str(row))
                      & (data.column_number == str(column))]
    base = dict(year=year, value=np.nan, raw_value=np.nan, reason="Нет ячейки / отчета",
                quality_flag="missing", source_file="", source_page="", form_number=str(form),
                table_code=str(table), row_number=str(row), column_number=str(column), mapping="")
    if subset.empty:
        return base
    if len(subset) != 1:
        raise ValueError("Неоднозначная ячейка: нельзя суммировать дубликаты.")
    record = subset.iloc[0]
    base.update(raw_value=record.value_num, quality_flag=record.quality_flag,
                source_file=record.source_file, source_page=record.source_page,
                mapping=f"Форма {form} · таблица {table} · строка {row} · графа {column}")
    if not np.isfinite(record.value_num) or record.value_num < 0:
        base["reason"] = "Нет допустимого неотрицательного числа"
    elif record.value_status in {"text", "not_applicable", "conflict"}:
        base["reason"] = "Текст, X или конфликт в источнике"
    elif str(record.row_number_inferred).lower() in {"true", "1"}:
        base["reason"] = "Номер строки восстановлен неоднозначно"
    elif record.quality_flag not in POLICIES[policy]:
        base["reason"] = "Исключено выбранным режимом качества"
    elif pattern and record.quality_flag == "digital" and not re.search(pattern, re.sub(r"[\s\-–—\u00ad]", "", record.metric_name.lower())):
        base["reason"] = "Нужна проверка смысла графы / редакции формы"
    else:
        base.update(value=float(record.value_num), reason="")
    return base


def metric_series(data, metric_id, policy=DEFAULT_POLICY):
    """Построить ряд без интерполяции с учетом редакции формы и качества."""
    spec = CATALOG[metric_id]
    if "slot" in spec:
        form, table, row, old_col, new_col, pattern = spec["slot"]
        records = []
        for year in years_of(data):
            column = new_col if year >= 2023 else old_col
            # В короткой таблице 2120 нумерация в Excel 2020/2023 начинается
            # с графы 3, в Word — с графы 1. Проверено по названиям CSV.
            if table == "2120" and year in {2020, 2023}:
                column = {"working_age": "4", "working_age_employed": "6"}.get(metric_id, column)
            if table == "2150" and year in {2020, 2023}:
                column = str(int(column) + 2)
            records.append(cell_value(data, year, form, table, row, column, policy, pattern))
        result = pd.DataFrame(records)
    else:
        components = [metric_series(data, key, policy).set_index("year") for key in spec["dependencies"]]
        rows = []
        for year in years_of(data):
            left, right = (part.loc[year] for part in components)
            value = np.nan
            reason = "; ".join(dict.fromkeys(str(r) for r in [left.reason, right.reason] if r))
            if np.isfinite(left.value) and np.isfinite(right.value):
                if spec["operation"] == "repeat":
                    if left.value > 0 and 0 <= right.value <= left.value:
                        value = (left.value - right.value) / left.value * 100
                    else:
                        reason = "Невозможный итог или подмножество больше итога"
                elif right.value > 0:
                    value = left.value / right.value * spec["scale"]
                    if spec["scale"] == 100 and value > 100:
                        value = np.nan
                        reason = "Подмножество больше итога"
                else:
                    reason = "Нулевой знаменатель: доля не вычисляется"
            quality = "digital" if all(p.loc[year].quality_flag == "digital" for p in components) else "mixed"
            if any(p.loc[year].quality_flag.startswith("ocr") for p in components):
                quality = "ocr_derived"
            rows.append(dict(year=year, value=value, raw_value=np.nan, reason=reason,
                             quality_flag=quality, source_file="; ".join(dict.fromkeys(p.loc[year].source_file for p in components)),
                             source_page="; ".join(str(p.loc[year].source_page) for p in components),
                             mapping=" / ".join(p.loc[year].mapping for p in components)))
        result = pd.DataFrame(rows)
    result["metric_id"] = metric_id
    result["label"] = spec["label"]
    result["unit"] = spec["unit"]
    # fill_method отсутствует: pandas не должен протянуть прошлый год через пропуск.
    result["yoy_pct"] = result.value.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan) * 100
    return result


def build_series(data, policy=DEFAULT_POLICY):
    """Один словарь рядов для всех экранов и экспортов."""
    return {key: metric_series(data, key, policy) for key in CATALOG}


def point(series, year):
    """Вернуть точку выбранного года; не подменять ее ближайшим доступным."""
    selected = series.loc[series.year == year]
    return selected.iloc[0] if not selected.empty else pd.Series(dict(value=np.nan, yoy_pct=np.nan, reason="Нет года", quality_flag="missing"))


def change_pct(before, after):
    """Относительное изменение; при нулевой базе показать только абсолютное."""
    return (after / before - 1) * 100 if np.isfinite(before) and np.isfinite(after) and before > 0 else np.nan


def period_comparison(series_map, before_years, after_years):
    """Описательное сравнение средних и прозрачное количество точек."""
    rows = []
    for key, series in series_map.items():
        before = series.loc[series.year.isin(before_years) & series.value.notna()]
        after = series.loc[series.year.isin(after_years) & series.value.notna()]
        b, a = before.value.mean(), after.value.mean()
        rows.append(dict(metric_id=key, indicator=CATALOG[key]["label"], before=b, after=a,
                         change_pct=change_pct(b, a), difference=a - b, n_before=len(before), n_after=len(after),
                         years_before=", ".join(map(str, before.year)), years_after=", ".join(map(str, after.year)),
                         unit=CATALOG[key]["unit"]))
    return pd.DataFrame(rows)


def detect_anomalies(series_map, threshold=15.0):
    """Объяснимые сигналы: порог YoY и MAD только по прошлым годовым изменениям.

    Минимум пять прошлых изменений для MAD; пропущенный год не образует YoY.
    Порог — пользовательское правило внимания, не клинический норматив.
    Для малых чисел (<20 в одном из двух лет) процентный сигнал подавлен.
    """
    alerts = []
    for key, series in series_map.items():
        ordered = series.sort_values("year").reset_index(drop=True)
        for index, record in ordered.iterrows():
            delta = record.yoy_pct
            if not np.isfinite(delta):
                continue
            previous = ordered.iloc[index - 1]
            small = CATALOG[key]["unit"] != "%" and min(previous.value, record.value) < 20
            reasons = []
            if abs(delta) >= threshold and not small:
                reasons.append(f"Годовое изменение {delta:+.1f}% ≥ порога {threshold:g}% по модулю")
            historical = ordered.loc[:index - 1, "yoy_pct"].dropna().tail(6)
            score = np.nan
            if len(historical) >= 5 and not small:
                median = float(historical.median())
                mad = float((historical - median).abs().median())
                if mad > 0:
                    score = 0.6745 * (delta - median) / mad
                    if abs(score) >= 3.5:
                        reasons.append(f"Изменение отличается от последних {len(historical)} прошлых изменений: robust z={score:.1f}")
            if reasons:
                alerts.append(dict(metric_id=key, year=int(record.year), indicator=CATALOG[key]["label"],
                                   value=record.value, previous=previous.value, change_pct=delta, robust_z=score,
                                   reason="; ".join(reasons), quality_flag=record.quality_flag,
                                   priority=CATALOG[key]["priority"], action=CATALOG[key]["action"], mapping=record.get("mapping", "")))
    columns = ["metric_id", "year", "indicator", "value", "previous", "change_pct", "robust_z", "reason", "quality_flag", "priority", "action", "mapping"]
    return pd.DataFrame(alerts, columns=columns).sort_values(["priority", "year"], ascending=[True, False])


def quality_checks(data, policy=DEFAULT_POLICY):
    """Проверить присутствие отчетов и балансы без суммирования иерархий."""
    rows = []
    for year in years_of(data):
        for form in ["10", "36"]:
            if not ((data.year == year) & (data.form_number == form)).any():
                rows.append(dict(year=year, check="Нет исходного отчета", detail=f"Форма {form}", severity="Нет данных"))
        for table in ["2000", "3000"]:
            for label, total, parts in [("Возрастной баланс", "4", list(map(str, range(6, 12)))),
                                        ("Диагностический баланс", "4", None)]:
                whole = cell_value(data, year, "10", table, "1", total, policy)
                if parts:
                    children = [cell_value(data, year, "10", table, "1", col, policy) for col in parts]
                else:
                    group_rows = diagnostic_rows(data, year, table)
                    children = [cell_value(data, year, "10", table, row, "4", policy) for row in group_rows.values()]
                    if len(children) != 3:
                        continue
                values = [x["value"] for x in children]
                if np.isfinite(whole["value"]) and all(np.isfinite(v) for v in values):
                    difference = sum(values) - whole["value"]
                    if abs(difference) > 0.01:
                        rows.append(dict(year=year, check=label, detail=f"Форма 10 / {table}: сумма − итог = {difference:g}", severity="Проверить источник"))
        for table, col in [("2000", "12"), ("2000", "13")]:
            form36 = "2100" if col == "12" else "2110"
            a = cell_value(data, year, "10", table, "1", col, policy)["value"]
            b = cell_value(data, year, "36", form36, "1", "11" if year >= 2023 else "10", policy)["value"]
            if np.isfinite(a) and np.isfinite(b) and abs(a - b) > 0.01:
                rows.append(dict(year=year, check="Согласование форм 10 / 36", detail=f"10/{table}/{col} против 36/{form36}: разница {a - b:g}", severity="Проверить учет"))
    return pd.DataFrame(rows, columns=["year", "check", "detail", "severity"])


def normalize_text(value):
    """Нормализовать название/МКБ, сохранив содержательный смысл."""
    return re.sub(r"[\s\-–—\u00ad]", "", str(value).lower())


def diagnostic_rows(data, year, table):
    """Найти три основные группы по названиям, а не номеру строки.

    В 2013 году непсихотическая группа и интеллектуальные нарушения
    находятся на строках 14 и 22; в следующих редакциях — 15 и 24.
    """
    rows = data.loc[(data.form_number == "10") & (data.table_code == str(table))
                    & (data.year == year), ["row_number", "entity_name"]].drop_duplicates()
    patterns = {"psychoses": r"^психозыисостояния|^психозыи\(или\)состояния",
                "nonpsychotic": r"^психическиерасстройстванепсихотичес|^непсихотическиепсихическиерасстройств",
                "intellectual": r"^умственнаяотсталость"}
    result = {}
    for key, pattern in patterns.items():
        selected = rows.loc[rows.entity_name.map(normalize_text).str.contains(pattern, regex=True), "row_number"].unique()
        if len(selected) == 1:
            result[key] = str(selected[0])
    return result


def diagnosis_history(data, table, current_year, current_row, column, policy):
    """Сравнить выбранный диагноз только при совпадении МКБ/определения."""
    current = data.loc[(data.form_number == "10") & (data.table_code == str(table))
                       & (data.year == current_year) & (data.row_number == str(current_row))].iloc[0]
    code, name = normalize_text(current.icd10), normalize_text(current.entity_name)
    records = []
    for year in years_of(data):
        rows = data.loc[(data.form_number == "10") & (data.table_code == str(table))
                        & (data.year == year), ["row_number", "icd10", "entity_name"]].drop_duplicates()
        if code:
            matching = rows.loc[rows.icd10.map(normalize_text).eq(code), "row_number"].unique()
        else:
            matching = rows.loc[rows.entity_name.map(normalize_text).eq(name), "row_number"].unique()
        if len(matching) == 1:
            item = cell_value(data, year, "10", table, matching[0], column, policy)
        else:
            item = dict(year=year, value=np.nan, quality_flag="definition_changed", reason="Нет однозначного совпадения МКБ / названия строки")
        records.append(item)
    result = pd.DataFrame(records)
    result["label"] = current.entity_name
    result["unit"] = "чел. по форме"
    return result


def population_frame(content):
    """Проверить пользовательские среднегодовые знаменатели для на 100 тыс."""
    data = pd.read_csv(BytesIO(content), dtype=str)
    if not {"year", "population_total"}.issubset(data.columns):
        raise ValueError("В населении нужны year и population_total.")
    data["year"] = pd.to_numeric(data.year, errors="raise").astype(int)
    data["population_total"] = pd.to_numeric(data.population_total, errors="coerce")
    if data.year.duplicated().any() or not data.population_total.gt(0).all():
        raise ValueError("Население: один положительный знаменатель на каждый год.")
    return data


def per_100k(series, population):
    """Применить знаменатель того же года; отсутствие населения не дает ноль."""
    output = series.merge(population[["year", "population_total"]], on="year", how="left", validate="one_to_one")
    output["value"] = output.value / output.population_total * 100000
    output["yoy_pct"] = output.value.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan) * 100
    output["unit"] = "на 100 тыс. жителей"
    return output


def capacity_scenario(visits, positions, demand_pct, extra_positions):
    """Арифметический сценарий; не прогноз потребности и не норматив."""
    if not np.isfinite(visits) or not np.isfinite(positions) or positions <= 0:
        return dict(current=np.nan, scenario=np.nan, visits=np.nan)
    new_positions = positions + extra_positions
    if new_positions <= 0:
        return dict(current=visits / positions, scenario=np.nan, visits=np.nan)
    new_visits = visits * (1 + demand_pct / 100)
    return dict(current=visits / positions, scenario=new_visits / new_positions, visits=new_visits)
