"""Подключить полностью проверенный output нового парсера к СППР."""
import argparse
from pathlib import Path
import shutil
import tempfile

from analytics import load_frames
from indicators import load_population


def main():
    parser = argparse.ArgumentParser(description='Обновить данные СППР из output парсера')
    parser.add_argument('parser_output', type=Path)
    args = parser.parse_args()
    source = args.parser_output.resolve()
    root = Path(__file__).resolve().parent
    medical = load_frames([source / 'medical/psychiatry_all_long.csv'])
    load_population(source / 'population/population_denominators_2013_2024.csv')
    if set(medical.region.str.strip()) != {'Иркутская область'}:
        raise ValueError('Медицинская территория не совпадает с населением.')
    for form in ['10', '36']:
        if not (source / 'medical' / f'form_{form}_all_years_long.csv').is_file():
            raise FileNotFoundError('В output отсутствует файл формы ' + form)
    # Сначала копируем весь комплект в staging. Ошибка чтения источника
    # не должна оставить приложение без его старых файлов.
    with tempfile.TemporaryDirectory(prefix='data_update_', dir=root) as temp:
        stage = Path(temp)
        for folder in ['medical', 'population', 'exports']:
            shutil.copytree(source / folder, stage / folder)
        shutil.copy2(source / 'build_report.json', stage / 'build_report.json')
        for folder in ['medical', 'population', 'exports']:
            target = root / ('data/' + folder if folder != 'exports' else folder)
            shutil.copytree(stage / folder, target, dirs_exist_ok=True)
    shutil.copy2(source / 'build_report.json', root / 'docs/build_report.json')
    print('Данные СППР обновлены. Перезапусти приложение, чтобы сбросить старый сеанс.')


if __name__ == '__main__':
    main()
