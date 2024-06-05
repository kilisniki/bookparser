import openpyxl

def parse_vocabulary(filename):
    # Открыть файл Excel
    wb = openpyxl.load_workbook(filename)
    ws = wb.active

    # Создать пустой словарь для хранения данных
    word_map = {}

    # Пройти по всем строкам начиная со второй (первая строка содержит заголовки)
    for row in ws.iter_rows(min_row=2, values_only=True):
        word = row[0]
        count = int(row[1]) if row[1] else 0  # Приведение count к числу
        forms = row[2].split(', ') if row[2] else []
        sentences = row[3].split(' | ') if row[3] else []
        status = row[4]

        # Заполнить словарь
        word_map[word] = {
            'count': count,
            'forms': forms,
            'chosen_sentences': sentences,
            'status': status
        }

    return word_map
