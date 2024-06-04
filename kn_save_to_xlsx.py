# save_to_excel.py
import openpyxl
from openpyxl.worksheet.datavalidation import DataValidation

def save_to_excel(word_map, filename):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(['Word', 'Count', 'Forms', 'Sentences', 'Status'])

    # Create a data validation object
    dv = DataValidation(
        type="list",
        formula1='"Known,Unknown"',
        allow_blank=True
    )
    dv.error = 'Invalid entry. Please select from the list.'
    dv.errorTitle = 'Invalid Entry'
    dv.prompt = 'Please select from the list'
    dv.promptTitle = 'Status Selection'

    # Apply the data validation to the Status column
    ws.add_data_validation(dv)

    for word, data in word_map.items():
        ws.append([
            word,
            data['count'],
            ', '.join(data['forms']),
            ' | '.join(data['chosen_sentences']),
            ''
        ])

    # Adjust the data validation range to include all rows in the Status column
    dv.add(f'E2:E{ws.max_row}')

    wb.save(filename)
