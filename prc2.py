#ТВ-23, Хворостяний Артем
#  20.Перетворення дат. Створіть функцію, яка конвертує нестандартні формати дат у datetime64.

import pandas as pd

def convert_to_datetime(dates):
    formats = ["%d-%m-%Y", "%Y/%m/%d", "%B %d, %Y", "%d.%m.%Y"]
    
    converted_dates = []
    for date in dates:
        parsed_date = None
        for fmt in formats:
            try:
                parsed_date = pd.to_datetime(date, format=fmt)
                break
            except ValueError:
                continue
        converted_dates.append(parsed_date.strftime('%Y-%m-%d') if parsed_date else 'Invalid Date')
    
    return converted_dates

dates_list = ["12-01-2024", "2024/02/15", "March 5, 2023", "01.04.2022"]
converted_dates = convert_to_datetime(dates_list)
print(converted_dates)
