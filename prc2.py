#ТВ-23, Хворостяний Артем
#  20.Перетворення дат. Створіть функцію, яка конвертує нестандартні формати дат у datetime64.

import pandas as pd
import kagglehub
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
# path = kagglehub.dataset_download("ishajangir/crime-data")
file_path = "/Users/artem/.cache/kagglehub/datasets/ishajangir/crime-data/versions/1/Crime_Data_from_2020_to_Present.csv"

df = pd.read_csv(file_path)

# Виведення всіх колонок
print("\n\n\nСписок колонок у датасеті:")
print(df.columns)

# print("Path to dataset files:", path)

print("\n\n\nПерші 10 записів у датасеті:")
print(df.head(10))
print("\n\n\nТипи даних у датафреймі:\n", df.dtypes)
def convert_to_datetime(df, column_name):
    df[column_name] = pd.to_datetime(df[column_name], errors='coerce', format="%m/%d/%Y %I:%M:%S %p")
    return df

# Конвертуємо дати
df = convert_to_datetime(df, 'Date Rptd')
df = convert_to_datetime(df, 'DATE OCC')

# Виведемо перші 5 рядків після конвертації
print("\n\n\n", df[['Date Rptd', 'DATE OCC']].head())
print("\n\n\nТипи даних у датафреймі:\n", df.dtypes)
