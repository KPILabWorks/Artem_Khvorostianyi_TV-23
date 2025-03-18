# 20 Збережіть великі енергетичні дані у форматах CSV, Parquet, HDF5 та порівняйте швидкість читання/запису з попередньою реалізацією у SQL.
# Цей розширений аналіз дозволить оцінити ефективність нових методів у порівнянні з класичними підходами.

import pandas as pd
import pyarrow.parquet as pq
import h5py
import sqlite3
import time

# Створення DataFrame з енергетичними даними
df = pd.DataFrame({
    'timestamp': ['2025-03-01', '2025-03-02'],
    'energy_usage': [500, 600],
    'temperature': [22.5, 23.0]
})

# Запис і читання даних
results = []

# 1. Запис у формат CSV
start = time.time()
df.to_csv('energy_data.csv', index=False)
csv_write_time = time.time() - start
results.append(('CSV Write', csv_write_time))

# 2. Запис у формат Parquet
start = time.time()
df.to_parquet('energy_data.parquet')
parquet_write_time = time.time() - start
results.append(('Parquet Write', parquet_write_time))

# 3. Запис у формат HDF5 (потрібно привести дані до числових типів)
start = time.time()
df['timestamp'] = pd.to_datetime(df['timestamp'])  # Якщо хочете зберігати дату в HDF5
with h5py.File('energy_data.h5', 'w') as f:
    f.create_dataset('energy_data', data=df[['energy_usage', 'temperature']].to_numpy())
hdf5_write_time = time.time() - start
results.append(('HDF5 Write', hdf5_write_time))

# 4. Запис у базу даних SQL (SQLite)
conn = sqlite3.connect('energy_data.db')
cursor = conn.cursor()
cursor.execute('''
CREATE TABLE IF NOT EXISTS energy_data (
    timestamp TEXT,
    energy_usage REAL,
    temperature REAL
)
''')
start = time.time()
df.to_sql('energy_data', conn, if_exists='replace', index=False)
sql_write_time = time.time() - start
results.append(('SQL Write', sql_write_time))

# 5. Читання з CSV
start = time.time()
df_csv = pd.read_csv('energy_data.csv')
csv_read_time = time.time() - start
results.append(('CSV Read', csv_read_time))

# 6. Читання з Parquet
start = time.time()
df_parquet = pd.read_parquet('energy_data.parquet')
parquet_read_time = time.time() - start
results.append(('Parquet Read', parquet_read_time))

# 7. Читання з HDF5
start = time.time()
with h5py.File('energy_data.h5', 'r') as f:
    df_hdf5 = pd.DataFrame(f['energy_data'][:], columns=df.columns[1:])
hdf5_read_time = time.time() - start
results.append(('HDF5 Read', hdf5_read_time))

# 8. Читання з SQL
start = time.time()
df_sql = pd.read_sql('SELECT * FROM energy_data', conn)
sql_read_time = time.time() - start
results.append(('SQL Read', sql_read_time))

# Закриття з'єднання з базою даних SQL
conn.close()

# Сортуємо результати за часом
results.sort(key=lambda x: x[1])

# Перетворюємо в список для виведення у вигляді таблиці
result_df = pd.DataFrame(results, columns=["Operation", "Time (seconds)"])

# Виведення результатів у вигляді таблиці
print(result_df)
