# 25 Виконайте швидке попереднє обчислення статистичних ознак для машинного навчання на великому наборі даних про енергетичне споживання, використовуючи Featuretools.
#  Оцініть час та точність для різних стратегій створення ознак.

import pandas as pd
import featuretools as ft
import time
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


file_path =  "/Users/artem/Docs/Life/HomeWork/III Course/ЗбірДаних/PR3/household_power_consumption.csv"


df = pd.read_csv(file_path, sep=";", encoding="utf-8")
if len(df.columns) == 1:
    df = pd.read_csv(file_path, sep=",", encoding="utf-8")


print("Перших 5 рядків датасету:")
print(df.head())
print("\nОпис датасету:")
print(df.describe())
print("\nНазви колонок:")
print(df.columns.tolist())

# Попередня обробка даних
df.columns = df.columns.str.strip()
df["datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S")
df = df.drop(columns=["Date", "Time"])

for col in df.columns:
    if col != "datetime":
        df[col] = pd.to_numeric(df[col], errors="coerce")


time_features = df[["datetime"]].copy()
time_features["month"] = time_features["datetime"].dt.month
time_features["day"] = time_features["datetime"].dt.day
time_features["hour"] = time_features["datetime"].dt.hour
time_features["weekday"] = time_features["datetime"].dt.weekday


es = ft.EntitySet(id="energy_data")


es = es.add_dataframe(
    dataframe_name="power_consumption",
    dataframe=df,
    index="index",
    time_index="datetime",
)


es = es.add_dataframe(
    dataframe_name="time_features",
    dataframe=time_features,
    index="datetime",
)


es = es.add_relationship(
    parent_dataframe_name="time_features",
    parent_column_name="datetime",
    child_dataframe_name="power_consumption",
    child_column_name="datetime",
)


results = []
depths = [1, 2, 3]
for depth in depths:
    start_time = time.time()
    feature_matrix, feature_defs = ft.dfs(
        entityset=es,
        target_dataframe_name="power_consumption",
        trans_primitives=["month", "day", "hour", "weekday"],
        agg_primitives=["mean", "sum", "min", "max", "std"],
        max_depth=depth,
    )
    elapsed_time = time.time() - start_time
    results.append({
        "max_depth": depth,
        "time": elapsed_time,
        "num_features": len(feature_defs)
    })
    

    print(f"\nГлибина {depth}: {len(feature_defs)} ознак")
    print(feature_defs)


results_df = pd.DataFrame(results)
print("Глибина моделі | Час виконання (с) | Кількість характеристик")
print(results_df.to_string(index=False))
