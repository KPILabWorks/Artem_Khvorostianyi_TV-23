import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Завантаження та підготовка даних
raw_data_1 = pd.read_csv('/Users/artem/Docs/Life/HomeWork/III Course/ЗбірДаних/PR6/Audio data.csv')
raw_data_2 = pd.read_csv('/Users/artem/Docs/Life/HomeWork/III Course/ЗбірДаних/PR6/Audio data_1.csv')

raw_data_1.columns = ['Timestamp', 'Value']
raw_data_2.columns = ['Timestamp', 'Value']

# Візуалізація вихідного сигналу
plt.figure(figsize=(26, 9))
for idx, dataset in enumerate([(raw_data_1, 'Сигнал з тихого місця', '#1f77b4'),
                               (raw_data_2, 'Сигнал з гучного місця', '#ff7f0e')]):
    plt.subplot(1, 2, idx + 1)
    plt.plot(dataset[0]['Timestamp'], dataset[0]['Value'], color=dataset[2])
    plt.title(dataset[1])
    plt.xlabel("Час (мс)")
    plt.ylabel("Амплітуда")

plt.tight_layout()
plt.show()

# Функція згладжування сигналу за допомогою DWT
def wavelet_smoothing(data_array, wavelet_type='db4', dwt_level=4):
    coeff = pywt.wavedec(data_array, wavelet_type, level=dwt_level)
    coeff[1:] = [np.zeros_like(level) for level in coeff[1:]]
    return pywt.waverec(coeff, wavelet_type)[:len(data_array)]

smooth_signal_1 = wavelet_smoothing(raw_data_1['Value'].values)
smooth_signal_2 = wavelet_smoothing(raw_data_2['Value'].values)

plt.figure(figsize=(14, 6))
for i, (time, smoothed, title, color) in enumerate([
    (raw_data_1['Timestamp'], smooth_signal_1, 'Згладжений сигнал (тихо)', '#2ca02c'),
    (raw_data_2['Timestamp'], smooth_signal_2, 'Згладжений сигнал (гучно)', '#d62728')
]):
    plt.subplot(1, 2, i+1)
    plt.plot(time, smoothed, color=color)
    plt.title(title)
    plt.xlabel("Час (мс)")
    plt.ylabel("Амплітуда")
plt.tight_layout()
plt.show()

# Частота дискретизації
def estimate_sample_rate(time_col):
    return 1000 / np.mean(np.diff(time_col))

sampling_rate_1 = estimate_sample_rate(raw_data_1['Timestamp'])
sampling_rate_2 = estimate_sample_rate(raw_data_2['Timestamp'])

# Обчислення FFT
def compute_fft(values, sample_rate):
    N = len(values)
    fft_vals = fft(values)
    freq_axis = fftfreq(N, 1 / sample_rate)
    return freq_axis[:N // 2], np.abs(fft_vals[:N // 2])

freqs_1, amps_1 = compute_fft(raw_data_1['Value'].values, sampling_rate_1)
freqs_2, amps_2 = compute_fft(raw_data_2['Value'].values, sampling_rate_2)

plt.figure(figsize=(14, 6))
for i, (x, y, label, col) in enumerate([
    (freqs_1, amps_1, 'Фур\'є спектр (тихо)', '#17becf'),
    (freqs_2, amps_2, 'Фур\'є спектр (гучно)', '#bcbd22')
]):
    plt.subplot(1, 2, i+1)
    plt.plot(x, y, color=col)
    plt.title(label)
    plt.xlabel("Частота (Гц)")
    plt.ylabel("Амплітуда")
plt.tight_layout()
plt.show()

# CWT
scale_range = np.arange(1, 128)
cwt_1, _ = pywt.cwt(raw_data_1['Value'].values, scale_range, 'morl')
cwt_2, _ = pywt.cwt(raw_data_2['Value'].values, scale_range, 'morl')

plt.figure(figsize=(14, 6))
for i, (cwt_data, time_data, label) in enumerate([
    (cwt_1, raw_data_1['Timestamp'], "CWT (тихо)"),
    (cwt_2, raw_data_2['Timestamp'], "CWT (гучно)")
]):
    plt.subplot(1, 2, i+1)
    plt.imshow(cwt_data, extent=[time_data.min(), time_data.max(), scale_range.min(), scale_range.max()],
               cmap='coolwarm', aspect='auto', vmax=abs(cwt_data).max(), vmin=-abs(cwt_data).max())
    plt.title(label)
    plt.xlabel("Час (мс)")
    plt.ylabel("Масштаб")
plt.tight_layout()
plt.show()

# Піки сигналу
peak_ids_1, _ = find_peaks(raw_data_1['Value'], height=0)
peak_ids_2, _ = find_peaks(raw_data_2['Value'], height=0)

plt.figure(figsize=(14, 6))
for i, (data, peaks, title) in enumerate([
    (raw_data_1, peak_ids_1, "Піки (тихо)"),
    (raw_data_2, peak_ids_2, "Піки (гучно)")
]):
    plt.subplot(1, 2, i+1)
    plt.plot(data['Timestamp'], data['Value'], label='Сигнал', alpha=0.7)
    plt.plot(data['Timestamp'].iloc[peaks], data['Value'].iloc[peaks], "x", color='magenta', label='Піки')
    plt.title(title)
    plt.xlabel("Час (мс)")
    plt.ylabel("Амплітуда")
    plt.legend()
plt.tight_layout()
plt.show()

# Витягування ознак сигналу
def signal_stats(signal_array):
    return {
        'Середнє': np.mean(signal_array),
        'Стд. відхилення': np.std(signal_array),
        'Дисперсія': np.var(signal_array),
        'Амплітуда': np.ptp(signal_array),
        'Медіана': np.median(signal_array),
        'Енергія': np.sum(signal_array**2)
    }

print("Ознаки (тихе місце):")
for key, val in signal_stats(raw_data_1['Value'].values).items():
    print(f"{key}: {val:.4f}")

print("\nОзнаки (гучне місце):")
for key, val in signal_stats(raw_data_2['Value'].values).items():
    print(f"{key}: {val:.4f}")

# Кластеризація KMeans
def clusterize_signal(data_series, clusters=3):
    reshaped = data_series.reshape(-1, 1)
    km = KMeans(n_clusters=clusters, random_state=0)
    assigned = km.fit_predict(reshaped)
    return assigned, km

clust_labels_1, model_k1 = clusterize_signal(raw_data_1['Value'].values)
clust_labels_2, model_k2 = clusterize_signal(raw_data_2['Value'].values)

plt.figure(figsize=(14, 6))
for i, (data, labels, title) in enumerate([
    (raw_data_1, clust_labels_1, "Кластери (тихо)"),
    (raw_data_2, clust_labels_2, "Кластери (гучно)")
]):
    plt.subplot(1, 2, i+1)
    plt.scatter(data['Timestamp'], data['Value'], c=labels, cmap='plasma', s=10)
    plt.title(title)
    plt.xlabel("Час (мс)")
    plt.ylabel("Амплітуда")
plt.tight_layout()
plt.show()

# Лінійна регресія
def regression_fit(time_series, signal_series):
    model = LinearRegression()
    X = time_series.values.reshape(-1, 1)
    y = signal_series.values
    model.fit(X, y)
    y_pred = model.predict(X)
    return model, y_pred

linreg_1, pred_line_1 = regression_fit(raw_data_1['Timestamp'], raw_data_1['Value'])
linreg_2, pred_line_2 = regression_fit(raw_data_2['Timestamp'], raw_data_2['Value'])

plt.figure(figsize=(14, 6))
for i, (df, pred, title) in enumerate([
    (raw_data_1, pred_line_1, "Тренд (тихо)"),
    (raw_data_2, pred_line_2, "Тренд (гучно)")
]):
    plt.subplot(1, 2, i+1)
    plt.plot(df['Timestamp'], df['Value'], label='Справжній сигнал', alpha=0.5)
    plt.plot(df['Timestamp'], pred, label='Прогноз', color='crimson')
    plt.title(title)
    plt.xlabel("Час (мс)")
    plt.ylabel("Амплітуда")
    plt.legend()
plt.tight_layout()
plt.show()

# Прогнозування регресією
def regression_forecast(time_data, value_data):
    X = time_data.values.reshape(-1, 1)
    y = value_data.values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_tr, y_tr)
    predictions = model.predict(X_te)
    print(f"  MSE: {mean_squared_error(y_te, predictions):.4f}")
    print(f"  R² : {r2_score(y_te, predictions):.4f}")
    return model

print("\nПрогнозування (тихо):")
regression_forecast(raw_data_1['Timestamp'], raw_data_1['Value'])

print("\nПрогнозування (гучно):")
regression_forecast(raw_data_2['Timestamp'], raw_data_2['Value'])
