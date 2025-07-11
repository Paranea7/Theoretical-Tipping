import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import detrend
from scipy.fft import fft
from multiprocessing import Pool

# 定义参数
k0 = 10
alpha = 1
beta = 1
dt = 0.01
time_end = 400
num_steps = int(time_end / dt)
w = 0.1


def rk4_vectorized(x0, dt, num_steps):
    x_values = np.zeros(num_steps)
    x_values[0] = x0

    for i in range(1, num_steps):
        t = i * dt

        # 根据时间t计算r和k
        k = k0
        r = 0.47 + 0.05 * np.sin(w * t)

        x_prev = x_values[i - 1]
        k1 = dt * (r * x_prev * (1 - x_prev / k) - (beta * x_prev ** 2) / (alpha ** 2 + x_prev ** 2))
        k2 = dt * (r * (x_prev + 0.5 * k1) * (1 - (x_prev + 0.5 * k1) / k) - (beta * (x_prev + 0.5 * k1) ** 2) / (
                    alpha ** 2 + (x_prev + 0.5 * k1) ** 2))
        k3 = dt * (r * (x_prev + 0.5 * k2) * (1 - (x_prev + 0.5 * k2) / k) - (beta * (x_prev + 0.5 * k2) ** 2) / (
                    alpha ** 2 + (x_prev + 0.5 * k2) ** 2))
        k4 = dt * (r * (x_prev + k3) * (1 - (x_prev + k3) / k) - (beta * x_prev ** 2) / (alpha ** 2 + x_prev ** 2))

        x_values[i] = x_prev + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return x_values


# 计算多个指标
def calculate_metrics(window):
    variance = np.var(window)
    mean = np.mean(window)
    c0 = np.sum((window - mean) ** 2)

    autocorr = 0
    if c0 != 0:
        lag = 1
        c_lag = np.sum((window[:-lag] - mean) * (window[lag:] - mean))
        autocorr = c_lag / (c0 / (len(window) - lag))

    return_rate = np.concatenate(([0], (window[1:] - window[:-1]) / window[:-1]))
    detrended = detrend(window)
    df_analysis = np.sqrt(np.mean(detrended ** 2))

    spectral_red = np.abs(fft(window))
    speed = np.std(window)

    return variance, autocorr, return_rate, df_analysis, spectral_red, speed


def sliding_window_analysis(series, window_size):
    n = len(series)
    results = {
        'variance': [],
        'autocorrelation': [],
        'return_rate': [],
        'detrended_fluctuation_analysis': [],
        'spectral_reddening': [],
        'speed_of_travelling_waves': [],
    }

    with Pool() as pool:
        for start in range(n - window_size + 1):
            end = start + window_size
            window = series[start:end]
            metrics = pool.apply(calculate_metrics, (window,))
            results['variance'].append(metrics[0])
            results['autocorrelation'].append(metrics[1])
            results['return_rate'].append(metrics[2])
            results['detrended_fluctuation_analysis'].append(metrics[3])
            results['spectral_reddening'].append(metrics[4])
            results['speed_of_travelling_waves'].append(metrics[5])

    return results


if __name__ == '__main__':
    initial_condition = 2.22
    data = rk4_vectorized(x0=initial_condition, dt=dt, num_steps=num_steps)

    plt.figure(figsize=(10, 5))
    time = np.linspace(0, time_end, num_steps)
    plt.plot(time, data, label='Time Series', color='blue')
    plt.title('Time Series over Time')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid()
    plt.show()

    window_size = 50
    results = sliding_window_analysis(data, window_size)

    time_windows = np.arange(len(results['variance'])) * dt

    plt.figure(figsize=(10, 5))
    plt.plot(time_windows, results['variance'], label='Variance', color='blue')
    plt.title('Variance over Time')
    plt.xlabel('Time')
    plt.ylabel('Variance')
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(time_windows, results['autocorrelation'], label='Autocorrelation (lag=1)', color='orange')
    plt.title('Autocorrelation (lag=1) over Time')
    plt.xlabel('Time')
    plt.ylabel('Autocorrelation (lag=1)')
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure(figsize=(10, 5))
    return_rate_mean = [np.mean(rate) for rate in results['return_rate']]
    plt.plot(time_windows, return_rate_mean, label='Mean Return Rate', color='green')
    plt.title('Mean Return Rate over Time')
    plt.xlabel('Time')
    plt.ylabel('Mean Return Rate')
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(time_windows, results['detrended_fluctuation_analysis'], label='Detrended Fluctuation Analysis', color='red')
    plt.title('Detrended Fluctuation Analysis over Time')
    plt.xlabel('Time')
    plt.ylabel('Detrended Fluctuation')
    plt.legend()
    plt.grid()
    plt.show()

    spectral_mean = [np.mean(sr) for sr in results['spectral_reddening']]
    plt.figure(figsize=(10, 5))
    plt.plot(time_windows, spectral_mean, label='Mean Spectral Reddening', color='purple')
    plt.title('Mean Spectral Reddening over Time')
    plt.xlabel('Time')
    plt.ylabel('Mean Spectral Reddening')
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(time_windows, results['speed_of_travelling_waves'], label='Speed of Travelling Waves', color='black')
    plt.title('Speed of Travelling Waves over Time')
    plt.xlabel('Time')
    plt.ylabel('Speed')
    plt.legend()
    plt.grid()
    plt.show()