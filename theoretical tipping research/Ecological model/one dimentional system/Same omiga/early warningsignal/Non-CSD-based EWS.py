import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
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
        k = k0 + 2 * np.sin(w * t)
        r = 0.47 + 0.05 * np.sin(w * t)

        x_prev = x_values[i - 1]
        k1 = dt * (r * x_prev * (1 - x_prev / k) - (beta * x_prev ** 2) / (alpha ** 2 + x_prev ** 2))
        k2 = dt * (r * (x_prev + 0.5 * k1) * (1 - (x_prev + 0.5 * k1) / k) - (beta * (x_prev + 0.5 * k1) ** 2) / (alpha ** 2 + (x_prev + 0.5 * k1) ** 2))
        k3 = dt * (r * (x_prev + 0.5 * k2) * (1 - (x_prev + 0.5 * k2) / k) - (beta * (x_prev + 0.5 * k2) ** 2) / (alpha ** 2 + (x_prev + 0.5 * k2) ** 2))
        k4 = dt * (r * (x_prev + k3) * (1 - (x_prev + k3) / k) - (beta * x_prev ** 2) / (alpha ** 2 + x_prev ** 2))

        x_values[i] = x_prev + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return x_values

def calculate_window_statistics(window):
    return skew(window), kurtosis(window), np.mean(window)

def hurst_exponent(ts):
    N = len(ts)
    L = range(2, N // 2)
    R_S = []

    for l in L:
        sub_series = [ts[i:i + l] for i in range(N - l)]
        R = [np.max(s) - np.min(s) for s in sub_series]
        S = [np.std(s) for s in sub_series]
        R_S.append(np.mean(np.array(R) / np.array(S)))

    log_L = np.log(L)
    log_R_S = np.log(R_S)
    hurst = np.polyfit(log_L, log_R_S, 1)[0]
    return hurst

def calculate_moving_statistics(data, window_size):
    num_windows = len(data) - window_size + 1

    with Pool() as pool:
        results = pool.map(calculate_window_statistics, [data[i:i + window_size] for i in range(num_windows)])

    skewness, kurt, mean_flux = zip(*results)
    return np.array(skewness), np.array(kurt), np.array(mean_flux)

def calculate_moving_hurst(data, window_size):
    num_windows = len(data) - window_size + 1

    with Pool() as pool:
        hurst_values = pool.map(hurst_exponent, [data[i:i + window_size] for i in range(num_windows)])

    return np.array(hurst_values)

def plot_time_series(x_values, dt):
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(len(x_values)) * dt, x_values)
    plt.title('Time Series')
    plt.xlabel('Time (s)')
    plt.ylabel('Value')
    plt.grid()
    plt.show()

def plot_skewness(skewness, dt):
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(len(skewness)) * dt, skewness)
    plt.title('Skewness')
    plt.xlabel('Time (s)')
    plt.ylabel('Skewness')
    plt.grid()
    plt.show()

def plot_kurtosis(kurt, dt):
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(len(kurt)) * dt, kurt)
    plt.title('Kurtosis')
    plt.xlabel('Time (s)')
    plt.ylabel('Kurtosis')
    plt.grid()
    plt.show()

def plot_mean_flux(mean_flux, dt):
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(len(mean_flux)) * dt, mean_flux)
    plt.title('Mean Flux')
    plt.xlabel('Time (s)')
    plt.ylabel('Mean Flux')
    plt.grid()
    plt.show()

def plot_hurst(hurst_values, dt):
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(len(hurst_values)) * dt, hurst_values)
    plt.title('Hurst Exponent')
    plt.xlabel('Time (s)')
    plt.ylabel('Hurst Exponent')
    plt.grid()
    plt.show()

def main():
    x0 = 2.22  # 初始值
    x_values = rk4_vectorized(x0, dt, num_steps)

    window_size = 50
    skewness, kurt, mean_flux = calculate_moving_statistics(x_values, window_size)
    hurst_values = calculate_moving_hurst(x_values, window_size)

    plot_time_series(x_values, dt)
    plot_skewness(skewness, dt)
    plot_kurtosis(kurt, dt)
    plot_mean_flux(mean_flux, dt)
    plot_hurst(hurst_values, dt)

    for i, hurst in enumerate(hurst_values):
        print(f"Window {i + 1}: Hurst指数 = {hurst:.4f}")

if __name__ == '__main__':
    main()