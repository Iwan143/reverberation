import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

file = "Хлопок.wav"

fs, x = wavfile.read(file)
x = x.astype(float)
if x.ndim > 1:
    x = x.mean(axis=1)
x /= np.max(np.abs(x))
print(fs)

i0 = np.argmax(np.abs(x))
x = x[i0:]
t = np.arange(len(x)) / fs

E = np.sum(x**2)
exx = np.cumsum(x[::-1]**2)[::-1]
exx_db = 20 * np.log10(exx / exx[0])

idx_60 = np.where(exx_db <= -60)[0]
rt60 = t[idx_60[0]] if len(idx_60) else None

print(f"Энергия сигнала: {E:.6f}")
if rt60 is not None:
    print(f"RT60 : {rt60:.3f} с")
else:
    print("неправильно")

plt.figure(figsize=(10,4))
plt.plot(t, x)
plt.title("Сигнал после пика")
plt.xlabel("Время, с")
plt.ylabel("Амплитуда")
plt.grid()

plt.figure(figsize=(10,4))
plt.plot(t, exx_db, label="Огибающая")
plt.axhline(-60, color="r", linestyle="--", label="-60 dB")
if rt60 is not None:
    plt.axvline(rt60, color="g", linestyle="--", label=f"RT60 = {rt60:.3f} c")
plt.title("Энергетическое затухание")
plt.xlabel("Время, с")
plt.ylabel("Уровень дБ")
plt.legend()
plt.grid()
plt.show()