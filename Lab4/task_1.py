import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Експериментальні дані
X = np.array([13.33, 21, 63.75, 20.87, 40.42, 30.27])
Y = np.array([10.48, 21.03, 23.02, 41.25, 27.16, 51.5])

# Визначаємо лінійну функцію для апроксимації
def linear_func(x, a, b):
    return a * x + b

# Застосовуємо метод найменших квадратів для знаходження параметрів a і b
params, _ = curve_fit(linear_func, X, Y)
a, b = params

# Створюємо графік
plt.scatter(X, Y, color='blue', label='Експериментальні дані')  # Експериментальні точки
plt.plot(X, linear_func(X, a, b), color='red', label=f'Апроксимація: Y = {a:.2f}X + {b:.2f}')  # Лінійна апроксимація

# Додаємо підписи та легенду
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.title('Експериментальні точки та апроксимуюча пряма')
plt.grid(True)
plt.show()
