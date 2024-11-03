import numpy as np
import matplotlib.pyplot as plt

# Вектори даних
x = np.array([0.1, 0.3, 0.4, 0.6, 0.7])
y = np.array([3.2, 3, 1, 1.8, 1.9])

# Знаходимо коефіцієнти інтерполяційного полінома 4-го степеня
coefficients = np.polyfit(x, y, 4)
polynomial = np.poly1d(coefficients)

# Визначення значень функції в точках 0.2 та 0.5
y_02 = polynomial(0.2)
y_05 = polynomial(0.5)

# Виведемо значення функції у проміжних точках
print(f"Значення функції в точці x = 0.2: {y_02}")
print(f"Значення функції в точці x = 0.5: {y_05}")

# Побудова графіка інтерполюючого полінома
x_range = np.linspace(0.1, 0.7, 100)  # діапазон для графіку
y_range = polynomial(x_range)

plt.plot(x, y, 'o', label='Табличні точки')  # експериментальні точки
plt.plot(x_range, y_range, '-', label='Інтерполяційний поліном')  # графік полінома
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Інтерполяція поліномом 4-го степеня')
plt.grid(True)
plt.show()
