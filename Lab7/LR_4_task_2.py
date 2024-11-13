from sklearn.svm import SVC
from sklearn.metrics import pairwise_distances_argmin
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

iris = load_iris()
X = iris['data']  # Завантаження даних із набору "iris"
y = iris['target']  # Завантаження цільових міток із набору "iris"

# Створення об'єкта KMeans з неправильними аргументами (потребує виправлення)
kmeans = KMeans(n_clusters=8, init='k-means++', n_init=10, max_iter=300, tol=0.0001, 
                verbose=0, random_state=None, algorithm='auto')

kmeans = KMeans(n_clusters=5)  # Ініціалізація KMeans з 5 кластерами

kmeans.fit(X)  # Навчання моделі KMeans на даних X

y_kmeans = kmeans.predict(X)  # Прогнозування кластерів для X

# Візуалізація даних із кольоровими мітками кластерів
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_  # Отримання центрів кластерів
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)  # Візуалізація центрів кластерів

# Визначення функції для пошуку кластерів
def find_clusters(X, n_clusters, rseed=2):
    rng = np.random.RandomState(rseed)  # Створення об'єкта генератора випадкових чисел
    i = rng.permutation(X.shape[0])[:n_clusters]  # Випадковий вибір початкових центрів
    centers = X[i]  # Ініціалізація центрів кластерів
    while True:
        # Присвоєння кожній точці найближчого центру кластера
        labels = pairwise_distances_argmin(X, centers)

        # Обчислення нових центрів як середнє значення точок у кожному кластері
        new_centers = np.array([X[labels == i].mean(0) for i in range(n_clusters)])

        # Перевірка, чи змінилися центри; якщо ні, припинити цикл
        if np.all(centers == new_centers):
            break
        centers = new_centers  # Оновлення центрів
    return centers, labels  # Повернення центрів і міток

centers, labels = find_clusters(X, 3)  # Виклик функції для пошуку 3 кластерів
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')  # Візуалізація з новими мітками
plt.show()

centers, labels = find_clusters(X, 3, rseed=0)  # Виклик функції з іншим значенням rseed
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')  # Візуалізація результатів
plt.show()

# Прогнозування кластерів за допомогою KMeans з 3 кластерами
labels = KMeans(3, random_state=0).fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')  # Візуалізація кластерів
plt.show()
