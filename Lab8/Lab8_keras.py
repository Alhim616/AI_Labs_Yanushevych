import tensorflow as tf
import numpy as np

n_samples = 1000
X_data = np.random.uniform(0, 1, (n_samples, 1)).astype(np.float32)
y_data = 2 * X_data + 1 + np.random.normal(0, 0.2, (n_samples, 1)).astype(np.float32)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_dim=1, use_bias=True, kernel_initializer='random_normal', bias_initializer='zeros')
])

model.compile(optimizer=tf.optimizers.SGD(learning_rate=0.001), loss='mean_squared_error')

model.fit(X_data, y_data, epochs=20000, batch_size=100, verbose=100)

print(f"Final parameters: k = {model.weights[0].numpy()[0][0]}, b = {model.weights[1].numpy()}")
