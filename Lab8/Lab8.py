import tensorflow as tf
import numpy as np

n_samples, batch_size, num_steps = 1000, 100, 20000

X_data = np.random.uniform(0, 1, (n_samples, 1)).astype(np.float32)
y_data = 2 * X_data + 1 + np.random.normal(0, 0.2, (n_samples, 1)).astype(np.float32)

k = tf.Variable(tf.random.normal((1, 1), stddev=0.01, dtype=tf.float32), name='slope')
b = tf.Variable(tf.zeros((1,), dtype=tf.float32), name='bias')

def model(X):
    return tf.matmul(X, k) + b

def loss_fn(y_true, y_pred):
    return tf.reduce_sum(tf.square(y_true - y_pred))

optimizer = tf.optimizers.SGD(learning_rate=0.001)

for step in range(num_steps):
    indices = np.random.choice(n_samples, batch_size)
    X_batch, y_batch = X_data[indices], y_data[indices]

    with tf.GradientTape() as tape:
        y_pred = model(X_batch)
        loss = loss_fn(y_batch, y_pred)

    gradients = tape.gradient(loss, [k, b])
    optimizer.apply_gradients(zip(gradients, [k, b]))

    if (step + 1) % 100 == 0:
        print(f"Step {step + 1}: loss = {loss.numpy()}, k = {k.numpy()[0][0]}, b = {b.numpy()[0]}")
