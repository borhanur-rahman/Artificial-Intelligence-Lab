import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential, metrics
from tensorflow.keras.layers import Dense, Input


def calculate_y(x):
    return 5 * x**2 + 10 * x - 2

def generate_samples(n):
    x = np.random.uniform(-20, 20, n)
    y = calculate_y(x)
    return x, y

def normalize(values):
    maximum = np.max(values)
    minimum = np.min(values)
    normed = (2 * (values - minimum) / (maximum - minimum)) - 1
    return normed, minimum, maximum

def denormalize(normed, minimum, maximum):
    return ((normed + 1) / 2) * (maximum - minimum) + minimum



n = 5000
x, y = generate_samples(n)

x_norm, x_min, x_max = normalize(x)
y_norm, y_min, y_max = normalize(y)


x_train, y_train = x_norm[:int(n*0.9)], y_norm[:int(n*0.9)]
x_val,   y_val   = x_norm[int(n*0.9):int(n*0.95)], y_norm[int(n*0.9):int(n*0.95)]
x_test,  y_test  = x_norm[int(n*0.95):], y_norm[int(n*0.95):]


train_dataset = (tf.data.Dataset.from_tensor_slices((x_train, y_train))
                 .shuffle(buffer_size=len(x_train))
                 .batch(32))
val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(32)


model = Sequential([
    Input(shape=(1,)),
    Dense(64, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1)
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss='mean_squared_error',
              metrics=[metrics.R2Score(name='r2')])



fitted_model = model.fit(train_dataset, epochs=20, validation_data=val_dataset)
history = fitted_model.history



plt.figure(figsize=(12,5))


plt.subplot(1,2,1)
plt.plot(history['loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
plt.title("Training vs Validation Loss")
plt.legend()


plt.subplot(1,2,2)
plt.plot(history['r2'], label='Train R²')
plt.plot(history['val_r2'], label='Val R²')
plt.xlabel("Epochs")
plt.ylabel("R² Score")
plt.title("Training vs Validation R²")
plt.legend()

plt.show()

y_pred_norm = model.predict(x_test).flatten()

y_pred = denormalize(y_pred_norm, y_min, y_max)
y_true = denormalize(y_test, y_min, y_max)
x_true  = denormalize(x_test, x_min, x_max)

plt.figure(figsize=(6,6))
plt.scatter(x_test, y_test, label="True (Norm)", alpha=0.5)
plt.scatter(x_test, y_pred_norm, label="Predicted (Norm)", alpha=0.5)
plt.xlabel("x (normalized)")
plt.ylabel("y (normalized)")
plt.title("True vs Predicted (Normalized)")
plt.legend()
plt.show()

plt.figure(figsize=(6,6))
plt.scatter(x_true, y_true, label="True", alpha=0.5)
plt.scatter(x_true, y_pred, label="Predicted", alpha=0.5)
plt.xlabel("x")
plt.ylabel("y")
plt.title("True vs Predicted (Original Scale)")
plt.legend()
plt.show()
plt.savefig("loss_plot.png", dpi=300, bbox_inches='tight')
plt.savefig("prediction_plot.png", dpi=300, bbox_inches='tight')

