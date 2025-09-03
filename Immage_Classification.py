from tensorflow.keras.models import Sequential  
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np

(x_train, y_train), (x_test, y_test) = mnist.load_data()



x_train = x_train / 255.0
x_test = x_test / 255.0

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Split test data into validation and new test sets (50% each)
val_size = len(x_test) // 2
x_val, x_test_new = x_test[:val_size], x_test[val_size:]
y_val, y_test_new = y_test[:val_size], y_test[val_size:]

model = Sequential([
    Flatten(input_shape=(28,28)),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


history = model.fit(x_train, y_train, epochs=10, batch_size=64,
                    validation_data=(x_val, y_val))

loss, acc = model.evaluate(x_test_new, y_test_new)
print(f"Test Accuracy: {acc*100:.2f}%")

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Number of Epoch')
plt.ylabel('Accuracy in Percentage')
plt.legend()
plt.show()



# Predict on the first test image
sample_image = x_test_new[0].reshape(1, 28, 28)  # reshape for the model
prediction = model.predict(sample_image)
predicted_label = np.argmax(prediction)

plt.imshow(x_test_new[0], cmap='gray')
plt.title(f"Predicted Label: {predicted_label}")
plt.show()

print(f"Model prediction probabilities: {prediction}")
