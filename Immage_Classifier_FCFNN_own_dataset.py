from google.colab import files
uploaded = files.upload()




import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.models import Model


data = np.load("mnist_custom.npz")

X_train = data['trainX']
X_test  = data['testX']
y_train = data['trainY']
y_test  = data['testY']

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)


X_train = X_train.astype("float32") / 255.0
X_test  = X_test.astype("float32") / 255.0


inputs = Input((28,28,1))
x = Flatten()(inputs)
x = Dense(256, activation="relu")(x)
x = Dense(128, activation="relu")(x)
x = Dense(64, activation="relu")(x)
x = Dense(32, activation="relu")(x)
outputs = Dense(10, activation="softmax")(x)

model = Model(inputs, outputs)


model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test, y_test)
)



sample_index = 0
sample_image = X_test[sample_index]
sample_label = y_test[sample_index]

sample_input = np.expand_dims(sample_image, axis=0)
pred = model.predict(sample_input)
pred_class = np.argmax(pred, axis=1)[0]

plt.imshow(sample_image.squeeze(), cmap="gray")
plt.title(f"True: {sample_label}, Predicted: {pred_class}")
plt.show()


num_samples = 10
plt.figure(figsize=(12,4))

for i in range(num_samples):
    img = X_test[i]
    true_label = y_test[i]

    pred = model.predict(np.expand_dims(img, axis=0))
    pred_class = np.argmax(pred, axis=1)[0]

    plt.subplot(2,5,i+1)
    plt.imshow(img.squeeze(), cmap="gray")
    plt.title(f"T:{true_label}, P:{pred_class}")
    plt.axis("off")

plt.tight_layout()
plt.show()
