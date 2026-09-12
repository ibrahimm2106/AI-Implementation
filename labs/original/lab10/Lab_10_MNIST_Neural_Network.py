import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, utils
from tensorflow.keras.datasets import mnist

(train_X, train_Y), (test_X, test_Y) = mnist.load_data()
train_X, test_X = train_X / 255.0, test_X / 255.0
train_Y_cat = utils.to_categorical(train_Y, 10)
test_Y_cat = utils.to_categorical(test_Y, 10)

model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax"),
])
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
history = model.fit(train_X, train_Y_cat, epochs=10, validation_split=0.33)
test_loss, test_accuracy = model.evaluate(test_X, test_Y_cat)
predictions = model.predict(test_X)
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Loss: {test_loss:.4f}")
print("Predicted Digit:", np.argmax(predictions[0]))
print("True Label:", test_Y[0])
