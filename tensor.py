import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# Загрузка данных MNIST
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

# Нормализация и изменение размерности данных
train_images, test_images = train_images / 255.0, test_images / 255.0
train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))

# Создание модели
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10)
])

# Компиляция модели
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Обучение модели
history = model.fit(train_images, train_labels, epochs=30, validation_data=(test_images, test_labels))

# Сохранение весов модели
model.save_weights('path_to_my.weights.h5')

# Оценка модели
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nТочность на тестовых данных:', test_acc)

