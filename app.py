from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

# Загрузка модели
from tensorflow.keras.layers import Input

model = tf.keras.models.Sequential([
    Input(shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10)
])
model.load_weights('path_to_my.weights.h5')

'''def preprocess_image(image_data):
    """ Преобразование входящих данных изображения для предсказания """
    logging.info("Преобразование изображения")
    image = Image.open(io.BytesIO(image_data))
    image = image.resize((28, 28)).convert('L')
    image = np.array(image)
    image = image.reshape(1, 28, 28, 1)
    image = 1 - image.astype('float32') / 255.0
    return image'''
def preprocess_image(image_data):
    """ Преобразование входящих данных изображения для предсказания """
    image = Image.open(io.BytesIO(image_data))
    image = image.resize((28, 28)).convert('L')  # Преобразование в чёрно-белый формат
    image = np.array(image)
    if np.mean(image) > 127:  # Среднее значение больше 127 предполагает светлый фон
        image = 255 - image  # Инвертирование изображения
    image = image / 255.0  # Нормализация
    image = image.reshape(1, 28, 28, 1)
    return image


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/save', methods=['POST'])
def save_image():
    if 'image' not in request.files:
        logging.error("Изображение не предоставлено")
        return jsonify({'error': 'No image provided'}), 400
    image_file = request.files['image']
    image_bytes = image_file.read()
    image = Image.open(io.BytesIO(image_bytes))
    image.save("desired_directory/my_canvas_image.png")  # Сохраняем изображение
    return jsonify({'message': 'Image saved successfully'})


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        logging.error("Изображение не предоставлено")
        return jsonify({'error': 'No image provided'}), 400
    image_file = request.files['image']
    image_bytes = image_file.read()
    image = preprocess_image(image_bytes)
    logging.info("Получение предсказания")
    predictions = model.predict(image)
    predicted_digit = np.argmax(predictions)
    logging.info(f"Предсказанная цифра: {predicted_digit}")
    return jsonify({'prediction': int(predicted_digit)})

if __name__ == '__main__':
    app.run(debug=True)

