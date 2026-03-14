from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import os

app = Flask(__name__)

# Загружаем модель, если есть
model = None
if os.path.exists('my_model.keras'):
    model = tf.keras.models.load_model('my_model.keras')
    print("Модель загружена")


@app.route('/')
def index():
    # Просто показываем страницу
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Получаем числа из формы
    x1 = float(request.form['x1'])
    x2 = float(request.form['x2'])

    # Превращаем в массив для модели
    data = np.array([[x1, x2]])

    # Предсказание
    if model:
        result = model.predict(data)[0][0]
    else:
        # Заглушка, если модели нет
        result = (x1 + x2) / 2

    # Показываем результат на той же странице
    return render_template('index.html', prediction=result)


if __name__ == '__main__':
    app.run(debug=True)