import os

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential

tf.keras.utils.set_random_seed(17)

class_names = ['История', 'Экономика', 'Математика', 'Астрономия', 'Экология']

data = {
    'text': [
        'Историк изучает летопись о торговле средневекового города.',
        'Музей хранит исторические документы о строительстве старой крепости.',
        'Историк сравнивает архивные документы о жизни купеческих семей.',
        'Летопись описывает путешествие послов в соседнее государство.',
        'Музей открыл выставку о быте жителей средневекового города.',
        'Историк восстанавливает историю крепости по старинным картам.',
        'Экономист объясняет, как инфляция меняет стоимость товаров.',
        'Рынок реагирует на увеличение спроса ростом цен.',
        'Экономист рассчитывает расходы предприятия и прибыль от продажи товаров.',
        'Инфляция снижает покупательную способность денежных накоплений.',
        'Рынок труда меняется при открытии новых предприятий.',
        'Экономист оценивает влияние налогов на прибыль предприятия.',
        'Математик решает уравнение с одной неизвестной величиной.',
        'Геометрия позволяет вычислить площадь и периметр треугольника.',
        'Математик проверяет формулу для суммы членов прогрессии.',
        'Уравнение задает зависимость между двумя переменными.',
        'Геометрия изучает свойства окружности и прямоугольного треугольника.',
        'Математик использует формулу для вычисления объема цилиндра.',
        'Астроном направляет телескоп на далекую звезду.',
        'Планета движется по орбите вокруг своей звезды.',
        'Астроном изучает снимки галактики, полученные через телескоп.',
        'Телескоп помогает наблюдать поверхность Луны и кольца Сатурна.',
        'Планета отражает свет звезды и меняет положение на небе.',
        'Астроном рассчитывает время следующего солнечного затмения.',
        'Эколог исследует загрязнение воды в лесном озере.',
        'Заповедник защищает редкие виды растений и животных.',
        'Эколог оценивает влияние отходов на состояние почвы.',
        'Загрязнение воздуха ухудшает состояние растений в городе.',
        'Заповедник сохраняет лес и места обитания перелетных птиц.',
        'Эколог предлагает сократить отходы и организовать переработку пластика.',
    ],
    'label': np.repeat(np.arange(5), 6),
}
df = pd.DataFrame(data)

train_texts, test_texts, y_train, y_test = train_test_split(
    df['text'], df['label'].to_numpy(),
    test_size=1 / 3, random_state=17, stratify=df['label'],
)

vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(train_texts).toarray().astype('float32')
X_test = vectorizer.transform(test_texts).toarray().astype('float32')

print('Всего предложений:', len(df))
print('Количество классов:', len(class_names))
print('Размеры матриц: обучение', X_train.shape, ', тест', X_test.shape)
print('Объектов каждого класса в обучении:', np.bincount(y_train))
print('Объектов каждого класса в тесте:', np.bincount(y_test))

model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(len(class_names), activation='softmax'),
])
model.compile(
    optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'],
)
history = model.fit(X_train, y_train, epochs=100, batch_size=4, verbose=0)

train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f'\nТочность на обучении: {train_acc:.2%}')
print(f'Точность на тесте: {test_acc:.2%}')

test_predictions = model.predict(X_test, verbose=0)
print('\nПроверка отложенных предложений:')
for text, correct, prediction in zip(test_texts, y_test, test_predictions):
    predicted = int(np.argmax(prediction))
    print(f'{text}\n  Правильно: {class_names[correct]}; модель: {class_names[predicted]}')

new_sentences = [
    'Историк изучает архивные документы о восстановлении крепости.',
    'Экономист сравнивает прибыль предприятия и стоимость товаров.',
    'Математик использует уравнение для вычисления неизвестной величины.',
    'Астроном наблюдает звезду и ее планету через телескоп.',
    'Эколог изучает загрязнение почвы и предлагает переработку отходов.',
]
new_vectors = vectorizer.transform(new_sentences).toarray().astype('float32')
predictions = model.predict(new_vectors, verbose=0)

print('\nКлассификация новых предложений:')
for sentence, prediction in zip(new_sentences, predictions):
    predicted_class = int(np.argmax(prediction))
    print(f'\nТекст: {sentence}')
    print('Тема:', class_names[predicted_class])
    print(f'Оценка Softmax: {prediction[predicted_class]:.2%}')

print('\nНабор небольшой: результаты демонстрируют работу модели на учебных текстах.')
