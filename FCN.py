# -*- coding: utf-8 -*-
#Реализация слабой локализации с помощью скользящего окна.ipynb

import random
import imageio
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
print('TensorFlow version:', tf.__version__)

# Подготовка датасета
(train_x, train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()
train_x = train_x.reshape(-1, 28, 28, 1).astype(np.float32) / 255.
test_x = test_x.reshape(-1, 28, 28, 1).astype(np.float32) / 255.
np.random.seed(10)
from pathlib import Path
path = Path("model_1")
path.mkdir(exist_ok=True)# создаем папку на диске
cpt_filename = "checkpoint.hdf5"
cpt_path = str(path / cpt_filename)
### Визуализация нескольких образцов из MNIST
def plot_some_samples(some_samples):
    fig = plt.figure(figsize=(10, 6))
    for j in range(some_samples.shape[0]):
        ax = fig.add_subplot(4, 8, j+1)
        ax.imshow(some_samples[j,:,:,0], cmap='gray')
        plt.xticks([]), plt.yticks([])
    plt.show()
plot_some_samples(train_x[:32, ...])

### Аугментация: добавление образцов класса "фон"
bg_samples = 100000 # Количество обучающих образцов из класса "фон"
bg_train_x = np.zeros((bg_samples, 28, 28, 1), dtype=np.float32)
bg_train_y = np.ones((bg_samples,), dtype=np.int32)*10
src_idxs = random.choices(range(train_x.shape[0]), k=bg_samples)
sh = train_x.shape[1]
sw = train_x.shape[2]
for i in range(bg_samples):
    dh = random.randint(sh//4, 3*sh//4) * random.choice([-1, 1])
    dw = random.randint(sw//4, 3*sw//4) * random.choice([-1, 1])

    sample = train_x[src_idxs[i], ...]

    bg_train_x[i, max(-dh,0):min(sh-dh,sh), max(-dw,0):min(sw-dw, sw), :] = \
        sample[max( dh,0):min(sh+dh,sh), max( dw,0):min(sw+dw, sw), :]
plot_some_samples(bg_train_x[:32, ...])

### Аугментация: добавление образцов исходных классов цифр
sh_samples = 50000 # Количество дополнительно сгенерированных образов для цифр (со смещениями)
sh_train_x = np.zeros((sh_samples, 28, 28, 1), dtype=np.float32)
sh_train_y = np.zeros((sh_samples,), dtype=np.int32)
src_idxs = random.sample(range(train_x.shape[0]), sh_samples)
sh = train_x.shape[1]
sw = train_x.shape[2]
for i in range(sh_samples):
    dh = random.randint(0, sh//4) * random.choice([-1, 1])
    dw = random.randint(0, sw//4) * random.choice([-1, 1])

    sample = train_x[src_idxs[i], ...]

    sh_train_x[i, max(-dh,0):min(sh-dh,sh), max(-dw,0):min(sw-dw, sw), :] = \
        sample[max( dh,0):min(sh+dh,sh), max( dw,0):min(sw+dw, sw), :]
    sh_train_y[i] = train_y[src_idxs[i]]
plot_some_samples(sh_train_x[:32, ...])

### Объединение исходного датасета MNSIT и двух новых сгенерированных
train_x = np.concatenate((train_x, bg_train_x, sh_train_x), axis=0)
train_y = np.concatenate((train_y, bg_train_y, sh_train_y), axis=0)

# Обучение классификатора
NUM_CLASSES = 11
NUM_EPOCHS = 3
BATCH_SIZE = 64

# Классификационная модель
def get_compiled_model():
    """
    Функция возвращает скомпилированную модель для бинарной классификации
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (5, 5), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2), (2, 2), padding='same'),
        tf.keras.layers.Conv2D(64, (5, 5), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2), (2, 2), padding='same'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(NUM_CLASSES, activation='softmax'),
    ])
    # Подготовка модели к обучению
    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
    return model

# """### Обучение модели"""
model = get_compiled_model() # определим случайно инициализированную модель
# checkpoint = tf.keras.callbacks.ModelCheckpoint(cpt_path, monitor='loss', verbose=1, save_best_only=True, mode='min')
# model.fit(train_x, train_y, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, verbose=1, callbacks=[checkpoint])
model = tf.keras.models.load_model("model_1/checkpoint.hdf5")
"""### Оценка качества классификационной модели"""
loss, acc = model.evaluate(test_x, test_y)
print(f"Accuracy of restored model {acc*100 :.2f}%")
# Слабая локализация с помощью скользящего окна

if True: # Сгенерировать случайное изображение

    img = np.zeros((100, 130, 1), dtype=np.float32)
    def add_digit(img, digit):
        ofs = (random.randint(0, img.shape[0]-digit.shape[0]),
               random.randint(0, img.shape[1]-digit.shape[1]))
        img[ofs[0]:ofs[0]+digit.shape[0], ofs[1]:ofs[1]+digit.shape[1], :] += digit
        img = np.clip(img, 0.0, 1.0)
        return img
    for _ in range(3):
        digit = test_x[random.randint(0, test_x.shape[0])]
        img = add_digit(img, digit)

else: # Загрузить готовое изображение 'digits.png'

    INPUT_IMAGE_FPTAH = 'digits.png'
    img = imageio.imread(INPUT_IMAGE_FPTAH, pilmode="RGB")
    img = img.astype(np.float32)/255.
    img = np.mean(img, axis=2, keepdims=True)

# Превращение входной картинки в RGB
# (для визуализации и последующего смешивания с цветной тепловой картой)
img_clr = np.tile(img, (1, 1, 3))
_=plt.imshow(img_clr)
plt.show()
# Подготовка буфера для тепловых карт

inp_shape = (train_x.shape[1], train_x.shape[2]) # размер входа для классификатора

heatmaps = np.zeros((
    img.shape[0] - inp_shape[0] + 1,
    img.shape[1] - inp_shape[1] + 1,
    NUM_CLASSES))

# Запуск классификатора в режиме скользящего окна
for i in range(heatmaps.shape[0]):
    for j in range(heatmaps.shape[1]):
        window = img[i:i+inp_shape[0], j:j+inp_shape[1], :]
        heatmaps[i,j,:] = model.predict(window[None, ...])[0, ...]

diff = (img.shape[0]-heatmaps.shape[0],
        img.shape[1]-heatmaps.shape[1],)

heatmaps = np.pad(heatmaps, (
    (diff[0]//2,diff[0]-diff[0]//2),
    (diff[1]//2,diff[1]-diff[1]//2),
    (0, 0)
), 'constant')
# Визуализация тепловых карт для разных классов

for clss in range(11):
    heatmap = heatmaps[..., clss]
    heatmap_clr = plt.get_cmap('jet')(heatmap)[..., :3]
    print('Heatmap for class:', clss)
    plt.imshow(img_clr*0.5 + heatmap_clr*0.5)
    plt.show()