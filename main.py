import tarfile
import os

"""
# Cesty k datasetu
tar_file_path = '/Users/tomashorak/image_classification/102flowers.tar'
extracted_dir = '/Users/tomashorak/Desktop/flowerapp'

# Rozbalení .tar souboru
with tarfile.open(tar_file_path, 'r') as tar:
    tar.extractall(path=extracted_dir)

print("Rozbalení dokončeno")

"""
import scipy.io

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, models, optimizers, preprocessing

import numpy as np
import tensorflow as tf

from keras import layers, models, optimizers, preprocessing, utils, saving
from keras import ops

import scipy.io



# Cesty k datasetu
images_dir = '/Users/tomashorak/image_classification/jpg'
labels_file = '/Users/tomashorak/image_classification/imagelabels.mat'
splits_file = '/Users/tomashorak/image_classification/setid.mat'

# Načtení popisků a rozdělení dat
labels = scipy.io.loadmat(labels_file)['labels'][0]
splits = scipy.io.loadmat(splits_file)

train_idx = splits['trnid'][0] - 1
val_idx = splits['valid'][0] - 1
test_idx = splits['tstid'][0] - 1

def load_image(file_path):
    img = tf.keras.preprocessing.image.load_img(file_path, target_size=(150, 150))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    return img

# Funkce pro načtení obrázků a popisků
def get_data(indices):
    images = []
    image_labels = []
    for idx in indices:
        file_path = os.path.join(images_dir, f'image_{idx+1:05d}.jpg')
        img = load_image(file_path)
        images.append(img)
        image_labels.append(labels[idx] - 1)
    return np.vstack(images), np.array(image_labels)

# Načtení trénovacích, validačních a testovacích dat
x_train, y_train = get_data(train_idx)
x_val, y_val = get_data(val_idx)
x_test, y_test = get_data(test_idx)

# omezení vzorku na 5
x_train_small, y_train_small = x_train, y_train

# Příprava dat pro TensorFlow

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator()
val_datagen = tf.keras.preprocessing.image.ImageDataGenerator()

train_generator = train_datagen.flow(x_train_small, tf.keras.utils.to_categorical(y_train_small, 102), batch_size=19)
val_generator = val_datagen.flow(x_val, tf.keras.utils.to_categorical(y_val, 102), batch_size=19)




# Definice modelu
model_save_path = '/Users/tomashorak/image_classification/model.keras'

# Podmíněné načtení modelu
if os.path.exists(model_save_path):
    # Načtení uloženého modelu
    model = saving.load_model(model_save_path)
    print("Načtení uloženého modelu.")
else:
    model = keras.Sequential([
    layers.Conv2D(32, (2, 2), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (2, 2), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (2, 2), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(102, activation='softmax')
])

# Kompilace modelu
model.compile(
    optimizer=optimizers.Adam(learning_rate= 0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

import math

# Trénování modelu
history = model.fit(
    train_generator,
    steps_per_epoch=20,
    validation_data=val_generator,
    validation_steps=10,
    epochs=300
)
model.save(model_save_path)

# Vyhodnocení modelu na validačních datech
validation_loss, validation_accuracy = model.evaluate(val_generator)
print(f'Validation Accuracy: {validation_accuracy * 100:.2f}%')



labels = {"0": "pink primrose", "1": "hard-leaved pocket orchid", "2": "canterbury bells", "3": "sweet pea", "4": "english marigold", "5": "tiger lily", "6": "moon orchid", "7": "bird of paradise", "8": "monkshood", "9": "globe thistle", "10": "snapdragon", "11": "colt's foot", "12": "king protea", "13": "spear thistle", "14": "yellow iris", "15": "globe-flower", "16": "purple coneflower", "17": "peruvian lily", "18": "balloon flower", "19": "giant white arum lily", "20": "fire lily", "21": "pincushion flower", "22": "fritillary", "23": "red ginger", "24": "grape hyacinth", "25": "corn poppy", "26": "prince of wales feathers", "27": "stemless gentian", "28": "artichoke", "29": "sweet william", "30": "carnation", "31": "garden phlox", "32": "love in the mist", "33": "mexican aster", "34": "alpine sea holly", "35": "ruby-lipped cattleya", "36": "cape flower", "37": "great masterwort", "38": "siam tulip", "39": "lenten rose", "40": "barbeton daisy", "41": "daffodil", "42": "sword lily", "43": "poinsettia", "44": "bolero deep blue", "45": "wallflower", "46": "marigold", "47": "buttercup", "48": "oxeye daisy", "49": "common dandelion", "50": "petunia", "51": "wild pansy", "52": "primula", "53": "sunflower", "54": "pelargonium", "55": "bishop of llandaff", "56": "gaura", "57": "geranium", "58": "orange dahlia", "59": "pink-yellow dahlia", "60": "cautleya spicata", "61": "japanese anemone", "62": "black-eyed susan", "63": "silverbush", "64": "californian poppy", "65": "osteospermum", "66": "spring crocus", "67": "bearded iris", "68": "windflower", "69": "tree poppy", "70": "gazania", "71": "azalea", "72": "water lily", "73": "rose", "74": "thorn apple", "75": "morning glory", "76": "passion flower", "77": "lotus", "78": "toad lily", "79": "anthurium", "80": "frangipani", "81": "clematis", "82": "hibiscus", "83": "columbine", "84": "desert-rose", "85": "tree mallow", "86": "magnolia", "87": "cyclamen", "88": "watercress", "89": "canna lily", "90": "hippeastrum", "91": "bee balm", "92": "ball moss", "93": "foxglove", "94": "bougainvillea", "95": "camellia", "96": "mallow", "97": "mexican petunia", "98": "bromelia", "99": "blanket flower", "100": "trumpet creeper", "101": "blackberry lily"}


# Funkce pro předpověď číselné třídy květiny
def predict_houseplant(img_path):
    img = utils.load_img(img_path, target_size=(150, 150))
    img_array = utils.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    return predicted_class




# Příklad použití
img_path = '/Users/tomashorak/image_classification/image_00001.jpg'
predicted_class = predict_houseplant(img_path)
print(f'Predicted class: {predicted_class}')
print(f'Predicted flower name: {labels[str(predicted_class)]}')



