import json
import PIL
import os
import cv2
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.nasnet import NASNetLarge
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input

DATA_FOLDER = "data"
HTML_FOLDER = "data\\html"
JSON_FOLDER = "data\\json"
IMG_FOLDER = "data\\img"

IMAGE_SIZE = (128,128)
IMAGE_FULL_SIZE = (128,128, 3)
batchSize = 8

TYPE_LIST = ['G', 'R', 'W', 'L', 'P', 'M', 'D', 'F', 'Y', 'C', 'N']  # Expand if needed
TYPE_TO_IDX = {t: i for i, t in enumerate(TYPE_LIST)}

def prepareData():
    df = pd.read_csv("pokemon.csv")
    # prepare all images and labels as a numpy array
    allImages = []
    allLabels = []

    for ix, (name) in enumerate(df[['Name']].values):
        #print(name[0])
        imgFile = os.path.join(IMG_FOLDER, name[0] + '.jpg')
        jsonFile = os.path.join(JSON_FOLDER, name[0] + '.json')

        if not os.path.exists(imgFile) or not os.path.exists(jsonFile):
            #print("no data for: ", name[0])
            continue

        img = cv2.imread(imgFile)
        resized = cv2.resize(img, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
        allImages.append(resized)

        with open(jsonFile, 'r', encoding='utf-8') as f:
            jsonData = json.load(f)
            allLabels.append(jsonData)

    print(len(allImages))
    print(len(allLabels))

    with open("temp\\allPokemonLabels.json", "w", encoding="utf-8") as f:
        json.dump(allLabels, f, ensure_ascii=False, indent=2)

    allImages = np.array(allImages, dtype=np.uint8)
    allLabels = np.array(allLabels)

    np.save("temp\\allPokemonImages.npy", allImages)
    np.save("temp\\allPokemonLabels.npy", allLabels)
    
    '''
    from sklearn.preprocessing import LabelEncoder

    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(allLabels)

    # Save label mapping for inference
    class_names = label_encoder.classes_
    np.save("temp\\allDogClassNames.npy", class_names)

    # Use encoded labels (for model training)
    allLabels = np.array(encoded_labels)

    np.save("temp\\dogLabels.npy", allLabels)

    print("finish save te data ...")
    '''

def encode_label(data):
    vector = []

    # HP
    hp = int(data.get("hp", "0"))
    vector.append(hp)

    # Type (one-hot)
    type_vec = [0] * len(TYPE_LIST)
    t = data.get("type", "")
    if t in TYPE_TO_IDX:
        type_vec[TYPE_TO_IDX[t]] = 1
    vector.extend(type_vec)

    # Stage
    stage = int(data.get("stage", "0"))
    vector.append(stage)

    # Retreat
    retreat = int(data.get("retreat", "0"))
    vector.append(retreat)

    # Weak (use first char for simplicity)
    weak_vec = [0] * len(TYPE_LIST)
    w = data.get("weak", "")
    if w:
        weak_type = w[0]
        if weak_type in TYPE_TO_IDX:
            weak_vec[TYPE_TO_IDX[weak_type]] = 1
    vector.extend(weak_vec)

    # Resist (same as weak)
    resist_vec = [0] * len(TYPE_LIST)
    r = data.get("resist", "")
    if r:
        resist_type = r[0]
        if resist_type in TYPE_TO_IDX:
            resist_vec[TYPE_TO_IDX[resist_type]] = 1
    vector.extend(resist_vec)

    # Attacks — take up to 2
    for i in range(2):
        if i < len(data["attacks"]):
            atk = data["attacks"][i]
            damage = atk.get("damage", "0").replace("+", "").replace("-", "").strip()
            try:
                vector.append(int(damage))
            except:
                vector.append(0)
            cost_len = len(atk.get("cost", ""))
            vector.append(cost_len)
        else:
            vector.extend([0, 0])  # pad missing attack

    return np.array(vector, dtype=np.float32)

def train():
    allImages = np.load("temp\\allPokemonImages.npy")
    with open("temp\\allPokemonLabels.json", "r", encoding="utf-8") as f:
        allLabels = json.load(f)
    #allLabels = np.load("temp\\allPokemonLabels.npy")

    #print(allImages.shape)
    #print(allLabels.shape)

    numerical_labels = [encode_label(d) for d in allLabels]
    label_array = np.array(numerical_labels)

    allImagesForModel = allImages.astype(np.float32) / 255.0

    # create train and test data
    X_train, X_test, y_train, y_test = train_test_split(allImagesForModel, label_array, test_size=0.3)

    print("X_train, X_test, y_train, y_test ----> shape:")
    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)

    # build moodel
    input_shape = allImages.shape[1:]       # e.g. (128, 128, 3)
    output_dim = label_array.shape[1]         # label vector length

    task_prompt = "<s_pokemon-card>"

    model = Sequential([
        Input(shape=input_shape),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(output_dim)   # No activation for regression; use 'softmax' if classification
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])  # use 'categorical_crossentropy' if classifying

    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=30,
        batch_size=batchSize
    )

    model.save("pokemon_model.keras")
    

#prepareData()
train()