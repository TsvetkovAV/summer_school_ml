from fastai.vision import transform, image
from keras.applications import VGG16, imagenet_utils
from keras.preprocessing.image import img_to_array, load_img, array_to_img
from sklearn import linear_model, svm, metrics
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import os
import re

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

TRAIN_DIR = 'superbowllsh/train/'
TEST_DIR = 'superbowllsh/test/'
LABELS = ['cleaned', 'dirty']
BATCH_SIZE = 16
LE_PATH = os.path.sep.join(["output", "le.cpickle"])
BASE_CSV_PATH = "output"
# set the path to the serialized model after training
MODEL_PATH = 'output/model.cpickle'

def reshape_torch_to_keras(array):
    r = []
    for layer in array:
        layer = np.expand_dims(layer, 2)
        if not len(r):
            r = layer
            continue
        r = np.c_[r, layer]
    return r

def augment_image(path, size=(224, 224), x=10):
    img = image.open_image(path)
    augmented = [img.apply_tfms(tfms=tfms[0], size=size, padding_mode='border') for i in range(x)]
    orig = load_img(path, target_size=size)
    return [orig] + [array_to_img(aug_img.data, data_format='channels_first') for aug_img in augmented]


def train():
    # load the VGG16 network and initialize the label encoder
    model = VGG16(weights="imagenet", include_top=False)
    tfms = transform.get_transforms(do_flip=True, flip_vert=True, max_rotate=30., max_zoom=1.05)
    train_x = np.array([])
    train_y = np.array([])
    # Collect all train images
    train_imgs = np.array([])
    for y, label in enumerate(LABELS):
        img_paths = list(paths.list_images(TRAIN_DIR + label))
        for path in img_paths:
            # Augment original image 5x times + original image, all in size = (244, 244, 3)
            augmented_imgs = np.array([img_to_array(img) / 255 for img in augment_image(path, x=4)])
            augmented_y = np.ones(augmented_imgs.shape[0]) * y
            if train_imgs.shape == (0,):
                train_imgs = augmented_imgs
                train_y = augmented_y
                continue
            train_imgs = np.r_[train_imgs, augmented_imgs]
            train_y = np.r_[train_y, augmented_y]
    dataset_size = train_imgs.shape[0]
    for start in range(0, dataset_size, BATCH_SIZE):
        batchImages = train_imgs[start:np.min((start + BATCH_SIZE, dataset_size))]
        features = model.predict(batchImages, batch_size=np.min((BATCH_SIZE, len(batchImages))))
        features = features.reshape((features.shape[0], np.prod(features.shape[1:])))
        if train_x.shape == (0,):
            train_x = features
            continue
        train_x = np.r_[train_x, features]
    return train_x, train_y


def process_train(train_x, train_y):
    train_xy = np.c_[train_x, train_y]
    np.random.shuffle(train_xy)

    # train the model
    model = linear_model.LogisticRegression(solver="lbfgs", multi_class="auto", max_iter=(10 ** 7))
    model.fit(train_xy[:, :-1], train_xy[:, -1])

    # serialize the model to disk
    f = open(MODEL_PATH, "wb")
    f.write(pickle.dumps(model))
    f.close()


def test(stop):
    model = VGG16(weights="imagenet", include_top=False)
    img_paths = list(paths.list_images(TEST_DIR))
    img_paths = sorted(img_paths, key=lambda path: int(re.findall('{}([0-9]*).jpg'.format(TEST_DIR), path)[0]))[:10]
    dataset_size = len(img_paths)
    test_x = np.array([])
    ids = []
    for start in range(0, dataset_size, BATCH_SIZE):
        batchImages = np.array([])
        for path in img_paths[start:np.min((start + BATCH_SIZE, dataset_size))]:
            id = re.findall('{}([0-9]*).jpg'.format(TEST_DIR), path)[0]
            ids += [id]
            img = load_img(path, target_size=(224, 224))
            img = img_to_array(img) / 255
            if batchImages.shape == (0,):
                batchImages = np.array([img])
                continue
            batchImages = np.r_[batchImages, np.array([img])]
        features = model.predict(batchImages, batch_size=batchImages.shape[0])
        features = features.reshape((features.shape[0], np.prod(features.shape[1:])))
        if test_x.shape == (0,):
            test_x = features
            continue
        test_x = np.r_[test_x, features]

    # deserialize model of classifier
    model = pickle.load(open(MODEL_PATH, 'rb'))

    # evaluate the model
    preds = model.predict(test_x)
    df = pd.DataFrame(
        np.array([[str(i).zfill(4), LABELS[int(p)]] for i, p in enumerate(preds)]),
        columns=('id', 'label')
    )
    df.to_csv(BASE_CSV_PATH + "/test_prediction.csv", index=False)
    return df

df = test(10)
df.describe()
for i, label in df.values[:10]:
    img_paths = list(paths.list_images(TEST_DIR))
    img = image.open_image(TEST_DIR+i+'.jpg')
    img.show(title=label)