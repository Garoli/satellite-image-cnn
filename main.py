import numpy as np
import os
import cv2
import random
from shapely.geometry import Point, MultiPolygon, Polygon
from shapely.wkt import loads as wkt_loads
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.models import Sequential
from keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, Activation, Dropout
import PIL.Image as Image
import pandas as pd
import tifffile as tiff
import builtins
from threading import Thread

# Explications du fonctionnement interne de notre algorithme

## Création et chargement du dataset

### Chargement des images

#### load\_image_tiff(filename)
#fonction qui charge une image tiff dont le chemin est passé en paramètre.
#dimension du tableau retourné : Y * X * nb_canaux
def load_image_tiff(filename):
    img = tiff.imread(filename)
    img = np.rollaxis(img, 0, 3)
    return img

#### load_image\_n\_dim(filename, nb\_canaux\_voulus, image\_chargée\_en\_couleur=True)
#front\_end pour load\_image\_3dim et load\_image\_4dim
#fonction qui charge les données d'une image .jpg dont le chemin complet est passé en paramètre.
def load_image_n_dim(filename,n, color=True):
    if n == 3:
        return load_image_3dim(filename, color)
    elif n == 4:
        return load_image_4dim(filename, color)
    else:
        return ArithmeticError

def load_image_3dim(filename, color=True):
    if color or is_grey_scale(filename):
        return cv2.imread(filename, 1)
    else:
        img_grey = cv2.imread(filename, 0)
        return np.rollaxis(np.array([img_grey, img_grey, img_grey]), 0, 3)


def load_image_4dim(filename, color=True):
    if not is_grey_scale(filename) and color:
        img_color = cv2.imread(filename, 1)
        w, h, l = img_color.shape
        img_array = np.zeros(shape=(w, h, 4))
        for i in range(w):
            for j in range(h):
                img_array[i, j] = np.insert(img_color[i, j], 3, 0)
    else:
        img_grey = cv2.imread(filename, 0)
        w, h \
            = img_grey.shape
        img_array = np.zeros(shape=(w, h, 4))
        for i in range(w):
            for j in range(h):
                img_array[i, j, 3] = img_grey[i, j]
    return img_array

#### create\_datas\_percent(path, n, percent\_training=0.65, color=True)
#charge toutes les images du dataset en deux groupes distincts de manière aléatoire.
#Chaque label sera autant représenté que les autres dans les sets finaux.
#- le taux percent\_training représente la proportion d'images allant dans le set d'entrainement
#- le reste des images vont dans le set de test
def create_datas_percent(path, n, percent_training=0.65, color=True):
    i = 0
    train = []
    test = []
    min_nb_files = -1

    for lab in os.listdir(path):
        in_path = os.path.join(path, lab)
        result = len([filename for filename in os.listdir(in_path) if os.path.isfile(os.path.join(in_path, filename))])
        if min_nb_files == -1 or min_nb_files > result:
            min_nb_files = result

    nb_train = int(min_nb_files * percent_training)
    nb_test = int(min_nb_files * (1 - percent_training))

    for lab in os.listdir(path):
        in_path = os.path.join(path, lab)
        files_list = os.listdir(in_path)
        random.shuffle(files_list)
        for img in files_list[0:nb_train]:
            img_array = load_image_n_dim(os.path.join(in_path, img), n, color)
            train.append([img_array, i])
        for img in files_list[nb_train:nb_train + nb_test]:
            img_array = load_image_n_dim(os.path.join(in_path, img), n, color)
            test.append([img_array, i])
        i += 1
    return train, test

### Découpe des images et polygones

#### slice\_image(chevauchement\_en\_pixels, tableau\_image, taille\_sortie)
#fonction qui découpe une image de taille quelconque en un set d'images de taille normalisée.
#Elle prend en  paramètre :
#- Le chevauchement en pixel de la partie qui vas se répéter d'une image à l'autre.
#- L'image importée (sous la forme d'un  tableau)
#- La taille de l'image de sortie (carrée)
#La fonction parcourt l'image à découper. Pour chaque portion à extraire, elle réalise une copie de la 'sous-image' dans
#un set qui est retourné.
def slice_image(chevauch, image, s = 64):
    crop = []
    shape = np.array(image).shape
    stepx = range(int(shape[0] / (s - chevauch)))
    stepy = range(int(shape[1] / (s - chevauch)))
    for x in stepx:
        for y in stepy:
            tmpImg = image[(s - chevauch) * x: ((s - chevauch) * x) + s, (s - chevauch) * y: ((s - chevauch) * y) + s]
            if not np.max(tmpImg) == 0 and np.array(tmpImg).shape == (s,s,3):
                crop.append(tmpImg)
    return np.array(crop)


#### dichotomy\_polygon(i, imgs, img, bbox, poly, decoupe)
#fonction trèèès back-end, récursive.
#- i : le numéro de l'image dans imgs
#- imgs : le tableau final qu'on récupère après plusieurs appels de la fonction. contient une image détourée par label.
#- img : l'image extraite du fichier jpg ou tiff
#- bbox : la bounding box entourant le polygone
#- poly : le polygone à détourer
#- decoupe : booléen représentant s'il s'agit d'une découpe horizontale ou verticale
#Elle ajoute le contenu du polygone sur une image noire qui sera ajoutée à imgs.
def dichotomy_poly(i, imgs, img, bbox, poly, decoupe):
    if (bbox[2][0] - bbox[0][0]) < 10 or (bbox[2][1] - bbox[0][1]) < 10:
        for x in range(bbox[0][1], bbox[2][1]):
            for y in range(bbox[0][0], bbox[2][0]):
                if Point(y, x).within(poly):
                    imgs[i][x][y] = img[x][y]

    else:
        if Polygon(bbox).intersects(poly):
            if decoupe:
                ne = int((bbox[2][0] + bbox[0][0]) / 2)
                box1 = [bbox[0], (ne, bbox[0][1]), (ne, bbox[2][1]), bbox[3]]
                box2 = [(ne, bbox[0][1]), bbox[1], bbox[2], (ne, bbox[3][1])]
            else:
                ne = int((bbox[2][1] + bbox[0][1]) / 2)
                box1 = [bbox[0], bbox[1], (bbox[2][0], ne), (bbox[3][0], ne)]
                box2 = [(bbox[0][0], ne), (bbox[1][0], ne), bbox[2], bbox[3]]

            dichotomy_poly(i, imgs, img, box1, poly, not decoupe)
            dichotomy_poly(i, imgs, img, box2, poly, not decoupe)
        else:
            if Point(bbox[0][0], bbox[0][1]).within(poly):
                imgs[i][bbox[0][1]: bbox[2][1]][bbox[0][0]: bbox[2][0]] = img[bbox[0][1]: bbox[2][1]][bbox[0][0]: bbox[2][0]]


##### get\_polygon\_list(csv\_data, image\_id, label)
#fonction qui retourne un tableau contenant toutes les coordonnées des polygones de cette image, regroupées par label.
def get_polygon_list(csv_data, image_id, label):
    df_image = csv_data[csv_data.ImageId == image_id]
    multipoly_def = df_image[df_image.ClassType == label].MultipolygonWKT
    polygonList = []
    if len(multipoly_def) > 0:
        assert(len(multipoly_def) == 1)
        polygonList = wkt_loads(multipoly_def.values[0])
    return polygonList

##### bounding\_box(coords)
#fonction qui retourne un bounding box encadrant un polygone dont les coordonnées sont passées en paramètre.
def bounding_box(coords):
    min_x = 10000 # start with something much higher than expected min
    min_y = 10000
    max_x = -1 # start with something much lower than expected max
    max_y = -1
    for item in coords:
        if item[0] < min_x:
            min_x = item[0]
        if item[0] > max_x:
            max_x = item[0]
        if item[1] < min_y:
            min_y = item[1]
        if item[1] > max_y:
            max_y = item[1]
    return [(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)]

#### load\_image\_poly(path, file, nb\_classes=10)
#fonction qui charge une image tiff et le fichier csv. Elle isole les polygones de l'image par label.
#Elle retourne une image par label qui contient uniquement ses polygones représentatifs.
#Elle prend en paramètre le chemin et le nom du fichier .tiff et le nombre de labels du fichier .csv.
def load_image_poly(path, file, nb_classes = 10):
    CsvFile = pd.read_csv('train_wkt_v4.csv')
    filename = os.path.join(path,file)
    imageId = os.path.splitext(file)[0]
    img = load_image_tiff(filename)
    img = ((img - np.min(img))/(np.max(img) - np.min(img)))
    imgs = []
    for i in range(nb_classes):
        imgs.append(np.zeros(shape=img.shape, dtype=float))
        img = np.array(img)
        shape = img.shape
        polygons = get_polygon_list(CsvFile, imageId, i + 1)
        if not polygons == []:
            for poly in polygons :
                shapi = shape[0], shape[1]
                poly = _convert_coordinates_to_raster(np.array(list(poly.exterior.coords)), shapi, get_xmax_ymin(imageId))
                bbox = bounding_box(poly)
                if not len(img[bbox[0][1]:bbox[2][1]][bbox[0][0]:bbox[2][0]]) == 0:
                    poly = Polygon(poly)
                    dichotomy_poly(i, imgs, img, bbox, poly, True)
                    #t = tD.Decoupe(i, imgs, img, bbox, poly, True)
                    #t.start()
    return imgs

#### create\_data\_img(path, file, chevauchement = 48, size\_image\_out = 64, nbLabel = 10)
#fonction qui utilise load\_image\_poly et slice\_image.
#Elle génère toutes les images de taille normalisée contenant les polygones isolés par classe pour une image dont le
#chemin est assé en paramètre.
def create_data_img(path, file, output_path, chevauchement = 48, size_image_out = 64, nbLabel = 10):
    iLabel = 0
    for img in load_image_poly(path, file, nb_classes=nbLabel):
        if not np.max(img) == 0:
            x = 0
            for i in slice_image(chevauchement, img, size_image_out):
                name = os.path.splitext(file)[0] + str(x) + '.jpg'
                plt.imsave(os.path.join(output_path, str(iLabel), name), i)
                x += 1
        iLabel += 1


#### convert\_images\_to\_raster(), get\_xmax\_ymin()
#fonction utilitaires importées du code source de kaggle.
#Elles permettent de passer les coordonnées en pixels
def _convert_coordinates_to_raster(coords, img_size, xymax):
    # __author__ = visoft
    # https://www.kaggle.com/visoft/dstl-satellite-imagery-feature-detection/export-pixel-wise-mask
    Xmax, Ymax = xymax
    H, W = img_size
    W1 = 1.0 * W * W / (W + 1)
    H1 = 1.0 * H * H / (H + 1)
    xf = W1 / Xmax
    yf = H1 / Ymax
    coords[:, 1] *= yf
    coords[:, 0] *= xf
    coords_int = np.round(coords).astype(np.int32)
    return coords_int

#### image\_to\_y\_x\_pixels(image, y, x)
#copie une image dans une nouvelle image de dimension (y,x)
def image_to_y_x_pixels(image, y, x):
    y1, x1, l = image.shape
    new_image = np.zeros(dtype=int, shape=(y, x, l))
    for i in range(y1):
        for j in range(x1):
            new_image[i, j] = image[i, j]
    return new_image

def get_xmax_ymin(imageId):
    # __author__ = visoft
    grid_sizes_panda  = pd.read_csv('grid_sizes.csv', names=['ImageId', 'Xmax', 'Ymin'], skiprows=1)
    # https://www.kaggle.com/visoft/dstl-satellite-imagery-feature-detection/export-pixel-wise-mask
    xmax, ymin = grid_sizes_panda[grid_sizes_panda.ImageId == imageId].iloc[0, 1:].astype(float)
    return (xmax, ymin)

### Fonctions utilitaires

#### show\_image(filename)
#affiche l'image à l'écran
def show_image(filename):
    image = mpimg.imread(filename)
    plt.axis("off")
    plt.imshow(image)
    plt.show()


#### is\_grey\_scale(filename)
#retourne True si l'image dont l'adresse est passée en paramètre est en nuances de gris
def is_grey_scale(filename):
    img = cv2.imread(filename, 1)
    w, h, l = img.shape
    for i in range(w):
        for j in range(h):
            r, g, b = img[i, j]
            if r != g != b:
                return False
    return True

### Threading

#Nous avons d'utiliser des threads afin d'optimiser la génération d'images à partir de polygones.

#i\\_Thread est une variable globale qui décrit l'index de la prochaine image à découper
i_Thread = 0

#### class Load(Thread)
#cette petite classe appelle create\\_data\_img() tant que i\_Thread est inférieur au nombre d'images total (i.e qu'il
#reste des images à découper)
class Load(Thread):
    def __init__(self, imgs, path):
        Thread.__init__(self)
        self.imgs = imgs
        self.path = path

    def run(self):
        global i_Thread
        while i_Thread < len(self.imgs):
            tmp = i_Thread
            i_Thread += 1
            create_data_img(self.path, self.imgs[tmp])
        return


#### multi\_thread\_start(nb\start=4)
#crée n threads (ci-dessus) afin qu'ils découpent les images simultanément.
def multi_ThreadStart(nbThread=4):
    path = "D:\\data\\"
    imgs = os.listdir(path)
    for n in range(nbThread):
        t = Load(imgs, path)
        t.start()


## Deep Learning

### Chargement du modèle

#### create\_model\_entonnoir\_generique(dim\_input, nb\_output)
#crée un modèle adapté à la taille des entrées et sorties passées en paramètre.
#Cette fonction génère
#- n couches convolutionnelles, n étant le nombre de canaux de l'image
#- y couches denses, dont la taille décrémente à raison 0.5.
#- du dropout de taux 0.15
#puis compile le modèle et le retourne.
def create_model_entonoir_generique(dim_input, nb_output):
    model = Sequential()  # initialize neural network with keras
    y, x, n = dim_input

    model.add(Conv2D(32, (3,3), input_shape=(y, x, n), activation='relu', data_format="channels_last"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    for i in range(n-1):
        model.add(Dropout(0.15))
        model.add(Conv2D(32, (3, 3)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
    # make it flat
    model.add(Flatten())

    i = y
    while i > nb_output:
        model.add(Dropout(0.15))
        model.add(Dense(int(i), activation="relu"))
        i /= 2

    model.add(Dense(nb_output, activation='sigmoid'))

    # makes the neural networks with all layers
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])
    return model

#### create\_model\_entonnoir\_n\_dim(n) !!! Fonction obsolète
#ancêtre de create\_model\_entonnoir\_generique() uniquement adaptée à une taille de 64 \* 64 \* n
def create_model_entonoir_n_dim(n):
    IMG_SIZE = 64
    model = Sequential()  # initialize neural network with keras

    for i in range(n):
        model.add(Conv2D(IMG_SIZE, (3, 3), input_shape=(IMG_SIZE, IMG_SIZE, n), activation='relu', data_format="channels_last"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))

    # make it flat
    model.add(Flatten())
    model.add(Dense(IMG_SIZE, activation="relu"))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(16, activation="relu"))
    model.add(Dense(8, activation="relu"))
    model.add(Dense(5, activation='sigmoid'))

    # makes the neural networks with all layers
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])
    return model


### Apprentissage

#### main\_generique(dim\_input, nb\_sorties, path, epoch=1)
#crée un environnement de travail appelant l'ensemble des fonctions nécessaires à l'éxecution correcte de l'algorithme.
#- dim\_input : forme des images
#- nb\_sorties : nombre de labels
#- path : chemin vers le dataset
#- epoch : nombre de répétition d'apprentissage

def main_generique(dim_input, nb_sortie, path, epochs = 1):
    model = create_model_entonoir_generique(dim_input, nb_sortie)
    data_training, data_test = create_datas_percent(path, dim_input[2], color=False)
    random.shuffle(data_training)
    nb_imgcharge = 0
    nb_imgchargemax = 1000
    X = []
    y = np.zeros(shape=(nb_imgchargemax, nb_sortie))
    i = 0
    for data, label in data_training:
        img = image_to_y_x_pixels(np.array(data), dim_input[0], dim_input[1])
        X.append(img)
        y[nb_imgcharge][label] = 1
        i += 1
        nb_imgcharge+=1
        if i == len(data_training):
            break
        if nb_imgcharge >= nb_imgchargemax:
            nb_imgcharge = 0
            X = np.array(X)
            model.fit(X, y, steps_per_epoch=1000, epochs=epochs, shuffle=True)  # runs the algorithme
            accuracy_on_test(data_test, model, dim_input)
            X = []
            nb_imgcharge = 0
            if len(data_training)-i >= nb_imgchargemax:
                y = np.zeros(shape=(nb_imgchargemax, nb_sortie))
            else :
                y = np.zeros(shape=(len(data_training)-i, nb_sortie))

    X = np.array(X)

    model.fit(X, y, steps_per_epoch=1000, epochs=epochs, shuffle=True)  # runs the algorithm
    save_model(model)
    accuracy_on_test(data_test, model, dim_input)
    return model

##### main\_n\_dim(n, path) !!! Fonction obsolète
#ancêtre de la fonction main\_generique uniquement adaptée à une taille de 64 \* 64 \* n
def main_n_dim(n, path):
    IMG_SIZE = 64
    data_training, data_test = create_datas_percent(path, n, color=False)
    random.shuffle(data_training)
    X = []
    y = np.zeros(shape=(len(data_training), 5))
    i = 0
    for data, label in data_training:
        if np.array(data).shape == (IMG_SIZE, IMG_SIZE, n):
            X.append(np.array(data).reshape(IMG_SIZE, IMG_SIZE, n))
            y[i][label] = 1
            i += 1
            if i == len(data_training):
                break
    X = np.array(X)
    model = create_model_entonoir_n_dim(n)
    model.compile(loss="binary_crossentropy", optimizer="adam",
                  metrics=['accuracy'])  # makes the neural networks with all layers

    model.fit(X, y, steps_per_epoch=1000, epochs=1, shuffle=True)  # runs the algorithm
    save_model(model)
    accuracy_on_test(data_test, model)


### Sauvegarde et test du modèle

#### save\_model(model)
#enregistre le .json et le .h5 représentant le modèle.
def save_model(model):
    js = model.to_json()
    with open("model.json", 'w') as json_file:
        json_file.write(js)
    model.save_weights("model.h5")

### Prédiction

#### accuracy\_on\_test(data\_test, model, dim\_input)
#calcule la précision du modèle pré-entrainé en prédisant les labels des données de test
#- data\_test : les données d'image de test
#- model : le modèle
#- dim\_input : les dimensions des images
def accuracy_on_test(data_test, model, dim_input):
    n = t = 0
    for img, label in data_test:
        img = image_to_y_x_pixels(img, dim_input[0], dim_input[1])
        w, h, c = img.shape
        img = np.reshape(img, [1, w, h, c])
        if model.predict_classes(img) == label:
            t += 1
        n += 1
    print("accuracy : ", (t/n)*100)

#### predict()
#prédit le label auquel appartient une image à l'aide du modèle pré-entrainé


main_generique((64, 64, 3), 7, "d:\\Users\\Penjenaire56\\IMG\\")