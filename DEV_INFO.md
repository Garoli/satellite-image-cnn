# Explications du fonctionnement interne de notre algorithme

## Création et chargement du dataset

### Chargement des images

#### load\_image\_tiff(filename)
fonction qui charge une image tiff dont le chemin est passé en paramètre.
dimension du tableau retourné : Y \* X \* nb\_canaux

#### load\_image\_n\_dim(filename, nb\_canaux\_voulus, image\_chargée\_en\_couleur=True)
front\_end pour load\_image\_3dim et load\_image\_4dim
fonction qui charge les données d'une image .jpg dont le chemin complet est passé en paramètre.

#### create\_datas\_percent(path, n, percent\_training=0.65, color=True)
charge toutes les images du dataset en deux groupes distincts de manière aléatoire.
Chaque label sera autant représenté que les autres dans les sets finaux.
- le taux percent\_training représente la proportion d'images allant dans le set d'entrainement
- le reste des images vont dans le set de test

### Découpe des images et polygones

#### slice\_image(chevauchement\_en\_pixels, tableau\_image, taille\_sortie)
fonction qui découpe une image de taille quelconque en un set d'images de taille normalisée.
Elle prend en  paramètre :
- Le chevauchement en pixel de la partie qui vas se répéter d'une image à l'autre.
- L'image importée (sous la forme d'un  tableau)
- La taille de l'image de sortie (carrée)

La fonction parcourt l'image à découper. Pour chaque portion à extraire, elle réalise une copie de la 'sous-image' dans
un set qui est retourné.

#### dichotomy\_polygon(i, imgs, img, bbox, poly, decoupe)
fonction trèèès back-end, récursive.
- i : le numéro de l'image dans imgs
- imgs : le tableau final qu'on récupère après plusieurs appels de la fonction. contient une image détourée par label.
- img : l'image extraite du fichier jpg ou tiff
- bbox : la bounding box entourant le polygone
- poly : le polygone à détourer
- decoupe : booléen représentant s'il s'agit d'une découpe horizontale ou verticale
Elle ajoute le contenu du polygone sur une image noire qui sera ajoutée à imgs. 


##### get\_polygon\_list(csv\_data, image\_id, label)
fonction qui retourne un tableau contenant toutes les coordonnées des polygones de cette image, regroupées par label.

##### bounding\_box(coords)
fonction qui retourne un bounding box encadrant un polygone dont les coordonnées sont passées en paramètre.

#### load\_image\_poly(path, file, nb\_classes=10)
fonction qui charge une image tiff et le fichier csv. Elle isole les polygones de l'image par label.
Elle retourne une image par label qui contient uniquement ses polygones représentatifs.
Elle prend en paramètre le chemin et le nom du fichier .tiff et le nombre de labels du fichier .csv.

#### create\_data\_img(path, file, chevauchement = 48, size\_image\_out = 64, nbLabel = 10)
fonction qui utilise load\_image\_poly et slice\_image.
Elle génère toutes les images de taille normalisée contenant les polygones isolés par classe pour une image dont le
chemin est assé en paramètre.

#### convert\_images\_to\_raster(), get\_xmax\_ymin()
fonction utilitaires importées du code source de kaggle.
Elles permettent de passer les coordonnées en pixels

#### image\_to\_y\_x\_pixels(image, y, x)
copie une image dans une nouvelle image de dimension (y,x)

### Fonctions utilitaires

#### show\_image(filename)
affiche l'image à l'écran

#### is\_grey\_scale(filename)
retourne True si l'image dont l'adresse est passée en paramètre est en nuances de gris

### Threading

Nous avons d'utiliser des threads afin d'optimiser la génération d'images à partir de polygones.

i\\_Thread est une variable globale qui décrit l'index de la prochaine image à découper

#### class Load(Thread)
cette petite classe appelle create\\_data\_img() tant que i\_Thread est inférieur au nombre d'images total (i.e qu'il
reste des images à découper)

#### multi\_thread\_start(nb\start=4)
crée n threads (ci-dessus) afin qu'ils découpent les images simultanément.

## Deep Learning

### Chargement du modèle

#### create\_model\_entonnoir\_generique(dim\_input, nb\_output)
crée un modèle adapté à la taille des entrées et sorties passées en paramètre.
Cette fonction génère 
- n couches convolutionnelles, n étant le nombre de canaux de l'image
- y couches denses, dont la taille décrémente à raison 0.5.
- du dropout de taux 0.15
puis compile le modèle et le retourne.

#### create\_model\_entonnoir\_n\_dim(n) !!! Fonction obsolète
ancêtre de create\_model\_entonnoir\_generique() uniquement adaptée à une taille de 64 \* 64 \* n

### Apprentissage

#### main\_generique(dim\_input, nb\_sorties, path, epoch=1)
crée un environnement de travail appelant l'ensemble des fonctions nécessaires à l'éxecution correcte de l'algorithme.
- dim\_input : forme des images
- nb\_sorties : nombre de labels
- path : chemin vers le dataset
- epoch : nombre de répétition d'apprentissage

##### main\_n\_dim(n, path) !!! Fonction obsolète
ancêtre de la fonction main\_generique uniquement adaptée à une taille de 64 \* 64 \* n

### Sauvegarde et test du modèle

#### save\_model(model)
enregistre le .json et le .h5 représentant le modèle.

### Prédiction

#### accuracy\_on\_test(data\_test, model, dim\_input)
calcule la précision du modèle pré-entrainé en prédisant les labels des données de test
- data\_test : les données d'image de test
- model : le modèle
- dim\_input : les dimensions des images

#### predict()
prédit le label auquel appartient une image à l'aide du modèle pré-entrainé

