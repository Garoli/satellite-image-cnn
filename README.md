# Projet S4 - Reconnaissance des objets depuis des images Sat

Le projet consiste à reconnaître des objets depuis des images satellitaires tels que : la forêt, l’eau, la mer, les habitations, la route, etc.

## Pour débuter

### Prérequis

Le projet est réalisé en python. 
Vous allez avoir besoin d'un interpréteur python 3.2 ou d'une version ultérieure
Vous allez avoir besoin de conda ou de pip pour installer les dépendances nécéssaires.

[Télécharger Conda](https://www.anaconda.com/distribution/#download-section)

### Installation des dépendances

Il vous faut installer tensorflow-gpu avec conda si vous disposez d'une carte graphique :

```sh
conda install tensorflow-gpu
```

Les dépendances nécessaires sont :
keras numpy matplotlib os opencv random shapely PIL pandas tifffile builtins threading

```sh
conda install keras numpy matplotlib os opencv random shapely PIL pandas tifffile builtins threading
```
ou
```sh
pip install keras numpy matplotlib os opencv random shapely PIL pandas tifffile builtins threading
```

### Récupération de notre code source

Vous pouvez au choix télécharger le fichier main.py, ou récupérer le projet complet
avec dataset depuis gitlab.

```sh
git clone https://gitlab.com/Garoli/projet_s4_images_satellites.git
```

## Utilisation


### Préparation des données

Il faut un set important de données pour arriver à des bons résultats de prédiction.
Les images doivent être toutes de même taille et être organisées selon l'arborescence suivante :
```
images/
       label1/
	          image11.jpg
			  image12.jpg
			  ...
	   label2/
	          image21.jpg
			  image22.jpg
              ...
	   ...
	   labeln/
	          imagen1.jpg
			  ...
```

#### Création du dataset en live

Notre projet contient un utilitaire qui permet de découper des grandes images en petites images de même format.

utilisation :

```sh
slice_image(chevauchement_en_pixels, tableau_image, taille_sortie)
```

avec :
- chevauchement_en_pixels : le nombre de pixels qui seront inclus dans plusieurs images côte à côte.
- tableau_image : un tableau 3D correspondant à l'image chargée préalablement par la fonction
                  load_image_n_dim(chemin_du_fichier, nombre_de_canaux_de_l_image)
- taille_sortie : la taille des images découpées (défaut : 64)

#### Création d'un dataset persistent avec un fichier csv

Notre projet contient une fonction permettant de créer un répertoire contenatn les images découpées et classées
généré à partir d'une image tiff et d'un fichier csv qui contient des informations sur la découpe (des polygones
séparant les différents labels associés aux parties de l'image)
```
create_data_img(path, file, output_path, chevauchement, size_image_out, number_of_labels) 
```
avec :
- path : le dossier contenant l'image à découper
- file : le nom de l'image
- output_path : le dossier de sortie (possédant préalablement un dossier par catégorie)
- chevauchement : le nombre de pixels qui seront inclus dans plusieurs images côte à côte.
- size_image_out : la taille des images de sortie
- number_of_labels : le nombre de labels (correspond aux dossier de l'image et à la 1e colonne du csv)

### Apprentissage du modèle

la fonction ``` main_generique(shape, nb_labels, path)``` entraine le modèle de reconnaissance d'images et le retourne.
avec :
- shape : la dimension de l'image d'entrée
- nb_labels : le nombre de catégories différentes
- path : le dossier qui contient le dataset
- epoque : le nombre de répétitions d'entrainement (défault : 1)


### Sauvegarde des données

La fonction ```save_model(model)``` sauvegarde le modèle. Ainsi, il ne faut pas le réentrainer à chaque fois qu'on veut
faire une prédiction.

### Prédiction du modèle

La fonction ```predict(image_tableau)``` prend en paramètre une image importée (via ```load_image_n_ndim()```)
et "prédit" de quelle catégorie appartient l'image

## Auteurs

* [**Brood Sarah**](gitlab.com/Garoli) - *Développeuse*
* **Caracciolo Nathan** - *Développeur*
* **Célié Kévin** - *Développeur*
* [**Pompeani Paco**](gitlab.com/ppom) - *Développeur*

