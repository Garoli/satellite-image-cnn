import os
import cv2

def is_grey_scale(filename):
    img = cv2.imread(filename, 1)
    w,h,l = img.shape

    for i in range(w):
        for j in range(h):
            r,g,b = img[i,j]
            if r != g != b: return False
    return True


def rm_if_greyscale(is_greyscale, directoryname):
    path1 = directoryname
    i = 0
    for lab in os.listdir(path1):
        path2 = os.path.join(path1, lab)
        for file in os.listdir(path2):
            if is_grey_scale(file) == is_greyscale:
                os.remove( os.path.join(path2,file) )
                i += 1
    print(i, " files deleted in ", path1)

>>>>>>> aba02a6f0bfceaa013f160dcaa0e77e1646a4785

""" Voici un exemple de suppression
rm_if_greyscale(False, "imagesNG")
rm_if_greyscale(True, "imagesRGB")
"""
