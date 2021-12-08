import numpy as np

#TODO -> ratio aire bbox aire polygon  moyen
#TODO -> répartition label premier échantillon
#TODO -> histogramme aire bbox
#TODO -> histogramme aire polygo
#TODO -> Répartition polygon label





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


def getArea(x,y):
    return 0.5*np.abs(np.dots(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))


def moyenneTailleBbox():



