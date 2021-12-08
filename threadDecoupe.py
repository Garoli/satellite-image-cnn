from threading import Thread
from shapely.geometry import Point, MultiPolygon, Polygon


class Decoupe(Thread):

    def __init__(self, i, imgs, img, bbox, poly, decoupe):
        Thread.__init__(self)
        self.i = i
        self.imgs = imgs
        self.img = img
        self.bbox = bbox
        self.poly = poly
        self.decoupe = decoupe

    def run(self):
        if (self.bbox[2][0] - self.bbox[0][0]) < 10 or (self.bbox[2][1] - self.bbox[0][1]) < 10:
            for x in range(self.bbox[0][1], self.bbox[2][1]):
                for y in range(self.bbox[0][0], self.bbox[2][0]):
                    if Point(y, x).within(self.poly):
                        self.imgs[self.i][x][y] = self.img[x][y]

        else:
            if Polygon(self.bbox).intersects(self.poly):
                if self.decoupe:
                    ne = int((self.bbox[2][0] + self.bbox[0][0]) / 2)
                    box1 = [self.bbox[0], (ne, self.bbox[0][1]), (ne, self.bbox[2][1]), self.bbox[3]]
                    box2 = [(ne, self.bbox[0][1]), self.bbox[1], self.bbox[2], (ne, self.bbox[3][1])]
                else:
                    ne = int((self.bbox[2][1] + self.bbox[0][1]) / 2)
                    # print(self.bbox[2][1], self.bbox[0][1], ne)
                    box1 = [self.bbox[0], self.bbox[1], (self.bbox[2][0], ne), (self.bbox[3][0], ne)]
                    box2 = [(self.bbox[0][0], ne), (self.bbox[1][0], ne), self.bbox[2], self.bbox[3]]

                # todo mettre un  décalage plutot que des redimentionnement
                t1 = Decoupe(self.i, self.imgs, self.img, box1, self.poly, not self.decoupe)
                t2 = Decoupe(self.i, self.imgs, self.img, box2, self.poly, not self.decoupe)
                t1.start()
                t2.start()
            else:
                if Point(self.bbox[0][0], self.bbox[0][1]).within(self.poly):
                    self.imgs[self.i][self.bbox[0][1]: self.bbox[2][1]][self.bbox[0][0]: self.bbox[2][0]] = self.img[
                                                                                                            self.bbox[
                                                                                                                0][1]:
                                                                                                            self.bbox[
                                                                                                                2][1]][
                                                                                                            self.bbox[
                                                                                                                0][0]:
                                                                                                            self.bbox[
                                                                                                                2][0]]
