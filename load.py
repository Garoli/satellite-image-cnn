from keras.models import model_from_json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def showImage(filename):
    image = mpimg.imread(filename)
    plt.axis("off")
    plt.imshow(image)
    plt.show()


def test(images, labels):

    print("accuracy : ", (t/n)*100)



js = open("./neural_network_files/model.json", "r")
model_json = js.read()
js.close()

model = model_from_json(model_json)
model.load_weights("./neural_network_files/model.h5")

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])


