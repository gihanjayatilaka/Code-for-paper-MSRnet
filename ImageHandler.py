import cv2
import numpy as np
import os
import keras.backend as K
import shutil
from sklearn.metrics import confusion_matrix

class ImageHandler():
    def __init__(self, image_size):
        assert len(image_size) == 2
        K.set_image_data_format('channels_last')
        self.img_size = image_size

    def load_images(self, path, n=1e5):
        print("Loading ... " + path, end=' ')
        generator = self.yield_image(path)
        tmp = []
        count = 0
        for img in generator:
            # print("DEBUG : ",img.shape)
            tmp.append(img)
            count += 1
            if count > n:
                break
        print("Done!")
        return np.array(tmp)

    def yield_image(self, path):
        # r=root, d=directories, f = files
        for r, d, f in os.walk(path):
            for file in sorted(f):
                img = cv2.imread(os.path.join(r, file))


                print("File name",os.path.join(r, file))
                # print("Loaded img size",img.shape)
                img = cv2.resize(img, self.img_size)
                yield img

    def preprocess_images(self, imgs):
        return imgs/256.0#imgs / 128.0 - 1

    def inv_preprocess_images(self, arr):
        return arr*256.0#(arr + 1.0) * 128

    def save_images(self, path, images):
        try:
            shutil.rmtree(path)
        except:
            print("")

        os.mkdir(path)
        print("Saving {} images".format(len(images)))
        for i in range(len(images)):
            cv2.imwrite(path + "/{:07d}.jpg".format(i), images[i])
        print("Dataset saved successfully in " + path)

    def create_dataset(self, path):

        true_imgs = self.load_images(path+'/input/')

        dark_imgs = true_imgs *np.random.rand()

        self.save_images(path + "/true/", true_imgs)
        self.save_images(path + "/dark/", dark_imgs)