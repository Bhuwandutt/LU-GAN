import pandas as pd
import numpy as np
import cv2
import os
from PIL import Image

CSV_DIR = os.getcwd()+'/LU-GAN/csv/'
IMAGE_DIR = os.getcwd()+'/LU-GAN/data/imgs/'


class Equalize(object):
    def __init__(self,mode="Normal"):

        self.mode = mode
        self.equlizer = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    def __call__(self, image):
        if self.mode=="Normal":
            equ = cv2.equalizeHist(image)
        elif self.mode=="CLAHE":
            equ = self.equlizer.apply(image)
        # print("Equalize")
        return equ


class ToTensor(object):

    """Convert darray in sample to Tensors"""
    def __call__(self, image):
        # print("To Tensor")
        # torch image: channel * H * W
        h, w = image.shape[:2]
        # print(f'{image}')
        image = image.reshape((1, h, w))/255
        image = (image - 0.5) / 0.5
        return image


def read_png(filename):
    # print(filename)
    image = None
    if os.path.exists(filename):
        imag = Image.open(filename)
        image = imag.convert('L')

        # image = cv2.cvtColor(imag, cv2.COLOR_BGR2GRAY)
    return image


def vcn_images(uid: int):  # Get the list of images that don't have both frontal and lateral images.

    df = pd.read_csv(CSV_DIR + 'indiana_projections.csv')
    df_ = df.loc[(df['uid'] == uid)].values
    if len(df_) == 2:
        return True
    return False


class Rescale(object):
    """Rescale the image in the sample to a given size
    Args:
        Output_size(tuple): Desired output size
            tuple for output_size
    """

    def __init__(self, output_sizes):
        new_h, new_w = output_sizes
        self.resize = (int(new_h), int(new_w))

    def __call__(self, image):
        img = np.resize(image, new_shape=self.resize)
        # print("Rescale")
        return img
