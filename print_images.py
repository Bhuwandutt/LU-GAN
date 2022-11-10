import cv2

img =cv2.imread('/Users/bhuwandutt/Documents/GitHub/LU-GAN/data/imgs/1_IM-0001-3001.dcm.png', 0)
image = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
print(image.shape)


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

        img = cv2.resize(image, dsize=self.resize, interpolation=cv2.INTER_CUBIC)
        print("Rescale")
        return img



Resale = Rescale(output_sizes = (256,256) )

print(Resale(image).shape)

