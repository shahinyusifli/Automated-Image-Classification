import numpy as np
from keras.preprocessing import image


class ImageInput:


    def __init__(self, input_img_url, img_height, img_width):
        
        self.input_img_url = input_img_url
        self.image_height = img_height
        self.img_width = img_width
    
    def process_image(self):

        img = image.load_img(r(self.input_img_url))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)

        return img

    def execute(self):

        self.process_image()


        