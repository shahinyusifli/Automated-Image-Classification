from os import name
import tensorflow as tf
import numpy as np
import pathlib

from tensorflow._api.v2 import image


class LoadImages:
    
    def __init__(self, batch_size, image_height, image_weight, 
                data_destination, file_name):
        
        self.batch_size = batch_size
        self.image_height = image_height
        self.image_weight = image_weight
        self.data_dst = data_destination
        self.fname = file_name

    def create_direction(self):

        data_dir = tf.keras.utils.get_file(origin=self.data_dst,
                                            fname=self.fname,
                                            untar=True)

        data_dir = pathlib.Path(data_dir)

        image_count = len(list(data_dir.glob('*/*.jpg')))

        return data_dir, image_count
        




    

    