import tensorflow as tf
import numpy as np
from tensorflow.python.data.ops.dataset_ops import AUTOTUNE
from tensorflow.python.ops.gen_string_ops import string_format
import os




class CreateInputPipeline:
    
    
    def __init__(self, data_direction, train_test_split_size,
                image_count, buffer_size, batch_size, img_height, img_width ):

        self.data_dir = data_direction
        self.tr_tst_size = train_test_split_size
        self.img_count = image_count
        self.buf_size = buffer_size
        self.bch_size = batch_size
        self.img_hgt = img_height
        self.img_wdh = img_width

    def create_tf_data(self):
        
        list_dataset = tf.data.Dataset.list_files(str(self.data_dir/'*/*'), shuffle=False)
        list_dataset = list_dataset.shuffle(self.img_count, reshuffle_each_iteration=False)

        train_size = int(self.img_count * self.tr_tst_size)
        train_ds = list_dataset.take(train_size)
        test_ds = list_dataset.skip(train_size)

        
        return train_ds, test_ds
        
    def find_class_names(self):

        class_names = np.array(
            sorted([item.name for item in self.data_dir.glob('*') if item.name != 'LICENSE.txt'])
            )
        
        return class_names
        

    def show_length_dataset(self):
        pass


    def get_label(self, file_path):

        class_names = self.find_class_names()
        parts = tf.strings.split(file_path, os.path.sep)
        label = parts[-2] == class_names
        return tf.argmax(label)


    def decode(self, image):

        img = tf.io.decode_jpeg(image, channels=3)
        return tf.image.resize(img, [self.img_hgt, self.img_wdh])


    def process_path(self, file_path):
        
        label = self.get_label(file_path)
        img = tf.io.read_file(file_path)
        img = self.decode(img)

        return img, label

    def process_datasets(self):

        train_ds, test_ds = self.create_tf_data()
        train_ds = train_ds.map(self.process_path, num_parallel_calls=AUTOTUNE)
        test_ds = test_ds.map(self.process_path, num_parallel_calls=AUTOTUNE)

        return train_ds, test_ds 

    def show_procesed_length_dataset(self):
        pass

    def data_augmentation(self):
        
        train_ds = self.create_tf_data()
        label = self.get_label(train_ds) 
        

    def configure_for_performance(self, data_set):
        
        ds = data_set.cache()
        ds = ds.shuffle(buffer_size=self.buf_size)
        #data argument
        ds = ds.batch(self.bch_size)
        ds = ds.prefetch(buffer_size=AUTOTUNE)

        return ds



    def visualize_data(self):
        pass

    def execute(self):
        
        train_ds, test_ds = self.process_datasets()
        train_ds = self.configure_for_performance(train_ds)
        test_ds = self.configure_for_performance(test_ds)

        return train_ds, test_ds


