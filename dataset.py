import nibabel as nib

import tensorflow.compat.v2 as tf
tf.compat.v1.enable_v2_behavior()
import glob
import pydicom
import numpy as np
import os
import tensorflow_datasets as tfds

def load_and_preprocess(image_path,label):
    image = pydicom.dcmread(image_path.numpy().decode())
    pixel_array = image.pixel_array.astype(np.float32)
    normalized_pixel_array = pixel_array / np.max(pixel_array)
    dicom_tensor = tf.convert_to_tensor(normalized_pixel_array)
    expanded_tensor = tf.expand_dims(dicom_tensor, axis=-1)  # Expand along the first axis
    expanded_tensor = tf.tile(expanded_tensor, [1, 1, 3]) 
    return expanded_tensor

def create_dataset():
    label_names = {'A':0,'B':1,'E':2,'G':3}
    image_paths  = glob.glob(os.path.join('/media/wenuka/New Volume-G/01.FYP/Dataset/Lung_CT_Dataset/Lung-PET-CT-Dx-NBIA-Manifest-122220/Lung-PET-CT-Dx', '**', '*.dcm'), recursive=True)
    labels = [label_names[x.split('/')[9][8]] for x in image_paths]
    dataset = tf.data.Dataset.from_tensor_slices((image_paths,labels))
    dataset = dataset.map(lambda x, y: (tf.py_function(load_and_preprocess, [x, y], tf.float32), y), num_parallel_calls=tf.data.AUTOTUNE)
    return dataset


