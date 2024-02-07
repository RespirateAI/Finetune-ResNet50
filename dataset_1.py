import tensorflow as tf
import pydicom
import numpy as np
import glob
import os

def parse_function(filename, label):
    image = pydicom.dcmread(filename)
    pixel_array = image.pixel_array.astype(np.float32)
    normalized_pixel_array = pixel_array / np.max(pixel_array)
    dicom_tensor = tf.convert_to_tensor(normalized_pixel_array)
    expanded_tensor = tf.expand_dims(dicom_tensor, axis=-1)  # Expand along the first axis
    expanded_tensor = tf.tile(expanded_tensor, [1, 1, 3]) 
    data_out = {'image':expanded_tensor,'label':label}
    return expanded_tensor,label

def dataset_maker(list_paths,list_lables):
    filenames = tf.constant(list_paths)
    labels = tf.constant(list_lables)
    dataset = tf.data.Dataset.from_tensor_slices((filenames,labels))
    dataset = dataset.map(parse_function)
    return dataset

def load_one_sample(image_path,label):
    image = pydicom.dcmread(image_path.numpy().decode())
    pixel_array = image.pixel_array.astype(np.float32)
    normalized_pixel_array = pixel_array / np.max(pixel_array)
    dicom_tensor = tf.convert_to_tensor(normalized_pixel_array)
    expanded_tensor = tf.expand_dims(dicom_tensor, axis=-1)  # Expand along the first axis
    expanded_tensor = tf.tile(expanded_tensor, [1, 1, 3]) 
    data_out = {'image':expanded_tensor,'label':label.numpy()}
    return data_out

def wrapper_load(img_path, label_path):
  img, label = tf.py_function(func = load_one_sample, inp = [img_path, label_path], Tout = [tf.float32, tf.uint8])
  return img, label

label_names = {'A':0,'B':1,'E':2,'G':3}
image_paths  = glob.glob(os.path.join('/media/wenuka/New Volume-G/01.FYP/Dataset/Lung_CT_Dataset/Lung-PET-CT-Dx-NBIA-Manifest-122220/Lung-PET-CT-Dx', '**', '*.dcm'), recursive=True)
labels = [label_names[x.split('/')[9][8]] for x in image_paths]

# dataset = dataset_maker(image_paths,labels)


dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels)).map(wrapper_load)
final_dataset = dataset.batch(16)
iterator = iter(final_dataset)
x = next(iterator)
print(x)