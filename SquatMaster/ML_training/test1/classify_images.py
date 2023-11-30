import json
import os
from os import path
# # Open the json file containing the classifications
# with open("/home/rafik/PROJECTS/pose1/Material/hill_raise/vott_output/vott-json-export/test2-export.json", "r") as f:
#    classification = json.load(f)
# Create a set which contains all the classes
# classes = set([i["names"] for i in classification.values()])
# # For each of the classes make a folder to contain them
# for c in classes:
#     os.makedirs(c)
# # For each image entry in the json move the image to the folder named it's class
# for image_number, image_data in classification.items():
#     os.rename(image_data["/home/rafik/PROJECTS/pose1/Material/hill_raise/vott_output/vott-json-export"], path.join(image_data["tags"], "{}.jpg".format(image_number)))


from tensorflow.data import Dataset, TFRecordDataset
from tensorflow.data.experimental import TFRecordWriter

# # Construct a small dataset
# ds = Dataset.from_tensor_slices([b'abc', b'123'])

# # Write the dataset to a TFRecord
# writer = TFRecordWriter(PATH)
# writer.write(ds)
    
# with open("/home/rafik/PROJECTS/pose1/Material/hill_raise/vott_output/test2-TFRecords-export/IMG_3676.MOV#t=0.133333.tfrecord") as PATH:

import tensorflow as tf
import tfrecord

filename = '/home/rafik/PROJECTS/pose1/Material/hill_raise/vott_output/test2-TFRecords-export/IMG_3676.MOV#t=0.133333.tfrecord'


# Store the returned generator object into the variable 'loader'
loader = tfrecord.tfrecord_loader(filename, None, {
    "image": "byte",
    "id": "byte",
    "class": "int"
})

import imageio.v3 as iio
import matplotlib.pyplot as plt

# Get the first item from the generator
data = next(loader)

# Print the item's class
print(data['class'])

# Return the image as a Numpy array
img_arr = iio.imread(data['image'])

# Print the image shape
print(img_arr.shape)

# Visualize the image
plt.imshow(img_arr)



