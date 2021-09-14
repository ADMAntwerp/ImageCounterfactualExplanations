import os
import multiprocessing
import argparse

import matplotlib.pylab as plt
import PIL.Image as Image
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from skimage.segmentation import quickshift

from isedc.sedc_t2_fast import sedc_t2_fast


parser = argparse.ArgumentParser("CF Image Generator For MobileNetV2 Model")
parser.add_argument("--data", help="Data Folder", type=str, required=True)
parser.add_argument("--output", help="Output Folder", type=str, required=True)
parser.add_argument("--cclass", help="Counterfactual class, labels names are available "
                              "in https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt",
                    type=str, required=True)
parser.add_argument("--jobs", help="Number of jobs", type=int, default=1)
parser.add_argument("--mode", help="Mode", type=str, default="blur")
parser.add_argument("--timeout", help="Max generation time in seconds", type=int, default=60)


args = parser.parse_args()

data_folder = args.data
output_folder = args.output
cf_class = args.cclass
number_jobs = args.jobs
cf_mode = args.mode
cf_timeout = args.timeout

# Get labels
labels_path = tf.keras.utils.get_file('ImageNetLabels.txt','https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
imagenet_labels = np.array(open(labels_path).read().splitlines())

# Verify if counterfactual mode is an existent mode
cf_modes = ['mean', 'blur', 'random', 'inpaint']
if cf_mode not in cf_modes:
    raise f'Counterfactual mode {cf_mode} does not exist, the allowed modes are {",".join(cf_modes)}'

# Create output folder if not exist
if not os.path.exists(output_folder):
    os.mkdir(output_folder)

# Verify if counterfactual class exist
cf_class_idx = np.where(np.array(list(map(lambda x: str(x).lower(), imagenet_labels))) == str(cf_class).lower())[0]

# Verify if counterfactual class is correct
if len(cf_class_idx) != 1:
    raise f'Counterfactual class {cf_class} is not right'

# Get the counterfactual class label
cf_class_label = imagenet_labels[cf_class_idx[0]]

# Deactivate GPU since it will run on multiple processes
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Import model
classifier_url ="https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/2"

# Define image shape
IMAGE_SHAPE = (224, 224)

def find_cf_region(i_data):

    # Configure classifier
    classifier = tf.keras.Sequential([
        hub.KerasLayer(classifier_url, input_shape=IMAGE_SHAPE + (3,), trainable=True)
    ])

    img_path = i_data[0]
    output_folder = i_data[1]

    img_filename = img_path.split('/')[-1]

    image = Image.open(f'{img_path}')
    image = image.resize(IMAGE_SHAPE)
    image = np.array(image) / 255.0
    image = image[:, :, 0:3]

    segments = quickshift(image, kernel_size=4, max_dist=200, ratio=0.2)

    explanation, segments_in_explanation, perturbation, new_class = sedc_t2_fast(
        image,
        classifier,
        segments,
        np.where(imagenet_labels == cf_class_label)[0][0],
        cf_mode,
        max_time=cf_timeout)

    if explanation is not None:
        mask_true_single = np.isin(segments, segments_in_explanation)

        mask_true_full = []
        for row in mask_true_single:
            mask_true_col = []
            for col in row:
                if col:
                    mask_true_col.append([1, 1, 1])
                else:
                    mask_true_col.append([0, 0, 0])
            mask_true_full.append(mask_true_col)

        mask_true_full = np.array(mask_true_full)

        Image.fromarray(
            ((image * (mask_true_full == 0) + (mask_true_full != 0) * (0, 1, 0)) * 255).astype(np.uint8)).save(
            f'{output_folder}/{img_filename}')


img_factual = os.listdir(data_folder)
img_factual_paths = [[data_folder+x, output_folder] for x in img_factual]

p = multiprocessing.Pool(number_jobs)

p.map(find_cf_region, img_factual_paths)
