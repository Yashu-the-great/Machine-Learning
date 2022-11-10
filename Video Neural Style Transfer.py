import tensorflow as tf
import numpy as np
import matplotlib as mpl
import os

os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'

mpl.rcParams['figure.figsize'] = (12, 12)
mpl.rcParams['axes.grid'] = False

style_path = tf.keras.utils.get_file('What_is_Contemporary_Art_2048x.png','https://cdn.shopify.com/s/files/1/2126/2505/articles/What_is_Contemporary_Art_2048x.png?v=1549290999') #v

def load_img(path_to_img):
  max_dim = 512
  img = tf.io.read_file(path_to_img)
  img = tf.image.decode_image(img, channels=3)
  img = tf.image.convert_image_dtype(img, tf.float32)

  shape = tf.cast(tf.shape(img)[:-1], tf.float32)
  long_dim = max(shape)
  scale = max_dim / long_dim

  new_shape = tf.cast(shape * scale, tf.int32)

  img = tf.image.resize(img, new_shape)
  img = img[tf.newaxis, :]
  return img

style_image = load_img(style_path)

import tensorflow_hub as hub
hub_handle = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
hub_module = hub.load(hub_handle)

def crop_center(image):
  shape = image.shape
  new_shape = min(shape[1], shape[2])
  offset_y = max(shape[1] - shape[2], 0) // 2
  offset_x = max(shape[2] - shape[1], 0) // 2
  image = tf.image.crop_to_bounding_box(
      image, offset_y, offset_x, new_shape, new_shape)
  return image


def resize_image_to_square(image_np, image_size=(256,256), preserve_aspect_ratio=True):
    image_np_extra = image_np.astype(np.float32)[np.newaxis, ...]
    if image_np_extra.max() > 1.0:
        image_np_extra = image_np_extra / 255.
    if len(image_np_extra.shape) == 3:
      image_np_extra = tf.stack([image_np_extra, image_np_extra, image_np_extra], axis=-1)
    image_np_extra = crop_center(image_np_extra)
    image_np_extra = tf.image.resize(image_np_extra, image_size, preserve_aspect_ratio=True)
    return image_np_extra

import cv2

def start():
    cap = cv2.VideoCapture(0)
    frame = ...
    for i in range(10):
        success, frame = cap.read()
    image_rgb_np = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized_image_np = resize_image_to_square(image_rgb_np, image_size=(1280,720))
        
    outputs = hub_module(tf.constant(resized_image_np), tf.constant(style_image))
    stylized_image = outputs[0]

    image_pil = tf.keras.preprocessing.image.array_to_img(stylized_image[0])
    image_bgr_np=cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        
    return image_bgr_np

def showFinal():
    cv2.imshow("img",start())
    if cv2.waitKey(0) & 0xFF == ord('q'):
        return

if __name__ == "__main__":
    while True:
        showFinal()