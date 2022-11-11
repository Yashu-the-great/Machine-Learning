import tensorflow as tf
import numpy as np
import os
import cv2, time
import tensorflow_hub as hub

hub_handle = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
hub_module = hub.load(hub_handle)
os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'

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

styles = [
    load_img(tf.keras.utils.get_file('64b343edeb1352560742df89b89b09b5.jpg','https://i.pinimg.com/originals/64/b3/43/64b343edeb1352560742df89b89b09b5.jpg')),
    load_img(tf.keras.utils.get_file('What_is_Contemporary_Art_2048x.png','https://cdn.shopify.com/s/files/1/2126/2505/articles/What_is_Contemporary_Art_2048x.png?v=1549290999')),
    load_img(tf.keras.utils.get_file('5eeb813aa9575.jpeg','https://d1ee3oaj5b5ueh.cloudfront.net/2020/06/5eeb813aa9575.jpeg')),
    load_img(tf.keras.utils.get_file('violet-fire-colours-hand-painted-background_23-2148427580.jpg','https://img.freepik.com/free-vector/violet-fire-colours-hand-painted-background_23-2148427580.jpg?w=2000')),
    load_img(tf.keras.utils.get_file('image.jpg','https://media.timeout.com/images/103225866/750/562/image.jpg')),
    load_img(tf.keras.utils.get_file('starry-night-van-gogh.jpg','https://practicalpages.files.wordpress.com/2010/02/starry-night-van-gogh.jpg')),
    #https://artincontext.org/wp-content/uploads/2021/05/Abstract-Artists.jpg
    ]
styles_size = len(styles)
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

def start():
    imgs = []
    cap = cv2.VideoCapture(0)
    frame = ...

    for i in range(2):
        success, frame = cap.read()
    
    image_rgb_np = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized_image_np = resize_image_to_square(image_rgb_np, image_size=(1280,720))
    for s in styles:
        outputs = hub_module(tf.constant(resized_image_np), tf.constant(s))
        stylized_image = outputs[0]

        image_pil = tf.keras.preprocessing.image.array_to_img(stylized_image[0])
        image_bgr_np=cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        imgs.append(image_bgr_np)
    return imgs

def showFinal():
    img = np.concatenate((start()), axis=1)
    cv2.imshow("Thee art is eft", img)
    if cv2.waitKey(0) & 0xFF == ord('s'):
        cv2.imwrite(f"./Images/{time.time()}.png", img)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        return

if __name__ == "__main__":
    while True:

        showFinal()