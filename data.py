import tensorflow as tf
import glob
import pathlib
#AUTOTUNE=tf.data.experimental.AUTOTUNE

def get_paths(directory):
    dir=pathlib.Path.cwd()/directory
    all_image_paths=list(dir.glob('*'))
    return [str(path) for path in all_image_paths]


def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [192, 192])
    image /= 255.0  # normalize to [0,1] range
    #image = tf.image.convert_image_dtype(image, tf.float16)
    return image

def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)

def from_path_to_tensor(paths, batch_size):
    path_ds=tf.data.Dataset.from_tensor_slices(paths)
    ds=path_ds.map(load_and_preprocess_image, num_parallel_calls=1)
    #ds=ds.repeat()
    ds=ds.shuffle(5000)
    ds=ds.batch(batch_size)
    ds=ds.prefetch(buffer_size=1)
    return ds
