import tensorflow as tf
import numpy as np
import PIL
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import image_transform_net


def save_image(image, file_path):
    '''
    Save an image as a jpg file. The image is given as 
    a numpy array with pixel values between 0 and 255.
    
    :param image:
        The numpy array of the image.
        type: ndarray
    :param file_path:
        The full path to save the mixed image, 
        i.e. image path + image name
        type: str
    :return:
        Save the image as a jpeg file.
    '''
    
    # Ensure the pixel values are between 0 and 255.
    image = np.clip(image, 0.0, 255.0)
    
    # Convert to bytes.
    image = image.astype(np.uint8)
    
    # Write the image file in jpeg format.
    with open(file_path, 'wb') as file:
        PIL.Image.fromarray(image).save(file, 'jpeg')


def feed_forward(image_path, output_path, checkpoint_dir):
    '''
    Since we already have a pre-trained checkpoint, we could 
    just use it to generate a mixed image.
    
    :param image_path:
        The path and filename that you are going to tranfer.
        type: str
    :param output_path:
        The path to store the mixed image, including its filename.
        type: str
    :param checkpoint_dir:
        The path and filename of the pre-trained checkpoint.
        type: str
    :return:
        Save the mixed image.
    '''
    
    # Build a graph and a session.
    with tf.Graph().as_default(), tf.Session() as sess:
        # Read content image from a file as a numpy array.
        content_image = mpimg.imread(image_path)
        
        # Since image transform net requires a 4-D array,
        # we'll have to expand a dimension at axis = 0.
        content_image = np.expand_dims(content_image, axis=0)
        
        # Define a 4-D placeholder for image.
        image_holder = tf.placeholder(
            tf.float32, content_image.shape, 'input_image')
        
        # Let image flow through image transform net.
        output_image = image_transform_net.net(image_holder)
        
        # Restore the pre-trained checkpoint.
        saver = tf.train.Saver()
        saver.restore(sess, checkpoint_dir)
        
        # Run the session.
        feed_dict = {image_holder: content_image}
        mixed_image = sess.run(output_image, feed_dict)
        
        # Save the mixed image.
        with open(output_path, 'wb') as file:
            save_image(mixed_image[0], output_path)

if __name__ == '__main__':
    IMAGE_PATH = ''
    OUTPUT_PATH = ''
    CHECKPOINT_DIR = ''
    feed_forward(IMAGE_PATH, OUTPUT_PATH, CHECKPOINT_DIR)


