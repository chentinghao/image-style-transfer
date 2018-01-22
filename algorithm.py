import functools, pdb, time, os, random
os.environ['TPP_CPP_MIN_LOG_LEVEL'] = '2' # ignore some warnings

import tensorflow as tf
import numpy as np
import scipy.misc

import vgg19
import image_transform_net
import play


def save_image(output_path, image):
    '''
    Save image in the output path.
    
    :param: output_path:
        Output path.
        type: str
    :param: image:
        Image to save.
        type: ndarray
    :return:
        Nothing, but save image in the path.
    '''
    image = np.clip(image, 0, 255).astype(np.uint8)
    scipy.misc.imsave(output_path, image)

def get_image(path, image_shape=False):
    '''
    Reshape the image from the path.
    
    :param path:
        Path to the image.
        type: str
    :param image_shape:
        Shape of the image. Default is false.
        type: 3-D tuple
    :return:
        Resized image.
        type: ndarray
    '''
    image = scipy.misc.imread(path, mode='RGB')
    if not (len(image.shape) == 3 and image.shape[2] == 3):
        image = np.dstack((image,image,image))
    if image_shape != False:
        image = scipy.misc.imresize(image, image_shape)
    return image

def list_files(file_dir):
    '''
    List all the filenames in the directory.
    
    :param file_dir:
        The path of the directory.
        type: str
    :return:
        A list of all the filenames in the directory.
        type: list
    '''
    # Generate the file names in a directory tree by walking the tree.
    for dirpath, dirnames, filenames in os.walk(file_dir):
        # filenames will return a list of filenames.
        list_names = filenames
        
        # Don't return other directory tree.
        break
    
    return list_names

def get_files(img_dir):
    '''
    List all the filenames with full path in the directory.
    
    :param img_dir:
        The path of the directory
    :return:
        A list of all the filenames with full path in the directory.
        type: list
    '''
    # Get all the filenames from the directory.
    files = list_files(img_dir)
    
    return [os.path.join(img_dir, i) for i in files]


STYLE_LAYERS = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']
CONTENT_LAYER = 'relu4_2'

def tensor_size(tensor):
    '''
    Helper function for calculating the size of each tensor.
    :param tensor:
        Get the shape of a tensor and multiply all the them to obtain the shape.
    :return:
        Size of the tensors.
    '''
    from operator import mul
    return functools.reduce(mul, (d.value for d in tensor.get_shape()[1:]), 1)


def style_transfer(content_target, style_target, content_weight, style_weight, 
                   denoise_weight, vgg_path, epochs=2, print_iterations=100, batch_size=4,
                   save_path='saver/fns.ckpt', slow=False, learning_rate=1e-3, debug=False):
    '''
    Main style transfer algorithm.
    
    :param content_target:
        A list of path of the content targets (images) for training and evaluation.
        type: list
    :param style_target:
        The path of the style target (image).
        type: str
    :param content_weight:
        Lambda of L2 loss function for content target.
        type: int or float
    :param style_weight:
        Lambda of L2 loss function for style target.
        type: int or float
    :param denoise_weight:
        Lambda of L2 loss function for denoising.
        type: int or float
    :param vgg_path:
        Path to VGG19 model.
        type: str
    :param epochs:
        Number of epochs.
        type: int
    :param print_iterations:
        When to print out the iterations.
        type: int
    :param batch_size:
        The size of batch for training.
        type: int
    :param save_path:
        The path to save checkpoints.
        type: str
    :param slow:
        For the usage of debug.
        type: bool
    :param learning_rate:
        Learning rate for optimizer.
        type: int or float
    :param debug:
        For the usage of debug.
        type: bool
    :return:
        Tuple of loss, total number of batches, total epoch
        _preds, losses, num_batches, epoch
    '''
    # For the usage of debug
    if slow:
        batch_size = 1
    
    cut = len(content_target) % batch_size
    if cut > 0:
        content_target = content_target[:-cut]
        print('Train dataset has been cut.')
    
    batch_shape = (batch_size, 256, 256, 3)
    
    # Loss function for style features
    # Create a dict for storing style features.
    style_features = {}
    
    # We are going to call VGG19 graph from outside, so we'll 
    # have to set tf.Graph() as default.
    with tf.Graph().as_default(), tf.Session() as sess:
        # Expand new dim at axis = 0 for batch size.
        style_target = np.array([style_target])
        
        # Create a 4-D, float32 placeholder for style target.
        style_image = tf.placeholder(tf.float32, style_target.shape, 'style_image')
        
        # Style target minus mean pixel.
        # mean pixel = np.array([123.68, 116.779, 103.939])
        style_image = vgg19.preprocess(style_image)
        
        # Let style target flow through VGG19 model.
        net = vgg19.net(vgg_path, style_image)
        
        for layer in STYLE_LAYERS:
            # Feed style targets into style image
            feed_dict = {style_image: style_target}
            features = net[layer].eval(feed_dict)
            
            # Reshape [batch_size, height, width, channel] to 
            # [batch_size * height * width, channel]
            features = np.reshape(features, (-1, features.shape[3]))
            
            # Calculate gram metrix and normalize it
            gram = np.matmul(features.T, features) / features.size
            
            # Store the gram metrices for loss function.
            style_features[layer] = gram
    
   

    # Build the main graph.
    with tf.Graph().as_default(), tf.Session() as sess:
        # Create a 4-D, float32 placeholder for content target.
        X_content = tf.placeholder(tf.float32, batch_shape, 'input_image')
        
        # Content image minus mean pixel.
        content_pre = vgg19.preprocess(X_content)
        
        # Create a dict for storing content features.
        content_features = {}
        
        # Let content target flow through VGG19 model
        content_net = vgg19.net(vgg_path, content_pre)
        
        # Store the content features for loss function.
        content_features[CONTENT_LAYER] = content_net[CONTENT_LAYER]
        
        if slow:
            output_image = tf.Variable(
                tf.random_normal(inp_image.get_shape()) * 0.256)
        else:
            # Let input image flow through image transform net.
            output_image = image_transform_net.net(X_content / 255.0)
            
            # Output image minus mean pixel.
            output_pre = vgg19.preprocess(output_image)
        
        # Let output image flow through VGG19 model in order to calculate 
        # the features of output image.
        output_features = vgg19.net(vgg_path, output_pre)
        
        # Calculate the size of content features
        content_size = tensor_size(content_features[CONTENT_LAYER]) * batch_size
        
        assert tensor_size(content_features[CONTENT_LAYER]) == \
        tensor_size(output_features[CONTENT_LAYER])
        
        # Calculate L2 loss function between output features and content features.
        # L2 loss without half the norm.
        l2_loss = 2 * tf.nn.l2_loss(
            output_features[CONTENT_LAYER] - content_features[CONTENT_LAYER])
        
        # Normalize L2 loss and multiply its weight (a.k.a. lambda).
        content_loss = content_weight * (l2_loss / content_size)
        
        
        # Calculate style loss function.
        style_losses = []
        
        for style_layer in STYLE_LAYERS:
            # Output features in each style layer.
            layer = output_features[style_layer]
            
            # Get the batch_size, height, width, channel from each style layer.
            bs, height, width, channel = map(lambda i: i.value, layer.get_shape())
            
            # Calculate the size.
            size = height * width * channel
            
            # Try to calculate gram metrix for output features in each style layer.
            # Reshape output features in each style layer.
            output_f_style = tf.reshape(layer, (bs, height * width, channel))
            
            # Transpose output features.
            output_f_style_T = tf.transpose(output_f_style, perm=[0,2,1])
            
            # Calculate gram metrix for output features.
            output_gram = tf.matmul(output_f_style_T, output_f_style) / size
            
            style_gram = style_features[style_layer]
            
            # Calculate L2 loss function between output features and style features.
            # L2 loss without half the norm.
            l2_loss = 2 * tf.nn.l2_loss(output_gram - style_features[style_layer])
            
            # Normalize L2 loss and store them in style_losses.
            style_losses.append(l2_loss / style_features[style_layer].size)
        
        # Normalize style loss function and multiply its weight (a.k.a. lambda).
        style_loss = style_weight * functools.reduce(tf.add, style_losses) / batch_size
        
        # This creates the loss function for denoising the mixed image. The algorithm 
        # is called Total Variation Denoising and essentially just shifts the image 
        # one pixel in the x and y axis. We calculate L2 loss without half the norm 
        # for x and y axis.
        Y_denoise = 2 * tf.nn.l2_loss(output_image[:, 1:, :, :] - output_image[:, :-1, :, :])
        X_denoise = 2 * tf.nn.l2_loss(output_image[:, :, 1:, :] - output_image[:, :, :-1, :])
        
        # Calculate the size of Y_denoise and X_denoise.
        Y_size = tensor_size(output_image[:,1:,:,:])
        X_size = tensor_size(output_image[:,:,1:,:])
        
        # Normalize total variation denoising and multiply its weight (a.k.a. lambda).
        denoise_loss = \
        denoise_weight * (X_denoise / X_size + Y_denoise / Y_size) / batch_size
        
        # Total loss.
        loss = content_loss + style_loss + denoise_loss
        
        # Use Adam as our optimizer.
        optimizer = tf.train.AdamOptimizer(learning_rate)
        
        # Minimize the loss function.
        train_op = optimizer.minimize(loss)
        
        # Initialize all variables.
        init = tf.global_variables_initializer()
        sess.run(init)
        
        uid = random.randint(1, 100)
        print("UID: %s" % uid)
        
        # Start training.
        for epoch in range(epochs):
            num_examples = len(content_target)
            num_batches = 0
            
            while num_batches * batch_size < num_examples:
                start_time = time.time()
                
                # Decide the range of batch for training.
                start = num_batches * batch_size
                end = start + batch_size
                
                # Initialize X_batch with zeros.
                X_batch = np.zeros(batch_shape, np.float32)
                
                # Fill content targets into X_batch
                for i, img_p in enumerate(content_target[start: end]):
                    X_batch[i] = get_image(img_p, (256,256,3)).astype(np.float32)

                # Done calculating current batch, iterate to next batch.
                num_batches += 1
                assert X_batch.shape[0] == batch_size

                # Run the graph to start training.
                feed_dict = {X_content: X_batch}
                train_op.run(feed_dict=feed_dict)
                
                end_time = time.time()
                delta_time = end_time - start_time
                
                if debug:
                    print('UID: {}, batch time: {}'.format(uid, delta_time))
                
                is_print_iter = int(num_batches) % print_iterations == 0
                
                if slow:
                    is_print_iter = epoch % print_iterations == 0
                
                is_last = epoch == epochs - 1 and num_batches * batch_size >= num_examples
                should_print = is_print_iter or is_last
                
                if should_print:
                    to_get = [style_loss, content_loss, denoise_loss, loss, output_image]
                    
                    # Run test.
                    test_feed_dict = {X_content: X_batch}
                    _style_loss, _content_loss, _denoise_loss, _loss, _preds = \
                    sess.run(to_get, test_feed_dict)
                    
                    
                    losses = (_style_loss, _content_loss, _denoise_loss, _loss)
                    
                    if slow:
                        _preds = vgg19.unprocess(_preds)
                    
                    # Save the checkpoint for this operation.
                    else:
                        saver = tf.train.Saver()
                        saver.save(sess, save_path)
                    
                    yield(_preds, losses, num_batches, epoch)


# # Parameters for training.
# NUM_EPOCHS = 2
# BATCH_SIZE = 20
# CHECKPOINT_ITERATIONS = 1000
# LEARNING_RATE = 1e-3

# # Parameters for loss functions.
# CONTENT_WEIGHT = 1.5e1
# STYLE_WEIGHT = 1e2
# DENOISE_WEIGHT = 2e2

# # Parameters for files
# CHECKPOINT_DIR = 'checkpoints'
# STYLE_PATH = 'images/udnie.jpg'
# TRAIN_PATH = 'train2014'
# TEST_PATH = 'images/tinghao.jpg'
# TEST_DIR = 'images'
# VGG_PATH = 'imagenet-vgg-verydeep-19.mat'
# SLOW = False


# style_target = get_image(STYLE_PATH)

# if not SLOW:
#     content_target = get_files(TRAIN_PATH)
# elif TEST_PATH:
#     content_target = [TEST_PATH]

# if SLOW:
#     if NUM_EPOCHS < 10:
#         NUM_EPOCHS = 1000
#     if LEARNING_RATE < 1:
#         LEARNING_RATE = 1e1

# kwargs = {'slow': SLOW,
#           'epochs': NUM_EPOCHS,
#           'print_iterations': CHECKPOINT_ITERATIONS,
#           'batch_size': BATCH_SIZE,
#           'save_path': os.path.join(CHECKPOINT_DIR, 'fns.ckpt'),
#           'learning_rate': LEARNING_RATE}

# args = [content_target,
#         style_target,
#         CONTENT_WEIGHT,
#         STYLE_WEIGHT,
#         DENOISE_WEIGHT,
#         VGG_PATH]

# for preds, losses, i, epoch in style_transfer(*args, **kwargs):
#     style_loss, content_loss, denoise_loss, loss = losses

#     print('Epoch {}, Iteration: {}, Loss: {}'.format(epoch, i, loss))

#     print('style loss: {}, content loss: {}, denoise loss: {}'.format(
#         style_loss, content_loss, denoise_loss))

#     if TEST_PATH:
#         assert TEST_DIR != False
#         preds_path = '{}/{}_{}.png'.format(TEST_DIR, epoch, i)

#         if not SLOW:
#             ckpt_dir = os.path.dirname(CHECKPOINT_DIR)
#             play.feed_forward(TEST_PATH, preds_path, CHECKPOINT_DIR)
#         else:
#             save_image(preds_path, preds)

# cmd_text = 'python play.py --checkpoint {} ...'.format(CHECKPOINT_DIR)
# print('Training complete. For evaluation:\n    `{}`'.format(cmd_text))