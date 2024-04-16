import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.applications import VGG19
from keras.preprocessing import image as kp_image
from keras import Model
from keras import layers
import time

# Load VGG19 model pretrained on ImageNet dataset
vgg = VGG19(include_top=False, weights='imagenet')
vgg.trainable = False

# Content layer where will pull our feature maps
content_layers = ['block5_conv2'] 

# Style layer we are interested in
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1', 
                'block4_conv1', 
                'block5_conv1']

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)

# Function to load and preprocess images
def load_img(path_to_img):
    max_dim = 512
    img = kp_image.load_img(path_to_img)
    img = kp_image.img_to_array(img)

    img = tf.image.resize(img, (max_dim,max_dim))
    img = img/255.0
    return img

# Function to display the image
def imshow(image, title=None):
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)

    plt.imshow(image)
    if title:
        plt.title(title)

# Function to preprocess image for VGG
def preprocess_img(image):
    image = tf.cast(image, dtype=tf.float32)
    image = keras.applications.vgg19.preprocess_input(image)
    return image

# Function to deprocess image for display
def deprocess_img(processed_img):
    x = processed_img.copy()
    if len(x.shape) == 4:
        x = np.squeeze(x, 0)
    assert len(x.shape) == 3, ("Input to deprocess image must be an image of "
                             "dimension [1, height, width, channel] or [height, width, channel]")
    if len(x.shape) != 3:
        raise ValueError("Invalid input to deprocessing image")

    # Perform the inverse of the preprocessiing step
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]

    x = np.clip(x, 0, 255).astype('uint8')
    return x

# Function to get content and style representations
def get_feature_representations(model, content_path, style_path):
    # Load and preprocess content image
    content_image = load_img(content_path)
    content_image = preprocess_img(content_image)

    # Load and preprocess style image
    style_image = load_img(style_path)
    style_image = preprocess_img(style_image)

    # Get the content and style feature representations
    style_outputs = model(style_image)
    content_outputs = model(content_image)

    # Get the style features
    style_features = [style_layer[0] for style_layer in style_outputs[:num_style_layers]]

    # Get the content features
    content_features = [content_layer[0] for content_layer in content_outputs[num_style_layers:]]
    return style_features, content_features

# Function to compute gram matrix
def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    return result/(num_locations)

# Function to compute the style loss
def style_loss(style_targets, style_outputs):
    style_loss = tf.add_n([tf.reduce_mean((style_targets[i]-style_outputs[i])**2) 
                           for i in range(len(style_targets))])
    return style_loss

# Function to compute the content loss
def content_loss(content_targets, content_outputs):
    content_loss = tf.add_n([tf.reduce_mean((content_targets[i]-content_outputs[i])**2) 
                             for i in range(len(content_targets))])
    return content_loss

# Function to compute total variation loss
def total_variation_loss(image):
    x_deltas = image[:,:,1:,:] - image[:,:,:-1,:]
    y_deltas = image[:,1:,:,:] - image[:,:-1,:,:]

    return tf.reduce_sum(tf.abs(x_deltas)) + tf.reduce_sum(tf.abs(y_deltas))

# Function to compute total loss
def compute_loss(model, loss_weights, init_image, gram_style_features, content_features):
    style_weight, content_weight = loss_weights

    model_outputs = model(init_image)

    style_output_features = model_outputs[:num_style_layers]
    content_output_features = model_outputs[num_style_layers:]

    style_score = style_loss(gram_style_features, style_output_features)
    content_score = content_loss(content_features, content_output_features)
    tv_score = total_variation_loss(init_image)

    style_score *= style_weight / num_style_layers
    content_score *= content_weight / num_content_layers
    tv_score *= 1e-6
    return style_score + content_score + tv_score

# Function to compute gradients
@tf.function()
def compute_grads(cfg):
    with tf.GradientTape() as tape:
        all_loss = compute_loss(**cfg)
    total_loss = all_loss
    return tape.gradient(total_loss, cfg['init_image']), all_loss

# Function to run style transfer
def style_transfer(content_path, style_path, num_iterations=1000, content_weight=1e3, style_weight=1e-2): 
    model = vgg

    for layer in model.layers:
        layer.trainable = False

    style_features, content_features = get_feature_representations(model, content_path, style_path)
    gram_style_features = [gram_matrix(style_feature) for style_feature in style_features]

    init_image = load_img(content_path)
    init_image = tf.Variable(init_image, dtype=tf.float32)

    opt = tf.optimizers.Adam(learning_rate=5, beta_1=0.99, epsilon=1e-1)

    best_loss, best_img = float('inf'), None

    loss_weights = (style_weight, content_weight)

    cfg = {
        'model': model,
        'loss_weights': loss_weights,
        'init_image': init_image,
        'gram_style_features': gram_style_features,
        'content_features': content_features
    }

    norm_means = np.array([103.939, 116.779, 123.68])
    min_vals = -norm_means
    max_vals = 255 - norm_means   

    imgs = []
    for i in range(num_iterations):
        grads, all_loss = compute_grads(cfg)
        loss, style_score, content_score, tv_score = all_loss
        opt.apply_gradients([(grads, init_image)])
        clipped = tf.clip_by_value(init_image, min_vals, max_vals)
        init_image.assign(clipped)
        if loss < best_loss:
            best_loss = loss
            best_img = deprocess_img(init_image.numpy())
        if i % 100 == 0:
            print(f"Iteration: {i}, Total loss: {loss}")
    return best_img

content_path = 'content_image.jpg'
style_path = 'style_image.jpg'
output_image = style_transfer(content_path, style_path)
plt.imshow(output_image)
plt.axis('off')
plt.show()