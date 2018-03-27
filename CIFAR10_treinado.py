import tensorflow as tf
import numpy as np
from PIL import Image
from resizeimage import resizeimage
import matplotlib.pyplot as plt


#Placeholders
x = tf.placeholder(tf.float32, shape = [None, 32, 32, 3])

#Helpers
def init_weights(shape):
     init_random_dist = tf.truncated_normal(shape, stddev = 0.1)
     return tf.Variable(init_random_dist)

def init_bias(shape):
    init_bias_vals = tf.constant(0.1, shape = shape)
    return tf.Variable(init_bias_vals)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = "SAME")

def max_pool_2by2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

def convolutional_layer(input_x, shape):
    W = init_weights(shape)
    b = init_bias([shape[3]])
    return tf.nn.relu(conv2d(input_x, W) + b)

def normal_full_layer(input_layer, size):
    input_size = int(input_layer.get_shape()[1])
    W = init_weights([input_size, size])
    b = init_bias([size])
    return tf.matmul(input_layer, W) + b


#Creating Layers
convo_1 = convolutional_layer(x, shape = [4, 4, 3, 32])
convo_1_pooling = max_pool_2by2(convo_1)

convo_2 = convolutional_layer(convo_1_pooling, shape = [4, 4, 32, 64])
convo_2_pooling = max_pool_2by2(convo_2)

convo_3 = convolutional_layer(convo_2_pooling, shape = [4, 4, 64, 128])
convo_3_pooling = max_pool_2by2(convo_3)

convo_3_flat = tf.reshape(convo_3_pooling, [-1, 4*4*128])

full_layer_1 = tf.nn.relu(normal_full_layer(convo_3_flat, 1024))

y_pred = normal_full_layer(full_layer_1, 10)

########## FEED THE IMAGE YOU WANT PROVIDING THE DIRECTORY INSIDE THE EMPTY STRING ################
img = Image.open("") #Image Here
img_rsz = resizeimage.resize_contain(img, [32, 32])
img_input = img_rsz.convert("RGB")
img_input = np.asarray(img_input)
img_input = tf.reshape(img_input, [-1, 32, 32, 3])

saver = tf.train.Saver()

#GRAPH SESSION
with tf.Session() as sess:
    saver.restore(sess, "trained_model_CIFAR10/cifar10_cnn.ckpt")

    match = tf.argmax(y_pred, 1)

    print("\nThe image is: ")
    guess = sess.run(match, feed_dict = {x: img_input.eval()})

    labels = {"l0": "Airplane!", "l1": "Automobile!", "l2": "Bird!", "l3": "Cat!", "l4": "Deer!", "l5": "Dog!",
              "l6": "Frog!", "l7": "Horse!", "l8": "Ship!", "l9": "Truck!"}
    predictions = []
    for i in range(10):
        if guess == [i]:
            prediction = labels["l" + str(i)]
            print(prediction)
            predictions.append(prediction)

    #PLOTTING IMAGE AND POST-FILTERS
    after_filter_1 = convo_1_pooling
    img_plot = sess.run(after_filter_1, feed_dict = {x: img_input.eval()})
    after_filter_2 = convo_2_pooling
    img_plot_2 = sess.run(after_filter_2, feed_dict = {x: img_input.eval()})
    after_filter_3 = convo_3_pooling
    img_plot_3 = sess.run(after_filter_3, feed_dict = {x: img_input.eval()})
    fig = plt.figure()
    fig.canvas.set_window_title('Images and Results')
    plt.axis("off")
    ax1 = fig.add_subplot(3, 2, 1)
    ax1.imshow(img_rsz)
    plt.title("Input")
    plt.xlabel("px")
    plt.ylabel("px")
    ax2 = fig.add_subplot(3, 2, 2)
    ax2.imshow(img_plot[0, :, :, 0])
    plt.title("Convolution 1")
    plt.xlabel("px")
    plt.ylabel("px")
    ax3 = fig.add_subplot(3, 2, 3)
    ax3.imshow(img_plot_2[0, :, :, 0])
    plt.title("Convolution 2")
    plt.xlabel("px")
    plt.ylabel("px")
    ax4 = fig.add_subplot(3, 2, 4)
    ax4.imshow(img_plot_3[0, :, :, 0])
    plt.title("Convolution 3")
    plt.xlabel("px")
    plt.ylabel("px")
    ax5 = fig.add_subplot(3, 2, 5)
    ax5.imshow(img)
    plt.title("Original")
    ax6 = fig.add_subplot(3, 2, 6, adjustable = "box")
    t = plt.text(.5, .5, predictions[0], horizontalalignment='center', verticalalignment='center', size = 20, bbox=dict(facecolor='green', alpha=0.5, boxstyle="round"))
    plt.axis("off")
    plt.title("Output")
    fig.subplots_adjust(wspace = .001,hspace = 0.8)
    plt.show()
