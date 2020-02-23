import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import flatten
from sklearn.utils import shuffle
from ops import *


EPOCHS = 10
BATCH_SIZE = 64
global totalScore
totalScore = 0

data_folder = 'MNISTDD_train+valid/'

#load the data
bBoxesTrain = np.load(data_folder + 'train_bboxes.npy')
XValuesTrain = np.load(data_folder + 'train_X.npy')
YValuesTrain = np.load(data_folder + "train_Y.npy")
y_train = np.load(data_folder + "train_Y.npy")
YTrainNew = y_train
print("XValuesTrain", XValuesTrain.shape)
print("YValuesTrain", YValuesTrain.shape)

#reshape
trainImagesHolder = []
for i in range(XValuesTrain.shape[0]):
    reshapedImage = XValuesTrain[i].reshape(64, 64)
    reshapedImage = np.expand_dims(reshapedImage, axis=2)  # expand to 64x64x1
    trainImagesHolder.append(reshapedImage)
np.save("trainImages", trainImagesHolder)

#load data
bBoxesValidation = np.load(data_folder + 'valid_bboxes.npy')
XValuesValidation = np.load(data_folder + 'valid_X.npy')
YValuesValidation = np.load(data_folder + "valid_Y.npy")
y_val = np.load(data_folder + "valid_Y.npy")
YValidNew = y_val
print("XValuesValidation", XValuesValidation.shape)
print("YValuesValidation", YValuesValidation.shape)

#reshape
validationImagesHolder = []
for i in range(XValuesValidation.shape[0]):
    reshapedImage = XValuesValidation[i].reshape(64, 64)
    reshapedImage = np.expand_dims(reshapedImage, axis=2)  # expand to 64x64x1
    validationImagesHolder.append(reshapedImage)
np.save("validationImages", validationImagesHolder)

XTrain = np.load("trainImages.npy")
XValid = np.load("validationImages.npy")
print("XTrainValues reshaped", XTrain.shape)
print("XValidationValues reshaped", XValid.shape)


def LeNet(x):
    # Layer 1: Convolutional.
    num_channels, filter_size, num_filters, strides = 1, 3, 32, [1, 2, 2, 1]
    conv_layer1, conv_layer_pool1 = create_conv_layer(x, filter_size, num_channels, num_filters, strides,
                                                      use_batch_norm=False, use_max_pool=True,
                                                      layer_name="conv_layer1")

    print("conv_layer1", conv_layer1)

    # Layer 2: Convolutional.
    num_channels, filter_size, num_filters, strides = 32, 3, 64, [1, 1, 1, 1]
    conv_layer2, conv_layer_pool2 = create_conv_layer(conv_layer1, filter_size, num_channels, num_filters, strides,
                                                      use_batch_norm=False, use_max_pool=True,
                                                      layer_name="conv_layer2")

    print("conv_layer2", conv_layer2)

    # Layer 3: Convolutional.
    num_channels, filter_size, num_filters, strides = 64, 3, 64, [1, 1, 1, 1]
    conv_layer3, conv_layer_pool3 = create_conv_layer(conv_layer2, filter_size, num_channels, num_filters, strides,
                                                      use_batch_norm=False, use_max_pool=True,
                                                      layer_name="conv_layer3")

    print("conv_layer3", conv_layer3)

    conv_layer4_flat = tf.contrib.layers.flatten(conv_layer3)  # Flatten

    print("conv_layer4_flat", conv_layer4_flat)

    # Layer 4: Fully Connected.
    num_inputs, num_outputs = int(conv_layer4_flat.shape[1]), 84
    fc1 = create_fully_connected(conv_layer4_flat, num_inputs, num_outputs)

    print("fc1", fc1)

    # Layer 5: Fully Connected.
    num_inputs, num_outputs = 84, 64
    fc2 = create_fully_connected(fc1, num_inputs, num_outputs)

    print("fc2", fc2)


    num_inputs, num_outputs = 64, 10
    logits = create_fully_connected(fc2, num_inputs, num_outputs, use_relu=False)

    print("logits", logits)

    return logits

# Begin definition of graph
imagesPlaceholder = tf.placeholder(tf.float32, shape=[None, 64, 64, 1])
labelsPlaceholder = tf.placeholder(tf.int64, shape=[None, 2])


rate = 0.001

logits = LeNet(imagesPlaceholder)
logits2 = LeNet(imagesPlaceholder)
y_pred = [logits, logits2]
y_pred_cls = tf.transpose(tf.argmax(y_pred, axis=2))
print("y_pred_cls shape", y_pred_cls.shape)

loss1 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labelsPlaceholder[:, 0]))
loss2 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits2, labels=labelsPlaceholder[:, 1]))
loss = loss1 + loss2

optimizer = tf.train.AdamOptimizer(learning_rate = rate)
grads_and_vars = optimizer.compute_gradients(loss, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
training_operation = optimizer.apply_gradients(grads_and_vars)

def accuracy(predictions, labels):
    return (100.0 * np.sum(predictions == labels) / predictions.shape[1] / predictions.shape[0])

def add_gradient_summaries(grads_and_vars):
    for grad, var in grads_and_vars:
        if grad is not None:
            tf.summary.histogram(var.op.name + "/gradient", grad)

for var in tf.trainable_variables():
    tf.summary.histogram(var.op.name + "/histogram", var)

add_gradient_summaries(grads_and_vars)
tf.summary.scalar('loss_operation', loss)
merged_summary_op= tf.summary.merge_all()
summary_writer = tf.summary.FileWriter('logs/')

saver = tf.train.Saver(max_to_keep=0)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(XTrain)

    print("Training...")
    global_step = 0
    for i in range(EPOCHS):
        X_train, y_train = shuffle(XTrain, YTrainNew)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            _, summaries = sess.run([training_operation, merged_summary_op], feed_dict={imagesPlaceholder: batch_x, labelsPlaceholder: batch_y})
            if global_step % 100 == 1:
                summary_writer.add_summary(summaries, global_step=global_step)
            global_step += 1

        print("EPOCH {} ...".format(i + 1))
        batch_predictions = sess.run(y_pred_cls, feed_dict={imagesPlaceholder: batch_x, labelsPlaceholder: batch_y})
        print("Test Minibatch accuracy at step %d: %.4f" % (global_step, accuracy(batch_predictions, batch_y)))

        val_predictions = sess.run(y_pred_cls, feed_dict={imagesPlaceholder: XValid, labelsPlaceholder: YValidNew})
        print("Validation accuracy at step %d: %.4f" % (global_step, accuracy(val_predictions, YValidNew)))
        print()
        saver.save(sess, 'ckpt/lenet', global_step=i)

    print("Model saved")
