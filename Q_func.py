from __future__ import absolute_import, division, print_function, unicode_literals

# import matplotlib.pyplot as plt
import tensorflow as tf
keras = tf.keras
import numpy as np


def format_example(image):

    IMG_SIZE = 160  # All images will be resized to 160x160
    image = tf.cast(image, tf.float32)
    image = (image/127.5) - 1
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    return image


def load_model(train_batches):
    # SPLIT_WEIGHTS = (8, 1, 1)
    IMG_SIZE = 160  # All images will be resized to 160x160
    IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

    # Create the base model from the pre-trained model MobileNet V2
    base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                   include_top=False,
                                                   weights='imagenet')

    image_batch = train_batches[0]
    train_images = np.zeros((len(image_batch), image_batch[0].shape[0], image_batch[0].shape[1], 3))

    for i in range(len(image_batch)):
        train_images[i, :, :, :] = image_batch[i]

    train_images = format_example(train_images)

    feature_batch = base_model(train_images)
    # feature_batch = base_model(image_batch)

    # print(feature_batch.shape)

    base_model.trainable = False

    base_model.summary()

    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    feature_batch_average = global_average_layer(feature_batch)
    print(feature_batch_average.shape)

    # prediction_layer = keras.layers.Dense(3, activation='softmax')
    prediction_layer = keras.layers.Dense(3)
    prediction_batch = prediction_layer(feature_batch_average)
    print(prediction_batch.shape)

    model = tf.keras.Sequential([
        base_model,
        global_average_layer,
        prediction_layer
    ])

    base_learning_rate = 0.0001
    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
                  loss='categorical_crossentropy')  # ,metrics=['accuracy'])

    model.summary()

    # len(model.trainable_variables)

    return model


def Q_func_update(train_batches, model, epoches):

    image_batch = train_batches[0]
    train_images = np.zeros((len(image_batch), image_batch[0].shape[0], image_batch[0].shape[1], 3))

    for i in range(len(image_batch)):
        train_images[i, :, :, :] = image_batch[i]

    y = np.round(train_batches[1]).astype(int) - 1
    y = tf.keras.utils.to_categorical(y,num_classes=3, dtype='float32')

    train_images = format_example(train_images)

    history = model.fit([train_images, y], epochs=epoches, steps_per_epoch=1)

    return model
