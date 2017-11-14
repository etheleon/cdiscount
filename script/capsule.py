import numpy as np
import feather
import os
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from keras import callbacks
from keras.utils.vis_utils import plot_model

from keras.callbacks import TensorBoard
from time import time

from capsuleclass import Length, Mask, PrimaryCap, CapsuleLayer

from keras import layers, models
from keras import backend as K
from keras.utils import to_categorical

from sklearn.model_selection import train_test_split
import cv2
from PIL import Image
import bson

def CapsNet(input_shape, n_class, num_routing):
    """
    A Capsule Network on MNIST.
    :param input_shape: data shape, 4d, [None, width, height, channels]
    :param n_class: number of classes
    :param num_routing: number of routing iterations
    :return: A Keras Model with 2 inputs and 2 outputs
    """
    x = layers.Input(shape=input_shape)

    # Layer 1: Just a conventional Conv2D layer
    conv1 = layers.Conv2D(filters=256, kernel_size=9, strides=1, padding='valid', activation='relu', name='conv1')(x)

    # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_vector]
    primarycaps = PrimaryCap(conv1, dim_vector=8, n_channels=32, kernel_size=9, strides=2, padding='valid')

    # Layer 3: Capsule layer. Routing algorithm works here.
    digitcaps = CapsuleLayer(num_capsule=n_class, dim_vector=16, num_routing=num_routing, name='digitcaps')(primarycaps)

    # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
    # If using tensorflow, this will not be necessary. :)
    out_caps = Length(name='out_caps')(digitcaps)

    # Decoder network.
    y = layers.Input(shape=(n_class,))
    masked = Mask()([digitcaps, y])  # The true label is used to mask the output of capsule layer.
    x_recon = layers.Dense(512, activation='relu')(masked)
    x_recon = layers.Dense(1024, activation='relu')(x_recon)
    x_recon = layers.Dense(128*128*3, activation='sigmoid')(x_recon)
    x_recon = layers.Reshape(target_shape=[128, 128, 3], name='out_recon')(x_recon)

    # two-input-two-output keras Model
    return models.Model([x, y], [out_caps, x_recon])

def margin_loss(y_true, y_pred):
    """
    Margin loss for Eq.(4). When y_true[i, :] contains not just one `1`, this loss should work too. Not test it.
    :param y_true: [None, n_classes]
    :param y_pred: [None, num_capsule]
    :return: a scalar loss value.
    """
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))

    return K.mean(K.sum(L, 1))


# define model
model = CapsNet(input_shape=[128, 128, 3],
                n_class=5270,
                num_routing=3)
model.summary()

try:
    plot_model(model, to_file='model.png', show_shapes=True)
except Exception as e:
    print('No fancy plot {}'.format(e))

#mnist###################################################################################################
#      data_train = pd.read_csv('/data/uesu/cdiscount/data/mnist/train.csv')
#      X_full = data_train.iloc[:,1:]
#      y_full = data_train.iloc[:,:1]
#      x_train, x_test, y_train, y_test = train_test_split(X_full, y_full, test_size = 0.3)
#      # x_train, x_test, y_train, y_test = train_test_split(test[0], test[1], test_size = 0.3)
# #
#      x_train = x_train.values.reshape(-1, 28, 28, 1).astype('float32') / 255.
#      #[num, width, height, channel]
#      x_test = x_test.values.reshape(-1, 28, 28, 1).astype('float32') / 255.
#      #[num, width, height, channel]
#      y_train = to_categorical(y_train.astype('float32'))
#      y_test = to_categorical(y_test.astype('float32'))
####################################################################################################

# slow method
# def myown():
#     i = 0
#     for example in bson.decode_file_iter(open("/data/uesu/cdiscount/cdiscount-kernel/data/train.bson", 'rb')):
#         print(i)
#         i = i + 1
#         if i > 100:
#             break
#         else:
#             for image in example['imgs']:
#                 Image.fromarray(cv2.imdecode(np.frombuffer(image['picture'], dtype=np.uint8), cv2.IMREAD_COLOR)).save("/data/uesu/cdiscount/sampleImages/{}.jpg".format(i))
#     print("done")
# myown()
#
# for i in range(100):
#     imagesss = next(slowGen)
#     Image.fromarray(imagesss.astype(np.uint8)).save("/data/uesu/cdiscount/sampleImages/{}.jpg".format(i))

def myFastGenerator(length=100000):
    """
        file1 = open("/data/uesu/cdiscount/cdiscount-kernel/data/images_128x128.bin", 'rb')
        file1.seek(0)
        test = file1.read(7555)
        aimage = cv2.imdecode(np.frombuffer(test, dtype=np.uint8), cv2.IMREAD_COLOR)
        file1.seek(7555)
        test = file1.read(7853)
        bimage = cv2.imdecode(np.frombuffer(test, dtype=np.uint8), cv2.IMREAD_COLOR)
    """
    #labels = pd.read_csv("/data/uesu/cdiscount/cdiscount-kernel/data/images_128x128.csv").head(length)
    labels = feahter.read_dataframe("/data/uesu/cdiscount/data/meta.feather").head(length)
    with open("/data/uesu/cdiscount/cdiscount-kernel/data/images_128x128.bin", 'rb') as file_:
        # for class_, offset, size in list(examples)[start:end]:
        for startend in list(zip([start for start in range(1,length+1, 100)],[end for end in range(100,length+100, 100)])):
            examples = zip(
                labels['class'][startend[0]:startend[1]],
                labels['offset'][startend[0]:startend[1]],
                labels['size'][startend[0]:startend[1]]
                )
            tempArr = []
            for class_, offset, size in examples:
                file_.seek(offset)
                tempArr.append(cv2.imdecode(np.frombuffer(file_.read(size), dtype=np.uint8), cv2.IMREAD_COLOR))
            yield (np.concatenate([y[np.newaxis,:,:,:] for y in tempArr], axis=0), labels['class'][startend[0]:startend[1]].values)

def myFastGeneratorVal(length=100000):
    """
        file1 = open("/data/uesu/cdiscount/cdiscount-kernel/data/images_128x128.bin", 'rb')
        file1.seek(0)
        test = file1.read(7555)
        aimage = cv2.imdecode(np.frombuffer(test, dtype=np.uint8), cv2.IMREAD_COLOR)
        file1.seek(7555)
        test = file1.read(7853)
        bimage = cv2.imdecode(np.frombuffer(test, dtype=np.uint8), cv2.IMREAD_COLOR)
    """
    # labels = pd.read_csv("/data/uesu/cdiscount/cdiscount-kernel/data/images_128x128.csv").tail(length)
    labels = feahter.read_dataframe("/data/uesu/cdiscount/data/meta.feather").tail(length)
    with open("/data/uesu/cdiscount/cdiscount-kernel/data/images_128x128.bin", 'rb') as file_:
        # for class_, offset, size in list(examples)[start:end]:
        for startend in list(zip([start for start in range(1,length+1, 100)],[end for end in range(100,length+100, 100)])):
            examples = zip(
                labels['class'][startend[0]:startend[1]],
                labels['offset'][startend[0]:startend[1]],
                labels['size'][startend[0]:startend[1]]
                )
            tempArr = []
            for class_, offset, size in examples:
                file_.seek(offset)
                tempArr.append(cv2.imdecode(np.frombuffer(file_.read(size), dtype=np.uint8), cv2.IMREAD_COLOR))
            yield (np.concatenate([y[np.newaxis,:,:,:] for y in tempArr], axis=0), labels['class'][startend[0]:startend[1]].values)

def train(data, model, epoch_size_frac=1.0):
    """
    Training a CapsuleNet
    :param model: the CapsuleNet model
    :param data: a tuple containing training and testing data, like `((x_train, y_train), (x_test, y_test))`
    :param args: arguments
    :return: The trained model
    """
    # unpacking the data
    (x_train, y_train), (x_test, y_test) = data
    # generator for data cause too big
    y_train = to_categorical(y_train.astype('float32'), 5270)
    y_test = to_categorical(y_test.astype('float32'), 5270)

    # callbacks
    log = callbacks.CSVLogger('log.csv')
    checkpoint = callbacks.ModelCheckpoint('weights-{epoch:02d}.h5',
                                           save_best_only=True, save_weights_only=True, verbose=1)
    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: 0.001 * np.exp(-epoch / 10.))

    # compile the model
    model.compile(optimizer='adam',
                  loss=[margin_loss, 'mse'],
                  loss_weights=[1., 0.0005],
                  metrics={'out_caps': 'accuracy'})

    # """
    # Training without data augmentation:
    model.fit([x_train, y_train], [y_train, x_train], batch_size=20, epochs=10, validation_data=[[x_test, y_test], [y_test, x_test]], callbacks=[log, checkpoint])
    # """

    # -----------------------------------Begin: Training with data augmentation -----------------------------------#
    def train_generator(x, y, batch_size, shift_fraction=0.):
        train_datagen = ImageDataGenerator(width_shift_range=shift_fraction,
                                           height_shift_range=shift_fraction)  # shift up to 2 pixel for MNIST
        generator = train_datagen.flow(x, y, batch_size=batch_size)
        while 1:
            x_batch, y_batch = generator.next()
            yield ([x_batch, y_batch], [y_batch, x_batch])

    # Training with data augmentation. If shift_fraction=0., also no augmentation.
    tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

    # model.fit_generator(generator=train_generator(x_train, y_train, 64, 0.1),
    # model.fit_generator(generator=myFastGenerator(),
    #                     # steps_per_epoch=int(epoch_size_frac*y_train.shape[0] / 64),
    #                     steps_per_epoch=100,
    #                     epochs=10,
    #                     validation_data=myFastGeneratorVal(),
    #                     validation_steps=100,
    #   )
                        # callbacks=[log, checkpoint, lr_decay])
                        # callbacks=[log, checkpoint, lr_decay, tensorboard])
    # -----------------------------------End: Training with data augmentation -----------------------------------#

    model.save_weights('trained_model.h5')
    print('Trained model saved to \'trained_model.h5\'')

    return model


train(data, model=model, epoch_size_frac = 0.1) # do 10% of an epoch (takes too long)

import matplotlib
matplotlib.use("Agg")

def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(np.sqrt(num))
    height = int(np.ceil(float(num)/width))
    shape = generated_images.shape[1:3]
    image = np.zeros((height*shape[0], width*shape[1]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = \
            img[:, :, 0]
    return image

def test(model, data):
    x_test, y_test = data
    y_pred, x_recon = model.predict([x_test, y_test], batch_size=100)
    print('-'*50)
    print('Test acc:', np.sum(np.argmax(y_pred, 1) == np.argmax(y_test, 1))/y_test.shape[0])

    import matplotlib.pyplot as plt
    from PIL import Image

    img = combine_images(np.concatenate([x_test[:50],x_recon[:50]]))
    image = img * 255
    Image.fromarray(image.astype(np.uint8)).save("real_and_recon.png")
    print()
    print('Reconstructed images are saved to ./real_and_recon.png')
    print('-'*50)
    plt.imshow(plt.imread("real_and_recon.png", ))
    plt.show()

test(model=model, data=(x_test[:100], y_test[:100]))
