# This code trains different combinations of GAN models to colorize grayscale images.
# Generator and discriminator models, hyperparameters are different for each model.
# After every training results are saved for comparison.
# This code created by alperenlcr@gmail.com and sevval.bulburu@std.yildiz.edu.tr


###############
### IMPORTS ###
###############

import os, sys, cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
from time import time
from skimage.color import rgb2lab, lab2rgb
from tensorflow.keras.layers import ( # type: ignore
    Activation, AveragePooling2D, BatchNormalization, Conv2D, Conv2DTranspose,
    Dense, Dropout, Flatten, Input, LeakyReLU, ReLU, UpSampling2D)
from tensorflow.keras.datasets import cifar10 # type: ignore
from tensorflow.keras.models import Sequential, Model # type: ignore


#################
### CONSTANTS ###
#################

IMAGE_SIZE = 32
EPOCHS = 150
BATCH_SIZE = 128    # this will be changed in the testing loop
SHUFFLE_BUFFER_SIZE = 100
WORKDIR = "/home/alperenlcr/bitirme/gan-image-colorizer"


######################################
### DATASET LOAD AND PREPROCESSING ###
######################################

# Load CIFAR-10 dataset and convert images to LAB color space
def generate_dataset(images, debug=False):
    X = []
    Y = []

    for i in images:
        lab_image_array = rgb2lab(i / 255)
        x = lab_image_array[:, :, 0]
        y = lab_image_array[:, :, 1:]
        y /= 128  # normalize

        if debug:
            fig = plt.figure()
            fig.add_subplot(1, 2, 1)
            plt.imshow(i / 255)

            fig.add_subplot(1, 2, 2)
            plt.imshow(lab2rgb(np.dstack((x, y * 128))))
            plt.show()

        X.append(x.reshape(IMAGE_SIZE, IMAGE_SIZE, 1))
        Y.append(y)

    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)

    return X, Y

# Load data from computer if exists, otherwise download and preprocess
def load_data(force=False):
    is_saved_arrays_exist = os.path.isfile(os.path.join(WORKDIR, 'X_train.npy'))

    if not is_saved_arrays_exist or force:
        (train_images, _), (test_images, _) = cifar10.load_data()
        X_train, Y_train = generate_dataset(train_images)
        X_test, Y_test = generate_dataset(test_images)
        print('Saving processed data to computer')
        np.save(os.path.join(WORKDIR, 'X_train.npy'), X_train)
        np.save(os.path.join(WORKDIR, 'Y_train.npy'), Y_train)
        np.save(os.path.join(WORKDIR, 'X_test.npy'), X_test)
        np.save(os.path.join(WORKDIR, 'Y_test.npy'), Y_test)
    else:
        print('Loading processed data from computer')
        X_train = np.load(os.path.join(WORKDIR, 'X_train.npy'))
        Y_train = np.load(os.path.join(WORKDIR, 'Y_train.npy'))
        X_test = np.load(os.path.join(WORKDIR, 'X_test.npy'))
        Y_test = np.load(os.path.join(WORKDIR, 'Y_test.npy'))

    return X_train, Y_train, X_test, Y_test


X_train, Y_train, X_test, Y_test = load_data()

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test))

train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)


##############
### MODELS ###
##############

def generator1():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 1)))
    model.add(Conv2D(64, (3, 3), padding='same', strides=2, activation='relu'))
    model.add(BatchNormalization())
    # model = MaxPooling2D(pool_size=(2, 2))(model)

    model.add(Conv2D(128, (3, 3), padding='same', activation='relu', strides=2))
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    # model = MaxPooling2D(pool_size=(2, 2))(model)

    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(UpSampling2D(size=(2,2)))
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())

    model.add(UpSampling2D(size=(2,2)))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())

    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(2, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    # self.model = BatchNormalization()(self.model)
    # self.model = merge(inputs=[self.g_input, self.model], mode='concat')
    # self.model = Activation('linear')(self.model)
    return model

def generator2(): # autoencoder generator model
    def downsample(filters, kernel_size, apply_batchnorm=True):
        initializer = tf.random_uniform_initializer(0, 0.02)
        model = Sequential()
        model.add(Conv2D(filters, kernel_size, strides=2, padding='same',
                        kernel_initializer=initializer, use_bias=False))
        
        if apply_batchnorm:
            model.add(BatchNormalization())

        model.add(LeakyReLU())
        return model


    def upsample(filters, kernel_size, apply_dropout=False):
        initializer = tf.random_uniform_initializer(0, 0.02)
        model = Sequential()
        model.add(Conv2DTranspose(filters, kernel_size, strides=2, padding='same',
                                kernel_initializer=initializer, use_bias=False))
        model.add(BatchNormalization())

        if apply_dropout:
            model.add(Dropout(0.5))

        model.add(ReLU())
        return model

    inputs = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1))

    # Downsampling layers
    # 1: (BATCH_SIZE, 16, 16, 32)
    # 2: (BATCH_SIZE, 8, 8, 64)
    # 3: (BATCH_SIZE, 4, 4, 128)
    # 4: (BATCH_SIZE, 2, 2, 256)
    # 5: (BATCH_SIZE, 1, 1, 256)

    downstack = [
        downsample(32, 4, apply_batchnorm=False),
        downsample(64, 4),
        downsample(128, 4),
        downsample(256, 4),
        downsample(256, 4)
    ]

    # Upsampling layers
    # 1: (BATCH_SIZE, 1, 1, 256)
    # 2: (BATCH_SIZE, 1, 1, 128)
    # 3: (BATCH_SIZE, 1, 1, 64)
    # 4: (BATCH_SIZE, 1, 1, 32)
    
    upstack = [
        upsample(256, 4, apply_dropout=True),
        upsample(128, 4),
        upsample(64, 4),
        upsample(32, 4),
    ]

    initializer = tf.random_uniform_initializer(0, 0.02)
    output_layer = Conv2DTranspose(2, 3, strides=2, padding='same',
                                   kernel_initializer=initializer,
                                   activation='tanh')
    
    x = inputs

    # Downsampling layers
    skips = []
    for dm in downstack:
        x = dm(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling layers
    for um, skip in zip(upstack, skips):
        x = um(x)
        x = tf.keras.layers.Concatenate()([x, skip])
    
    x = output_layer(x)

    return Model(inputs=inputs, outputs=x)


def discriminator1():
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(4, 4), strides=(2, 2), padding='SAME', activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))) # input_shape=(32, 32, 3)
    model.add(Conv2D(filters=64, kernel_size=(4, 4), strides=(2, 2), padding='SAME', activation='relu')) # input_shape=(16, 16, 32)
    model.add(Dropout(.25))
    model.add(Conv2D(filters=128, kernel_size=(4, 4), strides=(2, 2), padding='SAME', activation='relu')) # input_shape=(8, 8, 64)
    model.add(Dropout(.25))
    # current_shape=(4, 4, 128)

    model.add(Flatten()) # 2048
    model.add(Dense(256))   # 256
    model.add(LeakyReLU(.2))
    model.add(BatchNormalization())
    model.add(Dropout(.5))
    model.add(Dense(1))    # 1
    model.add(Activation('sigmoid'))

    return model


def discriminator2():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)))   # input_shape=(32, 32, 3)
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))    # 32x32x32
    model.add(AveragePooling2D(pool_size=(2, 2)))   # 16x16x32
    model.add(Dropout(.25)) # 16x16x32

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))    # 16x16x64
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))    # 16x16x64
    model.add(AveragePooling2D(pool_size=(2, 2)))   # 8x8x64
    model.add(Dropout(.25)) # 8x8x64

    model.add(Flatten())    # 4096
    model.add(Dense(512))   # 512
    model.add(LeakyReLU(.2))
    model.add(BatchNormalization())
    model.add(Dropout(.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    return model


######################
### LOSS FUNCTIONS ###
######################

LAMBDA = 100
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = cross_entropy(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = cross_entropy(tf.zeros_like(disc_generated_output), disc_generated_output)
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss


def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = cross_entropy(tf.ones_like(disc_generated_output), disc_generated_output)
    # mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    total_gen_loss = gan_loss + (LAMBDA * l1_loss)
    return total_gen_loss, gan_loss, l1_loss


################
### TRAINING ###
################

def train(generator, discriminator, generator_optimizer, discriminator_optimizer, batch_size, csv_file):
    BATCH_SIZE = batch_size
    print('Training started')
    checkpoint_dir = os.path.join(csv_file[:-4]+'_checkpoints')
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    if generator_optimizer['name'] == 'adam':
        generator_optimizer = tf.keras.optimizers.Adam(learning_rate=generator_optimizer['learning_rate'], beta_1=generator_optimizer['beta_1'])
    if discriminator_optimizer['name'] == 'adam':
        discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=discriminator_optimizer['learning_rate'], beta_1=discriminator_optimizer['beta_1'])
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                    discriminator_optimizer=discriminator_optimizer,
                                    generator=generator,
                                    discriminator=discriminator)
    manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)

    summary_log_file = os.path.join(
        WORKDIR, 'tf-summary', datetime.now().strftime("%Y%m%d-%H%M%S"))
    summary_writer = tf.summary.create_file_writer(summary_log_file)

    @tf.function
    def train_step(input_image, target, epoch):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = generator(input_image, training=True)

            tf.keras.layers.concatenate([input_image, target])
            disc_real_output = discriminator(tf.keras.layers.concatenate([input_image, target]), training=True)
            disc_generated_output = discriminator(tf.keras.layers.concatenate([input_image, gen_output]), training=True)

            gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(
                disc_generated_output, gen_output, target)
            disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

        generator_gradients = gen_tape.gradient(gen_total_loss,
                                                generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss,
                                                    discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(generator_gradients,
                                            generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                discriminator.trainable_variables))

        with summary_writer.as_default():
            tf.summary.scalar('gen_total_loss', gen_total_loss, step=epoch)
            tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=epoch)
            tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=epoch)
            tf.summary.scalar('disc_loss', disc_loss, step=epoch)
        
        return gen_total_loss, disc_loss


    checkpoint.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print('Restored from {}'.format(manager.latest_checkpoint))
    else:
        print('Initializing from scratch')


    for e in range(EPOCHS):
        start_time = time()
        gen_loss_total = disc_loss_total = 0
        for input_image, target in train_dataset:
            gen_loss, disc_loss = train_step(input_image, target, e)
            gen_loss_total += gen_loss
            disc_loss_total += disc_loss

        time_taken = time() - start_time

        if (e + 1) % 10 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)
        
        print('Epoch {}: gen loss: {}, disc loss: {}, time: {:.2f}s'.format(
            e + 1, gen_loss_total / BATCH_SIZE, disc_loss_total / BATCH_SIZE, time_taken))
        
        with open(csv_file, 'a') as f:
            f.write(f'{e+1},{gen_loss_total / BATCH_SIZE},{disc_loss_total / BATCH_SIZE},{time_taken}\n')

    # plot and save
    sample_count = 20
    Y_hat = generator(X_test[:sample_count])

    for idx, (x, y, y_hat) in enumerate(zip(X_test[:sample_count], Y_test[:sample_count], Y_hat)):

        # Original RGB image
        orig_lab = np.dstack((x, y * 128))
        orig_rgb = lab2rgb(orig_lab)

        # Grayscale version of the original image
        grayscale_lab = np.dstack((x, np.zeros((IMAGE_SIZE, IMAGE_SIZE, 2))))
        grayscale_rgb = lab2rgb(grayscale_lab)

        # Colorized image
        predicted_lab = np.dstack((x, y_hat * 128))
        predicted_rgb = lab2rgb(predicted_lab)

        # print(idx)
        # convert to cv2 format
        grayscale_rgb = (grayscale_rgb * 255).astype(np.uint8)
        orig_rgb = (orig_rgb * 255).astype(np.uint8)
        predicted_rgb = (predicted_rgb * 255).astype(np.uint8)
        # concat grayscale_rgb, orig_rgb, predicted_rgb
        img = np.concatenate((grayscale_rgb, orig_rgb, predicted_rgb), axis=1)
        os.makedirs(csv_file[:-4], exist_ok=True)
        cv2.imwrite(csv_file[:-4]+f'/{idx}.png', img)


######################
### TESTING MODELS ###
######################

# This part of the code is used to test different combinations of models and hyperparameters.
testing_models = \
{
    'generator_functions': [generator1, generator2],
    'discriminator_functions': [discriminator1, discriminator2],
    'generator_optimizers': [{'name':'adam', 'learning_rate':2e-4, 'beta_1':0.5}],
    'discriminator_optimizers': [{'name':'adam', 'learning_rate':2e-3, 'beta_1':0.7}],
    'batch_size': [128, 256]
}

testing_matrix = [] # every possible combination of models and hyperparameters
for g in testing_models['generator_functions']:
    for d in testing_models['discriminator_functions']:
        for go in testing_models['generator_optimizers']:
            for do in testing_models['discriminator_optimizers']:
               for b in testing_models['batch_size']:
                    testing_matrix.append((g, d, go, do, b))

saving_path = os.path.join(WORKDIR, 'results_grid_search')
os.makedirs(saving_path, exist_ok=True)
for test in testing_matrix:
    file_name = f'{test[0]}-{test[1]}-{test[2]}-{test[3]}-{test[4]}.csv'
    with open(os.path.join(saving_path, file_name), 'w') as f:
        f.write('epoch,gen_loss,disc_loss,time\n')
    train(test[0](), test[1](), test[2], test[3], test[4], os.path.join(saving_path, file_name))
