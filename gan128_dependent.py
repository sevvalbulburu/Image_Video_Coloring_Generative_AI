# After grid search for 32 pixel images train, two models were selected and modified to work on 128 pixel images,
# This code created by alperenlcr@gmail.com and sevval.bulburu@std.yildiz.edu.tr


###############
### IMPORTS ###
###############

import os, cv2
import numpy as np
import tensorflow as tf
import tensorflow_io as tfio
from datetime import datetime
from time import time
from skimage.color import rgb2lab, lab2rgb
from tensorflow.keras.layers import ( # type: ignore
    Activation, AveragePooling2D, BatchNormalization, Conv2D, Conv2DTranspose,
    Dense, Dropout, Flatten, Input, LeakyReLU, ReLU, UpSampling2D, MaxPool2D)
from tensorflow.keras.models import Sequential, Model # type: ignore


#################
### CONSTANTS ###
#################

IMAGE_SIZE = 128
EPOCHS = 200
BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 25
WORKDIR = "/home/alperenlcr/bitirme/"


######################################
### DATASET LOAD AND PREPROCESSING ###
######################################

# Dataset loading
def load_dataset(p):
    def input_map(rgb):
        # crop the left half of the image
        prev_frame = rgb[:, :, :IMAGE_SIZE, :]
        next_frame = rgb[:, :, IMAGE_SIZE:, :]
        lab = tfio.experimental.color.rgb_to_lab(next_frame/255)
        l = lab[:, :, :, :1]
        ab = output_map(prev_frame)
        return tf.concat([l, ab], axis=-1)

    def output_map(rgb):
        lab = tfio.experimental.color.rgb_to_lab(rgb/255)
        return lab[:, :, :, 1:]/128

    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # Dataset loading
    train_dataset_l = tf.keras.preprocessing.image_dataset_from_directory(
        WORKDIR+p, image_size=(IMAGE_SIZE, IMAGE_SIZE*2), batch_size=BATCH_SIZE, shuffle=False, label_mode=None
    )
    train_dataset_l = train_dataset_l.map(input_map)

    train_dataset_ab = tf.keras.preprocessing.image_dataset_from_directory(
        WORKDIR+p, image_size=(IMAGE_SIZE, IMAGE_SIZE*2), batch_size=BATCH_SIZE, shuffle=False, label_mode=None
    )
    # take the right half of the image
    train_dataset_ab = train_dataset_ab.map(lambda x: x[:, :, IMAGE_SIZE:, :])
    train_dataset_ab = train_dataset_ab.map(output_map)

    train_dataset = tf.data.Dataset.zip((train_dataset_l, train_dataset_ab))

    return train_dataset

train_dataset = load_dataset('village_island_dataset_train_dependent/rgb/')
test_dataset = load_dataset('village_island_dataset_test_dependent/rgb/')

##############
### MODELS ###
##############

#disc for shape 128x128
def discriminator():
    model = Sequential()
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)))
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(AveragePooling2D(pool_size=(2, 2)))
    model.add(Dropout(.25))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(AveragePooling2D(pool_size=(2, 2)))
    model.add(Dropout(.25))

    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(AveragePooling2D(pool_size=(2, 2)))
    model.add(Dropout(.25))
    
    model.add(Flatten())
    model.add(Dense(512))
    model.add(LeakyReLU(.2))
    model.add(BatchNormalization())
    model.add(Dropout(.3))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

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

    inputs = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))

    # Downsampling layers
    # 1: (BATCH_SIZE, 16, 16, 32)
    # 2: (BATCH_SIZE, 8, 8, 64)
    # 3: (BATCH_SIZE, 4, 4, 128)
    # 4: (BATCH_SIZE, 2, 2, 256)
    # 5: (BATCH_SIZE, 1, 1, 256)

    downstack = [
        downsample(128, 4, apply_batchnorm=False),
        downsample(256, 4),
        downsample(512, 4),
        downsample(1024, 4),
        downsample(1024, 4)
    ]

    # Upsampling layers
    # 1: (BATCH_SIZE, 1, 1, 256)
    # 2: (BATCH_SIZE, 1, 1, 128)
    # 3: (BATCH_SIZE, 1, 1, 64)
    # 4: (BATCH_SIZE, 1, 1, 32)
    
    upstack = [
        upsample(1024, 4, apply_dropout=True),
        upsample(512, 4),
        upsample(256, 4),
        upsample(128, 4),
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

def train(generator, discriminator, generator_optimizer, discriminator_optimizer, csv_file, skip_training=False):
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

    @tf.function(experimental_relax_shapes=True)
    def train_step(input_image, target, epoch):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = generator(input_image, training=True)

            tf.keras.layers.concatenate([input_image, target])
            disc_real_output = discriminator(tf.keras.layers.concatenate([input_image[:,:,:,:1], target]), training=True)
            disc_generated_output = discriminator(tf.keras.layers.concatenate([input_image[:,:,:,:1], gen_output]), training=True)

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
        skip_training = True
    else:
        print('Initializing from scratch')


    if not skip_training:
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

    # # plot all batch of images
    # count = 0
    # for input_image, target in test_dataset:
    #     count += 1
    #     cv2.imwrite(csv_file[:-4]+"sil_"+str(count)+'.png', generate_images(generator, input_image, target))


######################
### TESTING MODELS ###
######################
def generate_images(model, test_input, tar):
    prediction = model(test_input, training=True)
    img_list = []
    for i in range(BATCH_SIZE):
        grayscale_lab = np.dstack((test_input[i][:, :, 0], np.zeros((IMAGE_SIZE, IMAGE_SIZE, 2))))
        grayscale_rgb = lab2rgb(grayscale_lab)
        gray = (grayscale_rgb * 255).astype(np.uint8)

        orig_lab = np.dstack((test_input[i][:, :, 0], tar[i] * 128))
        orig_rgb = lab2rgb(orig_lab)
        ground_truth = (orig_rgb * 255).astype(np.uint8)
        ground_truth = cv2.cvtColor(ground_truth, cv2.COLOR_RGB2BGR)

        predicted_lab = np.dstack((test_input[i][:, :, 0], prediction[i] * 128))
        predicted_rgb = lab2rgb(predicted_lab)
        predicted = (predicted_rgb * 255).astype(np.uint8)
        predicted = cv2.cvtColor(predicted, cv2.COLOR_RGB2BGR)

        display_list = [gray, ground_truth, predicted]
        img = np.concatenate(display_list, axis=1)
        img_list.append(img)
    merged_img = np.concatenate(img_list, axis=0)
    return merged_img


def generate_image(model, test_input, tar):
    prediction = model(np.array([test_input]), training=True)
    grayscale_lab = np.dstack((test_input[:, :, 0], np.zeros((IMAGE_SIZE, IMAGE_SIZE, 2))))
    grayscale_rgb = lab2rgb(grayscale_lab)
    gray = (grayscale_rgb * 255).astype(np.uint8)

    orig_lab = np.dstack((test_input[:, :, 0], tar * 128))
    orig_rgb = lab2rgb(orig_lab)
    ground_truth = (orig_rgb * 255).astype(np.uint8)
    ground_truth = cv2.cvtColor(ground_truth, cv2.COLOR_RGB2BGR)

    predicted_lab = np.dstack((test_input[:, :, 0], prediction[0] * 128))
    predicted_rgb = lab2rgb(predicted_lab)
    predicted = (predicted_rgb * 255).astype(np.uint8)
    predicted = cv2.cvtColor(predicted, cv2.COLOR_RGB2BGR)

    display_list = [gray, ground_truth, predicted]
    img = np.concatenate(display_list, axis=1)
    return img



generator_dependent = generator2()
discriminator_dependent = discriminator()
train(generator_dependent, discriminator_dependent,\
        {'name': 'adam', 'learning_rate': 0.0004, 'beta_1': 0.5}, \
        {'name': 'adam', 'learning_rate': 0.0004, 'beta_1': 0.5}, \
        WORKDIR+'village_island_dependent.csv', skip_training=True)
print()
print('Training completed')
print()


if False:
    print("Testing started")
    # plot all batch of images
    count = 0
    for input_image, target in test_dataset:
        count += 1
        cv2.imwrite(WORKDIR+"sil2_"+str(count)+'.png', generate_images(generator_dependent, input_image, target))
    print("Testing completed")
