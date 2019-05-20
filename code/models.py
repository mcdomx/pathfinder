# Pathfinder - Support Script - models.py
# Author - Mark McDonald (2019)

# This script contains the base project model and supporting
# data generators.

import os
import random
import time
import numpy as np
from keras import regularizers
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dropout
from keras.models import Model

# Returns a tuple that includes a model description dictionary
# and a Keras untrained model.
# Arguments:
#   input_height:    the model's input height
#   input_width:     the model's input width
#   op_height:       the model's output height
#   op_width:        the model output width
#   l2_lambda:       if a value exists, regularization will be used with this value
#   epochs:          The number of epochs the model is trained with
#   final_activation: the final activation function
#   color:           Boolean value that indicates if model trains color or b&w
#   rot:             The augmentation rotation value
#   zoom:            The augmentation zoom value
#   hflip:           The augmentation horizontal flip value (T/F)
#   vflip:           The augmentation vertical flip value (T/F)
#   fill_mode:       fill mode used for augmentation (nearest or reflect are most reasonable)
#   dropout:         If 0, no dropout layers will be used.
#                    Else, a dropout layer is added with the provided value.
#   notes:           Notes that can be added to the description dictionary

# Return Value: List of path for each video created.
# Summary:
#   Image and mask data is extracted from each *.mat file.
#   Each mask is overlayed on its respective image.
#   Each overlayed image is added to a video sequence and saved
def get_autoenc_model(input_height=360,
                      input_width=480,
                      op_height=360,
                      op_width=480,
                      l2_lambda=0,
                      epochs=20,
                      final_activation='relu',
                      color=True,
                      rot=0,
                      zoom=0,
                      hflip=True,
                      vflip=False,
                      fill_mode=None,
                      dropout=0,
                      notes="no notes"):

    if fill_mode is None:   fill_mode = 'nearest'

    # use epoch time to label the results directory and file
    epoch_time = int(time.time())

    description = dict( Model_Type="CNN_encoder2400",
                        input_height=input_height,
                        input_width=input_width,
                        color=color,
                        output_height=op_height,
                        output_width=op_width,
                        l2_lambda=l2_lambda,
                        final_activation=final_activation,
                        optimizer='adadelta',
                        loss='binary_crossentropy',
                        learn_rate=None,
                        batch_size=64,
                        epochs=epochs,
                        rotation=rot,
                        zoom=zoom,
                        hflip=hflip,
                        vflip=vflip,
                        dropout=dropout,
                        epoch_time=epoch_time,
                        fill_mode=fill_mode)

    description.update(notes=notes)

    name = str(description["epoch_time"])
    name = name + "_" + description["Model_Type"]
    name = name + "_epochs=" + str(description["epochs"])
    name = name + "_notes=" + description["notes"]
    description.update(name=name)

    # Support for black and white images
    channels = 3
    if not color:
        channels = 1

    # MODEL DESIGN
    # Note that some layers are optionally added based on arguments provided
    input_img = Input(shape=(input_height, input_width, channels))

    # -- ENCODER --
    if l2_lambda == 0:
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    else:
        x = Conv2D(16, (3, 3), activation='relu', padding='same',
                   activity_regularizer=regularizers.l2(l2_lambda))(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)

    if dropout != 0:
        x = Dropout(dropout)(x)  # a single dropout works well with no regularization

    if l2_lambda == 0:
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    else:
        x = Conv2D(8, (3, 3), activation='relu', padding='same',
                   activity_regularizer=regularizers.l2(l2_lambda))(x)

    x = MaxPooling2D((2, 2), padding='same')(x)

    # if dropout != 0:
    #     x = Dropout(dropout)(x)  # trying additional dropouts

    if l2_lambda == 0:
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    else:
        x = Conv2D(8, (3, 3), activation='relu', padding='same',
                   activity_regularizer=regularizers.l2(l2_lambda))(x)

    x = MaxPooling2D((2, 2), padding='same')(x)

    # if dropout != 0:
    #     x = Dropout(dropout)(x)  # trying additional dropouts

    if l2_lambda == 0:
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    else:
        x = Conv2D(8, (3, 3), activation='relu', padding='same',
                   activity_regularizer=regularizers.l2(l2_lambda))(x)

    encoded = MaxPooling2D((3, 3), padding='same')(x)
    encoder = Model(input_img, encoded)

    # at this point the representation is (15, 20, 8) i.e. 2400-dimensional

    # -- DECODER --
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((3, 3))(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation=final_activation, padding='same')(x)

    autoencoder = Model(input_img, decoded)

    return description, autoencoder


# Data Generators

# Returns a generator used for training data.
# This generator will augment images.
# If arguments are not supplied, values from description dictionary are used.
# Arguments:
#   train_img_dir:    the model's input height
#   train_mask_dir:     the model's input width
#   description:       the model's output height
#   color:           Boolean value that indicates if model trains color or b&w
#   rot:             The augmentation rotation value
#   zoom:            The augmentation zoom value
#   hflip:           The augmentation horizontal flip value (T/F)
#   vflip:           The augmentation vertical flip value (T/F)
#   fill_mode:       Fill mode to be used for augmentation (nearest and reflect are most reasonable)
#   augment:         Boolean that indicates if images are augmented or not
#                    Validation generators should not be augmented
# Return Value:      Returns a generator that will provide an augmented image and
#                    identically augmented mask.
# Summary:
#   Masks and images are augmented with the same values by fitting a
#   test image and mask.  Same 'seed' value must be used for both generators.
def get_img_mask_generator(train_img_dir,
                        train_mask_dir,
                        description,
                        color=True,
                        rot=0,
                        wsr=0,
                        hsr=0,
                        zoom=0,
                        hflip=None,
                        vflip=None,
                        fill_mode=None,
                        augment=True):
    print("Creating generator:")
    print("\t"+train_img_dir)
    print("\t"+train_mask_dir)
    print("\tAugemntation: ", augment)
    batch_size = description["batch_size"]
    input_height = description["input_height"]
    input_width = description["input_width"]
    output_height = description["output_height"]
    output_width = description["output_width"]

    # Determine if argument values or description values are used
    if rot == 0:    rot = description["rotation"]
    if zoom == 0:   zoom = description["zoom"]
    if fill_mode is None:   fill_mode = description["fill_mode"]

    color_mode = 'rgb'
    if color is None:
        color = description["color"]
        if not color:   color_mode = 'grayscale'

    if hflip is None:   hflip = description["hflip"]
    else:               hflip = True

    if vflip is None:   vflip = description["vflip"]
    else:               vflip = True

    # Create ImageDataGenerator object with augmentation
    if augment:
        # Create dictionary for data generator arguments
        data_gen_args = dict(rotation_range=rot,
                             width_shift_range=wsr,
                             zoom_range=zoom,
                             height_shift_range=hsr,
                             horizontal_flip=hflip,
                             vertical_flip=vflip,
                             fill_mode=fill_mode)

        # the mask data is only 0's and 1's so does not need to be rescaled
        # The other augmentations must be the same as the image datagenerator above
        mask_datagen = ImageDataGenerator(**data_gen_args)

        # Add rescaling factor to the image generator
        data_gen_args.update(rescale=1. / 255)
        image_datagen = ImageDataGenerator(**data_gen_args)

    # Create ImageDataGenerator object without augmentation
    else:
        image_datagen = ImageDataGenerator(rescale=1. / 255)
        mask_datagen = ImageDataGenerator()

    # We are going to augment the image and mask in the same exact way.
    # To do this, we will create two identical data generators.
    # Since the augmentation is random, we need to make sure the random
    # change is applied the same to both.  We can do this be fitting the
    # data generators with the same seed.  The fit method only requires a
    # single sample image and mask, but it requires a 4th dimension as the
    # first element.

    # Get samples
    sample_img_path = os.path.join(train_img_dir, 'data')
    sample_img_path = os.path.join(sample_img_path, os.listdir(sample_img_path)[0])
    img = image.load_img(sample_img_path)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    sample_mask_path = os.path.join(train_mask_dir, 'data')
    sample_mask_path = os.path.join(sample_mask_path, os.listdir(sample_mask_path)[0])
    mask = image.load_img(sample_mask_path)
    mask_array = image.img_to_array(mask)
    mask_array = np.expand_dims(mask_array, axis=0)

    # Provide the same seed and keyword arguments to the fit and flow methods
    seed = random.randint(1, 100)
    image_datagen.fit(img_array, augment=False, seed=seed)
    mask_datagen.fit(mask_array, augment=False, seed=seed)

    image_generator = image_datagen.flow_from_directory(
        train_img_dir,
        target_size=(input_height, input_width),
        color_mode=color_mode,
        class_mode=None,
        batch_size=batch_size,
        shuffle=True,
        seed=seed)

    mask_generator = mask_datagen.flow_from_directory(
        train_mask_dir,
        target_size=(output_height, output_width),
        color_mode='grayscale',
        class_mode=None,
        batch_size=batch_size,
        shuffle=True,
        seed=seed)

    # combine generators into one which yields image and masks
    generator = zip(image_generator, mask_generator)

    return generator
